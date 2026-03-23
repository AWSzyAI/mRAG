#!/usr/bin/env python3
import argparse
import copy
import gc
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import torch
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "github/LLaVA-NeXT"))
sys.path.append(str(ROOT_DIR / "github/MRAG-Bench/eval"))
sys.path.append(str(ROOT_DIR / "github/magiclens"))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from utils.dataloader import bench_data_loader  # noqa: E402
from data_utils import process_img  # noqa: E402
from inference import load_model as load_magiclens_model  # noqa: E402
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_question_and_options(question_blob: str):
    marker = "\n Choices:\n"
    if marker not in question_blob:
        return question_blob, {}

    question_text, choices_blob = question_blob.split(marker, 1)
    options = {}
    for line in choices_blob.splitlines():
        m = re.match(r"^([A-D]):\s*(.*)$", line.strip())
        if m:
            options[m.group(1)] = m.group(2)
    return question_text, options


def extract_choice(text: str) -> str:
    text_up = str(text).upper()
    candidates = []

    for choice in ("A", "B", "C", "D"):
        for m in re.finditer(rf"\({choice}\)", text_up):
            candidates.append((m.start(), choice))

    if not candidates:
        for choice in ("A", "B", "C", "D"):
            for m in re.finditer(rf"\b{choice}\b", text_up):
                candidates.append((m.start(), choice))

    if not candidates:
        return "N/A"
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def is_oom_error(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def parse_int_list(s: str):
    vals = []
    for part in re.split(r"[,\s]+", str(s).strip()):
        if not part:
            continue
        try:
            n = int(part)
        except ValueError:
            continue
        if n > 0 and n not in vals:
            vals.append(n)
    return vals


def resolve_bpe_path(explicit_path: str) -> str:
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    env_path = os.environ.get("MAGICLENS_BPE_PATH", "")
    if env_path and os.path.exists(env_path):
        return env_path
    candidates = [
        ROOT_DIR / "models/bpe_simple_vocab_16e6.txt.gz",
        ROOT_DIR
        / "github/LLaVA-NeXT/llava/model/multimodal_encoder/dev_eva_clip/eva_clip/bpe_simple_vocab_16e6.txt.gz",
        Path.home() / ".cache/scenic/clip/bpe_simple_vocab_16e6.txt.gz",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return ""


def build_data_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_name=args.dataset_name,
        test_size=10**9,
        use_rag=True,
        use_retrieved_examples=False,
        extra_prompt="",
        max_rag_images=args.max_rag_images,
        hf_cache_dir=args.hf_cache_dir,
        hf_offline=args.hf_offline,
        hf_max_retries=args.hf_max_retries,
        dataset_heartbeat_sec=args.dataset_heartbeat_sec,
        hf_hub_etag_timeout=args.hf_hub_etag_timeout,
        hf_hub_download_timeout=args.hf_hub_download_timeout,
    )


def get_data_iter_and_total(data_args):
    try:
        sig = inspect.signature(bench_data_loader)
        if "return_total" in sig.parameters:
            data_iter, total = bench_data_loader(
                data_args, image_placeholder=DEFAULT_IMAGE_TOKEN, return_total=True
            )
            return data_iter, total
    except (TypeError, ValueError):
        pass

    data_iter = bench_data_loader(data_args, image_placeholder=DEFAULT_IMAGE_TOKEN)
    return data_iter, None


def pick_sample(args):
    data_args = build_data_args(args)
    data_iter, total = get_data_iter_and_total(data_args)
    if total is None:
        log("dataset_total=unknown (dataloader has no return_total)")
    else:
        log(f"dataset_total={total}")

    for idx, item in enumerate(data_iter):
        if args.sample_id is not None and str(item["id"]) == str(args.sample_id):
            return idx, item
        if args.sample_id is None and idx == args.sample_index:
            return idx, item
    raise ValueError(
        f"sample not found: sample_index={args.sample_index}, sample_id={args.sample_id}"
    )


def save_sample_images(image_files, out_dir: Path):
    saved = []
    for i, img in enumerate(image_files):
        role = "query" if i == 0 else f"rag_top{i}"
        path = out_dir / f"{role}.png"
        img.save(path)
        saved.append(
            {
                "role": role,
                "path": str(path),
                "name": path.name,
            }
        )
    return saved


def parse_llava_jsonl(jsonl_path: str, qs_id: str, method_name: str):
    if not jsonl_path:
        return None
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"{method_name} jsonl not found: {jsonl_path}")

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("qs_id")) != str(qs_id):
                continue
            output = str(row.get("output", "")).strip()
            return {
                "source": f"jsonl:{p}",
                "raw_output": output,
                "pred_choice": extract_choice(output),
            }
    raise ValueError(f"qs_id={qs_id} not found in {jsonl_path}")


def clear_torch_cuda_cache(reason: str = ""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    if reason:
        log(f"[CUDA] cache cleared ({reason})")


def run_llava_decode_once(item, args, num_beams: int, max_new_tokens: int):
    model_name = "llava_qwen"
    llava_args = {
        "multimodal": True,
        "attn_implementation": "sdpa",
        "overwrite_config": {"image_aspect_ratio": "pad"},
    }
    tokenizer = None
    model = None
    image_processor = None
    input_ids = None
    image_tensors = None
    image_sizes = None
    cont = None
    try:
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.llava_model_path, None, model_name, device_map="auto", **llava_args
        )
        model.eval()

        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], item["question"])
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        model_device = next(model.parameters()).device
        image_dtype = torch.float16 if model_device.type == "cuda" else torch.float32
        input_ids = (
            tokenizer_image_token(
                prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(model_device)
        )

        image_tensors = process_images(item["image_files"], image_processor, model.config)
        image_tensors = [img.to(dtype=image_dtype, device=model_device) for img in image_tensors]
        image_sizes = [img.size for img in item["image_files"]]

        with torch.inference_mode():
            cont = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                max_new_tokens=max_new_tokens,
                num_beams=max(1, int(num_beams)),
                do_sample=False,
                temperature=0.0,
            )
        text = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return {
            "source": "live",
            "raw_output": text,
            "pred_choice": extract_choice(text),
            "num_beams": int(num_beams),
            "max_new_tokens": int(max_new_tokens),
        }
    finally:
        del cont, input_ids, image_tensors, image_sizes, image_processor, model, tokenizer
        clear_torch_cuda_cache(
            f"after llava decode num_beams={int(num_beams)} max_new_tokens={int(max_new_tokens)}"
        )


def run_llava_live(item, args):

    log("running llava greedy (num_beams=1)")
    greedy = run_llava_decode_once(item, args, num_beams=1, max_new_tokens=args.max_new_tokens)

    beam_size = max(1, int(args.llava_beam_size))
    log(f"running llava beam (num_beams={beam_size})")

    fallback_tokens = max(1, min(int(args.max_new_tokens), int(args.llava_oom_max_new_tokens)))
    fallback_last_tokens = max(
        1, min(int(args.max_new_tokens), int(args.llava_oom_final_max_new_tokens))
    )
    attempts = [(beam_size, int(args.max_new_tokens), "primary")]
    if not args.disable_llava_oom_fallback:
        for b in parse_int_list(args.llava_oom_fallback_beams):
            attempts.append((b, fallback_tokens, "oom_fallback"))
        attempts.append((1, fallback_last_tokens, "oom_last_resort"))

    uniq = []
    seen = set()
    for beams, tok, reason in attempts:
        key = (max(1, int(beams)), max(1, int(tok)))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((key[0], key[1], reason))

    beam = None
    last_exc = None
    for idx, (beams, tok, reason) in enumerate(uniq, start=1):
        if idx > 1:
            log(
                f"[OOM-FALLBACK] retry beam attempt {idx}/{len(uniq)} "
                f"(num_beams={beams}, max_new_tokens={tok})"
            )
        try:
            beam = run_llava_decode_once(item, args, num_beams=beams, max_new_tokens=tok)
            if idx > 1:
                beam["fallback_used"] = True
                beam["fallback_reason"] = reason
            break
        except Exception as exc:
            if not is_oom_error(exc) or idx == len(uniq):
                raise
            last_exc = exc
            log(
                f"[OOM] beam attempt failed (num_beams={beams}, max_new_tokens={tok}). "
                "Clearing CUDA cache and retrying..."
            )
            clear_torch_cuda_cache("beam oom retry")

    if beam is None and last_exc is not None:
        raise last_exc
    return greedy, beam


def run_llava(item, args):
    greedy = parse_llava_jsonl(args.llava_greedy_jsonl, item["id"], "llava_greedy")
    beam = parse_llava_jsonl(args.llava_beam_jsonl, item["id"], "llava_beam")

    if args.skip_llava:
        if greedy is None or beam is None:
            raise ValueError(
                "--skip-llava is set, but --llava-greedy-jsonl / --llava-beam-jsonl do not both contain this sample."
            )
        return greedy, beam

    if greedy is None or beam is None:
        live_greedy, live_beam = run_llava_live(item, args)
        if greedy is None:
            greedy = live_greedy
        if beam is None:
            beam = live_beam

    return greedy, beam


def magiclens_encode(model, params, tokenizer_fn, image_path: str, text: str):
    image = process_img(image_path, 224)
    tokens = np.array(tokenizer_fn(text))
    out = model.apply(params, {"ids": jnp.array(tokens), "image": jnp.array(image)})
    return np.array(out["multimodal_embed_norm"])[0]


def rank_rag_images(query_embed, rag_paths, rag_embeds):
    sims = [float(np.dot(query_embed, r_embed)) for r_embed in rag_embeds]
    ranked_pairs = sorted(zip(rag_paths, sims), key=lambda x: x[1], reverse=True)
    rows = []
    for rank, (path, score) in enumerate(ranked_pairs, start=1):
        rows.append(
            {
                "rank": rank,
                "image": Path(path).name,
                "path": str(path),
                "score": float(score),
            }
        )
    return rows


def run_magiclens(question_text, options, sample_images, args):
    bpe_path = resolve_bpe_path(args.bpe_path)
    if bpe_path:
        tokenizer_fn = clip_tokenizer.build_tokenizer(bpe_path=bpe_path)
    else:
        tokenizer_fn = clip_tokenizer.build_tokenizer()

    model, params = load_magiclens_model(args.magiclens_model_size, args.magiclens_model_path)
    query_path = sample_images[0]["path"]
    rag_paths = [x["path"] for x in sample_images[1:]]
    rag_embeds = [magiclens_encode(model, params, tokenizer_fn, p, "") for p in rag_paths]

    question_only_embed = magiclens_encode(model, params, tokenizer_fn, query_path, question_text)
    question_only_rank = rank_rag_images(question_only_embed, rag_paths, rag_embeds)

    option_scores = {}
    option_topk = {}
    for letter in ("A", "B", "C", "D"):
        if letter not in options:
            continue
        q_text = f"{question_text}\nChoice {letter}: {options[letter]}"
        q_embed = magiclens_encode(model, params, tokenizer_fn, query_path, q_text)
        ranked = rank_rag_images(q_embed, rag_paths, rag_embeds)
        option_scores[letter] = ranked[0]["score"] if ranked else float("-inf")
        option_topk[letter] = ranked[: min(args.top_k, len(ranked))]

    pred_choice = max(option_scores.items(), key=lambda kv: kv[1])[0] if option_scores else "N/A"
    return {
        "pred_choice": pred_choice,
        "option_scores": option_scores,
        "option_topk": option_topk,
        "pred_choice_topk": option_topk.get(pred_choice, []),
        "question_only_topk": question_only_rank[: min(args.top_k, len(question_only_rank))],
    }


def attach_correctness(report):
    gt = str(report["sample"]["gt_choice"])
    report["llava_greedy"]["is_correct"] = report["llava_greedy"]["pred_choice"] == gt
    report["llava_beam"]["is_correct"] = report["llava_beam"]["pred_choice"] == gt
    report["magiclens"]["is_correct"] = report["magiclens"]["pred_choice"] == gt


def table_rows(rows):
    if not rows:
        return ["| rank | image | score |", "| --- | --- | --- |", "| - | - | - |"]
    out = ["| rank | image | score |", "| --- | --- | --- |"]
    for row in rows:
        out.append(f"| {row['rank']} | {row['image']} | {row['score']:.6f} |")
    return out


def write_report(report, out_dir: Path):
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# One Sample Compare")
    lines.append("")
    lines.append(f"- sample_index: {report['sample']['sample_index']}")
    lines.append(f"- qs_id: {report['sample']['qs_id']}")
    lines.append(f"- scenario: {report['sample']['scenario']}")
    lines.append(f"- aspect: {report['sample']['aspect']}")
    lines.append(f"- gt_choice: {report['sample']['gt_choice']}")
    lines.append(f"- gt_answer: {report['sample']['gt_answer']}")
    lines.append("")
    lines.append("## Prompt")
    lines.append(report["sample"]["prompt_instruction"])
    lines.append("")
    lines.append("## Question")
    lines.append(report["sample"]["question_text"])
    lines.append("")
    lines.append("## Options")
    for k, v in report["sample"]["options"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Retrieved Images (dataset order)")
    for row in report["sample"]["images"]:
        lines.append(f"- {row['role']}: {row['path']}")
    lines.append("")
    lines.append("## LLaVA Greedy")
    lines.append(f"- source: {report['llava_greedy'].get('source', 'live')}")
    lines.append(f"- pred_choice: {report['llava_greedy']['pred_choice']}")
    lines.append(f"- is_correct: {report['llava_greedy']['is_correct']}")
    lines.append(f"- raw_output: {report['llava_greedy']['raw_output']}")
    lines.append("")
    lines.append(f"## LLaVA Beam ({report['llava_beam'].get('num_beams', 'N/A')})")
    lines.append(f"- source: {report['llava_beam'].get('source', 'live')}")
    lines.append(f"- pred_choice: {report['llava_beam']['pred_choice']}")
    lines.append(f"- is_correct: {report['llava_beam']['is_correct']}")
    lines.append(f"- raw_output: {report['llava_beam']['raw_output']}")
    lines.append("")
    lines.append("## MagicLens")
    lines.append(f"- pred_choice: {report['magiclens']['pred_choice']}")
    lines.append(f"- is_correct: {report['magiclens']['is_correct']}")
    lines.append("- option_scores")
    for k, v in report["magiclens"]["option_scores"].items():
        lines.append(f"  - {k}: {v:.6f}")
    lines.append("- question_only_topk")
    lines.extend(table_rows(report["magiclens"]["question_only_topk"]))
    lines.append("")
    lines.append("### topk per option")
    for k, rows in report["magiclens"]["option_topk"].items():
        lines.append(f"#### Option {k}")
        lines.extend(table_rows(rows))
        lines.append("")
    lines.append("")
    lines.append("## Final Compare")
    lines.append(f"- gt_choice: {report['sample']['gt_choice']}")
    lines.append(f"- llava_greedy: {report['llava_greedy']['pred_choice']}")
    lines.append(f"- llava_beam5: {report['llava_beam']['pred_choice']}")
    lines.append(f"- magiclens: {report['magiclens']['pred_choice']}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare one MRAG-Bench sample with LLaVA greedy/beam and MagicLens."
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="uclanlp/MRAG-Bench")
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    parser.add_argument("--hf-offline", action="store_true")
    parser.add_argument("--hf-max-retries", type=int, default=8)
    parser.add_argument("--hf-hub-etag-timeout", type=int, default=30)
    parser.add_argument("--hf-hub-download-timeout", type=int, default=600)
    parser.add_argument("--dataset-heartbeat-sec", type=int, default=10)
    parser.add_argument("--max-rag-images", type=int, default=3)

    parser.add_argument(
        "--llava-model-path",
        type=str,
        default=str(ROOT_DIR / "models/llava-onevision-qwen2-7b-ov"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--llava-beam-size", type=int, default=5)
    parser.add_argument("--llava-oom-fallback-beams", type=str, default="3,2,1")
    parser.add_argument("--llava-oom-max-new-tokens", type=int, default=32)
    parser.add_argument("--llava-oom-final-max-new-tokens", type=int, default=16)
    parser.add_argument("--disable-llava-oom-fallback", action="store_true")
    parser.add_argument("--llava-greedy-jsonl", type=str, default="")
    parser.add_argument("--llava-beam-jsonl", type=str, default="")
    parser.add_argument("--skip-llava", action="store_true")

    parser.add_argument(
        "--magiclens-model-path",
        type=str,
        default=str(ROOT_DIR / "models/magic_lens_clip_base.pkl"),
    )
    parser.add_argument("--magiclens-model-size", type=str, default="base", choices=["base", "large"])
    parser.add_argument("--bpe-path", type=str, default="")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "log/one_sample_compare"),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("loading one sample from MRAG-Bench")
    sample_index, item = pick_sample(args)
    question_blob = item.get("prompt_question_part", item["question"])
    question_text, options = parse_question_and_options(question_blob)
    sample_images = save_sample_images(item["image_files"], out_dir)
    log(f"sample selected: index={sample_index} qs_id={item['id']}")

    log("running llava comparison")
    llava_greedy, llava_beam = run_llava(item, args)
    clear_torch_cuda_cache("before magiclens")

    log("running magiclens comparison")
    magiclens = run_magiclens(question_text, options, sample_images, args)

    report = {
        "sample": {
            "sample_index": sample_index,
            "qs_id": item["id"],
            "scenario": item["scenario"],
            "aspect": item["aspect"],
            "gt_choice": item["gt_choice"],
            "gt_answer": item["answer"],
            "prompt_instruction": item.get("prompt_instruction_part", ""),
            "question_text": question_text,
            "options": options,
            "images": sample_images,
        },
        "llava_greedy": llava_greedy,
        "llava_beam": llava_beam,
        "magiclens": magiclens,
    }
    attach_correctness(report)

    json_path, md_path = write_report(report, out_dir)
    log(f"report_json={json_path}")
    log(f"report_md={md_path}")
    log("done")


if __name__ == "__main__":
    main()
