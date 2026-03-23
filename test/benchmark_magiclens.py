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
import uuid
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import torch
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "github/LLaVA-NeXT"))
sys.path.append(str(ROOT_DIR / "github/MRAG-Bench/eval"))
sys.path.append(str(ROOT_DIR / "github/magiclens"))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from utils.dataloader import bench_data_loader  # noqa: E402
from inference import load_model as load_magiclens_model  # noqa: E402


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def log_torch_cuda_env() -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    names = []
    for i in range(count):
        try:
            names.append(torch.cuda.get_device_name(i))
        except Exception:
            names.append("<unknown>")
    log(
        "torch_cuda="
        f"available={available}, count={count}, visible={cuda_visible}, devices={names}"
    )


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
        use_retrieved_examples=not bool(args.use_gt),
        extra_prompt="",
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


def infer_dataset_total(dataset_name: str):
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="test")
        return len(ds)
    except Exception as e:
        log(f"[WARN] infer_dataset_total failed: {e}")
        return None


def preprocess_pil_image(image, size: int = 224) -> np.ndarray:
    arr = jnp.array(image.convert("RGB"))[jnp.newaxis, ...]
    arr = arr / (arr.max() + 1e-12)
    arr = jax.image.resize(arr, (1, size, size, 3), method="bilinear")
    return np.array(arr)


def build_magiclens_encoder(model, params, disable_jit: bool = False):
    def encode(ids, image):
        out = model.apply(params, {"ids": ids, "image": image})
        return out["multimodal_embed_norm"]

    if disable_jit:
        return encode
    encode = jax.jit(encode)
    return encode


def trim_rag_images(image_files, max_rag_images: int):
    if not image_files:
        return image_files
    if max_rag_images < 0:
        return image_files
    return [image_files[0]] + list(image_files[1 : 1 + max_rag_images])


def rerank_rag_images(encode_fn, tokenizer_fn, question_text: str, image_files):
    if len(image_files) < 2:
        return image_files, []

    query_image = image_files[0]
    rag_images = image_files[1:]

    rag_img_batch = np.concatenate([preprocess_pil_image(img, 224) for img in rag_images], axis=0)
    rag_tok_batch = np.concatenate([np.array(tokenizer_fn("")) for _ in rag_images], axis=0)
    rag_embeds = np.asarray(encode_fn(jnp.array(rag_tok_batch), jnp.array(rag_img_batch)))

    q_img = preprocess_pil_image(query_image, 224)
    q_tok = np.array(tokenizer_fn(question_text))
    q_embed = np.asarray(encode_fn(jnp.array(q_tok), jnp.array(q_img)))[0]

    sims = np.matmul(rag_embeds, q_embed)
    order = np.argsort(-sims)
    reranked = [query_image] + [rag_images[int(i)] for i in order.tolist()]
    rank_info = [
        {
            "new_rank": int(rank + 1),
            "orig_rag_index": int(i + 1),
            "score": float(sims[int(i)]),
        }
        for rank, i in enumerate(order.tolist())
    ]
    return reranked, rank_info


def load_reference_predictions(path: str):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"reference jsonl not found: {path}")
    ref = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qs_id = str(row.get("qs_id"))
            ref[qs_id] = extract_choice(str(row.get("output", "")))
    return ref


def load_llava(args):
    model_name = "llava_qwen"
    visible_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    dm_arg = str(args.llava_device_map).strip().lower()
    if dm_arg in ("single", "cuda", "cuda:0", "0"):
        llava_device_map = {"": "cuda:0"}
    elif re.fullmatch(r"(?:cuda:)?\d+", dm_arg):
        device_idx = dm_arg.split(":")[-1]
        llava_device_map = {"": f"cuda:{device_idx}"}
    else:
        llava_device_map = args.llava_device_map
    if llava_device_map == "auto" and visible_cuda_count >= 2:
        llava_device_map = "balanced"
        log("llava_device_map auto->balanced (multi-GPU detected)")

    llava_args = {
        "multimodal": True,
        "overwrite_config": {"image_aspect_ratio": "pad"},
    }
    if args.llava_attn_implementation:
        llava_args["attn_implementation"] = args.llava_attn_implementation
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.llava_model_path,
        None,
        model_name,
        device_map=llava_device_map,
        load_4bit=bool(args.llava_load_4bit),
        load_8bit=bool(args.llava_load_8bit),
        **llava_args,
    )
    model.eval()
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        counts = {}
        for v in device_map.values():
            key = str(v)
            counts[key] = counts.get(key, 0) + 1
        summary = ", ".join([f"{k}:{counts[k]}" for k in sorted(counts.keys())])
        log(f"llava_hf_device_map={summary}")
        has_cpu_offload = any(str(v) == "cpu" for v in device_map.values())
        if has_cpu_offload:
            log("[WARN] LLaVA has CPU offload. This usually causes major slowdown.")
            if not args.llava_allow_cpu_offload:
                raise RuntimeError(
                    "LLaVA CPU offload detected and llava_allow_cpu_offload is disabled. "
                    "Try one of: "
                    "(1) CUDA_VISIBLE_DEVICES=0,1 (if available), "
                    "(2) LLAVA_DEVICE_MAP=single, "
                    "(3) LLAVA_LOAD_4BIT=1."
                )
    return tokenizer, model, image_processor


def llava_answer(tokenizer, model, image_processor, item, image_files, args):
    question_part = item.get("prompt_question_part", "")
    if not question_part:
        question_blob = item.get("question", "")
        marker = "\n Choices:\n"
        if marker in question_blob:
            question_part = question_blob.split(marker, 1)[0] + marker + question_blob.split(marker, 1)[1]
        else:
            question_part = question_blob

    if len(image_files) <= 1:
        instruction = "Answer with the option's letter from the given choices directly. "
    else:
        instruction = (
            "You will be given one question concerning several images. "
            "The first image is the input image, others are retrieved examples to help you. "
            "Answer with the option's letter from the given choices directly. "
        )
    image_tokens = DEFAULT_IMAGE_TOKEN * max(1, len(image_files))
    user_query = f"{instruction}{image_tokens}\n{question_part}"

    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], user_query)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    model_device = next(model.parameters()).device
    image_dtype = torch.float16 if model_device.type == "cuda" else torch.float32

    input_ids = (
        tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model_device)
    )
    image_tensors = process_images(image_files, image_processor, model.config)
    image_tensors = [img.to(dtype=image_dtype, device=model_device) for img in image_tensors]
    image_sizes = [img.size for img in image_files]

    with torch.inference_mode():
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0.0,
            num_beams=max(1, int(args.llava_num_beams)),
            max_new_tokens=max(1, int(args.llava_max_new_tokens)),
        )
    return tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser(
        description="MagicLens reranks RAG images; final A/B/C/D is answered by LLaVA."
    )
    parser.add_argument("--dataset-name", type=str, default="uclanlp/MRAG-Bench")
    parser.add_argument(
        "--answers-file",
        type=str,
        default=str(ROOT_DIR / "github/MRAG-Bench/magiclens_rerank_llava_results.jsonl"),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 means run all remaining samples.",
    )
    parser.add_argument(
        "--max-rag-images",
        type=int,
        default=5,
        help="Baseline-aligned default: keep up to 5 retrieved images (+1 query image).",
    )
    parser.add_argument(
        "--magiclens-model-path",
        type=str,
        default=str(ROOT_DIR / "models/magic_lens_clip_base.pkl"),
    )
    parser.add_argument("--magiclens-model-size", type=str, default="base", choices=["base", "large"])
    parser.add_argument("--bpe-path", type=str, default="")
    parser.add_argument("--magiclens-disable-jit", action="store_true")
    parser.add_argument(
        "--magiclens-clear-cache-every",
        type=int,
        default=0,
        help="If >0, periodically call jax.clear_caches() and gc.collect().",
    )
    parser.add_argument("--disable-magiclens-rerank", action="store_true")
    parser.add_argument(
        "--llava-model-path",
        type=str,
        default=str(ROOT_DIR / "models/llava-onevision-qwen2-7b-ov"),
    )
    parser.add_argument(
        "--llava-device-map",
        type=str,
        default="auto",
        help="auto / single / cuda:0 / balanced / sequential",
    )
    parser.add_argument(
        "--llava-attn-implementation",
        type=str,
        default="sdpa",
        help="e.g. sdpa / flash_attention_2 / empty string",
    )
    parser.add_argument("--llava-load-4bit", action="store_true")
    parser.add_argument("--llava-load-8bit", action="store_true")
    parser.add_argument("--llava-allow-cpu-offload", action="store_true")
    parser.add_argument("--llava-max-new-tokens", type=int, default=4096)
    parser.add_argument("--llava-num-beams", type=int, default=1)
    parser.add_argument(
        "--llava-greedy-jsonl",
        type=str,
        default="",
        help="Optional baseline jsonl for agreement/accuracy comparison.",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default=str(ROOT_DIR / "log/magiclens_rerank_llava_summary.json"),
    )
    parser.add_argument(
        "--use-gt",
        "--use_GT",
        dest="use_gt",
        action="store_true",
        help="Use gt_images as RAG examples (oracle-GT mode).",
    )
    parser.add_argument(
        "--no-use-gt",
        "--no-use_GT",
        dest="use_gt",
        action="store_false",
        help="Use retrieved_images as RAG examples (real retrieval mode).",
    )
    parser.set_defaults(use_gt=True)
    args = parser.parse_args()

    answers_path = Path(args.answers_file)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    bpe_path = resolve_bpe_path(args.bpe_path)
    if bpe_path:
        tokenizer_fn = clip_tokenizer.build_tokenizer(bpe_path=bpe_path)
        log(f"using_bpe_path={bpe_path}")
    else:
        tokenizer_fn = clip_tokenizer.build_tokenizer()
        log("using_bpe_path=<default>")

    log("loading MagicLens model")
    try:
        ml_model, ml_params = load_magiclens_model(args.magiclens_model_size, args.magiclens_model_path)
    except Exception as e:
        msg = str(e).lower()
        if "cuinit" in msg or "unable to load the cuda libraries" in msg or "cuda error" in msg:
            raise RuntimeError(
                "JAX CUDA backend init failed. "
                "Run on a GPU node with valid CUDA runtime, "
                "or rerun with JAX_PLATFORMS=cpu."
            ) from e
        raise
    jax_backend = jax.default_backend()
    disable_jit = bool(args.magiclens_disable_jit)
    if jax_backend == "cpu" and not disable_jit:
        disable_jit = True
        log("JAX backend is CPU; auto-disabling MagicLens JIT for stability.")
    cache_clear_every = int(args.magiclens_clear_cache_every)
    if cache_clear_every <= 0 and jax_backend == "cpu":
        cache_clear_every = 200
    log(
        f"magiclens_backend={jax_backend}, disable_jit={disable_jit}, "
        f"clear_cache_every={cache_clear_every}"
    )

    ml_encode_fn = build_magiclens_encoder(ml_model, ml_params, disable_jit=disable_jit)
    log("MagicLens model ready")

    log(f"loading LLaVA model from {args.llava_model_path}")
    log_torch_cuda_env()
    llava_tokenizer, llava_model, llava_image_processor = load_llava(args)
    log("LLaVA model ready")

    ref_preds = load_reference_predictions(args.llava_greedy_jsonl)
    if ref_preds:
        log(f"loaded_reference_predictions={len(ref_preds)}")

    data_args = build_data_args(args)
    rag_source = "gt_images" if args.use_gt else "retrieved_images"
    log(f"rag_source={rag_source}")
    data_iter, total = get_data_iter_and_total(data_args)
    if total is None:
        total = infer_dataset_total(args.dataset_name)
        if total is None:
            log("dataset_total=unknown (dataloader has no return_total)")
        else:
            log(f"dataset_total={total} (inferred via load_dataset)")
    else:
        log(f"dataset_total={total}")

    processed = 0
    correct = 0
    by_scenario = {}
    overlap = 0
    agree = 0
    ref_correct = 0
    run_correct_on_overlap = 0

    with open(answers_path, "w", encoding="utf-8", buffering=1) as out:
        for idx, item in enumerate(tqdm(data_iter, desc="MagicLens-rerank + LLaVA")):
            if idx < args.start_index:
                continue
            if args.max_samples > 0 and processed >= args.max_samples:
                break

            question_blob = item.get("prompt_question_part", item["question"])
            question_text, _ = parse_question_and_options(question_blob)
            image_files = trim_rag_images(item["image_files"], args.max_rag_images)

            rerank_info = []
            if not args.disable_magiclens_rerank:
                image_files, rerank_info = rerank_rag_images(
                    ml_encode_fn, tokenizer_fn, question_text, image_files
                )

            raw_output = llava_answer(
                llava_tokenizer,
                llava_model,
                llava_image_processor,
                item,
                image_files,
                args,
            )
            pred_choice = extract_choice(raw_output)

            qs_id = str(item["id"])
            gt_choice = str(item["gt_choice"])
            scenario = str(item["scenario"])

            row = {
                "qs_id": qs_id,
                "prompt": item["prompt"],
                "output": raw_output,
                "gt_answer": item["answer"],
                "shortuuid": uuid.uuid4().hex,
                "model_id": f"llava_qwen7b_magiclens_rerank_{args.magiclens_model_size}",
                "gt_choice": gt_choice,
                "scenario": scenario,
                "aspect": item["aspect"],
                "meta_pred_choice": pred_choice,
                "meta_rag_count": max(0, len(image_files) - 1),
                "meta_rag_source": rag_source,
                "meta_magiclens_rerank_disabled": bool(args.disable_magiclens_rerank),
                "meta_magiclens_rag_ranks": rerank_info,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

            processed += 1
            is_correct = pred_choice == gt_choice
            if is_correct:
                correct += 1

            stat = by_scenario.setdefault(scenario, {"total": 0, "correct": 0})
            stat["total"] += 1
            if is_correct:
                stat["correct"] += 1

            ref_choice = ref_preds.get(qs_id)
            if ref_choice is not None:
                overlap += 1
                if is_correct:
                    run_correct_on_overlap += 1
                if ref_choice == gt_choice:
                    ref_correct += 1
                if ref_choice == pred_choice:
                    agree += 1

            if cache_clear_every > 0 and processed % cache_clear_every == 0:
                try:
                    jax.clear_caches()
                except Exception:
                    pass
                gc.collect()
                log(f"cleared_jax_caches_at={processed}")

    by_scenario_acc = {}
    for scenario, stat in by_scenario.items():
        total_s = max(1, stat["total"])
        by_scenario_acc[scenario] = round(stat["correct"] * 100.0 / total_s, 2)

    overall_acc = round(correct * 100.0 / max(1, processed), 2)
    summary = {
        "answers_file": str(answers_path),
        "total_processed": processed,
        "overall_accuracy": overall_acc,
        "mode": "magiclens_rerank_plus_llava_answer",
        "rag_source": rag_source,
        "use_gt": bool(args.use_gt),
        "max_rag_images": args.max_rag_images,
        "llava_model_path": args.llava_model_path,
        "llava_num_beams": args.llava_num_beams,
        "llava_max_new_tokens": args.llava_max_new_tokens,
        "magiclens_rerank_disabled": bool(args.disable_magiclens_rerank),
        "magiclens_disable_jit": bool(disable_jit),
        "magiclens_clear_cache_every": int(cache_clear_every),
        "by_scenario": by_scenario_acc,
    }

    if overlap > 0:
        run_acc_on_overlap = round(run_correct_on_overlap * 100.0 / overlap, 2)
        ref_acc = round(ref_correct * 100.0 / overlap, 2)
        agree_rate = round(agree * 100.0 / overlap, 2)
        summary["reference_compare"] = {
            "reference_file": args.llava_greedy_jsonl,
            "overlap_samples": overlap,
            "agree_count": agree,
            "agree_rate": agree_rate,
            "reference_accuracy": ref_acc,
            "run_accuracy_on_overlap": run_acc_on_overlap,
            "accuracy_delta_vs_reference": round(run_acc_on_overlap - ref_acc, 2),
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"answers_file={answers_path}")
    log(f"summary_file={summary_path}")
    log(f"overall_accuracy={overall_acc}% ({correct}/{processed})")
    if overlap > 0:
        cmp = summary["reference_compare"]
        log(
            "reference_agree_rate="
            f"{cmp['agree_rate']}% ({cmp['agree_count']}/{cmp['overlap_samples']})"
        )
        log(f"accuracy_delta_vs_reference={cmp['accuracy_delta_vs_reference']}%")


if __name__ == "__main__":
    main()
