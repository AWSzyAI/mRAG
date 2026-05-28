#!/usr/bin/env python3
"""Run real E11_4-style retrieval and open-ended answering on InfoSeek samples."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
TEST_DIR = ROOT_DIR / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
sys.path.append(str(ROOT_DIR / "github" / "magiclens"))
sys.path.append(str(ROOT_DIR / "github" / "LLaVA-NeXT"))
sys.path.append(str(ROOT_DIR / "github" / "scenic"))

from src.mrag import envfile as core_envfile  # noqa: E402
from src.mrag import gemma4_dims as core_gemma4_dims  # noqa: E402
from src.mrag import gemma4_loader as core_gemma4_loader  # noqa: E402


@dataclass
class InfoSeekResult:
    sample_id: int
    data_id: str
    image_id: str
    question: str
    query_image_path: str = ""
    query_dims: list[str] = field(default_factory=list)
    query_rationales: list[str] = field(default_factory=list)
    retrieval_queries: list[str] = field(default_factory=list)
    query_gen_time_sec: float = 0.0
    query_gen_error: str | None = None
    per_dim_retrieval: list[list[dict[str, Any]]] = field(default_factory=list)
    retrieval_results: list[dict[str, Any]] = field(default_factory=list)
    retrieval_time_sec: float = 0.0
    retrieval_error: str | None = None
    answerer: str = "gemma4"
    predicted_answer: str = ""
    raw_output: str = ""
    answer_gen_time_sec: float = 0.0
    answer_gen_error: str | None = None
    answer_input_images: int = 0
    reference_answers: list[str] = field(default_factory=list)
    exact_match: bool = False
    fuzzy_match: bool = False
    status: str = "pending"
    total_time_sec: float = 0.0
    error_message: str | None = None


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [E11_4_InfoSeek_REAL] {msg}", flush=True)


def load_sample_metadata(sample_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with open(sample_dir / "sample_metadata.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])
    meta = {
        "split": data.get("split"),
        "total_sampled": data.get("total_sampled"),
        "random_seed": data.get("random_seed"),
    }
    log(f"loaded_samples={len(samples)} seed={meta['random_seed']}")
    return meta, samples


def load_reference_answers(path: str | None) -> dict[str, list[str]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, list[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            out[str(key)] = [str(v) for v in value if str(v).strip()]
        elif isinstance(value, str) and value.strip():
            out[str(key)] = [value.strip()]
        elif isinstance(value, dict):
            vals = value.get("answers") or value.get("reference_answers") or value.get("answer")
            if isinstance(vals, list):
                out[str(key)] = [str(v) for v in vals if str(v).strip()]
            elif isinstance(vals, str) and vals.strip():
                out[str(key)] = [vals.strip()]
    return out


def normalize_answer(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^(?:the\s+answer\s+is|answer\s*:|answer\s+is)\s+", "", text, flags=re.I)
    return " ".join(text.split())


def evaluate_answer(predicted: str, refs: list[str]) -> tuple[bool, bool]:
    pred = normalize_answer(predicted).lower()
    refs_norm = [normalize_answer(r).lower() for r in refs if normalize_answer(r)]
    if not pred or not refs_norm:
        return False, False
    if any(pred == ref for ref in refs_norm):
        return True, True
    pred_words = set(re.findall(r"[a-z0-9]+", pred))
    for ref in refs_norm:
        ref_words = set(re.findall(r"[a-z0-9]+", ref))
        if pred_words and ref_words and len(pred_words & ref_words) / max(len(pred_words), len(ref_words)) >= 0.5:
            return False, True
    return False, False


def resolve_image_path(image_dir: Path, image_id: str) -> Path:
    stem = image_id
    candidates = []
    for suffix in ("", ".jpg", ".jpeg", ".png", ".JPG"):
        candidates.append(image_dir / f"{stem}{suffix}")
    if image_dir.name == "all":
        parent_images = image_dir.parent
        for suffix in ("", ".jpg", ".jpeg", ".png", ".JPG"):
            candidates.append(parent_images / f"{stem}{suffix}")
    else:
        all_dir = image_dir / "all"
        for suffix in ("", ".jpg", ".jpeg", ".png", ".JPG"):
            candidates.append(all_dir / f"{stem}{suffix}")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"image not found for image_id={image_id} under {image_dir}")


def compact_rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": int(r.get("rank", i + 1)),
            "path": str(r.get("path", "")),
            "score": float(r.get("score", 0.0)),
        }
        for i, r in enumerate(rows)
    ]


def compact_fused_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, r in enumerate(rows):
        row = {
            "rank": int(r.get("rank", i + 1)),
            "path": str(r.get("path", "")),
        }
        if "fusion_score" in r:
            row["fusion_score"] = float(r["fusion_score"])
        elif "fused_score" in r:
            row["fusion_score"] = float(r["fused_score"])
        if "score" in r:
            row["score"] = float(r["score"])
        out.append(row)
    return out


def answer_open_with_gemma4(processor, model, image_paths: list[str], question: str, max_new_tokens: int) -> str:
    content: list[dict[str, str]] = []
    for path in image_paths:
        content.append({"type": "image", "image": str(Path(path).expanduser().resolve())})
    content.append(
        {
            "type": "text",
            "text": (
                "Answer this open-ended visual question concisely.\n"
                "The first image is the query image. Later images are retrieved visual evidence.\n"
                "Use the evidence only if it helps. Return a short answer, not a letter and not an explanation.\n\n"
                f"Question: {question}"
            ),
        }
    )
    inputs = core_gemma4_loader.prepare_inputs(processor, model, [{"role": "user", "content": content}])
    input_len = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
    text = processor.tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()
    if hasattr(processor, "parse_response"):
        try:
            parsed = processor.parse_response(text)
            if isinstance(parsed, dict) and parsed.get("content"):
                text = str(parsed["content"]).strip()
        except Exception:
            pass
    return normalize_answer(text)


def load_llava_open_helpers():
    import pipeline_multi_dim_rag as pmr

    load_llava, _ = pmr._load_llava_helpers()
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    def answer(tokenizer, model, image_processor, question: str, pil_images: list[Image.Image], args) -> str:
        if len(pil_images) <= 1:
            instruction = "Answer this open-ended visual question concisely. "
        else:
            instruction = (
                "You will be given one query image followed by retrieved visual evidence images. "
                "Answer this open-ended visual question concisely. "
            )
        image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * max(1, len(pil_images)))
        prompt_question = f"{instruction}{image_tokens}\nQuestion: {question}"
        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], prompt_question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        model_device = next(model.parameters()).device
        image_dtype = torch.float16 if model_device.type == "cuda" else torch.float32
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model_device)
        image_tensors = process_images(pil_images, image_processor, model.config)
        image_tensors = [img.to(dtype=image_dtype, device=model_device) for img in image_tensors]
        image_sizes = [img.size for img in pil_images]
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
        return normalize_answer(tokenizer.batch_decode(cont, skip_special_tokens=True)[0])

    return load_llava, answer


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real E11_4 InfoSeek open-ended benchmark")
    p.add_argument("--sample-dir", required=True)
    p.add_argument("--image-dir", default="data/infoseek/images/all")
    p.add_argument("--output-dir", default="")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--reference-answers", default="")
    p.add_argument("--resume-from-existing", action="store_true")

    p.add_argument("--corpus-dir", default=os.environ.get("CORPUS_DIR", "data/image_corpus"))
    p.add_argument("--retriever", choices=["magiclens", "none"], default=os.environ.get("RETRIEVER", "magiclens"))
    p.add_argument("--corpus-cache-dir", default=os.environ.get("CORPUS_CACHE_DIR", str(ROOT_DIR / ".cache" / "mrag")))
    p.add_argument("--n-dims", type=int, default=4)
    p.add_argument("--dim-top-k", type=int, default=5)
    p.add_argument("--final-top-k", type=int, default=5)
    p.add_argument("--fusion-strategy", choices=["rrf", "score_sum", "voting"], default="rrf")
    p.add_argument("--dim-generator-type", choices=["gemma4_local", "raw_question", "heuristic"], default="gemma4_local")
    p.add_argument("--gemma4-local-dir", default=os.environ.get("GEMMA4_LOCAL_DIR", str(ROOT_DIR / "models" / "gemma4-e2b")))
    p.add_argument("--gemma4-model-id", default=os.environ.get("GEMMA4_MODEL_ID", "google/gemma-4-E2B-it"))
    p.add_argument("--gemma4-device", default=os.environ.get("GEMMA4_DEVICE", "cuda:0"))
    p.add_argument("--gemma4-max-new-tokens", type=int, default=256)
    p.add_argument("--gemma4-dim-rationale", action="store_true")
    p.add_argument("--gemma4-answer-max-new-tokens", type=int, default=64)
    p.add_argument("--gemma4-answer-max-images", type=int, default=6)
    p.add_argument("--gemma4-allow-torch-below-2-4", action="store_true")
    p.add_argument("--gemma4-hf-token", default="")

    p.add_argument("--magiclens-model-path", default=str(ROOT_DIR / "models" / "magic_lens_clip_base.pkl"))
    p.add_argument("--magiclens-model-size", choices=["base", "large"], default="base")
    p.add_argument("--magiclens-batch-size", type=int, default=16)
    p.add_argument("--magiclens-disable-jit", action="store_true")
    p.add_argument("--magiclens-platform", default=os.environ.get("MAGICLENS_PLATFORM", "cpu"))

    p.add_argument("--final-answerer", choices=["gemma4", "llava", "none"], default=os.environ.get("FINAL_ANSWERER", "gemma4"))
    p.add_argument("--llava-model-path", default=os.environ.get("LLAVA_MODEL_PATH", str(ROOT_DIR / "models" / "llava-onevision-qwen2-7b-ov")))
    p.add_argument("--llava-device-map", default=os.environ.get("LLAVA_DEVICE_MAP", "auto"))
    p.add_argument("--llava-attn-implementation", default=os.environ.get("LLAVA_ATTN_IMPLEMENTATION", "sdpa"))
    p.add_argument("--llava-load-4bit", action="store_true")
    p.add_argument("--llava-load-8bit", action="store_true")
    p.add_argument("--llava-allow-cpu-offload", action="store_true")
    p.add_argument("--llava-max-images", type=int, default=1)
    p.add_argument("--llava-max-new-tokens", type=int, default=64)
    p.add_argument("--llava-num-beams", type=int, default=1)
    return p


def heuristic_dims(question: str, n_dims: int) -> list[str]:
    core = question.rstrip("?")
    dims = [
        f"Identify the entity or object in the query image needed to answer: {core}",
        f"Find visual evidence and context related to: {core}",
        f"Look for labels, attributes, or distinctive details relevant to: {core}",
        f"Find supporting examples or similar images that may reveal: {core}",
    ]
    while len(dims) < n_dims:
        dims.append(f"Find additional visual evidence related to: {core}")
    return dims[:n_dims]


def run(args: argparse.Namespace) -> None:
    core_envfile.load_dotenv(ROOT_DIR / ".env")
    log(f"python={sys.version.split()[0]} platform={platform.platform()}")
    log(f"torch_cuda_available={torch.cuda.is_available()} cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    log(
        "config "
        f"retriever={args.retriever} dim_generator={args.dim_generator_type} "
        f"answerer={args.final_answerer} magiclens_platform={args.magiclens_platform} "
        f"max_samples={args.max_samples}"
    )

    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir) if args.output_dir else sample_dir.parent / "benchmark_real"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "e11_4_infoseek_results.jsonl"
    summary_path = output_dir / "e11_4_infoseek_summary.json"
    log(f"outputs results={results_path} summary={summary_path}")

    metadata, samples = load_sample_metadata(sample_dir)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
        log(f"max_samples applied; running_samples={len(samples)}")
    refs_by_id = load_reference_answers(args.reference_answers)

    seen: set[str] = set()
    if args.resume_from_existing and results_path.is_file():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        seen.add(str(row.get("data_id", "")))
                    except json.JSONDecodeError:
                        pass
        log(f"resume_from_existing=1 seen={len(seen)}")

    tokenizer_fn = encode_fn = corpus_paths = corpus_embeds = None
    if args.retriever == "magiclens":
        log("MagicLens stage: importing jax/pipeline/model helpers")
        import jax
        import pipeline_multi_dim_rag as pmr
        from inference import load_model as load_magiclens_model
        from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
        from src.mrag import indexing as core_indexing
        from src.mrag import magiclens as core_magiclens

        if args.magiclens_platform:
            jax.config.update("jax_platforms", args.magiclens_platform)
        log(f"jax_backend={jax.default_backend()}")
        bpe_path = pmr.resolve_bpe_path("")
        log(f"MagicLens stage: building tokenizer bpe_path={bpe_path or '<default>'}")
        tokenizer_fn = clip_tokenizer.build_tokenizer(bpe_path=bpe_path) if bpe_path else clip_tokenizer.build_tokenizer()
        log(f"MagicLens stage: loading model path={args.magiclens_model_path} size={args.magiclens_model_size}")
        ml_model, ml_params = load_magiclens_model(args.magiclens_model_size, args.magiclens_model_path)
        disable_jit = bool(args.magiclens_disable_jit) or jax.default_backend() == "cpu"
        encode_fn = core_magiclens.build_magiclens_encoder(ml_model, ml_params, disable_jit=disable_jit)
        log(f"MagicLens stage: encoder ready disable_jit={disable_jit}")
        log(f"MagicLens stage: loading/building corpus index corpus_dir={args.corpus_dir} cache_dir={args.corpus_cache_dir}")
        corpus_paths, corpus_embeds = core_indexing.load_or_build_magiclens_corpus_index(args, encode_fn, tokenizer_fn)
        log(f"corpus_size={len(corpus_paths)}")
    else:
        log("retriever=none; answering query image without retrieved evidence")

    needs_gemma4 = args.dim_generator_type == "gemma4_local" or args.final_answerer == "gemma4"
    gemma_processor = gemma_model = None
    if needs_gemma4:
        dev = args.gemma4_device.strip()
        log(f"Gemma4 stage: loading model local_dir={args.gemma4_local_dir} model_id={args.gemma4_model_id} device={dev}")
        if dev.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(dev)
        device_map: str | dict = dev if dev.startswith("cuda") else "auto"
        if dev == "cpu":
            device_map = {"": "cpu"}
        token = (args.gemma4_hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None
        gemma_processor, gemma_model = core_gemma4_loader.load_processor_and_model(
            args.gemma4_model_id,
            Path(args.gemma4_local_dir),
            device_map=device_map,
            token=token,
            allow_torch_below_2_4=bool(args.gemma4_allow_torch_below_2_4),
        )
        gemma_model.eval()
        log(f"Gemma4 ready device={dev}")

    llava_tokenizer = llava_model = llava_image_processor = llava_answer = None
    if args.final_answerer == "llava":
        log(f"LLaVA stage: loading model path={args.llava_model_path}")
        load_llava, llava_answer = load_llava_open_helpers()
        llava_tokenizer, llava_model, llava_image_processor = load_llava(args)
        log("LLaVA ready")

    image_dir = Path(args.image_dir)
    out_mode = "a" if args.resume_from_existing else "w"
    processed = completed = failed = exact = fuzzy = missing_refs = 0
    dim_times: list[float] = []
    ret_times: list[float] = []
    ans_times: list[float] = []
    total_t0 = time.time()
    log(f"loop_start samples={len(samples)} resume_skipped={len(seen)}")

    with open(results_path, out_mode, encoding="utf-8", buffering=1) as out:
        for idx, sample in enumerate(samples):
            data_id = str(sample.get("data_id", ""))
            if data_id in seen:
                continue
            if processed == 0:
                log(f"first_sample data_id={data_id} image_id={sample.get('image_id', '')}")
            if processed and processed % 10 == 0:
                log(f"progress processed={processed}/{len(samples)} completed={completed} failed={failed}")
            sample_t0 = time.time()
            result = InfoSeekResult(
                sample_id=int(sample.get("sample_id", idx)),
                data_id=data_id,
                image_id=str(sample.get("image_id", "")),
                question=str(sample.get("question", "")),
                answerer=args.final_answerer,
            )
            try:
                query_image_path = resolve_image_path(image_dir, result.image_id)
                result.query_image_path = str(query_image_path)

                t0 = time.time()
                if args.dim_generator_type == "raw_question":
                    dims = [result.question]
                    rationales = []
                elif args.dim_generator_type == "heuristic":
                    dims = heuristic_dims(result.question, args.n_dims)
                    rationales = []
                else:
                    if args.gemma4_dim_rationale:
                        plan = core_gemma4_dims.generate_retrieval_plan_with_rationales_gemma4(
                            gemma_processor,
                            gemma_model,
                            query_image=query_image_path,
                            question=result.question,
                            n_dims=args.n_dims,
                            max_new_tokens=max(640, args.gemma4_max_new_tokens),
                        )
                        dims = plan.get("queries", [])
                        rationales = plan.get("rationales", [])
                    else:
                        dims = core_gemma4_dims.generate_retrieval_instructions_gemma4(
                            gemma_processor,
                            gemma_model,
                            query_image=query_image_path,
                            question=result.question,
                            n_dims=args.n_dims,
                            max_new_tokens=args.gemma4_max_new_tokens,
                        )
                        rationales = []
                if not dims:
                    dims = heuristic_dims(result.question, args.n_dims)
                    result.query_gen_error = "empty dimension generation; used heuristic fallback"
                result.query_dims = list(dims)[: args.n_dims]
                result.query_rationales = list(rationales)
                result.retrieval_queries = list(result.query_dims)
                result.query_gen_time_sec = time.time() - t0
                dim_times.append(result.query_gen_time_sec)

                fused = []
                if args.retriever == "magiclens":
                    from src.mrag import multi_dim_pipeline as mdp

                    t0 = time.time()
                    per_dim, fused = mdp.multi_dim_magiclens_retrieve_and_fuse(
                        Image.open(query_image_path).convert("RGB"),
                        result.retrieval_queries,
                        corpus_paths,
                        corpus_embeds,
                        encode_fn,
                        tokenizer_fn,
                        dim_top_k=args.dim_top_k,
                        fusion_strategy=args.fusion_strategy,
                        final_top_k=args.final_top_k,
                    )
                    result.per_dim_retrieval = [compact_rank_rows(rows) for rows in per_dim]
                    result.retrieval_results = compact_fused_rows(fused)
                    result.retrieval_time_sec = time.time() - t0
                    ret_times.append(result.retrieval_time_sec)
                else:
                    result.per_dim_retrieval = []
                    result.retrieval_results = []
                    result.retrieval_time_sec = 0.0
                    ret_times.append(0.0)

                t0 = time.time()
                evidence_paths = [str(query_image_path)] + [str(row["path"]) for row in fused]
                if args.final_answerer == "gemma4":
                    evidence_paths = evidence_paths[: max(1, args.gemma4_answer_max_images)]
                    raw = answer_open_with_gemma4(
                        gemma_processor,
                        gemma_model,
                        evidence_paths,
                        result.question,
                        args.gemma4_answer_max_new_tokens,
                    )
                    result.raw_output = raw
                    result.predicted_answer = normalize_answer(raw)
                    result.answer_input_images = len(evidence_paths)
                elif args.final_answerer == "llava":
                    evidence_paths = evidence_paths[: max(1, args.llava_max_images)]
                    pil_images = [Image.open(path).convert("RGB") for path in evidence_paths]
                    raw = llava_answer(llava_tokenizer, llava_model, llava_image_processor, result.question, pil_images, args)
                    result.raw_output = raw
                    result.predicted_answer = normalize_answer(raw)
                    result.answer_input_images = len(pil_images)
                else:
                    result.raw_output = ""
                    result.predicted_answer = ""
                    result.answer_input_images = len(evidence_paths)
                result.answer_gen_time_sec = time.time() - t0
                ans_times.append(result.answer_gen_time_sec)

                result.reference_answers = refs_by_id.get(result.data_id, [])
                if not result.reference_answers:
                    missing_refs += 1
                result.exact_match, result.fuzzy_match = evaluate_answer(result.predicted_answer, result.reference_answers)
                result.status = "completed"
                completed += 1
                if result.exact_match:
                    exact += 1
                if result.fuzzy_match:
                    fuzzy += 1
            except Exception as exc:
                result.status = "failed"
                result.error_message = str(exc)
                failed += 1
                log(f"sample_error data_id={result.data_id} image_id={result.image_id}: {exc}")
            result.total_time_sec = time.time() - sample_t0
            out.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            processed += 1

    total_sec = time.time() - total_t0
    scored = completed - missing_refs
    summary = {
        "experiment": "E11_4_InfoSeek_10K_REAL",
        "dataset": metadata.get("split"),
        "total_samples_requested": len(samples),
        "processed_this_run": processed,
        "completed": completed,
        "failed": failed,
        "missing_reference_answers": missing_refs,
        "scored_samples": max(0, scored),
        "exact_match_count": exact,
        "fuzzy_match_count": fuzzy,
        "exact_match_rate": exact / scored if scored > 0 else None,
        "fuzzy_match_rate": fuzzy / scored if scored > 0 else None,
        "random_seed": metadata.get("random_seed"),
        "n_dims": args.n_dims,
        "dim_top_k": args.dim_top_k,
        "final_top_k": args.final_top_k,
        "retriever": args.retriever,
        "fusion_strategy": args.fusion_strategy,
        "dim_generator_type": args.dim_generator_type,
        "final_answerer": args.final_answerer,
        "magiclens_jax_platform": args.magiclens_platform,
        "avg_dim_gen_time_sec": round(sum(dim_times) / max(1, len(dim_times)), 3),
        "avg_retrieval_time_sec": round(sum(ret_times) / max(1, len(ret_times)), 3),
        "avg_answer_gen_time_sec": round(sum(ans_times) / max(1, len(ans_times)), 3),
        "total_time_sec": round(total_sec, 3),
        "avg_time_per_processed_sample": round(total_sec / max(1, processed), 3),
        "outputs": {
            "results_jsonl": str(results_path),
            "summary_json": str(summary_path),
        },
        "note": "InfoSeek Entity metadata has no built-in GT answers; accuracy is null unless --reference-answers is supplied.",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"summary_saved={summary_path}")
    log(f"completed={completed} failed={failed} scored={scored} total_sec={total_sec:.1f}")


def main() -> None:
    parser = build_arg_parser()
    run(parser.parse_args())


if __name__ == "__main__":
    main()
