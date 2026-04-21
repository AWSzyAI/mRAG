#!/usr/bin/env python3
"""
End-to-end multi-dimension RAG pipeline (MRAG-Bench + MagicLens + LLaVA).

Module layout (all reusable logic lives under ``src/mrag/``):

1. ``src.mrag.query_planner`` — build chat prompts and call an LLM (API or local HF)
   to produce N complementary English retrieval instructions.
2. ``src.mrag.magiclens`` — encode query image + each instruction; score corpus.
3. ``src.mrag.multi_dim_pipeline`` — run retrieval per instruction, then fuse lists.
4. ``src.mrag.fusion`` — RRF / score-sum / voting (used by multi_dim_pipeline).
5. ``src.mrag.indexing`` — load or build cached MagicLens corpus embeddings.
6. ``benchmark_corpus_rag`` (test/) — dataset iteration, LLaVA I/O, logging helpers.

Environment (copy ``example.env`` to ``.env`` at repo root; do not commit secrets):

  - ``DIM_GENERATOR_API_KEY``, ``DIM_GENERATOR_API_BASE``, ``DIM_GENERATOR_MODEL`` — API planner.
  - ``HF_TOKEN`` — Hugging Face (downloads / gated Gemma).
  - ``GEMMA4_LOCAL_DIR``, ``GEMMA4_DEVICE``, ``GEMMA4_MODEL_ID`` — local Gemma4 multimodal planner.
  - ``MAGICLENS_BPE_PATH`` optional; corpus / model paths via CLI defaults.

  **无网 GPU 节点**：在**有网机器（如 infinity）**完成下载与缓存，再同步到 GPU；见 ``doc/GEMMA4_MULTI_DIM_PIPELINE.md`` 的「有网 / 无网分工」。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _preload_dotenv_before_jax() -> None:
    """Load ``.env`` before ``import jax`` so ``JAX_PLATFORMS`` / offline hub flags apply."""
    p = ROOT_DIR / ".env"
    if not p.is_file():
        return
    try:
        from src.mrag.envfile import load_dotenv

        load_dotenv(p)
    except Exception:
        try:
            raw = p.read_text(encoding="utf-8")
            for line in raw.splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, val = s.split("=", 1)
                k = k.strip()
                if not k:
                    continue
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                    val = val[1:-1]
                cur = os.environ.get(k)
                if cur is None or cur == "":
                    os.environ[k] = val
        except OSError:
            pass


_preload_dotenv_before_jax()

import jax
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
from tqdm.auto import tqdm

sys.path.append(str(ROOT_DIR / "github/LLaVA-NeXT"))
sys.path.append(str(ROOT_DIR / "github/MRAG-Bench/eval"))
sys.path.append(str(ROOT_DIR / "github/magiclens"))

from inference import load_model as load_magiclens_model  # noqa: E402

from benchmark_corpus_rag import (  # noqa: E402
    extract_choice,
    iter_bench_queries,
    llava_answer,
    load_llava,
    log,
    log_torch_cuda_env,
    parse_question_and_options,
    resolve_bpe_path,
)
from src.mrag import envfile as core_envfile
from src.mrag import gemma4_dims as core_gemma4_dims
from src.mrag import gemma4_loader as core_gemma4_loader
from src.mrag import indexing as core_indexing
from src.mrag import magiclens as core_magiclens
from src.mrag import mrag_bench as core_mrag_bench
from src.mrag import multi_dim_pipeline as mdp
from src.mrag import query_planner as qp


def _resolve_repo_path(path_str: str, root: Path) -> Path:
    p = Path(path_str).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-dimension query LLM + MagicLens + fusion + LLaVA pipeline")
    p.add_argument("--dataset-name", type=str, default="uclanlp/MRAG-Bench")
    p.add_argument("--corpus-dir", type=str, required=True)
    p.add_argument("--corpus-cache-dir", type=str, default=str(ROOT_DIR / "results/corpus_index"))

    p.add_argument("--n-dims", type=int, default=3)
    p.add_argument("--dim-top-k", type=int, default=5)
    p.add_argument("--final-top-k", type=int, default=5)
    p.add_argument("--fusion-strategy", type=str, default="rrf", choices=sorted(mdp.FUSION_STRATEGIES))

    p.add_argument(
        "--dim-generator-type",
        type=str,
        default=os.environ.get("DIM_GENERATOR_TYPE", "api"),
        choices=["api", "local", "gemma4_local"],
        help="api=OpenAI-compatible HTTP; local=HF text-generation pipeline; gemma4_local=Gemma4 multimodal (query image + question).",
    )
    p.add_argument(
        "--dim-generator-model",
        type=str,
        default=os.environ.get("DIM_GENERATOR_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        help="API / local HF model id for dimension lines (ignored when type=gemma4_local).",
    )
    p.add_argument(
        "--dim-generator-api-base",
        type=str,
        default=os.environ.get("DIM_GENERATOR_API_BASE", "https://api.siliconflow.cn/v1"),
    )
    p.add_argument(
        "--dim-generator-api-key",
        type=str,
        default=os.environ.get("DIM_GENERATOR_API_KEY", ""),
        help="Prefer setting DIM_GENERATOR_API_KEY in .env (not committed).",
    )
    p.add_argument("--dim-generator-temperature", type=float, default=0.3)
    p.add_argument("--fallback-instruction", type=str, default="")

    p.add_argument(
        "--gemma4-local-dir",
        type=str,
        default=os.environ.get("GEMMA4_LOCAL_DIR", str(ROOT_DIR / "models" / "gemma4-e2b")),
        help="Local snapshot dir with config.json (E2B-it weights).",
    )
    p.add_argument(
        "--gemma4-model-id",
        type=str,
        default=os.environ.get("GEMMA4_MODEL_ID", "google/gemma-4-E2B-it"),
        help="Hub id used when local dir has no weights (download separately).",
    )
    p.add_argument("--gemma4-device", type=str, default=os.environ.get("GEMMA4_DEVICE", "cuda:1"))
    p.add_argument(
        "--gemma4-max-new-tokens",
        type=int,
        default=int((os.environ.get("GEMMA4_DIM_MAX_NEW_TOKENS") or "512").strip() or "512"),
    )
    p.add_argument(
        "--gemma4-hf-token",
        type=str,
        default=os.environ.get("GEMMA4_HF_TOKEN", "") or os.environ.get("HF_TOKEN", ""),
        help="Defaults to HF_TOKEN from .env if set.",
    )
    p.add_argument(
        "--gemma4-allow-torch-below-2-4",
        action="store_true",
        help="Skip PyTorch>=2.4 guard (not recommended).",
    )

    p.add_argument("--magiclens-model-path", type=str, default=str(ROOT_DIR / "models/magic_lens_clip_base.pkl"))
    p.add_argument("--magiclens-model-size", type=str, default="base", choices=["base", "large"])
    p.add_argument("--magiclens-batch-size", type=int, default=16)
    p.add_argument("--bpe-path", type=str, default="")
    p.add_argument("--magiclens-disable-jit", action="store_true")

    p.add_argument("--llava-model-path", type=str, default=str(ROOT_DIR / "models/llava-onevision-qwen2-7b-ov"))
    p.add_argument("--llava-device-map", type=str, default="auto")
    p.add_argument("--llava-attn-implementation", type=str, default="sdpa")
    p.add_argument("--llava-load-4bit", action="store_true")
    p.add_argument("--llava-load-8bit", action="store_true")
    p.add_argument("--llava-allow-cpu-offload", action="store_true")
    p.add_argument("--llava-max-new-tokens", type=int, default=4096)
    p.add_argument("--llava-num-beams", type=int, default=1)

    p.add_argument("--answers-file", type=str, default=str(ROOT_DIR / "log/E8/e8_multi_dim_rag_results.jsonl"))
    p.add_argument("--summary-out", type=str, default=str(ROOT_DIR / "log/E8/e8_multi_dim_rag_summary.json"))
    p.add_argument("--save-dimensions-jsonl", type=str, default="")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0)
    return p


def run_benchmark(args: argparse.Namespace) -> None:
    core_envfile.load_dotenv(ROOT_DIR / ".env")
    core_mrag_bench.ensure_mrag_hf_cache_env()

    answers_path = Path(args.answers_file)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    dims_path = Path(args.save_dimensions_jsonl) if args.save_dimensions_jsonl else None
    if dims_path:
        dims_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = (args.dim_generator_api_key or os.environ.get("DIM_GENERATOR_API_KEY", "")).strip()
    dim_model_tag = args.gemma4_model_id if args.dim_generator_type == "gemma4_local" else args.dim_generator_model

    log_torch_cuda_env()
    log(
        f"n_dims={args.n_dims} dim_top_k={args.dim_top_k} final_top_k={args.final_top_k} fusion={args.fusion_strategy}"
    )
    log(f"dim_generator_type={args.dim_generator_type} dim_model={dim_model_tag}")

    bpe_path = resolve_bpe_path(args.bpe_path)
    tokenizer_fn = (
        clip_tokenizer.build_tokenizer(bpe_path=bpe_path)
        if bpe_path
        else clip_tokenizer.build_tokenizer()
    )
    ml_model, ml_params = load_magiclens_model(args.magiclens_model_size, args.magiclens_model_path)
    disable_jit = bool(args.magiclens_disable_jit)
    if jax.default_backend() == "cpu" and not disable_jit:
        disable_jit = True
    encode_fn = core_magiclens.build_magiclens_encoder(ml_model, ml_params, disable_jit=disable_jit)
    log("MagicLens ready")

    corpus_paths, corpus_embeds = core_indexing.load_or_build_magiclens_corpus_index(args, encode_fn, tokenizer_fn)
    log(f"corpus_size={len(corpus_paths)}")

    llava_tokenizer, llava_model, llava_image_processor = load_llava(args)

    local_pipeline = None
    gemma_processor = None
    gemma_model = None
    if args.dim_generator_type == "local":
        local_pipeline = qp.load_local_pipeline(
            args.dim_generator_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        )
        log(f"local_dim_generator_ready={args.dim_generator_model}")

    if args.dim_generator_type == "gemma4_local":
        gdir = _resolve_repo_path(args.gemma4_local_dir, ROOT_DIR)
        token = (args.gemma4_hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None
        dev = args.gemma4_device.strip()
        if dev.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(dev)
        device_map: str | dict = dev if dev.startswith("cuda") else "auto"
        if dev == "cpu":
            device_map = {"": "cpu"}
        if not (gdir / "config.json").is_file():
            log(f"warning: no config.json under {gdir} — from_pretrained may hit the Hub (needs HF_TOKEN if gated).")
        gemma_processor, gemma_model = core_gemma4_loader.load_processor_and_model(
            args.gemma4_model_id,
            gdir,
            device_map=device_map,
            token=token,
            allow_torch_below_2_4=bool(args.gemma4_allow_torch_below_2_4),
        )
        gemma_model.eval()
        log(f"gemma4_dim_generator_ready model_id={args.gemma4_model_id} local_dir={gdir} device={dev}")

    try:
        total = len(load_dataset(args.dataset_name, split="test"))
    except Exception:
        total = None

    processed = 0
    correct = 0
    by_scenario: dict[str, dict] = {}
    dim_gen_times: list[float] = []
    retrieval_times: list[float] = []
    dim_gen_failures = 0

    dims_out = open(dims_path, "w", encoding="utf-8") if dims_path else None

    with open(answers_path, "w", encoding="utf-8", buffering=1) as out:
        for idx, item in enumerate(tqdm(iter_bench_queries(args.dataset_name), total=total, desc="pipeline-multi-dim")):
            if idx < args.start_index:
                continue
            if args.max_samples > 0 and processed >= args.max_samples:
                break

            question_text, _ = parse_question_and_options(item["prompt_question_part"])

            t0 = time.time()
            try:
                if args.dim_generator_type == "api":
                    dim_instructions = qp.generate_retrieval_instructions(
                        question_text,
                        args.n_dims,
                        backend="api",
                        api_base=args.dim_generator_api_base,
                        api_key=api_key,
                        api_model=args.dim_generator_model,
                        temperature=args.dim_generator_temperature,
                    )
                elif args.dim_generator_type == "gemma4_local":
                    dim_instructions = core_gemma4_dims.generate_retrieval_instructions_gemma4(
                        gemma_processor,
                        gemma_model,
                        query_image=item["query_image"],
                        question=question_text,
                        n_dims=args.n_dims,
                        max_new_tokens=args.gemma4_max_new_tokens,
                    )
                else:
                    dim_instructions = qp.generate_retrieval_instructions(
                        question_text,
                        args.n_dims,
                        backend="local",
                        local_pipeline=local_pipeline,
                    )
            except Exception as e:
                log(f"dimension_gen_error qs_id={item.get('id')}: {e}")
                dim_instructions = []
            t_dim = time.time() - t0
            dim_gen_times.append(t_dim)

            if not dim_instructions:
                dim_gen_failures += 1
                dim_instructions = (
                    [question_text] if not args.fallback_instruction else [args.fallback_instruction]
                )

            t0 = time.time()
            per_dim, fused = mdp.multi_dim_magiclens_retrieve_and_fuse(
                item["query_image"],
                dim_instructions,
                corpus_paths,
                corpus_embeds,
                encode_fn,
                tokenizer_fn,
                dim_top_k=args.dim_top_k,
                fusion_strategy=args.fusion_strategy,
                final_top_k=args.final_top_k,
            )
            t_ret = time.time() - t0
            retrieval_times.append(t_ret)

            image_files = [item["query_image"]]
            for fi in fused:
                with Image.open(fi["path"]) as img:
                    image_files.append(img.convert("RGB"))

            raw_output = llava_answer(llava_tokenizer, llava_model, llava_image_processor, item, image_files, args)
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
                "model_id": f"llava_qwen7b_multidim_{dim_model_tag.split('/')[-1]}_{args.fusion_strategy}",
                "gt_choice": gt_choice,
                "scenario": scenario,
                "aspect": item["aspect"],
                "meta_pred_choice": pred_choice,
                "meta_rag_count": len(fused),
                "meta_rag_source": "multi_dimension_magiclens_pipeline",
                "meta_n_dims": args.n_dims,
                "meta_dim_instructions": dim_instructions,
                "meta_fusion_strategy": args.fusion_strategy,
                "meta_dim_generator_model": dim_model_tag,
                "meta_dim_generator_type": args.dim_generator_type,
                "meta_fused_retrieval": [
                    {"rank": f["rank"], "path": f["path"], "fusion_score": f.get("fusion_score", f.get("fused_score"))}
                    for f in fused
                ],
                "meta_per_dim_retrieval": [
                    [{"rank": r["rank"], "path": r["path"], "score": r["score"]} for r in dim_res]
                    for dim_res in per_dim
                ],
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

            if dims_out:
                dims_out.write(
                    json.dumps(
                        {"qs_id": qs_id, "question": question_text, "instructions": dim_instructions, "dim_gen_time": round(t_dim, 3)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            processed += 1
            is_correct = pred_choice == gt_choice
            if is_correct:
                correct += 1
            stat = by_scenario.setdefault(scenario, {"total": 0, "correct": 0})
            stat["total"] += 1
            if is_correct:
                stat["correct"] += 1

    if dims_out:
        dims_out.close()

    by_scenario_acc = {sc: round(100.0 * st["correct"] / max(1, st["total"]), 2) for sc, st in by_scenario.items()}
    summary = {
        "dataset_name": args.dataset_name,
        "processed": processed,
        "correct": correct,
        "accuracy": round(100.0 * correct / max(1, processed), 2),
        "n_dims": args.n_dims,
        "dim_top_k": args.dim_top_k,
        "final_top_k": args.final_top_k,
        "fusion_strategy": args.fusion_strategy,
        "dim_generator_type": args.dim_generator_type,
        "dim_generator_model": dim_model_tag,
        "dim_gen_failures": dim_gen_failures,
        "avg_dim_gen_time_sec": round(sum(dim_gen_times) / max(1, len(dim_gen_times)), 3),
        "avg_retrieval_time_sec": round(sum(retrieval_times) / max(1, len(retrieval_times)), 3),
        "by_scenario_accuracy": by_scenario_acc,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"summary_saved={summary_path}")
    log(f"accuracy={summary['accuracy']}% processed={processed} dim_gen_failures={dim_gen_failures}")


def main() -> None:
    core_envfile.load_dotenv(ROOT_DIR / ".env")
    parser = build_arg_parser()
    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
