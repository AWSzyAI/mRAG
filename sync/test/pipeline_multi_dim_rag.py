#!/usr/bin/env python3
"""
Multi-dimension RAG pipeline (MRAG-Bench + Gemma4/query planner + MagicLens, optional LLaVA).

Module layout (all reusable logic lives under ``src/mrag/``):

1. ``src.mrag.query_planner`` — build chat prompts and call an LLM (API or local HF)
   to produce N complementary English retrieval instructions. The ``raw_question``
   mode skips rewriting and sends the original question+choices to retrieval.
2. ``src.mrag.magiclens`` — encode query image + each instruction; score corpus.
3. ``src.mrag.multi_dim_pipeline`` — run retrieval per instruction, then fuse lists.
4. ``src.mrag.fusion`` — RRF / score-sum / voting (used by multi_dim_pipeline).
5. ``src.mrag.indexing`` — load or build cached MagicLens corpus embeddings.
6. Optional LLaVA or Gemma4 answer stage for end-to-end A/B/C/D evaluation.

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
import platform
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


def _preconfigure_jax_before_import() -> None:
    """Set MagicLens/JAX runtime knobs before JAX sees the process environment."""
    if os.environ.get("MRAG_ALLOW_HF_NETWORK", "").strip().lower() not in ("1", "true", "yes"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    platform = (os.environ.get("MAGICLENS_JAX_PLATFORMS") or "").strip()
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--magiclens-platform" and i + 1 < len(argv):
            platform = argv[i + 1].strip()
            break
        if arg.startswith("--magiclens-platform="):
            platform = arg.split("=", 1)[1].strip()
            break

    if platform:
        os.environ["JAX_PLATFORMS"] = platform
    elif not os.environ.get("JAX_PLATFORMS"):
        os.environ["JAX_PLATFORMS"] = "cpu"


_preload_dotenv_before_jax()
_preconfigure_jax_before_import()

import jax
import numpy as np
import torch
from PIL import Image
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
from tqdm.auto import tqdm

sys.path.append(str(ROOT_DIR / "github/LLaVA-NeXT"))
sys.path.append(str(ROOT_DIR / "github/MRAG-Bench/eval"))
sys.path.append(str(ROOT_DIR / "github/magiclens"))

from inference import load_model as load_magiclens_model  # noqa: E402
from utils.dataloader import bench_data_loader  # noqa: E402

from src.mrag import envfile as core_envfile
from src.mrag import gemma4_dims as core_gemma4_dims
from src.mrag import gemma4_loader as core_gemma4_loader
from src.mrag import indexing as core_indexing
from src.mrag import magiclens as core_magiclens
from src.mrag import mrag_bench as core_mrag_bench
from src.mrag import multi_dim_pipeline as mdp
from src.mrag import query_planner as qp
from src.mrag import runtime as core_runtime
from src.mrag import text as core_text


log = core_runtime.log
log_torch_cuda_env = core_runtime.log_torch_cuda_env
extract_choice = core_text.extract_choice
parse_question_and_options = core_text.parse_question_and_options
resolve_bpe_path = core_text.resolve_bpe_path


def _resolve_repo_path(path_str: str, root: Path) -> Path:
    p = Path(path_str).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def _load_llava_helpers():
    from src.mrag.transformers_llava_compat import ensure_modeling_utils_chunking_compat

    ensure_modeling_utils_chunking_compat()

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    from src.mrag import llava as core_llava

    def _patch_llava_assign_meta_compat() -> None:
        """Fallback to non-assign loading when meta-device guard is triggered."""
        try:
            import llava.model.builder as lbuilder
        except Exception as e:
            log(f"warning: could not import llava.model.builder for assign/meta patch: {e}")
            return

        def _wrap_assign_helper(module_obj, helper_name: str, module_tag: str) -> None:
            helper = getattr(module_obj, helper_name, None)
            if helper is None or getattr(helper, "_mrag_assign_meta_patch", False):
                return

            def patched(model_cls, model_name_or_path, **kwargs):
                try:
                    return helper(model_cls, model_name_or_path, **kwargs)
                except RuntimeError as e:
                    msg = str(e)
                    if "meta device context manager" not in msg and "torch.set_default_device('meta')" not in msg:
                        raise
                    log(f"patched {module_tag}.{helper_name}: retrying without assign=True due to meta-device guard")
                    return model_cls.from_pretrained(model_name_or_path, **kwargs)

            patched._mrag_assign_meta_patch = True
            setattr(module_obj, helper_name, patched)
            log(f"patched {module_tag}.{helper_name} meta-device compatibility")

        _wrap_assign_helper(lbuilder, "_from_pretrained_with_assign", "llava.model.builder")

        try:
            import llava.model.multimodal_encoder.siglip_encoder as siglip_encoder
        except Exception as e:
            log(f"warning: could not import siglip_encoder for assign/meta patch: {e}")
        else:
            _wrap_assign_helper(
                siglip_encoder,
                "_from_pretrained_with_assign",
                "llava.model.multimodal_encoder.siglip_encoder",
            )

    def _patch_llava_qwen_rope_parameters() -> None:
        try:
            from llava.model.language_model import llava_qwen as lq
        except Exception as e:
            log(f"warning: could not patch LlavaQwen rope_parameters: {e}")
            return

        if getattr(lq.LlavaQwenForCausalLM, "_mrag_rope_parameters_patch", False):
            return

        def ensure_rope_parameters(config):
            if getattr(config, "rope_parameters", None) is None:
                config.rope_parameters = {}
            config.rope_parameters.setdefault("rope_type", "default")
            config.rope_parameters.setdefault("rope_theta", getattr(config, "rope_theta", 1000000.0))

        def patched_init(self, config, *model_args, **model_kwargs):
            ensure_rope_parameters(config)
            lq.Qwen2ForCausalLM.__init__(self, config)
            config.model_type = "llava_qwen"
            config.rope_scaling = None
            # Critical for transformers>=4.5x meta-init guard:
            # prevent vision tower nested from_pretrained inside model __init__.
            # It will be loaded later by LLaVA builder once outer model load exits.
            config.delay_load = True
            # Some LLaVA-NeXT checkpoints set mm_tunable_parts to include
            # `mm_vision_tower`, which forces immediate vision-tower load even
            # when delay_load=True. Clear these flags during outer meta-init.
            if getattr(config, "unfreeze_mm_vision_tower", False):
                config.unfreeze_mm_vision_tower = False
            mm_parts = getattr(config, "mm_tunable_parts", None)
            if isinstance(mm_parts, str) and "mm_vision_tower" in mm_parts:
                cleaned = ",".join(
                    part.strip() for part in mm_parts.split(",") if part.strip() and part.strip() != "mm_vision_tower"
                )
                config.mm_tunable_parts = cleaned
            elif isinstance(mm_parts, (list, tuple)) and any(str(part).strip() == "mm_vision_tower" for part in mm_parts):
                config.mm_tunable_parts = [part for part in mm_parts if str(part).strip() != "mm_vision_tower"]
            ensure_rope_parameters(config)

            self.model = lq.LlavaQwenModel(config)
            self.lm_head = lq.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()

        lq.LlavaQwenForCausalLM.__init__ = patched_init
        lq.LlavaQwenForCausalLM._mrag_rope_parameters_patch = True
        log("patched LlavaQwenForCausalLM rope_parameters compatibility")

    _patch_llava_assign_meta_compat()
    _patch_llava_qwen_rope_parameters()

    def load_llava(args):
        return core_llava.load_llava(args, load_pretrained_model, log)

    def llava_answer(tokenizer, model, image_processor, item, image_files, args):
        return core_llava.llava_answer(
            tokenizer,
            model,
            image_processor,
            item,
            image_files,
            args,
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
            conv_templates,
            tokenizer_image_token,
            process_images,
        )

    return load_llava, llava_answer


def _ensure_query_image_path(item: dict, cache_dir: Path) -> Path:
    raw_path = item.get("query_image_path")
    if raw_path:
        path = Path(str(raw_path)).expanduser()
        if path.is_file():
            return path.resolve()

    query_image = _query_image_from_bench_item(item)
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(item["id"]))
    path = cache_dir / f"{safe_id}.png"
    if not path.is_file():
        query_image.save(path)
    return path.resolve()


def _query_image_from_bench_item(item: dict):
    if "query_image" in item:
        return item["query_image"]
    image_files = item.get("image_files") or []
    if image_files:
        return image_files[0]
    raise KeyError("MRAG-Bench item has neither query_image nor image_files[0]")


def _path_payload(path_str: str) -> dict:
    p = Path(path_str)
    return {"path": str(path_str), "filename": p.name, "id": p.stem}


def _rank_row_payload(row: dict) -> dict:
    payload = _path_payload(str(row["path"]))
    payload.update(
        {
            "rank": int(row["rank"]),
            "score": float(row["score"]) if "score" in row else None,
        }
    )
    if "fusion_score" in row:
        payload["fusion_score"] = float(row["fusion_score"])
    if "fusion_votes" in row:
        payload["fusion_votes"] = int(row["fusion_votes"])
    return payload


def _fusion_trace(per_dim: list[list[dict]], fused: list[dict], strategy: str) -> list[dict]:
    by_path: dict[str, list[dict]] = {}
    for dim_idx, rows in enumerate(per_dim, start=1):
        for row in rows:
            path = str(row["path"])
            entry = {
                "dim_index": dim_idx,
                "rank": int(row["rank"]),
                "score": float(row.get("score", 0.0)),
            }
            if strategy == "rrf":
                entry["rrf_k"] = 60
                entry["contribution"] = 1.0 / (60 + int(row["rank"]))
            elif strategy == "score_sum":
                entry["contribution"] = float(row.get("score", 0.0))
            elif strategy == "voting":
                entry["vote"] = 1
                entry["score_contribution"] = float(row.get("score", 0.0))
            by_path.setdefault(path, []).append(entry)

    out = []
    for row in fused:
        path = str(row["path"])
        payload = _rank_row_payload(row)
        payload["source_hits"] = by_path.get(path, [])
        if strategy == "rrf":
            payload["fusion_formula"] = "sum(1 / (60 + per_dimension_rank))"
            payload["fusion_score_recomputed"] = float(sum(h["contribution"] for h in payload["source_hits"]))
        elif strategy == "score_sum":
            payload["fusion_formula"] = "sum(per_dimension_similarity_score)"
            payload["fusion_score_recomputed"] = float(sum(h["contribution"] for h in payload["source_hits"]))
        elif strategy == "voting":
            payload["fusion_formula"] = "sort by vote_count desc, then summed similarity score desc"
            payload["fusion_votes_recomputed"] = int(sum(h["vote"] for h in payload["source_hits"]))
            payload["fusion_score_recomputed"] = float(sum(h["score_contribution"] for h in payload["source_hits"]))
        out.append(payload)
    return out


def _augment_question_with_image_descriptions(question_with_choices: str, descriptions: list[dict]) -> str:
    if not descriptions:
        return question_with_choices
    lines = [
        "Additional visual evidence descriptions generated by Gemma4. "
        "Use them as auxiliary evidence together with the images; answer with only the option letter.",
    ]
    for desc in descriptions:
        lines.append(f"{desc['image_label']} ({desc['filename']}): {desc['description']}")
    return "\n".join(lines) + "\n\n" + question_with_choices


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-dimension query LLM + MagicLens + fusion pipeline")
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
        choices=["api", "local", "gemma4_local", "raw_question"],
        help=(
            "api=OpenAI-compatible HTTP; local=HF text-generation pipeline; "
            "gemma4_local=Gemma4 multimodal (query image + question); "
            "raw_question=no rewrite, use original question+choices as the single retrieval instruction."
        ),
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
        "--gemma4-dim-rationale",
        action="store_true",
        help="For gemma4_local, generate per-dimension rationale and store it in trace/dims outputs.",
    )
    p.add_argument(
        "--dim-retrieval-use-rationale",
        action="store_true",
        help="Prefix each retrieval query with its rationale: '<rationale>; <query>' to inject explicit CoT-style guidance into MagicLens retrieval.",
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
    p.add_argument(
        "--magiclens-platform",
        type=str,
        default=os.environ.get("MAGICLENS_JAX_PLATFORMS", os.environ.get("JAX_PLATFORMS", "cpu")),
        choices=["cpu", "cuda", "gpu"],
        help="JAX platform for MagicLens. Default cpu avoids GPU cuDNN/runtime conflicts on shared nodes.",
    )

    p.add_argument(
        "--final-answerer",
        type=str,
        default=os.environ.get("FINAL_ANSWERER", "none"),
        choices=["none", "llava", "gemma4"],
        help="none writes Gemma4 dimensions + MagicLens candidates only; llava/gemma4 run A/B/C/D evaluation.",
    )
    p.add_argument("--llava-model-path", type=str, default=str(ROOT_DIR / "models/llava-onevision-qwen2-7b-ov"))
    p.add_argument("--llava-device-map", type=str, default="auto")
    p.add_argument("--llava-attn-implementation", type=str, default="sdpa")
    p.add_argument("--llava-load-4bit", action="store_true")
    p.add_argument("--llava-load-8bit", action="store_true")
    p.add_argument("--llava-allow-cpu-offload", action="store_true")
    p.add_argument("--llava-max-new-tokens", type=int, default=4096)
    p.add_argument("--llava-num-beams", type=int, default=1)
    p.add_argument(
        "--llava-max-images",
        type=int,
        default=1,
        help="Maximum number of images passed to LLaVA per sample (including query image). "
        "Use 1 for stability if multi-image inference triggers CUDA device-side asserts.",
    )
    p.add_argument(
        "--gemma4-answer-max-new-tokens",
        type=int,
        default=int((os.environ.get("GEMMA4_ANSWER_MAX_NEW_TOKENS") or "64").strip() or "64"),
        help="Max new tokens for Gemma4 final answerer. Keep small because only A/B/C/D is expected.",
    )
    p.add_argument(
        "--gemma4-answer-max-images",
        type=int,
        default=int((os.environ.get("GEMMA4_ANSWER_MAX_IMAGES") or "6").strip() or "6"),
        help="Maximum images passed to Gemma4 final answerer, including query image.",
    )

    p.add_argument("--answers-file", type=str, default=str(ROOT_DIR / "log/E8/e8_multi_dim_rag_results.jsonl"))
    p.add_argument("--summary-out", type=str, default=str(ROOT_DIR / "log/E8/e8_multi_dim_rag_summary.json"))
    p.add_argument("--save-dimensions-jsonl", type=str, default="")
    p.add_argument(
        "--trace-jsonl",
        type=str,
        default="",
        help="Optional detailed per-sample trace JSONL with prompts, 5x5 retrieval, fusion, descriptions, timings, and scoring.",
    )
    p.add_argument(
        "--describe-final-images",
        action="store_true",
        help="Use Gemma4 to describe the query image and final fused images, then append descriptions to the LLaVA question.",
    )
    p.add_argument(
        "--gemma4-description-max-new-tokens",
        type=int,
        default=int((os.environ.get("GEMMA4_DESCRIPTION_MAX_NEW_TOKENS") or "160").strip() or "160"),
    )
    p.add_argument(
        "--query-image-cache-dir",
        type=str,
        default=str(ROOT_DIR / "results/query_images/mrag_bench"),
        help="Local PNG cache for dataset query images passed to Gemma4.",
    )
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument(
        "--samples-per-scenario",
        type=int,
        default=0,
        help=(
            "Stratified sampling cap per MRAG-Bench scenario. "
            "Use >0 for parameter sweeps so each visual task type contributes samples; "
            "0 disables this cap."
        ),
    )
    p.add_argument(
        "--resume-from-existing",
        action="store_true",
        help="Resume from existing outputs: skip qs_id already present in answers-file and append new rows.",
    )
    return p


def run_benchmark(args: argparse.Namespace) -> None:
    core_envfile.load_dotenv(ROOT_DIR / ".env")
    core_mrag_bench.ensure_mrag_hf_cache_env()
    if args.magiclens_platform:
        jax.config.update("jax_platforms", args.magiclens_platform)
        log(f"magiclens_jax_platform={args.magiclens_platform}")

    answers_path = Path(args.answers_file)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    dims_path = Path(args.save_dimensions_jsonl) if args.save_dimensions_jsonl else None
    if dims_path:
        dims_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = Path(args.trace_jsonl) if args.trace_jsonl else None
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
    query_image_cache_dir = _resolve_repo_path(args.query_image_cache_dir, ROOT_DIR)

    api_key = (args.dim_generator_api_key or os.environ.get("DIM_GENERATOR_API_KEY", "")).strip()
    if args.dim_generator_type == "gemma4_local":
        dim_model_tag = args.gemma4_model_id
    elif args.dim_generator_type == "raw_question":
        dim_model_tag = "raw_question_no_rewrite"
    else:
        dim_model_tag = args.dim_generator_model

    log_torch_cuda_env()
    log(
        f"n_dims={args.n_dims} dim_top_k={args.dim_top_k} final_top_k={args.final_top_k} fusion={args.fusion_strategy}"
    )
    if args.samples_per_scenario > 0:
        log(f"samples_per_scenario={args.samples_per_scenario}")
    log(f"dim_generator_type={args.dim_generator_type} dim_model={dim_model_tag}")
    if args.trace_jsonl:
        log(f"trace_jsonl={args.trace_jsonl}")
    if args.describe_final_images:
        log(f"describe_final_images=1 max_new_tokens={args.gemma4_description_max_new_tokens}")

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

    llava_tokenizer = llava_model = llava_image_processor = None
    llava_answer = None
    if args.final_answerer == "llava":
        unified_cuda = args.gemma4_device.strip()
        if unified_cuda.startswith("cuda"):
            old_map = str(args.llava_device_map)
            args.llava_device_map = unified_cuda
            log(f"unified_cuda_device={unified_cuda} llava_device_map_override: {old_map} -> {args.llava_device_map}")
        load_llava, llava_answer = _load_llava_helpers()
        llava_tokenizer, llava_model, llava_image_processor = load_llava(args)
        log("LLaVA model ready")
    else:
        log(f"final_answerer={args.final_answerer}; skipping LLaVA model load")

    local_pipeline = None
    gemma_processor = None
    gemma_model = None
    if args.dim_generator_type == "local":
        local_pipeline = qp.load_local_pipeline(
            args.dim_generator_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        )
        log(f"local_dim_generator_ready={args.dim_generator_model}")

    needs_gemma4 = (
        args.dim_generator_type == "gemma4_local"
        or args.describe_final_images
        or args.final_answerer == "gemma4"
    )
    if needs_gemma4:
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
        log(f"gemma4_model_ready model_id={args.gemma4_model_id} local_dir={gdir} device={dev}")

    data_args = argparse.Namespace(
        dataset_name=args.dataset_name,
        test_size=10**9,
        use_rag=False,
        use_retrieved_examples=False,
        extra_prompt="",
    )
    log("mrag_bench_loader=official bench_data_loader use_rag=False")
    data_iter, total = core_mrag_bench.get_data_iter_and_total(data_args, bench_data_loader, "<image>")
    if total is None:
        log("dataset_total=provided_by_official_dataloader_progress")
    else:
        log(f"dataset_total={total}")

    processed = 0
    correct = 0
    by_scenario: dict[str, dict] = {}
    dim_gen_times: list[float] = []
    retrieval_times: list[float] = []
    final_answer_times: list[float] = []
    dim_gen_failures = 0

    seen_qs_ids: set[str] = set()
    if args.resume_from_existing and answers_path.is_file():
        try:
            with open(answers_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    row = json.loads(s)
                    qs_id = str(row.get("qs_id", "")).strip()
                    if qs_id:
                        seen_qs_ids.add(qs_id)
                        processed += 1
                        is_correct = bool(row.get("meta_is_correct", False))
                        if is_correct:
                            correct += 1
                        scenario = str(row.get("scenario", "Unknown"))
                        stat = by_scenario.setdefault(scenario, {"total": 0, "correct": 0})
                        stat["total"] += 1
                        if is_correct:
                            stat["correct"] += 1
            log(f"resume_from_existing=1 loaded_existing_rows={len(seen_qs_ids)} from {answers_path}")
        except Exception as e:
            log(f"warning: failed to load existing answers for resume: {e}")
            seen_qs_ids = set()

    out_mode = "a" if args.resume_from_existing else "w"
    dims_out = open(dims_path, out_mode, encoding="utf-8", buffering=1) if dims_path else None
    trace_out = open(trace_path, out_mode, encoding="utf-8", buffering=1) if trace_path else None

    try:
        with open(answers_path, out_mode, encoding="utf-8", buffering=1) as out:
            for idx, item in enumerate(data_iter):
                if idx < args.start_index:
                    continue
                if args.max_samples > 0 and processed >= args.max_samples:
                    break
                qs_id = str(item.get("id"))
                if qs_id in seen_qs_ids:
                    continue
                scenario = str(item.get("scenario", "Unknown"))
                if args.samples_per_scenario > 0:
                    scenario_seen = by_scenario.get(scenario, {}).get("total", 0)
                    if scenario_seen >= args.samples_per_scenario:
                        continue

                sample_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
                sample_t0 = time.time()
                question_with_choices = item["prompt_question_part"]
                question_text, _ = parse_question_and_options(question_with_choices)
                query_image = _query_image_from_bench_item(item)
                query_image_path = _ensure_query_image_path(item, query_image_cache_dir)

                t0 = time.time()
                dim_gen_error = None
                dim_rationales: list[str] = []
                dim_generation_raw_text = ""
                try:
                    if args.dim_generator_type == "raw_question":
                        dim_instructions = [question_with_choices]
                    elif args.dim_generator_type == "api":
                        dim_instructions = qp.generate_retrieval_instructions(
                            question_with_choices,
                            args.n_dims,
                            backend="api",
                            api_base=args.dim_generator_api_base,
                            api_key=api_key,
                            api_model=args.dim_generator_model,
                            temperature=args.dim_generator_temperature,
                        )
                    elif args.dim_generator_type == "gemma4_local":
                        if args.gemma4_dim_rationale:
                            plan = core_gemma4_dims.generate_retrieval_plan_with_rationales_gemma4(
                                gemma_processor,
                                gemma_model,
                                query_image=query_image_path,
                                question=question_with_choices,
                                n_dims=args.n_dims,
                                max_new_tokens=max(640, args.gemma4_max_new_tokens),
                            )
                            dim_instructions = plan.get("queries", [])
                            dim_rationales = plan.get("rationales", [])
                            dim_generation_raw_text = str(plan.get("raw_text", ""))
                        else:
                            dim_instructions = core_gemma4_dims.generate_retrieval_instructions_gemma4(
                                gemma_processor,
                                gemma_model,
                                query_image=query_image_path,
                                question=question_with_choices,
                                n_dims=args.n_dims,
                                max_new_tokens=args.gemma4_max_new_tokens,
                            )
                    else:
                        dim_instructions = qp.generate_retrieval_instructions(
                            question_with_choices,
                            args.n_dims,
                            backend="local",
                            local_pipeline=local_pipeline,
                        )
                except Exception as e:
                    dim_gen_error = str(e)
                    log(f"dimension_gen_error qs_id={item.get('id')}: {e}")
                    dim_instructions = []
                t_dim = time.time() - t0
                dim_gen_times.append(t_dim)

                if not dim_instructions:
                    dim_gen_failures += 1
                    dim_instructions = (
                        [question_with_choices] if not args.fallback_instruction else [args.fallback_instruction]
                    )
                    if not dim_rationales:
                        dim_rationales = ["fallback: dimension generation failed, using fallback instruction"]

                retrieval_queries = list(dim_instructions)
                if args.dim_retrieval_use_rationale and dim_rationales:
                    retrieval_queries = []
                    for i, q in enumerate(dim_instructions):
                        r = dim_rationales[i] if i < len(dim_rationales) else ""
                        r = " ".join(str(r).split())
                        q = " ".join(str(q).split())
                        retrieval_queries.append(f"{r}; {q}" if r else q)

                t0 = time.time()
                per_dim, fused = mdp.multi_dim_magiclens_retrieve_and_fuse(
                    query_image,
                    retrieval_queries,
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

                descriptions: list[dict] = []
                t_desc = 0.0
                if args.describe_final_images:
                    t0 = time.time()
                    desc_inputs = [{"image_label": "query_image", "path": str(query_image_path)}]
                    for rank, f in enumerate(fused, start=1):
                        desc_inputs.append({"image_label": f"retrieved_rank_{rank}", "path": str(f["path"])})
                    for desc_input in desc_inputs:
                        desc_path = desc_input["path"]
                        try:
                            text = core_gemma4_dims.describe_image_for_question_gemma4(
                                gemma_processor,
                                gemma_model,
                                image_path=desc_path,
                                question=question_with_choices,
                                image_label=desc_input["image_label"],
                                max_new_tokens=args.gemma4_description_max_new_tokens,
                            )
                            err = None
                        except Exception as e:
                            text = ""
                            err = str(e)
                            log(f"image_description_error qs_id={item.get('id')} image={desc_input['image_label']}: {e}")
                        pinfo = _path_payload(desc_path)
                        descriptions.append(
                            {
                                "image_label": desc_input["image_label"],
                                "path": pinfo["path"],
                                "filename": pinfo["filename"],
                                "id": pinfo["id"],
                                "description": text,
                                "error": err,
                            }
                        )
                    t_desc = time.time() - t0

                final_prompt_question_part = question_with_choices
                if descriptions:
                    final_prompt_question_part = _augment_question_with_image_descriptions(question_with_choices, descriptions)

                t_final_answer = 0.0
                final_answer_error = None
                final_answer_image_files = [query_image]
                if args.final_answerer == "llava":
                    image_files = [query_image]
                    for fi in fused:
                        with Image.open(fi["path"]) as img:
                            image_files.append(img.convert("RGB"))
                    max_images = max(1, int(args.llava_max_images))
                    if len(image_files) > max_images:
                        image_files = image_files[:max_images]
                    llava_item = dict(item)
                    llava_item["prompt_question_part"] = final_prompt_question_part
                    llava_item["question"] = final_prompt_question_part
                    t0 = time.time()
                    try:
                        raw_output = llava_answer(
                            llava_tokenizer,
                            llava_model,
                            llava_image_processor,
                            llava_item,
                            image_files,
                            args,
                        )
                        pred_choice = extract_choice(raw_output)
                    except Exception as e:
                        raw_output = ""
                        pred_choice = "N/A"
                        final_answer_error = str(e)
                        log(f"llava_answer_error qs_id={item.get('id')}: {e}")
                    t_final_answer = time.time() - t0
                    final_answer_image_files = image_files
                elif args.final_answerer == "gemma4":
                    image_paths = [str(query_image_path)] + [str(fi["path"]) for fi in fused]
                    max_images = max(1, int(args.gemma4_answer_max_images))
                    if len(image_paths) > max_images:
                        image_paths = image_paths[:max_images]
                    t0 = time.time()
                    try:
                        raw_output = core_gemma4_dims.answer_question_with_evidence_gemma4(
                            gemma_processor,
                            gemma_model,
                            image_paths=image_paths,
                            question=final_prompt_question_part,
                            max_new_tokens=args.gemma4_answer_max_new_tokens,
                        )
                        pred_choice = extract_choice(raw_output)
                    except Exception as e:
                        raw_output = ""
                        pred_choice = "N/A"
                        final_answer_error = str(e)
                        log(f"gemma4_answer_error qs_id={item.get('id')}: {e}")
                    t_final_answer = time.time() - t0
                    final_answer_image_files = image_paths
                else:
                    image_files = [query_image]
                    raw_output = ""
                    pred_choice = "N/A"
                    final_answer_image_files = image_files

                qs_id = str(item["id"])
                gt_choice = str(item["gt_choice"])
                scenario = str(item.get("scenario", scenario))
                is_correct = args.final_answerer in ("llava", "gemma4") and pred_choice == gt_choice
                sample_total = time.time() - sample_t0
                final_answer_times.append(t_final_answer)
                row = {
                    "qs_id": qs_id,
                    "prompt": item["prompt"],
                    "output": raw_output,
                    "gt_answer": item["answer"],
                    "shortuuid": uuid.uuid4().hex,
                    "model_id": f"{args.final_answerer}_multidim_{dim_model_tag.split('/')[-1]}_{args.fusion_strategy}",
                    "gt_choice": gt_choice,
                    "scenario": scenario,
                    "aspect": item["aspect"],
                    "meta_pred_choice": pred_choice,
                    "meta_is_correct": is_correct,
                    "meta_rag_count": len(fused),
                    "meta_rag_source": "multi_dimension_magiclens_pipeline",
                    "meta_n_dims": args.n_dims,
                    "meta_dim_instructions": dim_instructions,
                    "meta_dim_rationales": dim_rationales,
                    "meta_dim_retrieval_queries": retrieval_queries,
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
                    "meta_timings_sec": {
                        "dim_generation": round(t_dim, 3),
                        "magiclens_retrieval_and_fusion": round(t_ret, 3),
                        "gemma4_image_descriptions": round(t_desc, 3),
                        "final_answer": round(t_final_answer, 3),
                        "sample_total": round(sample_total, 3),
                    },
                    "meta_final_answer_error": final_answer_error,
                    "meta_final_answer_image_count": len(final_answer_image_files),
                    "meta_llava_error": final_answer_error if args.final_answerer == "llava" else None,
                    "meta_llava_image_count": len(final_answer_image_files) if args.final_answerer == "llava" else 0,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

                if trace_out:
                    trace = {
                        "trace_schema": "gemma4_multi_dim_rag_trace_v1",
                        "sample": {
                            "index": idx,
                            "qs_id": qs_id,
                            "scenario": scenario,
                            "aspect": item["aspect"],
                            "gt_choice": gt_choice,
                            "gt_answer": item["answer"],
                            "started_at": sample_started_at,
                        },
                        "runtime": {
                            "python": sys.version.split()[0],
                            "platform": platform.platform(),
                            "torch_cuda_available": torch.cuda.is_available(),
                            "torch_cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                            "jax_backend": jax.default_backend(),
                        },
                        "models": {
                            "dim_generator_type": args.dim_generator_type,
                            "dim_generator_model": dim_model_tag,
                            "gemma4_local_dir": str(_resolve_repo_path(args.gemma4_local_dir, ROOT_DIR)),
                            "gemma4_device": args.gemma4_device,
                            "magiclens_model_size": args.magiclens_model_size,
                            "magiclens_model_path": str(_resolve_repo_path(args.magiclens_model_path, ROOT_DIR)),
                            "magiclens_platform": args.magiclens_platform,
                            "final_answerer": args.final_answerer,
                            "llava_model_path": str(_resolve_repo_path(args.llava_model_path, ROOT_DIR)),
                            "gemma4_answer_model": args.gemma4_model_id if args.final_answerer == "gemma4" else None,
                        },
                        "input": {
                            "query_image": _path_payload(str(query_image_path)),
                            "question": question_text,
                            "question_with_choices": question_with_choices,
                            "official_prompt": item["prompt"],
                        },
                        "dimension_generation": {
                            "n_dims_requested": args.n_dims,
                            "prompt_input": {
                                "query_image_path": str(query_image_path),
                                "question_with_choices": question_with_choices,
                                "max_new_tokens": args.gemma4_max_new_tokens,
                                "with_rationale": bool(args.gemma4_dim_rationale),
                            },
                            "queries": [
                                {
                                    "dim_index": i,
                                    "query": instr,
                                    "rationale": dim_rationales[i - 1] if i - 1 < len(dim_rationales) else "",
                                    "retrieval_query": retrieval_queries[i - 1] if i - 1 < len(retrieval_queries) else instr,
                                }
                                for i, instr in enumerate(dim_instructions, start=1)
                            ],
                            "raw_generation_text": dim_generation_raw_text,
                            "error": dim_gen_error,
                            "time_sec": round(t_dim, 3),
                        },
                        "magiclens_retrieval": {
                            "dim_top_k": args.dim_top_k,
                            "calls": [
                                {
                                    "dim_index": i,
                                    "query": dim_instructions[i - 1] if i - 1 < len(dim_instructions) else "",
                                    "top_k": [_rank_row_payload(r) for r in rows],
                                }
                                for i, rows in enumerate(per_dim, start=1)
                            ],
                            "time_sec": round(t_ret, 3),
                        },
                        "fusion": {
                            "strategy": args.fusion_strategy,
                            "input_candidate_count": sum(len(rows) for rows in per_dim),
                            "unique_candidate_count": len({str(r["path"]) for rows in per_dim for r in rows}),
                            "final_top_k": args.final_top_k,
                            "selected": _fusion_trace(per_dim, fused, args.fusion_strategy),
                        },
                        "gemma4_image_descriptions": {
                            "enabled": bool(args.describe_final_images),
                            "max_new_tokens": args.gemma4_description_max_new_tokens,
                            "items": descriptions,
                            "time_sec": round(t_desc, 3),
                        },
                        "final_answer": {
                            "answerer": args.final_answerer,
                            "enabled": args.final_answerer in ("llava", "gemma4"),
                            "image_sequence": [
                                {"slot": 0, "image_label": "query_image", **_path_payload(str(query_image_path))}
                            ]
                            + [
                                {"slot": rank, "image_label": f"retrieved_rank_{rank}", **_path_payload(str(f["path"]))}
                                for rank, f in enumerate(fused, start=1)
                            ],
                            "prompt_question_part": final_prompt_question_part,
                            "raw_output": raw_output,
                            "pred_choice": pred_choice,
                            "gt_choice": gt_choice,
                            "is_correct": is_correct,
                            "error": final_answer_error,
                            "time_sec": round(t_final_answer, 3),
                        },
                        "llava_answer": {
                            "enabled": args.final_answerer == "llava",
                            "image_sequence": [
                                {"slot": 0, "image_label": "query_image", **_path_payload(str(query_image_path))}
                            ]
                            + [
                                {"slot": rank, "image_label": f"retrieved_rank_{rank}", **_path_payload(str(f["path"]))}
                                for rank, f in enumerate(fused, start=1)
                            ],
                            "prompt_question_part": final_prompt_question_part,
                            "raw_output": raw_output,
                            "pred_choice": pred_choice,
                            "gt_choice": gt_choice,
                            "is_correct": is_correct,
                            "time_sec": round(t_final_answer, 3) if args.final_answerer == "llava" else 0.0,
                        },
                        "timings_sec": row["meta_timings_sec"],
                    }
                    trace_out.write(json.dumps(trace, ensure_ascii=False) + "\n")

                if dims_out:
                    dims_out.write(
                        json.dumps(
                            {
                                "qs_id": qs_id,
                                "question": question_text,
                                "question_with_choices": question_with_choices,
                                "instructions": dim_instructions,
                                "rationales": dim_rationales,
                                "retrieval_queries": retrieval_queries,
                                "raw_generation_text": dim_generation_raw_text,
                                "dim_gen_time": round(t_dim, 3),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                processed += 1
                if is_correct:
                    correct += 1
                stat = by_scenario.setdefault(scenario, {"total": 0, "correct": 0})
                stat["total"] += 1
                if is_correct:
                    stat["correct"] += 1
    finally:
        if trace_out:
            trace_out.close()

    if dims_out:
        dims_out.close()

    by_scenario_acc = (
        {sc: round(100.0 * st["correct"] / max(1, st["total"]), 2) for sc, st in by_scenario.items()}
        if args.final_answerer in ("llava", "gemma4")
        else {}
    )
    summary = {
        "dataset_name": args.dataset_name,
        "processed": processed,
        "correct": correct,
        "accuracy": round(100.0 * correct / max(1, processed), 2) if args.final_answerer in ("llava", "gemma4") else None,
        "n_dims": args.n_dims,
        "dim_top_k": args.dim_top_k,
        "final_top_k": args.final_top_k,
        "fusion_strategy": args.fusion_strategy,
        "dim_generator_type": args.dim_generator_type,
        "dim_generator_model": dim_model_tag,
        "final_answerer": args.final_answerer,
        "magiclens_jax_platform": args.magiclens_platform,
        "dim_gen_failures": dim_gen_failures,
        "avg_dim_gen_time_sec": round(sum(dim_gen_times) / max(1, len(dim_gen_times)), 3),
        "avg_retrieval_time_sec": round(sum(retrieval_times) / max(1, len(retrieval_times)), 3),
        "avg_final_answer_time_sec": round(sum(final_answer_times) / max(1, len(final_answer_times)), 3),
        "by_scenario_accuracy": by_scenario_acc,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"summary_saved={summary_path}")
    if args.final_answerer in ("llava", "gemma4"):
        log(f"accuracy={summary['accuracy']}% processed={processed} dim_gen_failures={dim_gen_failures}")
    else:
        log(f"retrieval_only processed={processed} dim_gen_failures={dim_gen_failures}")


def main() -> None:
    core_envfile.load_dotenv(ROOT_DIR / ".env")
    parser = build_arg_parser()
    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
