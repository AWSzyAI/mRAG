"""MRAG-Bench iteration via HuggingFace ``datasets`` (local Hub cache under ``MRAG_HF_HOME``)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

if os.environ.get("MRAG_ALLOW_HF_NETWORK", "").strip().lower() not in ("1", "true", "yes"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from datasets import load_dataset

# ``src/mrag/mrag_bench.py`` → repo root is two levels up
REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_mrag_hf_cache_env() -> None:
    """Point ``HF_HOME`` / ``HF_DATASETS_CACHE`` at the MRAG project cache so ``load_dataset`` stays local.

    Resolution order:

    1. ``MRAG_HF_HOME`` if set (absolute, or relative to repo root).
    2. Else if ``<repo>/github/MRAG-Bench/.cache/huggingface-mrag`` exists (common submodule layout).
    3. Else if ``<repo>/models/huggingface-mrag`` exists (``main.py`` default layout).

    Uses ``os.environ.setdefault`` so explicit ``HF_HOME`` / ``HF_DATASETS_CACHE`` win.
    """
    allow_network = os.environ.get("MRAG_ALLOW_HF_NETWORK", "").strip().lower() in ("1", "true", "yes")
    if not allow_network:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    raw = (os.environ.get("MRAG_HF_HOME") or "").strip()
    hf_home: Path | None = None
    if raw:
        p = Path(raw).expanduser()
        hf_home = p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()
    else:
        for cand in (
            REPO_ROOT / "github" / "MRAG-Bench" / ".cache" / "huggingface-mrag",
            REPO_ROOT / "models" / "huggingface-mrag",
        ):
            if cand.is_dir():
                hf_home = cand.resolve()
                break
    if hf_home is None:
        return

    hf_home.mkdir(parents=True, exist_ok=True)
    hub = hf_home / "hub"
    dss = hf_home / "datasets"
    hub.mkdir(parents=True, exist_ok=True)
    dss.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hub))
    os.environ.setdefault("HF_DATASETS_CACHE", str(dss))

    if os.environ.get("MRAG_HF_OFFLINE", "").strip().lower() in ("1", "true", "yes"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def build_data_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_name=args.dataset_name,
        test_size=10**9,
        use_rag=True,
        use_retrieved_examples=not bool(args.use_gt),
        extra_prompt="",
    )


def get_data_iter_and_total(data_args, bench_data_loader, image_placeholder):
    try:
        import inspect

        sig = inspect.signature(bench_data_loader)
        if "return_total" in sig.parameters:
            data_iter, total = bench_data_loader(
                data_args, image_placeholder=image_placeholder, return_total=True
            )
            return data_iter, total
    except (TypeError, ValueError):
        pass

    data_iter = bench_data_loader(data_args, image_placeholder=image_placeholder)
    return data_iter, None


def infer_dataset_total(dataset_name: str):
    ensure_mrag_hf_cache_env()
    try:
        ds = load_dataset(dataset_name, split="test", download_mode="reuse_dataset_if_exists")
        return len(ds)
    except Exception:
        return None


def iter_bench_queries(dataset_name: str):
    ensure_mrag_hf_cache_env()
    ds = load_dataset(dataset_name, split="test", download_mode="reuse_dataset_if_exists")
    for item in ds:
        qs = item["question"]
        image = item["image"].convert("RGB")
        image_path = getattr(item["image"], "filename", None)
        prompt_question_part = (
            f"{qs}\n Choices:\n"
            f"A: {item['A']}\n"
            f"B: {item['B']}\n"
            f"C: {item['C']}\n"
            f"D: {item['D']}"
        )
        yield {
            "id": item["id"],
            "question": prompt_question_part,
            "prompt_question_part": prompt_question_part,
            "prompt": prompt_question_part,
            "answer": item["answer"],
            "gt_choice": item["answer_choice"],
            "scenario": item["scenario"],
            "aspect": item["aspect"],
            "query_image": image,
            "query_image_path": image_path if image_path and Path(str(image_path)).is_file() else None,
        }
