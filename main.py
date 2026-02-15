#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download


def strtobool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def resolve_path(value: str, root: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _snapshot_download(
    repo_id: str,
    local_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> str:
    kwargs = {
        "repo_id": repo_id,
    }
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return snapshot_download(**kwargs)


def _cleanup_dataset_cache(dataset_id: str, hf_datasets_cache: Path) -> int:
    if not hf_datasets_cache.exists():
        return 0

    key = dataset_id.lower().replace("/", "___")
    removed = 0
    for child in hf_datasets_cache.iterdir():
        name = child.name.lower()
        if key in name or name.endswith(".incomplete"):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
            removed += 1
    return removed


def _cleanup_dataset_download_cache(hf_datasets_cache: Path) -> int:
    removed = 0
    for name in ("downloads", "downloads-extracted"):
        target = hf_datasets_cache / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            removed += 1
    return removed


def _cleanup_dataset_hub_cache(dataset_id: str, hf_hub_cache: Path) -> int:
    if not hf_hub_cache.exists():
        return 0

    repo_key = f"datasets--{dataset_id.replace('/', '--')}".lower()
    removed = 0
    for child in hf_hub_cache.iterdir():
        if repo_key in child.name.lower():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
            removed += 1
    return removed


def main() -> None:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Prefetch MRAG model/dataset assets to local cache for offline runtime."
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("MRAG_MODEL_ID", "lmms-lab/llava-onevision-qwen2-7b-ov"),
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--model-local-dir",
        default=os.getenv(
            "MRAG_MODEL_LOCAL_DIR", str(root / "models/llava-onevision-qwen2-7b-ov")
        ),
        help="Local directory to save model files",
    )
    parser.add_argument(
        "--hf-home",
        default=os.getenv("MRAG_HF_HOME", str(root / "models/huggingface-mrag")),
        help="HF cache root (HF_HOME)",
    )
    parser.add_argument(
        "--unset-proxy",
        action="store_true",
        default=strtobool(os.getenv("MRAG_UNSET_PROXY", "0")),
        help="Unset proxy env vars before download",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=os.getenv("HF_ENDPOINT", "https://hf-mirror.com"),
        help="HF endpoint (mirror)",
    )
    parser.add_argument(
        "--hf-hub-etag-timeout",
        type=int,
        default=int(os.getenv("HF_HUB_ETAG_TIMEOUT", "30")),
        help="HF Hub etag timeout (seconds)",
    )
    parser.add_argument(
        "--hf-hub-download-timeout",
        type=int,
        default=int(os.getenv("HF_HUB_DOWNLOAD_TIMEOUT", "600")),
        help="HF Hub download timeout (seconds)",
    )
    parser.add_argument(
        "--hf-hub-enable-hf-transfer",
        type=int,
        choices=[0, 1],
        default=int(os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "0")),
        help="Set HF_HUB_ENABLE_HF_TRANSFER (0/1)",
    )
    parser.add_argument(
        "--hf-hub-disable-xet",
        type=int,
        choices=[0, 1],
        default=int(os.getenv("HF_HUB_DISABLE_XET", "1")),
        help="Set HF_HUB_DISABLE_XET (0/1)",
    )
    parser.add_argument(
        "--hf-max-retries",
        type=int,
        default=int(os.getenv("HF_MAX_RETRIES", "8")),
        help="Retry count for datasets downloads",
    )
    parser.add_argument(
        "--dataset-id",
        default=os.getenv("MRAG_DATASET_ID", "uclanlp/MRAG-Bench"),
        help="Hugging Face dataset id",
    )
    parser.add_argument(
        "--dataset-split",
        default=os.getenv("MRAG_DATASET_SPLIT", "test"),
        help="Dataset split to materialize locally",
    )
    parser.add_argument(
        "--vision-tower-id",
        default=os.getenv("MRAG_VISION_TOWER_ID", "google/siglip-so400m-patch14-384"),
        help="Extra repo to prefetch into HF cache for LLaVA vision tower",
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        default=strtobool(os.getenv("MRAG_SKIP_MODEL_DOWNLOAD", "0")),
        help="Skip downloading the LLaVA model repo",
    )
    parser.add_argument(
        "--skip-dataset-download",
        action="store_true",
        default=strtobool(os.getenv("MRAG_SKIP_DATASET_DOWNLOAD", "0")),
        help="Skip downloading dataset split",
    )
    parser.add_argument(
        "--skip-vision-tower-download",
        action="store_true",
        default=strtobool(os.getenv("MRAG_SKIP_VISION_TOWER_DOWNLOAD", "0")),
        help="Skip prefetching vision tower repo",
    )
    args = parser.parse_args()

    if args.unset_proxy:
        for key in (
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
        ):
            os.environ.pop(key, None)

    model_local_dir = resolve_path(args.model_local_dir, root)
    hf_home = resolve_path(args.hf_home, root)
    hf_hub_cache = hf_home / "hub"
    hf_datasets_cache = hf_home / "datasets"
    model_local_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_cache.mkdir(parents=True, exist_ok=True)
    hf_datasets_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(hf_datasets_cache)
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(args.hf_hub_etag_timeout)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.hf_hub_download_timeout)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = str(args.hf_hub_enable_hf_transfer)
    os.environ["HF_HUB_DISABLE_XET"] = str(args.hf_hub_disable_xet)

    print(f"[ENV] MODEL_ID={args.model_id}", flush=True)
    print(f"[ENV] MODEL_LOCAL_DIR={model_local_dir}", flush=True)
    print(f"[ENV] HF_ENDPOINT={os.environ['HF_ENDPOINT']}", flush=True)
    print(f"[ENV] HF_HOME={os.environ['HF_HOME']}", flush=True)
    print(f"[ENV] HF_HUB_CACHE={os.environ['HF_HUB_CACHE']}", flush=True)
    print(f"[ENV] HF_DATASETS_CACHE={os.environ['HF_DATASETS_CACHE']}", flush=True)
    print(
        "[ENV] HF_HUB_ETAG_TIMEOUT="
        f"{os.environ['HF_HUB_ETAG_TIMEOUT']} "
        "HF_HUB_DOWNLOAD_TIMEOUT="
        f"{os.environ['HF_HUB_DOWNLOAD_TIMEOUT']} "
        "HF_HUB_ENABLE_HF_TRANSFER="
        f"{os.environ['HF_HUB_ENABLE_HF_TRANSFER']} "
        "HF_HUB_DISABLE_XET="
        f"{os.environ['HF_HUB_DISABLE_XET']}",
        flush=True,
    )
    print(
        "[ENV] http_proxy="
        f"{os.environ.get('http_proxy', '<unset>')} "
        "https_proxy="
        f"{os.environ.get('https_proxy', '<unset>')} "
        "all_proxy="
        f"{os.environ.get('all_proxy', '<unset>')}",
        flush=True,
    )

    if args.skip_model_download:
        print("[SKIP] model download", flush=True)
    else:
        path = _snapshot_download(
            args.model_id,
            local_dir=model_local_dir,
            cache_dir=hf_hub_cache,
        )
        print(f"[OK] model_cached_at={path}", flush=True)

    if args.skip_vision_tower_download:
        print("[SKIP] vision tower prefetch", flush=True)
    else:
        vt_path = _snapshot_download(args.vision_tower_id, cache_dir=hf_hub_cache)
        print(f"[OK] vision_tower_cached_at={vt_path}", flush=True)

    if args.skip_dataset_download:
        print("[SKIP] dataset download", flush=True)
    else:
        from datasets import DownloadConfig, load_dataset
        from datasets.exceptions import NonMatchingSplitsSizesError

        print(
            f"[INFO] materializing dataset split: {args.dataset_id} ({args.dataset_split})",
            flush=True,
        )
        download_config = DownloadConfig(
            max_retries=max(1, int(args.hf_max_retries)),
            resume_download=True,
        )

        download_mode = "reuse_dataset_if_exists"
        try:
            ds = load_dataset(
                args.dataset_id,
                split=args.dataset_split,
                cache_dir=str(hf_datasets_cache),
                download_config=download_config,
                download_mode=download_mode,
            )
        except NonMatchingSplitsSizesError:
            removed_ds = _cleanup_dataset_cache(args.dataset_id, hf_datasets_cache)
            removed_dl = _cleanup_dataset_download_cache(hf_datasets_cache)
            removed_hub = _cleanup_dataset_hub_cache(args.dataset_id, hf_hub_cache)
            print(
                "[WARN] Dataset cache metadata/content mismatch detected. "
                f"Removed datasets={removed_ds}, downloads={removed_dl}, hub={removed_hub}. "
                "Retrying with force_redownload...",
                flush=True,
            )
            try:
                ds = load_dataset(
                    args.dataset_id,
                    split=args.dataset_split,
                    cache_dir=str(hf_datasets_cache),
                    download_config=download_config,
                    download_mode="force_redownload",
                )
            except NonMatchingSplitsSizesError as e:
                raise RuntimeError(
                    "Dataset prefetch still failed after full cache cleanup. "
                    "Please manually remove cache dirs and retry:\n"
                    f"  rm -rf {hf_datasets_cache}/uclanlp___mrag-bench*\n"
                    f"  rm -rf {hf_datasets_cache}/downloads {hf_datasets_cache}/downloads-extracted\n"
                    f"  rm -rf {hf_hub_cache}/datasets--uclanlp--MRAG-Bench\n"
                    "Then rerun: bash test/data_models.sh"
                ) from e
        print(f"[OK] dataset_rows={len(ds)}", flush=True)

    print("[DONE] asset prefetch completed", flush=True)


if __name__ == "__main__":
    main()
