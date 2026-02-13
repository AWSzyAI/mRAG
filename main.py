#!/usr/bin/env python3
import argparse
import inspect
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def strtobool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def resolve_path(value: str, root: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def main() -> None:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Download MRAG LLaVA-OneVision model to local directory."
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

    sig = inspect.signature(snapshot_download)
    kwargs = {
        "repo_id": args.model_id,
        "local_dir": str(model_local_dir),
        "resume_download": True,
    }
    if "local_dir_use_symlinks" in sig.parameters:
        kwargs["local_dir_use_symlinks"] = False

    path = snapshot_download(**kwargs)
    print(f"[OK] model_cached_at={path}", flush=True)


if __name__ == "__main__":
    main()
