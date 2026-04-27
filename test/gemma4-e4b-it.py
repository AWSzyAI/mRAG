#!/usr/bin/env python3
"""
Gemma 4 E4B-it download + smoke test for E15.

This is the E4B counterpart of ``test/gemma4.py``. It downloads the model to
``models/gemma4-e4b-it`` by default, then optionally loads it on a GPU and runs
one text-only plus one image-text generation check.

Typical server commands:

    # Download only. Requires HF_TOKEN or an authenticated huggingface-cli login.
    python test/gemma4-e4b-it.py --mode download

    # Download if needed, then run a smoke test on cuda:0.
    python test/gemma4-e4b-it.py --mode both --device cuda:0

    # Use a different model/cache path if the server layout differs.
    python test/gemma4-e4b-it.py \
      --mode download \
      --model-id google/gemma-4-E4B-it \
      --local-dir ./models/gemma4-e4b-it

After this pre-test passes, run E15 with:

    bash test/E15_gemma4_e4b_answerer.sh
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_LOCAL_DIR = ROOT / "models" / "gemma4-e4b-it"
MIN_WEIGHT_FILE_BYTES = 1024 * 1024


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_repo_dotenv(path: Path) -> None:
    try:
        from src.mrag.envfile import load_dotenv
    except Exception as e:
        log(f"skip .env load: {e}")
        return
    load_dotenv(path)


def resolve_local_dir(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def find_weight_files(local_dir: Path) -> list[Path]:
    hits: list[Path] = []
    for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
        hits.extend(p for p in local_dir.glob(pattern) if p.is_file())
    return sorted(hits)


def has_real_weight_files(local_dir: Path) -> bool:
    return any(p.stat().st_size >= MIN_WEIGHT_FILE_BYTES for p in find_weight_files(local_dir))


def require_download_complete(local_dir: Path) -> None:
    if has_real_weight_files(local_dir):
        return
    found = find_weight_files(local_dir)
    if found:
        details = ", ".join(f"{p.name}={p.stat().st_size}B" for p in found)
        raise SystemExit(
            f"Local directory has only tiny/incomplete weight pointer files: {local_dir} ({details}).\n"
            "Re-run download after ensuring Hugging Face/Xet can fetch large files."
        )
    raise SystemExit(
        f"Local directory has config files but no model weights: {local_dir}\n"
        "Run: python test/gemma4-e4b-it.py --mode download\n"
        "Expected at least one large file such as model.safetensors."
    )


def cmd_download(model_id: str, local_dir: Path, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    log(f"snapshot_download repo={model_id} -> {local_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token or None,
        resume_download=True,
    )
    require_download_complete(local_dir)
    log("download_done")


def cmd_run_lazy(args: argparse.Namespace) -> None:
    # Import only for run mode so download-only nodes do not need torch/transformers loaded.
    from gemma4 import cmd_run

    cmd_run(args)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gemma 4 E4B-it: server download + GPU smoke test for E15"
    )
    p.add_argument(
        "--mode",
        choices=("download", "run", "both"),
        default="both",
        help="download=only fetch weights; run=only smoke test; both=download if needed then run",
    )
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument(
        "--local-dir",
        type=str,
        default=str(DEFAULT_LOCAL_DIR),
        help="Local snapshot directory. Default: repo models/gemma4-e4b-it",
    )
    p.add_argument("--device", type=str, default=os.environ.get("GEMMA4_DEVICE", "cuda:0"))
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--hf-token", type=str, default="", help="Defaults to HF_TOKEN / HUGGING_FACE_HUB_TOKEN")
    p.add_argument("--image", type=str, default="", help="Optional local image for multimodal smoke test")
    p.add_argument(
        "--skip-download-check",
        action="store_true",
        help="Allow run mode to load from Hub if local config.json is missing",
    )
    p.add_argument("--skip-vision", action="store_true", help="Only run text generation")
    p.add_argument(
        "--no-synthetic-image",
        action="store_true",
        help="Do not create .cache/gemma4_smoke_96.png when no image is found",
    )
    p.add_argument(
        "--allow-torch-below-2-4",
        action="store_true",
        help="Skip the PyTorch>=2.4 guard inherited from the Gemma4 loader",
    )
    return p


def main() -> None:
    load_repo_dotenv(ROOT / ".env")
    args = build_arg_parser().parse_args()
    local_dir = resolve_local_dir(args.local_dir)
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if args.mode in ("download", "both"):
        if args.mode == "both" and (local_dir / "config.json").exists() and has_real_weight_files(local_dir):
            log(f"local model already exists; skip download: {local_dir}")
        else:
            cmd_download(args.model_id, local_dir, token)

    if args.mode in ("run", "both"):
        if not args.skip_download_check:
            require_download_complete(local_dir)
        args.local_dir = str(local_dir)
        cmd_run_lazy(args)


if __name__ == "__main__":
    main()
