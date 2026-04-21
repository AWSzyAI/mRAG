#!/usr/bin/env python3
"""
Print a concise picture of where MRAG-Bench / HF cache / image corpus live on disk.

Run on the server (repo root) and paste stdout into chat or CI logs::

    python scripts/inspect_data_layout.py
    python scripts/inspect_data_layout.py --root /public/home/hzh/mRAG
    python scripts/inspect_data_layout.py --json-out log/data_layout.json

Pick one corpus image path for tools like ``test/gemma4.py --image``::

    python scripts/inspect_data_layout.py --print-one-corpus-image

Uses only the standard library (no torch/datasets required).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def repo_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root.resolve()
    return Path(__file__).resolve().parents[1]


def rel(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root))
    except ValueError:
        return str(p)


def walk_limited(root: Path, max_depth: int, max_entries: int) -> list[str]:
    """Breadth-first style listing up to max_depth, cap lines."""
    lines: list[str] = []
    if not root.exists():
        lines.append(f"  (missing) {root}")
        return lines
    count = 0

    def rec(prefix: Path, depth: int) -> None:
        nonlocal count
        if count >= max_entries or depth > max_depth:
            return
        try:
            children = sorted(prefix.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except OSError as e:
            lines.append(f"  (list error) {prefix}: {e}")
            return
        for ch in children:
            if count >= max_entries:
                return
            sym = "/" if ch.is_dir() else ""
            lines.append(f"{'  ' * depth}{ch.name}{sym}")
            count += 1
            if ch.is_dir() and depth < max_depth:
                rec(ch, depth + 1)

    lines.append(str(root))
    rec(root, 1)
    if count >= max_entries:
        lines.append(f"  ... truncated after {max_entries} entries")
    return lines


def count_images_capped(root: Path, cap: int) -> tuple[int, bool]:
    if not root.is_dir():
        return 0, False
    n = 0
    for i, p in enumerate(root.rglob("*")):
        if i >= cap:
            return n, True
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            n += 1
    return n, False


def first_image_under(root: Path) -> Path | None:
    """Depth-first, sorted names per directory; first image file."""
    if not root.is_dir():
        return None
    for dirpath, _, names in os.walk(root, topdown=True):
        for n in sorted(names):
            suf = Path(n).suffix.lower()
            if suf in IMAGE_EXT:
                return Path(dirpath) / n
    return None


def should_count_images(path: Path, mrag_hf: Path) -> bool:
    """Avoid rglob over entire HF hub/datasets trees (can be millions of files)."""
    try:
        path = path.resolve()
        mrag_hf = mrag_hf.resolve()
    except OSError:
        return True
    if path == mrag_hf:
        return False
    if path.name in ("hub", "datasets"):
        return False
    try:
        path.relative_to(mrag_hf / "hub")
        return False
    except ValueError:
        pass
    try:
        path.relative_to(mrag_hf / "datasets")
        return False
    except ValueError:
        pass
    return True


def hf_dataset_dirs(datasets_cache: Path) -> list[str]:
    if not datasets_cache.is_dir():
        return []
    out = []
    for ch in sorted(datasets_cache.iterdir()):
        name = ch.name.lower()
        if "mrag" in name or "uclanlp" in name:
            out.append(ch.name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect MRAG data / corpus / HF cache layout")
    ap.add_argument("--root", type=Path, default=None, help="Repo root (default: parent of scripts/)")
    ap.add_argument("--max-depth", type=int, default=2, help="Per-directory tree depth for samples")
    ap.add_argument("--max-entries", type=int, default=120, help="Cap tree listing entries")
    ap.add_argument("--count-cap", type=int, default=200_000, help="Max rglob steps when counting corpus images")
    ap.add_argument("--json-out", type=Path, default=None, help="Write machine-readable summary JSON")
    ap.add_argument(
        "--print-one-corpus-image",
        action="store_true",
        help="Print absolute path of first image under corpus candidates (for --image=...)",
    )
    args = ap.parse_args()

    root = repo_root(args.root)
    env_keys = (
        "CORPUS_DIR",
        "MRAG_HF_HOME",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "HF_ENDPOINT",
        "MRAG_HF_OFFLINE",
    )
    env = {k: os.environ.get(k, "") for k in env_keys}

    mrag_hf = Path(env["MRAG_HF_HOME"] or (root / "models" / "huggingface-mrag")).expanduser()
    corpus_from_env = Path(env["CORPUS_DIR"]).expanduser() if env.get("CORPUS_DIR") else None

    candidates: list[Path] = []
    if corpus_from_env and corpus_from_env != Path("."):
        candidates.append(corpus_from_env)
    for rel_p in (
        root / "data" / "image_corpus",
        root / "data",
        mrag_hf,
        mrag_hf / "datasets",
        mrag_hf / "hub",
    ):
        if rel_p not in candidates:
            candidates.append(rel_p)

    if args.print_one_corpus_image:
        search_order: list[Path] = []
        if corpus_from_env and corpus_from_env.is_dir():
            search_order.append(corpus_from_env)
        p = root / "data" / "image_corpus"
        if p.is_dir() and p not in search_order:
            search_order.append(p)
        d = root / "data"
        if d.is_dir() and d not in search_order:
            search_order.append(d)
        for c in candidates:
            if c in search_order or not c.is_dir():
                continue
            if c.name == "image_corpus":
                search_order.append(c)
        for c in search_order:
            hit = first_image_under(c)
            if hit is not None:
                print(hit.resolve())
                return 0
        print("(no image found under CORPUS_DIR or data/image_corpus)", file=sys.stderr)
        return 1

    report: dict = {"repo_root": str(root), "env": env, "paths": []}

    print("=== mRAG data layout inspector ===")
    print(f"repo_root={root}")
    print("--- relevant env ---")
    for k, v in env.items():
        print(f"  {k}={v!r}")

    for base in candidates:
        exists = base.exists()
        rec: dict = {"path": str(base), "exists": exists, "is_dir": base.is_dir() if exists else False}
        if exists and base.is_dir():
            if should_count_images(base, mrag_hf):
                n_img, truncated = count_images_capped(base, args.count_cap)
                rec["image_files_recursive"] = n_img
                rec["image_count_truncated"] = truncated
            else:
                rec["image_files_recursive"] = None
                rec["image_count_note"] = "skipped_rglob_hf_cache_root"
            rec["tree_sample"] = walk_limited(base, args.max_depth, args.max_entries)
        report["paths"].append(rec)

        print(f"--- path: {base} ---")
        print(f"  exists={exists}")
        if exists and base.is_dir():
            if rec.get("image_files_recursive") is None:
                print(f"  image_files_recursive=skipped ({rec.get('image_count_note', '')})")
            else:
                print(
                    f"  image_files_recursive(~)={rec['image_files_recursive']} "
                    f"truncated={rec.get('image_count_truncated', False)}"
                )
            for line in rec.get("tree_sample", []):
                print(line)
        print()

    ds = mrag_hf / "datasets"
    print("--- HF datasets cache (MRAG-related dir names) ---")
    names = hf_dataset_dirs(ds)
    if not names:
        print(f"  (none or missing) {ds}")
    else:
        for n in names:
            print(f"  {ds / n}")
    report["mrag_hf_datasets_mrag_related"] = [str(ds / n) for n in names]

    print("--- notes ---")
    print("  MRAG-Bench 由 HuggingFace ``datasets`` 加载；缓存目录名通常含 ``uclanlp___mrag-bench``。")
    print("  全库检索语料为扁平/任意子目录下的图片：与 ``list_corpus_images`` / ``--corpus-dir`` 一致。")
    print("  详见仓库 ``doc/DATA_LAYOUT.md``。")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"wrote_json={args.json_out.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
