#!/usr/bin/env python3
"""Compute Recall@5 / Recall@N for E0-E13 style MRAG experiments.

Definitions used in the paper:
- Recall@5: per-sample fraction of GT evidence images appearing in the final
  answer evidence list (normally Top-5), averaged over samples.
- Recall@N: per-sample fraction of GT evidence images appearing anywhere in the
  pre-fusion candidate pool. For single-list experiments this equals Top-5.
- GT hits@N: duplicate-aware count of GT hits in the pre-fusion pool. This is
  useful for multi-dim query rewriting, where the same GT image can be hit by
  several dimensions.

The script is designed to be run on the server where MRAG-Bench cache and
`data/image_corpus` are available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm_id(value: Any) -> str:
    return str(value).strip()


def image_dhash(img: Image.Image, hash_size: int = 16) -> str:
    img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())
    bits = []
    for y in range(hash_size):
        row = y * (hash_size + 1)
        for x in range(hash_size):
            bits.append(1 if pixels[row + x] > pixels[row + x + 1] else 0)
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return f"{value:0{hash_size * hash_size // 4}x}"


def load_image_from_dataset_value(value: Any) -> Image.Image:
    import io

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and "bytes" in value:
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    return value.convert("RGB")


def load_dataset_items(dataset_name: str, split: str, hash_size: int) -> dict[str, dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    out: dict[str, dict[str, Any]] = {}
    for item in ds:
        qs_id = norm_id(item["id"])
        scenario = str(item.get("scenario", "Unknown"))
        gt_values = list(item.get("gt_images") or [])
        if scenario == "Incomplete" and gt_values:
            gt_values = [gt_values[0]]
        gt_hashes = []
        for value in gt_values:
            gt_hashes.append(image_dhash(load_image_from_dataset_value(value), hash_size=hash_size))

        retrieved_hashes = []
        for value in list(item.get("retrieved_images") or []):
            retrieved_hashes.append(image_dhash(load_image_from_dataset_value(value), hash_size=hash_size))
        if scenario == "Incomplete" and retrieved_hashes:
            retrieved_hashes = [retrieved_hashes[0]]

        out[qs_id] = {
            "scenario": scenario,
            "gt_hashes": gt_hashes,
            "retrieved_hashes": retrieved_hashes,
        }
    return out


def build_corpus_hash_index(corpus_dir: Path, hash_size: int) -> dict[str, str]:
    """Map corpus basename to perceptual hash."""
    index: dict[str, str] = {}
    if not corpus_dir.is_dir():
        return index
    paths = [p for p in corpus_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    for i, path in enumerate(paths, 1):
        try:
            with Image.open(path) as img:
                index[path.name] = image_dhash(img, hash_size=hash_size)
        except Exception:
            continue
        if i % 2000 == 0:
            print(f"[recall] indexed corpus images: {i}/{len(paths)}")
    return index


def read_jsonl(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def result_paths_from_row(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (final_paths, candidate_pool_paths)."""
    final_paths: list[str] = []
    candidate_paths: list[str] = []

    if isinstance(row.get("meta_fused_retrieval"), list):
        final_paths = [str(x.get("path", "")) for x in row["meta_fused_retrieval"] if x.get("path")]
        for dim in row.get("meta_per_dim_retrieval") or []:
            for x in dim:
                if x.get("path"):
                    candidate_paths.append(str(x["path"]))
        return final_paths, candidate_paths or final_paths

    if isinstance(row.get("meta_corpus_retrieval"), list):
        base = [str(x.get("path", "")) for x in row["meta_corpus_retrieval"] if x.get("path")]
        ranks = row.get("meta_magiclens_rag_ranks") or []
        if ranks:
            by_rank = {int(x.get("rank", i + 1)): p for i, (x, p) in enumerate(zip(row["meta_corpus_retrieval"], base))}
            ordered = []
            for r in sorted(ranks, key=lambda z: int(z.get("new_rank", 10**9))):
                idx = int(r.get("orig_rag_index", 0))
                if idx in by_rank:
                    ordered.append(by_rank[idx])
            final_paths = ordered or base
        else:
            final_paths = base
        return final_paths[:5], base

    return [], []


def trace_paths_from_row(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    final_paths: list[str] = []
    candidate_paths: list[str] = []
    fusion = row.get("fusion") or {}
    for x in fusion.get("selected") or []:
        if x.get("path"):
            final_paths.append(str(x["path"]))
    retrieval = row.get("magiclens_retrieval") or {}
    for call in retrieval.get("calls") or []:
        for x in call.get("top_k") or []:
            if x.get("path"):
                candidate_paths.append(str(x["path"]))
    return final_paths, candidate_paths or final_paths


def hashes_for_paths(paths: list[str], corpus_hash: dict[str, str]) -> list[str]:
    out = []
    for p in paths:
        name = Path(p).name
        h = corpus_hash.get(name)
        if h:
            out.append(h)
    return out


def score_one(gt_hashes: list[str], final_hashes: list[str], candidate_hashes: list[str]) -> dict[str, float]:
    gt_set = set(gt_hashes)
    denom = max(1, len(gt_set))
    final_unique = set(final_hashes)
    cand_unique = set(candidate_hashes)
    hit5 = len(gt_set & final_unique)
    hitn_unique = len(gt_set & cand_unique)
    hitn_dup = sum(1 for h in candidate_hashes if h in gt_set)
    return {
        "gt_count": len(gt_set),
        "final_count": len(final_hashes),
        "candidate_count": len(candidate_hashes),
        "recall_at_5": hit5 / denom,
        "recall_at_n": hitn_unique / denom,
        "gt_hits_at_5": hit5,
        "gt_hits_at_n": hitn_dup,
    }


def summarize(exp: str, rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {"experiment": exp, "status": "missing", "processed": 0}
    keys = ["recall_at_5", "recall_at_n", "gt_hits_at_5", "gt_hits_at_n", "candidate_count"]
    out = {"experiment": exp, "status": "ok", "processed": len(rows)}
    for k in keys:
        out[k] = sum(float(r[k]) for r in rows) / len(rows)
    return out


def compute_experiment(exp: dict[str, Any], dataset: dict[str, dict[str, Any]], corpus_hash: dict[str, str]) -> dict[str, Any]:
    mode = exp["mode"]
    name = exp["name"]
    rows: list[dict[str, float]] = []

    if mode == "none":
        return {"experiment": name, "status": "not_applicable", "processed": 0}

    if mode == "gt_oracle":
        for item in dataset.values():
            gt = item["gt_hashes"]
            rows.append(score_one(gt, gt, gt))
        return summarize(name, rows)

    if mode == "official_retrieved":
        for item in dataset.values():
            gt = item["gt_hashes"]
            retrieved = item["retrieved_hashes"]
            rows.append(score_one(gt, retrieved[:5], retrieved))
        return summarize(name, rows)

    path = ROOT / exp["path"]
    if not path.exists():
        return {"experiment": name, "status": "missing", "processed": 0}

    for row in read_jsonl(path):
        qs_id = norm_id(row.get("qs_id") or (row.get("sample") or {}).get("qs_id"))
        if qs_id not in dataset:
            continue
        gt = dataset[qs_id]["gt_hashes"]
        if mode == "trace":
            final_paths, cand_paths = trace_paths_from_row(row)
        else:
            final_paths, cand_paths = result_paths_from_row(row)
        final_hashes = hashes_for_paths(final_paths, corpus_hash)
        cand_hashes = hashes_for_paths(cand_paths, corpus_hash)
        rows.append(score_one(gt, final_hashes, cand_hashes))
    return summarize(name, rows)


def default_experiments() -> list[dict[str, str]]:
    return [
        {"name": "E0", "mode": "gt_oracle"},
        {"name": "E1", "mode": "gt_oracle"},
        {"name": "E2", "mode": "official_retrieved"},
        {"name": "E3", "mode": "results", "path": "log/E3/e3_clip_corpus_rag_results.jsonl"},
        {"name": "E4", "mode": "none"},
        {"name": "E5", "mode": "gt_oracle"},
        {"name": "E6", "mode": "results", "path": "log/E6/e6_clip_corpus_magiclens_rerank_results.jsonl"},
        {"name": "E7", "mode": "results", "path": "log/E7/e7_magiclens_corpus_rag_results.jsonl"},
        {"name": "E8", "mode": "trace", "path": "log/E8_full/e8_full_trace.jsonl"},
        {"name": "E9", "mode": "trace", "path": "log/E9/e9_gemma4_answer_trace.jsonl"},
        {"name": "E10", "mode": "trace", "path": "log/E10/e10_no_rewrite_trace.jsonl"},
        {"name": "E11_1", "mode": "trace", "path": "log/E11/E11_1/e11_1_trace.jsonl"},
        {"name": "E11_2", "mode": "trace", "path": "log/E11/E11_2/e11_2_trace.jsonl"},
        {"name": "E11_3", "mode": "trace", "path": "log/E11/E11_3/e11_3_trace.jsonl"},
        {"name": "E11_4", "mode": "trace", "path": "log/E11/E11_4/e11_4_trace.jsonl"},
        {"name": "E11_5/E8", "mode": "trace", "path": "log/E8_full/e8_full_trace.jsonl"},
        {"name": "E12_1", "mode": "trace", "path": "log/E12/E12_1/e12_1_trace.jsonl"},
        {"name": "E12_2", "mode": "trace", "path": "log/E12/E12_2/e12_2_trace.jsonl"},
        {"name": "E12_3", "mode": "trace", "path": "log/E12/E12_3/e12_3_trace.jsonl"},
        {"name": "E12_4", "mode": "trace", "path": "log/E12/E12_4/e12_4_trace.jsonl"},
        {"name": "E12_5/E8", "mode": "trace", "path": "log/E8_full/e8_full_trace.jsonl"},
        {"name": "E13_1", "mode": "trace", "path": "log/E13/E13_1_score_sum/e13_1_score_sum_trace.jsonl"},
        {"name": "E13_2", "mode": "trace", "path": "log/E13/E13_2_voting/e13_2_voting_trace.jsonl"},
    ]


def plot_recall(rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r.get("status") == "ok" and str(r["experiment"]) in {"E2", "E3", "E6", "E7", "E8", "E13_1", "E13_2"}]
    if not ok:
        ok = [r for r in rows if r.get("status") == "ok"]
    labels = [r["experiment"] for r in ok]
    r5 = [100 * float(r["recall_at_5"]) for r in ok]
    rn = [100 * float(r["recall_at_n"]) for r in ok]

    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=180)
    x = range(len(labels))
    width = 0.38
    ax.bar([i - width / 2 for i in x], r5, width=width, label="Recall@5", color="#3b73b9")
    ax.bar([i + width / 2 for i in x], rn, width=width, label="Recall@N", color="#d97941")
    ax.set_xticks(list(x), labels, rotation=25, ha="right")
    ax.set_ylabel("GT evidence recall (%)")
    ax.set_title("GT Evidence Recall Across Retrieval Experiments")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", default="uclanlp/MRAG-Bench")
    ap.add_argument("--split", default="test")
    ap.add_argument("--corpus-dir", default="data/image_corpus")
    ap.add_argument("--hash-size", type=int, default=16)
    ap.add_argument("--csv-out", default="log/E13/e0_e13_recall_metrics.csv")
    ap.add_argument("--json-out", default="log/E13/e0_e13_recall_metrics.json")
    ap.add_argument("--plot-out", default="paper/images/e0_e13_recall.png")
    args = ap.parse_args()

    print("[recall] loading MRAG-Bench metadata")
    dataset = load_dataset_items(args.dataset_name, args.split, args.hash_size)
    print("[recall] indexing corpus images")
    corpus_hash = build_corpus_hash_index(ROOT / args.corpus_dir, args.hash_size)
    print(f"[recall] corpus indexed={len(corpus_hash)}")

    rows = [compute_experiment(exp, dataset, corpus_hash) for exp in default_experiments()]

    csv_path = ROOT / args.csv_out
    json_path = ROOT / args.json_out
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "status", "processed", "recall_at_5", "recall_at_n", "gt_hits_at_5", "gt_hits_at_n", "candidate_count"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_recall(rows, ROOT / args.plot_out)
    print(f"[recall] wrote {csv_path}")
    print(f"[recall] wrote {json_path}")
    print(f"[recall] wrote {ROOT / args.plot_out}")


if __name__ == "__main__":
    main()
