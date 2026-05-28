#!/usr/bin/env python3
"""Run open-ended InfoSeek questions through the current multi-dim rewrite stage.

This script is intentionally lightweight:
- sample N InfoSeek Entity questions
- generate multi-dimensional retrieval instructions for each question
- optionally use the existing query planner when credentials / local models are available
- otherwise fall back to a deterministic heuristic rewrite so the pipeline remains runnable

The output is a JSONL file with per-sample rewrite results that can be inspected
or later fed into retrieval / answer stages.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.mrag import query_planner as qp


@dataclass
class RewriteSample:
    data_id: str
    image_id: str
    question: str
    backend: str
    n_dims: int
    instructions: list[str]
    retrieval_queries: list[str]
    used_fallback: bool
    elapsed_sec: float
    error: str | None = None


def _heuristic_rewrite(question: str, n_dims: int) -> list[str]:
    question_l = question.lower().strip()
    core = question.rstrip("?")
    dims = [
        f"Identify the specific entity or object needed to answer: {core}",
        f"Find visual evidence about the context or scene related to: {core}",
        f"Look for attributes, labels, or distinctive details relevant to: {core}",
    ]

    if "where" in question_l or "place" in question_l or "location" in question_l:
        dims[0] = f"Find the exact place, landmark, or location that answers: {core}"
        dims[1] = f"Find surrounding context or geographic cues related to: {core}"
    elif "who" in question_l or "person" in question_l or "name" in question_l:
        dims[0] = f"Identify the person or named entity that answers: {core}"
        dims[1] = f"Find visual cues about appearance, role, or identity for: {core}"
    elif "what" in question_l:
        dims[0] = f"Identify the object, concept, or category needed for: {core}"
        dims[1] = f"Find supporting visual context for: {core}"

    if n_dims <= len(dims):
        return dims[:n_dims]

    while len(dims) < n_dims:
        dims.append(f"Find additional evidence or alternatives related to: {core}")
    return dims[:n_dims]


def _rewrite_question(question: str, n_dims: int, backend: str, api_base: str, api_key: str, api_model: str):
    if backend == "raw_question":
        return [question], "raw_question", True, None

    if backend in ("api", "auto"):
        if api_base and api_key and api_model:
            try:
                instructions = qp.generate_retrieval_instructions(
                    question,
                    n_dims,
                    backend="api",
                    api_base=api_base,
                    api_key=api_key,
                    api_model=api_model,
                )
                if instructions:
                    return instructions, "api", False, None
            except Exception as exc:
                api_error = str(exc)
                if backend == "api":
                    return [], "api", True, api_error
        else:
            api_error = "missing api credentials"
            if backend == "api":
                return [], "api", True, api_error

    return _heuristic_rewrite(question, n_dims), "heuristic", True, None


def _load_infoseek_samples(data_root: str, split: str, max_samples: int):
    split_to_path = {
        "entity_test": Path(data_root) / "Entity" / "infoseek_test.jsonl",
        "entity_train": Path(data_root) / "Entity" / "infoseek_train.jsonl",
        "entity_val": Path(data_root) / "Entity" / "infoseek_val.jsonl",
        "human": Path(data_root) / "Human" / "infoseek_human.jsonl",
    }
    jsonl_path = split_to_path[split]
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"InfoSeek split file not found: {jsonl_path}")

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(records) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                {
                    "data_id": str(obj.get("data_id", "")),
                    "image_id": str(obj.get("image_id", "")),
                    "question": str(obj.get("question", "")),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="InfoSeek open-ended multi-dim rewrite smoke run")
    parser.add_argument("--data-root", default="/mnt/d/mRAG/data/infoseek", help="InfoSeek root directory")
    parser.add_argument("--split", default="entity_test", choices=["entity_test", "entity_train", "entity_val", "human"], help="InfoSeek split")
    parser.add_argument("--max-samples", type=int, default=100, help="Number of samples to rewrite")
    parser.add_argument("--n-dims", type=int, default=3, help="Number of rewrite dimensions")
    parser.add_argument(
        "--backend",
        choices=["auto", "api", "raw_question"],
        default="auto",
        help="Rewrite backend: auto falls back to heuristic when no API is configured",
    )
    parser.add_argument("--api-base", default=os.environ.get("DIM_GENERATOR_API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("DIM_GENERATOR_API_KEY", ""))
    parser.add_argument("--api-model", default=os.environ.get("DIM_GENERATOR_MODEL", ""))
    parser.add_argument(
        "--output-jsonl",
        default=str(Path("log") / "infoseek_open_ended_multidim_100.jsonl"),
        help="Where to write per-sample rewrite outputs",
    )
    parser.add_argument(
        "--summary-json",
        default=str(Path("log") / "infoseek_open_ended_multidim_100_summary.json"),
        help="Where to write summary stats",
    )
    args = parser.parse_args()

    records = _load_infoseek_samples(args.data_root, args.split, args.max_samples)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    total_t0 = time.time()
    used_backends: dict[str, int] = {}
    fallback_count = 0
    errors = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for record in records:
            t0 = time.time()
            instructions, used_backend, used_fallback, error = _rewrite_question(
                record["question"],
                args.n_dims,
                args.backend,
                args.api_base,
                args.api_key,
                args.api_model,
            )
            elapsed = time.time() - t0
            used_backends[used_backend] = used_backends.get(used_backend, 0) + 1
            if used_fallback:
                fallback_count += 1
            if error:
                errors += 1

            row = RewriteSample(
                data_id=record["data_id"],
                image_id=record["image_id"],
                question=record["question"],
                backend=used_backend,
                n_dims=args.n_dims,
                instructions=instructions,
                retrieval_queries=instructions,
                used_fallback=used_fallback,
                elapsed_sec=elapsed,
                error=error,
            )
            fout.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    summary = {
        "split": args.split,
        "sample_size": len(records),
        "requested_n_dims": args.n_dims,
        "backend_requested": args.backend,
        "used_backends": used_backends,
        "fallback_count": fallback_count,
        "errors": errors,
        "total_elapsed_sec": round(time.time() - total_t0, 3),
        "output_jsonl": str(out_path),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()