#!/usr/bin/env python3
"""Summarize MRAG parameter sweep outputs into CSV/JSON/Markdown."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCENARIOS = [
    "Angle",
    "Partial",
    "Scope",
    "Obstruction",
    "Temporal",
    "Deformation",
    "Incomplete",
    "Biological",
    "Others",
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def trace_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "trace_rows": 0,
            "avg_input_candidates": None,
            "avg_unique_candidates": None,
            "avg_unique_ratio": None,
            "avg_sample_total_time_sec": None,
            "avg_description_time_sec": None,
        }
    rows = 0
    input_total = 0.0
    unique_total = 0.0
    sample_total = 0.0
    sample_total_rows = 0
    description_total = 0.0
    description_rows = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            fusion = row.get("fusion") or {}
            input_total += float(fusion.get("input_candidate_count") or 0)
            unique_total += float(fusion.get("unique_candidate_count") or 0)
            timings = row.get("timings_sec") or {}
            if timings.get("sample_total") is not None:
                sample_total += float(timings.get("sample_total") or 0)
                sample_total_rows += 1
            if timings.get("gemma4_image_descriptions") is not None:
                description_total += float(timings.get("gemma4_image_descriptions") or 0)
                description_rows += 1
            rows += 1
    avg_input = input_total / rows if rows else None
    avg_unique = unique_total / rows if rows else None
    return {
        "trace_rows": rows,
        "avg_input_candidates": round(avg_input, 3) if avg_input is not None else None,
        "avg_unique_candidates": round(avg_unique, 3) if avg_unique is not None else None,
        "avg_unique_ratio": round(unique_total / input_total, 4) if input_total else None,
        "avg_sample_total_time_sec": round(sample_total / sample_total_rows, 3) if sample_total_rows else None,
        "avg_description_time_sec": round(description_total / description_rows, 3) if description_rows else None,
    }


def pct(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "exp",
        "sweep",
        "n_dims",
        "dim_top_k",
        "processed",
        "accuracy",
        "avg_total",
        "avg_unique",
        "unique_ratio",
    ]
    lines = ["# 参数实验结果汇总", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [
            row.get("exp", ""),
            row.get("sweep", ""),
            row.get("n_dims", ""),
            row.get("dim_top_k", ""),
            row.get("processed", ""),
            pct(row.get("accuracy")),
            row.get("avg_sample_total_time_sec", ""),
            row.get("avg_unique_candidates", ""),
            row.get("avg_unique_ratio", ""),
        ]
        lines.append("| " + " | ".join(map(str, values)) + " |")

    lines.extend(["", "## 分场景准确率", ""])
    scenario_headers = ["exp", *SCENARIOS]
    lines.append("| " + " | ".join(scenario_headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(scenario_headers)) + " |")
    for row in rows:
        values = [row.get("exp", "")]
        values.extend(pct(row.get(f"acc_{sc}")) for sc in SCENARIOS)
        lines.append("| " + " | ".join(map(str, values)) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_rows(root: Path, include_e8: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        config = read_json(exp_dir / "config.json") or {}
        prefix = config.get("prefix") or exp_dir.name.lower()
        summary = read_json(exp_dir / f"{prefix}_summary.json")
        if summary is None:
            summaries = sorted(exp_dir.glob("*_summary.json"))
            summary = read_json(summaries[0]) if summaries else None
        trace_path = exp_dir / f"{prefix}_trace.jsonl"
        if not trace_path.is_file():
            traces = sorted(exp_dir.glob("*_trace.jsonl"))
            trace_path = traces[0] if traces else trace_path
        stats = trace_stats(trace_path)
        row = {
            "exp": exp_dir.name,
            "status": "ok" if summary else "missing",
            "sweep": config.get("sweep", ""),
            "n_dims": config.get("n_dims", summary.get("n_dims") if summary else ""),
            "dim_top_k": config.get("dim_top_k", summary.get("dim_top_k") if summary else ""),
            "final_top_k": config.get("final_top_k", summary.get("final_top_k") if summary else ""),
            "fusion_strategy": config.get("fusion_strategy", summary.get("fusion_strategy") if summary else ""),
            "processed": summary.get("processed") if summary else "",
            "correct": summary.get("correct") if summary else "",
            "accuracy": summary.get("accuracy") if summary else "",
            "dim_gen_failures": summary.get("dim_gen_failures") if summary else "",
            "avg_dim_gen_time_sec": summary.get("avg_dim_gen_time_sec") if summary else "",
            "avg_retrieval_time_sec": summary.get("avg_retrieval_time_sec") if summary else "",
            "avg_final_answer_time_sec": summary.get("avg_final_answer_time_sec") if summary else "",
            **stats,
        }
        if summary:
            if row.get("avg_sample_total_time_sec") is None:
                total = 0.0
                for key in ("avg_dim_gen_time_sec", "avg_retrieval_time_sec", "avg_final_answer_time_sec"):
                    try:
                        total += float(summary.get(key) or 0)
                    except Exception:
                        pass
                row["avg_sample_total_time_sec"] = round(total, 3)
            by_scenario = summary.get("by_scenario_accuracy") or {}
            for sc in SCENARIOS:
                row[f"acc_{sc}"] = by_scenario.get(sc, "")
        rows.append(row)

    if include_e8:
        e8 = read_json(Path("log/E8_full/e8_full_summary.json"))
        if e8:
            stats = trace_stats(Path("log/E8_full/e8_full_trace.jsonl"))
            row = {
                "exp": "E8_baseline_n5_k5",
                "status": "ok",
                "sweep": "baseline",
                "n_dims": e8.get("n_dims", 5),
                "dim_top_k": e8.get("dim_top_k", 5),
                "final_top_k": e8.get("final_top_k", 5),
                "fusion_strategy": e8.get("fusion_strategy", "rrf"),
                "processed": e8.get("processed"),
                "correct": e8.get("correct"),
                "accuracy": e8.get("accuracy"),
                "dim_gen_failures": e8.get("dim_gen_failures"),
                "avg_dim_gen_time_sec": e8.get("avg_dim_gen_time_sec"),
                "avg_retrieval_time_sec": e8.get("avg_retrieval_time_sec"),
                "avg_final_answer_time_sec": e8.get("avg_final_answer_time_sec"),
                **stats,
            }
            if row.get("avg_sample_total_time_sec") is None:
                total = 0.0
                for key in ("avg_dim_gen_time_sec", "avg_retrieval_time_sec", "avg_final_answer_time_sec"):
                    try:
                        total += float(e8.get(key) or 0)
                    except Exception:
                        pass
                row["avg_sample_total_time_sec"] = round(total, 3)
            by_scenario = e8.get("by_scenario_accuracy") or {}
            for sc in SCENARIOS:
                row[f"acc_{sc}"] = by_scenario.get(sc, "")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("log/ParamSweep"))
    parser.add_argument("--csv-out", type=Path, default=Path("log/ParamSweep/param_sweep_results.csv"))
    parser.add_argument("--json-out", type=Path, default=Path("log/ParamSweep/param_sweep_results.json"))
    parser.add_argument("--md-out", type=Path, default=Path("log/ParamSweep/param_sweep_results.md"))
    parser.add_argument("--include-e8-baseline", action="store_true")
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.root, include_e8=args.include_e8_baseline)
    rows.sort(
        key=lambda r: (
            str(r.get("sweep", "")),
            int(r.get("n_dims") or 0),
            int(r.get("dim_top_k") or 0),
            str(r.get("exp")),
        )
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with args.csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, args.md_out)

    print(f"rows={len(rows)}")
    print(f"csv={args.csv_out}")
    print(f"json={args.json_out}")
    print(f"md={args.md_out}")


if __name__ == "__main__":
    main()
