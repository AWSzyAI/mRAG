#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "paper" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


SCENARIO_ORDER = [
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

SCENARIO_LABELS = {
    "Obstruction": "Occlusion",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scenario_display(name: str) -> str:
    return SCENARIO_LABELS.get(name, name)


def norm_scenario_dict(raw: dict) -> dict:
    out = {}
    for k, v in raw.items():
        if k == "Occlusion":
            out["Obstruction"] = v
        else:
            out[k] = v
    return out


def make_dataset_scenario_distribution():
    overlap = load_json(ROOT / "log" / "retrieved_vs_gt_overlap_summary.json")
    counts = [overlap["by_scenario"][k]["n"] for k in SCENARIO_ORDER]
    labels = [scenario_display(k) for k in SCENARIO_ORDER]

    plt.figure(figsize=(10, 4.8))
    bars = plt.bar(labels, counts, color="#1f4e79")
    plt.title("MRAG-Bench Scenario Distribution")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=25, ha="right")
    for b, val in zip(bars, counts):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, str(val), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "dataset_scenario_distribution.png", dpi=220)
    plt.close()


def make_overlap_recall_by_scenario():
    overlap = load_json(ROOT / "log" / "retrieved_vs_gt_overlap_summary.json")
    recall = [overlap["by_scenario"][k]["avg_recall"] * 100 for k in SCENARIO_ORDER]
    no_overlap = [overlap["by_scenario"][k]["no_overlap"] / overlap["by_scenario"][k]["n"] * 100 for k in SCENARIO_ORDER]
    labels = [scenario_display(k) for k in SCENARIO_ORDER]
    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(10.8, 5.0))
    plt.bar(x - width / 2, recall, width=width, color="#2a7f62", label="Avg. GT Recall in Retrieved (%)")
    plt.bar(x + width / 2, no_overlap, width=width, color="#c54a4a", label="No-overlap Ratio (%)")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Percentage")
    plt.title("Retrieved-vs-GT Coverage by Scenario")
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "overlap_recall_by_scenario.png", dpi=220)
    plt.close()


def load_experiment_metrics():
    data = {}
    data["E0"] = {
        "overall": 59.05,
        "by_scenario": norm_scenario_dict(load_json(ROOT / "log" / "E0-MRAG_BENCH_baseline" / "llava_one_vision_gt_rag_results_score.json")),
    }
    data["E1"] = {
        "overall": 60.16,
        "by_scenario": norm_scenario_dict(load_json(ROOT / "log" / "E1-magiclens对GT进行rerank" / "magiclens_rerank_gt_summary.json")["by_scenario"]),
    }
    data["E2"] = {
        "overall": 51.74,
        "by_scenario": norm_scenario_dict(load_json(ROOT / "log" / "E2-magiclens不用GT真RAG" / "magiclens_rerank_llava_retrieved_rag_summary.json")["by_scenario"]),
    }
    data["E3"] = {
        "overall": 50.41,
        "by_scenario": norm_scenario_dict(load_json(ROOT / "log" / "E3" / "e3_clip_corpus_rag_summary.json")["by_scenario_accuracy"]),
    }
    data["E4"] = {
        "overall": 53.14,
        "by_scenario": norm_scenario_dict(load_json(ROOT / "log" / "E4" / "e4_llava_no_rag_results_score.json")),
    }
    data["E5"] = {
        "overall": 60.16,
        "by_scenario": norm_scenario_dict(load_json(ROOT / "log" / "E5" / "e5_magiclens_gt_norerank_summary.json")["by_scenario"]),
    }
    data["E6"] = {
        "overall": 50.33,
        "by_scenario": norm_scenario_dict(load_json(ROOT / "log" / "E6" / "e6_clip_corpus_magiclens_rerank_summary.json")["by_scenario_accuracy"]),
    }
    data["E7"] = {
        "overall": 47.97,
        "by_scenario": norm_scenario_dict(load_json(ROOT / "log" / "E7" / "e7_magiclens_corpus_rag_summary.json")["by_scenario_accuracy"]),
    }
    return data


def make_overall_accuracy_bar(exp):
    order = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]
    vals = [exp[k]["overall"] for k in order]
    colors = ["#4c78a8", "#4c78a8", "#72b7b2", "#f58518", "#9d755d", "#4c78a8", "#e45756", "#54a24b"]

    plt.figure(figsize=(10.4, 4.8))
    bars = plt.bar(order, vals, color=colors)
    plt.ylim(45, 62.5)
    plt.ylabel("Overall Accuracy (%)")
    plt.title("Overall Accuracy across E0-E7")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "overall_accuracy_e0_e7.png", dpi=220)
    plt.close()


def make_corpus_scenario_compare(exp):
    order = ["E3", "E6", "E7"]
    labels = [scenario_display(k) for k in SCENARIO_ORDER]
    x = np.arange(len(labels))
    width = 0.25
    colors = ["#f58518", "#e45756", "#54a24b"]

    plt.figure(figsize=(11.5, 5.2))
    for idx, run in enumerate(order):
        vals = [exp[run]["by_scenario"][k] for k in SCENARIO_ORDER]
        plt.bar(x + (idx - 1) * width, vals, width=width, label=run, color=colors[idx])
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("Scenario-wise Accuracy: Corpus Retrieval Pipelines")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "corpus_scenario_compare.png", dpi=220)
    plt.close()


def make_rerank_control_compare(exp):
    order = ["E0", "E1", "E5", "E2", "E3", "E4"]
    labels = ["E0 GT", "E1 GT+ML", "E5 GT no-rerank", "E2 Ret+ML", "E3 CLIP", "E4 No-RAG"]
    vals = [exp[k]["overall"] for k in order]

    plt.figure(figsize=(10.8, 4.8))
    bars = plt.bar(labels, vals, color=["#4c78a8", "#4c78a8", "#9ecae9", "#72b7b2", "#f58518", "#9d755d"])
    plt.ylim(48, 62.5)
    plt.ylabel("Overall Accuracy (%)")
    plt.title("Control and Rerank Comparisons")
    plt.xticks(rotation=20, ha="right")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "control_rerank_compare.png", dpi=220)
    plt.close()


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    make_dataset_scenario_distribution()
    make_overlap_recall_by_scenario()
    exp = load_experiment_metrics()
    make_overall_accuracy_bar(exp)
    make_corpus_scenario_compare(exp)
    make_rerank_control_compare(exp)
    print("[OK] analysis figures generated")


if __name__ == "__main__":
    main()
