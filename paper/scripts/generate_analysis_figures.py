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


def annotate_bars(ax, bars, fmt="{:.2f}", dy=0.6, fontsize=7):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


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

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(labels, counts, color="#1f4e79")
    ax.set_title("MRAG-Bench Scenario Distribution")
    ax.set_ylabel("Number of Samples")
    ax.tick_params(axis="x", rotation=25)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    annotate_bars(ax, bars, fmt="{:.0f}", dy=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "dataset_scenario_distribution.png", dpi=220)
    plt.close(fig)


def make_overlap_recall_by_scenario():
    overlap = load_json(ROOT / "log" / "retrieved_vs_gt_overlap_summary.json")
    recall = [overlap["by_scenario"][k]["avg_recall"] * 100 for k in SCENARIO_ORDER]
    no_overlap = [overlap["by_scenario"][k]["no_overlap"] / overlap["by_scenario"][k]["n"] * 100 for k in SCENARIO_ORDER]
    labels = [scenario_display(k) for k in SCENARIO_ORDER]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    bars1 = ax.bar(x - width / 2, recall, width=width, color="#2a7f62", label="Avg. GT Recall in Retrieved (%)")
    bars2 = ax.bar(x + width / 2, no_overlap, width=width, color="#c54a4a", label="No-overlap Ratio (%)")
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", rotation=25)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 70)
    ax.set_title("Retrieved-vs-GT Coverage by Scenario")
    ax.legend(frameon=False, fontsize=9)
    annotate_bars(ax, bars1, fmt="{:.1f}", dy=0.8, fontsize=7)
    annotate_bars(ax, bars2, fmt="{:.1f}", dy=0.8, fontsize=7)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "overlap_recall_by_scenario.png", dpi=220)
    plt.close(fig)


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

    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    bars = ax.bar(order, vals, color=colors)
    ax.set_ylim(45, 62.5)
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("Overall Accuracy across E0-E7")
    annotate_bars(ax, bars, fmt="{:.2f}", dy=0.15, fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "overall_accuracy_e0_e7.png", dpi=220)
    plt.close(fig)


def make_corpus_scenario_compare(exp):
    order = ["E3", "E6", "E7"]
    labels = [scenario_display(k) for k in SCENARIO_ORDER]
    x = np.arange(len(labels))
    width = 0.25
    colors = ["#f58518", "#e45756", "#54a24b"]

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for idx, run in enumerate(order):
        vals = [exp[run]["by_scenario"][k] for k in SCENARIO_ORDER]
        bars = ax.bar(x + (idx - 1) * width, vals, width=width, label=run, color=colors[idx])
        annotate_bars(ax, bars, fmt="{:.1f}", dy=0.45, fontsize=6)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", rotation=25)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(18, 64)
    ax.set_title("Scenario-wise Accuracy: Corpus Retrieval Pipelines")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "corpus_scenario_compare.png", dpi=220)
    plt.close(fig)


def make_corpus_scenario_radar(exp):
    order = ["E3", "E6", "E7"]
    names = {
        "E3": "E3 CLIP-RAG",
        "E6": "E6 CLIP+ML Rerank",
        "E7": "E7 MagicLens-RAG",
    }
    colors = {
        "E3": "#f58518",
        "E6": "#e45756",
        "E7": "#54a24b",
    }
    labels = [scenario_display(k) for k in SCENARIO_ORDER]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.8, 7.6), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(20, 65)
    ax.set_yticks([20, 30, 40, 50, 60])
    ax.set_yticklabels(["20", "30", "40", "50", "60"], fontsize=8)
    ax.grid(alpha=0.35)
    ax.set_title("Scenario Ability Map: Corpus Retrieval Pipelines", pad=22)

    for run in order:
        vals = [exp[run]["by_scenario"][k] for k in SCENARIO_ORDER]
        vals += vals[:1]
        ax.plot(angles, vals, color=colors[run], linewidth=2.2, label=names[run])
        ax.fill(angles, vals, color=colors[run], alpha=0.10)

    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "corpus_scenario_radar.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_rerank_control_compare(exp):
    order = ["E0", "E1", "E5", "E2", "E3", "E4"]
    labels = ["E0 GT", "E1 GT+ML", "E5 GT no-rerank", "E2 Ret+ML", "E3 CLIP", "E4 No-RAG"]
    vals = [exp[k]["overall"] for k in order]

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    bars = ax.bar(labels, vals, color=["#4c78a8", "#4c78a8", "#9ecae9", "#72b7b2", "#f58518", "#9d755d"])
    ax.set_ylim(48, 62.5)
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("Control and Rerank Comparisons")
    ax.tick_params(axis="x", rotation=20)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    annotate_bars(ax, bars, fmt="{:.2f}", dy=0.12, fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "control_rerank_compare.png", dpi=220)
    plt.close(fig)


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    make_dataset_scenario_distribution()
    make_overlap_recall_by_scenario()
    exp = load_experiment_metrics()
    make_overall_accuracy_bar(exp)
    make_corpus_scenario_compare(exp)
    make_corpus_scenario_radar(exp)
    make_rerank_control_compare(exp)
    print("[OK] analysis figures generated")


if __name__ == "__main__":
    main()
