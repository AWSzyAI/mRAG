#!/usr/bin/env python3
"""Generate paper figures from experiment logs.

常用方式：

    python paper/scripts/generate_analysis_figures.py

这会重新生成 `paper/images/` 下的全部分析图，其中
`corpus_scenario_radar.png` 默认画 E0--E7 的全部可比实验，并按雷达
多边形面积从大到小排序。

如果后续只想重画部分雷达图，可以改 `RADAR_RUNS`；如果新增 E8 全量
结果，则在 `load_experiment_metrics()` 里补充 E8 的 summary 路径，
并把 `"E8"` 加入 `RADAR_RUNS` 即可。
"""
import json
import os
from pathlib import Path

# 在受限环境里，Matplotlib/fontconfig 默认会尝试写用户目录缓存。
# 这里提前指定到 /tmp，避免重画图时出现一串无关的权限警告。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mrag_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mrag_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

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

# 雷达图展示的实验集合。之前在 `make_corpus_scenario_radar()` 中写死为
# ["E3", "E6", "E7"]，所以论文中的雷达图只展示了 corpus 检索线。
# 现在默认展示 E0--E7；实际绘制顺序会按雷达多边形面积从大到小自动排序。
RADAR_RUNS = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]

EXPERIMENT_NAMES = {
    "E0": "E0 GT-RAG",
    "E1": "E1 GT+ML Rerank",
    "E2": "E2 Retrieved+ML",
    "E3": "E3 CLIP-RAG",
    "E4": "E4 No-RAG",
    "E5": "E5 GT no-rerank",
    "E6": "E6 CLIP+ML Rerank",
    "E7": "E7 MagicLens-RAG",
}

EXPERIMENT_COLORS = {
    "E0": "#4c78a8",
    "E1": "#2f6b9a",
    "E2": "#72b7b2",
    "E3": "#f58518",
    "E4": "#9d755d",
    "E5": "#9ecae9",
    "E6": "#e45756",
    "E7": "#54a24b",
}

# 这两个旧 score 文件缺少 Obstruction/Occlusion 一列；论文总表中已有
# 对应数值。为了让全量雷达图能覆盖九类场景，这里显式补齐缺失项。
SCENARIO_VALUE_PATCHES = {
    "E0": {"Obstruction": 67.59},
    "E4": {"Obstruction": 57.41},
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


def patch_missing_scenarios(run: str, by_scenario: dict) -> dict:
    patched = dict(by_scenario)
    patched.update(SCENARIO_VALUE_PATCHES.get(run, {}))
    return patched


def radar_polygon_area(vals: list[float]) -> float:
    """Return the polar polygon area for equally spaced radar axes."""
    theta = 2 * np.pi / len(vals)
    return 0.5 * np.sin(theta) * sum(
        vals[idx] * vals[(idx + 1) % len(vals)] for idx in range(len(vals))
    )


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
        "by_scenario": patch_missing_scenarios(
            "E0",
            norm_scenario_dict(load_json(ROOT / "log" / "E0-MRAG_BENCH_baseline" / "llava_one_vision_gt_rag_results_score.json")),
        ),
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
        "by_scenario": patch_missing_scenarios(
            "E4",
            norm_scenario_dict(load_json(ROOT / "log" / "E4" / "e4_llava_no_rag_results_score.json")),
        ),
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
    """Draw the scenario radar chart.

    The output filename is kept as `corpus_scenario_radar.png` because
    `paper/content.tex` already references it. The content is now the full
    E0--E7 scenario ability map rather than only E3/E6/E7. The figure uses
    two side-by-side radar panels: a 0--100 global view and a tighter view
    whose outer radius equals the maximum value among all plotted runs.
    """
    order = RADAR_RUNS
    labels = [scenario_display(k) for k in SCENARIO_ORDER]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    values_by_run = {}
    for run in order:
        missing = [k for k in SCENARIO_ORDER if k not in exp[run]["by_scenario"]]
        if missing:
            raise KeyError(f"{run} is missing scenario values: {missing}")
        values_by_run[run] = [exp[run]["by_scenario"][k] for k in SCENARIO_ORDER]

    max_value = max(max(vals) for vals in values_by_run.values())
    areas_by_run = {run: radar_polygon_area(vals) for run, vals in values_by_run.items()}
    order = sorted(order, key=lambda run: areas_by_run[run], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.8), subplot_kw={"polar": True})

    panel_specs = [
        {
            "ax": axes[0],
            "title": "Global scale (0-100)",
            "ylim": (0, 100),
            "yticks": [0, 20, 40, 60, 80, 100],
        },
        {
            "ax": axes[1],
            "title": f"Zoomed scale (max={max_value:.2f})",
            "ylim": (20, max_value),
            "yticks": [20, 30, 40, 50, 60, max_value],
        },
    ]

    handles = []
    labels_for_legend = []
    for spec in panel_specs:
        ax = spec["ax"]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(*spec["ylim"])
        ax.set_yticks(spec["yticks"])
        ax.set_yticklabels([f"{tick:.0f}" if tick != max_value else f"{tick:.2f}" for tick in spec["yticks"]], fontsize=7.5)
        ax.grid(alpha=0.35)
        ax.set_title(spec["title"], pad=22, fontsize=12)

        for run in order:
            vals = values_by_run[run] + values_by_run[run][:1]
            line = ax.plot(
                angles,
                vals,
                color=EXPERIMENT_COLORS[run],
                linewidth=1.8,
                label=EXPERIMENT_NAMES.get(run, run),
            )[0]
            ax.fill(angles, vals, color=EXPERIMENT_COLORS[run], alpha=0.045)
            if ax is axes[0]:
                handles.append(line)
                labels_for_legend.append(EXPERIMENT_NAMES.get(run, run))

    fig.suptitle("Scenario Ability Map: E0-E7 Experiments", y=0.98, fontsize=17)
    fig.legend(
        handles,
        labels_for_legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
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
