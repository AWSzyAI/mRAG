#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict

from datasets import load_dataset, load_from_disk


def _count_images(value):
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    return 1


def _sorted_counter_items(counter):
    def key_fn(k):
        if isinstance(k, int):
            return (0, k)
        try:
            return (0, int(str(k)))
        except Exception:
            return (1, str(k))

    return sorted(counter.items(), key=lambda kv: key_fn(kv[0]))


def _print_counter(title, counter, total):
    print(title)
    for k, v in _sorted_counter_items(counter):
        ratio = (v / total * 100.0) if total else 0.0
        print(f"  {k}: {v} ({ratio:.2f}%)")
    if not counter:
        print("  <empty>")
    print()


def _counter_to_dict(counter):
    out = {}
    for k, v in _sorted_counter_items(counter):
        out[str(k)] = v
    return out


def _plot_hist(counter, out_path, title, xlabel):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib is required for plotting: {e}") from e

    items = _sorted_counter_items(counter)
    if not items:
        raise RuntimeError("counter is empty, cannot plot histogram")

    xs = [int(k) for k, _ in items]
    ys = [int(v) for _, v in items]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    bars = ax.bar(xs, ys, color="#3a7ca5", edgecolor="#214a63")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Sample Count")
    ax.set_xticks(xs)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    max_y = max(ys) if ys else 0
    y_offset = max(2, int(max_y * 0.01))
    for bar, y in zip(bars, ys):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            str(y),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect MRAG-Bench image field distributions."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="uclanlp/MRAG-Bench",
        help="Dataset name for datasets.load_dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split name",
    )
    parser.add_argument(
        "--from-disk",
        action="store_true",
        help="Use datasets.load_from_disk on --dataset path",
    )
    parser.add_argument(
        "--from-results-jsonl",
        type=str,
        default="",
        help=(
            "Optional results jsonl path (e.g. MagicLens output) to derive "
            "retrieved count from `meta_rag_count` without loading dataset."
        ),
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save summary JSON",
    )
    parser.add_argument(
        "--plot-retrieved-hist",
        type=str,
        default="",
        help="Optional output image path for retrieved_images count histogram",
    )
    args = parser.parse_args()

    q_image_count = Counter()
    q_image_type = Counter()

    gt_count_raw = Counter()
    gt_count_after_incomplete_rule = Counter()

    retrieved_count = Counter()

    scenario_total = Counter()
    scenario_gt_raw = defaultdict(Counter)
    scenario_gt_after = defaultdict(Counter)
    scenario_q_images = defaultdict(Counter)

    if args.from_results_jsonl:
        source = f"results_jsonl:{args.from_results_jsonl}"
        total = 0
        with open(args.from_results_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                total += 1
                scenario = item.get("scenario", "<missing>")
                scenario_total[scenario] += 1

                if "meta_rag_count" in item:
                    ret_cnt = int(item.get("meta_rag_count", 0))
                    retrieved_count[ret_cnt] += 1
                elif "retrieved_images" in item:
                    ret_cnt = _count_images(item.get("retrieved_images"))
                    retrieved_count[ret_cnt] += 1
    else:
        if args.from_disk:
            ds = load_from_disk(args.dataset)
            try:
                ds = ds[args.split]
            except Exception:
                pass
            source = f"disk:{args.dataset}"
        else:
            ds = load_dataset(args.dataset, split=args.split)
            source = f"hf:{args.dataset}"

        total = len(ds)

        for item in ds:
            scenario = item.get("scenario", "<missing>")
            scenario_total[scenario] += 1

            img_value = item.get("image")
            q_cnt = _count_images(img_value)
            q_image_count[q_cnt] += 1
            q_image_type[type(img_value).__name__] += 1
            scenario_q_images[scenario][q_cnt] += 1

            gt_value = item.get("gt_images")
            gt_raw = _count_images(gt_value)
            gt_count_raw[gt_raw] += 1
            scenario_gt_raw[scenario][gt_raw] += 1

            gt_after = gt_raw
            if scenario == "Incomplete" and gt_after > 0:
                gt_after = 1
            gt_count_after_incomplete_rule[gt_after] += 1
            scenario_gt_after[scenario][gt_after] += 1

            if "retrieved_images" in item:
                ret_cnt = _count_images(item.get("retrieved_images"))
                retrieved_count[ret_cnt] += 1

    print(f"Source: {source}")
    print(f"Split: {args.split}")
    print(f"Total samples: {total}\n")

    if q_image_count:
        _print_counter("Question image count per sample (`image`):", q_image_count, total)
    if gt_count_raw:
        _print_counter(
            "GT image count per sample (`gt_images`) [raw]:", gt_count_raw, total
        )
    if gt_count_after_incomplete_rule:
        _print_counter(
            "GT image count per sample after dataloader Incomplete rule:",
            gt_count_after_incomplete_rule,
            total,
        )
    if retrieved_count:
        _print_counter(
            "Retrieved image count per sample (`retrieved_images`) [raw]:",
            retrieved_count,
            total,
        )

    if q_image_type:
        print("Question image value types (`type(item['image']).__name__`):")
        for k, v in _sorted_counter_items(q_image_type):
            ratio = (v / total * 100.0) if total else 0.0
            print(f"  {k}: {v} ({ratio:.2f}%)")
        print()

    print("Per-scenario summary:")
    for scenario, scen_total in sorted(scenario_total.items(), key=lambda kv: kv[0]):
        print(f"- {scenario} (n={scen_total})")
        if scenario_q_images:
            qimg = ", ".join(
                [f"{k}:{v}" for k, v in _sorted_counter_items(scenario_q_images[scenario])]
            )
            print(f"  question_images: {qimg or '<empty>'}")
        if scenario_gt_raw:
            raw = ", ".join(
                [f"{k}:{v}" for k, v in _sorted_counter_items(scenario_gt_raw[scenario])]
            )
            print(f"  gt_images_raw: {raw or '<empty>'}")
        if scenario_gt_after:
            after = ", ".join(
                [f"{k}:{v}" for k, v in _sorted_counter_items(scenario_gt_after[scenario])]
            )
            print(f"  gt_images_after_incomplete_rule: {after or '<empty>'}")
    print()

    summary = {
        "source": source,
        "split": args.split,
        "total_samples": total,
        "question_image_count": _counter_to_dict(q_image_count),
        "question_image_types": _counter_to_dict(q_image_type),
        "gt_images_count_raw": _counter_to_dict(gt_count_raw),
        "gt_images_count_after_incomplete_rule": _counter_to_dict(
            gt_count_after_incomplete_rule
        ),
        "retrieved_images_count": _counter_to_dict(retrieved_count),
        "per_scenario": {
            scenario: {
                "total": scen_total,
                "question_images": _counter_to_dict(scenario_q_images[scenario]),
                "gt_images_raw": _counter_to_dict(scenario_gt_raw[scenario]),
                "gt_images_after_incomplete_rule": _counter_to_dict(
                    scenario_gt_after[scenario]
                ),
            }
            for scenario, scen_total in sorted(
                scenario_total.items(), key=lambda kv: kv[0]
            )
        },
    }

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON summary to: {args.save_json}")

    if args.plot_retrieved_hist:
        if not retrieved_count:
            raise RuntimeError("dataset has no `retrieved_images` field")
        _plot_hist(
            retrieved_count,
            args.plot_retrieved_hist,
            title="MRAG-Bench: retrieved_images Count Distribution",
            xlabel="retrieved_images per Sample",
        )
        print(f"Saved histogram to: {args.plot_retrieved_hist}")


if __name__ == "__main__":
    main()
