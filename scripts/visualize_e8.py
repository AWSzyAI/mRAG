#!/usr/bin/env python3
"""Visualize E8 multi-dim MagicLens retrieval results.

Generates:
1) A pure-image 1x5 collage using per-dimension top-1 retrieval images.
2) A pure-image 1x(1+5) collage with query image + fused final top-5 retrieval images.
3) A pure-image dimension x rank grid. No labels are drawn inside the figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def resolve_image_path(raw_path: str, repo_root: Path, allow_local_fallback: bool) -> Path:
    p = Path(raw_path)
    if p.exists():
        return p
    if not allow_local_fallback:
        return p
    remote_prefix = "/public/home/hzh/mRAG/"
    if raw_path.startswith(remote_prefix):
        candidate = repo_root / raw_path[len(remote_prefix) :]
        if candidate.exists():
            return candidate
    return p


def load_or_placeholder(path: Path, label: str, tile_size: int) -> Image.Image:
    if path.exists():
        img = Image.open(path).convert("RGB")
        return img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tile_size, tile_size), color=(45, 45, 45))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    msg = f"MISSING\n{label}\n{path.name}"
    draw.multiline_text((12, 12), msg, fill=(240, 240, 240), font=font, spacing=4)
    return canvas


def draw_collage(
    images: list[Image.Image],
    output_path: Path,
    tile_size: int = 240,
) -> None:
    cols = len(images)
    gap = max(6, tile_size // 32)
    pad = gap
    width = pad * 2 + cols * tile_size + (cols - 1) * gap
    height = pad * 2 + tile_size
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y0 = pad
    for i, img in enumerate(images):
        x = pad + i * (tile_size + gap)
        canvas.paste(img, (x, y0))
        draw.rectangle((x - 1, y0 - 1, x + tile_size, y0 + tile_size), outline=(210, 210, 210), width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def draw_dim_topk_grid(
    sample_trace: dict[str, Any],
    output_path: Path,
    repo_root: Path,
    allow_local_fallback: bool,
    tile_size: int = 220,
) -> dict[str, Any]:
    dim_calls = sample_trace["magiclens_retrieval"]["calls"]
    rows = len(dim_calls)
    cols = max((len(c.get("top_k", [])) for c in dim_calls), default=0)
    if rows == 0 or cols == 0:
        raise ValueError("No per-dimension retrieval rows to render.")

    gap = max(5, tile_size // 40)
    pad = gap
    width = pad * 2 + cols * tile_size + (cols - 1) * gap
    height = pad * 2 + rows * tile_size + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Duplicate hit highlighting by image id.
    id_counts: dict[str, int] = {}
    for call in dim_calls:
        for r in call.get("top_k", []):
            rid = str(r.get("id", ""))
            if rid:
                id_counts[rid] = id_counts.get(rid, 0) + 1

    missing: list[str] = []
    for r, call in enumerate(dim_calls):
        y = pad + r * (tile_size + gap)
        topk = call.get("top_k", [])
        for c in range(cols):
            x = pad + c * (tile_size + gap)
            if c < len(topk):
                row = topk[c]
                p = resolve_image_path(str(row.get("path", "")), repo_root, allow_local_fallback)
                if not p.exists():
                    missing.append(str(p))
                img = load_or_placeholder(p, label=f"d{r+1}r{c+1}", tile_size=tile_size)
                canvas.paste(img, (x, y))
                rid = str(row.get("id", ""))
                dup = id_counts.get(rid, 0)
                border = (210, 70, 70) if dup > 1 else (210, 210, 210)
                draw.rectangle((x - 1, y - 1, x + tile_size, y + tile_size), outline=border, width=1)
            else:
                draw.rectangle((x, y, x + tile_size, y + tile_size), outline=(220, 220, 220))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {"missing_paths": sorted(set(missing)), "rows": rows, "cols": cols}


def render_for_sample(
    sample_trace: dict[str, Any],
    output_dir: Path,
    repo_root: Path,
    allow_local_fallback: bool,
    strict_missing: bool,
    tile_size: int = 240,
) -> dict[str, Any]:
    sample = sample_trace["sample"]
    qs_id = sample["qs_id"]

    dim_calls = sample_trace["magiclens_retrieval"]["calls"]
    dim_queries = [c["query"] for c in dim_calls]

    per_dim_top1_paths = [
        resolve_image_path(c["top_k"][0]["path"], repo_root, allow_local_fallback) for c in dim_calls if c.get("top_k")
    ]
    per_dim_imgs = [
        load_or_placeholder(p, label=f"dim{i + 1}", tile_size=tile_size) for i, p in enumerate(per_dim_top1_paths)
    ]
    dim_collage_path = output_dir / f"qs_{qs_id}_dim_top1_1x5.png"
    draw_collage(
        images=per_dim_imgs,
        output_path=dim_collage_path,
        tile_size=tile_size,
    )

    grid_path = output_dir / f"qs_{qs_id}_dim_topk_grid.png"
    grid_meta = draw_dim_topk_grid(
        sample_trace,
        output_path=grid_path,
        repo_root=repo_root,
        allow_local_fallback=allow_local_fallback,
        tile_size=max(180, min(tile_size, 240)),
    )

    query_path = resolve_image_path(sample_trace["input"]["query_image"]["path"], repo_root, allow_local_fallback)
    query_img = load_or_placeholder(query_path, label="query", tile_size=tile_size)

    final_selected = sample_trace["fusion"]["selected"][:5]
    final_paths = [resolve_image_path(x["path"], repo_root, allow_local_fallback) for x in final_selected]
    final_imgs = [load_or_placeholder(p, label=f"top{i + 1}", tile_size=tile_size) for i, p in enumerate(final_paths)]

    merged_images = [query_img] + final_imgs
    final_collage_path = output_dir / f"qs_{qs_id}_query_plus_final_top5_1x6.png"
    draw_collage(
        images=merged_images,
        output_path=final_collage_path,
        tile_size=tile_size,
    )

    missing_paths = [str(p) for p in [query_path, *per_dim_top1_paths, *final_paths] if not p.exists()]
    if strict_missing and missing_paths:
        missing_msg = "\n".join(missing_paths)
        raise FileNotFoundError(f"Missing images ({len(missing_paths)}):\n{missing_msg}")

    return {
        "qs_id": qs_id,
        "dim_queries": dim_queries,
        "dim_collage_path": str(dim_collage_path),
        "final_collage_path": str(final_collage_path),
        "dim_topk_grid_path": str(grid_path),
        "llava_prompt_question_part": sample_trace["llava_answer"]["prompt_question_part"],
        "missing_image_paths": sorted(set(missing_paths + grid_meta["missing_paths"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize log/E8 retrieval results.")
    parser.add_argument(
        "--trace-jsonl",
        default="log/E8/gemma4_multi_dim_smoke_trace.jsonl",
        help="Trace JSONL path.",
    )
    parser.add_argument("--qs-id", default="0", help="Which qs_id to visualize.")
    parser.add_argument("--all", action="store_true", help="Render all qs_id in trace JSONL.")
    parser.add_argument(
        "--out-dir",
        default="log/E8/figures",
        help="Output directory for generated collages.",
    )
    parser.add_argument("--tile-size", type=int, default=240, help="Single image tile size.")
    parser.add_argument(
        "--allow-local-fallback",
        action="store_true",
        help="If absolute server path is missing, try mapping /public/home/hzh/mRAG/* to local repo root.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail immediately if any referenced image path does not exist.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    traces = load_jsonl(Path(args.trace_jsonl))
    out_dir = Path(args.out_dir)
    reports: list[dict[str, Any]] = []
    selected = traces if args.all else [next((x for x in traces if x["sample"]["qs_id"] == str(args.qs_id)), None)]
    if not args.all and selected[0] is None:
        raise SystemExit(f"qs_id={args.qs_id} not found in {args.trace_jsonl}")
    for st in selected:
        if st is None:
            continue
        report = render_for_sample(
            st,
            out_dir,
            repo_root,
            allow_local_fallback=args.allow_local_fallback,
            strict_missing=args.strict_missing,
            tile_size=args.tile_size,
        )
        reports.append(report)
        print(f"Saved: {report['dim_collage_path']}")
        print(f"Saved: {report['final_collage_path']}")
        print(f"Saved: {report['dim_topk_grid_path']}")
        if report["missing_image_paths"]:
            print(f"Warning: qs_id={report['qs_id']} missing={len(report['missing_image_paths'])}")
    report_path = out_dir / ("all_visualization_report.json" if args.all else f"qs_{args.qs_id}_visualization_report.json")
    payload: Any = reports if args.all else reports[0]
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
