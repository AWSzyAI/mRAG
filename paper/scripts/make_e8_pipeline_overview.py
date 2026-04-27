#!/usr/bin/env python3
"""Create a dense E8 pipeline overview figure from existing E8 visual assets."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_PAPER = ROOT / "paper" / "images"
OUT_LOG = ROOT / "log" / "E8" / "figures"

FINAL_COLLAGE = ROOT / "log" / "E8" / "figures" / "qs_0_query_plus_final_top5_1x6.png"
GRID_COLLAGE = ROOT / "log" / "E8" / "figures" / "qs_0_dim_topk_grid.png"
TRACE_PATH = ROOT / "log" / "E8_full" / "e8_full_trace.jsonl"


W, H = 2400, 1560
PAD = 34

NAVY = "#111827"
MUTED = "#475569"
PURPLE = "#6d28d9"
GREEN = "#15803d"
ORANGE = "#ea580c"
BLUE = "#1d4ed8"
BG = "#f8fafc"
PANEL = "#ffffff"
GRID_BG = "#f1f5f9"
LINE = "#cbd5e1"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


F_TITLE = font(38, True)
F_SUB = font(22, False)
F_H = font(28, True)
F_M = font(22, True)
F_S = font(18, False)
F_XS = font(15, False)
F_NUM = font(20, True)


def rounded(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_center(draw: ImageDraw.ImageDraw, box, text, fnt, fill=NAVY, spacing=4):
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        b = draw.textbbox((0, 0), line, font=fnt)
        widths.append(b[2] - b[0])
        heights.append(b[3] - b[1])
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = y0 + (y1 - y0 - total_h) / 2
    for line, tw, th in zip(lines, widths, heights):
        draw.text((x0 + (x1 - x0 - tw) / 2, y), line, font=fnt, fill=fill)
        y += th + spacing


def arrow(draw: ImageDraw.ImageDraw, start, end, fill=NAVY, width=5):
    draw.line([start, end], fill=fill, width=width)
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head = 18
    wing = 10
    p1 = (x1, y1)
    p2 = (x1 - head * ux + wing * px, y1 - head * uy + wing * py)
    p3 = (x1 - head * ux - wing * px, y1 - head * uy - wing * py)
    draw.polygon([p1, p2, p3], fill=fill)


def paste_fit(canvas: Image.Image, img: Image.Image, box, cover=False):
    x0, y0, x1, y1 = map(int, box)
    bw, bh = x1 - x0, y1 - y0
    iw, ih = img.size
    scale = max(bw / iw, bh / ih) if cover else min(bw / iw, bh / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    if cover:
        left = max(0, (nw - bw) // 2)
        top = max(0, (nh - bh) // 2)
        img = img.crop((left, top, left + bw, top + bh))
        canvas.paste(img, (x0, y0))
    else:
        canvas.paste(img, (x0 + (bw - nw) // 2, y0 + (bh - nh) // 2))


def tile_cells(canvas: Image.Image, source: Image.Image, box, rows: int, cols: int, gutter: int = 8):
    """Crop source into a rows x cols grid and retile it to fill box."""
    x0, y0, x1, y1 = map(int, box)
    bw, bh = x1 - x0, y1 - y0
    cell_w = (bw - gutter * (cols - 1)) / cols
    cell_h = (bh - gutter * (rows - 1)) / rows
    sw, sh = source.size
    src_w = sw / cols
    src_h = sh / rows
    for r in range(rows):
        for c in range(cols):
            crop = source.crop((
                int(c * src_w),
                int(r * src_h),
                int((c + 1) * src_w),
                int((r + 1) * src_h),
            ))
            dst = (
                int(x0 + c * (cell_w + gutter)),
                int(y0 + r * (cell_h + gutter)),
                int(x0 + c * (cell_w + gutter) + cell_w),
                int(y0 + r * (cell_h + gutter) + cell_h),
            )
            paste_fit(canvas, crop, dst, cover=True)


def load_trace():
    with TRACE_PATH.open("r", encoding="utf-8") as f:
        return json.loads(f.readline())


def main():
    OUT_PAPER.mkdir(parents=True, exist_ok=True)
    OUT_LOG.mkdir(parents=True, exist_ok=True)

    trace = load_trace()
    dims = [q["query"] for q in trace["dimension_generation"]["queries"]]
    final = Image.open(FINAL_COLLAGE).convert("RGB")
    grid = Image.open(GRID_COLLAGE).convert("RGB")

    # Crop image groups from existing visual assets.
    query_img = final.crop((20, 118, 262, 360))
    final_top5 = final.crop((276, 118, 1543, 360))
    retrieval_5x5 = grid.crop((294, 45, 1540, 1290))

    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # Full-width header.
    draw.rectangle((0, 0, W, 132), fill=NAVY)
    draw.text((PAD, 24), "E8: Gemma4-Guided Multi-Dimensional Query Rewriting + MagicLens Hybrid Retrieval", font=F_TITLE, fill="white")
    draw.text((PAD, 76), "query image + question -> 5 retrieval dimensions -> 5x5 MagicLens evidence pool -> RRF Top-5 -> fixed VLM answer", font=F_SUB, fill="#cbd5e1")

    # Panel geometry.
    left = (PAD, 154, 520, 1192)
    grid_panel = (536, 154, W - PAD, 1192)
    bottom = (PAD, 1210, W - PAD, H - PAD)

    rounded(draw, left, 18, PANEL, "#bfdbfe", 3)
    rounded(draw, grid_panel, 18, PANEL, "#bbf7d0", 3)
    rounded(draw, bottom, 18, PANEL, "#fed7aa", 3)

    # Left panel: query and planning.
    draw.text((left[0] + 22, left[1] + 18), "(a) Query + planning", font=F_H, fill=PURPLE)
    q_box = (left[0] + 30, left[1] + 72, left[2] - 30, left[1] + 430)
    rounded(draw, q_box, 16, "#eff6ff", "#60a5fa", 3)
    paste_fit(canvas, query_img, (q_box[0] + 18, q_box[1] + 18, q_box[2] - 18, q_box[3] - 76), cover=True)
    text_center(draw, (q_box[0] + 10, q_box[3] - 58, q_box[2] - 10, q_box[3] - 12), "Question: Can you identify this animal?\nChoices: A/B/C/D", F_S, fill=NAVY)

    planner = (left[0] + 30, left[1] + 456, left[2] - 30, left[1] + 594)
    rounded(draw, planner, 16, "#f5f3ff", "#8b5cf6", 3)
    text_center(draw, planner, "Gemma4 Planner\nimage + question + choices\n-> 5 retrieval dimensions", F_M, fill=NAVY)

    y = left[1] + 620
    for idx, dim in enumerate(dims, start=1):
        box = (left[0] + 30, y, left[2] - 30, y + 72)
        rounded(draw, box, 14, "#ffffff", "#ddd6fe", 2)
        draw.ellipse((box[0] + 14, box[1] + 17, box[0] + 50, box[1] + 53), fill="#ede9fe", outline="#7c3aed", width=2)
        text_center(draw, (box[0] + 14, box[1] + 17, box[0] + 50, box[1] + 53), f"{idx}", F_NUM, fill=PURPLE)
        wrapped = "\n".join(wrap(dim, width=42)[:2])
        draw.text((box[0] + 64, box[1] + 10), wrapped, font=F_XS, fill=NAVY)
        y += 78

    # Main retrieval panel.
    draw.text((grid_panel[0] + 22, grid_panel[1] + 18), "(b) Per-dimension MagicLens retrieval: 5 dimensions x Top-5", font=F_H, fill=GREEN)
    draw.text((grid_panel[2] - 552, grid_panel[1] + 24), "Rows: Gemma4 queries | Columns: ranks 1-5", font=F_S, fill=MUTED)
    grid_img_box = (grid_panel[0] + 126, grid_panel[1] + 92, grid_panel[2] - 26, grid_panel[3] - 26)
    tile_cells(canvas, retrieval_5x5, grid_img_box, 5, 5, gutter=7)

    # Labels flush to the 5x5 grid.
    gx0, gy0, gx1, gy1 = grid_img_box
    row_h = (gy1 - gy0) / 5
    for i in range(5):
        y0 = int(gy0 + i * row_h)
        label_box = (grid_panel[0] + 18, y0 + 10, grid_panel[0] + 114, y0 + int(row_h) - 10)
        rounded(draw, label_box, 12, "#ecfdf5", "#22c55e", 2)
        text_center(draw, label_box, f"t_{i+1}", F_M, fill=GREEN)
    for j in range(5):
        x = int(gx0 + (j + 0.5) * (gx1 - gx0) / 5)
        text_center(draw, (x - 72, grid_panel[1] + 58, x + 72, grid_panel[1] + 86), f"rank {j+1}", F_S, fill=MUTED)

    # Bottom: fusion and final Top-5.
    draw.text((bottom[0] + 24, bottom[1] + 18), "(c) RRF evidence fusion and fixed VLM answering", font=F_H, fill=ORANGE)
    final_box = (bottom[0] + 570, bottom[1] + 70, bottom[0] + 1710, bottom[3] - 24)
    rounded(draw, final_box, 16, "#fff7ed", "#fb923c", 3)
    tile_cells(canvas, final_top5, (final_box[0] + 20, final_box[1] + 48, final_box[2] - 20, final_box[3] - 20), 1, 5, gutter=8)
    text_center(draw, (final_box[0], final_box[1] + 8, final_box[2], final_box[1] + 42), "Final RRF Evidence Set (Top-5 candidates)", F_M, fill=ORANGE)

    # Compact process boxes in bottom band.
    steps = [
        ((bottom[0] + 30, bottom[1] + 80, bottom[0] + 245, bottom[3] - 36), "RRF Fusion\nsum 1/(60+rank)\nselect Top-5", ORANGE),
        ((bottom[0] + 300, bottom[1] + 80, bottom[0] + 520, bottom[3] - 36), "Gemma4\nEvidence\nDescriptions", PURPLE),
        ((bottom[0] + 1760, bottom[1] + 80, bottom[0] + 1990, bottom[3] - 36), "LLaVA-OneVision\nfixed VLM\nA/B/C/D", BLUE),
        ((bottom[0] + 2038, bottom[1] + 80, bottom[2] - 24, bottom[3] - 36), "E8 full result\n1353 samples\n56.32% Acc\n+8.35 vs E7", GREEN),
    ]
    for box, label, color in steps:
        rounded(draw, box, 15, "#ffffff", color, 3)
        text_center(draw, box, label, F_M if "E8" not in label else F_S, fill=NAVY)

    arrow(draw, (bottom[0] + 245, (bottom[1] + bottom[3]) // 2), (bottom[0] + 300, (bottom[1] + bottom[3]) // 2), ORANGE, 5)
    arrow(draw, (bottom[0] + 520, (bottom[1] + bottom[3]) // 2), (final_box[0] - 16, (bottom[1] + bottom[3]) // 2), ORANGE, 5)
    arrow(draw, (final_box[2] + 16, (bottom[1] + bottom[3]) // 2), (bottom[0] + 1760, (bottom[1] + bottom[3]) // 2), BLUE, 5)
    arrow(draw, (bottom[0] + 1990, (bottom[1] + bottom[3]) // 2), (bottom[0] + 2038, (bottom[1] + bottom[3]) // 2), GREEN, 5)

    # Cross-panel arrows.
    arrow(draw, (left[2] - 4, planner[1] + 68), (grid_panel[0] + 112, grid_panel[1] + 250), PURPLE, 5)
    arrow(draw, (grid_panel[0] + 110, grid_panel[3] - 18), (bottom[0] + 140, bottom[1] + 80), ORANGE, 5)

    out_png = OUT_PAPER / "e8_pipeline_overview.png"
    out_pdf = OUT_PAPER / "e8_pipeline_overview.pdf"
    canvas.save(out_png, quality=95)
    canvas.save(out_pdf, "PDF", resolution=300)
    canvas.save(OUT_LOG / "e8_pipeline_overview.png", quality=95)
    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()
