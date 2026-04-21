#!/usr/bin/env python3
"""
Gemma 4 E2B — download + GPU smoke / benchmark.

1) 下载节点：将权重拉到仓库 ``models/gemma4-e2b``（可用 ``--local-dir`` 覆盖）。
2) GPU 节点：默认在 ``cuda:1`` 上加载并跑文本 / 图文多模态各一轮。
3) 打印显存、首包时间（近似 TTFT）、整段与解码阶段 token/s。

依赖（Gemma 4 + 当前 Hugging Face 栈）::

    # transformers 新发行版要求 PyTorch >= 2.4，否则会「禁用」Torch，AutoProcessor 无法加载 Gemma4VideoProcessor
    pip install -U 'torch>=2.4' 'torchvision' transformers accelerate huggingface_hub pillow

    # CUDA 12.1 示例（按 https://pytorch.org 调整 index-url）
    # pip install --upgrade 'torch>=2.4' 'torchvision' --index-url https://download.pytorch.org/whl/cu121

若模型门禁，需先接受协议并设置 ``HF_TOKEN`` 或 ``huggingface-cli login``。

**transformers 过旧**（无 ``AutoModelForImageTextToText`` / ``AutoModelForMultimodalLM``）时，脚本会依次尝试
``AutoModelForCausalLM`` / ``AutoModel``；图文仍失败时请升级::

    pip install -U 'transformers>=4.51' accelerate

仅测文本可加 ``--skip-vision``。

示例::

    python test/gemma4.py --mode download
    python test/gemma4.py --mode run --device "cuda:0"
    python test/gemma4.py --mode run --skip-vision

服务器无 ``paper/images`` 时：若已设置 ``CORPUS_DIR`` 或存在 ``data/image_corpus``，会自动取其中第一张图做图文测试；
也可 ``--image $(python scripts/inspect_data_layout.py --print-one-corpus-image)``。目录说明见 ``doc/DATA_LAYOUT.md``。

---
(llava) [hzh@gpu01 mRAG]$ python test/gemma4.py --mode run --image "$(python scripts/inspect_data_layout.py --print-one-corpus-image)"
[2026-04-20 11:14:41] [mem:before_load] cuda:0 (NVIDIA A100-SXM4-80GB) allocated=0.000 GiB reserved=0.000 GiB max_allocated=0.000 GiB
[2026-04-20 11:14:41] [mem:before_load] cuda:1 (NVIDIA A100-SXM4-80GB) allocated=0.000 GiB reserved=0.000 GiB max_allocated=0.000 GiB
[2026-04-20 11:14:44] loading_from=local:/public/home/hzh/mRAG/models/gemma4-e2b
[2026-04-20 11:14:44] transformers_version=5.5.4
Loading weights: 100%|█████████████████████████████████████| 1951/1951 [00:07<00:00, 276.35it/s]
[2026-04-20 11:14:58] model_class=AutoModelForImageTextToText
[2026-04-20 11:14:58] [mem:after_load] cuda:0 (NVIDIA A100-SXM4-80GB) allocated=0.000 GiB reserved=0.000 GiB max_allocated=0.000 GiB
[2026-04-20 11:14:58] [mem:after_load] cuda:1 (NVIDIA A100-SXM4-80GB) allocated=9.508 GiB reserved=9.602 GiB max_allocated=9.508 GiB
[2026-04-20 11:14:58] === 文本-only 生成 ===
[2026-04-20 11:15:01] text_stats={'text': 'RAM是系统内存用于程序运行，而VRAM是显存专门用于图形处理和存储显卡数据的内存。', 'n_prompt_tokens': 47, 'n_new_tokens': 26, 'wall_generate_sec': 2.459, 'ttft_sec': 1.2787, 'tokens_per_sec_all': 10.57, 'tokens_per_sec_after_first': 22.03, 'peak_gib_on_device': 9.524}
[2026-04-20 11:15:01] text_parse_response={'role': 'assistant', 'content': 'RAM是系统内存用于程序运行，而VRAM是显存专门用于图形处理和存储显卡数据的内存。'}
[2026-04-20 11:15:01] === 多模态（文本+图片）image=/public/home/hzh/mRAG/data/image_corpus/Biological_0_gt_409E7C55-DDA5-442E-BEF368457F16CAA7.jpg ===
[2026-04-20 11:15:05] vision_stats={'text': '这张图片展示了一个被切开的苹果，可以看到其白色的果肉和略带黄色的果皮。', 'n_prompt_tokens': 299, 'n_new_tokens': 25, 'wall_generate_sec': 2.8433, 'ttft_sec': 1.7064, 'tokens_per_sec_all': 8.79, 'tokens_per_sec_after_first': 21.99, 'peak_gib_on_device': 10.022}
[2026-04-20 11:15:05] vision_parse_response={'role': 'assistant', 'content': '这张图片展示了一个被切开的苹果，可以看到其白色的果肉和略带黄色的果皮。'}
[2026-04-20 11:15:05] [mem:after_generate] cuda:0 (NVIDIA A100-SXM4-80GB) allocated=0.000 GiB reserved=0.000 GiB max_allocated=0.000 GiB
[2026-04-20 11:15:05] [mem:after_generate] cuda:1 (NVIDIA A100-SXM4-80GB) allocated=9.523 GiB reserved=10.090 GiB max_allocated=10.022 GiB
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mrag.envfile import load_dotenv as load_repo_dotenv
from src.mrag.gemma4_loader import load_processor_and_model, prepare_inputs
_CORPUS_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _first_image_under(dirpath: Path) -> Path | None:
    """Walk until first image (sorted names per dir); for corpus smoke only."""
    if not dirpath.is_dir():
        return None
    for dp, _, names in os.walk(dirpath, topdown=True):
        for n in sorted(names):
            if Path(n).suffix.lower() in _CORPUS_IMAGE_EXTS:
                return Path(dp) / n
    return None


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def log_cuda_memory(tag: str) -> None:
    import torch

    if not torch.cuda.is_available():
        log(f"[mem:{tag}] CUDA 不可用")
        return
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        alloc = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        peak = torch.cuda.max_memory_allocated(i) / (1024**3) if hasattr(torch.cuda, "max_memory_allocated") else 0.0
        log(f"[mem:{tag}] cuda:{i} ({name}) allocated={alloc:.3f} GiB reserved={reserved:.3f} GiB max_allocated={peak:.3f} GiB")


def resolve_local_dir(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (Path(__file__).resolve().parent / p).resolve()
    return p


def cmd_download(model_id: str, local_dir: Path, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    log(f"snapshot_download repo={model_id} -> {local_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token or None,
        resume_download=True,
    )
    log("download_done")


def _decode_new_tokens(processor, input_len: int, output_ids) -> tuple[str, int]:
    """Decode only generated suffix; return (text, n_new_tokens)."""
    new_ids = output_ids[0, input_len:]
    n = int(new_ids.shape[-1])
    text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, n


def generate_with_timings(model, processor, inputs: dict, max_new_tokens: int) -> dict:
    """流式生成：首个文本块时间近似 TTFT；同一次 ``generate`` 的返回值统计 new token 与吞吐。"""
    import logging

    import torch
    from transformers import GenerationConfig, TextIteratorStreamer

    input_len = inputs["input_ids"].shape[-1]
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    # 显式 GenerationConfig，避免默认里的 top_p/top_k 在 Gemma4 上触发 “flags not valid” 提示
    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
    gen_kwargs = dict(**inputs, generation_config=gen_cfg, streamer=streamer)

    if torch.cuda.is_available() and inputs["input_ids"].device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(inputs["input_ids"].device.index)

    out_box: list = []
    streamed_chunks: list[str] = []
    t_start = time.perf_counter()
    ttft: float | None = None
    first = True

    gen_log_utils = logging.getLogger("transformers.generation.utils")
    gen_log_cfg = logging.getLogger("transformers.generation.configuration_utils")

    def _run():
        with torch.inference_mode():
            prev_u = gen_log_utils.level
            prev_c = gen_log_cfg.level
            gen_log_utils.setLevel(logging.ERROR)
            gen_log_cfg.setLevel(logging.ERROR)
            try:
                out_box.append(model.generate(**gen_kwargs))
            finally:
                gen_log_utils.setLevel(prev_u)
                gen_log_cfg.setLevel(prev_c)

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    for ch in streamer:
        if first:
            ttft = time.perf_counter() - t_start
            first = False
        streamed_chunks.append(ch)
    th.join()
    wall = time.perf_counter() - t_start

    out = out_box[0] if out_box else None
    if out is None:
        log("generate 返回 None（部分版本在 streamer 下如此），用流式拼接文本估算 token 数")
        text_full = "".join(streamed_chunks).strip()
        enc = processor.tokenizer.encode(text_full, add_special_tokens=False)
        n_new = len(enc)
    else:
        text_full, n_new = _decode_new_tokens(processor, input_len, out)
    peak_gb = 0.0
    if torch.cuda.is_available() and inputs["input_ids"].device.type == "cuda":
        idx = inputs["input_ids"].device.index
        peak_gb = torch.cuda.max_memory_allocated(idx) / (1024**3)

    avg_all = n_new / wall if wall > 0 else 0.0
    decode_wall = max(wall - (ttft or 0.0), 1e-6)
    avg_decode = n_new / decode_wall if n_new else 0.0

    return {
        "text": text_full.strip(),
        "n_prompt_tokens": int(input_len),
        "n_new_tokens": n_new,
        "wall_generate_sec": round(wall, 4),
        "ttft_sec": round(ttft, 4) if ttft is not None else None,
        "tokens_per_sec_all": round(avg_all, 2),
        "tokens_per_sec_after_first": round(avg_decode, 2),
        "peak_gib_on_device": round(peak_gb, 3),
    }


def build_text_messages() -> list[dict]:
    """Gemma4 ``apply_chat_template`` 要求 ``content`` 为部件列表，不能是纯字符串。"""
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [{"type": "text", "text": "用一句话说明 RAM 和 VRAM 的区别，不要换行。"}],
        },
    ]


def ensure_synthetic_smoke_image() -> Path:
    """在无仓库资源时写入 96×96 PNG（边长可被 48 整除，符合 Gemma4 图像约束的常见要求）。"""
    from PIL import Image

    cache = ROOT / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "gemma4_smoke_96.png"
    if path.is_file() and path.stat().st_size > 0:
        return path
    img = Image.new("RGB", (96, 96), color=(42, 88, 168))
    img.save(path, format="PNG")
    log(f"wrote_synthetic_smoke_image={path}")
    return path


def build_vision_messages(image_path: Path) -> list[dict]:
    # Gemma4 processor expects local file path or http(s) URL; file:// URI may be rejected.
    img_path = str(image_path.expanduser().resolve())
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                # {"type": "text", "text": "用一句话详细描述这张图片里的内容。"},
                {"type": "text", "text": "Use one but concise sentence to describe the content of the image in detail."},
            ],
        }
    ]


def cmd_run(args: argparse.Namespace) -> None:
    import torch

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    device_map = args.device if args.device.startswith("cuda") else "auto"
    if device_map == "cpu":
        device_map = {"": "cpu"}

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    local_dir = resolve_local_dir(args.local_dir)

    if not (local_dir / "config.json").exists() and not args.skip_download_check:
        log(f"本地目录缺少权重: {local_dir} ，请先在本机或其它节点执行: python test/gemma4.py --mode download")
        sys.exit(2)

    log_cuda_memory("before_load")
    processor, model = load_processor_and_model(
        args.model_id,
        local_dir,
        device_map=device_map,
        token=token,
        allow_torch_below_2_4=args.allow_torch_below_2_4,
    )
    model.eval()
    log_cuda_memory("after_load")

    # --- 纯文本 ---
    log("=== 文本-only 生成 ===")
    text_inputs = prepare_inputs(processor, model, build_text_messages(), enable_thinking=False)
    stats_t = generate_with_timings(model, processor, text_inputs, max_new_tokens=args.max_new_tokens)
    log(f"text_stats={stats_t}")
    if hasattr(processor, "parse_response"):
        try:
            log(f"text_parse_response={processor.parse_response(stats_t['text'])}")
        except Exception as e:
            log(f"parse_response skipped: {e}")

    # --- 图文 ---
    if args.skip_vision:
        log("--skip-vision：跳过图文测试")
    else:
        img_path = Path(args.image) if args.image else None
        if img_path is None:
            for scan in (
                Path(os.environ["CORPUS_DIR"]).expanduser()
                if os.environ.get("CORPUS_DIR")
                else None,
                ROOT / "data" / "image_corpus",
                ROOT / "data",
            ):
                if scan is None or not scan.is_dir():
                    continue
                hit = _first_image_under(scan)
                if hit is not None:
                    img_path = hit
                    log(f"vision_probe_image_from_corpus={img_path}")
                    break
        if img_path is None:
            for c in (
                ROOT / "paper" / "images" / "overview_plots.png",
                ROOT / "paper" / "images" / "corpus_scenario_compare.png",
            ):
                if c.is_file():
                    img_path = c
                    break
        if (img_path is None or not img_path.is_file()) and not args.no_synthetic_image:
            img_path = ensure_synthetic_smoke_image()
            log(f"仓库内无候选图，使用内置测试图: {img_path}")
        if img_path is None or not img_path.is_file():
            log("未找到测试图片；可用 --image 指定，或去掉 --no-synthetic-image 以生成内置图；跳过图文测试")
        else:
            log(f"=== 多模态（文本+图片）image={img_path} ===")
            try:
                vision_inputs = prepare_inputs(
                    processor, model, build_vision_messages(img_path), enable_thinking=False
                )
                stats_v = generate_with_timings(model, processor, vision_inputs, max_new_tokens=args.max_new_tokens)
                log(f"vision_stats={stats_v}")
                if hasattr(processor, "parse_response"):
                    try:
                        log(f"vision_parse_response={processor.parse_response(stats_v['text'])}")
                    except Exception as e:
                        log(f"parse_response skipped: {e}")
            except Exception as e:
                log(f"图文测试失败: {e}")
                log("可尝试: pip install -U 'transformers>=4.51' accelerate  或  --skip-vision 只测文本")

    log_cuda_memory("after_generate")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gemma 4 E2B: download + cuda:1 benchmark")
    p.add_argument(
        "--mode",
        choices=("download", "run", "both"),
        default="both",
        help="download=仅拉权重; run=仅推理; both=先下载再推理（已存在则跳过下载）",
    )
    p.add_argument("--model-id", type=str, default="google/gemma-4-E2B-it", help="Hub repo id（含指令版建议 -it）")
    p.add_argument(
        "--local-dir",
        type=str,
        default=str(ROOT / "models" / "gemma4-e2b"),
        help="相对 test/ 或绝对路径；默认仓库 models/gemma4-e2b",
    )
    p.add_argument("--device", type=str, default="cuda:1", help='例如 cuda:1 或 {"": "cuda:0"} 用字符串 cuda:0')
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--hf-token", type=str, default="", help="或设置环境变量 HF_TOKEN")
    p.add_argument("--image", type=str, default="", help="图文测试用的本地图片路径")
    p.add_argument(
        "--skip-download-check",
        action="store_true",
        help="未检测到本地权重也继续从 Hub 拉（需网络）",
    )
    p.add_argument("--skip-vision", action="store_true", help="不跑图文多模态，仅文本（旧 transformers 可用）")
    p.add_argument(
        "--no-synthetic-image",
        action="store_true",
        help="无 --image 且仓库无 paper/images 时不自动生成内置测试图（默认会生成 .cache/gemma4_smoke_96.png）",
    )
    p.add_argument(
        "--allow-torch-below-2-4",
        action="store_true",
        help="跳过 PyTorch>=2.4 检查（不推荐；新 transformers 下仍常因 Processor 失败）",
    )
    return p


def main() -> None:
    load_repo_dotenv(ROOT / ".env")
    args = build_arg_parser().parse_args()
    local_dir = resolve_local_dir(args.local_dir)
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if args.mode in ("download", "both"):
        if args.mode == "both" and (local_dir / "config.json").exists():
            log(f"本地已有权重目录，跳过下载: {local_dir}")
        else:
            cmd_download(args.model_id, local_dir, token)

    if args.mode in ("run", "both"):
        cmd_run(args)


if __name__ == "__main__":
    main()
