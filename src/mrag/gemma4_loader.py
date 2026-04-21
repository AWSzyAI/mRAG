"""Shared Gemma 4 (E2B-it, etc.) processor + model loading for tests and pipeline."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from .runtime import log


def _parse_torch_version_tuple() -> tuple[int, int, int] | None:
    import torch

    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(torch.__version__).split("+")[0])
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _require_torch_for_gemma4_processor(*, allow_torch_below_2_4: bool) -> None:
    if allow_torch_below_2_4:
        return
    import torch

    tup = _parse_torch_version_tuple()
    if tup is None or tup >= (2, 4, 0):
        return
    log(
        "当前 PyTorch=%s，而本机 transformers 在导入时会要求 PyTorch>=2.4，否则内部会禁用 Torch，"
        "进而导致 Gemma4VideoProcessor / AutoProcessor 报「PyTorch was not found」。\n"
        "请升级 PyTorch 与 Torchvision（与 CUDA 匹配），例如 CUDA 12.1:\n"
        "  pip install --upgrade 'torch>=2.4' 'torchvision>=0.19' "
        "--index-url https://download.pytorch.org/whl/cu121\n"
        "若暂时不能升级，可为 Gemma4 单独建 conda 环境；强行继续可传 "
        "--allow-torch-below-2-4（仍大概率在加载 Processor 时失败）。"
        % torch.__version__
    )
    sys.exit(3)


def pick_torch_dtype():
    import torch

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_processor_and_model(
    model_id: str,
    local_dir: Path,
    device_map: str | dict,
    token: str | None,
    *,
    allow_torch_below_2_4: bool = False,
):
    """Try ImageTextToText / MultimodalLM / CausalLM / AutoModel in order."""
    _require_torch_for_gemma4_processor(allow_torch_below_2_4=allow_torch_below_2_4)
    import transformers
    from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor

    use_local = (local_dir / "config.json").exists()
    src = str(local_dir) if use_local else model_id
    log(f"loading_from={'local:' + src if use_local else 'hub:' + src}")
    log(f"transformers_version={transformers.__version__}")

    processor = AutoProcessor.from_pretrained(src, token=token or None, trust_remote_code=True)
    dt = pick_torch_dtype()
    kwargs = dict(
        token=token or None,
        trust_remote_code=True,
        device_map=device_map,
        dtype=dt,
    )

    model = None
    last_err: Exception | None = None

    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(src, **kwargs)
        log("model_class=AutoModelForImageTextToText")
    except ImportError as e:
        log(f"skip AutoModelForImageTextToText (not in this transformers): {e}")
    except Exception as e:
        last_err = e
        log(f"AutoModelForImageTextToText.from_pretrained failed: {e}")

    if model is None:
        try:
            from transformers import AutoModelForMultimodalLM

            model = AutoModelForMultimodalLM.from_pretrained(src, **kwargs)
            log("model_class=AutoModelForMultimodalLM")
        except ImportError as e:
            log(f"skip AutoModelForMultimodalLM (not in this transformers): {e}")
        except Exception as e:
            last_err = e
            log(f"AutoModelForMultimodalLM.from_pretrained failed: {e}")

    if model is None:
        try:
            model = AutoModelForCausalLM.from_pretrained(src, **kwargs)
            log(
                "model_class=AutoModelForCausalLM "
                "(旧版 transformers；图文 apply_chat_template 可能失败，建议: pip install -U 'transformers>=4.51' accelerate)"
            )
        except Exception as e:
            last_err = e
            log(f"AutoModelForCausalLM.from_pretrained failed: {e}")

    if model is None:
        try:
            model = AutoModel.from_pretrained(src, **kwargs)
            log("model_class=AutoModel (fallback)")
        except Exception as e:
            last_err = e
            log(f"AutoModel.from_pretrained failed: {e}")

    if model is None:
        hint = (
            "当前环境的 transformers 过旧，不包含 Gemma 4 所需入口类，且 CausalLM/AutoModel 加载也失败。\n"
            "请升级后再跑图文与官方示例一致：\n"
            "  pip install -U 'transformers>=4.51' accelerate huggingface_hub\n"
            "（若仍失败可再提高版本或对照 https://huggingface.co/google/gemma-4-E2B-it 说明）"
        )
        raise RuntimeError(f"{hint}\nLast error: {last_err}")

    return processor, model


def prepare_inputs(processor, model, messages: list[dict], enable_thinking: bool):
    import torch

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}
    return inputs
