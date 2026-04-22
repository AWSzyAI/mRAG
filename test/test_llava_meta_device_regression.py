#!/usr/bin/env python3
"""
Regression test for the LLaVA meta-device loading bug.

Background:
- LLaVA-NeXT uses `from_pretrained(assign=True)` in parts of the loading path.
- Passing `device_map="auto"/"balanced"` (or any non-None map) can route through
  accelerate/meta initialization and crash with:
  "You are using from_pretrained with a meta device context manager..."

This script validates our wrapper (`src.mrag.llava.load_llava`) now:
1) DOES NOT pass `device_map` to the builder callback.
2) Moves non-quantized model to an explicit target device after load.

It is intentionally model-free and fast; no HF download is required.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.mrag.llava import load_llava


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.eval_called = False
        self.to_calls: list[str] = []

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device=None, *args, **kwargs):
        self.to_calls.append(str(device))
        return super().to(device=device, *args, **kwargs)


@dataclass
class Args:
    llava_model_path: str = "dummy/path"
    llava_device_map: str = "balanced"
    llava_attn_implementation: str = "sdpa"
    llava_load_4bit: bool = False
    llava_load_8bit: bool = False
    llava_allow_cpu_offload: bool = True


def run_once(force_cuda_visible: bool) -> None:
    args = Args()
    fake_model = TinyModel()
    seen = {}

    original_cuda_available = torch.cuda.is_available
    original_cuda_count = torch.cuda.device_count

    try:
        if force_cuda_visible:
            torch.cuda.is_available = lambda: True
            torch.cuda.device_count = lambda: 2
        else:
            torch.cuda.is_available = lambda: False
            torch.cuda.device_count = lambda: 0

        def fake_builder(model_path, model_base, model_name, **kwargs):
            seen["kwargs"] = kwargs
            if kwargs.get("device_map", "__missing__") is not None:
                raise AssertionError(
                    f"Regression: device_map must be None for assign=True path, got {kwargs.get('device_map')!r}"
                )
            return object(), fake_model, object(), 2048

        logs = []
        load_llava(args, fake_builder, logs.append)

        if not fake_model.eval_called:
            raise AssertionError("Regression: model.eval() was not called")

        expected_device = "cuda:0" if force_cuda_visible else "cpu"
        if expected_device not in fake_model.to_calls:
            raise AssertionError(
                f"Regression: model.to({expected_device!r}) was not called; got {fake_model.to_calls!r}"
            )
    finally:
        torch.cuda.is_available = original_cuda_available
        torch.cuda.device_count = original_cuda_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression test for LLaVA meta-device load bug.")
    parser.add_argument(
        "--cuda-visible",
        action="store_true",
        help="Simulate CUDA-visible environment to verify model.to('cuda:0') path.",
    )
    args = parser.parse_args()

    run_once(force_cuda_visible=args.cuda_visible)
    mode = "cuda-visible" if args.cuda_visible else "cpu-only"
    print(f"[PASS] llava meta-device regression test ({mode})")


if __name__ == "__main__":
    main()
