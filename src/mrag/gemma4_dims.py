"""Use Gemma 4 multimodal (e.g. E2B-it) to produce MagicLens retrieval instruction lines."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import GenerationConfig

from .gemma4_loader import prepare_inputs
from .query_planner import DIMENSION_SYSTEM_PROMPT, parse_dimension_lines


def build_dimension_vision_messages(question: str, n_dims: int, image_path: str | Path) -> list[dict]:
    """Chat messages: system text + user image + question (Gemma4 content parts)."""
    system_text = DIMENSION_SYSTEM_PROMPT.format(n_dims=n_dims)
    img = str(Path(image_path).expanduser().resolve())
    user_text = (
        f"Question about the query image:\n{question}\n\n"
        f"Output exactly {n_dims} instruction lines as specified above. No preamble."
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def generate_retrieval_instructions_gemma4(
    processor,
    model,
    *,
    query_image: str | Path,
    question: str,
    n_dims: int,
    max_new_tokens: int = 512,
    enable_thinking: bool = False,
) -> list[str]:
    messages = build_dimension_vision_messages(question, n_dims, query_image)
    inputs = prepare_inputs(processor, model, messages, enable_thinking=enable_thinking)
    input_len = int(inputs["input_ids"].shape[-1])
    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
    gen_log_utils = logging.getLogger("transformers.generation.utils")
    gen_log_cfg = logging.getLogger("transformers.generation.configuration_utils")
    prev_u, prev_c = gen_log_utils.level, gen_log_cfg.level
    gen_log_utils.setLevel(logging.ERROR)
    gen_log_cfg.setLevel(logging.ERROR)
    try:
        with torch.inference_mode():
            out = model.generate(**inputs, generation_config=gen_cfg)
    finally:
        gen_log_utils.setLevel(prev_u)
        gen_log_cfg.setLevel(prev_c)
    new_ids = out[0, input_len:]
    text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    if hasattr(processor, "parse_response"):
        try:
            parsed = processor.parse_response(text)
            if isinstance(parsed, dict) and parsed.get("content"):
                text = str(parsed["content"])
        except Exception:
            pass
    return parse_dimension_lines(text, n_dims)
