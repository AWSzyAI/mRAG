"""Use Gemma 4 multimodal (e.g. E2B-it) to produce MagicLens retrieval instruction lines."""

from __future__ import annotations

import logging
import json
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
        f"Question and answer choices about the query image:\n{question}\n\n"
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


def generate_retrieval_plan_with_rationales_gemma4(
    processor,
    model,
    *,
    query_image: str | Path,
    question: str,
    n_dims: int,
    max_new_tokens: int = 768,
    enable_thinking: bool = False,
) -> dict:
    """Generate dimension queries with short rationales as structured JSON.

    Output schema:
    {
      "dimensions": [
        {"query": "...", "rationale": "..."},
        ...
      ]
    }
    """
    img = str(Path(query_image).expanduser().resolve())
    schema_hint = (
        '{\n'
        '  "dimensions": [\n'
        '    {"query": "identity: ...", "rationale": "..."},\n'
        '    {"query": "attribute: ...", "rationale": "..."}\n'
        "  ]\n"
        "}"
    )
    user_text = (
        "You are a retrieval planner for visual QA.\n"
        f"Given the image and question, produce exactly {n_dims} complementary retrieval dimensions.\n"
        "For each dimension, include:\n"
        '- "query": one concise image-search instruction in English (<=30 words)\n'
        '- "rationale": one short reason (<=25 words) explaining why this dimension helps answer the question\n\n'
        "Return ONLY valid JSON. No markdown fences. No extra text.\n"
        f"JSON schema example:\n{schema_hint}\n\n"
        f"Question with choices:\n{question}"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text},
            ],
        }
    ]
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
    text = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    if hasattr(processor, "parse_response"):
        try:
            parsed = processor.parse_response(text)
            if isinstance(parsed, dict) and parsed.get("content"):
                text = str(parsed["content"]).strip()
        except Exception:
            pass
    # Try robust JSON extraction.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    obj = json.loads(text)
    dims = obj.get("dimensions", []) if isinstance(obj, dict) else []
    queries: list[str] = []
    rationales: list[str] = []
    for d in dims:
        if not isinstance(d, dict):
            continue
        q = " ".join(str(d.get("query", "")).split())
        r = " ".join(str(d.get("rationale", "")).split())
        if q:
            queries.append(q)
            rationales.append(r)
        if len(queries) >= n_dims:
            break
    return {"queries": queries, "rationales": rationales, "raw_text": text}


def describe_image_for_question_gemma4(
    processor,
    model,
    *,
    image_path: str | Path,
    question: str,
    image_label: str,
    max_new_tokens: int = 160,
    enable_thinking: bool = False,
) -> str:
    """Describe one image with the downstream multiple-choice question in mind."""
    img = str(Path(image_path).expanduser().resolve())
    user_text = (
        f"You are preparing evidence for a multiple-choice visual question.\n"
        f"Image label: {image_label}\n"
        f"Question and answer choices:\n{question}\n\n"
        "Describe only visual evidence in this image that may help answer the question. "
        "Be concise, factual, and do not choose an option."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text},
            ],
        }
    ]
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
    text = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    if hasattr(processor, "parse_response"):
        try:
            parsed = processor.parse_response(text)
            if isinstance(parsed, dict) and parsed.get("content"):
                text = str(parsed["content"]).strip()
        except Exception:
            pass
    return " ".join(text.split())
