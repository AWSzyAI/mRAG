"""LLM-based multi-dimension retrieval instruction generation for mRAG."""

from __future__ import annotations

import re
from typing import Literal

DIMENSION_SYSTEM_PROMPT = """\
You are a retrieval query planner for a visual question answering system.
Given a question about an image, decompose the retrieval need into {n_dims} \
complementary search dimensions. Each dimension should target a different \
aspect of evidence that helps answer the question.

Output EXACTLY {n_dims} lines, one per dimension. Each line is a short, \
concrete image-search instruction (in English, <=30 words) that could be \
passed to an instruction-based image retrieval model.

Dimension guidelines:
- identity: preserve the specific object/species/brand identity in the query image
- attribute: seek images showing the attribute or property the question asks about
- complementary-view: find images from a different viewpoint or life stage
- context: find images showing the broader scene or habitat
- comparison: find images of visually similar but distinct alternatives

Do NOT number the lines. Do NOT add any explanation. Output ONLY the {n_dims} \
instruction lines."""

DIMENSION_USER_TEMPLATE = """\
Question: {question}
Image description (if available): {description}

Generate {n_dims} retrieval instructions."""


def build_dimension_prompt(question: str, n_dims: int, description: str = "") -> list[dict]:
    system_msg = DIMENSION_SYSTEM_PROMPT.format(n_dims=n_dims)
    user_msg = DIMENSION_USER_TEMPLATE.format(
        question=question, n_dims=n_dims, description=description or "(not provided)"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def parse_dimension_lines(text: str, n_dims: int) -> list[str]:
    """Split model output into non-empty lines; strip bullets / numbering."""
    lines: list[str] = []
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        ln = re.sub(r"^\d+[\.\)\-]\s*", "", ln)
        ln = ln.lstrip("-*• ").strip()
        if ln:
            lines.append(ln)
    return lines[:n_dims]


def call_api_generate_dimensions(
    messages: list[dict],
    api_base: str,
    api_key: str,
    model_name: str,
    n_dims: int,
    temperature: float = 0.3,
    max_retries: int = 3,
) -> list[str]:
    from .llm_client import chat_completions_text

    text = chat_completions_text(
        api_base,
        api_key,
        model_name,
        messages,
        temperature=temperature,
        max_tokens=512,
        max_retries=max_retries,
    )
    return parse_dimension_lines(text, n_dims)


def _messages_to_text_prompt(messages: list[dict]) -> str:
    blocks = []
    for m in messages:
        role = str(m.get("role", "user")).upper()
        blocks.append(f"[{role}]\n{m.get('content', '')}")
    blocks.append("[ASSISTANT]")
    return "\n\n".join(blocks)


def load_local_pipeline(model_name: str, torch_dtype=None, device_map: str | dict = "auto"):
    from transformers import pipeline

    kwargs: dict = {"trust_remote_code": True}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        kwargs["device_map"] = device_map
    return pipeline("text-generation", model=model_name, **kwargs)


def call_local_generate_dimensions(messages: list[dict], local_pipeline, n_dims: int) -> list[str]:
    prompt = _messages_to_text_prompt(messages)
    outputs = local_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.3)
    if not outputs:
        return []
    full = outputs[0].get("generated_text", "")
    if isinstance(full, str) and full.startswith(prompt):
        text = full[len(prompt) :]
    else:
        text = str(full)
    return parse_dimension_lines(text, n_dims)


def generate_retrieval_instructions(
    question: str,
    n_dims: int,
    *,
    backend: Literal["api", "local"] = "api",
    image_description: str = "",
    api_base: str = "",
    api_key: str = "",
    api_model: str = "",
    temperature: float = 0.3,
    local_pipeline=None,
) -> list[str]:
    """High-level: build chat messages and return up to ``n_dims`` instruction lines."""
    messages = build_dimension_prompt(question, n_dims, description=image_description)
    if backend == "api":
        if not api_key or not api_base:
            raise ValueError("backend=api requires api_base and api_key")
        if not api_model:
            raise ValueError("backend=api requires api_model")
        return call_api_generate_dimensions(
            messages, api_base, api_key, api_model, n_dims, temperature=temperature
        )
    if backend == "local":
        if local_pipeline is None:
            raise ValueError("backend=local requires local_pipeline")
        return call_local_generate_dimensions(messages, local_pipeline, n_dims)
    raise ValueError(f"unknown backend: {backend}")
