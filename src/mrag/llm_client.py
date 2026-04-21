"""OpenAI-compatible Chat Completions client (SiliconFlow, Together, OpenRouter, etc.)."""

from __future__ import annotations

import time
from typing import Any


def chat_completions_text(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    max_tokens: int = 512,
    timeout_sec: float = 120.0,
    max_retries: int = 3,
) -> str:
    """POST /v1/chat/completions; return assistant message content (may be empty)."""
    import httpx

    base = api_base.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout_sec) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            return str(data["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"chat_completions failed after {max_retries} attempts: {last_err}")
