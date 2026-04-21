import os
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def parse_question_and_options(question_blob: str):
    marker = "\n Choices:\n"
    if marker not in question_blob:
        return question_blob, {}

    question_text, choices_blob = question_blob.split(marker, 1)
    options = {}
    for line in choices_blob.splitlines():
        m = re.match(r"^([A-D]):\s*(.*)$", line.strip())
        if m:
            options[m.group(1)] = m.group(2)
    return question_text, options


def extract_choice(text: str) -> str:
    text_up = str(text).upper()
    candidates = []

    for choice in ("A", "B", "C", "D"):
        for m in re.finditer(rf"\({choice}\)", text_up):
            candidates.append((m.start(), choice))

    if not candidates:
        for choice in ("A", "B", "C", "D"):
            for m in re.finditer(rf"\b{choice}\b", text_up):
                candidates.append((m.start(), choice))

    if not candidates:
        return "N/A"
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def resolve_bpe_path(explicit_path: str) -> str:
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    env_path = os.environ.get("MAGICLENS_BPE_PATH", "")
    if env_path and os.path.exists(env_path):
        return env_path
    candidates = [
        ROOT_DIR / "models/bpe_simple_vocab_16e6.txt.gz",
        ROOT_DIR
        / "github/LLaVA-NeXT/llava/model/multimodal_encoder/dev_eva_clip/eva_clip/bpe_simple_vocab_16e6.txt.gz",
        Path.home() / ".cache/scenic/clip/bpe_simple_vocab_16e6.txt.gz",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return ""
