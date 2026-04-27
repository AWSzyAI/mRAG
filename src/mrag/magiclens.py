from pathlib import Path
import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .clip_retriever import l2_normalize

_TOKENIZER_CONTEXT_RE = re.compile(r"too long for context length", re.IGNORECASE)


def _head_tail_words(words: list[str], keep: int) -> str:
    if keep >= len(words):
        return " ".join(words)
    if keep <= 0:
        return ""
    head = (keep + 1) // 2
    tail = keep - head
    return " ".join(words[:head] + words[-tail:]) if tail else " ".join(words[:head])


def preprocess_pil_image(image, size: int = 224) -> np.ndarray:
    arr = jnp.array(image.convert("RGB"))[jnp.newaxis, ...]
    arr = arr / (arr.max() + 1e-12)
    arr = jax.image.resize(arr, (1, size, size, 3), method="bilinear")
    return np.array(arr)


def build_magiclens_encoder(model, params, disable_jit: bool = False):
    def encode(ids, image):
        out = model.apply(params, {"ids": ids, "image": image})
        return out["multimodal_embed_norm"]

    if disable_jit:
        return encode
    return jax.jit(encode)


def _safe_tokenize_clip_text(tokenizer_fn, text: str) -> np.ndarray:
    """Tokenize CLIP text, truncating overlong prompts to a valid head/tail excerpt."""
    try:
        return np.array(tokenizer_fn(text))
    except RuntimeError as exc:
        if not _TOKENIZER_CONTEXT_RE.search(str(exc)):
            raise

    words = str(text).split()
    if not words:
        return np.array(tokenizer_fn(""))

    lo, hi = 0, len(words)
    best_tok = None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = _head_tail_words(words, mid)
        try:
            tok = np.array(tokenizer_fn(candidate))
        except RuntimeError as exc:
            if not _TOKENIZER_CONTEXT_RE.search(str(exc)):
                raise
            hi = mid - 1
            continue
        best_tok = tok
        lo = mid + 1

    if best_tok is not None:
        warnings.warn(
            "MagicLens CLIP query exceeded tokenizer context; truncated to fit 77-token limit.",
            RuntimeWarning,
            stacklevel=2,
        )
        return best_tok

    # Rare path: a single whitespace token can still exceed the tokenizer limit.
    chars = str(text)
    lo, hi = 0, len(chars)
    best_tok = np.array(tokenizer_fn(""))
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = chars[:mid]
        try:
            tok = np.array(tokenizer_fn(candidate))
        except RuntimeError as exc:
            if not _TOKENIZER_CONTEXT_RE.search(str(exc)):
                raise
            hi = mid - 1
            continue
        best_tok = tok
        lo = mid + 1
    warnings.warn(
        "MagicLens CLIP query exceeded tokenizer context; truncated to fit 77-token limit.",
        RuntimeWarning,
        stacklevel=2,
    )
    return best_tok


def rerank_rag_images(encode_fn, tokenizer_fn, question_text: str, image_files):
    if len(image_files) < 2:
        return image_files, []

    query_image = image_files[0]
    rag_images = image_files[1:]

    rag_img_batch = np.concatenate([preprocess_pil_image(img, 224) for img in rag_images], axis=0)
    rag_tok_batch = np.concatenate([np.array(tokenizer_fn("")) for _ in rag_images], axis=0)
    rag_embeds = np.asarray(encode_fn(jnp.array(rag_tok_batch), jnp.array(rag_img_batch)))

    q_img = preprocess_pil_image(query_image, 224)
    q_tok = _safe_tokenize_clip_text(tokenizer_fn, question_text)
    q_embed = np.asarray(encode_fn(jnp.array(q_tok), jnp.array(q_img)))[0]

    sims = np.matmul(rag_embeds, q_embed)
    order = np.argsort(-sims)
    reranked = [query_image] + [rag_images[int(i)] for i in order.tolist()]
    rank_info = [
        {
            "new_rank": int(rank + 1),
            "orig_rag_index": int(i + 1),
            "score": float(sims[int(i)]),
        }
        for rank, i in enumerate(order.tolist())
    ]
    return reranked, rank_info


def encode_magiclens_images(paths, encode_fn, tokenizer_fn, batch_size: int):
    embeds = []
    empty_tok = np.array(tokenizer_fn(""))
    for start in tqdm(range(0, len(paths), batch_size), desc="MagicLens corpus encode"):
        batch_paths = paths[start : start + batch_size]
        batch_imgs = np.concatenate(
            [preprocess_pil_image(Image.open(p).convert("RGB"), 224) for p in batch_paths],
            axis=0,
        )
        batch_toks = np.concatenate([empty_tok for _ in batch_paths], axis=0)
        feats = np.asarray(encode_fn(jnp.array(batch_toks), jnp.array(batch_imgs)))
        embeds.append(l2_normalize(feats).astype(np.float32))
    return np.concatenate(embeds, axis=0)


def retrieve_corpus_paths_ranked(
    query_image,
    instruction: str,
    corpus_paths,
    corpus_embeds,
    encode_fn,
    tokenizer_fn,
    top_k: int,
) -> list[dict]:
    """MagicLens corpus retrieval for one text instruction + query image.

    Returns fusion-friendly rows: ``rank`` (1-based), ``path``, ``score`` (cosine sim).
    Does not load candidate images into memory.
    """
    q_img = preprocess_pil_image(query_image, 224)
    q_tok = _safe_tokenize_clip_text(tokenizer_fn, instruction)
    q_embed = np.asarray(encode_fn(jnp.array(q_tok), jnp.array(q_img)))[0]
    q_embed = l2_normalize(q_embed[None, :])[0]
    sims = corpus_embeds @ q_embed
    k = min(max(1, int(top_k)), len(corpus_paths))
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    ranked: list[dict] = []
    for rank, idx in enumerate(top_idx.tolist(), start=1):
        path = corpus_paths[idx]
        ranked.append({"rank": rank, "path": str(path), "score": float(sims[idx])})
    return ranked


def retrieve_corpus_images_magiclens(
    query_image,
    question_text: str,
    corpus_paths,
    corpus_embeds,
    encode_fn,
    tokenizer_fn,
    top_k: int,
):
    q_img = preprocess_pil_image(query_image, 224)
    q_tok = _safe_tokenize_clip_text(tokenizer_fn, question_text)
    q_embed = np.asarray(encode_fn(jnp.array(q_tok), jnp.array(q_img)))[0]
    q_embed = l2_normalize(q_embed[None, :])[0]
    sims = corpus_embeds @ q_embed
    k = min(max(1, int(top_k)), len(corpus_paths))
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    ranked = []
    for rank, idx in enumerate(top_idx.tolist(), start=1):
        path = corpus_paths[idx]
        with Image.open(path) as img:
            ranked.append(
                {
                    "rank": rank,
                    "path": str(path),
                    "score": float(sims[idx]),
                    "image": img.convert("RGB"),
                }
            )
    return ranked
