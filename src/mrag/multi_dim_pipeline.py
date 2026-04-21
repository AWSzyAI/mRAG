"""Orchestrate multi-instruction MagicLens retrieval, fusion, and (optional) VLM answer prep."""

from __future__ import annotations

from typing import Callable

from .fusion import fuse_reciprocal_rank, fuse_score_sum, fuse_voting
from .magiclens import retrieve_corpus_paths_ranked

FusionFn = Callable[[list[list[dict]], int], list[dict]]

FUSION_STRATEGIES: dict[str, FusionFn] = {
    "rrf": lambda results, k: fuse_reciprocal_rank(results, final_k=k),
    "score_sum": lambda results, k: fuse_score_sum(results, final_k=k),
    "voting": lambda results, k: fuse_voting(results, final_k=k),
}


def get_fusion_fn(name: str) -> FusionFn:
    if name not in FUSION_STRATEGIES:
        raise ValueError(f"unknown fusion strategy {name!r}; choose from {sorted(FUSION_STRATEGIES)}")
    return FUSION_STRATEGIES[name]


def retrieve_per_instruction_magiclens(
    query_image,
    instructions: list[str],
    corpus_paths,
    corpus_embeds,
    encode_fn,
    tokenizer_fn,
    dim_top_k: int,
) -> list[list[dict]]:
    """Run MagicLens corpus retrieval once per instruction string."""
    out: list[list[dict]] = []
    for instr in instructions:
        rows = retrieve_corpus_paths_ranked(
            query_image,
            instr,
            corpus_paths,
            corpus_embeds,
            encode_fn,
            tokenizer_fn,
            dim_top_k,
        )
        out.append(rows)
    return out


def fuse_retrieval_lists(
    per_dim_results: list[list[dict]],
    fusion_strategy: str,
    final_top_k: int,
) -> list[dict]:
    fn = get_fusion_fn(fusion_strategy)
    return fn(per_dim_results, final_top_k)


def multi_dim_magiclens_retrieve_and_fuse(
    query_image,
    instructions: list[str],
    corpus_paths,
    corpus_embeds,
    encode_fn,
    tokenizer_fn,
    *,
    dim_top_k: int,
    fusion_strategy: str,
    final_top_k: int,
) -> tuple[list[list[dict]], list[dict]]:
    """Full retrieval + fusion in one call."""
    per_dim = retrieve_per_instruction_magiclens(
        query_image,
        instructions,
        corpus_paths,
        corpus_embeds,
        encode_fn,
        tokenizer_fn,
        dim_top_k,
    )
    fused = fuse_retrieval_lists(per_dim, fusion_strategy, final_top_k)
    return per_dim, fused


def fused_paths_for_vlm(fused_rows: list[dict]) -> list[str]:
    return [str(r["path"]) for r in fused_rows]
