"""Core modules for the mRAG project.

Experiment drivers under ``test/`` should import from here. Main pieces:

- ``mrag_bench`` — MRAG-Bench ``load_dataset`` iteration + ``ensure_mrag_hf_cache_env`` (``github/MRAG-Bench/.cache/huggingface-mrag`` / ``MRAG_HF_HOME``)
- ``query_planner`` / ``llm_client`` — LLM-based multi-dimension retrieval instructions
- ``gemma4_loader`` / ``gemma4_dims`` — Gemma 4 multimodal local dimension lines for MagicLens
- ``envfile`` — load ``.env`` without extra dependencies
- ``transformers_llava_compat`` — patch ``modeling_utils`` for LLaVA on newer ``transformers``
- ``magiclens`` — MagicLens encoding and corpus retrieval
- ``multi_dim_pipeline`` — per-dimension retrieval + fusion orchestration
- ``fusion`` — RRF / score-sum / voting over ranked path lists
- ``indexing`` — cached corpus embedding index (CLIP / MagicLens)
"""
