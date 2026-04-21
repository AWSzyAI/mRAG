#!/usr/bin/env python3
"""
Backward-compatible entrypoint for the E8 multi-dimension RAG benchmark.

Implementation lives in ``test/pipeline_multi_dim_rag.py`` with reusable modules
under ``src/mrag/`` (query_planner, magiclens, multi_dim_pipeline, fusion, indexing).
"""
from pipeline_multi_dim_rag import main

if __name__ == "__main__":
    main()
