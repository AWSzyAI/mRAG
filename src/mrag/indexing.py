import hashlib
import json
from pathlib import Path

import numpy as np

from .clip_retriever import encode_clip_images, list_corpus_images
from .magiclens import encode_magiclens_images
from .runtime import log


def corpus_signature(corpus_dir: Path, retriever_type: str, retriever_name: str) -> str:
    payload = f"{corpus_dir.resolve()}::{retriever_type}::{retriever_name}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def load_or_build_clip_corpus_index(args, clip_processor, clip_model, device: str):
    corpus_dir = Path(args.corpus_dir).expanduser().resolve()
    image_paths = list_corpus_images(corpus_dir)
    cache_dir = Path(args.corpus_cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = corpus_signature(corpus_dir, "clip", args.clip_model_name)
    npy_path = cache_dir / f"{sig}_embeddings.npy"
    json_path = cache_dir / f"{sig}_paths.json"

    if npy_path.exists() and json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cached_paths = meta.get("paths", [])
        if len(cached_paths) == len(image_paths) and cached_paths:
            log(f"loading_cached_corpus_index={npy_path}")
            embeddings = np.load(npy_path)
            return image_paths, embeddings
        log("cached_corpus_index_mismatch=rebuild")

    log(f"building_corpus_index images={len(image_paths)} corpus_dir={corpus_dir}")
    embeddings = encode_clip_images(image_paths, clip_processor, clip_model, device, args.clip_batch_size)
    np.save(npy_path, embeddings)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"corpus_dir": str(corpus_dir), "paths": [str(p) for p in image_paths]}, f, ensure_ascii=False)
    log(f"saved_corpus_index={npy_path}")
    return image_paths, embeddings


def load_or_build_magiclens_corpus_index(args, encode_fn, tokenizer_fn):
    corpus_dir = Path(args.corpus_dir).expanduser().resolve()
    image_paths = list_corpus_images(corpus_dir)
    cache_dir = Path(args.corpus_cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    retriever_name = f"magiclens_{args.magiclens_model_size}"
    sig = corpus_signature(corpus_dir, "magiclens", retriever_name)
    npy_path = cache_dir / f"{sig}_embeddings.npy"
    json_path = cache_dir / f"{sig}_paths.json"

    if npy_path.exists() and json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cached_paths = meta.get("paths", [])
        if len(cached_paths) == len(image_paths) and cached_paths:
            log(f"loading_cached_corpus_index={npy_path}")
            embeddings = np.load(npy_path)
            return image_paths, embeddings
        log("cached_corpus_index_mismatch=rebuild")

    log(f"building_corpus_index images={len(image_paths)} corpus_dir={corpus_dir}")
    embeddings = encode_magiclens_images(image_paths, encode_fn, tokenizer_fn, args.magiclens_batch_size)
    np.save(npy_path, embeddings)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"corpus_dir": str(corpus_dir), "paths": [str(p) for p in image_paths]}, f, ensure_ascii=False)
    log(f"saved_corpus_index={npy_path}")
    return image_paths, embeddings
