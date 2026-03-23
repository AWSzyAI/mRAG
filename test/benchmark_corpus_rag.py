#!/usr/bin/env python3
import argparse
import copy
import gc
import hashlib
import inspect
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
from tqdm.auto import tqdm
from transformers import AutoProcessor, CLIPModel


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "github/LLaVA-NeXT"))
sys.path.append(str(ROOT_DIR / "github/MRAG-Bench/eval"))
sys.path.append(str(ROOT_DIR / "github/magiclens"))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from inference import load_model as load_magiclens_model  # noqa: E402


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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


def rerank_rag_images(encode_fn, tokenizer_fn, question_text: str, image_files):
    if len(image_files) < 2:
        return image_files, []

    query_image = image_files[0]
    rag_images = image_files[1:]

    rag_img_batch = np.concatenate([preprocess_pil_image(img, 224) for img in rag_images], axis=0)
    rag_tok_batch = np.concatenate([np.array(tokenizer_fn("")) for _ in rag_images], axis=0)
    rag_embeds = np.asarray(encode_fn(jnp.array(rag_tok_batch), jnp.array(rag_img_batch)))

    q_img = preprocess_pil_image(query_image, 224)
    q_tok = np.array(tokenizer_fn(question_text))
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


def log_torch_cuda_env() -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    names = []
    for i in range(count):
        try:
            names.append(torch.cuda.get_device_name(i))
        except Exception:
            names.append("<unknown>")
    log(
        "torch_cuda="
        f"available={available}, count={count}, visible={cuda_visible}, devices={names}"
    )


def load_llava(args):
    model_name = "llava_qwen"
    visible_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    dm_arg = str(args.llava_device_map).strip().lower()
    if dm_arg == "auto" and visible_cuda_count == 1:
        llava_device_map = {"": "cuda:0"}
        log("llava_device_map auto->single (single GPU detected)")
    elif dm_arg in ("single", "cuda", "cuda:0", "0"):
        llava_device_map = {"": "cuda:0"}
    elif re.fullmatch(r"(?:cuda:)?\d+", dm_arg):
        device_idx = dm_arg.split(":")[-1]
        llava_device_map = {"": f"cuda:{device_idx}"}
    else:
        llava_device_map = args.llava_device_map
    if llava_device_map == "auto" and visible_cuda_count >= 2:
        llava_device_map = "balanced"
        log("llava_device_map auto->balanced (multi-GPU detected)")

    llava_args = {
        "multimodal": True,
        "overwrite_config": {"image_aspect_ratio": "pad"},
    }
    if args.llava_attn_implementation:
        llava_args["attn_implementation"] = args.llava_attn_implementation
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.llava_model_path,
        None,
        model_name,
        device_map=llava_device_map,
        load_4bit=bool(args.llava_load_4bit),
        load_8bit=bool(args.llava_load_8bit),
        **llava_args,
    )
    model.eval()
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        counts = {}
        for v in device_map.values():
            key = str(v)
            counts[key] = counts.get(key, 0) + 1
        summary = ", ".join([f"{k}:{counts[k]}" for k in sorted(counts.keys())])
        log(f"llava_hf_device_map={summary}")
        has_cpu_offload = any(str(v) == "cpu" for v in device_map.values())
        if has_cpu_offload and not args.llava_allow_cpu_offload:
            raise RuntimeError(
                "LLaVA CPU offload detected and llava_allow_cpu_offload is disabled. "
                "Try one of: CUDA_VISIBLE_DEVICES=0,1 or LLAVA_DEVICE_MAP=single or LLAVA_LOAD_4BIT=1."
            )
    return tokenizer, model, image_processor


def llava_answer(tokenizer, model, image_processor, item, image_files, args):
    question_part = item.get("prompt_question_part", item["question"])
    if len(image_files) <= 1:
        instruction = "Answer with the option's letter from the given choices directly. "
    else:
        instruction = (
            "You will be given one question concerning several images. "
            "The first image is the input image, others are retrieved examples to help you. "
            "Answer with the option's letter from the given choices directly. "
        )
    image_tokens = DEFAULT_IMAGE_TOKEN * max(1, len(image_files))
    user_query = f"{instruction}{image_tokens}\n{question_part}"

    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], user_query)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    model_device = next(model.parameters()).device
    image_dtype = torch.float16 if model_device.type == "cuda" else torch.float32
    input_ids = (
        tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model_device)
    )
    image_tensors = process_images(image_files, image_processor, model.config)
    image_tensors = [img.to(dtype=image_dtype, device=model_device) for img in image_tensors]
    image_sizes = [img.size for img in image_files]

    with torch.inference_mode():
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0.0,
            num_beams=max(1, int(args.llava_num_beams)),
            max_new_tokens=max(1, int(args.llava_max_new_tokens)),
        )
    return tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()


def load_reference_predictions(path: str):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"reference jsonl not found: {path}")
    ref = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ref[str(row.get("qs_id"))] = extract_choice(str(row.get("output", "")))
    return ref


def corpus_signature(corpus_dir: Path, retriever_type: str, retriever_name: str) -> str:
    payload = f"{corpus_dir.resolve()}::{retriever_type}::{retriever_name}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def list_corpus_images(corpus_dir: Path):
    paths = [p for p in sorted(corpus_dir.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    if not paths:
        raise FileNotFoundError(f"no images found under corpus dir: {corpus_dir}")
    return paths


def load_clip_encoder(model_name: str, device: str):
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return processor, model


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return x / denom


def encode_clip_images(paths, processor, model, device: str, batch_size: int):
    embeds = []
    for start in tqdm(range(0, len(paths), batch_size), desc="CLIP corpus encode"):
        batch_paths = paths[start : start + batch_size]
        images = []
        for p in batch_paths:
            with Image.open(p) as img:
                images.append(img.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            feats = model.get_image_features(**inputs)
        feats = feats.detach().float().cpu().numpy()
        embeds.append(l2_normalize(feats).astype(np.float32))
    return np.concatenate(embeds, axis=0)


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


def encode_magiclens_images(paths, encode_fn, tokenizer_fn, batch_size: int):
    embeds = []
    empty_tok = np.array(tokenizer_fn(""))
    for start in tqdm(range(0, len(paths), batch_size), desc="MagicLens corpus encode"):
        batch_paths = paths[start : start + batch_size]
        batch_imgs = np.concatenate([preprocess_pil_image(Image.open(p).convert("RGB"), 224) for p in batch_paths], axis=0)
        batch_toks = np.concatenate([empty_tok for _ in batch_paths], axis=0)
        feats = np.asarray(encode_fn(jnp.array(batch_toks), jnp.array(batch_imgs)))
        embeds.append(l2_normalize(feats).astype(np.float32))
    return np.concatenate(embeds, axis=0)


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


def encode_clip_query_image(image, processor, model, device: str) -> np.ndarray:
    inputs = processor(images=[image.convert("RGB")], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        feats = model.get_image_features(**inputs)
    feats = feats.detach().float().cpu().numpy()
    return l2_normalize(feats)[0]


def retrieve_corpus_images(query_image, corpus_paths, corpus_embeds, processor, model, device: str, top_k: int):
    query_embed = encode_clip_query_image(query_image, processor, model, device)
    sims = corpus_embeds @ query_embed
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
    q_tok = np.array(tokenizer_fn(question_text))
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


def iter_bench_queries(dataset_name: str):
    ds = load_dataset(dataset_name, split="test")
    for item in ds:
        qs = item["question"]
        prompt_question_part = (
            f"{qs}\n Choices:\n"
            f"A: {item['A']}\n"
            f"B: {item['B']}\n"
            f"C: {item['C']}\n"
            f"D: {item['D']}"
        )
        yield {
            "id": item["id"],
            "question": prompt_question_part,
            "prompt_question_part": prompt_question_part,
            "prompt": prompt_question_part,
            "answer": item["answer"],
            "gt_choice": item["answer_choice"],
            "scenario": item["scenario"],
            "aspect": item["aspect"],
            "query_image": item["image"].convert("RGB"),
        }


def maybe_load_magiclens(args):
    needs_magiclens = (not args.disable_magiclens_rerank) or args.retriever_type == "magiclens"
    if not needs_magiclens:
        return None, None, None
    bpe_path = resolve_bpe_path(args.bpe_path)
    if bpe_path:
        tokenizer_fn = clip_tokenizer.build_tokenizer(bpe_path=bpe_path)
        log(f"using_bpe_path={bpe_path}")
    else:
        tokenizer_fn = clip_tokenizer.build_tokenizer()
        log("using_bpe_path=<default>")
    log("loading MagicLens model")
    ml_model, ml_params = load_magiclens_model(args.magiclens_model_size, args.magiclens_model_path)
    disable_jit = bool(args.magiclens_disable_jit)
    if jax.default_backend() == "cpu" and not disable_jit:
        disable_jit = True
        log("JAX backend is CPU; auto-disabling MagicLens JIT for stability.")
    encode_fn = build_magiclens_encoder(ml_model, ml_params, disable_jit=disable_jit)
    log("MagicLens model ready")
    return tokenizer_fn, encode_fn, disable_jit


def main():
    parser = argparse.ArgumentParser(description="True corpus-based MRAG benchmark for E3/E6/E7.")
    parser.add_argument("--dataset-name", type=str, default="uclanlp/MRAG-Bench")
    parser.add_argument("--corpus-dir", type=str, required=True)
    parser.add_argument("--corpus-cache-dir", type=str, default=str(ROOT_DIR / "results/corpus_index"))
    parser.add_argument("--retriever-type", type=str, default="clip", choices=["clip", "magiclens"])
    parser.add_argument("--clip-model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--answers-file", type=str, default=str(ROOT_DIR / "github/MRAG-Bench/results/corpus_rag_results.jsonl"))
    parser.add_argument("--summary-out", type=str, default=str(ROOT_DIR / "log/corpus_rag_summary.json"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--disable-magiclens-rerank", action="store_true")
    parser.add_argument("--magiclens-model-path", type=str, default=str(ROOT_DIR / "models/magic_lens_clip_base.pkl"))
    parser.add_argument("--magiclens-model-size", type=str, default="base", choices=["base", "large"])
    parser.add_argument("--magiclens-batch-size", type=int, default=16)
    parser.add_argument("--bpe-path", type=str, default="")
    parser.add_argument("--magiclens-disable-jit", action="store_true")
    parser.add_argument("--magiclens-clear-cache-every", type=int, default=0)
    parser.add_argument("--llava-model-path", type=str, default=str(ROOT_DIR / "models/llava-onevision-qwen2-7b-ov"))
    parser.add_argument("--llava-device-map", type=str, default="auto")
    parser.add_argument("--llava-attn-implementation", type=str, default="sdpa")
    parser.add_argument("--llava-load-4bit", action="store_true")
    parser.add_argument("--llava-load-8bit", action="store_true")
    parser.add_argument("--llava-allow-cpu-offload", action="store_true")
    parser.add_argument("--llava-max-new-tokens", type=int, default=4096)
    parser.add_argument("--llava-num-beams", type=int, default=1)
    parser.add_argument("--llava-greedy-jsonl", type=str, default="")
    args = parser.parse_args()

    answers_path = Path(args.answers_file)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    log_torch_cuda_env()
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"clip_device={clip_device}")
    llava_tokenizer, llava_model, llava_image_processor = load_llava(args)
    tokenizer_fn, magiclens_encode_fn, _ = maybe_load_magiclens(args)
    clip_processor = None
    clip_model = None
    if args.retriever_type == "clip":
        clip_processor, clip_model = load_clip_encoder(args.clip_model_name, clip_device)
        corpus_paths, corpus_embeds = load_or_build_clip_corpus_index(args, clip_processor, clip_model, clip_device)
    else:
        if magiclens_encode_fn is None or tokenizer_fn is None:
            raise RuntimeError("MagicLens retriever selected, but MagicLens model failed to load.")
        corpus_paths, corpus_embeds = load_or_build_magiclens_corpus_index(args, magiclens_encode_fn, tokenizer_fn)
    ref_preds = load_reference_predictions(args.llava_greedy_jsonl)
    if ref_preds:
        log(f"loaded_reference_predictions={len(ref_preds)}")

    try:
        total = len(load_dataset(args.dataset_name, split="test"))
    except Exception:
        total = None
    if total is not None:
        log(f"dataset_total={total}")

    processed = 0
    correct = 0
    by_scenario = {}
    overlap = 0
    agree = 0
    ref_correct = 0
    run_correct_on_overlap = 0

    with open(answers_path, "w", encoding="utf-8", buffering=1) as out:
        for idx, item in enumerate(tqdm(iter_bench_queries(args.dataset_name), total=total, desc="Corpus-RAG + LLaVA")):
            if idx < args.start_index:
                continue
            if args.max_samples > 0 and processed >= args.max_samples:
                break

            question_text, _ = parse_question_and_options(item["prompt_question_part"])
            if args.retriever_type == "clip":
                retrieved = retrieve_corpus_images(
                    item["query_image"],
                    corpus_paths,
                    corpus_embeds,
                    clip_processor,
                    clip_model,
                    clip_device,
                    args.top_k,
                )
            else:
                retrieved = retrieve_corpus_images_magiclens(
                    item["query_image"],
                    question_text,
                    corpus_paths,
                    corpus_embeds,
                    magiclens_encode_fn,
                    tokenizer_fn,
                    args.top_k,
                )
            image_files = [item["query_image"]] + [row["image"] for row in retrieved]

            rerank_info = []
            if not args.disable_magiclens_rerank:
                image_files, rerank_info = rerank_rag_images(
                    magiclens_encode_fn, tokenizer_fn, question_text, image_files
                )

            raw_output = llava_answer(
                llava_tokenizer,
                llava_model,
                llava_image_processor,
                item,
                image_files,
                args,
            )
            pred_choice = extract_choice(raw_output)

            qs_id = str(item["id"])
            gt_choice = str(item["gt_choice"])
            scenario = str(item["scenario"])
            row = {
                "qs_id": qs_id,
                "prompt": item["prompt"],
                "output": raw_output,
                "gt_answer": item["answer"],
                "shortuuid": uuid.uuid4().hex,
                "model_id": f"llava_qwen7b_{args.retriever_type}_corpus_rag",
                "gt_choice": gt_choice,
                "scenario": scenario,
                "aspect": item["aspect"],
                "meta_pred_choice": pred_choice,
                "meta_rag_count": max(0, len(image_files) - 1),
                "meta_rag_source": f"corpus_{args.retriever_type}_topk",
                "meta_corpus_dir": str(Path(args.corpus_dir).expanduser().resolve()),
                "meta_retriever_type": args.retriever_type,
                "meta_clip_model_name": args.clip_model_name if args.retriever_type == "clip" else None,
                "meta_magiclens_rerank_disabled": bool(args.disable_magiclens_rerank),
                "meta_magiclens_rag_ranks": rerank_info,
                "meta_corpus_retrieval": [
                    {"rank": r["rank"], "path": r["path"], "score": r["score"]} for r in retrieved
                ],
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

            processed += 1
            is_correct = pred_choice == gt_choice
            if is_correct:
                correct += 1
            stat = by_scenario.setdefault(scenario, {"total": 0, "correct": 0})
            stat["total"] += 1
            if is_correct:
                stat["correct"] += 1

            ref_choice = ref_preds.get(qs_id)
            if ref_choice is not None:
                overlap += 1
                if is_correct:
                    run_correct_on_overlap += 1
                if ref_choice == gt_choice:
                    ref_correct += 1
                if ref_choice == pred_choice:
                    agree += 1

            if (
                magiclens_encode_fn is not None
                and args.magiclens_clear_cache_every > 0
                and processed % int(args.magiclens_clear_cache_every) == 0
            ):
                try:
                    jax.clear_caches()
                except Exception:
                    pass
                gc.collect()
                log(f"cleared_jax_caches_at={processed}")

    by_scenario_acc = {}
    for scenario, stat in by_scenario.items():
        total_s = max(1, stat["total"])
        by_scenario_acc[scenario] = round(100.0 * stat["correct"] / total_s, 2)

    summary = {
        "dataset_name": args.dataset_name,
        "processed": processed,
        "correct": correct,
        "accuracy": round(100.0 * correct / max(1, processed), 2),
        "top_k": int(args.top_k),
        "retriever_type": args.retriever_type,
        "clip_model_name": args.clip_model_name if args.retriever_type == "clip" else None,
        "magiclens_model_size": args.magiclens_model_size if args.retriever_type == "magiclens" else None,
        "corpus_dir": str(Path(args.corpus_dir).expanduser().resolve()),
        "magiclens_rerank_enabled": not bool(args.disable_magiclens_rerank),
        "by_scenario_accuracy": by_scenario_acc,
        "reference_overlap": overlap,
        "reference_agreement": round(100.0 * agree / max(1, overlap), 2) if overlap else None,
        "reference_accuracy": round(100.0 * ref_correct / max(1, overlap), 2) if overlap else None,
        "run_accuracy_on_overlap": round(100.0 * run_correct_on_overlap / max(1, overlap), 2) if overlap else None,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"summary_saved={summary_path}")
    log(f"accuracy={summary['accuracy']} processed={processed}")


if __name__ == "__main__":
    main()
