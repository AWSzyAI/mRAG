from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, CLIPModel


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return x / denom


def load_clip_encoder(model_name: str, device: str):
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return processor, model


def list_corpus_images(corpus_dir: Path):
    paths = [p for p in sorted(corpus_dir.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    if not paths:
        raise FileNotFoundError(f"no images found under corpus dir: {corpus_dir}")
    return paths


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
