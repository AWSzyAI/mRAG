#!/usr/bin/env python3
import argparse
import io
import json
import os
import shutil
from pathlib import Path

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_NAME = "uclanlp/MRAG-Bench"

PIPELINES = {
    "E4": {
        "result_jsonl": ROOT_DIR / "log/E4/e4_llava_no_rag_results.jsonl",
        "summary_json": ROOT_DIR / "log/E4/e4_llava_no_rag_results_score.json",
        "log_file": ROOT_DIR / "log/E4/E4.log",
        "mode": "no_rag",
        "title": "E4: LLaVA + No-RAG",
    },
    "E2": {
        "result_jsonl": ROOT_DIR / "log/E2-magiclens不用GT真RAG/magiclens_rerank_llava_retrieved_rag_results.jsonl",
        "summary_json": ROOT_DIR / "log/E2-magiclens不用GT真RAG/magiclens_rerank_llava_retrieved_rag_summary.json",
        "log_file": ROOT_DIR / "log/E2-magiclens不用GT真RAG/benchmark_magiclens_real_rag.log",
        "mode": "dataset_retrieved_rerank",
        "title": "E2: Retrieved-RAG + MagicLens rerank",
    },
    "E3": {
        "result_jsonl": ROOT_DIR / "log/E3/e3_clip_corpus_rag_results.jsonl",
        "summary_json": ROOT_DIR / "log/E3/e3_clip_corpus_rag_summary.json",
        "log_file": ROOT_DIR / "log/E3/E3.log",
        "mode": "corpus_clip",
        "title": "E3: CLIP direct corpus retrieval",
    },
    "E7": {
        "result_jsonl": ROOT_DIR / "log/E7/e7_magiclens_corpus_rag_results.jsonl",
        "summary_json": ROOT_DIR / "log/E7/e7_magiclens_corpus_rag_summary.json",
        "log_file": ROOT_DIR / "log/E7/E7.log",
        "mode": "corpus_magiclens",
        "title": "E7: MagicLens direct corpus retrieval",
    },
}


def pil_from_hf_image(obj):
    if isinstance(obj, Image.Image):
        return obj.convert("RGB")
    if isinstance(obj, dict) and "bytes" in obj:
        return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")
    raise TypeError(f"unsupported image object: {type(obj)}")


def load_results(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_sample(sample_index: int = None, sample_id: str = None, cache_dir: str = None):
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split="test", cache_dir=cache_dir)
    if sample_id is not None:
        for idx, item in enumerate(ds):
            if str(item["id"]) == str(sample_id):
                return idx, item
        raise ValueError(f"sample_id not found: {sample_id}")
    if sample_index is None or sample_index < 0 or sample_index >= len(ds):
        raise ValueError(f"invalid sample_index: {sample_index}")
    return sample_index, ds[sample_index]


def parse_question_and_options(prompt: str):
    marker = "\n Choices:\n"
    if marker not in prompt:
        return prompt, {}
    question_text, choices_blob = prompt.split(marker, 1)
    options = {}
    for line in choices_blob.splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            options[k.strip()] = v.strip()
    return question_text, options


def image_token_block(num_images: int):
    return "<image>" * max(1, num_images)


def build_llava_instruction(num_images: int, use_rag: bool):
    if not use_rag or num_images <= 1:
        return f"Answer with the option's letter from the given choices directly. {image_token_block(1)}"
    return (
        "You will be given one question concerning several images. "
        "The first image is the input image, others are retrieved examples to help you. "
        f"Answer with the option's letter from the given choices directly. {image_token_block(num_images)}"
    )


def save_pil_image(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def copy_path_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def build_bundle(pipeline: str, row: dict, sample_item: dict, sample_index: int, out_dir: Path):
    cfg = PIPELINES[pipeline]
    mode = cfg["mode"]

    query_img = pil_from_hf_image(sample_item["image"])
    gt_images = [pil_from_hf_image(x) for x in sample_item["gt_images"]]
    retrieved_images = [pil_from_hf_image(x) for x in sample_item.get("retrieved_images", [])]
    scenario = sample_item["scenario"]
    if scenario == "Incomplete":
        gt_images = gt_images[:1]
        retrieved_images = retrieved_images[:1]

    question_text, options = parse_question_and_options(str(row["prompt"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_images = []
    query_path = out_dir / "query.png"
    save_pil_image(query_img, query_path)
    saved_images.append({"role": "query", "path": str(query_path), "name": query_path.name})

    magiclens_query_instruction = None
    retrieval_rows = []

    if mode == "no_rag":
        llava_num_images = 1
    elif mode == "dataset_retrieved_rerank":
        orig_dir = out_dir / "retrieved_original"
        rerank_dir = out_dir / "retrieved_reranked"
        orig_rows = []
        for i, img in enumerate(retrieved_images, start=1):
            p = orig_dir / f"rag_orig{i}.png"
            save_pil_image(img, p)
            orig_rows.append({"rank": i, "role": f"rag_orig{i}", "path": str(p), "name": p.name})

        reranked = list(retrieved_images)
        rerank_meta = row.get("meta_magiclens_rag_ranks", []) or []
        if rerank_meta:
            reordered = []
            for item in sorted(rerank_meta, key=lambda x: int(x["new_rank"])):
                orig_idx = int(item["orig_rag_index"]) - 1
                if 0 <= orig_idx < len(retrieved_images):
                    reordered.append(retrieved_images[orig_idx])
            if len(reordered) == len(retrieved_images):
                reranked = reordered
        reranked_rows = []
        for i, img in enumerate(reranked, start=1):
            p = rerank_dir / f"rag_top{i}.png"
            save_pil_image(img, p)
            reranked_rows.append({"rank": i, "role": f"rag_top{i}", "path": str(p), "name": p.name})

        saved_images.extend(orig_rows)
        saved_images.extend(reranked_rows)
        retrieval_rows = reranked_rows
        magiclens_query_instruction = question_text
        llava_num_images = 1 + len(reranked_rows)
    elif mode in ("corpus_clip", "corpus_magiclens"):
        corpus_rows = row.get("meta_corpus_retrieval", []) or []
        rag_dir = out_dir / "corpus_retrieval"
        for item in corpus_rows:
            src = Path(item["path"])
            dst = rag_dir / f"rag_top{int(item['rank'])}{src.suffix.lower() or '.png'}"
            copied = copy_path_if_exists(src, dst)
            retrieval_rows.append(
                {
                    "rank": int(item["rank"]),
                    "path": str(dst if copied else src),
                    "source_path": str(src),
                    "score": float(item["score"]),
                    "copied": copied,
                    "name": dst.name if copied else src.name,
                }
            )
        saved_images.extend(
            [{"role": f"rag_top{r['rank']}", "path": r["path"], "name": r["name"]} for r in retrieval_rows]
        )
        if mode == "corpus_magiclens":
            magiclens_query_instruction = question_text
        llava_num_images = 1 + len(retrieval_rows)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    llava_instruction = build_llava_instruction(llava_num_images, use_rag=(llava_num_images > 1))
    full_prompt = f"{llava_instruction}\n{question_text}\n Choices:\n" + "\n".join(
        [f"{k}: {v}" for k, v in options.items()]
    )

    prompt_payload = {
        "llava_instruction": llava_instruction,
        "question_text": question_text,
        "options": options,
        "llava_full_prompt": full_prompt,
        "magiclens_query_instruction": magiclens_query_instruction,
    }
    with open(out_dir / "prompts.json", "w", encoding="utf-8") as f:
        json.dump(prompt_payload, f, ensure_ascii=False, indent=2)

    with open(out_dir / "result_row.json", "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)

    if cfg["summary_json"].exists():
        copy_path_if_exists(cfg["summary_json"], out_dir / cfg["summary_json"].name)
    if cfg["log_file"].exists():
        copy_path_if_exists(cfg["log_file"], out_dir / cfg["log_file"].name)

    report = {
        "pipeline": pipeline,
        "title": cfg["title"],
        "sample": {
            "sample_index": sample_index,
            "qs_id": str(row["qs_id"]),
            "scenario": sample_item["scenario"],
            "aspect": sample_item["aspect"],
            "gt_choice": sample_item["answer_choice"],
            "gt_answer": sample_item["answer"],
        },
        "prompts": prompt_payload,
        "images": saved_images,
        "retrieval": retrieval_rows,
        "result": {
            "raw_output": row["output"],
            "pred_choice": row.get("meta_pred_choice") or extract_choice_simple(str(row["output"])),
            "gt_choice": row["gt_choice"],
            "is_correct": (row.get("meta_pred_choice") or extract_choice_simple(str(row["output"]))) == str(row["gt_choice"]),
        },
    }

    if pipeline == "E2":
        report["rerank"] = row.get("meta_magiclens_rag_ranks", [])
    if pipeline in ("E3", "E7"):
        report["retriever_type"] = row.get("meta_retriever_type", "clip" if pipeline == "E3" else "magiclens")

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md = []
    md.append(f"# {cfg['title']} Demo Bundle")
    md.append("")
    md.append(f"- sample_qs_id: {row['qs_id']}")
    md.append(f"- scenario: {sample_item['scenario']}")
    md.append(f"- aspect: {sample_item['aspect']}")
    md.append(f"- gt_choice: {sample_item['answer_choice']}")
    md.append(f"- gt_answer: {sample_item['answer']}")
    md.append("")
    md.append("## Prompts")
    md.append(f"- LLaVA instruction: `{llava_instruction}`")
    if magiclens_query_instruction:
        md.append(f"- MagicLens query instruction: `{magiclens_query_instruction}`")
    md.append("")
    md.append("## Question")
    md.append(question_text)
    md.append("")
    md.append("## Options")
    for k, v in options.items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Images")
    for item in saved_images:
        md.append(f"- {item['role']}: {item['path']}")
    if retrieval_rows:
        md.append("")
        md.append("## Retrieval")
        for item in retrieval_rows:
            line = f"- rank {item['rank']}: {item['path']}"
            if "score" in item:
                line += f" (score={item['score']:.6f})"
            md.append(line)
    md.append("")
    md.append("## Result")
    md.append(f"- raw_output: {row['output']}")
    md.append(f"- pred_choice: {report['result']['pred_choice']}")
    md.append(f"- is_correct: {report['result']['is_correct']}")
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def extract_choice_simple(text: str) -> str:
    text = str(text).upper()
    for c in ("A", "B", "C", "D"):
        if f"({c})" in text:
            return c
    for c in ("A", "B", "C", "D"):
        if c in text.split():
            return c
    return "N/A"


def main():
    parser = argparse.ArgumentParser(description="Export a one-sample demo bundle from existing pipeline results.")
    parser.add_argument("--pipeline", required=True, choices=sorted(PIPELINES.keys()))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = PIPELINES[args.pipeline]
    if not cfg["result_jsonl"].exists():
        raise FileNotFoundError(f"result jsonl not found: {cfg['result_jsonl']}")

    rows = load_results(cfg["result_jsonl"])
    if args.sample_id is not None:
        row = next((r for r in rows if str(r["qs_id"]) == str(args.sample_id)), None)
        if row is None:
            raise ValueError(f"sample_id not found in result jsonl: {args.sample_id}")
        _, sample_item = load_sample(sample_id=args.sample_id, cache_dir=args.hf_cache_dir)
        sample_tag = f"sample_{args.sample_id}"
    else:
        if args.sample_index < 0 or args.sample_index >= len(rows):
            raise ValueError(f"invalid sample_index: {args.sample_index}")
        row = rows[args.sample_index]
        _, sample_item = load_sample(sample_id=str(row["qs_id"]), cache_dir=args.hf_cache_dir)
        sample_tag = f"sample{args.sample_index}"

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = ROOT_DIR / "log/demo_review" / args.pipeline / sample_tag
    build_bundle(args.pipeline, row, sample_item, sample_index, out_dir)
    print(f"[OK] demo_bundle={out_dir}")


if __name__ == "__main__":
    main()
