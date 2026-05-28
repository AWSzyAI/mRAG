#!/usr/bin/env python3
"""
为 E11_4 在 InfoSeek 上准备 10000 样本的采样列表

生成：
- sample_indices.json: 采样的行号列表
- sample_metadata.json: 采样样本的详细元数据（data_id, image_id, question）
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import argparse


def load_infoseek_metadata(
    data_root: str,
    split: str = "entity_test",
    max_total: int = None
) -> List[Dict[str, Any]]:
    """加载 InfoSeek 数据集元数据"""
    split_to_path = {
        "entity_test": Path(data_root) / "Entity" / "infoseek_test.jsonl",
        "entity_train": Path(data_root) / "Entity" / "infoseek_train.jsonl",
        "entity_val": Path(data_root) / "Entity" / "infoseek_val.jsonl",
        "human": Path(data_root) / "Human" / "infoseek_human.jsonl",
    }
    
    jsonl_path = split_to_path[split]
    if not jsonl_path.exists():
        raise FileNotFoundError(f"File not found: {jsonl_path}")
    
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if max_total and len(records) >= max_total:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append({
                "line_idx": line_idx,
                "data_id": obj.get("data_id", ""),
                "image_id": obj.get("image_id", ""),
                "question": obj.get("question", ""),
            })
    return records


def prepare_samples(
    data_root: str,
    split: str = "entity_test",
    sample_size: int = 10000,
    random_seed: int = 42,
    output_dir: str = None
) -> Dict[str, Any]:
    """准备指定大小的随机采样"""
    
    if output_dir is None:
        output_dir = str(Path(data_root).parent / f"infoseek_{split}_{sample_size}k")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/3] 加载 {split} 元数据...")
    records = load_infoseek_metadata(data_root, split)
    print(f"      总共 {len(records)} 条记录")
    
    # 检查是否有足够的样本
    if len(records) < sample_size:
        print(f"[WARN] 只有 {len(records)} 条记录，少于请求的 {sample_size}")
        sample_size = len(records)
        sampled_indices = list(range(len(records)))
    else:
        random.seed(random_seed)
        sampled_indices = sorted(random.sample(range(len(records)), sample_size))
    
    print(f"[2/3] 采样 {len(sampled_indices)} 条记录 (seed={random_seed})...")
    
    sampled_records = [records[i] for i in sampled_indices]
    
    # 保存采样索引
    indices_file = output_dir / "sample_indices.json"
    with open(indices_file, "w") as f:
        json.dump({
            "split": split,
            "total_in_split": len(records),
            "sample_size": len(sampled_indices),
            "random_seed": random_seed,
            "indices": sampled_indices,
        }, f, indent=2)
    print(f"      ✓ 采样索引: {indices_file}")
    
    # 保存样本元数据
    metadata_file = output_dir / "sample_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump({
            "split": split,
            "total_sampled": len(sampled_records),
            "random_seed": random_seed,
            "samples": [
                {
                    "sample_id": i,  # 在采样内的序号 0-9999
                    "line_idx": r["line_idx"],  # 原始文件中的行号
                    "data_id": r["data_id"],
                    "image_id": r["image_id"],
                    "question": r["question"],
                }
                for i, r in enumerate(sampled_records)
            ]
        }, f, ensure_ascii=False, indent=2)
    print(f"      ✓ 样本元数据: {metadata_file}")
    
    # 保存为 JSONL 格式（便于流式处理）
    jsonl_file = output_dir / "samples.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for i, r in enumerate(sampled_records):
            row = {
                "sample_id": i,
                "line_idx": r["line_idx"],
                "data_id": r["data_id"],
                "image_id": r["image_id"],
                "question": r["question"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"      ✓ JSONL 格式: {jsonl_file}")
    
    print(f"[3/3] 生成统计摘要...")
    summary = {
        "experiment": "E11_4_InfoSeek_10K",
        "dataset": split,
        "total_in_split": len(records),
        "sampled_count": len(sampled_indices),
        "random_seed": random_seed,
        "output_directory": str(output_dir),
        "files_generated": [
            "sample_indices.json",
            "sample_metadata.json",
            "samples.jsonl",
        ],
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"      ✓ 摘要: {summary_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="为 E11_4 在 InfoSeek 上准备 10000 样本"
    )
    parser.add_argument(
        "--data-root",
        default="/mnt/d/mRAG/data/infoseek",
        help="InfoSeek 数据根目录"
    )
    parser.add_argument(
        "--split",
        default="entity_test",
        choices=["entity_test", "entity_train", "entity_val", "human"],
        help="要采样的数据分割"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="采样大小"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--output-dir",
        help="输出目录（默认为 {data_root}/../infoseek_{split}_{sample_size}k）"
    )
    
    args = parser.parse_args()
    
    summary = prepare_samples(
        data_root=args.data_root,
        split=args.split,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
    )
    
    print("\n" + "="*60)
    print("✓ 采样完成!")
    print("="*60)
    print(f"输出目录: {summary['output_directory']}")
    print(f"采样数量: {summary['sampled_count']}")
    print(f"随机种子: {summary['random_seed']}")
    print("\n下一步: 运行 E11_4 benchmark")
    print(f"  bash test/E11_4_infoseek.sh --sample-dir {summary['output_directory']}")


if __name__ == "__main__":
    main()
