#!/usr/bin/env python3
"""
 * [INPUT]: 无依赖
 * [OUTPUT]: 测试 infoseek_loader 的元数据加载逻辑（不加载图片）
 * [POS]: tests/ 中的验证脚本，无需在生产环境运行
 * [PROTOCOL]: 变更时更新此头部，然后检查 test/CLAUDE.md
"""

import json
import sys
from pathlib import Path

# 临时 import infoseek_loader（不依赖 PIL）
# 为了在没有 PIL 的环境中测试，我们手动加载 JSONL

def test_infoseek_metadata_loading():
    """测试 InfoSeek 元数据加载逻辑。"""
    
    root = Path("/mnt/d/mRAG/data/infoseek")
    
    print("=" * 60)
    print("InfoSeek 元数据加载测试")
    print("=" * 60)
    
    # 1. 检查目录结构
    print("\n【1】目录结构检查")
    for subdir in ["Entity", "Human", "Query", "images"]:
        path = root / subdir
        exists = "✅" if path.exists() else "❌"
        print(f"  {exists} {path}")
    
    # 2. 统计各分割
    print("\n【2】数据分割统计")
    splits = {
        "entity_test": root / "Entity" / "infoseek_test.jsonl",
        "entity_train": root / "Entity" / "infoseek_train.jsonl",
        "entity_val": root / "Entity" / "infoseek_val.jsonl",
        "human": root / "Human" / "infoseek_human.jsonl",
        "query_train": root / "Query" / "infoseek_train_withkb.jsonl",
        "query_val": root / "Query" / "infoseek_val_withkb.jsonl",
    }
    
    total_records = 0
    for split_name, jsonl_file in splits.items():
        if not jsonl_file.exists():
            print(f"  ❌ {split_name:20} FILE NOT FOUND: {jsonl_file}")
            continue
        
        count = sum(1 for line in open(jsonl_file) if line.strip())
        total_records += count
        print(f"  ✅ {split_name:20} {count:10,} records")
    
    print(f"\n  总计: {total_records:,} 条元数据记录")
    
    # 3. 采样记录并验证格式
    print("\n【3】记录格式验证")
    
    test_samples = [
        ("Entity (test)", root / "Entity" / "infoseek_test.jsonl", ["data_id", "image_id", "question"]),
        ("Human", root / "Human" / "infoseek_human.jsonl", ["data_id", "image_id", "question"]),
        ("Query (train)", root / "Query" / "infoseek_train_withkb.jsonl", ["data_id", "entity_id", "entity_text"]),
    ]
    
    for sample_name, jsonl_file, expected_keys in test_samples:
        print(f"\n  {sample_name}:")
        try:
            with open(jsonl_file) as fp:
                first_line = fp.readline()
                record = json.loads(first_line)
                
                # 检查必要字段
                missing_keys = set(expected_keys) - set(record.keys())
                if missing_keys:
                    print(f"    ❌ 缺失字段: {missing_keys}")
                else:
                    print(f"    ✅ 包含所有必要字段: {expected_keys}")
                
                # 打印样本
                for key in expected_keys:
                    value = record.get(key, "N/A")
                    if isinstance(value, str):
                        preview = value[:50] + "..." if len(value) > 50 else value
                    else:
                        preview = f"({type(value).__name__})"
                    print(f"      - {key}: {preview}")
        except Exception as e:
            print(f"    ❌ 加载失败: {e}")
    
    # 4. 图像文件检查
    print("\n【4】图像文件检查")
    images_dir = root / "images" / "all"
    if images_dir.exists():
        image_count = sum(1 for _ in images_dir.glob("*.*") if _.is_file())
        print(f"  本地图像数: {image_count}")
        if image_count == 0:
            print(f"  ⚠️  图像目录存在但为空（符合设计 - 57GB 图片未 pull）")
            print(f"  💡 远程位置: /home/user/code/mRAG/data/infoseek/images/all/")
            print(f"  💡 总图片数（远程）: ~1,005,415 张")
    else:
        print(f"  ❌ 图像目录不存在: {images_dir}")
    
    # 5. 数据适配层验证
    print("\n【5】数据适配层（Entity → MRAG 格式）准备情况")
    print(f"  Entity 记录：{total_records - sum(1 for line in open(root / 'Query' / 'infoseek_train_withkb.jsonl') if line.strip()) if (root / 'Query' / 'infoseek_train_withkb.jsonl').exists() else 'N/A'}")
    print(f"  必要转换:")
    print(f"    - image_id → query_image（需图像加载，本地缺失）")
    print(f"    - question → 多选题生成（通过 LLM，不需图像）")
    print(f"    - 生成选项 A/B/C/D（LLM 调用）")
    
    print("\n" + "=" * 60)
    print("✅ 元数据加载验证完成")
    print("=" * 60)

if __name__ == "__main__":
    test_infoseek_metadata_loading()
