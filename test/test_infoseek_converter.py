#!/usr/bin/env python3
"""
测试 InfoSeekConverter 的基本功能

包括:
1. 单个问题转换
2. 缓存验证
3. 批量转换
"""

import sys
import json
import tempfile
from pathlib import Path

# 添加 src 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mrag.infoseek_converter import InfoSeekConverter, ConversionResult


def test_cache_mechanism():
    """测试缓存机制"""
    print("\n" + "="*60)
    print("【测试 1】缓存机制验证")
    print("="*60)
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        cache_db = f.name
    
    converter = InfoSeekConverter(
        llm_model="local",  # 使用本地模型避免 API 调用
        cache_db=cache_db,
        cache_enabled=True
    )
    
    question = "What place inflows lake?"
    
    print(f"\n问题: {question}")
    print("\n【第 1 次调用】应该调用 LLM（未缓存）...")
    result1 = converter.convert_single(
        question=question,
        data_id="test_001",
        image_id="oven_00001"
    )
    print(f"  缓存命中: {result1.is_cached}")
    print(f"  答案: {result1.answer}")
    print(f"  干扰项: {result1.distractors}")
    
    print("\n【第 2 次调用】应该命中缓存...")
    result2 = converter.convert_single(
        question=question,
        data_id="test_002",
        image_id="oven_00002"
    )
    print(f"  缓存命中: {result2.is_cached}")
    print(f"  ✅ 验证缓存: {result2.is_cached == True}")
    
    # 清理
    Path(cache_db).unlink()


def test_mrag_format_conversion():
    """测试 MRAG 格式转换"""
    print("\n" + "="*60)
    print("【测试 2】MRAG 格式转换")
    print("="*60)
    
    converter = InfoSeekConverter(llm_model="local")
    
    result = converter.convert_single(
        question="In what year did this become a hiking mountain?",
        data_id="test_003",
        image_id="oven_05517942"
    )
    
    mrag_fmt = result.to_mrag_format()
    
    print(f"\n原始结果类型: {type(result)}")
    print(f"MRAG 格式:")
    print(json.dumps(mrag_fmt, indent=2, ensure_ascii=False))
    
    # 验证格式
    required_keys = {"data_id", "image_id", "question", "options", "correct", "confidence"}
    has_all_keys = all(k in mrag_fmt for k in required_keys)
    print(f"\n✅ 格式检查: {has_all_keys}")
    
    options_keys = set(mrag_fmt["options"].keys())
    correct_keys = {"A", "B", "C", "D"}
    options_valid = options_keys == correct_keys
    print(f"✅ 选项检查: {options_valid}")


def test_batch_conversion():
    """测试批量转换"""
    print("\n" + "="*60)
    print("【测试 3】批量转换（5 个样本）")
    print("="*60)
    
    # 创建临时输入文件
    with tempfile.NamedTemporaryFile(mode='w', suffix=".jsonl", delete=False) as fin:
        input_file = fin.name
        # 写入 5 个测试样本
        test_samples = [
            {"data_id": "test_001", "image_id": "img_001", "question": "What is the capital of France?"},
            {"data_id": "test_002", "image_id": "img_002", "question": "Who wrote Romeo and Juliet?"},
            {"data_id": "test_003", "image_id": "img_003", "question": "What is the largest planet?"},
            {"data_id": "test_004", "image_id": "img_004", "question": "When was the internet invented?"},
            {"data_id": "test_005", "image_id": "img_005", "question": "What color is the sky?"},
        ]
        for sample in test_samples:
            fin.write(json.dumps(sample) + "\n")
    
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as fout:
        output_file = fout.name
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    converter = InfoSeekConverter(llm_model="local")
    
    stats = converter.batch_convert(
        input_jsonl=input_file,
        output_jsonl=output_file,
        max_samples=5,
        batch_size=2,
        skip_errors=True
    )
    
    print(f"\n转换统计:")
    print(f"  总计: {stats['total']}")
    print(f"  成功: {stats['success']}")
    print(f"  失败: {stats['failed']}")
    print(f"  缓存命中: {stats['cached']}")
    print(f"  平均信心度: {stats['avg_confidence']:.2f}")
    
    # 检查输出文件
    output_path = Path(output_file)
    if output_path.exists():
        with open(output_file) as f:
            lines = f.readlines()
        print(f"\n输出文件行数: {len(lines)}")
        
        if lines:
            first_record = json.loads(lines[0])
            print(f"\n第一条转换结果:")
            print(json.dumps(first_record, indent=2, ensure_ascii=False)[:200] + "...")
    
    # 清理
    Path(input_file).unlink()
    Path(output_file).unlink()


def demo_with_real_data():
    """使用真实 InfoSeek 数据演示（仅元数据，不调用 LLM）"""
    print("\n" + "="*60)
    print("【测试 4】真实数据加载演示（前 3 个样本）")
    print("="*60)
    
    infoseek_dir = Path("/mnt/d/mRAG/data/infoseek")
    entity_test_file = infoseek_dir / "Entity" / "infoseek_test.jsonl"
    
    if not entity_test_file.exists():
        print(f"⚠️  数据文件不存在: {entity_test_file}")
        return
    
    converter = InfoSeekConverter(llm_model="local")
    
    print(f"\n读取前 3 个样本...")
    with open(entity_test_file) as f:
        for idx, line in enumerate(f):
            if idx >= 3:
                break
            
            record = json.loads(line)
            print(f"\n【样本 {idx+1}】")
            print(f"  data_id: {record['data_id']}")
            print(f"  image_id: {record['image_id']}")
            print(f"  question: {record['question']}")
            
            # 不实际转换（避免 LLM 调用），只展示格式


if __name__ == "__main__":
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "  InfoSeekConverter 功能测试".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    try:
        test_cache_mechanism()
        test_mrag_format_conversion()
        test_batch_conversion()
        demo_with_real_data()
        
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
