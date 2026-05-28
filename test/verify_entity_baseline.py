#!/usr/bin/env python3
"""
InfoSeek 方案 D 快速基线验证脚本

目的: 用 Entity split 原生的多选题（不转换），测量基线准确率
时间: 1 小时内完成

结果输出:
- baseline_accuracy: 多选题本身的准确率 (常见答案分布)
- mrag_bench_compatibility: MRAG-Bench 评估框架是否适配
- data_quality_check: Entity split 质量评估
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import random
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_entity_split(jsonl_path: str, max_samples: int = None) -> List[Dict]:
    """加载 Entity split JSONL"""
    records = []
    try:
        with open(jsonl_path) as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break
                records.append(json.loads(line))
        logger.info(f"✅ 加载 {len(records)} 条记录")
        return records
    except Exception as e:
        logger.error(f"❌ 加载失败: {e}")
        raise


def check_mrag_compatibility(record: Dict) -> bool:
    """检查单条记录是否兼容 MRAG-Bench 格式"""
    required = {"data_id", "image_id", "question", "options", "correct"}
    has_options = isinstance(record.get("options"), dict)
    has_4_choices = has_options and len(record.get("options", {})) == 4
    has_correct = record.get("correct") in ["A", "B", "C", "D"]
    
    return all(k in record for k in required) and has_4_choices and has_correct


def get_question_difficulty(question: str) -> str:
    """简单的问题难度分类"""
    word_count = len(question.split())
    if word_count <= 3:
        return "easy"
    elif word_count <= 7:
        return "medium"
    else:
        return "hard"


def analyze_baseline_accuracy(records: List[Dict]) -> Dict[str, Any]:
    """
    分析基线准确率
    
    方法:
    1. 随机选择 (Random baseline): 25% (4选1)
    2. 大多数类 (Majority class): 最常出现的答案选项比例
    3. 简单启发式 (Heuristic): 某些选项出现频率高（如选 A）
    """
    if not records:
        return {}
    
    correct_answers = [r.get("correct") for r in records if r.get("correct")]
    answer_dist = {}
    for ans in correct_answers:
        answer_dist[ans] = answer_dist.get(ans, 0) + 1
    
    # 计算各基线
    random_baseline = 0.25  # 4选1，随机选 25%
    
    # 多数类基线
    if answer_dist:
        max_count = max(answer_dist.values())
        majority_baseline = max_count / len(records)
    else:
        majority_baseline = 0.25
    
    # 选 A 基线（常见的模型退化）
    a_count = answer_dist.get("A", 0)
    always_a_baseline = a_count / len(records)
    
    logger.info("\n【基线准确率分析】")
    logger.info(f"  随机选择基线: {random_baseline:.1%}")
    logger.info(f"  多数类基线:   {majority_baseline:.1%}")
    logger.info(f"  总选 A 基线:   {always_a_baseline:.1%}")
    logger.info(f"\n  答案分布: {dict(sorted(answer_dist.items()))}")
    
    return {
        "random_baseline": random_baseline,
        "majority_baseline": majority_baseline,
        "always_a_baseline": always_a_baseline,
        "answer_distribution": answer_dist,
    }


def analyze_data_quality(records: List[Dict]) -> Dict[str, Any]:
    """质量指标"""
    
    mrag_compatible = 0
    has_all_fields = 0
    question_lengths = []
    image_id_count = 0
    difficulty_dist = {"easy": 0, "medium": 0, "hard": 0}
    
    for r in records:
        if check_mrag_compatibility(r):
            mrag_compatible += 1
        
        required = {"data_id", "image_id", "question"}
        if all(k in r for k in required):
            has_all_fields += 1
            question_lengths.append(len(r["question"].split()))
            difficulty_dist[get_question_difficulty(r["question"])] += 1
        
        if r.get("image_id"):
            image_id_count += 1
    
    logger.info("\n【数据质量指标】")
    logger.info(f"  MRAG 兼容性:  {mrag_compatible}/{len(records)} ({mrag_compatible/len(records):.1%})")
    logger.info(f"  必要字段完整: {has_all_fields}/{len(records)} ({has_all_fields/len(records):.1%})")
    logger.info(f"  有图像 ID:   {image_id_count}/{len(records)} ({image_id_count/len(records):.1%})")
    
    if question_lengths:
        avg_len = sum(question_lengths) / len(question_lengths)
        logger.info(f"  平均问题长度: {avg_len:.1f} 字")
    
    logger.info(f"  难度分布: {difficulty_dist}")
    
    return {
        "mrag_compatible_ratio": mrag_compatible / len(records),
        "complete_fields_ratio": has_all_fields / len(records),
        "with_image_ratio": image_id_count / len(records),
        "avg_question_length": sum(question_lengths) / len(question_lengths) if question_lengths else 0,
        "difficulty_distribution": difficulty_dist
    }


def sample_and_display(records: List[Dict], n_samples: int = 5):
    """显示样本"""
    logger.info(f"\n【数据样本示例】({n_samples} 个)")
    samples = random.sample(records, min(n_samples, len(records)))
    
    for idx, record in enumerate(samples, 1):
        logger.info(f"\n  样本 {idx}:")
        logger.info(f"    数据 ID: {record.get('data_id', 'N/A')}")
        logger.info(f"    问题: {record.get('question', 'N/A')}")
        
        options = record.get("options", {})
        if options:
            logger.info(f"    选项:")
            for opt_key, opt_val in sorted(options.items()):
                mark = " ← 正确答案" if opt_key == record.get("correct") else ""
                logger.info(f"      {opt_key}: {opt_val}{mark}")


def main():
    # 配置
    entity_test_file = Path("/mnt/d/mRAG/data/infoseek/Entity/infoseek_test.jsonl")
    entity_train_file = Path("/mnt/d/mRAG/data/infoseek/Entity/infoseek_train.jsonl")
    
    logger.info("\n" + "█"*60)
    logger.info("█" + " "*58 + "█")
    logger.info("█" + "InfoSeek 方案 D 基线验证 (Entity split)".center(58) + "█")
    logger.info("█" + " "*58 + "█")
    logger.info("█"*60)
    
    # 加载数据
    if entity_test_file.exists():
        logger.info("\n【加载 Test split】")
        test_records = load_entity_split(str(entity_test_file), max_samples=10000)
    else:
        logger.error(f"❌ 文件不存在: {entity_test_file}")
        sys.exit(1)
    
    # 质量分析
    quality_test = analyze_data_quality(test_records)
    
    # 基线分析
    baselines_test = analyze_baseline_accuracy(test_records)
    
    # 样本展示
    sample_and_display(test_records, n_samples=3)
    
    # 总结
    logger.info("\n" + "="*60)
    logger.info("【决策建议】")
    logger.info("="*60)
    
    compatibility = quality_test["mrag_compatible_ratio"]
    if compatibility > 0.95:
        logger.info("✅ Entity split 质量优秀，可直接用方案 D")
        logger.info("   → 继续: python test/benchmark_infoseek.py --mode local")
        logger.info("            --input-jsonl data/infoseek/Entity/infoseek_test.jsonl")
        logger.info("            --max-samples 347980")
    else:
        logger.info(f"⚠️  兼容性 {compatibility:.1%}，需要检查数据格式")
    
    if baselines_test["majority_baseline"] < 0.4:
        logger.info("✅ 答案分布均衡 → 多选一任务有区分度")
    else:
        logger.info(f"⚠️  答案分布不均 (最多类 {baselines_test['majority_baseline']:.1%})")
        logger.info("   → 可考虑加权评估")
    
    logger.info(f"\n✅ 预期 MRAG-Bench 兼容性: {compatibility:.1%}")
    logger.info(f"✅ 方案 D 可行性: {'HIGH' if compatibility > 0.90 else 'MEDIUM' if compatibility > 0.80 else 'LOW'}")
    
    # 输出统计
    logger.info("\n【完整统计】")
    stats = {
        "test_split": {
            "total_records": len(test_records),
            "quality": quality_test,
            "baselines": baselines_test
        }
    }
    
    output_file = Path("/tmp/infoseek_baseline_report.json")
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"📊 详细报告已保存: {output_file}")
    
    logger.info("\n" + "="*60)


if __name__ == "__main__":
    main()
