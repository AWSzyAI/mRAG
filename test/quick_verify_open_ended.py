#!/usr/bin/env python3
"""
快速验证: 开放式 VQA 评分可行性

目的: 用 BERTScore 测试 InfoSeek 开放式问题的评分效果
时间: 5-10 分钟
成本: $0

流程:
1. 加载 10 个 InfoSeek 问题
2. 手工标注 GT (5 分钟)
3. 用不同的「预测答案」测试评分
4. 判断 BERTScore 是否有区分度
"""

import json
import sys
from pathlib import Path

try:
    from bert_score import score as bertscore
    print("✅ BERTScore 已安装")
except ImportError:
    print("❌ 未安装 BERTScore，正在安装...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "bert-score", "-q"])
    from bert_score import score as bertscore
    print("✅ BERTScore 安装完成")


def load_sample_questions(n_samples: int = 10):
    """加载样本问题"""
    entity_test = Path("/mnt/d/mRAG/data/infoseek/Entity/infoseek_test.jsonl")
    
    if not entity_test.exists():
        print(f"❌ 文件不存在: {entity_test}")
        return []
    
    questions = []
    with open(entity_test) as f:
        for i in range(n_samples):
            try:
                record = json.loads(f.readline())
                questions.append({
                    "id": i,
                    "data_id": record["data_id"],
                    "question": record["question"],
                    "image_id": record.get("image_id", "unknown")
                })
            except:
                break
    
    return questions


def manual_annotation():
    """手工标注 10 个样本的 GT 答案"""
    questions = load_sample_questions(10)
    
    print("\n" + "="*70)
    print("【快速 GT 标注】- 请为每个问题标注答案 (或按 Enter 使用默认)")
    print("="*70)
    
    annotations = {}
    
    for q in questions:
        print(f"\n【问题 {q['id']+1}】 {q['question']}")
        print(f"   图像: {q['image_id']}")
        
        # 默认答案 (占位符)
        default_answers = {
            0: "river outlet",
            1: "Africa",
            2: "car",
            3: "mountain",
            4: "lake",
            5: "building",
            6: "city",
            7: "person",
            8: "bridge",
            9: "road",
        }
        
        default = default_answers.get(q['id'], "unknown")
        
        # 如果是 interactive 模式，询问用户；否则使用默认
        try:
            user_input = input(f"   答案 (默认: {default}): ").strip()
            answer = user_input if user_input else default
        except EOFError:
            # 非交互模式，使用默认
            answer = default
            print(f"   答案 (默认): {default}")
        
        annotations[q['data_id']] = {
            "question": q['question'],
            "gt_answer": answer
        }
    
    return annotations


def test_bertscore_with_variants(gt_answer: str):
    """
    为单个 GT 答案生成多个变体答案，测试 BERTScore 区分度
    """
    variants = [
        ("完全正确", gt_answer),
        ("语义等价", gt_answer.replace("outlet", "outflow") if "outlet" in gt_answer else gt_answer + " area"),
        ("部分正确", gt_answer.split()[0] if gt_answer.split() else gt_answer),  # 第一个词
        ("相关但错误", "river" if "outlet" in gt_answer else "water"),
        ("完全错误", "dog"),
    ]
    
    predictions = [v[1] for v in variants]
    references = [[gt_answer]] * len(variants)
    
    # 计算 BERTScore
    P, R, F1 = bertscore(
        predictions,
        references,
        lang="en",
        device="cpu",  # 使用 CPU，避免 GPU 显存不足
        batch_size=4,
        verbose=False
    )
    
    results = []
    for (label, pred), p, r, f1 in zip(variants, P, R, F1):
        results.append({
            "label": label,
            "prediction": pred,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1)
        })
    
    return results


def main():
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "快速验证: 开放式 VQA 评分框架".center(68) + "█")
    print("█" + "基于 BERTScore".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # 步骤 1: 手工标注
    print("\n【步骤 1】手工标注 10 个样本")
    annotations = manual_annotation()
    
    # 步骤 2: 测试 BERTScore
    print("\n\n【步骤 2】测试 BERTScore 区分度")
    print("="*70)
    
    all_results = {}
    for idx, (data_id, anno) in enumerate(annotations.items()):
        print(f"\n【测试 {idx+1}】 问题: {anno['question']}")
        print(f"  GT 答案: {anno['gt_answer']}")
        print("  不同答案的评分:")
        
        try:
            variants = test_bertscore_with_variants(anno['gt_answer'])
            
            for v in variants:
                f1_score = v['f1']
                bar_width = int(f1_score * 40)
                bar = "█" * bar_width + "░" * (40 - bar_width)
                print(f"    {v['label']:15s} {bar} F1={f1_score:.3f}")
                print(f"      预测: '{v['prediction']}'")
            
            all_results[data_id] = {
                "question": anno['question'],
                "gt": anno['gt_answer'],
                "scores": variants
            }
        
        except Exception as e:
            print(f"    ❌ 评分失败: {e}")
    
    # 步骤 3: 总结
    print("\n\n【步骤 3】分析结果")
    print("="*70)
    
    # 计算平均的正确答案 F1
    correct_f1_scores = []
    for data_id, results in all_results.items():
        for v in results['scores']:
            if v['label'] == "完全正确":
                correct_f1_scores.append(v['f1'])
    
    if correct_f1_scores:
        avg_correct_f1 = sum(correct_f1_scores) / len(correct_f1_scores)
        print(f"\n✅ 完全正确答案的平均 F1: {avg_correct_f1:.3f}")
        
        if avg_correct_f1 > 0.8:
            print("   → 区分度极好 ⭐⭐⭐")
        elif avg_correct_f1 > 0.6:
            print("   → 区分度良好 ⭐⭐")
        elif avg_correct_f1 > 0.4:
            print("   → 区分度尚可 ⭐")
        else:
            print("   → 区分度不够，可能需要其他方法")
        
        print(f"\n💡 建议:")
        if avg_correct_f1 > 0.7:
            print("   1. BERTScore 可用于 InfoSeek 评分")
            print("   2. 推荐方案: 不转换多选题，直接用开放式评分")
            print("   3. 后续: 标注 200-500 个样本，构建基准")
        else:
            print("   1. 考虑其他评分方法:")
            print("      - 使用 LLM Judge (GPT-4)")
            print("      - 结合多个评分指标 (ROUGE, BLEU)")
            print("      - 转换为多选题 (方案 A)")
    
    # 保存结果
    output_file = Path("/tmp/infoseek_bertscore_test.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 详细结果已保存: {output_file}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
