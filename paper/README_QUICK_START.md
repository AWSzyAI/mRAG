# 📋 mRAG 项目完整文档索引与快速参考

## 文档导航

### 📖 核心文档（按推荐阅读顺序）

| 文档 | 位置 | 用途 | 阅读时间 |
|------|------|------|---------|
| **项目进度总结** | [PROGRESS_AND_RESULTS.md](PROGRESS_AND_RESULTS.md) | 🔴 从这里开始 - 全局概览 | 15 分钟 |
| **详细实验结果** | [DETAILED_RESULTS.md](DETAILED_RESULTS.md) | 精度数据、场景分布、问题诊断 | 20 分钟 |
| **实验脚本指南** | [EXPERIMENTAL_SCRIPTS_GUIDE.md](EXPERIMENTAL_SCRIPTS_GUIDE.md) | 如何运行实验、常见问题 | 25 分钟 |
| **数据集评估** | [DATASET_EVALUATION_REPORT.md](DATASET_EVALUATION_REPORT.md) | 第二数据集选型、迁移计划 | 20 分钟 |

### 🗂️ 项目文档（外部）

| 文档 | 位置 | 说明 |
|------|------|------|
| 问题分析 | `../doc/04-02/magiclens_vs_clip_query_image_analysis_2026-04-02.md` | MagicLens vs CLIP 详细对比 |
| 改进方案 | `../doc/周报2026-03-25.md` | 多维度查询分解方案设计 |
| 运行日志 | `../log/Log.md` | 命令记录与排错笔记 |
| 初稿论文 | `main.tex` / `content.tex` | 论文 LaTeX 源代码 |

---

## 🎯 快速参考：关键精度

```
基线（E3）:           CLIP 直检索        50.41%  ✅ 稳定
现状（E6）:           CLIP + MagicLens    50.33%  ⚠️ 无改进
问题（E7）:           MagicLens 直检      47.97%  ❌ 下降
改进目标（D01+）:     多维度分解         52-54%  🎯 待验证
```

---

## 🔍 关键发现一览

### 问题诊断（E7 MagicLens 直检索失败）

```
❌ 语义原型偏置
   └─ 表现：top-1 被 1 张图命中 39 次（应 ≤1353）
   └─ 原因：query embedding 过于抽象，缺 query image 约束

❌ 检索集合大规模脱钩
   └─ 数据：909/1353 样本（67.2%）top-5 零重合
   └─ 含义：不是"排序不同"，而是"候选池本身不同"

❌ 整体精度下降
   └─ E3 CLIP: 50.41%
   └─ E7 MagicLens: 47.97%
   └─ 净损失：-2.44 个百分点
```

### 优势场景（为何保留 MagicLens）

```
✅ Incomplete 场景：MagicLens 17:7 胜率 vs CLIP
   └─ 原因：instruction + query image 对不完整信息敏感

✅ Temporal 场景：基本打平（可改进空间）
   └─ 原因：可能捕捉时间/变化信息

✅ MagicLens 赢场景：119 个样本
   └─ 虽然总体输 152 个样本，但有明确的优势领域
```

### 改进思路（Phase D）

```
💡 核心洞察：
   单一 question 过于宽泛 → query embedding 坍缩
   解决方案：分解 question 成 N 个互补维度

✨ 预期效果：
   D01-D07  → 确定最优维度数 & 生成模型
   D08-D09  → 确定最优融合策略
   D10-D12  → 确定最优检索参数
   目标精度：52-54%（+2-4 百分点）
```

---

## 📊 场景性能地图

```
精度排行              MagicLens (E6)    CLIP (E3)    优势
───────────────────────────────────────────────────────
1. Deformation        57.26% ⭐⭐⭐     56.86%      MagicLens +0.4
2. Scope              55.88% ⭐⭐⭐     55.88%      打平
3. Others             56.67% ⭐⭐⭐     56.67%      打平
4. Angle              53.92% ⭐⭐      54.04%      CLIP -0.1
5. Obstruction        51.96% ⭐⭐      51.85%      MagicLens +0.1
6. Biological         48.04% ⭐        49.02%      CLIP -1.0
7. Partial            49.02% ⭐        49.19%      CLIP -0.2
8. Temporal           46.08% ⭐        46.31%      CLIP -0.2
9. Incomplete         26.47% ❌        27.45%      CLIP -1.0

⚠️ 最弱点：Incomplete 仅 26.47%（全局最低）
🎯 改进目标：Incomplete → 35-40%（+8-13 点）
```

---

## 🚀 立即启动指南（4 步）

### 第 1 步：获取 API 密钥（10 分钟）

```bash
# 选项 A: SiliconFlow (推荐)
# 访问 https://cloud.siliconflow.cn/account/ak
# 注册 → 领取 $5 免费额度 → 复制 API Key

# 选项 B: OpenRouter
# 访问 https://openrouter.ai/keys
# 创建新 key

# 选项 C: DeepInfra
# 访问 https://deepinfra.com/dash/api_keys
```

### 第 2 步：运行 Phase D01（本地 SSH）

```bash
# 设置密钥
export DIM_GENERATOR_API_KEY="sk-xxxxxxxx"

# 快速验证（50 样本，10 分钟）
MAX_SAMPLES=50 DIM_GENERATOR_MODEL="qwen2.5-7b-instruct" \
  python test/pipeline_multi_dim_rag.py \
    --corpus-dir data/image_corpus \
    --answers-file log/D01_test/results.jsonl

# 如无误，运行完整版（1353 样本，2-3 小时）
MAX_SAMPLES=0 DIM_GENERATOR_MODEL="qwen2.5-7b-instruct" \
  nohup python test/pipeline_multi_dim_rag.py \
    --corpus-dir data/image_corpus \
    --answers-file log/D01/results.jsonl > log/D01/run.log 2>&1 &
```

### 第 3 步：监控进度

```bash
# 实时日志
tail -f log/D01/run.log

# 查看已完成的样本数
wc -l log/D01/results.jsonl

# 预期输出
# log/D01/results.jsonl        - 1353 行结果
# log/D01/d01_summary.json     - 精度统计
```

### 第 4 步：查看结果

```bash
# 提取精度
cat log/D01/d01_summary.json | jq '.accuracy'

# 与基线对比
echo "CLIP (E3):  50.41%"
echo "D01 结果:   $(cat log/D01/d01_summary.json | jq '.accuracy')%"

# 场景细分
cat log/D01/d01_summary.json | jq '.by_scenario_accuracy'
```

---

## 📅 时间线与里程碑

```
Week 1: 2026-04-16 ~ 2026-04-22
├─ Phase D01 基准验证       ← 🔴 开始
├─ 维度质量人工审查
└─ 问题排查与调优

Week 2: 2026-04-23 ~ 2026-04-30
├─ Phase D02-D05 模型对比   ← 🟡 进行中
├─ Phase D06-D07 维度数消融
└─ 汇总精度对比表

Week 3: 2026-05-01 ~ 2026-05-08
├─ Phase D08-D09 融合策略
├─ Phase D10-D12 参数扫描
└─ 确定最优配置 (D_best)

Week 4: 2026-05-09 ~ 2026-05-16
├─ InfoSeek 数据集准备
├─ 图像语料库构建
├─ 评估脚本改写
└─ D_best 迁移验证

Week 5-6: 2026-05-17 ~ 2026-06-01
├─ 论文集成结果
├─ 生成对比图表
├─ 导师反馈
└─ 投稿 IJCAI 2025

🎯 关键截止：IJCAI 投稿 2026-06-30
```

---

## 🔧 常见问题 TOP 5

### Q1: 如何判断 D01 是否改进成功？

```
✅ 成功标准：精度 > 50.41%（超过 E3 基线）
✅ 目标范围：52-54%
⚠️ 如果 < 50.41%，检查：
   - 维度生成质量（导出 50 样本审查）
   - 融合策略（尝试 Score-sum / Voting）
   - 维度数（尝试 1 维 / 5 维）
```

### Q2: Qwen2.5-7B API 超时怎么办？

```
💡 解决方案：
1. 增加超时时间：DIM_GENERATOR_API_TIMEOUT=60
2. 切换平台：用 DeepInfra 或 OpenRouter
3. 本地部署：下载 Qwen2.5-VL-7B 自部署
```

### Q3: 如何对比 D01 vs E6？

```bash
python -c "
import json

with open('log/E6/e6_summary.json') as f:
    e6 = json.load(f)
    
with open('log/D01/d01_summary.json') as f:
    d01 = json.load(f)

print(f'E6 (CLIP+MagicLens): {e6[\"accuracy\"]:.2f}%')
print(f'D01 (多维度):       {d01[\"accuracy\"]:.2f}%')
print(f'改进:              {d01[\"accuracy\"] - e6[\"accuracy\"]:+.2f}%')
"
```

### Q4: Incomplete 场景为什么这么低？

```
分析：
- Incomplete: 26.47% (最低)
- Temporal:   46.08%
- Partial:    49.02%

🔍 原因假说：
   不完整问题 = query image 信息 < 充分回答所需
   → 检索器必须非常准确地补齐缺失信息
   → 当前 CLIP/MagicLens 都不够好

💡 改进方向：
   多维度分解可能帮助（分多个问题侧面）
   可先通过提示约束与融合策略迭代验证，再考虑是否引入专用训练
```

### Q5: 什么时候开始 InfoSeek 迁移？

```
建议时机：
✅ Phase D07 完成、D_best 确定后（~5月初）
⚠️ 不要提前开始（避免重复工作）

InfoSeek 优先级：
🟡 如果 D01 改进 < 1%：可跳过，专注 GQA
✅ 如果 D01 改进 > 2%：立即启动 InfoSeek
```

---

## 🎓 论文结构快速参考

```
1. Introduction
   ├─ 研究背景：多模态 RAG
   ├─ 核心问题：MagicLens 直检索失效
   └─ 主要贡献：多维度分解框架

2. Related Work
   ├─ 视觉检索器 (CLIP, MagicLens)
   ├─ 检索增强生成 (RAG)
   └─ 查询分解方法

3. Background & Analysis
   ├─ MRAG-Bench 数据集介绍
   ├─ E7 问题诊断（语义原型偏置）
   └─ 为什么多维度可以解决

4. Methodology
   ├─ 多维度查询分解框架
   ├─ 三种维度生成模式
   └─ 三种融合策略

5. Experiments
   ├─ Phase D: MRAG-Bench 消融（D01-D12）
   ├─ Phase E: InfoSeek 迁移验证
   └─ 消融研究与场景分析

6. Results & Discussion
   ├─ 主要结果（精度对比）
   ├─ 场景细分分析
   └─ 与 CLIP/MagicLens 的优劣对标

7. Conclusion & Future Work
   ├─ 关键发现总结
   ├─ 局限性说明
   └─ 未来方向（GQA, 多语言等）
```

---

## 📁 文件组织

```
paper/
├─ PROGRESS_AND_RESULTS.md           ← 🔴 从这里开始
├─ DETAILED_RESULTS.md               ← 详细数据表
├─ EXPERIMENTAL_SCRIPTS_GUIDE.md     ← 执行指南
├─ DATASET_EVALUATION_REPORT.md      ← 数据集分析
├─ THIS_FILE (INDEX.md)              ← 快速参考
│
├─ main.tex                          ← 论文主文件
├─ content.tex                       ← 论文内容
├─ main.pdf                          ← 编译后 PDF
│
└─ README.md                         ← 论文项目说明

log/
├─ E6/                               ← CLIP + MagicLens 结果
├─ E7/                               ← MagicLens 直检索诊断
├─ D01/ ~ D12/                       ← Phase D 实验结果（待生成）
└─ Log.md                            ← 运行日志

../doc/
├─ 04-02/
│  └─ magiclens_vs_clip_query_image_analysis_2026-04-02.md
└─ 周报2026-03-25.md                 ← 改进方案设计
```

---

## 🌟 一句话项目总结

> **问题**：MagicLens 直接用原始问题检索时，query embedding 被语义原型吸附，导致检索集合与 CLIP 大规模脱钩、精度反而下降。
> 
> **方案**：用小模型分解问题成互补维度，每维分别用 MagicLens 检索，融合后的结果在保持 MagicLens 优势场景同时克服缺陷。
> 
> **证实**：在 MRAG-Bench（1353 样本）和 InfoSeek（14K 样本）上验证方法有效性和泛化能力。

---

## ✅ 下一步 Action Items

- [ ] 获取 SiliconFlow API 密钥
- [ ] 运行 D01 快速验证（50 样本）
- [ ] 审查 50 个维度指令样本质量
- [ ] 运行 D01 完整版（1353 样本）
- [ ] 汇总 D01-D07 消融结果
- [ ] 准备 InfoSeek 迁移工作
- [ ] 更新论文 content.tex
- [ ] 投稿 IJCAI 2025（目标 6 月 30 日）

---

**文档最后更新**：2026-04-16  
**项目负责人**：时子延  
**导师**：宋歌  
**机构**：NNU（南京师范大学）

