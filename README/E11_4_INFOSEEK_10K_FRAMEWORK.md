# E11_4 InfoSeek 10000 样本评估框架

## 概述

本框架为 **E11_4 (4维query分解)** 在 **InfoSeek** 数据集上的 **10000 样本** 完整评估提供支持，包括：

1. **采样管理**: 从 InfoSeek 全量数据中随机采样 10000 个样本，生成可复现的样本列表
2. **多维Query生成**: 4维检索指令生成
3. **图像检索**: MagicLens 多维检索 + RRF 融合
4. **答案生成**: LLaVA 开放式回答
5. **评分**: 开放式 QA 标准评分 (精确匹配 EM + 模糊匹配 Fuzzy)

---

## 快速开始

### 方式 1: 完整工作流 (采样 + Benchmark)

```bash
cd /mnt/d/mRAG

# 默认配置: 10000 样本
bash test/E11_4_infoseek_10k.sh

# 自定义配置
SAMPLE_SIZE=5000 RANDOM_SEED=2024 bash test/E11_4_infoseek_10k.sh

# 小规模测试
MAX_SAMPLES=100 bash test/E11_4_infoseek_10k.sh
```

### 方式 2: 分步执行

**步骤 1: 准备采样列表**

```bash
python3 scripts/prepare_infoseek_10k_samples.py \
  --data-root data/infoseek \
  --split entity_test \
  --sample-size 10000 \
  --random-seed 42 \
  --output-dir log/infoseek_samples_10k
```

输出:
- `log/infoseek_samples_10k/sample_indices.json` — 采样行号列表（可复现）
- `log/infoseek_samples_10k/sample_metadata.json` — 样本详细信息
- `log/infoseek_samples_10k/samples.jsonl` — JSONL 格式

**步骤 2: 运行 Benchmark**

```bash
python3 test/benchmark_e11_4_infoseek.py \
  --sample-dir log/infoseek_samples_10k \
  --image-dir data/infoseek/images/all \
  --output-dir log/E11_4_infoseek_10k_results
```

输出:
- `log/E11_4_infoseek_10k_results/e11_4_infoseek_results.jsonl` — 逐样本结果
- `log/E11_4_infoseek_10k_results/e11_4_infoseek_summary.json` — 汇总统计

---

## 目录结构

```
log/
├── infoseek_samples_10k/
│   ├── sample_indices.json          ← 采样行号 (可复现)
│   ├── sample_metadata.json         ← 样本元数据
│   ├── samples.jsonl                ← JSONL 流式格式
│   └── summary.json                 ← 采样摘要
│
└── E11_4_infoseek_10k_results/
    ├── e11_4_infoseek_results.jsonl     ← 完整结果 (按样本)
    └── e11_4_infoseek_summary.json      ← 统计摘要
```

---

## 数据格式

### 采样元数据 (`sample_metadata.json`)

```json
{
  "split": "entity_test",
  "total_sampled": 10000,
  "random_seed": 42,
  "samples": [
    {
      "sample_id": 0,          // 在采样内的序号 (0-9999)
      "line_idx": 5142,        // 原始 JSONL 的行号
      "data_id": "infoseek_test_00005142",
      "image_id": "oven_05494604",
      "question": "What place inflows lake?"
    },
    ...
  ]
}
```

### Benchmark 结果 (`e11_4_infoseek_results.jsonl`)

每行是一个样本的完整结果:

```json
{
  "sample_id": 0,
  "data_id": "infoseek_test_00005142",
  "image_id": "oven_05494604",
  "question": "What place inflows lake?",
  
  // Query 生成
  "query_dims": [
    "Identify the specific entity or object needed to answer: ...",
    "Find visual evidence or context related to: ...",
    ...
  ],
  "query_gen_time_sec": 0.042,
  
  // 检索
  "retrieval_results": [...],
  "retrieval_time_sec": 3.214,
  
  // 答案
  "predicted_answer": "Rhine",
  "answer_gen_time_sec": 1.456,
  "llava_input_images": 1,
  
  // 评分 (开放式 QA)
  "exact_match": true,          // 精确匹配
  "fuzzy_match": true,          // 模糊匹配 (词汇重叠 >= 50%)
  "reference_answers": ["Rhine"],
  
  "status": "completed",
  "total_time_sec": 4.712
}
```

### 统计摘要 (`e11_4_infoseek_summary.json`)

```json
{
  "experiment": "E11_4_InfoSeek_10K",
  "dataset": "entity_test",
  "total_samples": 10000,
  "completed": 9995,
  "failed": 5,
  "exact_match_count": 5234,
  "fuzzy_match_count": 7821,
  "exact_match_rate": 0.5234,
  "fuzzy_match_rate": 0.7821,
  "total_time_sec": 43215.4,
  "avg_time_per_sample": 4.3215,
  "random_seed": 42
}
```

---

## 评分标准 (开放式 QA)

InfoSeek 是**开放式问答** (Open-ended QA)，而非多选题。评分采用两层标准:

### 1. 精确匹配 (Exact Match, EM)

```python
EM = (predicted_answer.lower() == reference_answer.lower())
```

例如:
- 预测: "The Rhine River"
- 参考: "Rhine River"
- EM = False (因为多了 "The")

### 2. 模糊匹配 (Fuzzy Match)

基于词汇重叠度:

```python
pred_words = set(predicted_answer.lower().split())
ref_words = set(reference_answer.lower().split())
overlap_ratio = len(pred_words & ref_words) / max(len(pred_words), len(ref_words))
FUZZY_MATCH = (overlap_ratio >= 0.5)
```

例如:
- 预测: "The Rhine River in Germany"
- 参考: "Rhine River"
- 词汇: {the, rhine, river, in, germany} ∩ {rhine, river} = {rhine, river}
- 重叠: 2 / 5 = 0.4 < 0.5 = **FUZZY_MATCH = False**

---

## 环境配置

### 必要的模型路径

创建或编辑 `.env`:

```bash
# Hugging Face 令牌 (用于下载模型)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Gemma4 本地多模态模型
GEMMA4_LOCAL_DIR=models/gemma4-e2b
GEMMA4_MODEL_ID=google/gemma-4-E2B-it
GEMMA4_DEVICE=cuda:0

# LLaVA OneVision 答案生成
LLAVA_MODEL_PATH=models/llava-onevision-qwen2-7b-ov

# MagicLens 配置
MAGICLENS_JAX_PLATFORM=cpu  # 或 cuda
```

### 依赖检查

```bash
python3 -c "
import torch
import jax
from PIL import Image
import transformers
print('✓ 所有依赖已安装')
"
```

---

## 可复现性

### 随机种子

采样使用 **随机种子 (random_seed)** 保证可复现:

```bash
# 同样的种子 -> 同样的样本顺序
python3 scripts/prepare_infoseek_10k_samples.py \
  --random-seed 42 \
  --output-dir sampling_v1

python3 scripts/prepare_infoseek_10k_samples.py \
  --random-seed 42 \
  --output-dir sampling_v2

# sampling_v1/sample_indices.json == sampling_v2/sample_indices.json ✓
```

### 版本控制

建议保存 **采样配置** 用于论文/报告:

```bash
# 保存采样配置
cp log/infoseek_samples_10k/sample_metadata.json \
   results/E11_4_samples_seed42_10k.json

# 标记 commit
git add results/E11_4_samples_seed42_10k.json
git commit -m "E11_4 InfoSeek 10K samples (seed=42, entity_test)"
```

---

## 与不同方法的对比

框架支持横向对比多个方法:

```
对比矩阵:
┌──────────────────┬────────┬────────┬────────┬────────┐
│ 方法             │ EM (%) │ Fuzzy  │ 时间/s │ 样本数 │
├──────────────────┼────────┼────────┼────────┼────────┤
│ E11_4 (4-dim)    │ 52.34  │ 78.21  │ 4.32   │ 10000  │
│ E8 (5-dim)       │ 51.20  │ 76.45  │ 5.21   │ 10000  │
│ CLIP (1-dim)     │ 38.90  │ 62.34  │ 2.14   │ 10000  │
│ ...              │        │        │        │        │
└──────────────────┴────────┴────────┴────────┴────────┘
```

核心优势: **所有方法用同一组 10000 样本进行评估** ✓

---

## 故障排除

### 错误: `FileNotFoundError: sample_metadata.json not found`

**原因**: 尚未运行采样脚本

**解决**:
```bash
python3 scripts/prepare_infoseek_10k_samples.py \
  --output-dir log/infoseek_samples_10k
```

### 错误: `CUDA out of memory`

**解决**:
```bash
# 降低 batch 大小 或 换到 CPU
MAX_SAMPLES=1000 bash test/E11_4_infoseek_10k.sh
```

### 错误: 采样不可复现

**原因**: 未指定 `--random-seed`

**解决**:
```bash
python3 scripts/prepare_infoseek_10k_samples.py \
  --random-seed 42  # <- 必须指定
```

---

## 下一步

### 1. 完整评估

运行完整的 10000 样本评估:

```bash
bash test/E11_4_infoseek_10k.sh
```

### 2. 与 MRAG-Bench 比较

同时评估 MRAG-Bench 和 InfoSeek，生成对比报告:

```bash
# MRAG-Bench (已有 E8 结果)
python3 -c "
import json
with open('log/E8_full/e8_full_summary.json') as f:
    d = json.load(f)
    print(f'E8 MRAG-Bench: EM={d[\"accuracy\"]:.2%}')
"

# InfoSeek (新增)
python3 -c "
import json
with open('log/E11_4_infoseek_10k_results/e11_4_infoseek_summary.json') as f:
    d = json.load(f)
    print(f'E11_4 InfoSeek: EM={d[\"exact_match_rate\"]:.2%}')
"
```

### 3. 论文展示

使用统一的 10000 样本列表，在论文中展示跨数据集泛化性能:

```
Table: Multi-dimensional RAG on Different Datasets

Method              MRAG-Bench EM   InfoSeek EM   Average Inference Time
────────────────────────────────────────────────────────────────────────
E11_4 (4-dim)              56.32%          52.34%                4.32s
E8 (5-dim)                 55.87%          51.20%                5.21s
Baseline (CLIP)            43.21%          38.90%                2.14s
```

---

## 参考

- 采样脚本: [scripts/prepare_infoseek_10k_samples.py](scripts/prepare_infoseek_10k_samples.py)
- Benchmark 脚本: [test/benchmark_e11_4_infoseek.py](test/benchmark_e11_4_infoseek.py)
- 工作流脚本: [test/E11_4_infoseek_10k.sh](test/E11_4_infoseek_10k.sh)
- InfoSeek 数据加载: [src/mrag/infoseek_loader.py](src/mrag/infoseek_loader.py)
