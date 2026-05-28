# E11_4 InfoSeek 10K - 快速开始指南

## 目标

在 InfoSeek 上用 **E11_4** (4维多维度Query分解) 方法评估 **10000 个样本**，得到完整的检索+答案+评分结果。

## 一键启动

```bash
cd /mnt/d/mRAG

# 完整运行（采样 + Benchmark）
bash test/E11_4_infoseek_10k.sh

# 查看结果
cat log/E11_4_infoseek_10k/benchmark/e11_4_infoseek_summary.json
tail -5 log/E11_4_infoseek_10k/benchmark/e11_4_infoseek_results.jsonl
```

## 在 nnu 上运行

从本地直接发起远端运行：

```bash
ssh nnu 'cd /home/user/code/mRAG && \
  source $(conda info --base)/etc/profile.d/conda.sh && \
  conda activate /home/user/env/envs/llava && \
  SAMPLE_SIZE=10000 SPLIT=entity_test RANDOM_SEED=42 \
  OUTPUT_DIR=log/E11_4_infoseek_10k \
  bash test/E11_4_infoseek_10k.sh'
```

如果已经登录到 nnu：

```bash
cd /home/user/code/mRAG
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/user/env/envs/llava

SAMPLE_SIZE=10000 SPLIT=entity_test RANDOM_SEED=42 \
OUTPUT_DIR=log/E11_4_infoseek_10k \
bash test/E11_4_infoseek_10k.sh
```

快速试跑可加 `MAX_SAMPLES=100`：

```bash
MAX_SAMPLES=100 SAMPLE_SIZE=10000 SPLIT=entity_test RANDOM_SEED=42 \
OUTPUT_DIR=log/E11_4_infoseek_10k_smoke \
bash test/E11_4_infoseek_10k.sh
```

远端跑完后，在本地拉回结果：

```bash
rsync -az nnu:/home/user/code/mRAG/log/E11_4_infoseek_10k/ \
  ./log/E11_4_infoseek_10k/
```

## 核心命令

### 1. 只生成采样列表（可复现）

```bash
python3 scripts/prepare_infoseek_10k_samples.py \
  --data-root data/infoseek \
  --split entity_test \
  --sample-size 10000 \
  --random-seed 42 \
  --output-dir log/infoseek_samples_10k
```

**输出:**
- `log/infoseek_samples_10k/sample_indices.json` ← 采样索引（可复现）
- `log/infoseek_samples_10k/sample_metadata.json` ← 样本详细信息
- `log/infoseek_samples_10k/samples.jsonl` ← JSONL 格式

### 2. 运行 Benchmark

```bash
python3 test/benchmark_e11_4_infoseek.py \
  --sample-dir log/infoseek_samples_10k \
  --image-dir data/infoseek/images/all \
  --output-dir log/E11_4_infoseek_results
```

**输出:**
- `log/E11_4_infoseek_results/e11_4_infoseek_results.jsonl` ← 逐样本结果
- `log/E11_4_infoseek_results/e11_4_infoseek_summary.json` ← 统计摘要

## 测试运行（快速验证）

```bash
# 仅处理 100 个样本用于快速测试
MAX_SAMPLES=100 bash test/E11_4_infoseek_10k.sh
```

## 查看结果

### 查看统计摘要

```bash
python3 -c "
import json
with open('log/E11_4_infoseek_10k/benchmark/e11_4_infoseek_summary.json') as f:
    stats = json.load(f)
    print(f'完成: {stats[\"completed\"]}/{stats[\"total_samples\"]}')
    print(f'精确匹配: {stats[\"exact_match_rate\"]:.2%}')
    print(f'模糊匹配: {stats[\"fuzzy_match_rate\"]:.2%}')
    print(f'平均耗时: {stats[\"avg_time_per_sample\"]:.2f}s/sample')
"
```

### 查看单个样本

```bash
# 查看前 3 个样本的结果
head -3 log/E11_4_infoseek_10k/benchmark/e11_4_infoseek_results.jsonl | \
  python3 -m json.tool
```

### 查看采样列表

```bash
# 查看采样的样本信息
python3 -c "
import json
with open('log/infoseek_samples_10k/sample_metadata.json') as f:
    data = json.load(f)
    for sample in data['samples'][:5]:
        print(f\"{sample['sample_id']:4d}: {sample['data_id']:30s} {sample['question'][:50]}\")
"
```

## 与其他方法对比

由于所有方法使用**同一采样列表**（seed=42），可以直接对比：

```bash
# 并行运行多个方法
# E11_4 (4-dim)
SAMPLE_SIZE=10000 bash test/E11_4_infoseek_10k.sh

# E8 (5-dim) - 如果有现成脚本
bash test/E8_infoseek_10k.sh

# 生成对比表格
python3 - <<'PY'
import json
methods = [
    ("E11_4 (4-dim)", "log/E11_4_infoseek_10k/benchmark/e11_4_infoseek_summary.json"),
    ("E8 (5-dim)", "log/E8_infoseek_10k/benchmark/e8_infoseek_summary.json"),
]
print("方法\t\tEM\t\tFuzzy\t\t平均耗时")
print("-" * 50)
for name, path in methods:
    try:
        with open(path) as f:
            d = json.load(f)
            em = d.get("exact_match_rate", 0)
            fm = d.get("fuzzy_match_rate", 0)
            avg_time = d.get("avg_time_per_sample", 0)
            print(f"{name:15s}\t{em:.1%}\t\t{fm:.1%}\t\t{avg_time:.2f}s")
    except:
        print(f"{name:15s}\tN/A\t\tN/A\t\tN/A")
PY
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sample-size` | 10000 | 采样大小 |
| `--split` | entity_test | InfoSeek 分割 (entity_test/entity_train/entity_val/human) |
| `--random-seed` | 42 | 随机种子（保证可复现） |
| `--max-samples` | None | Benchmark 时仅处理前 N 个样本（用于测试） |

## 环境依赖

如果 nnu 无法联网，优先按 [nnu 离线环境与代码同步规范](NNU_OFFLINE_ENV_SYNC.md) 操作。简要流程：先在本地把 `llava` conda 环境按 README/实验需求装好，再从项目根目录同步到服务器：

```bash
# 预览目标路径
make sync-env

# 打包本地 llava 环境并上传到 nnu:/home/user/env/envs/llava
make sync-env y

# 如果远端 llava 已存在且确认要替换
make sync-env y ENV_REPLACE=1

# 验证 torch/llava/scenic 等关键 import
make env-smoke
```

本地环境里需要先有 `conda-pack`：

```bash
conda activate llava
python -m pip install conda-pack
```

必须在 `.env` 中配置（或使用默认）：

```bash
# HuggingFace 令牌
HF_TOKEN=hf_xxxxxxxxxxxxx

# Gemma4 模型
GEMMA4_LOCAL_DIR=models/gemma4-e2b
GEMMA4_DEVICE=cuda:0

# LLaVA 模型
LLAVA_MODEL_PATH=models/llava-onevision-qwen2-7b-ov
```

## 常见问题

### Q: 如何确保采样是可复现的？
A: 指定 `--random-seed 42`，多次运行会得到相同的采样顺序。

### Q: 如何使用不同的采样？
A: 改变 `--random-seed` 参数，例如 `--random-seed 2024`。

### Q: 如何对比不同的方法？
A: 所有方法都使用 `log/infoseek_samples_10k/` 中的同一采样列表，直接对比结果即可。

### Q: 可以扩展到 20K 或 100K 样本吗？
A: 可以，改变 `--sample-size` 参数即可。

## 输出文件参考

```
log/E11_4_infoseek_10k/
├── sampling/
│   ├── sample_indices.json          ← 采样行号 (可复现)
│   ├── sample_metadata.json         ← 样本元数据
│   ├── samples.jsonl                ← JSONL 格式
│   └── summary.json                 ← 采样摘要
│
├── benchmark/
│   ├── e11_4_infoseek_results.jsonl ← 完整结果 (每行一个样本)
│   └── e11_4_infoseek_summary.json  ← 统计摘要
│
└── REPORT.txt                       ← 运行报告
```

## 后续步骤

1. ✓ 准备 10000 样本采样列表
2. ✓ 创建 E11_4 InfoSeek benchmark 脚本
3. ⬜ 集成完整的 MagicLens + LLaVA 推理流程
4. ⬜ 与 MRAG-Bench 结果进行对比
5. ⬜ 生成论文表格

---

**相关文档**: [README/E11_4_INFOSEEK_10K_FRAMEWORK.md](README/E11_4_INFOSEEK_10K_FRAMEWORK.md)
