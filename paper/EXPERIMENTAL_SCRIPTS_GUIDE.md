# 实验脚本与执行指南

## 一、当前可用的实验脚本

### 1. 基准测试框架

#### `test/benchmark_corpus_rag.py`
**用途**：完整 RAG 评估，支持多种检索器和重排器

**基础用法**：
```bash
cd /Users/szy/Downloads/mRAG

# 快速验证（50 样本）
MAX_SAMPLES=50 bash test/benchmark_corpus_rag.sh

# 完整评估（1353 样本，需 2-3 小时）
MAX_SAMPLES=0 bash test/benchmark_corpus_rag.sh
```

**主要参数**：
```python
--corpus-dir           # 图像语料库路径
--retriever-type       # 检索器类型: clip / magiclens
--magiclens-model-size # base / large
--top-k                # 检索返回数量，建议 5
--llava-greedy-jsonl   # 用于对照的 greedy 结果
```

**输出文件**：
- 预测结果：`log/{EID}/{EID}_*_results.jsonl`
- 运行摘要：`log/{EID}/{EID}_*_summary.json`
- 分数统计：`results/{EID}_*_results_score.json`

---

### 2. 多维度查询分解脚本（Phase D）

#### `test/pipeline_multi_dim_rag.py`（主入口）
**用途**：多维度查询分解 RAG 核心实现（`test/benchmark_multi_dimension_rag.py` 为兼容包装）

**关键特性**：
- 支持 API 和本地模型两种维度生成方式
- 支持 RRF、Score-sum、Voting 三种融合策略
- 输出与 E3/E7 格式兼容
- 额外输出维度指令 JSONL（方便审查）

**使用示例**：

```bash
# 方案 D01: Qwen2.5-7B + 3维 + RRF
DIM_GENERATOR_API_KEY="sk-xxxx" \
python test/pipeline_multi_dim_rag.py \
  --corpus-dir /Users/szy/Downloads/mRAG/data/image_corpus \
  --dim-generator-type api \
  --dim-generator-model qwen2.5-7b-instruct \
  --dim-generator-api-base https://api.siliconflow.cn/v1 \
  --n-dims 3 \
  --fusion-strategy rrf \
  --max-samples 0 \
  --corpus-cache-dir results/corpus_index \
  --answers-file log/D01/d01_*.jsonl \
  --summary-out log/D01/d01_*_summary.json
```

**维度生成模型 API 密钥获取**：

| 平台 | 地址 | 免费额度 | 获取方式 |
|------|------|---------|---------|
| **SiliconFlow** | https://api.siliconflow.cn/v1 | $5 新用户 | 微信登录 |
| OpenRouter | https://openrouter.ai/api/v1 | - | 支持支付宝 |
| DeepInfra | https://api.deepinfra.com/v1 | - | 信用卡 |

推荐使用 **SiliconFlow**，因为：
- 国内延迟低
- Qwen2.5-7B 成本极低（$0.05/M tokens）
- 1353 样本 × 3 维 ≈ 4K 调用 ≈ $0.2

**注意事项**：
- 首次运行需构建语料库索引（~5 分钟）
- API 超时设置为 30 秒，如遇网络不稳定可调整
- 输出维度指令 JSONL 可用于质量审查

---

### 3. 批量扫描脚本（Phase D 完整扫描）

#### `test/E8_multi_dim_sweep.sh`
**用途**：自动运行完整的 Phase D 消融实验（D01-D12）

**用法**：
```bash
# 完整扫描（所有 1353 样本）
DIM_GENERATOR_API_KEY="sk-xxxx" bash test/E8_multi_dim_sweep.sh

# 快速验证（每组 50 样本）
DIM_GENERATOR_API_KEY="sk-xxxx" MAX_SAMPLES=50 bash test/E8_multi_dim_sweep.sh

# 指定 API 平台
DIM_GENERATOR_API_KEY="sk-xxxx" \
DIM_GENERATOR_API_BASE="https://api.siliconflow.cn/v1" \
bash test/E8_multi_dim_sweep.sh
```

**扫描配置**（自动执行以下 12 组实验）：

```
D01: Qwen2.5-7B, 3维, 5+5, RRF           (基准)
D02: Qwen2.5-3B, 3维, 5+5, RRF           (消融模型)
D03: DeepSeek-R1, 3维, 5+5, RRF          (消融推理)
D04: Llama-3.1-8B, 3维, 5+5, RRF         (消融语言)
D05: Phi-3.5-mini, 3维, 5+5, RRF         (消融大小)
D06: Qwen2.5-7B, 1维, 5+5, RRF           (消融维度下限)
D07: Qwen2.5-7B, 5维, 5+5, RRF           (消融维度上限)
D08: Qwen2.5-7B, 3维, 5+5, Score-sum     (消融融合)
D09: Qwen2.5-7B, 3维, 5+5, Voting        (消融融合)
D10: Qwen2.5-7B, 3维, 3+5, RRF           (消融检索K)
D11: Qwen2.5-7B, 3维, 10+5, RRF          (消融检索K)
D12: Qwen2.5-7B, 3维, 5+3, RRF           (消融融合K)
```

**输出**：
- `log/E8/sweep_results.csv` - 12 组实验的精度对比
- `log/E8/{D01-D12}/` - 各实验详细日志

**预期运行时间**：
- 快速验证（50 样本）：~30 分钟
- 完整扫描（1353 样本）：~18-24 小时

---

## 二、服务器执行指南

### 1. 环境配置

**前提检查**：
```bash
# SSH 到服务器
ssh AB

# 检查 CUDA 可用性
nvidia-smi

# 查看当前显存占用
nvidia-smi | grep MiB

# 查看已安装的 Python 包
conda list | grep -E "torch|clip|llava"
```

**激活 Conda 环境**：
```bash
conda activate mrag
# 或
source activate mrag
```

### 2. 远程执行示例

#### 执行 E6（已完成，用于参考）
```bash
ssh AB
cd /public/home/hzh/mRAG

# 直接运行（无输出）
bash test/E6.sh

# 后台运行并重定向日志
nohup bash test/E6.sh > results/E6.log 2>&1 &

# 实时监控日志
tail -f results/E6.log
```

#### 执行 Phase D01（首先推荐）
```bash
ssh AB
cd /public/home/hzh/mRAG

# 设置 API 密钥
export DIM_GENERATOR_API_KEY="sk-xxxxxx"

# 运行 D01
nohup python test/pipeline_multi_dim_rag.py \
  --corpus-dir /public/home/hzh/mRAG/data/image_corpus \
  --dim-generator-type api \
  --dim-generator-model qwen2.5-7b-instruct \
  --dim-generator-api-base https://api.siliconflow.cn/v1 \
  --n-dims 3 \
  --fusion-strategy rrf \
  --max-samples 0 \
  --answers-file /public/home/hzh/mRAG/log/D01/d01_results.jsonl \
  --summary-out /public/home/hzh/mRAG/log/D01/d01_summary.json \
  > /public/home/hzh/mRAG/log/D01/d01.log 2>&1 &

# 监控进度
tail -f /public/home/hzh/mRAG/log/D01/d01.log
```

### 3. 显存管理

**清空缓存**：
```bash
# Python 中清空 CUDA 缓存
python -c "import torch; torch.cuda.empty_cache()"

# 或在脚本中设置环保变量
export MAGICLENS_CLEAR_CACHE_EVERY=200  # 每 200 样本清一次
```

**多 GPU 分布**：
```bash
# 分配特定 GPU
CUDA_VISIBLE_DEVICES=0,1 python test/benchmark_...

# 禁用 GPU 用 CPU（调试用）
JAX_PLATFORMS=cpu python test/benchmark_...
```

---

## 三、结果收集与分析

### 1. 从服务器拉取结果

```bash
# 拉取整个 log 文件夹
rsync -avz AB:/public/home/hzh/mRAG/log/ ./log/

# 拉取特定实验
rsync -avz AB:/public/home/hzh/mRAG/log/D01/ ./log/D01/

# 拉取 results 文件夹
rsync -avz AB:/public/home/hzh/mRAG/results/ ./results/
```

### 2. 评分与对比

```bash
# 用官方 score.py 评分
cd github/MRAG-Bench
python eval/score.py \
  --answers-file ../../log/D01/d01_results.jsonl \
  --dataset-name uclanlp/MRAG-Bench
```

**输出格式**：
```json
{
  "accuracy": 52.34,
  "by_scenario_accuracy": {
    "Scope": 57.8,
    "Deformation": 59.2,
    ...
  }
}
```

### 3. 实验对比统计

```bash
# 生成对比表格
python -c "
import json
import pandas as pd

results = {}
for exp_id in ['E3', 'E6', 'E7', 'D01', 'D02', ...]:
    with open(f'log/{exp_id}/{exp_id}_*_summary.json') as f:
        results[exp_id] = json.load(f)

df = pd.DataFrame(results).T
print(df[['accuracy', 'reference_accuracy', 'by_scenario_accuracy']])
"
```

---

## 四、故障排查

### 常见问题

#### 问题 1：JAX CUDA OOM
**症状**：`RuntimeError: CUDA out of memory`

**解决**：
```bash
# 方案 A: 禁用 JAX CUDA，用 CPU
JAX_PLATFORMS=cpu MAX_SAMPLES=50 bash test/benchmark_magiclens.sh

# 方案 B: 减小 batch size
MAGICLENS_BATCH_SIZE=8 bash test/benchmark_magiclens.sh

# 方案 C: 定期清空缓存
MAGICLENS_CLEAR_CACHE_EVERY=100 bash test/benchmark_magiclens.sh
```

#### 问题 2：LLaVA 显存爆炸
**症状**：LLaVA 初始化时显存溢出

**解决**：
```bash
# 切换到 4-bit 量化
LLAVA_LOAD_4BIT=1 bash test/benchmark_magiclens.sh

# 或选择 SDPA attention
LLAVA_ATTN_IMPLEMENTATION=sdpa bash test/benchmark_magiclens.sh

# 最激进：限制 token 生成
LLAVA_MAX_NEW_TOKENS=64 bash test/benchmark_magiclens.sh
```

#### 问题 3：API 超时
**症状**：`TimeoutError: API request timeout`

**解决**：
```bash
# 增加超时时间
DIM_GENERATOR_API_TIMEOUT=60 bash test/E8_multi_dim_sweep.sh

# 使用本地模型替代
# (见下一部分)
```

### 本地模型替代方案

如果 API 不稳定，可用本地 Qwen2.5-VL-7B（需要 GPU）：

```bash
# 下载模型
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./models/qwen2.5-vl-7b

# 修改脚本使用本地模型
DIM_GENERATOR_TYPE="local" \
DIM_GENERATOR_MODEL_PATH="./models/qwen2.5-vl-7b" \
bash test/E8_multi_dim_sweep.sh
```

---

## 五、快速参考：重要命令

```bash
# 查看 GPU 状态
nvidia-smi

# 查看进程
ps aux | grep python

# 杀死进程
kill -9 <PID>

# 查看实时日志
tail -f log/D01/d01.log

# 统计样本数
wc -l log/D01/d01_results.jsonl

# 提取精度
cat log/D01/d01_summary.json | jq '.accuracy'

# 比较两次运行
diff <(jq '.by_scenario_accuracy' log/E3/summary.json | sort) \
     <(jq '.by_scenario_accuracy' log/D01/d01_summary.json | sort)
```

---

## 六、下一步建议

### 第 1 周：D01 基准验证
1. 获取 SiliconFlow API 密钥
2. 运行 D01（3维 Qwen2.5-7B + RRF）
3. 导出 50 个维度指令样本，人工审查质量

### 第 2 周：消融实验 D02-D07
1. 运行模型对比（D02-D05）
2. 运行维度数对比（D06-D07）
3. 汇总精度对比表

### 第 3 周：融合策略与参数优化
1. 运行策略对比（D08-D09）
2. 运行参数扫描（D10-D12）
3. 确定最优配置

### 第 4 周：第二数据集迁移
1. InfoSeek 图像语料库构建
2. 迁移最优配置（D_best）
3. 交叉数据集验证

