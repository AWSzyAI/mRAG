# MagicLens Rerank + LLaVA 版 MRAG-Bench 运行说明

## 1. 脚本清单

| 脚本 | 用途 | 是否直接运行 |
|---|---|---|
| `test/benchmark_magiclens.sh` | 一键跑：MagicLens 重排 RAG + LLaVA 最终作答，并自动调用 `score.py` | 是（推荐） |
| `test/benchmark_magiclens.py` | 核心评测逻辑：MagicLens 只做检索重排，LLaVA 输出最终答案 | 否（可直接调试） |
| `test/compare_one.sh` | 单样本对比（LLaVA greedy/beam + MagicLens） | 是 |
| `test/compare_one_sample.py` | 单样本核心逻辑 | 否 |
| `github/MRAG-Bench/eval/score.py` | 官方评分脚本 | 是 |

## 2. 依赖前提

1. 当前目录在项目根：`/public/home/hzh/mRAG`（或你的本地仓库根目录）。
2. 可用 Python 环境里有以下依赖：`jax`、`scenic`、`torch`、`datasets`、`LLaVA-NeXT`、`magiclens` 依赖。
3. 模型文件存在：
   - `models/magic_lens_clip_base.pkl`（或 large）
   - `models/bpe_simple_vocab_16e6.txt.gz`（可选，存在时自动使用）
4. 数据集可访问（在线或本地缓存）：`uclanlp/MRAG-Bench`。

## 3. 一键跑（MagicLens 重排 + LLaVA 回答）

```bash
bash test/benchmark_magiclens.sh
```

默认行为：

1. 使用 `base` MagicLens：`models/magic_lens_clip_base.pkl`
2. 使用 LLaVA：`models/llava-onevision-qwen2-7b-ov`
3. 跑完整测试集（`MAX_SAMPLES=0`）
4. 输出答案文件：`github/MRAG-Bench/magiclens_rerank_llava_results.jsonl`
5. 自动执行：
   - `cd github/MRAG-Bench`
   - `python eval/score.py -i <answers_file>`

## 4. 常用运行方式

### 4.1 先跑 20 条 smoke test

```bash
MAX_SAMPLES=20 bash test/benchmark_magiclens.sh
```

### 4.2 从某个样本开始续跑

```bash
START_INDEX=300 bash test/benchmark_magiclens.sh
```

### 4.3 切到 large 模型

```bash
MAGICLENS_MODEL_SIZE=large \
MAGICLENS_MODEL_PATH=./models/magic_lens_clip_large.pkl \
bash test/benchmark_magiclens.sh
```

### 4.4 控制使用的 RAG 图像数量

```bash
MAX_RAG_IMAGES=1 bash test/benchmark_magiclens.sh
```

### 4.5 与 LLaVA greedy 结果对比一致率/精度差

```bash
LLAVA_GREEDY_JSONL=./github/MRAG-Bench/llava_one_vision_gt_rag_results.jsonl \
bash test/benchmark_magiclens.sh
```

## 5. 关键环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MAGICLENS_MODEL_SIZE` | `base` | `base` 或 `large` |
| `MAGICLENS_MODEL_PATH` | `./models/magic_lens_clip_base.pkl` | MagicLens 权重路径 |
| `MAX_RAG_IMAGES` | `5` | 每条样本最多保留多少张 RAG 图（默认对齐 baseline） |
| `START_INDEX` | `0` | 从第几条样本开始跑 |
| `MAX_SAMPLES` | `0` | 最多跑多少条，`0`=全量 |
| `LLAVA_MODEL_PATH` | `./models/llava-onevision-qwen2-7b-ov` | 最终回答用的 MLLM |
| `LLAVA_MAX_NEW_TOKENS` | `4096` | LLaVA 解码长度 |
| `LLAVA_NUM_BEAMS` | `1` | LLaVA beam 数（默认 greedy） |
| `DISABLE_MAGICLENS_RERANK` | `0` | 设为 `1` 时禁用重排（可做 ablation） |
| `ANSWERS_FILE` | `./github/MRAG-Bench/magiclens_rerank_llava_results.jsonl` | 预测输出 |
| `SUMMARY_OUT` | `./log/magiclens_rerank_llava_summary.json` | 运行摘要 |
| `LLAVA_GREEDY_JSONL` | `./github/MRAG-Bench/llava_one_vision_gt_rag_results.jsonl` | 可选对比基线 |
| `JAX_PLATFORMS` | `cuda` | `cuda` 或 `cpu` |
| `JAX_CUDA_REQUIRED` | `0` | `1` 时 CUDA 不可用直接失败；`0` 时自动回退 CPU |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | 设为 `true` 可减少分配开销，提升速度 |
| `XLA_PYTHON_CLIENT_ALLOCATOR` | `platform` | 设为 `default` 通常更快但更占显存 |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `<unset>` | 预分配显存比例（配合 `PREALLOCATE=true`） |

## 6. 输出文件位置

1. 预测结果：`github/MRAG-Bench/magiclens_rerank_llava_results.jsonl`
2. 运行摘要：`log/magiclens_rerank_llava_summary.json`
3. 评分结果（`score.py` 输出）：`github/MRAG-Bench/results/*.json`

## 7. 单样本对比（调试）

```bash
bash test/compare_one.sh
```

输出：

1. `log/one_sample_compare/sample*/report.md`
2. `log/one_sample_compare/sample*/report.json`

说明：单样本脚本用于调试与可解释性分析，不替代全量 benchmark。

## 8. 常见报错（JAX CUDA 初始化失败）

典型报错关键词：

- `cuInit(0) failed`
- `JAX was unable to load the CUDA libraries`

处理方式：

1. 在 GPU 计算节点执行（不要在无 GPU 的登录节点跑）。
2. 若只是先验证流程，可强制 CPU：

```bash
JAX_PLATFORMS=cpu MAX_SAMPLES=20 bash test/benchmark_magiclens.sh
```

3. 如果你希望“必须用 GPU，否则直接报错退出”：

```bash
JAX_CUDA_REQUIRED=1 bash test/benchmark_magiclens.sh
```

## 9. 共享 GPU 的提速建议

你这种“同卡上还有同学在跑”的情况，建议用温和配置：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_ALLOCATOR=default \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 \
bash test/benchmark_magiclens.sh
```

说明：

1. `0.20` 在 24GB 卡上大约预留 4.9GB 给当前进程。
2. 若对方负载稳定、你这边仍有空闲，可试 `0.25`。
3. 如果出现显存争抢或 OOM，就降回 `0.15` 或恢复默认配置。
