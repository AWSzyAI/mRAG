# MagicLens Benchmark TODO (2026-03-05)

> 当前状态：MagicLens 全量结果尚未正式跑完。以下为可直接执行的测试脚本。

- 显存占用：44405MiB /  81920MiB
- 脚本结束后会自动调用 `github/MRAG-Bench/eval/score.py`。

## 指定 GPU / 显存策略（共享 GPU 场景）

```bash
CUDA_VISIBLE_DEVICES=1 \
JAX_PLATFORMS=cpu \
LLAVA_DEVICE_MAP=single \
LLAVA_LOAD_4BIT=0 \
LLAVA_LOAD_8BIT=0 \
LLAVA_ATTN_IMPLEMENTATION=sdpa \
LLAVA_MAX_NEW_TOKENS=64 \
MAGICLENS_DISABLE_JIT=1 \
MAGICLENS_CLEAR_CACHE_EVERY=200 \
bash test/benchmark_magiclens.sh > results/benchmark_magiclens_gpu1.log

```

## 对照实验（禁用 MagicLens 重排，仅 LLaVA 原顺序）

```bash
cd /Users/szy/Downloads/project/mRAG
DISABLE_MAGICLENS_RERANK=1 MAX_SAMPLES=0 bash test/benchmark_magiclens.sh
```

用途：验证提升/下降是否来自 MagicLens 重排本身。

## 5) 可选：切换 large 权重

```bash
cd /Users/szy/Downloads/project/mRAG
MAGICLENS_MODEL_SIZE=large \
MAGICLENS_MODEL_PATH=/Users/szy/Downloads/project/mRAG/models/magic_lens_clip_large.pkl \
MAX_SAMPLES=0 \
bash test/benchmark_magiclens.sh
```

## 6) 关键输出文件

- 预测结果：`github/MRAG-Bench/magiclens_rerank_llava_results.jsonl`
- 运行摘要：`log/magiclens_rerank_llava_summary.json`
- 评分输出：终端打印（来自 `github/MRAG-Bench/eval/score.py`）

## 7) 若 JAX CUDA 报错（先保流程）

```bash
cd /Users/szy/Downloads/project/mRAG
JAX_PLATFORMS=cpu MAX_SAMPLES=20 bash test/benchmark_magiclens.sh
```
