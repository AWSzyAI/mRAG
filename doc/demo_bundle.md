# 单题 Demo 展示说明

当前仓库的单题 demo 分为两类：

1. 旧版：
   - `log/DEMO：做一道题/compare_one.sh`
   - 主要比较一条样本上的 `LLaVA greedy / beam / MagicLens`
   - 更偏调试，不是完整 pipeline 展示

2. 新版：
   - `test/demo_E4.sh`
   - `test/demo_E2.sh`
   - `test/demo_E3.sh`
   - `test/demo_E7.sh`
   - 用于从现有实验结果中导出“评审展示包”

## 展示包内容

每次导出会在 `log/demo_review/<pipeline>/sample*/` 下生成：

- `query.png`
- 检索图片
- `prompts.json`
- `result_row.json`
- `report.json`
- `report.md`
- 对应实验的 `summary.json`（如果存在）
- 对应实验的 `log`（如果存在）

## 用法

默认导出第 0 条：

```bash
bash test/demo_E4.sh
bash test/demo_E2.sh
bash test/demo_E3.sh
bash test/demo_E7.sh
```

导出指定样本：

```bash
SAMPLE_INDEX=12 bash test/demo_E3.sh
SAMPLE_ID=128 bash test/demo_E7.sh
```

## 说明

- `E4`：展示无检索基线
- `E2`：展示数据集自带 retrieved images + MagicLens rerank
- `E3`：展示 `CLIP` 从完整 corpus 的检索结果
- `E7`：展示 `MagicLens` 从完整 corpus 的检索结果

这样可以直观看到不同 pipeline 在同一道题上的：

- 输入图片
- 检索图片
- LLaVA instruction
- MagicLens query instruction
- 最终答案
- 原始日志与摘要
