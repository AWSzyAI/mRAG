# mRAG - 多模态检索增强生成评估项目

> **AI Assistant 请优先读取本文件**
> 最后更新: 2026-02-14

## AI 快速上下文（先看这里）

- 主开发端: `MacBook Air M2`（Git source of truth）。
- 服务器端: `AC/NNU`（运行环境，可临时改代码）。
- 当前会话常在服务器端，可能没有 `.git`；不要假设本机可直接 `git status`。
- 代码同步主命令:
  - `make sync` / `ms`: 本地 -> 服务器下发代码。
  - `make pull`: 预览拉取（默认只 dry-run，不落地）。
  - `make pull y`: 按 `pull_list.txt` 真正拉取。
  - `make pull result y` / `mr`: 按 `result.txt` 拉取结果文件。
- `make pull -y` 不可用（GNU make 会把 `-y` 识别成 make 参数）；使用尾参 `y`。
- 角色文件:
  - 本机: `.agent/ROLE.md`
  - 回流远端角色: `.agent/ROLE.<SYNC_HOST>.md`
  - `make config` 会自动初始化/同步 ROLE。

## M2 首次引导（当 M2 还是旧 Makefile）

如果 AC 上已更新 pull 机制，而 M2 还没有这些脚本，先在 M2 执行一次引导同步:

```bash
scp AC:/home/database/2025/mRAG/Makefile .
scp AC:/home/database/2025/mRAG/.alias .
scp AC:/home/database/2025/mRAG/pull_list.txt .
scp AC:/home/database/2025/mRAG/result.txt .
```

然后:

```bash
make pull
make pull y
```

## 🎯 项目目标

评估大型视觉-语言模型在多模态检索增强生成（mRAG）任务上的性能，使用 UCLA MRAG-Bench 基准测试。

## ⚡ 快速开始

```bash
# 1. 同步代码到服务器
ms

# 2. 运行评估
mc "cd github/MRAG-Bench && conda activate llava && bash eval/models/run_model.sh"

# 3. 查看结果
mc "cd github/MRAG-Bench && python eval/score.py -i llava_one_vision_gt_rag_results.jsonl"

# 4. 拉取回主开发端（默认预览）
make pull result
make pull result y
```

## 📊 当前状态

**Baseline 结果**: 60.31% 整体准确率

| 场景类型 | 准确率 |
|---------|--------|
| Obstruction (遮挡) | 66.67% |
| Partial (部分) | 66.67% |
| Scope | 63.73% |
| Temporal | 61.74% |
| Angle | 60.25% |
| Biological | 57.84% |
| Deformation | 56.86% |
| **Incomplete (待优化)** | **30.39%** ⚠️ |

## 🏗️ 技术栈

- **模型**: LLaVA One Vision (llava-onevision-qwen2-7b-ov)
- **基准测试**: MRAG-Bench (UCLA)
- **环境**: Python 3.10, PyTorch 2.1.2, CUDA 12.1
- **开发模式**: 本地 Mac + 远程 NNU 服务器 (GPU)

## 🔗 关键路径

| 类型 | 路径 |
|------|------|
| 评估脚本 | `github/MRAG-Bench/eval/models/run_model.sh` |
| 模型配置 | `github/MRAG-Bench/eval/models/llava_one_vision.py` |
| 评分脚本 | `github/MRAG-Bench/eval/score.py` |
| 远程模型 | `/home/user/.cache/huggingface/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov` |
| 远程数据集 | `/home/user/.cache/huggingface/datasets/uclanlp___mrag-bench` |

## 📝 当前任务

- [ ] 分析 Incomplete 场景低准确率原因
- [ ] 测试 MagicLens 模型集成
- [ ] 优化推理速度

## 🔍 下一步

详细信息请参考:
- 架构设计 → `ARCHITECTURE.md`
- 技术决策 → `DECISIONS.md`
- 工作日志 → `log.md`
