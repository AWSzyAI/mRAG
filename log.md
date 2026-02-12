# Work Log (2026-02-12)

## 1. Remote workflow / Makefile
- Added `sync` target using `rsync --exclude-from=.exclude` for local -> NNU sync.
- Added `cmd` target to run remote commands in `REMOTE_DIR` with conda env activation.
- `cmd` supports:
  - explicit command: `make cmd CMD='python main.py'`
  - compatibility mode: `make cmd nvidia-smi` (for old `mc` alias behavior)
  - fallback to previous command when no `CMD` is given.
- Added `alias` target that prints shell bootstrap:
  - `ms` -> `make sync`
  - `mc` function:
    - with args: `mc nvidia-smi` => run that command remotely
    - without args: execute previous local command remotely
- Added `config` target:
  - runs `ssh-copy-id -i ~/.ssh/id_rsa.pub NNU`
  - writes a managed `mRAG alias init` block into `~/.zshrc` (idempotent).

## 2. Exclude/sync behavior
- `.exclude` is now the single source for rsync excludes.
- Clarified behavior:
  - `--exclude` means “skip transfer of matched paths”
  - with current `sync` (`--delete` only), excluded remote paths are preserved.
- Added data-protection rules for remote-only large data:
  - `github/magiclens/data`
  - `github/magiclens/data/***`

## 3. Model eval scripts
- `github/MRAG-Bench/eval/models/run_model.sh`
  - made path handling robust (works from any CWD)
  - changed GPU default to `CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}`.
- `github/MRAG-Bench/eval/models/llava_one_vision.py`
  - set `attn_implementation="sdpa"` so `flash-attn` is optional.

## 4. Python entry script and HF access
- Fixed `main.py` shell syntax issue (`export ...` -> Python-compatible env usage).
- Set `HF_ENDPOINT` before importing `datasets` to avoid default `huggingface.co` path when mirror is needed.
- Current `main.py` uses:
  - `os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")`
  - then `from datasets import load_dataset`.

## 5. Dependency setup cleanup
- `requirements.txt` was normalized for pip format.
- Added build/runtime prerequisites for unstable env cases:
  - `numpy<2`, `setuptools<81`, `wheel`, `psutil`, `packaging`, `ninja`
- Kept `flash-attn` as a separate install step recommendation.

## 6. Key troubleshooting findings
- `nvidia-smi` command availability depends on execution host (local Mac vs remote Linux).
- Remote `conda activate` failed in non-interactive SSH until command strategy was fixed.
- HuggingFace failures were mainly network/DNS/proxy related on remote host:
  - bad/blocked connectivity to `huggingface.co`
  - mirror (`hf-mirror.com`) reachable.
- GitHub clone failures were tied to Clash selected node connectivity (`connect refused` on selected proxy).

## 7. Current recommended run path
1. `ms` (sync code to remote)
2. `mc nvidia-smi` (quick remote check)
3. `cd github/MRAG-Bench && conda activate llava && bash eval/models/run_model.sh && cd ../../`

## 8. Notes
- If aliases do not apply in current shell, run:
  - `eval "$(make -s alias)"`
- For new shells, `make config` already injects auto-bootstrap into `~/.zshrc`.

---

# 项目概述

## 项目简介

这是一个**多模态检索增强生成（mRAG）**研究项目，主要进行多模态RAG系统的评估和研究，使用UCLA的MRAG-Bench基准测试来评估视觉-语言模型的性能。

## 核心内容

### 1. 主要任务
- 测试和评估大型视觉-语言模型（LLaVA One Vision）在多模态检索任务上的表现
- 处理各种复杂的视觉场景：部分遮挡、生物图像、视角变化、形变、时序变化等

### 2. 技术架构
项目集成了三个主要的GitHub仓库：
- **MRAG-Bench**: UCLA的多模态RAG基准测试框架
- **LLaVA-NeXT**: 视觉-语言模型（用于推理）
- **MagicLens**: Google DeepMind的视觉模型项目

### 3. 开发工作流
由于NNU服务器不支持在线IDE工具（CodeX、Claude Code），采用本地+远程的混合开发模式：
- **本地（Mac）**: 编写和修改代码
- **远程（NNU服务器）**: 运行GPU密集型训练和推理
- **同步方式**: 通过rsync和Makefile自动化工具

主要命令：
- `ms` (make sync): 同步代码到服务器
- `mc` (make cmd): 在服务器上执行命令

### 4. 当前进展
已经跑出baseline结果：
- **整体准确率**: 60.31%
- 在不同场景下的表现：
  - 遮挡场景 (Obstruction): 66.67%
  - 部分可见 (Partial): 66.67%
  - 视角变化 (Angle): 60.25%
  - 不完整信息 (Incomplete): 30.39%（较低）

### 5. 解决的技术问题
- HuggingFace网络访问问题（使用国内镜像 hf-mirror.com）
- 远程conda环境激活
- Flash Attention依赖的可选配置（使用sdpa作为替代）
- Git clone的网络连接问题
- 本地Mac与远程Linux环境的命令兼容性

## 项目价值

这个项目属于**视觉-语言AI前沿研究**，专注于评估多模态模型在复杂视觉检索场景下的能力，对于理解和改进多模态RAG系统具有重要意义。
