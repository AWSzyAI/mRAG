# M2 - Ocean-NAT - node01 三段式工作流

## 1. 背景

当前项目采用三段式结构来满足以下现实约束：

- `M2`：主开发端，负责改代码和 Git 管理。
- `Ocean-NAT`：可联网节点，负责下载模型与数据集。
- `node01`：GPU 运行节点（RTX 4090），用于推理评估，但网络受限/不可联网。

目标是让 `node01` 在离线条件下稳定运行 MRAG-Bench，不再在运行时触发 HuggingFace 下载。

## 2. 三节点角色定义

### 2.1 M2（主开发端）

- 角色：`source of truth`
- 职责：
  - 编辑代码、维护脚本和文档
  - 通过 `make sync` 下发代码到服务器
  - 通过 `make pull` 回收结果与远端改动

### 2.2 Ocean-NAT（联网下载端）

- 角色：`asset prefetch node`
- 职责：
  - 执行 `main.py` / `test/data_models.sh`
  - 预下载并缓存全部离线运行所需资产：
    - 主模型：`lmms-lab/llava-onevision-qwen2-7b-ov`
    - vision tower：`google/siglip-so400m-patch14-384`
    - 数据集：`uclanlp/MRAG-Bench`（`test` split）

### 2.3 node01（离线 GPU 运行端）

- 角色：`offline runtime node`
- 职责：
  - 仅使用本地缓存与本地模型执行推理
  - 不做联网下载（`MRAG_HF_OFFLINE=1`）
  - 产出结果文件供主开发端回拉

## 3. 目录与缓存约定

统一约定在项目根目录下使用以下路径（避免多套 cache）：

- 模型目录：`./models/llava-onevision-qwen2-7b-ov`
- HF 根目录：`./models/huggingface-mrag`
- HF Hub 缓存：`./models/huggingface-mrag/hub`
- HF Datasets 缓存：`./models/huggingface-mrag/datasets`

对应脚本约定：

- 预下载：`main.py --hf-home ./models/huggingface-mrag`
- 评估：`MRAG_HF_HOME="$PWD/../../models/huggingface-mrag"`
- 离线：`MRAG_HF_OFFLINE=1`

## 4. 标准执行流程

### 步骤 A：M2 同步代码到服务器

```bash
cd /path/to/mRAG
make sync
```

### 步骤 B：Ocean-NAT 预下载资产（联网）

```bash
cd /public/home/hzh/mRAG
bash test/data_models.sh
```

或显式执行：

```bash
python main.py \
  --model-local-dir ./models/llava-onevision-qwen2-7b-ov \
  --hf-home ./models/huggingface-mrag \
  --hf-endpoint https://hf-mirror.com
```

### 步骤 C：node01 离线评估

```bash
cd /public/home/hzh/mRAG
conda activate llava
MRAG_HF_OFFLINE=1 bash test/baseline.sh
```

beam 版本：

```bash
MRAG_HF_OFFLINE=1 bash test/beam_5.sh
```

### 步骤 D：结果回传到 M2

```bash
make pull result
make pull result y
```

## 5. 当前需求与落地策略

### 5.1 需求

- `node01` 运行时不访问外网。
- 所有下载动作集中在 `Ocean-NAT`，由 `main.py` 完成。
- 缓存路径统一，避免“下载在 A 路径、运行读 B 路径”。

### 5.2 已落地策略

- `main.py` 已支持模型 + vision tower + 数据集一站式预下载。
- `run_model.sh` / `baseline.sh` / `beam_5.sh` 已对齐 `MRAG_HF_HOME=./models/huggingface-mrag`。
- `MRAG_HF_OFFLINE=1` 时，运行脚本会走离线逻辑并禁止在线依赖。

## 6. 常见问题与处理

### 6.1 `CondaError: Run 'conda init' before 'conda activate'`

触发场景：非交互 shell 直接 `conda activate`。  
处理：使用 `test/data_models.sh`（已内置 `conda shell.bash hook`）或手动执行：

```bash
eval "$(conda shell.bash hook)"
conda activate llava
```

### 6.2 `NonMatchingSplitsSizesError`（数据集缓存不一致）

触发场景：历史中断或损坏缓存。  
处理：`main.py` 已加入自动清理并重试；若仍失败，手动清理：

```bash
rm -rf ./models/huggingface-mrag/datasets/uclanlp___mrag-bench*
```

然后在 Ocean-NAT 重新跑 `bash test/data_models.sh`。

### 6.3 离线节点仍尝试联网

优先检查：

- 是否设置 `MRAG_HF_OFFLINE=1`
- `MRAG_HF_HOME` 是否指向 `./models/huggingface-mrag`
- Ocean-NAT 是否已完成完整预下载（模型、vision tower、数据集）

## 7. 验收标准

满足以下三条即表示三段式链路可用：

1. Ocean-NAT 执行 `bash test/data_models.sh` 无报错，并显示 dataset rows。
2. node01 执行 `MRAG_HF_OFFLINE=1 bash test/baseline.sh` 无网络相关错误。
3. M2 能通过 `make pull result y` 拉回 `jsonl` 与 `results/` 产物。
