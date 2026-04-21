# Gemma 4 E2B 多模态维度规划 + MagicLens 多维 RAG 流水线

本文说明如何用**本地 Gemma 4 E2B-it**（多模态）根据**查询图 + 问题**生成多条 **MagicLens 指令式检索 query**，再与融合、LLaVA 作答串联；环境变量与可复制命令见文末。

## 架构（数据流）

```mermaid
flowchart LR
  Q[MRAG-Bench 问题 + query 图] --> G[Gemma4 E2B-it\n多模态维度生成]
  G --> D1[指令维度 1]
  G --> D2[指令维度 2]
  G --> D3[指令维度 N]
  D1 --> M1[MagicLens 编码\nquery 图 + 指令]
  D2 --> M2[MagicLens ...]
  D3 --> M3[MagicLens ...]
  M1 --> F[RRF / score_sum / voting 融合]
  M2 --> F
  M3 --> F
  F --> L[LLaVA-OneVision\n多图问答]
  L --> A[选项预测]
```



- **Gemma 4（google/gemma-4-E2B-it）**：在 Hugging Face 文档中定位为 **E2B**（Edge-to-Bridge）规模的 **多模态指令模型**；本仓库用 `AutoProcessor` + `AutoModelForImageTextToText`（或新版 `AutoModelForMultimodalLM`）加载。输入为 **系统提示（要求输出 N 条英文检索指令）+ 用户消息（本地图片路径 + 问题文本）**；输出经 `parse_dimension_lines` 解析为 N 条短句，每条作为 MagicLens 的 **text 分支** 与同一 query 图组合检索。
- **MagicLens**：对每个维度，用 query 图像 + 该条英文指令编码，与语料库向量比对（实现见 `src/mrag/magiclens.py`、`src/mrag/multi_dim_pipeline.py`）。
- **融合**：多路 Top-K 列表合并（`src/mrag/fusion.py`）。
- **LLaVA**：将 query 图与融合后的检索图一并送入多模态问答（`test/benchmark_corpus_rag.py`）。

核心代码：


| 模块                               | 作用                                          |
| -------------------------------- | ------------------------------------------- |
| `src/mrag/gemma4_loader.py`      | 加载 Processor / 多模态模型（与 `test/gemma4.py` 共用） |
| `src/mrag/gemma4_dims.py`        | 构建图文 chat + `generate` + 解析 N 条指令           |
| `src/mrag/query_planner.py`      | 维度规划系统提示与 `parse_dimension_lines`           |
| `test/pipeline_multi_dim_rag.py` | CLI：`--dim-generator-type gemma4_local`     |


## 密钥与配置

将 `example.env` 复制为仓库根目录 `.env`，填写 `HF_TOKEN`（下载/门禁）、按需填写 `DIM_GENERATOR_API_KEY`（若仍用 API 模式）。**不要把真实密钥写入仓库或提交到 git。**

## 有网（infinity）与无网（GPU）分工

计算节点 **无外网**、登录机 **infinity 等有网** 时，把「下载 / 建缓存」和「跑实验」拆开，避免进程里任何 Hugging Face / 数据集拉取。

### 在 **infinity（有网）** 完成

1. **Gemma4 权重**：`python test/gemma4.py --mode download`（或 `huggingface-cli download` 到 `models/gemma4-e2b`）。
2. **LLaVA 权重**：按 `README.md` / 组内约定把 `llava-onevision-qwen2-7b-ov` 等目录准备完整。
3. **MagicLens 权重**：`models/magic_lens_clip_base.pkl`（或 large）已就位。
4. **MRAG-Bench 与 HF 缓存**：在本机执行一次 `load_dataset("uclanlp/MRAG-Bench", split="test")` 或跑通一小段 pipeline，使 `HF_HOME`、`HF_DATASETS_CACHE`（或 `~/.cache/huggingface`）里已有快照。
5. **语料**：`CORPUS_DIR` 指向的 `image_corpus` 整棵目录。

将上述 **仓库目录 + 缓存目录**（按需）打包或 `rsync` 到 GPU 节点同一相对路径，保证 GPU 上 **不需要再解析 `huggingface.co`**。

### 在 **GPU（无网）** 运行

1. 在 **仓库根** `.env`（或启动前 `export`）建议打开离线开关（见 `example.env` 注释）：

   - `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`：禁止静默联网；缺文件会立刻报错，便于发现未同步的缓存。
   - 将 `HF_HOME` / `HF_DATASETS_CACHE` 指到已从 infinity 拷过来的目录。

2. **`pipeline_multi_dim_rag.py` 会在 `import jax` 之前加载 `.env`**，因此可在 `.env` 里写 **`JAX_PLATFORMS=cpu`**：若出现日志里 **CuDNN 运行时版本低于 jaxlib 编译版本**（例如 runtime 9.1 vs compile 9.8）导致 `DNN library initialization failed`，可先用 CPU 跑通 MagicLens（会慢）；长期方案是在 GPU 节点升级 CuDNN / 换与系统匹配的 `jaxlib` wheel。

3. 启动命令仍在仓库根执行，且 **`gemma4_local` 只读本地目录**，不应对 Hub 发起请求；若仍看到对某个模型的 `HEAD https://huggingface.co/...`，说明某依赖在 import 阶段触发了 Hub，需在 infinity 上预生成同名缓存或升级/精简 import 链。

## 环境与依赖

- PyTorch **≥ 2.4**（与当前 transformers 对 Gemma4 processor 的要求一致）。
- `pip install -U 'transformers>=4.51' accelerate huggingface_hub pillow`
- 权重目录：`GEMMA4_LOCAL_DIR`（默认 `models/gemma4-e2b`），可先执行下载模式（见下）。

## 可复制命令

在仓库根目录（`mRAG/`），且已配置 `.env` 中的 `HF_TOKEN` 等：

```bash
# 1) 下载 Gemma4 E2B 到本地（仅需一次；已下载可跳过）
python test/gemma4.py --mode download

# 2) 单机冒烟：文本 + 图文（验证权重与 CUDA）
python test/gemma4.py --mode run
# 指定图与设备示例：
# python test/gemma4.py --mode run --device cuda:1 --image "$(python scripts/inspect_data_layout.py --print-one-corpus-image)"

# 3) 整条多维 RAG：Gemma4 生成维度 → MagicLens → 融合 → LLaVA
# 无网 GPU 示例（.env 中已配置 HF_*_OFFLINE 与 JAX_PLATFORMS=cpu 时）：
# set -a && source .env && set +a && export CORPUS_DIR=...
export CORPUS_DIR=data/image_corpus  # 语料根目录
python test/pipeline_multi_dim_rag.py \
  --corpus-dir "${CORPUS_DIR}" \
  --dim-generator-type gemma4_local \
  --gemma4-local-dir models/gemma4-e2b \
  --gemma4-model-id google/gemma-4-E2B-it \
  --gemma4-device cuda:1 \
  --n-dims 3 \
  --dim-top-k 5 \
  --final-top-k 5 \
  --fusion-strategy rrf \
  --max-samples 2 \
  --answers-file log/E8/gemma4_multi_dim_smoke.jsonl \
  --summary-out log/E8/gemma4_multi_dim_smoke_summary.json \
  --save-dimensions-jsonl log/E8/gemma4_multi_dim_smoke_dims.jsonl
```

等价入口：

```bash
python test/benchmark_multi_dimension_rag.py --corpus-dir "${CORPUS_DIR}" --dim-generator-type gemma4_local --max-samples 1
```

（其余参数与上表相同；路径相对于在 `test/` 下执行时由脚本内 `ROOT_DIR` 解析。）

## 与「仅文本 API / 本地 HF」的对比


| `--dim-generator-type` | 输入                | 适用场景                          |
| ---------------------- | ----------------- | ----------------------------- |
| `api`                  | 仅问题文本（+ 可选占位描述）   | 有硅基流动等 API key、无本地 Gemma 显存   |
| `local`                | 仅问题文本             | 本地 Qwen 等文本模型                 |
| `gemma4_local`         | **问题 + query 图像** | 利用视觉上下文分解检索维度，定制 MagicLens 指令 |


## 故障排除

- **`[Errno -2] Name or service not known` + `huggingface.co`**：GPU 无网时不应访问 Hub；在 infinity 备好缓存并设 `HF_HUB_OFFLINE=1` 等（见上文「有网 / 无网分工」）。
- **`DNN library initialization failed` / CuDNN 版本不匹配**：JAX GPU 与系统 CuDNN 不一致时，可在 `.env` 设 `JAX_PLATFORMS=cpu` 跑通流水线（见上文）；根治需对齐服务器 CuDNN 与 `jaxlib`。
- **`cannot import name 'apply_chunking_to_forward' / 'find_pruneable_heads_and_indices' from 'transformers.modeling_utils'` 等 LLaVA 导入失败**：新版 `transformers` 不再把这些符号挂在 `modeling_utils` 上（部分仅存在于 `pytorch_utils`，部分在 5.x 已从 `pytorch_utils` 删除）。`test/benchmark_corpus_rag.py` 会在导入 `llava` 前调用 `src.mrag.transformers_llava_compat.ensure_modeling_utils_chunking_compat()` 统一补全。其它脚本若直接 `import llava`，需先调用该函数，或对齐 LLaVA-NeXT 与 `transformers` 版本。
- **Processor 报 PyTorch 未找到**：升级 `torch`/`torchvision` 到 ≥ 2.4（见 `test/gemma4.py` 头部说明）。
- **图文报错 / 不接受 file://**：对 Gemma 使用**本地绝对路径**的 `image` 字段（本流水线已使用路径字符串）。
- **维度行数不足 N**：可调大 `--gemma4-max-new-tokens` 或检查 `log/` 中 `dimensions.jsonl` 的原始输出；失败时会回退到 `question` 或 `--fallback-instruction`。

更多数据目录说明见 `doc/DATA_LAYOUT.md`。