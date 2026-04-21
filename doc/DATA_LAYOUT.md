# 数据与缓存目录结构（mRAG）

本文说明本仓库在 **本地 / 服务器** 上常见数据布局，以及与脚本、环境变量的对应关系。若你换了机器或只同步了代码，可用自检脚本把实际路径打印出来。

## 一键自检（推荐在服务器仓库根执行）

```bash
cd /path/to/mRAG
python scripts/inspect_data_layout.py
python scripts/inspect_data_layout.py --json-out log/data_layout.json
```

从全库语料里取 **任意一张图** 的路径（给 `test/gemma4.py --image` 等用）：

```bash
python scripts/inspect_data_layout.py --print-one-corpus-image
```

## 1. MRAG-Bench（HuggingFace `datasets`）

- **加载方式**：代码里 `datasets.load_dataset("uclanlp/MRAG-Bench", split="test")`（见 `src/mrag/mrag_bench.py`）。仍是 Hub 上的**数据集 id**，但数据在磁盘上时只读 **本地 Arrow 缓存**，不联网。
- **缓存位置**：`test/pipeline_multi_dim_rag.py` / `benchmark_corpus_rag` 在访问数据集前会调用 **`ensure_mrag_hf_cache_env()`**（`src/mrag/mrag_bench.py`）：若已设置 **`MRAG_HF_HOME`** 则用之；否则若存在 **`github/MRAG-Bench/.cache/huggingface-mrag`** 或 **`models/huggingface-mrag`**，会自动 `setdefault` 设置 `HF_HOME`、`HF_HUB_CACHE`、`HF_DATASETS_CACHE`，让 `load_dataset` 命中你已在 infinity 准备好的目录。也可手动导出与 `main.py` 一致：

  `MRAG_HF_HOME`（默认 `./models/huggingface-mrag`）

  其下常见子目录：

  - `datasets/` —— `datasets` 库的 Arrow 缓存；目录名常含 **`uclanlp___mrag-bench`**（`/` 会变成 `___`）。
  - `hub/` —— Hub 元数据与快照（与 `datasets` 下载相关）。

- **离线**：设置 `MRAG_HF_OFFLINE=1` 等（详见 `module/node3.md`、`README.md` 实验段落）。

## 2. 全库图像语料（Corpus retrieval）

- **用途**：`test/benchmark_corpus_rag.py`、`test/pipeline_multi_dim_rag.py` 等通过 **`--corpus-dir`** 指向「仅含图片」的目录；`src/mrag/clip_retriever.py` 的 `list_corpus_images` 会 **递归** 收集常见后缀（`.png` / `.jpg` / …）。
- **本仓库约定路径**：`data/image_corpus/`（与 `README.md` 中「需上传到服务器」的 `data/image_corpus/*` 一致）。
- **服务器常见做法**：语料放在 NFS 或大盘上，用环境变量 **`CORPUS_DIR`** 指向该目录；实验脚本里等价于 `--corpus-dir $CORPUS_DIR`。

**不要求** 固定子目录命名：任意子树内散落的图片均可，只要根路径传给 `--corpus-dir`。

## 3. 其它数据（可选）

- **InfoSeek / OVEN 等**：见 `scripts/download_infoseek_images.py`、`paper/DATASET_EVALUATION_REPORT.md` 等；主实验仍以 MRAG-Bench + 上述 corpus 为主。

## 4. 与 `test/gemma4.py` 的关系

服务器上往往 **没有** `paper/images/`。`test/gemma4.py` 会按顺序尝试：

1. `--image` 显式路径  
2. **`CORPUS_DIR`** 或 **`data/image_corpus`** 下找到的第一张图（递归）  
3. 仓库内 `paper/images/*.png`（若存在）  
4. 仍无则生成 `.cache/gemma4_smoke_96.png`（可用 `--no-synthetic-image` 关闭）

## 5. 文档索引

| 主题 | 位置 |
|------|------|
| 项目当前状态 | `doc/CURRENT_STATUS_2026-04.md` |
| 实验脚本与 corpus 参数 | `paper/EXPERIMENTAL_SCRIPTS_GUIDE.md` |
| 远程 HF 缓存与离线 | `module/node3.md` |
| 本页 | `doc/DATA_LAYOUT.md` |

若你执行 `inspect_data_layout.py` 的输出与上表不一致，把 **JSON 或完整 stdout** 贴进 issue/讨论，可据此再改文档或脚本探测逻辑。
