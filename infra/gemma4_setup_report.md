# Gemma4 (google/gemma-4-E2B-it) 安装与缺失项报告

日期：2026-05-13

摘要
- 目标：在服务器上为 MRAG-Bench 配置 Gemma4 本地目录（`GEMMA4_LOCAL_DIR`），并列出还缺哪些模型与数据，以便能在服务器上完整运行 E11_4 / multi-dim 检验。
- 结论快速指引：在服务器上创建 Python 虚拟环境 -> 安装依赖 -> 使用 `huggingface_hub.snapshot_download` 或 `git lfs` 将 `google/gemma-4-E2B-it` 下载到 `models/gemma4-e2b` -> 下载/确认 LLaVA 最终回答模型与 HF cache -> 确保图像语料与 MagicLens 索引存在 -> 同步本地脚本并运行 E11_4 实验。

当前仓库状态（基于本地检查）
- `example.env` 指定：
  - `GEMMA4_LOCAL_DIR=models/gemma4-e2b`
  - `GEMMA4_MODEL_ID=google/gemma-4-E2B-it`
  - 注释有 `CORPUS_DIR=/path/to/image_corpus`
- 本地工作区未包含 `models/` 目录（仓库根下没有预装 Gemma4 权重）。
- 项目中已有对 LLaVA、MagicLens、image_corpus 的引用（请参见 `example.env` 与 `sync/Makefile` 中的路径引用），但模型/索引文件需要放到服务器上或 HF cache 中。

缺失项清单（需要在服务器上准备）
1. Gemma4 权重与配置：`google/gemma-4-E2B-it`（放在 `models/gemma4-e2b`）
2. LLaVA 最终回答模型（仓库/权重），用于最终 answerer（项目中通过 `MRAG_MODEL_LOCAL_DIR` 指定）
3. HF token / gated-model 访问凭证（如果 Gemma4 为 gated 模型）——设置 `GEMMA4_HF_TOKEN` 或 `HF_TOKEN` 环境变量
4. 图像语料与索引：`image_corpus` 或 MagicLens 的检索索引（若不存在，需要在服务器上构建）
5. 依赖环境：`transformers`, `huggingface_hub`, `torch`/`jax`/`flax`（取决于模型后端），`git-lfs` 等
6. 本地实验脚本：将本地 `test/*.py` 和 `test/*.sh`（例如 `test/E11_4_infoseek.sh`, `test/infoseek_open_ended_multidim.py`）同步到服务器并可执行

推荐的服务器操作步骤（以服务器 shell 为准）

1) 登录并进入项目目录（示例）
```
ssh user@nnu
cd /home/user/code/mRAG
```

2) 建立并激活虚拟环境
```
python3 -m venv ~/envs/mrag_gemma4
source ~/envs/mrag_gemma4/bin/activate
pip install --upgrade pip
```

3) 安装常见依赖（视 GPU/TPU 与后端选择调整）
```
pip install transformers huggingface_hub accelerate sentence-transformers torch torchvision git+https://github.com/huggingface/transformers.git
# 如果使用 JAX/Flax，则按 https://github.com/google/jax 的说明安装 jax/jaxlib/flax
pip install git-lfs
git lfs install
```

4) 下载 Gemma4 权重到 `models/gemma4-e2b`
- 推荐（Python API，支持断点续传）：
```
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="google/gemma-4-E2B-it", local_dir="models/gemma4-e2b", token=None)
PY
```
- 或使用 `git lfs` / `huggingface-cli`：
```
export HF_TOKEN="<your_token>"
huggingface-cli repo clone google/gemma-4-E2B-it models/gemma4-e2b
```

5) 验证文件存在
```
ls -lah models/gemma4-e2b
```
检查是否有 `config.json`, 权重文件（.bin/.msgpack/.npz 等）、tokenizer/bpe 文件等。

6) 配置环境变量（可写入服务器的 `.env` 或 systemd/unit）
```
export GEMMA4_LOCAL_DIR=models/gemma4-e2b
export GEMMA4_HF_TOKEN="$HF_TOKEN"
```

7) 下载/确认 LLaVA 最终回答模型（示例）
```
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="lmms-lab/llava-onevision-qwen2-7b-ov", local_dir="models/llava-onevision-qwen2-7b-ov")
PY
```
或按项目中 `MRAG_MODEL_LOCAL_DIR` 的值下载相应模型。

8) 确保图像语料（`CORPUS_DIR`）与 MagicLens 索引可用
- 如果服务器已有 `data/image_corpus`：确认路径，并在 `example.env` 中把 `CORPUS_DIR` 指向它。
- 如果没有：将远程或本地的图像集合 rsync 到服务器（注意磁盘空间），然后运行项目里的索引/构建脚本（如果有）。示例 rsync：
```
rsync -av --progress user@source:/path/to/image_corpus/ /home/user/code/mRAG/data/image_corpus/
```

9) 同步本地脚本到服务器并运行（我们修复了 `sync/Makefile` 的 shell 问题）
```
cd /home/user/code/mRAG
make sync y
# 或者手动 rsync test/ 到服务器上的相应路径
```

10) 运行 E11_4 实验（示例）
```
source ~/envs/mrag_gemma4/bin/activate
export HF_TOKEN="<your_token>"
export GEMMA4_LOCAL_DIR=models/gemma4-e2b
./test/E11_4_infoseek.sh  # 或直接运行 test/pipeline_multi_dim_rag.py，按项目说明传参数
```

验证项
- `models/gemma4-e2b` 下有 config/weights/tokenizer
- `models/...llava...` 下有回答器模型
- `data/image_corpus` 存在并可被 MagicLens 索引
- `.env`（或进程环境）包含 `HF_TOKEN` / `GEMMA4_HF_TOKEN`
- `test/E11_4_infoseek.sh` 与 `test/infoseek_open_ended_multidim.py` 在服务器可执行

注意事项与建议
- Gemma4 权重可能很大（数十 GB）；请预留足够磁盘空间并使用 `snapshot_download` 以便断点续传。
- JAX/Flax 与 PyTorch 的安装依赖于服务器的 GPU/OS/driver，按官方说明选择合适的二进制（特别是 `jaxlib`）。
- 如果模型受 gated 访问控制，需要有效的 HF token（并在 `example.env` 中设置）。
- 在推理时，确认 `GEMMA4_DEVICE`（如 `cuda:0`）是否匹配可用 GPU；在无 GPU 时可能需要用 CPU 模式，性能下降很大。

下一步建议
1. 在服务器上按上文步骤先下载 `google/gemma-4-E2B-it` 到 `models/gemma4-e2b`（或告知我我可生成一个可在服务器上直接运行的脚本）。
2. 同步本地 `test/` 脚本到服务器并执行一次小样本跑通（10 个样本），验证检索链与 answerer 能否工作。
3. 我已在仓库中添加两个脚本：
  - `infra/setup_gemma4.sh`：在服务器上运行以下载 Gemma4 与 LLaVA 模型（使用 `huggingface_hub.snapshot_download`），并写入基本环境变量提示。
  - `infra/check_server_requirements.sh`：在服务器上运行以验证模型、图像语料、GPU 与磁盘空间是否满足运行要求。
  请先把这两个脚本通过 `make sync y` 同步到服务器并在目标机器上以非 root 用户执行（脚本会打印检查结果与建议）。

报告作者：自动生成 by Copilot agent
