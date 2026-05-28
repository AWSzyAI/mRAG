# nnu 离线环境与代码同步规范

适用场景：本地机器可以联网，`nnu` 服务器不能联网或不能稳定访问 GitHub/PyPI/HuggingFace。原则是：**联网安装、下载、打包都在本地完成；nnu 只接收同步后的代码、源码依赖、wheel 或 conda 环境，然后离线运行。**

## 0. 固定工作目录

本地执行：

```bash
cd /mnt/d/mRAG
conda activate llava
```

nnu 执行：

```bash
cd /home/user/code/mRAG
conda activate llava
```

所有 `make sync*` / `ms y` 命令都必须在项目根目录运行，不要在 `github/scenic` 这类子目录里运行。

## 1. 优先同步普通代码

改了项目代码、脚本、README、`src/`、`test/` 等普通文件时：

```bash
ms y
```

如果只想看会同步什么：

```bash
ms
```

`ms y` 是项目同步的默认入口。不要为项目内文件单独手写 `rsync`；先把源码、补丁、离线 wheelhouse 放在项目目录内，再用 `ms y` 统一推到 nnu。

## 2. 完整同步 conda 环境

当 nnu 环境缺很多包，或者本地已经把 `llava` 环境配好时，用 `conda-pack` 整包迁移。

本地先确保有 `conda-pack`：

```bash
python -m pip install -U conda-pack
which conda-pack
```

预览远端路径：

```bash
make sync-env
```

打包并上传到 `nnu:/home/user/env/envs/llava`：

```bash
make sync-env y
```

如果远端已有旧环境，确认替换：

```bash
make sync-env y ENV_REPLACE=1
```

如果 `conda-pack` 报 `Files managed by conda were found to have been deleted/overwritten`，说明 `pip` 覆盖过 conda 管理的基础包。赶进度时可以忽略缺失文件继续打包：

```bash
make sync-env y ENV_REPLACE=1 ENV_PACK_IGNORE_MISSING=1
```

更干净的修复方式是先在本地重装基础包：

```bash
conda install -y --force-reinstall pip wheel setuptools
python -m pip install -U conda-pack
make sync-env y ENV_REPLACE=1
```

同步完成后验证：

```bash
make env-smoke
```

## 3. 只同步源码依赖

如果报错只是：

```text
ModuleNotFoundError: No module named 'scenic'
```

不要传 5G+ conda 环境。只同步源码包即可：

```bash
ms y
```

确认 `sync/.exclude` 没有排除需要上传的 `github/scenic/scenic/` 和相关 `test/*.py`。如果 `ms y` 输出里有 `Rsync transferred file size`，说明已经发生实际传输。

运行前设置：

```bash
export PYTHONPATH=/home/user/code/mRAG/github/scenic:/home/user/code/mRAG/github/magiclens:/home/user/code/mRAG/github/LLaVA-NeXT:$PYTHONPATH
```

验证：

```bash
python - <<'PY'
from scenic.projects.baselines.clip import tokenizer
print("scenic ok", tokenizer.__file__)
PY
```

## 4. 只补少量 Python 依赖

如果 nnu 报：

```text
ModuleNotFoundError: No module named 'absl'
```

或者类似 `ml_collections`、`immutabledict`、`clu`，优先用本地下载 wheel，再传到 nnu 离线安装。

本地：

```bash
mkdir -p .offline_wheels
python -m pip download -d .offline_wheels absl-py
ms y
```

nnu：

```bash
python -m pip install --no-index --find-links /home/user/code/mRAG/.offline_wheels absl-py
```

一次性准备 Scenic 常见依赖：

```bash
mkdir -p .offline_wheels
python -m pip download -d .offline_wheels \
  absl-py ml-collections immutabledict clu flax optax
ms y
```

nnu 离线安装：

```bash
python -m pip install --no-index --find-links /home/user/code/mRAG/.offline_wheels \
  absl-py ml-collections immutabledict clu flax optax
```

如果某个依赖会拉很大的包，先只补当前报错缺的包，验证后再继续。

如果验证 Scenic tokenizer 时报：

```text
ModuleNotFoundError: No module named 'tensorflow'
```

不要优先离线安装完整 TensorFlow。当前项目的 Scenic CLIP 只需要 `tensorflow.io.gfile` 做本地文件读写，已在本地 `github/scenic/scenic/projects/baselines/clip/download.py` 和 `model.py` 加了标准库 fallback。同步这两个小文件即可：

```bash
ms y
```

## 5. JAX 与 torch 的版本注意

在同一个 `llava` 环境里既跑 torch 又跑 JAX 时，不建议使用：

```bash
pip install -U "jax[cuda12]"
```

这个命令会安装或升级 pip CUDA/cuDNN 组件，可能把 `torch 2.5.1+cu121` 需要的 `nvidia-cudnn-cu12==9.1.0.70` 冲掉。优先用：

```bash
pip install -U "jax[cuda12-local]"
```

如果已经冲突，先恢复 torch 需要的 cuDNN，再装 local JAX：

```bash
python -m pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt
python -m pip uninstall -y nvidia-nvshmem-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-cccl-cu12
python -m pip install --force-reinstall nvidia-cudnn-cu12==9.1.0.70
python -m pip install -U "jax[cuda12-local]"
python -m pip check
```

## 6. E11_4 运行流程

nnu 上验证环境：

```bash
cd /home/user/code/mRAG
conda activate llava
export PYTHONPATH=/home/user/code/mRAG/github/scenic:/home/user/code/mRAG/github/magiclens:/home/user/code/mRAG/github/LLaVA-NeXT:$PYTHONPATH

python - <<'PY'
import torch
from scenic.projects.baselines.clip import tokenizer
print("torch cuda:", torch.cuda.is_available(), torch.cuda.device_count())
print("scenic ok:", tokenizer.__file__)
PY
```

完整运行：

```bash
bash test/E11_4_infoseek_10k.sh
```

如果采样已经存在，脚本会复用 `log/E11_4_infoseek_10k/sampling/`。需要重采样时：

```bash
FORCE_RESAMPLE=1 bash test/E11_4_infoseek_10k.sh
```

只跑 benchmark 阶段：

```bash
python test/benchmark_e11_4_infoseek.py \
  --sample-dir log/E11_4_infoseek_10k/sampling \
  --image-dir data/infoseek/images/all \
  --output-dir log/E11_4_infoseek_10k/benchmark
```

## 7. 常见卡点

`ms y` 输出看起来矛盾：

- `Files to upload: 0` 是同步脚本对“普通上传文件”的统计，不一定包含目录项、删除项、依赖目录等全部 rsync 行为。
- `Rsync transferred file size: 185.14M bytes` 才表示本次实际传输了数据。
- `(N/A)` 多见于目录、软链、删除或无法统计行数的条目，不是错误。
- `[WARN] cannot delete non-empty directory` 表示远端目录非空未删掉，常见于大数据、模型、旧 module 目录；只要不是本次要替换的代码路径，通常可以忽略。

同步大环境很慢：

- 先判断是不是只缺源码包或单个 wheel。
- 只缺 `scenic` 时，用 `ms y` 同步项目内 `github/scenic/scenic/`，不要传整个 conda 环境。
- 正在传 5G 环境包时，如果决定改走最小同步，可以 `Ctrl-C` 停掉。

底层同步报 `kex_exchange_identification: Connection closed by remote host`：

- 通常是 SSH 连接被远端临时限流，或另一个大传输还占着连接。
- 先停止大包上传，等几秒后重试小文件同步。

`pip install .` 在 nnu 上拉 GitHub 失败：

- 不要继续在 nnu 上装。
- 在本地装好后用 `make sync-env y`，或用 wheelhouse 离线补包。
