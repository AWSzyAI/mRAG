开发：HPC(SSH)

# mRAG

using magiclens to MRAG-BENCH

## 当前优先入口

如果你现在重新进入这个仓库，建议优先看：

1. `doc/CURRENT_STATUS_2026-04.md`
2. `doc/DATA_LAYOUT.md`（MRAG-Bench 缓存、`data/image_corpus`、`MRAG_HF_HOME` 等目录约定；服务器上可跑 `python scripts/inspect_data_layout.py` 自检）
3. `doc/04-02/magiclens_vs_clip_query_image_analysis_2026-04-02.md`
4. `paper/content.tex`
5. `Makefile`

当前项目主线已经不是单纯“把 MagicLens 接到 MRAG-Bench 上”，而是：

`CLIP 粗召回 + MagicLens 关系感知检索/重排 + 多维度 query decomposition + 论文整合`

当前代码中心正在迁移到 `src/mrag/`，当前实验入口主要在：

- `test/benchmark_corpus_rag.py`
- `test/benchmark_magiclens.py`
- `test/pipeline_multi_dim_rag.py`（主入口）
- `test/benchmark_multi_dimension_rag.py`（兼容包装入口，转调 `pipeline_multi_dim_rag.py`）

### 服务器数据布局（已验证）

在服务器仓库根目录（有 `Makefile`、`data/`、`models/` 的目录）执行：

```bash
python scripts/inspect_data_layout.py
python scripts/inspect_data_layout.py --print-one-corpus-image
```

当前已验证的典型结果（`/public/home/hzh/mRAG`）：

- 全库语料：`data/image_corpus`（约 19185 张）
- 数据总目录：`data`（约 19285 张，含 `COCO2017_100` 与 `image_corpus`）
- MRAG-Bench 缓存：`models/huggingface-mrag/datasets/uclanlp___mrag-bench`
- HF Hub 缓存：`models/huggingface-mrag/hub`

如果只想快速理解当前阶段，请先读 `doc/CURRENT_STATUS_2026-04.md`，再决定是进入 `paper/`、实验脚本还是同步工作流。

Gemma4 本地推理（文本 + 图文）建议命令：

```bash
python test/gemma4.py --mode run --image "$(python scripts/inspect_data_layout.py --print-one-corpus-image)"
```

当前 sync 工作流已经模块化：

- 根目录 `Makefile` 只负责 `include module/Makefile`
- 真正的 sync 配置与实现都在 `module/`
- 迁移到新项目时，优先复制整个 `module/`，不要再手工挑 `.sync_ssh/.alias/.exclude`

- 代码存本地+github，rsync到服务器
- 根据README配置环境并保存镜像
- 数据通过各种脚本直接从服务器端拉取下载
- 运行结果从服务器通过rsync拉取到本地分析



需要手动上传到服务器的文件有：
```
data/image_corpus/*
models/magic_lens_clip_base.pkl
models/magic_lens_clip_large.pkl
```



```
export LC_ALL=C
export LANG=C

apt-get update
apt-get install -y apt-transport-https ca-certificates gnupg curl

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
  > /etc/apt/sources.list.d/google-cloud-sdk.list

apt-get update
apt-get install -y google-cloud-cli

gsutil cp -R gs://gresearch/magiclens/models ./
```


## rsync
NNU服务器（VScode Remote-SSH）无法使用CodeX，Claude Code，只能在本地改代码，在服务器上运行。

服务器中
```
sudo apt install -y rsync
```

本地
```bash

brew install rsync

echo "140.82.112.4 github.com" >> /etc/hosts
cd github
git clone https://github.com/mragbench/MRAG-Bench.git
git clone https://github.com/google-deepmind/magiclens.git
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd ..


make config
eval "$(make -s alias)"

# 如果 M2 还是旧版本（没有 module 化 sync 和 pull_list 机制），先引导同步:
# scp AC:/home/database/2025/mRAG/Makefile .
# scp -r AC:/home/database/2025/mRAG/module .

# preview sync changes (dry-run only)
ms

# apply sync (local -> remote)
ms y

# preview pull changes using module/pull_list.txt (no actual download)
make pull

# apply pull using module/pull_list.txt
make pull y

# preview/apply result artifacts via module/result.txt
make pull result
make pull result y

# shortcut for: make pull result y
mr
```

## MRAG-BENCH环境

```bash
conda env remove -n llava -y
conda create -n llava python=3.10 -y
conda activate llava
# conda env create -f environment.yml
# conda env update -f /home/user/code/environment.yml

pip install -U pip setuptools wheel 
pip install "setuptools<81" wheel
pip install scipy joblib matplotlib nvidia-nccl-cu12 av open_clip_torch openai
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
pip install numpy==1.26.4 shortuuid datasets tqdm pillow requests \
  sentencepiece accelerate einops einops-exts timm decord \
  "httpx[socks]" huggingface_hub transformers==4.45.2 protobuf==3.20.3 
pip install -e ./github/LLaVA-NeXT --no-deps
pip install hf_transfer
pip install -U 'transformers>=4.51' accelerate huggingface_hub
pip install --upgrade 'torch>=2.4' 'torchvision>=0.19' --index-url https://download.pytorch.org/whl/cu121
# 与上面 torch 大版本对齐，否则 pip 会提示：torchaudio 2.1.x 需要 torch==2.1.2
pip install --upgrade torchaudio --index-url https://download.pytorch.org/whl/cu121

# 00:04:11

export HF_ENDPOINT=https://hf-mirror.com
mkdir -p "$PWD/models/huggingface-mrag"/{hub,datasets}
mkdir -p "$PWD/models"

conda activate llava

# 把 LLaVA-OneVision 模型单独下载到 mRAG/models，避免评估时边跑边下载。
# 这一步也会预下载 MRAG-Bench 数据集和 vision tower 到 --hf-home 对应目录。
python main.py \
  --model-local-dir ./models/llava-onevision-qwen2-7b-ov \
  --hf-home ./models/huggingface-mrag \
  --hf-endpoint https://hf-mirror.com



cd github/MRAG-Bench && \
CUDA_VISIBLE_DEVICES=0,1 \
MRAG_HF_HOME="$PWD/../../models/huggingface-mrag" \
MRAG_MODEL_LOCAL_DIR="$PWD/../../models/llava-onevision-qwen2-7b-ov" \
MRAG_NUM_BEAMS=5 \
MRAG_MAX_NEW_TOKENS=64 \
MRAG_MAX_RAG_IMAGES=3 \
HF_ENDPOINT=https://hf-mirror.com \
bash eval/models/run_model.sh


# 运行评估：若本地目录存在，run_model.sh 会优先使用本地模型。
cd github/MRAG-Bench && \
MRAG_HF_HOME="$PWD/../../models/huggingface-mrag" \
MRAG_MODEL_LOCAL_DIR="$PWD/../../models/llava-onevision-qwen2-7b-ov" \
HF_HUB_ENABLE_HF_TRANSFER=0 \
HF_ENDPOINT=https://hf-mirror.com \
bash eval/models/run_model.sh

# 纯离线节点运行（不允许联网）：
cd github/MRAG-Bench && \
MRAG_HF_HOME="$PWD/../../models/huggingface-mrag" \
MRAG_MODEL_LOCAL_DIR="$PWD/../../models/llava-onevision-qwen2-7b-ov" \
MRAG_HF_OFFLINE=1 \
bash eval/models/run_model.sh

cd ../../

cd github/MRAG-Bench && python eval/score.py -i llava_one_vision_gt_rag_results.jsonl && cd ../../

```



### 模型文件保存位置
（服务器重启会不会被清除？还是应该放在`/home/user/env/`下面才能持久化，或者直接放在./models ./data? 下次服务器重启如果文件不见了那就这样改，暂时默认路径先用着）：
```bash
(base) szy@szym2-2 mRAG % mc
Remote CMD: ls /home/user/.cache/huggingface/hub
datasets--uclanlp--MRAG-Bench
models--lmms-lab--llava-onevision-qwen2-7b-ov
version.txt
```
MRAG-BENCH数据集保存在：`/home/user/.cache/huggingface/datasets/uclanlp___mrag-bench`


## MagicLens环境

```bash

gsutil cp -R gs://gresearch/magiclens/models ./

conda create -n magic_lens python=3.10 -y
cd github
git clone https://github.com/google-research/scenic.git
cd scenic
pip install .
pip install -r scenic/projects/baselines/clip/requirements.txt
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install ftfy regex tqdm clip-anytorch
pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt
pip install -U "jax[cuda12]"



export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

make bpe
# 以上命令会把CLIP tokenizer词表放到:
# /home/user/code/mRAG/models/bpe_simple_vocab_16e6.txt.gz
# predict_one.py / inference.py / predict_coco100_one.py 会优先从这里加载

cd github/magiclens && conda activate llava && JAX_PLATFORMS=cuda python predict_one.py \
  --model_size base \
  --model_path ../../models/magic_lens_clip_base.pkl \
  --query_image ../../data/COCO2017_100/unlabeled2017/000000002505.jpg \
  --instruction "find the same image" \
  --target_image ../../data/COCO2017_100/unlabeled2017/000000007731.jpg \
  --print_embeddings \
  --embeddings_out ../../log/predict_one_embeddings.npz
cd ../..


```
居然在llava环境跑magiclens成功了（
```bash
(llava) ➜ mRAG cd github/magiclens && JAX_PLATFORMS=cuda python predict_one.py \
  --model_size base \
  --model_path ../../models/magic_lens_clip_base.pkl \
  --query_image ../../data/COCO2017_100/unlabeled2017/000000002505.jpg \
  --instruction "find the same image" \
  --target_image ../../data/COCO2017_100/unlabeled2017/000000007731.jpg \
  --print_embeddings \
  --embeddings_out ../../log/predict_one_embeddings.npz

[boot] importing predict_one dependencies...
/environment/miniconda3/envs/llava/lib/python3.10/site-packages/clip/clip.py:6: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import packaging
[boot] importing MagicLens dependencies...
[boot] dependencies imported.
[boot] dependencies imported.
[2026-02-13 09:11:32] predict_one started
[2026-02-13 09:11:32] args: model_size=base, model_path=../../models/magic_lens_clip_base.pkl, query_image=../../data/COCO2017_100/unlabeled2017/000000002505.jpg, target_image=../../data/COCO2017_100/unlabeled2017/000000007731.jpg
[2026-02-13 09:11:32] JAX backend=gpu, devices=[CudaDevice(id=0)]
[2026-02-13 09:11:32] model file: ../../models/magic_lens_clip_base.pkl (634.91 MB)
[2026-02-13 09:11:32] query file: ../../data/COCO2017_100/unlabeled2017/000000002505.jpg (0.50 MB)
[2026-02-13 09:11:32] target file: ../../data/COCO2017_100/unlabeled2017/000000007731.jpg (0.03 MB)
[2026-02-13 09:11:32] resolved local bpe_path=/home/featurize/work/mRAG/models/bpe_simple_vocab_16e6.txt.gz
[2026-02-13 09:11:32] building tokenizer from local BPE
[2026-02-13 09:11:32] tokenizer ready in 0.08s
[2026-02-13 09:11:32] loading model weights
[2026-02-13 09:11:32] Initializing model (size=base)
2026-02-13 09:11:35.318939: W external/xla/xla/service/gpu/autotuning/dot_search_space.cc:200] All configs were filtered out because none of them sufficiently match the hints. Maybe the hints set does not contain a good representative set of valid configs?Working around this by using the full hints set instead.
2026-02-13 09:11:43.695127: W external/xla/xla/service/gpu/autotuning/dot_search_space.cc:200] All configs were filtered out because none of them sufficiently match the hints. Maybe the hints set does not contain a good representative set of valid configs?Working around this by using the full hints set instead.
2026-02-13 09:11:44.590949: W external/xla/xla/service/gpu/autotuning/dot_search_space.cc:200] All configs were filtered out because none of them sufficiently match the hints. Maybe the hints set does not contain a good representative set of valid configs?Working around this by using the full hints set instead.
[2026-02-13 09:11:53] Model initialized
[2026-02-13 09:11:53] Loading checkpoint from ../../models/magic_lens_clip_base.pkl (0.62 GB)
[2026-02-13 09:11:54] Model loaded in 22.6s
[2026-02-13 09:11:54] model+params ready in 22.57s
[2026-02-13 09:11:54] encoding query
[2026-02-13 09:11:54] encode start: image=../../data/COCO2017_100/unlabeled2017/000000002505.jpg, text_len=19
[2026-02-13 09:11:55] image preprocessed in 0.42s, shape=(1, 224, 224, 3)
[2026-02-13 09:11:55] text tokenized in 0.00s, shape=(1, 77)
[2026-02-13 09:11:55] model.apply start (first call may include JAX compilation and can be slow)
[2026-02-13 09:11:56] model.apply done in 1.13s
[2026-02-13 09:11:56] encode total 1.55s
[2026-02-13 09:11:56] encoding target
[2026-02-13 09:11:56] encode start: image=../../data/COCO2017_100/unlabeled2017/000000007731.jpg, text_len=0
[2026-02-13 09:11:56] image preprocessed in 0.34s, shape=(1, 224, 224, 3)
[2026-02-13 09:11:56] text tokenized in 0.00s, shape=(1, 77)
[2026-02-13 09:11:56] model.apply start (first call may include JAX compilation and can be slow)
[2026-02-13 09:11:57] model.apply done in 0.98s
[2026-02-13 09:11:57] encode total 1.32s
[2026-02-13 09:11:57] similarity computed in 0.0000s
similarity=0.114134
embeddings_saved=../../log/predict_one_embeddings.npz
query_embedding_shape=(512,)
target_embedding_shape=(512,)
[2026-02-13 09:11:57] predict_one finished in 25.73s
```

```bash
(llava) ➜ mRAG python - <<'PY'
import numpy as np
x = np.load("log/predict_one_embeddings.npz")
print(x["query_embedding"].shape, x["target_embedding"].shape, x["similarity"])
PY

(512,) (512,) 0.114134476
```

- `query_embed = encode(query_image, instruction)`
- `target_embed = encode(target_image, "")`
- 输出 `similarity = dot(query_embed, target_embed)`


```

cd github/magiclens/ && conda activate py310 && python inference.py \
    --model_size base \
    --model_path ./models/magic_lens_clip_base.pkl \
    --dataset circo

cd github/magiclens/ && conda activate py310 && python inference.py \
  --model_size large \
  --model_path ./models/magic_lens_clip_large.pkl \
  --dataset circo \
  --device gpu \
  --batch_size 16


```


下载数据集和模型
```bash
bash test/data_models.sh 
```


## VPN
```bash
# 1) 准备 MMDB（若 jsdelivr 不通会自动尝试 github raw / ghproxy）
cd /public/home/hzh/mRAG
bash scripts/fetch_country_mmdb.sh env/Country.mmdb

# 2) 启动 clash（工作目录必须包含 config.yaml 和 Country.mmdb）
cd env
./clash -d .

# 3) 当前 shell 启用代理
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export all_proxy="socks5h://127.0.0.1:7891"
export no_proxy="localhost,127.0.0.1,::1"

# 4) 关闭代理
unset http_proxy https_proxy all_proxy no_proxy
```










# 结果
## baseline

```bash
(llava) root:~/code/MRAG-Bench# python eval/score.py -i /home/user/code/MRAG-Bench/llava_one_vision_gt_rag_results.jsonl
100%|███████████████████████████████████████████████████████| 1353/1353 [00:00<00:00, 312253.40it/s]
Overall Accuracy: 60.31%
==================================================
Partial:  66.67
Biological:  57.84
Obstruction:  66.67
Scope:  63.73
Temporal:  61.74
Incomplete:  30.39
Others:  67.5
Angle:  60.25
Deformation:  56.86
```
