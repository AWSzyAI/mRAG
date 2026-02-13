# mRAG
using magiclens to MRAG-BENCH

文件传输瓶颈：
```
.cache
data/
models/
github/magiclens/data/***
github/magiclens/models/***
github/MRAG-Bench/.cache/
!github/MRAG-Bench/eval/models
```



## rsync
NNU服务器（VScode Remote-SSH）无法使用CodeX，Claude Code，只能在本地改代码，在服务器上运行。
```bash
echo "140.82.112.4 github.com" >> /etc/hosts
cd github
git clone https://github.com/mragbench/MRAG-Bench.git
git clone https://github.com/google-deepmind/magiclens.git
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd ..


make config
eval "$(make -s alias)"

# make sync
ms 

# pull remote MRAG-Bench results back to local
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

# 00:04:11

export HF_ENDPOINT=https://hf-mirror.com
mkdir -p "$PWD/.cache/huggingface-mrag"/{hub,datasets}
mkdir -p "$PWD/models"

conda activate llava

# 把 LLaVA-OneVision 模型单独下载到 mRAG/models，避免评估时边跑边下载。
python main.py \
  --model-local-dir ./models/llava-onevision-qwen2-7b-ov \
  --hf-home ./models/huggingface-mrag \
  --hf-endpoint https://hf-mirror.com



cd github/MRAG-Bench && \
CUDA_VISIBLE_DEVICES=0,1 \
MRAG_HF_HOME="$PWD/.cache/huggingface-mrag" \
MRAG_MODEL_LOCAL_DIR="$PWD/../../models/llava-onevision-qwen2-7b-ov" \
MRAG_NUM_BEAMS=5 \
MRAG_MAX_NEW_TOKENS=64 \
MRAG_MAX_RAG_IMAGES=3 \
HF_ENDPOINT=https://hf-mirror.com \
bash eval/models/run_model.sh


# 运行评估：若本地目录存在，run_model.sh 会优先使用本地模型。
cd github/MRAG-Bench && \
MRAG_HF_HOME="$PWD/.cache/huggingface-mrag" \
MRAG_MODEL_LOCAL_DIR="$PWD/../../models/llava-onevision-qwen2-7b-ov" \
HF_HUB_ENABLE_HF_TRANSFER=0 \
HF_ENDPOINT=https://hf-mirror.com \
bash eval/models/run_model.sh

cd ../../

cd github/MRAG-Bench && python eval/score.py -i llava_one_vision_gt_rag_results.jsonl && cd ../../

```


用时：00:43:51.39
```bash
(py310) root:~/code/mRAG# cd github/MRAG-Bench && conda activate llava && bash eval/models/run_model.sh  && cd ../../
[ENV] HF_HOME=/home/user/.cache/huggingface-mrag
[ENV] HF_HUB_CACHE=/home/user/.cache/huggingface-mrag/hub
[ENV] HF_DATASETS_CACHE=/home/user/.cache/huggingface-mrag/datasets
[ENV] http_proxy=<unset> https_proxy=<unset> all_proxy=<unset>
[ENV] LD_LIBRARY_PATH=<unset>
Loaded LLaVA model: lmms-lab/llava-onevision-qwen2-7b-ov
You are using a model of type llava to instantiate a model of type llava_qwen. This is not supported for all configurations of models and can yield errors.
Overwriting config with {'image_aspect_ratio': 'pad'}
Loading vision tower: google/siglip-so400m-patch14-384
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████| 4/4 [03:47<00:00, 56.99s/it]
Model Class: LlavaQwenForCausalLM
[INFO] Loading MRAG-Bench test split...
[INFO] load_dataset(name=uclanlp/MRAG-Bench, split=test, offline=False, max_retries=1)
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████| 28/28 [00:00<00:00, 28940.49it/s]
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████| 28/28 [00:00<00:00, 29974.61it/s]
[INFO] Dataset loaded in 7.0s
[INFO] Dataset ready. total=1353
[INFO] Fetching first sample...
[INFO] First sample fetched. Starting evaluation.
MRAG-Bench Eval: 100%|█████████████████████████████████████████████████| 1353/1353 [35:02<00:00,  1.55s/sample, avg_s=1.6, eta_s=0, step_s=1.9]
(llava) root:~/code/mRAG# cd github/MRAG-Bench && python eval/score.py -i llava_one_vision_gt_rag_results.jsonl && cd ../../
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1353/1353 [00:00<00:00, 386863.00it/s]
Overall Accuracy: 60.24%
==================================================
Partial:  66.67 //246
Incomplete:  29.41 //102
Obstruction:  66.67 //108
Others:  67.5 //120
Angle:  60.25 //322
Deformation:  56.86 //102
Scope:  63.73 //102
Biological:  57.84 //102
Temporal:  61.74 //149
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
wget https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade "jax[cuda12_pip]" -f jax_cuda_releases.html

python -m pip install ftfy regex tqdm
python -m pip install clip-anytorch
python -m pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt
python -m pip install -U "jax[cuda12]"



export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

make bpe
# 以上命令会把CLIP tokenizer词表放到:
# /home/user/code/mRAG/models/bpe_simple_vocab_16e6.txt.gz
# predict_one.py / inference.py / predict_coco100_one.py 会优先从这里加载

cd github/magiclens && JAX_PLATFORMS=cuda python predict_one.py \
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
