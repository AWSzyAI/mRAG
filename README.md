# mRAG
mRAG

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



```

## MRAG-BENCH环境

```bash
conda env remove -n llava -y
conda create -n llava python=3.10 -y
conda activate llava
# conda env update -n llava -f environment.yml --prune

pip install -U pip setuptools wheel
pip install "setuptools<81" wheel
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
pip install numpy==1.26.4 shortuuid datasets tqdm pillow requests \
  sentencepiece accelerate einops einops-exts timm decord \
  "httpx[socks]" huggingface_hub transformers==4.45.2 protobuf==3.20.3
pip install -e ~/code/mRAG/github/LLaVA-NeXT --no-deps
pip install av open_clip_torch

export HF_ENDPOINT=https://hf-mirror.com
cd github/MRAG-Bench && conda activate llava && bash eval/models/run_model.sh  && cd ../../
# 这条评估不知道要运行多久，看不到进度条和ETC,加上

```

```bash

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


## VPN(暂时用不了，不知道为什么)
```bash
cd env/clash/ && ./clash -d .
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export all_proxy="socks5h://127.0.0.1:7891"
export no_proxy="localhost,127.0.0.1,::1"
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