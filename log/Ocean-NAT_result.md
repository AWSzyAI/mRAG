```bash
(llava) [hzh@node01 mRAG]$ bash test/baseline.sh 
[ENV] HF_HOME=/public/home/hzh/mRAG/github/MRAG-Bench/../../models/huggingface-mrag
[ENV] HF_HUB_CACHE=/public/home/hzh/mRAG/github/MRAG-Bench/../../models/huggingface-mrag/hub
[ENV] HF_DATASETS_CACHE=/public/home/hzh/mRAG/github/MRAG-Bench/../../models/huggingface-mrag/datasets
[ENV] HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
[ENV] MRAG_HF_OFFLINE=0 HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
[ENV] MRAG_MODEL_LOCAL_DIR=/public/home/hzh/mRAG/github/MRAG-Bench/../../models/llava-onevision-qwen2-7b-ov
[ENV] MRAG_MODEL_PATH=/public/home/hzh/mRAG/github/MRAG-Bench/../../models/llava-onevision-qwen2-7b-ov
[ENV] CUDA_VISIBLE_DEVICES=0,1
[ENV] http_proxy=<unset> https_proxy=<unset> all_proxy=<unset>
[ENV] LD_LIBRARY_PATH=<unset>
The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a one-time only operation. You can interrupt this and resume the migration later on by calling `transformers.utils.move_cache()`.
0it [00:00, ?it/s]
[INFO] Loading model from: /public/home/hzh/mRAG/github/MRAG-Bench/../../models/llava-onevision-qwen2-7b-ov
Loaded LLaVA model: /public/home/hzh/mRAG/github/MRAG-Bench/../../models/llava-onevision-qwen2-7b-ov
You are using a model of type llava to instantiate a model of type llava_qwen. This is not supported for all configurations of models and can yield errors.
Overwriting config with {'image_aspect_ratio': 'pad'}
Loading vision tower: google/siglip-so400m-patch14-384
Loading checkpoint shards: 100%|██████████████████| 4/4 [00:04<00:00,  1.07s/it]
Model Class: LlavaQwenForCausalLM
[INFO] Loading MRAG-Bench test split...
[INFO] Cache filesystem free space: 28225.9 GB / 59367.8 GB (/public/home/hzh/mRAG/github/MRAG-Bench/../../models/huggingface-mrag/datasets)
[INFO] load_dataset(name=uclanlp/MRAG-Bench, split=test, offline=False, max_retries=8)
Using the latest cached version of the dataset since uclanlp/MRAG-Bench couldn't be found on the Hugging Face Hub
Found the latest cached dataset configuration 'default' at /public/home/hzh/mRAG/github/MRAG-Bench/../../models/huggingface-mrag/datasets/uclanlp___mrag-bench/default/0.0.0/43620fde4044bb150e158c67ad7af6b6b11e1da2 (last modified on Sun Feb 15 13:37:52 2026).
[INFO] Dataset loaded in 1.2s
[INFO] Dataset ready. total=1353
[INFO] Fetching first sample...
[INFO] First sample fetched. Starting evaluation.
MRAG-Bench Eval:   0%| | 0/1353 [00:00<?, ?sample/s, stage=generate, step=1/1353The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
MRAG-Bench Eval:  17%|▏| 233/1353 [02:18<11:43,  1.59sample/s, avg_s=0.6, eta_s=/public/home/hzh/.conda/envs/llava/lib/python3.10/site-packages/PIL/Image.py:1039: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
  warnings.warn(
MRAG-Bench Eval: 100%|█| 1353/1353 [13:02<00:00,  1.73sample/s, avg_s=0.6, eta_s(llava) [hzh@node01 mRAG]$ cd github/MRAG-Bench && python eval/score.py -i llava_one_vision_gt_rag_results.jsonl && cd ../../
100%|███████████████████████████████████| 1353/1353 [00:00<00:00, 422333.36it/s]
Overall Accuracy: 59.05%
==================================================
Partial:  64.23
Temporal:  57.72
Obstruction:  67.59
Scope:  66.67
Incomplete:  29.41
Deformation:  56.86
Others:  64.17
Angle:  59.63
Biological:  55.88
(llava) [hzh@node01 mRAG]$ 

```


```bash
(llava) [hzh@node01 mRAG]$ bash test/beam_5.sh 
[ENV] HF_HOME=/public/home/hzh/mRAG/models/huggingface-mrag
[ENV] HF_HUB_CACHE=/public/home/hzh/mRAG/models/huggingface-mrag/hub
[ENV] HF_DATASETS_CACHE=/public/home/hzh/mRAG/models/huggingface-mrag/datasets
[ENV] HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
[ENV] MRAG_HF_OFFLINE=0 HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
[ENV] MRAG_MODEL_LOCAL_DIR=/public/home/hzh/mRAG/models/llava-onevision-qwen2-7b-ov
[ENV] MRAG_MODEL_PATH=/public/home/hzh/mRAG/models/llava-onevision-qwen2-7b-ov
[ENV] CUDA_VISIBLE_DEVICES=0,1
[ENV] http_proxy=<unset> https_proxy=<unset> all_proxy=<unset>
[ENV] LD_LIBRARY_PATH=<unset>
[INFO] Loading model from: /public/home/hzh/mRAG/models/llava-onevision-qwen2-7b-ov
Loaded LLaVA model: /public/home/hzh/mRAG/models/llava-onevision-qwen2-7b-ov
You are using a model of type llava to instantiate a model of type llava_qwen. This is not supported for all configurations of models and can yield errors.
Overwriting config with {'image_aspect_ratio': 'pad'}
Loading vision tower: google/siglip-so400m-patch14-384
Loading checkpoint shards: 100%|██████████████████| 4/4 [00:04<00:00,  1.09s/it]
Model Class: LlavaQwenForCausalLM
[INFO] Loading MRAG-Bench test split...
[INFO] Cache filesystem free space: 28227.2 GB / 59367.8 GB (/public/home/hzh/mRAG/models/huggingface-mrag/datasets)
[INFO] load_dataset(name=uclanlp/MRAG-Bench, split=test, offline=False, max_retries=8)
Using the latest cached version of the dataset since uclanlp/MRAG-Bench couldn't be found on the Hugging Face Hub
Found the latest cached dataset configuration 'default' at /public/home/hzh/mRAG/models/huggingface-mrag/datasets/uclanlp___mrag-bench/default/0.0.0/43620fde4044bb150e158c67ad7af6b6b11e1da2 (last modified on Sun Feb 15 13:37:52 2026).
[INFO] Dataset loaded in 0.1s
[INFO] Dataset ready. total=1353
[INFO] Fetching first sample...
[INFO] First sample fetched. Starting evaluation.
MRAG-Bench Eval:   0%| | 0/1353 [00:00<?, ?sample/s, stage=generate, step=1/1353The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
MRAG-Bench Eval:  17%|▏| 233/1353 [04:45<23:39,  1.27s/sample, avg_s=1.2, eta_s=/public/home/hzh/.conda/envs/llava/lib/python3.10/site-packages/PIL/Image.py:1039: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
  warnings.warn(
MRAG-Bench Eval: 100%|█| 1353/1353 [27:30<00:00,  1.22s/sample, avg_s=1.2, eta_s
100%|███████████████████████████████████| 1353/1353 [00:00<00:00, 431386.80it/s]
Overall Accuracy: 57.35%
==================================================
Incomplete:  30.39
Temporal:  59.06
Deformation:  55.88
Obstruction:  64.81
Others:  64.17
Biological:  55.88
Angle:  56.21
Partial:  60.98
Scope:  63.73
(llava) [hzh@node01 mRAG]$ 
```