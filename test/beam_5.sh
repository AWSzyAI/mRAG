cd github/MRAG-Bench && \
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
MRAG_NUM_BEAMS=5 \
MRAG_DO_SAMPLE=false \
MRAG_MAX_NEW_TOKENS=32 \
MRAG_MAX_RAG_IMAGES=1 \
MRAG_HF_HOME="$PWD/.cache/huggingface-mrag" \
MRAG_MODEL_LOCAL_DIR="$PWD/../../models/llava-onevision-qwen2-7b-ov" \
HF_ENDPOINT=https://hf-mirror.com \
bash eval/models/run_model.sh
cd ../../
cd github/MRAG-Bench && python eval/score.py -i llava_one_vision_gt_rag_results.jsonl && cd ../../