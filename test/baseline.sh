cd github/MRAG-Bench && \
CUDA_VISIBLE_DEVICES=0,1 \
MRAG_HF_HOME="$PWD/.cache/huggingface-mrag" \
MRAG_MODEL_LOCAL_DIR="$PWD/../../models/llava-onevision-qwen2-7b-ov" \
MRAG_MAX_NEW_TOKENS=64 \
MRAG_MAX_RAG_IMAGES=3 \
HF_ENDPOINT=https://hf-mirror.com \
bash eval/models/run_model.sh
