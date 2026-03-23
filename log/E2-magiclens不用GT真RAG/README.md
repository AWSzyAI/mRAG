CUDA_VISIBLE_DEVICES=1 \
LLAVA_DEVICE_MAP=single \
JAX_CUDA_REQUIRED=1 TORCH_CUDA_REQUIRED=1 \
bash test/benchmark_magiclens_real_rag.sh > benchmark_magiclens_real_rag.log 2>&1
