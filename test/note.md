XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_ALLOCATOR=default \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.60 \
JAX_CUDA_REQUIRED=1 \
bash test/benchmark_magiclens.sh  > benchmark_magiclens.log

XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_ALLOCATOR=default \
JAX_CUDA_REQUIRED=1 \
bash test/benchmark_magiclens.sh  > benchmark_magiclens.log
