import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset

print("hello world")

mrag_bench = load_dataset("uclanlp/MRAG-Bench", split="test")
print(mrag_bench[0])
print("hello world")
