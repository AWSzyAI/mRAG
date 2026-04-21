import os
import time

import torch


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def log_torch_cuda_env() -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    names = []
    for i in range(count):
        try:
            names.append(torch.cuda.get_device_name(i))
        except Exception:
            names.append("<unknown>")
    log(
        "torch_cuda="
        f"available={available}, count={count}, visible={cuda_visible}, devices={names}"
    )
