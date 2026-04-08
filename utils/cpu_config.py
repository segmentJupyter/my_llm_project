"""
Tune PyTorch / BLAS for CPU-only HPC nodes (e.g. 16 cores, 32 GB RAM).
Set TORCH_NUM_THREADS (default 16) before importing heavy torch workloads.
"""
from __future__ import annotations

import os


def configure_cpu_threads() -> int:
    n = int(os.environ.get("TORCH_NUM_THREADS", "16"))
    n = max(1, min(n, os.cpu_count() or 16))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    try:
        import torch

        torch.set_num_threads(n)
        interop = max(1, min(8, n // 4))
        torch.set_num_interop_threads(interop)
    except Exception:
        pass
    return n
