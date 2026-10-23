"""Reproducible resource controls for small laptop experiments.

PyTorch provides actual CUDA memory information. Multiple GPUs are listed but
never combined. OOM fallback is logged by the caller instead of hidden.
"""
from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path
from typing import Any

os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for matched controlled runs.

    Args:
        seed: Nonnegative pseudorandom seed used for sampling and initialization.
    Returns:
        None. Global generators are reset; existing checkpoint RNGs may override.
    Limitations:
        Cross-version and cross-device numerical identity is not guaranteed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)


def hardware_report(path: Path | None = None) -> dict[str, Any]:
    """Measure usable NVIDIA devices and save a machine-readable report.

    Args:
        path: Optional JSON destination; parent directories are created.
    Returns:
        CPU/Python/PyTorch details and per-device total/free/allocated memory.
    Limitations:
        Free memory varies with other processes; display-only GPUs may not be CUDA.
    """
    report: dict[str, Any] = {'python': platform.python_version(), 'platform': platform.platform(),
                            'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
                            'memory_is_not_pooled': True, 'gpus': []}
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        report['gpus'].append({'index': index, 'name': props.name, 'vram_bytes': total,
                              'free_bytes': free, 'allocated_bytes': torch.cuda.memory_allocated(index),
                              'peak_allocated_bytes': torch.cuda.max_memory_allocated(index)})
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def choose_device(requested: str = 'auto') -> torch.device:
    """Choose one CUDA device or CPU without assuming pooled GPU memory.

    Args:
        requested: 'auto', 'cpu', or a PyTorch device string such as 'cuda:0'.
    Returns:
        A device; unavailable CUDA requests fall back to CPU.
    """
    if requested == 'auto':
        requested = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if requested.startswith('cuda') and not torch.cuda.is_available():
        requested = 'cpu'
    return torch.device(requested)
