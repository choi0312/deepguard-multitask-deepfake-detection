from __future__ import annotations

import platform
from typing import Dict, Any

import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def system_info() -> Dict[str, Any]:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    return info
