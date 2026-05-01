from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    """Save a resolved configuration for experiment reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def get_output_dir(cfg: Dict[str, Any]) -> Path:
    out = Path(cfg["project"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out
