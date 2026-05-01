from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Any


class CsvLogger:
    """Append-only CSV logger for epoch-level metrics."""

    def __init__(self, path: str | Path, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        safe_row = {k: row.get(k, "") for k in self.fieldnames}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(safe_row)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
