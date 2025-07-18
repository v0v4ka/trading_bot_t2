from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_file(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if file_path.suffix in {".yaml", ".yml"}:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_file(data: Dict[str, Any], path: str) -> None:
    file_path = Path(path)
    if file_path.suffix in {".yaml", ".yml"}:
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def print_error(message: str) -> None:
    print(f"Error: {message}", flush=True)
