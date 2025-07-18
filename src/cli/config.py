import json
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {"source": "yahoo", "interval": "1d"},
    "backtest": {"start": "2024-01-01", "end": "2024-12-31"},
}


def load_config(path: str | None = None) -> Dict[str, Any]:
    if path is None:
        return DEFAULT_CONFIG.copy()
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if file_path.suffix in {".yaml", ".yml"}:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    if file_path.suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError("Unsupported config file format")


def validate_config(config: Dict[str, Any]) -> bool:
    if "data" not in config or "backtest" not in config:
        return False
    return True
