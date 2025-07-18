import json
from pathlib import Path

from src.cli.config import DEFAULT_CONFIG, load_config, validate_config


def test_load_default_config():
    cfg = load_config()
    assert cfg == DEFAULT_CONFIG


def test_load_json_config(tmp_path):
    config_path = tmp_path / "cfg.json"
    sample = {"data": {"source": "demo"}, "backtest": {"start": "2024"}}
    config_path.write_text(json.dumps(sample))
    cfg = load_config(str(config_path))
    assert cfg == sample


def test_validate_config():
    assert validate_config(DEFAULT_CONFIG)
    assert not validate_config({"invalid": True})
