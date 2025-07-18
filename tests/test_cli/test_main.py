import os
from importlib import reload

import pytest

from src.cli.main import main as cli_main


def test_version_output(capsys):
    cli_main(["version"])
    captured = capsys.readouterr()
    assert "Trading Bot CLI" in captured.out


def test_data_command(monkeypatch, capsys):
    monkeypatch.setenv("TBOT_TEST_MODE", "1")
    cli_main(
        [
            "data",
            "--symbol",
            "AAPL",
            "--interval",
            "1d",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-10",
        ]
    )
    captured = capsys.readouterr()
    assert "Fetched" in captured.out


def test_help_flag(capsys):
    with pytest.raises(SystemExit):
        cli_main(["-h"])
    captured = capsys.readouterr()
    assert "Trading Bot CLI" in captured.out
