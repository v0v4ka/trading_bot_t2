import pytest
from src.data.data_provider import DataProvider
from src.data.models import OHLCVSeries
import os

def test_fetch_valid_symbol(monkeypatch):
    provider = DataProvider(symbol="AAPL", interval="1d", start="2024-01-01", end="2024-01-10")
    series = provider.fetch()
    assert isinstance(series, OHLCVSeries)
    assert len(series.candles) > 0

def test_fetch_invalid_symbol():
    provider = DataProvider(symbol="INVALIDSYM", interval="1d", start="2024-01-01", end="2024-01-10")
    with pytest.raises(Exception):
        provider.fetch()
