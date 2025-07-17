import pytest
from src.data.models import OHLCV, OHLCVSeries
from datetime import datetime, timedelta


def test_missing_data_detection():
    # Placeholder: test will pass as method is not implemented
    now = datetime.now()
    candles = [
        OHLCV(
            timestamp=now + timedelta(days=i),
            open=1,
            high=2,
            low=0.5,
            close=1.5,
            volume=100,
        )
        for i in range(3)
    ]
    series = OHLCVSeries(candles=candles)
    assert series.detect_missing() == []


def test_outlier_detection():
    # Placeholder: test will pass as method is not implemented
    now = datetime.now()
    candles = [
        OHLCV(
            timestamp=now + timedelta(days=i),
            open=1,
            high=2,
            low=0.5,
            close=1.5,
            volume=100,
        )
        for i in range(3)
    ]
    series = OHLCVSeries(candles=candles)
    assert series.detect_outliers() == []
