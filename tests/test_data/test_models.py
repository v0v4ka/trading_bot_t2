import pytest
from src.data.models import OHLCV, OHLCVSeries
from datetime import datetime, timedelta

def test_ohlcv_validation():
    candle = OHLCV(
        timestamp=datetime.now(),
        open=100.0,
        high=110.0,
        low=90.0,
        close=105.0,
        volume=1000.0
    )
    assert candle.open == 100.0
    with pytest.raises(ValueError):
        OHLCV(timestamp=datetime.now(), open=-1, high=1, low=1, close=1, volume=1)

def test_ohlcvseries_chronological():
    now = datetime.now()
    candles = [
        OHLCV(timestamp=now + timedelta(minutes=i), open=1, high=2, low=0.5, close=1.5, volume=100) for i in range(3)
    ]
    series = OHLCVSeries(candles=candles)
    assert len(series.candles) == 3
    # Out of order
    candles_bad = [candles[2], candles[0], candles[1]]
    with pytest.raises(ValueError):
        OHLCVSeries(candles=candles_bad)
