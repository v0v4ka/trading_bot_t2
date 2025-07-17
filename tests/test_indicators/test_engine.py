import pandas as pd
from src.indicators.engine import IndicatorsEngine


def test_engine_all():
    df = pd.DataFrame(
        {
            "High": [1, 2, 5, 2, 1, 2, 5, 2, 1, 2, 5, 2, 1],
            "Low": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "Close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        }
    )
    df.index = pd.date_range("2024-01-01", periods=len(df))
    engine = IndicatorsEngine(df)
    results = engine.calculate_all()
    assert "fractals" in results
    assert "alligator" in results
    assert "awesome_oscillator" in results
    assert "ao_zero_crossings" in results
