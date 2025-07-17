import pandas as pd
from src.indicators.fractals import identify_fractals


def test_fractals_basic():
    data = {
        "High": [1, 2, 5, 2, 1, 2, 5, 2, 1],
        "Low": [1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range("2024-01-01", periods=len(df))
    results = identify_fractals(df)
    assert any(r["type"] == "up" for r in results)
    assert all("timestamp" in r and "price" in r for r in results)
