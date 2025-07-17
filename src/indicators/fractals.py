"""
Fractals indicator implementation for Bill Williams Trading Chaos system.
"""

from typing import List, Dict
import pandas as pd


def identify_fractals(df: pd.DataFrame) -> List[Dict]:
    """
    Identify up and down fractals in OHLCV DataFrame.
    Returns a list of dicts: {'type': 'up'|'down', 'timestamp': ..., 'price': ...}
    """
    results = []
    highs = df["High"].values
    lows = df["Low"].values
    for i in range(2, len(df) - 2):
        # Up fractal
        if (
            highs[i] > highs[i - 2]
            and highs[i] > highs[i - 1]
            and highs[i] > highs[i + 1]
            and highs[i] > highs[i + 2]
        ):
            results.append({"type": "up", "timestamp": df.index[i], "price": highs[i]})
        # Down fractal
        if (
            lows[i] < lows[i - 2]
            and lows[i] < lows[i - 1]
            and lows[i] < lows[i + 1]
            and lows[i] < lows[i + 2]
        ):
            results.append({"type": "down", "timestamp": df.index[i], "price": lows[i]})
    return results
