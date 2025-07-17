"""
Awesome Oscillator (AO) implementation for Bill Williams Trading Chaos system.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def awesome_oscillator(df: pd.DataFrame) -> pd.Series:
    median_price = (df["High"] + df["Low"]) / 2
    sma5 = median_price.rolling(window=5).mean()
    sma34 = median_price.rolling(window=34).mean()
    ao = sma5 - sma34
    return ao


def ao_zero_crossings(ao: pd.Series) -> pd.Series:
    return ao.shift(1) * ao < 0
