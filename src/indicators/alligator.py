"""
Alligator indicator implementation for Bill Williams Trading Chaos system.
"""
from typing import Dict
import pandas as pd
import numpy as np

def smma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Smoothed Moving Average (SMMA)."""
    smma = pd.Series(index=series.index, dtype=float)
    smma.iloc[:period] = series.iloc[:period].mean()
    
    for i in range(period, len(series)):
        smma.iloc[i] = (smma.iloc[i-1] * (period - 1) + series.iloc[i]) / period
    
    return smma

def alligator(df: pd.DataFrame) -> Dict[str, pd.Series]:
    jaw = smma(df['Close'], 13).shift(8)
    teeth = smma(df['Close'], 8).shift(5)
    lips = smma(df['Close'], 5).shift(3)
    return {
        'jaw': jaw,
        'teeth': teeth,
        'lips': lips
    }
