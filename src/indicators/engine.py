"""
Indicators Engine: Unified interface for Bill Williams indicators.
"""
from typing import Dict, Any
import pandas as pd
from .fractals import identify_fractals
from .alligator import alligator
from .awesome_oscillator import awesome_oscillator, ao_zero_crossings

class IndicatorsEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def calculate_all(self) -> Dict[str, Any]:
        results = {}
        results['fractals'] = identify_fractals(self.df)
        results['alligator'] = alligator(self.df)
        ao = awesome_oscillator(self.df)
        results['awesome_oscillator'] = ao
        results['ao_zero_crossings'] = ao_zero_crossings(ao)
        return results
