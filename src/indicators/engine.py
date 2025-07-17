"""
Indicators Engine: Unified interface for Bill Williams indicators.
"""

from typing import Any, Dict

import pandas as pd

from .alligator import alligator
from .awesome_oscillator import ao_zero_crossings, awesome_oscillator
from .fractals import identify_fractals


class IndicatorsEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def calculate_all(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        results["fractals"] = identify_fractals(self.df)
        results["alligator"] = alligator(self.df)
        ao = awesome_oscillator(self.df)
        results["awesome_oscillator"] = ao
        results["ao_zero_crossings"] = ao_zero_crossings(ao)
        return results
