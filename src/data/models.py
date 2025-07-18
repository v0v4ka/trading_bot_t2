from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic.v1 import BaseModel, root_validator, validator


class OHLCV(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @validator("open", "high", "low", "close", "volume")
    def check_positive(cls, v, field):
        if v < 0:
            raise ValueError(f"{field.name} must be non-negative")
        return v


class OHLCVSeries(BaseModel):
    candles: List[OHLCV]

    @root_validator(pre=True)
    def check_chronological(cls, values):
        candles = values.get("candles", [])
        timestamps = [c.timestamp for c in candles]
        if timestamps != sorted(timestamps):
            raise ValueError("Candles must be in chronological order")
        return values

    def detect_missing(self) -> List[datetime]:
        # Placeholder: implement missing data detection based on expected interval
        return []

    def detect_outliers(self) -> List[OHLCV]:
        # Placeholder: implement outlier detection (e.g., z-score)
        return []

    def to_dataframe(self) -> pd.DataFrame:
        """Convert OHLCVSeries to pandas DataFrame."""
        data = []
        for candle in self.candles:
            data.append(
                {
                    "timestamp": candle.timestamp,
                    "Open": candle.open,
                    "High": candle.high,
                    "Low": candle.low,
                    "Close": candle.close,
                    "Volume": candle.volume,
                }
            )
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df
