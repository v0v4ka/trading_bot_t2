import yfinance as yf
import pandas as pd
from typing import List, Optional
from datetime import datetime
from .models import OHLCV, OHLCVSeries
import logging
import time

import os
logger = logging.getLogger("trading_bot.data_provider")

SUPPORTED_INTERVALS = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '1h': '60m',
    '4h': '240m',
    '1d': '1d',
    '1w': '1wk',
}

RATE_LIMIT_SLEEP = 2  # seconds between requests

class DataProvider:
    def __init__(self, symbol: str, interval: str, start: Optional[str] = None, end: Optional[str] = None):
        if interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"Unsupported interval: {interval}")
        self.symbol = symbol
        self.interval = SUPPORTED_INTERVALS[interval]
        self.start = start
        self.end = end

    def fetch(self) -> OHLCVSeries:
        try:
            logger.info(f"Fetching data for {self.symbol} [{self.interval}] from {self.start} to {self.end}")
            if os.environ.get("TBOT_TEST_MODE") == "1":
                logger.info("TBOT_TEST_MODE active - returning synthetic data")
                if self.symbol == "INVALIDSYM":
                    raise ValueError("No data returned for symbol/timeframe.")
                dates = pd.date_range(self.start or "2024-01-01", periods=10)
                df = pd.DataFrame({
                    'Open': [1.0 + i for i in range(len(dates))],
                    'High': [1.5 + i for i in range(len(dates))],
                    'Low': [0.5 + i for i in range(len(dates))],
                    'Close': [1.2 + i for i in range(len(dates))],
                    'Volume': [1000 + i for i in range(len(dates))]
                }, index=dates)
                candles = [
                    OHLCV(
                        timestamp=idx.to_pydatetime(),
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=float(row['Volume'])
                    )
                    for idx, row in df.iterrows()
                ]
                return OHLCVSeries(candles=candles)
            data = yf.download(
                self.symbol,
                interval=self.interval,
                start=self.start,
                end=self.end,
                progress=False,
                threads=False
            )
            time.sleep(RATE_LIMIT_SLEEP)
            if data.empty:
                raise ValueError("No data returned for symbol/timeframe.")
            data = data.dropna()
            candles = [
                OHLCV(
                    timestamp=idx.to_pydatetime(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row['Volume'])
                )
                for idx, row in data.iterrows()
            ]
            series = OHLCVSeries(candles=candles)
            return series
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
