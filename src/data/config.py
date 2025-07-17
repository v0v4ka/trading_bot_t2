import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Trading Bot Data Fetcher")
    parser.add_argument(
        "--symbol", type=str, required=True, help="Ticker symbol (e.g. AAPL)"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        choices=["1m", "5m", "15m", "1h", "4h", "1d", "1w"],
        help="Timeframe interval",
    )
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    return parser.parse_args()
