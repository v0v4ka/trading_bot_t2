from __future__ import annotations

import argparse
import importlib.metadata
from typing import List

from src import __version__
from src.cli.config import load_config, validate_config
from src.data.data_provider import DataProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trading Bot CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument("--file", help="Path to config file")
    config_parser.add_argument("--show", action="store_true", help="Show config")

    data_parser = subparsers.add_parser("data", help="Fetch market data")
    data_parser.add_argument("--symbol", required=True)
    data_parser.add_argument(
        "--interval",
        default="1d",
        choices=["1m", "5m", "15m", "1h", "4h", "1d", "1w"],
    )
    data_parser.add_argument("--start")
    data_parser.add_argument("--end")

    subparsers.add_parser("version", help="Show CLI version")

    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--config")
    backtest_parser.add_argument("--symbol")
    backtest_parser.add_argument("--start")
    backtest_parser.add_argument("--end")

    subparsers.add_parser("visualize", help="Visualization placeholder")
    subparsers.add_parser("analyze", help="Analysis placeholder")
    subparsers.add_parser("logs", help="Show decision logs")

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(f"Trading Bot CLI {__version__}")
        return

    if args.command == "config":
        cfg = load_config(args.file)
        if not validate_config(cfg):
            print("Invalid configuration")
            return
        if args.show:
            print(cfg)
        else:
            print("Configuration loaded")
        return

    if args.command == "data":
        provider = DataProvider(
            symbol=args.symbol,
            interval=args.interval,
            start=args.start,
            end=args.end,
        )
        series = provider.fetch()
        print(f"Fetched {len(series.candles)} candles")
        return

    if args.command == "backtest":
        print("Backtesting not yet implemented")
        return

    if args.command in {"visualize", "analyze", "logs"}:
        print(f"Command '{args.command}' not yet implemented")
        return


if __name__ == "__main__":
    main()
