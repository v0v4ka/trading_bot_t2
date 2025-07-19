from __future__ import annotations

import argparse
import importlib.metadata
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from src import __version__
from src.cli.config import load_config, validate_config
from src.data.data_provider import DataProvider
from src.backtesting.config import BacktestConfig
from src.backtesting.engine import BacktestEngine
from src.visualization.charts import ChartVisualizer, create_quick_chart
from src.visualization.config import ChartTheme, BACKTESTING_CONFIG, PRESENTATION_CONFIG


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
    backtest_parser.add_argument("--symbol", default="AAPL")
    backtest_parser.add_argument("--start")
    backtest_parser.add_argument("--end")
    backtest_parser.add_argument("--days", type=int, default=60, help="Number of days to backtest")
    backtest_parser.add_argument("--initial-capital", type=float, default=10000.0)

    # Enhanced visualization parser
    visualize_parser = subparsers.add_parser("visualize", help="Create trading charts")
    visualize_parser.add_argument("--symbol", default="AAPL", help="Trading symbol")
    visualize_parser.add_argument("--days", type=int, default=60, help="Number of days of data")
    visualize_parser.add_argument("--theme", choices=[t.value for t in ChartTheme], default="light", help="Chart theme")
    visualize_parser.add_argument("--output", "-o", help="Output file path")
    visualize_parser.add_argument("--title", help="Chart title")
    visualize_parser.add_argument("--backtest", action="store_true", help="Include backtesting results")
    visualize_parser.add_argument("--width", type=int, default=16, help="Chart width")
    visualize_parser.add_argument("--height", type=int, default=12, help="Chart height")

    analyze_parser = subparsers.add_parser("analyze", help="Analysis placeholder")
    logs_parser = subparsers.add_parser("logs", help="Show decision logs")

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
        print("Running backtesting...")
        
        # Set test mode for synthetic data
        os.environ['TBOT_TEST_MODE'] = '1'
        
        # Create configuration
        start_date = datetime.now() - timedelta(days=args.days)
        end_date = datetime.now() - timedelta(days=1)
        
        config = BacktestConfig(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.initial_capital,
            signal_confidence_threshold=0.7,
            decision_confidence_threshold=0.6
        )
        
        # Run backtest
        engine = BacktestEngine(config)
        results = engine.run_backtest()
        
        # Display results
        print(f"\nBacktesting Results for {args.symbol}:")
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Total Trades: {len(results.trades)}")
        
        if results.trades:
            winning_trades = sum(1 for trade in results.trades 
                               if (trade.exit_price - trade.entry_price) * 
                                  (1 if trade.trade_type == 'BUY' else -1) > 0)
            win_rate = (winning_trades / len(results.trades)) * 100
            print(f"Winning Trades: {winning_trades}")
            print(f"Win Rate: {win_rate:.1f}%")
            
            if results.equity_curve:
                final_value = results.equity_curve[-1][1]
                total_return = ((final_value - args.initial_capital) / args.initial_capital) * 100
                print(f"Final Portfolio Value: ${final_value:,.2f}")
                print(f"Total Return: {total_return:.2f}%")
        
        return

    if args.command == "visualize":
        print(f"Creating chart for {args.symbol}...")
        
        # Set test mode for synthetic data
        os.environ['TBOT_TEST_MODE'] = '1'
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"charts/{args.symbol}_{timestamp}.png")
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get data
        start_date = datetime.now() - timedelta(days=args.days)
        end_date = datetime.now() - timedelta(days=1)
        
        provider = DataProvider(
            symbol=args.symbol,
            interval="1d",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d")
        )
        
        data_series = provider.fetch()
        df = data_series.to_dataframe()
        
        # Get backtest results if requested
        backtest_results = None
        if args.backtest:
            print("Running backtesting for chart annotations...")
            config = BacktestConfig(
                symbol=args.symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=10000.0,
                signal_confidence_threshold=0.7,
                decision_confidence_threshold=0.6
            )
            
            engine = BacktestEngine(config)
            backtest_results = engine.run_backtest()
            print(f"  Generated {len(backtest_results.trades)} trades for visualization")
        
        # Determine chart title
        if args.title:
            title = args.title
        else:
            period_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            if args.backtest and backtest_results:
                title = f"{args.symbol} Trading Analysis - {period_str} ({len(backtest_results.trades)} trades)"
            else:
                title = f"{args.symbol} Price Chart - {period_str}"
        
        # Parse theme
        theme = ChartTheme(args.theme)
        
        # Create chart with custom dimensions
        from src.visualization.config import ChartConfig
        config = ChartConfig(
            figure_size=(args.width, args.height)
        )
        
        if theme != ChartTheme.LIGHT:
            from src.visualization.config import get_theme_config
            config = get_theme_config(theme)
            config.figure_size = (args.width, args.height)
        
        visualizer = ChartVisualizer(config=config)
        
        if args.backtest and backtest_results:
            fig = visualizer.create_backtesting_summary_chart(
                data=df,
                backtest_results=backtest_results,
                title=title,
                save_path=output_path
            )
        else:
            fig = visualizer.create_integrated_chart(
                data=df,
                title=title,
                save_path=output_path
            )
        
        print(f"Chart saved to: {output_path}")
        
        # Display chart info
        print(f"Chart Details:")
        print(f"  Symbol: {args.symbol}")
        print(f"  Data Points: {len(df)}")
        print(f"  Theme: {args.theme}")
        print(f"  Dimensions: {args.width}x{args.height}")
        if args.backtest and backtest_results:
            print(f"  Trades Plotted: {len(backtest_results.trades)}")
        
        return

    if args.command in {"analyze", "logs"}:
        print(f"Command '{args.command}' not yet implemented")
        return


if __name__ == "__main__":
    main()
