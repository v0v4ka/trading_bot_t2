"""
Integration example demonstrating end-to-end backtesting workflow.

This module shows how to use the backtesting framework with the existing
Signal Detection and Decision Maker agents for historical simulation.
"""

import os
import sys
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.backtesting.config import (
    BacktestConfig,
    create_custom_config,
    get_scenario_config,
)
from src.backtesting.engine import BacktestEngine, run_simple_backtest
from src.backtesting.metrics import PerformanceAnalyzer
from src.logging.decision_logger import DecisionLogger


def run_basic_backtest_example():
    """Run a basic backtesting example with default configuration."""
    print("=" * 60)
    print("BASIC BACKTESTING EXAMPLE")
    print("=" * 60)

    # Create configuration
    config = BacktestConfig(
        symbol="AAPL",
        start_date=datetime.now() - timedelta(days=90),  # Last 3 months
        end_date=datetime.now(),
        initial_capital=10000.0,
        position_size=1000.0,
        signal_confidence_threshold=0.7,
        decision_confidence_threshold=0.75,
    )

    print(f"Configuration:")
    print(f"  Symbol: {config.symbol}")
    print(f"  Date Range: {config.start_date.date()} to {config.end_date.date()}")
    print(f"  Initial Capital: ${config.initial_capital:,.2f}")
    print(f"  Position Size: ${config.position_size:,.2f}")
    print(f"  Signal Threshold: {config.signal_confidence_threshold}")
    print(f"  Decision Threshold: {config.decision_confidence_threshold}")
    print()

    try:
        # Initialize and run backtest
        print("Initializing backtesting engine...")
        engine = BacktestEngine(config)

        print("Running backtest...")
        results = engine.run_backtest()

        # Generate performance report
        print("\nGenerating performance analysis...")
        analyzer = engine.get_performance_analyzer()
        metrics = analyzer.calculate_metrics()

        # Display summary
        print("\n" + "=" * 60)
        print("BACKTESTING RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Winning Trades: {metrics['winning_trades']}")
        print(f"Losing Trades: {metrics['losing_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Total Return: ${metrics['total_return']:.2f}")
        print(f"Total Return %: {metrics['total_return_percentage']:.2f}%")
        print(f"Final Capital: ${metrics['final_capital']:.2f}")

        if metrics["total_trades"] > 0:
            print(f"Average Trade Return: {metrics['average_trade_return']:.2f}%")
            print(f"Best Trade: ${metrics['best_trade']:.2f}")
            print(f"Worst Trade: ${metrics['worst_trade']:.2f}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"Max Drawdown: ${metrics['max_drawdown']:.2f}")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")

        print("=" * 60)

        return results, analyzer

    except Exception as e:
        print(f"Backtesting failed: {str(e)}")
        print("This is expected in the demo - check data availability and API keys")
        return None, None


def run_scenario_comparison_example():
    """Run backtesting with different predefined scenarios."""
    print("\n" + "=" * 60)
    print("SCENARIO COMPARISON EXAMPLE")
    print("=" * 60)

    scenarios = ["conservative", "balanced", "aggressive"]
    scenario_results = {}

    for scenario_name in scenarios:
        print(f"\nRunning {scenario_name} scenario...")

        try:
            # Get scenario configuration
            scenario = get_scenario_config(scenario_name)
            config = scenario.config

            # Update date range for demo
            config.start_date = datetime.now() - timedelta(days=60)
            config.end_date = datetime.now()

            print(f"  Signal Threshold: {config.signal_confidence_threshold}")
            print(f"  Decision Threshold: {config.decision_confidence_threshold}")
            print(f"  Position Size: ${config.position_size}")
            print(f"  Stop Loss: {config.stop_loss_percentage:.1%}")

            # Run backtest
            engine = BacktestEngine(config)
            results = engine.run_backtest()
            analyzer = engine.get_performance_analyzer()
            metrics = analyzer.calculate_metrics()

            scenario_results[scenario_name] = {
                "config": config,
                "results": results,
                "metrics": metrics,
            }

            print(f"  Total Trades: {metrics['total_trades']}")
            print(f"  Win Rate: {metrics['win_rate']:.2%}")
            print(
                f"  Total Return: ${metrics['total_return']:.2f} ({metrics['total_return_percentage']:.2f}%)"
            )

        except Exception as e:
            print(f"  Failed: {str(e)}")
            scenario_results[scenario_name] = None

    # Compare results
    print("\n" + "=" * 60)
    print("SCENARIO COMPARISON")
    print("=" * 60)
    print(
        f"{'Scenario':<12} {'Trades':<8} {'Win Rate':<10} {'Return %':<10} {'Sharpe':<8}"
    )
    print("-" * 60)

    for scenario_name in scenarios:
        if scenario_results[scenario_name]:
            metrics = scenario_results[scenario_name]["metrics"]
            print(
                f"{scenario_name:<12} {metrics['total_trades']:<8} "
                f"{metrics['win_rate']:<10.2%} "
                f"{metrics['total_return_percentage']:<10.2f}% "
                f"{metrics['sharpe_ratio']:<8.3f}"
            )
        else:
            print(f"{scenario_name:<12} {'Failed':<8}")

    print("=" * 60)

    return scenario_results


def run_custom_configuration_example():
    """Run backtesting with custom configuration."""
    print("\n" + "=" * 60)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("=" * 60)

    # Create custom configuration with specific parameters
    config = create_custom_config(
        symbol="MSFT",
        start_date=datetime(2023, 6, 1),
        end_date=datetime(2023, 9, 1),
        initial_capital=25000.0,
        position_size=2500.0,
        signal_confidence_threshold=0.65,
        decision_confidence_threshold=0.75,
        stop_loss_percentage=0.04,  # 4% stop loss
        max_position_size=0.15,  # 15% max position size
        use_llm_confirmation=True,
    )

    print(f"Custom Configuration:")
    print(f"  Symbol: {config.symbol}")
    print(f"  Period: {config.start_date.date()} to {config.end_date.date()}")
    print(f"  Capital: ${config.initial_capital:,.2f}")
    print(f"  Position Size: ${config.position_size:,.2f}")
    print(f"  Max Position %: {config.max_position_size:.1%}")
    print(f"  Stop Loss: {config.stop_loss_percentage:.1%}")
    print(f"  LLM Confirmation: {config.use_llm_confirmation}")

    try:
        # Run backtest with decision logging
        decision_logger = DecisionLogger()
        engine = BacktestEngine(config, decision_logger=decision_logger)

        print("\nRunning custom backtest...")
        results = engine.run_backtest()

        # Analyze results
        analyzer = engine.get_performance_analyzer()

        # Generate detailed report
        print("\n" + analyzer.generate_report(detailed=True))

        # Show trade summary if any trades occurred
        trade_summary = analyzer.get_trade_summary()
        if not trade_summary.empty:
            print("\nTRADE SUMMARY:")
            print(trade_summary.to_string(index=False))

        return results, analyzer

    except Exception as e:
        print(f"Custom backtest failed: {str(e)}")
        return None, None


def run_simple_backtest_example():
    """Demonstrate the convenience function for quick backtesting."""
    print("\n" + "=" * 60)
    print("SIMPLE BACKTEST FUNCTION EXAMPLE")
    print("=" * 60)

    try:
        print("Running simple backtest for TSLA (last 6 months)...")

        # Use convenience function
        results = run_simple_backtest(
            symbol="TSLA",
            days=180,
            initial_capital=15000.0,
            position_size=1500.0,
            signal_confidence_threshold=0.7,
        )

        # Quick analysis
        analyzer = PerformanceAnalyzer(results, initial_capital=15000.0)
        metrics = analyzer.calculate_metrics()

        print(f"\nQuick Results:")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(
            f"Total Return: ${metrics['total_return']:.2f} ({metrics['total_return_percentage']:.2f}%)"
        )

        if metrics["total_trades"] > 0:
            print(
                f"Average Trade Duration: {metrics['average_trade_duration_hours']:.1f} hours"
            )
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")

        return results

    except Exception as e:
        print(f"Simple backtest failed: {str(e)}")
        return None


def main():
    """Run all backtesting examples."""
    print("BACKTESTING FRAMEWORK INTEGRATION EXAMPLES")
    print("=" * 80)
    print()
    print("This module demonstrates the backtesting framework capabilities.")
    print("Note: Examples may fail without proper data access and API configuration.")
    print()

    # Run examples
    run_basic_backtest_example()
    run_scenario_comparison_example()
    run_custom_configuration_example()
    run_simple_backtest_example()

    print("\n" + "=" * 80)
    print("INTEGRATION EXAMPLES COMPLETED")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("- Basic backtesting configuration and execution")
    print("- Predefined scenario comparison (conservative, balanced, aggressive)")
    print("- Custom configuration with advanced parameters")
    print("- Decision logging integration")
    print("- Performance analysis and reporting")
    print("- Convenience functions for quick backtesting")
    print()
    print("Next Steps:")
    print("- Configure data provider with valid API keys")
    print("- Run backtests with real historical data")
    print("- Integrate with visualization for chart output")
    print("- Add CLI interface for user-friendly execution")


if __name__ == "__main__":
    main()
