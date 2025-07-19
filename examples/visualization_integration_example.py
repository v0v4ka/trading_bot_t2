"""
Visualization Integration Example

This script demonstrates the comprehensive visualization system capabilities,
showing how to create integrated charts with candlesticks, indicators, and
backtesting results.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
# Use non-interactive backend for headless environments
matplotlib.use('Agg')

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_provider import DataProvider
from src.backtesting.config import BacktestConfig
from src.backtesting.engine import BacktestEngine
from src.visualization.charts import ChartVisualizer, create_quick_chart
from src.visualization.config import ChartTheme, BACKTESTING_CONFIG, PRESENTATION_CONFIG


def create_sample_backtest_data():
    """Create sample backtesting data for visualization."""
    print("Creating sample backtesting data...")
    
    # Set test mode to use synthetic data
    os.environ['TBOT_TEST_MODE'] = '1'
    
    # Create backtesting configuration
    config = BacktestConfig(
        symbol="AAPL",
        start_date=datetime.now() - timedelta(days=60),
        end_date=datetime.now() - timedelta(days=1),
        initial_capital=10000.0,
        signal_confidence_threshold=0.7,
        decision_confidence_threshold=0.6
    )
    
    # Run backtest to generate data and results
    engine = BacktestEngine(config)
    results = engine.run_backtest()
    
    # Get the historical data used in backtesting
    data_provider = engine.data_provider
    data_series = data_provider.fetch()
    df = data_series.to_dataframe()
    
    print(f"Generated {len(df)} data points and {len(results.trades)} trades")
    return df, results


def demonstrate_basic_chart():
    """Demonstrate basic chart creation without backtesting results."""
    print("\n=== Basic Chart Demonstration ===")
    
    # Get sample data
    df, _ = create_sample_backtest_data()
    
    # Create basic chart with default settings
    fig = create_quick_chart(
        data=df,
        title="AAPL - Basic Candlestick Chart with Indicators",
        theme=ChartTheme.LIGHT,
        save_path="examples/charts/basic_chart.png"
    )
    
    print("Basic chart saved to: examples/charts/basic_chart.png")
    return fig


def demonstrate_backtesting_chart():
    """Demonstrate comprehensive backtesting chart."""
    print("\n=== Backtesting Chart Demonstration ===")
    
    # Get sample data and results
    df, results = create_sample_backtest_data()
    
    # Create comprehensive backtesting chart
    visualizer = ChartVisualizer(config=BACKTESTING_CONFIG)
    fig = visualizer.create_backtesting_summary_chart(
        data=df,
        backtest_results=results,
        title=f"AAPL Trading Strategy - Backtesting Results ({len(results.trades)} trades)",
        save_path="examples/charts/backtesting_summary.png"
    )
    
    print("Backtesting summary chart saved to: examples/charts/backtesting_summary.png")
    return fig


def demonstrate_theme_variations():
    """Demonstrate different chart themes."""
    print("\n=== Theme Variations Demonstration ===")
    
    # Get sample data
    df, results = create_sample_backtest_data()
    
    themes = [
        (ChartTheme.LIGHT, "Light Theme"),
        (ChartTheme.DARK, "Dark Theme"),
        (ChartTheme.PROFESSIONAL, "Professional Theme"),
        (ChartTheme.COLORBLIND_FRIENDLY, "Colorblind-Friendly Theme")
    ]
    
    for theme, description in themes:
        print(f"Creating {description.lower()}...")
        
        fig = create_quick_chart(
            data=df,
            backtest_results=results,
            title=f"AAPL - {description}",
            theme=theme,
            save_path=f"examples/charts/theme_{theme.value}.png"
        )
        
        print(f"  Saved to: examples/charts/theme_{theme.value}.png")


def demonstrate_custom_configuration():
    """Demonstrate custom chart configuration."""
    print("\n=== Custom Configuration Demonstration ===")
    
    # Get sample data
    df, results = create_sample_backtest_data()
    
    # Create custom configuration for presentation
    custom_config = PRESENTATION_CONFIG
    custom_config.candle_up_color = "#00ff88"  # Custom green
    custom_config.candle_down_color = "#ff4444"  # Custom red
    custom_config.decision_marker_base_size = 250  # Larger markers
    
    visualizer = ChartVisualizer(config=custom_config)
    fig = visualizer.create_integrated_chart(
        data=df,
        backtest_results=results,
        title="AAPL - Custom Configuration Example",
        save_path="examples/charts/custom_config.png"
    )
    
    print("Custom configuration chart saved to: examples/charts/custom_config.png")
    return fig


def demonstrate_performance_analysis():
    """Demonstrate chart with performance analysis."""
    print("\n=== Performance Analysis Demonstration ===")
    
    # Get sample data
    df, results = create_sample_backtest_data()
    
    # Calculate some basic performance metrics
    total_trades = len(results.trades)
    winning_trades = sum(1 for trade in results.trades 
                        if (trade.exit_price - trade.entry_price) * 
                           (1 if trade.trade_type == 'BUY' else -1) > 0)
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    # Create chart with detailed performance information
    visualizer = ChartVisualizer(theme=ChartTheme.PROFESSIONAL)
    fig = visualizer.create_backtesting_summary_chart(
        data=df,
        backtest_results=results,
        title=f"AAPL Strategy Performance Analysis\n"
              f"Trades: {total_trades} | Win Rate: {win_rate:.1f}% | "
              f"Capital: ${results.config.get('initial_capital', 0):,.0f}",
        save_path="examples/charts/performance_analysis.png"
    )
    
    print("Performance analysis chart saved to: examples/charts/performance_analysis.png")
    print(f"  Total Trades: {total_trades}")
    print(f"  Winning Trades: {winning_trades}")
    print(f"  Win Rate: {win_rate:.1f}%")
    
    return fig


def analyze_chart_components():
    """Analyze and describe chart components."""
    print("\n=== Chart Components Analysis ===")
    
    df, results = create_sample_backtest_data()
    
    print("Chart includes the following components:")
    print("1. Main Plot:")
    print("   - Candlestick price data (OHLC)")
    print("   - Alligator indicator lines (Jaw, Teeth, Lips)")
    print("   - Fractal up/down signals")
    print("   - Agent decision markers (BUY/SELL)")
    print("   - Trade P&L annotations")
    
    print("\n2. Awesome Oscillator Subplot:")
    print("   - Histogram showing momentum")
    print("   - Zero-line reference")
    print("   - Decision markers synchronized with main plot")
    
    print("\n3. Equity Curve Subplot:")
    print("   - Portfolio value over time")
    print("   - Initial capital reference line")
    print("   - Performance tracking")
    
    print("\n4. Additional Features:")
    print("   - Performance metrics text box")
    print("   - Configurable themes and styling")
    print("   - Export capabilities (PNG, PDF)")
    print("   - Responsive layout and scaling")


def main():
    """Run all visualization demonstrations."""
    print("Trading Bot Visualization System - Integration Example")
    print("=" * 60)
    
    # Create output directory
    Path("examples/charts").mkdir(parents=True, exist_ok=True)
    
    try:
        # Run demonstrations
        demonstrate_basic_chart()
        demonstrate_backtesting_chart()
        demonstrate_theme_variations()
        demonstrate_custom_configuration()
        demonstrate_performance_analysis()
        analyze_chart_components()
        
        print("\n" + "=" * 60)
        print("All visualization examples completed successfully!")
        print("\nGenerated charts can be found in: examples/charts/")
        print("\nChart files created:")
        
        chart_files = list(Path("examples/charts").glob("*.png"))
        for chart_file in sorted(chart_files):
            print(f"  - {chart_file.name}")
            
    except Exception as e:
        print(f"\nError during visualization demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
