"""
Simple visualization test script.

This script tests the visualization system without requiring backtesting
or OpenAI API keys.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Set up environment BEFORE any other imports
os.environ["TBOT_TEST_MODE"] = "1"

# Import matplotlib first to avoid logging conflicts
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

# Add src to path AFTER matplotlib import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.data_provider import DataProvider
from src.visualization.charts import create_quick_chart
from src.visualization.config import ChartTheme


def test_basic_visualization():
    """Test basic visualization without backtesting."""
    print("Testing basic visualization...")

    # Get sample data
    provider = DataProvider(
        symbol="AAPL",
        interval="1d",
        start=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        end=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    data_series = provider.fetch()
    df = data_series.to_dataframe()

    print(f"Loaded {len(df)} data points")

    # Create charts with different themes
    themes = [
        (ChartTheme.LIGHT, "light_theme_test.png"),
        (ChartTheme.DARK, "dark_theme_test.png"),
        (ChartTheme.PROFESSIONAL, "professional_theme_test.png"),
        (ChartTheme.COLORBLIND_FRIENDLY, "colorblind_theme_test.png"),
    ]

    # Create output directory
    Path("outputs/charts").mkdir(parents=True, exist_ok=True)

    for theme, filename in themes:
        print(f"Creating {theme.value} theme chart...")

        fig = create_quick_chart(
            data=df,
            title=f"AAPL - {theme.value.title()} Theme Test",
            theme=theme,
            save_path=f"outputs/charts/{filename}",
        )

        print(f"  Saved: outputs/charts/{filename}")

    print("\nVisualization test completed successfully!")
    print("Generated charts in outputs/charts/ directory")

    return True


if __name__ == "__main__":
    try:
        test_basic_visualization()
        print("✅ All visualization tests passed!")
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
