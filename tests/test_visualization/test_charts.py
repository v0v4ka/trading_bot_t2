"""
Tests for visualization chart creation and functionality.
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.backtesting.metrics import BacktestResults, Trade
from src.data.models import OHLCV, OHLCVSeries
from src.visualization.charts import ChartVisualizer, create_quick_chart
from src.visualization.config import ChartConfig, ChartTheme


class TestChartVisualizer:
    """Test ChartVisualizer class functionality."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
        np.random.seed(42)  # For reproducible results

        # Generate realistic price data
        base_price = 100.0
        data = []

        for i, date in enumerate(dates):
            # Simple random walk with some trend
            if i == 0:
                open_price = base_price
            else:
                open_price = data[-1]["Close"]

            # Random daily movement
            daily_change = np.random.normal(0, 2)
            close_price = open_price + daily_change

            # High and low based on open/close
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 1))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 1))

            volume = np.random.randint(1000, 10000)

            data.append(
                {
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                    "Volume": volume,
                }
            )

        df = pd.DataFrame(data, index=dates)
        return df

    @pytest.fixture
    def sample_backtest_results(self, sample_ohlcv_data):
        """Create sample backtesting results."""
        trades = [
            Trade(
                entry_time=sample_ohlcv_data.index[5],
                exit_time=sample_ohlcv_data.index[10],
                entry_price=sample_ohlcv_data.iloc[5]["Close"],
                exit_price=sample_ohlcv_data.iloc[10]["Close"],
                quantity=100,
                trade_type="BUY",
                transaction_cost=5.0,
                stop_loss_triggered=False,
            ),
            Trade(
                entry_time=sample_ohlcv_data.index[15],
                exit_time=sample_ohlcv_data.index[20],
                entry_price=sample_ohlcv_data.iloc[15]["Close"],
                exit_price=sample_ohlcv_data.iloc[20]["Close"],
                quantity=150,
                trade_type="SELL",
                transaction_cost=7.5,
                stop_loss_triggered=False,
            ),
        ]

        # Sample equity curve
        equity_curve = [
            (sample_ohlcv_data.index[0], 10000),
            (sample_ohlcv_data.index[10], 10500),
            (sample_ohlcv_data.index[20], 11000),
            (sample_ohlcv_data.index[-1], 11200),
        ]

        results = BacktestResults(config={"initial_capital": 10000})
        results.trades = trades
        results.equity_curve = equity_curve

        return results

    def test_chart_visualizer_initialization_default(self):
        """Test ChartVisualizer initialization with defaults."""
        visualizer = ChartVisualizer()

        assert isinstance(visualizer.config, ChartConfig)
        assert visualizer.config.figure_size == (16, 12)
        assert visualizer.config.dpi == 100

    def test_chart_visualizer_initialization_with_theme(self):
        """Test ChartVisualizer initialization with theme."""
        visualizer = ChartVisualizer(theme=ChartTheme.DARK)

        assert visualizer.config.background_color == "#2e2e2e"
        assert visualizer.config.candle_wick_color == "#cccccc"

    def test_chart_visualizer_initialization_with_custom_config(self):
        """Test ChartVisualizer initialization with custom config."""
        custom_config = ChartConfig(figure_size=(20, 16), dpi=150)
        visualizer = ChartVisualizer(config=custom_config)

        assert visualizer.config.figure_size == (20, 16)
        assert visualizer.config.dpi == 150

    def test_prepare_data_with_dataframe(self, sample_ohlcv_data):
        """Test data preparation with DataFrame input."""
        visualizer = ChartVisualizer()

        # Test with proper DataFrame
        prepared_data = visualizer._prepare_data(sample_ohlcv_data)

        assert isinstance(prepared_data, pd.DataFrame)
        assert list(prepared_data.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(prepared_data.index, pd.DatetimeIndex)

    def test_prepare_data_with_ohlcv_series(self, sample_ohlcv_data):
        """Test data preparation with OHLCVSeries input."""
        visualizer = ChartVisualizer()

        # Create OHLCVSeries from sample data
        ohlcv_list = []
        for timestamp, row in sample_ohlcv_data.iterrows():
            ohlcv_list.append(
                OHLCV(
                    timestamp=timestamp,
                    open=row["Open"],
                    high=row["High"],
                    low=row["Low"],
                    close=row["Close"],
                    volume=row["Volume"],
                )
            )

        ohlcv_series = OHLCVSeries(candles=ohlcv_list)
        prepared_data = visualizer._prepare_data(ohlcv_series)

        assert isinstance(prepared_data, pd.DataFrame)
        assert list(prepared_data.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_prepare_data_missing_columns(self):
        """Test data preparation with missing required columns."""
        visualizer = ChartVisualizer()

        # Create DataFrame with missing columns
        df = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                # Missing Low, Close, Volume
            },
            index=pd.date_range("2024-01-01", periods=3),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            visualizer._prepare_data(df)

    def test_prepare_data_invalid_index(self):
        """Test data preparation with invalid index."""
        visualizer = ChartVisualizer()

        # Create DataFrame without datetime index and no timestamp column
        df = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [95, 96, 97],
                "Close": [103, 104, 105],
                "Volume": [1000, 1100, 1200],
            }
        )  # No datetime index

        with pytest.raises(ValueError, match="Data must have datetime index"):
            visualizer._prepare_data(df)

    @patch("matplotlib.pyplot.subplots")
    def test_create_figure_layout(self, mock_subplots):
        """Test figure layout creation."""
        visualizer = ChartVisualizer()

        # Mock figure and axes
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        result = visualizer._create_figure_layout()

        # Verify subplots was called with correct parameters
        mock_subplots.assert_called_once()
        args, kwargs = mock_subplots.call_args

        assert args == (3, 1)  # 3 subplots, 1 column
        assert kwargs["figsize"] == visualizer.config.figure_size
        assert kwargs["dpi"] == visualizer.config.dpi
        assert "height_ratios" in kwargs["gridspec_kw"]

        assert result == mock_fig

    @patch("src.visualization.charts.alligator")
    def test_plot_alligator_success(self, mock_alligator, sample_ohlcv_data):
        """Test successful Alligator plotting."""
        visualizer = ChartVisualizer()
        mock_ax = Mock()

        # Mock alligator data
        mock_alligator.return_value = {
            "jaw": pd.Series([100, 101, 102], index=sample_ohlcv_data.index[:3]),
            "teeth": pd.Series([99, 100, 101], index=sample_ohlcv_data.index[:3]),
            "lips": pd.Series([98, 99, 100], index=sample_ohlcv_data.index[:3]),
        }

        # Should not raise exception
        visualizer._plot_alligator(mock_ax, sample_ohlcv_data)

        # Verify plot calls were made
        assert mock_ax.plot.call_count == 3  # Three lines (jaw, teeth, lips)

    @patch("src.visualization.charts.alligator")
    def test_plot_alligator_exception_handling(self, mock_alligator, sample_ohlcv_data):
        """Test Alligator plotting exception handling."""
        visualizer = ChartVisualizer()
        mock_ax = Mock()

        # Mock exception in calculation
        mock_alligator.side_effect = Exception("Calculation failed")

        # Should not raise exception, should log warning
        with patch("src.visualization.charts.logger") as mock_logger:
            visualizer._plot_alligator(mock_ax, sample_ohlcv_data)
            mock_logger.warning.assert_called_once()

    @patch("src.visualization.charts.identify_fractals")
    def test_plot_fractals_success(self, mock_identify_fractals, sample_ohlcv_data):
        """Test successful Fractal plotting."""
        visualizer = ChartVisualizer()
        mock_ax = Mock()
        mock_ax.get_legend.return_value = None  # No existing legend

        # Mock fractal data
        mock_identify_fractals.return_value = [
            {
                "timestamp": sample_ohlcv_data.index[5],
                "type": "up",
                "price": sample_ohlcv_data.iloc[5]["High"],
            },
            {
                "timestamp": sample_ohlcv_data.index[10],
                "type": "down",
                "price": sample_ohlcv_data.iloc[10]["Low"],
            },
        ]

        # Should not raise exception
        visualizer._plot_fractals(mock_ax, sample_ohlcv_data)

        # Verify scatter calls were made
        assert mock_ax.scatter.call_count == 2  # Two fractals

    @patch("src.visualization.charts.awesome_oscillator")
    def test_plot_awesome_oscillator_success(self, mock_ao, sample_ohlcv_data):
        """Test successful Awesome Oscillator plotting."""
        visualizer = ChartVisualizer()
        mock_ax = Mock()

        # Mock AO data
        ao_values = np.random.normal(0, 1, len(sample_ohlcv_data))
        mock_ao.return_value = pd.Series(ao_values, index=sample_ohlcv_data.index)

        # Should not raise exception
        visualizer._plot_awesome_oscillator(mock_ax, sample_ohlcv_data)

        # Verify bar plot and axhline calls
        mock_ax.bar.assert_called_once()
        mock_ax.axhline.assert_called_once_with(
            y=0, color="black", linestyle="-", linewidth=1
        )

    def test_plot_agent_decisions(self, sample_ohlcv_data, sample_backtest_results):
        """Test plotting agent decisions."""
        visualizer = ChartVisualizer()
        mock_ax = Mock()
        mock_ax.get_legend.return_value = None  # No existing legend

        # Should not raise exception
        visualizer._plot_agent_decisions(
            mock_ax, sample_ohlcv_data, sample_backtest_results
        )

        # Verify scatter calls for each trade
        assert mock_ax.scatter.call_count == len(sample_backtest_results.trades)

    def test_plot_trade_annotations(self, sample_ohlcv_data, sample_backtest_results):
        """Test plotting trade annotations."""
        visualizer = ChartVisualizer()
        mock_ax = Mock()

        # Should not raise exception
        visualizer._plot_trade_annotations(
            mock_ax, sample_ohlcv_data, sample_backtest_results
        )

        # Verify plot and annotate calls for each trade
        assert mock_ax.plot.call_count == len(sample_backtest_results.trades)
        assert mock_ax.annotate.call_count == len(sample_backtest_results.trades)

    def test_plot_equity_curve(self, sample_backtest_results):
        """Test plotting equity curve."""
        visualizer = ChartVisualizer()
        mock_ax = Mock()

        # Should not raise exception
        visualizer._plot_equity_curve(mock_ax, sample_backtest_results)

        # Verify plot calls
        mock_ax.plot.assert_called_once()
        mock_ax.axhline.assert_called_once()

    def test_plot_equity_curve_no_data(self):
        """Test plotting equity curve with no data."""
        visualizer = ChartVisualizer()
        mock_ax = Mock()

        # Create results with no equity curve
        results = BacktestResults(config={})
        results.equity_curve = []

        with patch("src.visualization.charts.logger") as mock_logger:
            visualizer._plot_equity_curve(mock_ax, results)
            mock_logger.warning.assert_called_once()

    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.setp")
    def test_configure_chart_appearance(
        self, mock_setp, mock_tight_layout, sample_ohlcv_data
    ):
        """Test chart appearance configuration."""
        visualizer = ChartVisualizer()

        # Create mock figure with axes
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_fig.axes = mock_axes

        # Mock legend handles/labels
        for ax in mock_axes:
            ax.get_legend_handles_labels.return_value = ([Mock()], ["Test Label"])

        # Should not raise exception
        visualizer._configure_chart_appearance(
            mock_fig, sample_ohlcv_data, "Test Title"
        )

        # Verify suptitle was set
        mock_fig.suptitle.assert_called_once_with(
            "Test Title", fontsize=16, weight="bold"
        )

        # Verify tight_layout was called
        mock_tight_layout.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    def test_save_chart(self, mock_savefig):
        """Test chart saving functionality."""
        visualizer = ChartVisualizer()
        mock_fig = Mock()
        save_path = Path("/tmp/test_chart.png")

        # Mock path operations
        with patch.object(Path, "mkdir") as mock_mkdir:
            visualizer._save_chart(mock_fig, save_path)

            # Verify mkdir was called on parent
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify savefig was called with correct parameters
        mock_fig.savefig.assert_called_once()
        args, kwargs = mock_fig.savefig.call_args
        assert args[0] == save_path
        assert kwargs["dpi"] == visualizer.config.export_dpi

    @patch("src.visualization.charts.ChartVisualizer.create_integrated_chart")
    def test_create_backtesting_summary_chart(
        self, mock_create_chart, sample_ohlcv_data, sample_backtest_results
    ):
        """Test backtesting summary chart creation."""
        visualizer = ChartVisualizer()
        mock_fig = Mock()
        mock_ax = Mock()
        mock_fig.axes = [mock_ax]  # Add axes attribute
        mock_create_chart.return_value = mock_fig

        original_size = visualizer.config.figure_size

        result = visualizer.create_backtesting_summary_chart(
            sample_ohlcv_data, sample_backtest_results, "Test Summary"
        )

        # Verify the chart was created
        mock_create_chart.assert_called_once()

        # Verify figure size was restored
        assert visualizer.config.figure_size == original_size

        assert result == mock_fig

    def test_create_integrated_chart_empty_data(self):
        """Test create integrated chart with empty data."""
        visualizer = ChartVisualizer()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Missing required columns"):
            visualizer.create_integrated_chart(empty_df)


class TestCreateQuickChart:
    """Test the create_quick_chart utility function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        data = {
            "Open": [100 + i for i in range(10)],
            "High": [105 + i for i in range(10)],
            "Low": [95 + i for i in range(10)],
            "Close": [103 + i for i in range(10)],
            "Volume": [1000 + i * 100 for i in range(10)],
        }
        return pd.DataFrame(data, index=dates)

    @patch("src.visualization.charts.ChartVisualizer.create_integrated_chart")
    def test_create_quick_chart_basic(self, mock_create_chart, sample_data):
        """Test basic quick chart creation."""
        mock_fig = Mock()
        mock_create_chart.return_value = mock_fig

        result = create_quick_chart(sample_data, title="Test Chart")

        # Verify ChartVisualizer was created and used
        mock_create_chart.assert_called_once_with(sample_data, None, "Test Chart", None)
        assert result == mock_fig

    @patch("src.visualization.charts.ChartVisualizer.create_integrated_chart")
    def test_create_quick_chart_with_all_params(self, mock_create_chart, sample_data):
        """Test quick chart creation with all parameters."""
        mock_fig = Mock()
        mock_create_chart.return_value = mock_fig

        mock_results = Mock()
        save_path = "/tmp/chart.png"

        result = create_quick_chart(
            sample_data,
            backtest_results=mock_results,
            title="Full Test",
            theme=ChartTheme.DARK,
            save_path=save_path,
        )

        mock_create_chart.assert_called_once_with(
            sample_data, mock_results, "Full Test", save_path
        )
        assert result == mock_fig
