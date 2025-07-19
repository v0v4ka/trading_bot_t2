"""
Comprehensive chart visualization system for trading bot analysis.

This module provides the ChartVisualizer class that creates integrated
candlestick charts with Bill Williams indicators and agent decision points.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from ..backtesting.metrics import BacktestResults, Trade
from ..data.models import OHLCV, OHLCVSeries
from ..indicators.alligator import alligator
from ..indicators.awesome_oscillator import awesome_oscillator
from ..indicators.fractals import identify_fractals
from .config import ChartConfig, ChartTheme, get_theme_config

logger = logging.getLogger("trading_bot.visualization.charts")


class ChartVisualizer:
    """
    Comprehensive chart visualization system for trading analysis.

    This class creates integrated charts combining candlestick price data,
    Bill Williams indicators, and agent decision points from backtesting results.
    """

    def __init__(
        self, config: Optional[ChartConfig] = None, theme: Optional[ChartTheme] = None
    ):
        """
        Initialize the chart visualizer.

        Args:
            config: Custom chart configuration (optional)
            theme: Predefined theme to use (optional)
        """
        if config:
            self.config = config
        elif theme:
            self.config = get_theme_config(theme)
        else:
            self.config = ChartConfig()

        logger.info(f"Initialized ChartVisualizer with theme: {theme or 'default'}")

    def create_integrated_chart(
        self,
        data: Union[pd.DataFrame, OHLCVSeries],
        backtest_results: Optional[BacktestResults] = None,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Create an integrated trading chart with all components.

        Args:
            data: OHLCV price data
            backtest_results: Optional backtesting results for decision overlay
            title: Chart title (optional)
            save_path: Path to save the chart (optional)

        Returns:
            matplotlib.figure.Figure: The created chart figure
        """
        logger.info("Creating integrated trading chart...")

        # Convert data to DataFrame if needed
        df = self._prepare_data(data)

        # Validate data
        if df.empty:
            raise ValueError("No data provided for chart creation")

        # Create figure and subplots
        fig = self._create_figure_layout()

        # Main chart: Candlesticks + Indicators + Decisions
        self._plot_candlesticks(fig.axes[0], df)
        self._plot_alligator(fig.axes[0], df)
        self._plot_fractals(fig.axes[0], df)

        if backtest_results:
            self._plot_agent_decisions(fig.axes[0], df, backtest_results)
            self._plot_trade_annotations(fig.axes[0], df, backtest_results)

        # Awesome Oscillator subplot
        self._plot_awesome_oscillator(fig.axes[1], df)
        if backtest_results:
            self._plot_decision_markers_ao(fig.axes[1], df, backtest_results)

        # Equity curve or confidence subplot
        if backtest_results and len(fig.axes) > 2:
            self._plot_equity_curve(fig.axes[2], backtest_results)

        # Configure chart appearance
        self._configure_chart_appearance(fig, df, title)

        # Save chart if path provided
        if save_path:
            self._save_chart(fig, save_path)

        logger.info("Integrated chart creation completed")
        return fig

    def create_backtesting_summary_chart(
        self,
        data: Union[pd.DataFrame, OHLCVSeries],
        backtest_results: BacktestResults,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Create a comprehensive backtesting summary chart.

        Args:
            data: OHLCV price data
            backtest_results: Backtesting results
            title: Chart title (optional)
            save_path: Path to save the chart (optional)

        Returns:
            matplotlib.figure.Figure: The created summary chart
        """
        logger.info("Creating backtesting summary chart...")

        # Use larger figure for summary
        original_size = self.config.figure_size
        self.config.figure_size = (20, 16)

        try:
            # Create the integrated chart
            fig = self.create_integrated_chart(data, backtest_results, title, save_path)

            # Add performance metrics text box
            self._add_performance_metrics_box(fig, backtest_results)

            return fig

        finally:
            # Restore original figure size
            self.config.figure_size = original_size

    def _prepare_data(self, data: Union[pd.DataFrame, OHLCVSeries]) -> pd.DataFrame:
        """Prepare and validate input data."""
        if isinstance(data, OHLCVSeries):
            df = data.to_dataframe()
        else:
            df = data.copy()

        # Ensure required columns exist
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            else:
                raise ValueError("Data must have datetime index or 'timestamp' column")

        # Sort by timestamp
        df = df.sort_index()

        return df

    def _create_figure_layout(self) -> plt.Figure:
        """Create the figure with appropriate subplot layout."""
        plt.style.use("default")  # Reset any previous styles

        fig, axes = plt.subplots(
            len(self.config.subplot_height_ratios),
            1,
            figsize=self.config.figure_size,
            dpi=self.config.dpi,
            gridspec_kw={"height_ratios": self.config.subplot_height_ratios},
            facecolor=self.config.background_color,
        )

        # Ensure axes is always a list
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        return fig

    def _plot_candlesticks(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot candlestick chart on the given axes."""
        logger.debug("Plotting candlesticks...")

        # Calculate candlestick colors
        up_mask = df["Close"] >= df["Open"]
        down_mask = df["Close"] < df["Open"]

        # Plot wicks (high-low lines)
        for i, (timestamp, row) in enumerate(df.iterrows()):
            ax.plot(
                [timestamp, timestamp],
                [row["Low"], row["High"]],
                color=self.config.candle_wick_color,
                linewidth=1,
                alpha=self.config.candle_alpha,
            )

        # Plot candle bodies
        width = (
            mdates.date2num(df.index[1]) - mdates.date2num(df.index[0])
            if len(df) > 1
            else 1
        )
        width *= 0.6  # Make candles 60% of available width

        for timestamp, row in df.iterrows():
            body_height = abs(row["Close"] - row["Open"])
            body_bottom = min(row["Open"], row["Close"])

            color = (
                self.config.candle_up_color
                if row["Close"] >= row["Open"]
                else self.config.candle_down_color
            )

            rect = Rectangle(
                (mdates.date2num(timestamp) - width / 2, body_bottom),
                width,
                body_height,
                facecolor=color,
                edgecolor=color,
                alpha=self.config.candle_alpha,
            )
            ax.add_patch(rect)

        ax.set_ylabel("Price", fontsize=self.config.font_size_labels)
        if self.config.show_grid:
            ax.grid(True, alpha=self.config.grid_alpha, color=self.config.grid_color)

    def _plot_alligator(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot Alligator indicator lines."""
        logger.debug("Plotting Alligator indicator...")

        try:
            alligator_data = alligator(df)

            # Plot Jaw (13-period, shifted 8)
            ax.plot(
                df.index,
                alligator_data["jaw"],
                color=self.config.alligator_jaw_color,
                linewidth=self.config.alligator_line_width,
                label="Alligator Jaw (13, 8)",
                alpha=0.8,
            )

            # Plot Teeth (8-period, shifted 5)
            ax.plot(
                df.index,
                alligator_data["teeth"],
                color=self.config.alligator_teeth_color,
                linewidth=self.config.alligator_line_width,
                label="Alligator Teeth (8, 5)",
                alpha=0.8,
            )

            # Plot Lips (5-period, shifted 3)
            ax.plot(
                df.index,
                alligator_data["lips"],
                color=self.config.alligator_lips_color,
                linewidth=self.config.alligator_line_width,
                label="Alligator Lips (5, 3)",
                alpha=0.8,
            )

        except Exception as e:
            logger.warning(f"Failed to plot Alligator indicator: {e}")

    def _plot_fractals(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot Fractal up/down signals."""
        logger.debug("Plotting Fractals...")

        try:
            fractals = identify_fractals(df)

            for fractal in fractals:
                timestamp = fractal["timestamp"]
                if timestamp in df.index:
                    # Get existing legend labels to avoid duplicates
                    existing_labels = []
                    legend = ax.get_legend()
                    if legend:
                        existing_labels = [t.get_text() for t in legend.get_texts()]

                    if fractal["type"] == "up":
                        ax.scatter(
                            timestamp,
                            fractal["price"],
                            marker="^",
                            s=self.config.fractal_marker_size,
                            color=self.config.fractal_up_color,
                            label=(
                                "Fractal Up"
                                if "Fractal Up" not in existing_labels
                                else ""
                            ),
                            zorder=5,
                        )
                    else:  # down fractal
                        ax.scatter(
                            timestamp,
                            fractal["price"],
                            marker="v",
                            s=self.config.fractal_marker_size,
                            color=self.config.fractal_down_color,
                            label=(
                                "Fractal Down"
                                if "Fractal Down" not in existing_labels
                                else ""
                            ),
                            zorder=5,
                        )

        except Exception as e:
            logger.warning(f"Failed to plot Fractals: {e}")

    def _plot_awesome_oscillator(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot Awesome Oscillator in subplot."""
        logger.debug("Plotting Awesome Oscillator...")

        try:
            ao_data = awesome_oscillator(df)

            # Plot histogram with color coding
            colors = [
                (
                    self.config.awesome_oscillator_positive_color
                    if val >= 0
                    else self.config.awesome_oscillator_negative_color
                )
                for val in ao_data
            ]

            ax.bar(
                df.index,
                ao_data,
                color=colors,
                alpha=self.config.awesome_oscillator_alpha,
                width=(
                    mdates.date2num(df.index[1]) - mdates.date2num(df.index[0])
                    if len(df) > 1
                    else 1
                ),
            )

            # Add zero line
            ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

            ax.set_ylabel("Awesome Oscillator", fontsize=self.config.font_size_labels)
            if self.config.show_grid:
                ax.grid(
                    True, alpha=self.config.grid_alpha, color=self.config.grid_color
                )

        except Exception as e:
            logger.warning(f"Failed to plot Awesome Oscillator: {e}")

    def _plot_agent_decisions(
        self, ax: plt.Axes, df: pd.DataFrame, backtest_results: BacktestResults
    ) -> None:
        """Plot agent decision points on the main chart."""
        logger.debug("Plotting agent decisions...")

        # Get trades from backtest results
        trades = backtest_results.trades

        for trade in trades:
            if trade.entry_time in df.index:
                price = df.loc[trade.entry_time, "Close"]

                # Determine marker properties based on trade type
                if trade.trade_type == "BUY":
                    color = self.config.buy_decision_color
                    marker = "^"
                    label_text = "Buy Decision"
                else:  # SELL
                    color = self.config.sell_decision_color
                    marker = "v"
                    label_text = "Sell Decision"

                # Calculate marker size based on confidence (if available)
                # For now, use base size since confidence isn't stored in Trade
                marker_size = self.config.decision_marker_base_size

                # Get existing legend labels to avoid duplicates
                existing_labels = []
                legend = ax.get_legend()
                if legend:
                    existing_labels = [t.get_text() for t in legend.get_texts()]

                ax.scatter(
                    trade.entry_time,
                    price,
                    marker=marker,
                    s=marker_size,
                    color=color,
                    alpha=self.config.decision_marker_alpha,
                    edgecolors="black",
                    linewidth=1,
                    zorder=10,
                    label=label_text if label_text not in existing_labels else "",
                )

    def _plot_trade_annotations(
        self, ax: plt.Axes, df: pd.DataFrame, backtest_results: BacktestResults
    ) -> None:
        """Plot trade entry/exit lines and P&L annotations."""
        logger.debug("Plotting trade annotations...")

        trades = backtest_results.trades

        for i, trade in enumerate(trades):
            if trade.entry_time in df.index and trade.exit_time in df.index:
                # Determine line color based on profit/loss
                pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                if trade.trade_type == "SELL":
                    pnl = -pnl  # Invert for short trades

                line_color = (
                    self.config.profit_line_color
                    if pnl > 0
                    else self.config.loss_line_color
                )

                # Draw line from entry to exit
                ax.plot(
                    [trade.entry_time, trade.exit_time],
                    [trade.entry_price, trade.exit_price],
                    color=line_color,
                    linewidth=self.config.trade_line_width,
                    alpha=self.config.trade_line_alpha,
                    linestyle="--",
                )

                # Add P&L annotation
                mid_time = trade.entry_time + (trade.exit_time - trade.entry_time) / 2
                mid_price = (trade.entry_price + trade.exit_price) / 2

                ax.annotate(
                    f"${pnl:.2f}",
                    xy=(mid_time, mid_price),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=self.config.font_size_annotations,
                    color=line_color,
                    weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

    def _plot_decision_markers_ao(
        self, ax: plt.Axes, df: pd.DataFrame, backtest_results: BacktestResults
    ) -> None:
        """Plot decision markers on Awesome Oscillator subplot."""
        logger.debug("Plotting decision markers on AO...")

        # Calculate AO for reference
        try:
            ao_data = awesome_oscillator(df)
            trades = backtest_results.trades

            for trade in trades:
                if trade.entry_time in df.index:
                    ao_value = (
                        ao_data.loc[trade.entry_time]
                        if trade.entry_time in ao_data.index
                        else 0
                    )

                    color = (
                        self.config.buy_decision_color
                        if trade.trade_type == "BUY"
                        else self.config.sell_decision_color
                    )

                    ax.scatter(
                        trade.entry_time,
                        ao_value,
                        marker="o",
                        s=80,
                        color=color,
                        alpha=0.8,
                        edgecolors="black",
                        linewidth=1,
                        zorder=10,
                    )

        except Exception as e:
            logger.warning(f"Failed to plot AO decision markers: {e}")

    def _plot_equity_curve(
        self, ax: plt.Axes, backtest_results: BacktestResults
    ) -> None:
        """Plot equity curve in bottom subplot."""
        logger.debug("Plotting equity curve...")

        if not backtest_results.equity_curve:
            logger.warning("No equity curve data available")
            return

        # Extract timestamps and values
        timestamps = [point[0] for point in backtest_results.equity_curve]
        values = [point[1] for point in backtest_results.equity_curve]

        ax.plot(
            timestamps,
            values,
            color=self.config.trade_entry_color,
            linewidth=2,
            label="Portfolio Value",
        )

        # Add initial capital line
        initial_capital = backtest_results.config.get("initial_capital", 10000)
        ax.axhline(
            y=initial_capital,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="Initial Capital",
        )

        ax.set_ylabel("Portfolio Value ($)", fontsize=self.config.font_size_labels)
        ax.legend(fontsize=self.config.font_size_legend)

        if self.config.show_grid:
            ax.grid(True, alpha=self.config.grid_alpha, color=self.config.grid_color)

    def _configure_chart_appearance(
        self, fig: plt.Figure, df: pd.DataFrame, title: Optional[str]
    ) -> None:
        """Configure overall chart appearance and formatting."""
        logger.debug("Configuring chart appearance...")

        # Set main title
        if title:
            fig.suptitle(title, fontsize=self.config.font_size_title, weight="bold")

        # Configure x-axis for all subplots
        for i, ax in enumerate(fig.axes):
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

            # Only show x-axis label on bottom subplot
            if i == len(fig.axes) - 1:
                ax.set_xlabel("Date", fontsize=self.config.font_size_labels)
            else:
                ax.set_xticklabels([])

        # Add legends to subplots that have labeled elements
        for ax in fig.axes:
            if ax.get_legend_handles_labels()[0]:  # Has legend items
                ax.legend(
                    fontsize=self.config.font_size_legend,
                    loc="upper left",
                    framealpha=0.9,
                )

        # Adjust layout
        plt.tight_layout()

        # Share x-axis across subplots
        for i in range(1, len(fig.axes)):
            fig.axes[i].sharex(fig.axes[0])

    def _add_performance_metrics_box(
        self, fig: plt.Figure, backtest_results: BacktestResults
    ) -> None:
        """Add performance metrics text box to the chart."""
        logger.debug("Adding performance metrics box...")

        # Calculate basic metrics
        total_trades = len(backtest_results.trades)
        if total_trades == 0:
            return

        winning_trades = sum(
            1
            for trade in backtest_results.trades
            if (trade.exit_price - trade.entry_price)
            * (1 if trade.trade_type == "BUY" else -1)
            > 0
        )
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        # Create metrics text
        metrics_text = f"""
        Trading Performance Summary
        ────────────────────────────
        Total Trades: {total_trades}
        Winning Trades: {winning_trades}
        Win Rate: {win_rate:.1f}%
        
        Initial Capital: ${backtest_results.config.get('initial_capital', 0):,.2f}
        """

        # Add text box to top-right of main subplot
        fig.axes[0].text(
            0.02,
            0.98,
            metrics_text.strip(),
            transform=fig.axes[0].transAxes,
            fontsize=self.config.font_size_annotations,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"
            ),
            family="monospace",
        )

    def _save_chart(self, fig: plt.Figure, save_path: Union[str, Path]) -> None:
        """Save the chart to specified path."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=self.config.export_dpi,
            bbox_inches=self.config.export_bbox_inches,
            pad_inches=self.config.export_pad_inches,
            facecolor=self.config.background_color,
        )

        logger.info(f"Chart saved to: {save_path}")


def create_quick_chart(
    data: Union[pd.DataFrame, OHLCVSeries],
    backtest_results: Optional[BacktestResults] = None,
    title: str = "Trading Chart",
    theme: ChartTheme = ChartTheme.LIGHT,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Quick utility function to create a trading chart with default settings.

    Args:
        data: OHLCV price data
        backtest_results: Optional backtesting results
        title: Chart title
        theme: Chart theme to use
        save_path: Optional path to save the chart

    Returns:
        matplotlib.figure.Figure: The created chart
    """
    visualizer = ChartVisualizer(theme=theme)
    return visualizer.create_integrated_chart(data, backtest_results, title, save_path)
