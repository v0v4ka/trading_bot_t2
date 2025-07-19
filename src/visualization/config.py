"""
Chart configuration and styling for trading visualization.

This module provides configuration classes and predefined themes for
customizing the appearance and behavior of trading charts.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Union


class ChartTheme(Enum):
    """Predefined chart themes for different use cases."""
    
    LIGHT = "light"
    DARK = "dark"
    PROFESSIONAL = "professional"
    COLORBLIND_FRIENDLY = "colorblind_friendly"


@dataclass
class ChartConfig:
    """
    Configuration class for chart visualization settings.
    
    This class encapsulates all visual and behavioral settings for
    trading charts, including colors, sizes, and display options.
    """
    
    # Chart dimensions and layout
    figure_size: Tuple[int, int] = (16, 12)
    dpi: int = 100
    subplot_height_ratios: Tuple[float, ...] = (3, 1, 1)  # Main, AO, Equity/Confidence
    
    # Candlestick styling
    candle_up_color: str = "#26a69a"  # Teal green
    candle_down_color: str = "#ef5350"  # Red
    candle_wick_color: str = "#4a4a4a"  # Dark gray
    candle_alpha: float = 0.8
    
    # Bill Williams Indicators
    alligator_jaw_color: str = "#1f77b4"  # Blue
    alligator_teeth_color: str = "#ff7f0e"  # Orange
    alligator_lips_color: str = "#2ca02c"  # Green
    alligator_line_width: float = 2.0
    
    fractal_up_color: str = "#00ff00"  # Bright green
    fractal_down_color: str = "#ff0000"  # Bright red
    fractal_marker_size: int = 100
    fractal_marker: str = "^"  # Up triangle for up fractals, v for down
    
    awesome_oscillator_positive_color: str = "#26a69a"  # Teal
    awesome_oscillator_negative_color: str = "#ef5350"  # Red
    awesome_oscillator_alpha: float = 0.7
    
    # Agent decision markers
    buy_decision_color: str = "#4caf50"  # Green
    sell_decision_color: str = "#f44336"  # Red
    hold_decision_color: str = "#ff9800"  # Orange
    decision_marker_base_size: int = 150
    decision_marker_alpha: float = 0.8
    
    # Trade annotations
    trade_entry_color: str = "#2196f3"  # Blue
    trade_exit_color: str = "#9c27b0"  # Purple
    profit_line_color: str = "#4caf50"  # Green
    loss_line_color: str = "#f44336"  # Red
    trade_line_width: float = 2.0
    trade_line_alpha: float = 0.6
    
    # Text and annotations
    font_size_title: int = 16
    font_size_labels: int = 12
    font_size_ticks: int = 10
    font_size_legend: int = 10
    font_size_annotations: int = 8
    
    # Grid and background
    show_grid: bool = True
    grid_alpha: float = 0.3
    grid_color: str = "#cccccc"
    background_color: str = "#ffffff"
    
    # Volume (if implemented)
    volume_up_color: str = "#26a69a"
    volume_down_color: str = "#ef5350"
    volume_alpha: float = 0.5
    
    # Confidence visualization
    confidence_high_threshold: float = 0.8
    confidence_medium_threshold: float = 0.6
    confidence_colormap: str = "RdYlGn"  # Red-Yellow-Green
    
    # Chart export settings
    export_dpi: int = 300
    export_bbox_inches: str = "tight"
    export_pad_inches: float = 0.2


def get_theme_config(theme: ChartTheme) -> ChartConfig:
    """
    Get a predefined chart configuration for a specific theme.
    
    Args:
        theme: The desired chart theme
        
    Returns:
        ChartConfig: Configured chart settings for the theme
    """
    base_config = ChartConfig()
    
    if theme == ChartTheme.DARK:
        base_config.background_color = "#2e2e2e"
        base_config.candle_wick_color = "#cccccc"
        base_config.grid_color = "#555555"
        
    elif theme == ChartTheme.PROFESSIONAL:
        base_config.figure_size = (20, 14)
        base_config.candle_up_color = "#00c851"
        base_config.candle_down_color = "#ff4444"
        base_config.font_size_title = 18
        base_config.font_size_labels = 14
        
    elif theme == ChartTheme.COLORBLIND_FRIENDLY:
        # Use colorblind-friendly palette
        base_config.candle_up_color = "#0173b2"  # Blue
        base_config.candle_down_color = "#de8f05"  # Orange
        base_config.alligator_jaw_color = "#029e73"  # Green
        base_config.alligator_teeth_color = "#cc78bc"  # Pink
        base_config.alligator_lips_color = "#ca9161"  # Brown
        base_config.buy_decision_color = "#0173b2"
        base_config.sell_decision_color = "#de8f05"
        
    return base_config


def create_custom_config(**kwargs) -> ChartConfig:
    """
    Create a custom chart configuration with specific overrides.
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        ChartConfig: Custom chart configuration
    """
    base_config = ChartConfig()
    
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    
    return base_config


# Predefined configurations for common use cases
BACKTESTING_CONFIG = ChartConfig(
    figure_size=(18, 14),
    subplot_height_ratios=(4, 1, 1),  # Emphasize main chart
    decision_marker_base_size=200,  # Larger markers for backtesting
    show_grid=True,
)

LIVE_TRADING_CONFIG = ChartConfig(
    figure_size=(14, 10),
    subplot_height_ratios=(3, 1),  # No equity curve for live
    decision_marker_alpha=1.0,  # Clear markers for live decisions
    candle_alpha=1.0,
)

PRESENTATION_CONFIG = ChartConfig(
    figure_size=(20, 16),
    dpi=150,
    font_size_title=20,
    font_size_labels=16,
    font_size_ticks=14,
    font_size_legend=14,
    export_dpi=400,
)
