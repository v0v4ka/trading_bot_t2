"""
Tests for visualization chart configuration.
"""

from unittest.mock import patch

import pytest

from src.visualization.config import (
    BACKTESTING_CONFIG,
    LIVE_TRADING_CONFIG,
    PRESENTATION_CONFIG,
    ChartConfig,
    ChartTheme,
    create_custom_config,
    get_theme_config,
)


class TestChartConfig:
    """Test ChartConfig dataclass functionality."""

    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = ChartConfig()

        # Test some key default values
        assert config.figure_size == (16, 12)
        assert config.dpi == 100
        assert config.candle_up_color == "#26a69a"
        assert config.candle_down_color == "#ef5350"
        assert config.show_grid is True
        assert config.font_size_title == 16

    def test_custom_config_values(self):
        """Test creating config with custom values."""
        config = ChartConfig(figure_size=(20, 16), dpi=150, candle_up_color="#00ff00")

        assert config.figure_size == (20, 16)
        assert config.dpi == 150
        assert config.candle_up_color == "#00ff00"
        # Ensure other defaults remain
        assert config.candle_down_color == "#ef5350"


class TestChartThemes:
    """Test chart theme functionality."""

    def test_light_theme_config(self):
        """Test light theme configuration."""
        config = get_theme_config(ChartTheme.LIGHT)

        # Light theme should use default values
        assert config.background_color == "#ffffff"
        assert config.candle_wick_color == "#4a4a4a"
        assert config.grid_color == "#cccccc"

    def test_dark_theme_config(self):
        """Test dark theme configuration."""
        config = get_theme_config(ChartTheme.DARK)

        # Dark theme should have modified colors
        assert config.background_color == "#2e2e2e"
        assert config.candle_wick_color == "#cccccc"
        assert config.grid_color == "#555555"

    def test_professional_theme_config(self):
        """Test professional theme configuration."""
        config = get_theme_config(ChartTheme.PROFESSIONAL)

        # Professional theme should have larger figure and fonts
        assert config.figure_size == (20, 14)
        assert config.font_size_title == 18
        assert config.font_size_labels == 14
        assert config.candle_up_color == "#00c851"
        assert config.candle_down_color == "#ff4444"

    def test_colorblind_friendly_theme(self):
        """Test colorblind-friendly theme configuration."""
        config = get_theme_config(ChartTheme.COLORBLIND_FRIENDLY)

        # Should use blue/orange instead of green/red
        assert config.candle_up_color == "#0173b2"
        assert config.candle_down_color == "#de8f05"
        assert config.buy_decision_color == "#0173b2"
        assert config.sell_decision_color == "#de8f05"


class TestCustomConfig:
    """Test custom configuration creation."""

    def test_create_custom_config_valid_params(self):
        """Test creating custom config with valid parameters."""
        config = create_custom_config(
            figure_size=(18, 14), dpi=200, candle_up_color="#custom_green"
        )

        assert config.figure_size == (18, 14)
        assert config.dpi == 200
        assert config.candle_up_color == "#custom_green"
        # Other values should remain default
        assert config.candle_down_color == "#ef5350"

    def test_create_custom_config_invalid_param(self):
        """Test creating custom config with invalid parameter."""
        with pytest.raises(ValueError, match="Unknown configuration parameter"):
            create_custom_config(invalid_parameter="value")

    def test_create_custom_config_no_params(self):
        """Test creating custom config with no parameters."""
        config = create_custom_config()

        # Should be identical to default config
        default_config = ChartConfig()
        assert config.figure_size == default_config.figure_size
        assert config.dpi == default_config.dpi


class TestPredefinedConfigs:
    """Test predefined configuration constants."""

    def test_backtesting_config(self):
        """Test backtesting configuration settings."""
        config = BACKTESTING_CONFIG

        assert config.figure_size == (18, 14)
        assert config.subplot_height_ratios == (4, 1, 1)
        assert config.decision_marker_base_size == 200
        assert config.show_grid is True

    def test_live_trading_config(self):
        """Test live trading configuration settings."""
        config = LIVE_TRADING_CONFIG

        assert config.figure_size == (14, 10)
        assert config.subplot_height_ratios == (3, 1)
        assert config.decision_marker_alpha == 1.0
        assert config.candle_alpha == 1.0

    def test_presentation_config(self):
        """Test presentation configuration settings."""
        config = PRESENTATION_CONFIG

        assert config.figure_size == (20, 16)
        assert config.dpi == 150
        assert config.font_size_title == 20
        assert config.font_size_labels == 16
        assert config.export_dpi == 400


class TestChartThemeEnum:
    """Test ChartTheme enum values."""

    def test_theme_values(self):
        """Test that all theme enum values are correct."""
        assert ChartTheme.LIGHT.value == "light"
        assert ChartTheme.DARK.value == "dark"
        assert ChartTheme.PROFESSIONAL.value == "professional"
        assert ChartTheme.COLORBLIND_FRIENDLY.value == "colorblind_friendly"

    def test_all_themes_have_configs(self):
        """Test that all themes can generate configurations."""
        for theme in ChartTheme:
            config = get_theme_config(theme)
            assert isinstance(config, ChartConfig)
            # Ensure each theme produces a valid config
            assert config.figure_size is not None
            assert config.dpi > 0
            assert config.font_size_title > 0
