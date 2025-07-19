#!/usr/bin/env python3
"""
Test script to verify environment configuration is working correctly.
Run this to check your .env file setup.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.env_config import TradingBotConfig, config


def test_env_config():
    """Test environment configuration loading."""
    print("🔧 Trading Bot Environment Configuration Test")
    print("=" * 50)

    # Test configuration loading
    try:
        test_config = TradingBotConfig()
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

    # Validate configuration
    errors = test_config.validate()
    if errors:
        print("❌ Configuration validation errors:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("✅ Configuration validation passed")

    # Display key settings
    print("\n📊 Key Configuration Settings:")
    print(f"   OpenAI Model: {test_config.openai_model}")
    print(f"   Test Mode: {test_config.test_mode}")
    print(f"   API Key Set: {'Yes' if test_config.openai_api_key else 'No'}")
    print(f"   Production Mode: {test_config.is_production_mode()}")
    print(f"   Signal Threshold: {test_config.signal_confidence_threshold}")
    print(f"   Decision Threshold: {test_config.decision_confidence_threshold}")
    print(f"   Default Symbol: {test_config.default_symbol}")
    print(f"   Default Capital: ${test_config.default_initial_capital:,.2f}")
    print(f"   Chart Theme: {test_config.default_chart_theme}")

    # Test global config instance
    print("\n🌐 Global Configuration Instance:")
    print(f"   Model: {config.openai_model}")
    print(f"   Test Mode: {config.test_mode}")

    return True


if __name__ == "__main__":
    success = test_env_config()

    if success:
        print("\n✅ Environment configuration test passed!")
        print("\n🚀 Ready to run trading bot commands:")
        print("   python -m src.cli.main config --show")
        print("   python -m src.cli.main backtest --symbol AAPL --days 60")
    else:
        print("\n❌ Environment configuration test failed!")
        print("   Please check your .env file and fix the issues above.")
        sys.exit(1)
