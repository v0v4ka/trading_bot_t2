
# trading_bot_t2

AI Multi-Agent Trading Bot MVP with backtesting, Bill Williams indicators, and LLM-powered agents. Implements APM framework for agentic project management.

## Overview

A modular AI multi-agent trading bot MVP featuring:
- **Bill Williams Trading Chaos** indicators (Fractals, Alligator, Awesome Oscillator)
- **Multi-Agent Architecture** with Signal Detection and Decision Maker agents
- **Comprehensive Backtesting** framework with performance analysis
- **Advanced Visualization** system with multiple themes and chart types
- **CLI Interface** for easy operation and automation
- **Structured Logging** and decision audit trails

## Project Structure

```
trading_bot_t2/
├── src/                    # Core source code
│   ├── data/              # Data handling and ingestion
│   ├── agents/            # Agent implementations (Signal, Decision Maker)
│   ├── indicators/        # Bill Williams indicators
│   ├── backtesting/       # Backtesting engine and metrics
│   ├── visualization/     # Chart generation and themes
│   ├── logging/           # Decision logging and analysis
│   └── cli/               # Command-line interface
├── tests/                 # Comprehensive test suite
├── examples/              # Usage examples and demonstrations
├── outputs/               # Generated charts and reports
└── docs/                  # Documentation and guides
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd trading_bot_t2

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Generate a basic chart
python -m src.cli.main visualize --symbol AAPL --days 30 --theme professional

# Run backtesting (requires OpenAI API key)
export OPENAI_API_KEY="your-api-key"
python -m src.cli.main backtest --symbol AAPL --days 60 --config balanced

# Test mode (no API key required)
TBOT_TEST_MODE=1 python examples/test_visualization_simple.py
```

## Examples

The `examples/` directory contains comprehensive demonstrations:

- **`test_visualization_simple.py`** - Visualization system showcase
- **`backtesting_integration_example.py`** - End-to-end backtesting
- **`agent_integration_example.py`** - Agent workflow demonstration

## Requirements

- **Python 3.9+**
- **Dependencies**: See `requirements.txt` and `pyproject.toml`
- **Optional**: OpenAI API key for agent-based features

## Command Line Interface

### Data Operations
```bash
# Fetch market data
python -m src.cli.main data --symbol AAPL --interval 1d --start 2024-01-01 --end 2024-01-10
```

### Visualization
```bash
# Basic chart
python -m src.cli.main visualize --symbol AAPL --days 30

# Advanced visualization
python -m src.cli.main visualize --symbol AAPL --days 60 \
  --theme professional --output analysis.png \
  --title "Technical Analysis" --width 20 --height 15
```

### Configuration
```bash
# Show configuration
python -m src.cli.main config --show

# Display version
python -m src.cli.main version
```

