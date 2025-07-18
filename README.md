
# trading_bot_t2

AI Multi-Agent Trading Bot MVP with backtest mode, Bill Williams indicators, and LLM-powered agents. Implements APM framework for agentic project management.

A modular AI multi-agent trading bot MVP for backtesting with Bill Williams Trading Chaos indicators.

## Project Structure

- `src/data/` - Data handling and ingestion
- `src/agents/` - Agent implementations
- `src/backtesting/` - Backtesting logic
- `src/visualization/` - Visualization tools
- `src/cli/` - Command-line interface

## Requirements
- Python 3.9+
- See `requirements.txt` and `pyproject.toml` for dependencies

## Command Line Interface

The project includes a simple CLI for common tasks. Run `python -m src.cli.main <command>`.

### Examples

Fetch market data:

```bash
python -m src.cli.main data --symbol AAPL --interval 1d --start 2024-01-01 --end 2024-01-10
```

Show default configuration:

```bash
python -m src.cli.main config --show
```

Display version:

```bash
python -m src.cli.main version
```

