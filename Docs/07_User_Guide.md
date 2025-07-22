# User Guide: AI Multi-Agent Trading Bot

## Setup Instructions
1. Clone the repository and enter the project directory.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your OpenAI API key and configuration options.

## API & Configuration Options
- All configuration is managed via `.env` and CLI arguments.
- Key options:
  - `OPENAI_API_KEY`: Required for agent decision logic.
  - `TBOT_TEST_MODE`: Set to `1` for offline/synthetic data testing.
  - `DEFAULT_SYMBOL`, `DEFAULT_INTERVAL`, `DEFAULT_INITIAL_CAPITAL`, etc.
  - See `.env` for all available options and scenarios.

## Usage Examples
### Backtesting
```bash
python -m src.cli.main backtest --symbol AAPL --days 60 --config balanced
```
### Visualization
```bash
python -m src.cli.main visualize --symbol AAPL --days 30 --theme professional --output analysis.png
```
### Data Fetching
```bash
python -m src.cli.main data --symbol AAPL --interval 1d --start 2024-01-01 --end 2024-01-10
```
### Configuration
```bash
python -m src.cli.main config --show
```

## Agent Decision Interpretation
- Decisions are logged with action (`BUY`, `SELL`, `HOLD`), confidence, and reasoning.
- Use `logs/` for audit trail and review.
- Decision context includes signals, confluence, and risk assessment.
- See `docs/05_Bill_Williams_Entry_Exit_Logic.md` for methodology.

## Tutorials
- See `examples/` for workflow demonstrations:
  - `agent_integration_example.py`: Multi-agent workflow
  - `backtesting_integration_example.py`: End-to-end backtesting
  - `test_visualization_simple.py`: Chart generation

## Troubleshooting
- Refer to `docs/06_Troubleshooting.md` for common issues and solutions.
- Use verbose logging and test mode for debugging.

## Support & Contributions
- Open issues or pull requests for bugs and enhancements.
- See `Memory_Bank.md` for project history and milestones.
