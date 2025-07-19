# Trading Bot Examples

This directory contains example scripts and demonstrations of the trading bot functionality.

## Available Examples

### Agent Integration Examples
- **`agent_integration_example.py`** - Demonstrates basic agent workflow integration
- **`decision_maker_integration_demo.py`** - Shows decision maker agent usage
- **`demo_decision_logging.py`** - Decision logging system demonstration

### Backtesting Examples
- **`backtesting_integration_example.py`** - Complete backtesting workflow example

### Visualization Examples
- **`visualization_integration_example.py`** - Chart visualization integration
- **`test_visualization_simple.py`** - Simple visualization testing script

## Running Examples

All examples can be run from the project root directory:

```bash
# From the trading_bot_t2 directory
python examples/visualization_integration_example.py
python examples/backtesting_integration_example.py
```

## Environment Requirements

Some examples require environment variables:
- `OPENAI_API_KEY` - For agent-based examples with real LLM integration
- `TBOT_TEST_MODE=1` - For running in test mode with synthetic data

## Output

Chart outputs will be saved to the `outputs/charts/` directory.
