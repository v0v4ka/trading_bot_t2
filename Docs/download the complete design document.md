
# AI-Powered Trading Bot Design Document

*Generated on 2025-07-10*

---

## 📌 Overview

This document outlines the architectural and implementation plan for an AI-powered trading bot using Bill Williams' *Trading Chaos* theory. The system integrates Alpaca for market data and order execution, and OpenAI (GPT-4) for high-level reasoning. It supports real-time trading, intelligent decision refinement, and a feedback-driven agentic loop for improving trade logic.

---

## 📋 Requirements

### Functional Requirements

1. **Strategy**: Follow Bill Williams' *Trading Chaos* theory
2. **Symbols**: Work with all available equity symbols from Alpaca
3. **Bars**: Use **daily bars** for scanning and decision making
4. **Order Logic**:
   - Use **market orders** for execution
   - Attach **stop-loss** based on the most recent opposite fractal
   - Replace pending orders if a **better entry** is detected
5. **Trade Constraints**:
   - Comply with Alpaca buying power, PDT, and open position limits
   - Detect constraint violations and log action plan
6. **Position Management**:
   - Support multiple positions
   - One position per symbol
   - Monitor trades for improved SL conditions
7. **Exit Logic**:
   - Use AO divergence, wave patterns, or GPT exit reasoning
8. **Advisor Loop**:
   - Entry advice: Only when not in market
   - Exit advice: Only when trade is active
9. **Logging**:
   - Persistent CSV logs for orders, decisions, PnL
   - Each order tagged with a UUID for traceability
10. **Auto-Plotting**:
   - Generate annotated candlestick plots per symbol per run
   - Overlay decision annotations, wave counts, fractals, AO

### Non-Functional Requirements

- Runs locally for testing; staging/production remote-ready
- CLI-based interface
- Environment-based configuration for token, log dir, thresholds

---

## 🔄 Trading State Machine

```plaintext
┌───────────────────────────────────────────────────────────────────────────────┐
│                                 NOT IN MARKET                                 │
└───────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
  +--------------------+       +---------------------+
  |   No Pending Order |──────▶|   Place Entry Order |
  +--------------------+       +---------------------+
          ▲                          │
          │                          ▼
  +--------------------+       +----------------------+
  |   Valid Fractal?   |◀──────│   Better Entry Found |
  +--------------------+       +----------------------+
          │                          │
          │ Fractal invalid /        ▼
          │ no confirmation   +-----------------------+
          └──────────────────▶|   Await Order Fill    |
                              +-----------------------+
                                       │
                                       ▼
                             ┌────────────────────────────┐
                             │         IN MARKET          │
                             └────────────────────────────┘
                                       │
                                       ▼
                         +-----------------------------+
                         |    Monitor Active Trade     |
                         |  AO / Advisor / PnL / SL    |
                         +-----------------------------+
                                       │
                      ┌────────────────┴──────────────┐
                      ▼                               ▼
             +-----------------+             +-----------------+
             | Update Stop-Loss|             |      Exit       |
             | Trail or Fractal|             | Sell / Flatten  |
             +-----------------+             +-----------------+
                                                   │
                                                   ▼
                                  Return to → NOT IN MARKET
```

---

## 🧠 Market Entry Phase

(continued below due to length)

---

## 🧠 Market Entry Phase (continued)

The market entry phase is critical. The bot uses both rule-based logic (fractals, AO, alligator) and GPT-based agents to evaluate entry quality.

### Key Criteria
- Two confirmed fractals
- AO histogram showing alignment
- Alligator lines spread and ordering
- Entry bar breakout confirmed
- Wave pattern suggests early trend

### Agentic Architecture
| Agent                  | Purpose                                                        |
|------------------------|----------------------------------------------------------------|
| Aggressive Advisor     | Argues **for** entering trade based on optimistic interpretation |
| Neutral Advisor        | Offers balanced, unbiased reasoning                            |
| Conservative Advisor   | Advises caution or avoidance                                   |
| Macro Advisor          | Reviews higher/lower timeframes for trend alignment            |
| Wave Classifier Agent  | Detects 5-wave patterns across timeframes                      |
| Market Filter Agent    | Confirms fractals, AO, and volume validity                     |
| Risk Assessment Agent  | Checks stop-loss placement, spread, and volatility             |
| Reviewer Agent         | Evaluates past actions to refine future decisions              |

---

## 📤 Advisor Interaction

```json
{
  "entry_context": {
    "symbol": "AAPL",
    "fractal": { "direction": "up", "confirmed": true, "level": 182.50 },
    "ao": [0.2, 0.5, 0.6],
    "alligator": { "lips": 180.1, "teeth": 179.8, "jaw": 179.0 },
    "trend": "bullish",
    "history": [20 daily bars...]
  }
}
```

### Advisor Output
```json
{
  "decision": "yes",
  "reason": "AO and fractal align, wave count at 2, uptrend just starting",
  "recommended_order": {
    "type": "market",
    "stop_loss": 178.50,
    "uuid": "b728a3cf"
  }
}
```

---

## 📈 Visualization Example

- Annotated candlestick charts (OHLC)
- Decision points, wave labels, stop-loss lines
- Dual-Y axis for price + indicators
- Generated automatically during simulations

(Chart generation code resides in `reports/chart_generator.py`)

---

## ⚙️ CLI Configuration

```bash
python bot.py --symbols AAPL TSLA --mode live --log-dir logs/ --model gpt-4o
```

Environment variables:
- `ALPACA_API_KEY`
- `OPENAI_API_KEY`
- `TRADING_MODE=backtest|live`

---

## 🧪 Test Cases

1. Detect confirmed fractals on synthetic uptrend
2. Replace old order if new fractal forms
3. Hold when corrective wave detected
4. Exit on AO divergence or wave 5
5. Validate stop-loss placement logic
6. Confirm that chart generation runs after every cycle
7. Ensure LLM cost control is enforced (skip LLM if no fractals)

---

## 📁 Project Structure

```
trading_bot/
├── bot.py
├── config/
├── data/
│   └── logs.csv
├── indicators/
│   └── fractals.py, ao.py, alligator.py
├── ai/
│   └── advisor.py, wave_classifier.py, macro.py
├── agents/
│   └── aggressive.py, neutral.py, conservative.py, risk.py
├── strategies/
│   └── chaos_theory.py
├── langgraph/
│   └── graph.yaml, runner.py
├── reports/
│   └── chart_generator.py, cli_reporter.py
```

---

## 🧠 Feedback & Learning Loop

Each agent receives reviewer feedback per trade cycle. This is stored in local logs and will evolve into adaptive learning logic to weight advisor decisions differently based on recent accuracy.

---

Document complete.
