# Bill Williams-Compliant Entry & Exit Logic

## Overview
This document describes the fully Bill Williams-compliant entry and exit logic implemented in the AI Multi-Agent Trading Bot. It covers rationale, methodology, agent workflow, and usage patterns for both entry and exit decisions.

---

## Entry Logic

### Rationale
Bill Williams' Trading Chaos methodology emphasizes staged entry, trend confirmation, and risk management. The bot uses multiple indicators (Fractals, Alligator, Awesome Oscillator) to ensure high-probability entries.

### Methodology
- **Staged Entry (Three Wise Men):**
  - Entry is allowed only when multiple (2+) Bill Williams signals align (e.g., Fractal breakout, Alligator awake, AO confirmation).
  - First entry uses full position size; subsequent add-ons use reverse pyramiding (0.5x, 0.25x, etc.).
- **Alligator State Filtering:**
  - Trades are filtered based on Alligator indicator state (awake/sleeping).
  - Single strong signal may be allowed if Alligator is awake and trend is confirmed.
- **Confluence Evaluation:**
  - Signal confluence is calculated using confidence scores and signal count.
  - Conflicting signals penalize confluence, resulting in HOLD or no trade.
- **Stop Loss Placement:**
  - Initial stop loss is placed below/above reversal bar for first entry.
  - Trailing stops are used for add-on entries, based on most recent swing or fractal.

### Usage
- Entry logic is encapsulated in the `DecisionMakerAgent`.
- Entry decisions require explicit signal context and market data.
- Position sizing and stop loss are automatically calculated.

---

## Exit Logic

### Rationale
Bill Williams exit logic focuses on timely position closure based on reversal signals, momentum shifts, and fractal breakouts.

### Methodology
- **Reversal Bar Exit:**
  - Exit is triggered by an opposite reversal bar outside the Alligator's mouth.
- **AO Zero-Line Cross:**
  - Exit is triggered when the Awesome Oscillator crosses the zero line in the opposite direction.
- **Opposite Fractal Breakout:**
  - Exit is triggered by a breakout of an opposite fractal.
- **Explicit Position Context:**
  - Exit logic considers the current position (long/short) and only triggers when reversal signals are valid for that position.

### Usage
- Exit logic is part of the `DecisionMakerAgent` and requires explicit position context.
- Exit decisions are logged and validated by comprehensive tests.

---

## Agent Workflow & Compliance
- All entry/exit decisions are logged for audit and compliance.
- Tests cover all edge cases, including staged entry, Alligator state, reverse pyramiding, stop loss, and exit logic.
- The agent is fully Bill Williams-compliant and validated by 28+ passing tests.

---

## References
- Bill Williams, "Trading Chaos"
- Project Memory Bank (see `Memory_Bank.md` for milestones)
- Source: `src/agents/decision_maker_agent.py`, `tests/test_agents/test_decision_maker_agent.py`
