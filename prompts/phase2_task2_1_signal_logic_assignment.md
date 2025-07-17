# Task Assignment Prompt: Phase 2, Task 2.1

## Task Title
Implement Trading Signal Logic for Market Entry Assessment

## Context
- The project is a multi-agent trading bot MVP using the APM framework.
- Phase 1 (data models, Yahoo Finance integration, Bill Williams indicators) is complete and tested.
- The current priority is to enable backtesting of market entry point assessment quality with minimal development.

## Objective
Develop a Python module that generates entry and exit signals for backtesting, using Bill Williams indicators (Fractals, Alligator, Awesome Oscillator) on historical OHLCV data.

## Requirements
- Define clear entry and exit rules based on the outputs of the implemented Bill Williams indicators.
- The logic should:
  - Accept OHLCV data and indicator outputs as input.
  - Output a list/array of entry and exit signals (buy/sell/hold or similar) for each bar.
  - Be compatible with the planned backtesting engine (next task).
- Include unit tests for signal logic, using realistic sample data.
- Document the rules and logic in code comments and/or a short markdown file.

## Deliverables
- `src/signals/` directory with one or more Python modules implementing the signal logic.
- Unit tests in `tests/test_signals/`.
- Documentation of entry/exit rules and usage.

## Success Criteria
- Signal logic is correct, deterministic, and reproducible.
- Unit tests cover typical and edge cases.
- Output is ready for use in the backtesting engine.

## References
- Product spec and Implementation_Plan.md
- Existing indicator modules in `src/indicators/`
- Sample OHLCV data and models in `src/data/`

---

*Please acknowledge understanding and state readiness to begin. Ask any clarifying questions if needed before starting implementation.*
