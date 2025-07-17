🛠️ AI Multi-Agent Trading Bot – Deep-Dive Design Document

Bill Williams Chaos Strategy, Alpaca Execution, GPT-4-powered Reasoning
Version 0.9 — July 2025
0 . Table of Contents

Overview & Goals
High-Level Architecture Diagram
Data Flow & Core Objects
Modules (detailed)
Agents (roles, I/O)
LangGraph State Machines
Indicator & Wave-Detection Algorithms
Order & Risk Management Layer
Reporting & Visualization Sub-system
CLI Design & Configuration
Logging / Persistence
Test & Simulation Harness
Future Extensions
1. Overview

The system is an agentic trading platform that implements Bill Williams’ Trading Chaos on daily bars, enriched by Elliott-wave macro context and GPT-4 reasoning.
Key design themes:

Separation of concerns: calculation vs reasoning
Cost efficiency: LLM invoked only after deterministic filters
Explainability: every trade tagged with rationale, wave phase, and UUID
Extensibility: plug-and-play agents, LangGraph orchestration
2. Architecture Diagram (ASCII)

       ┌─────────────┐
       │  Alpaca API │
       └────┬────────┘
            │ OHLCV
            ▼
┌───────────────────────────┐
│   indicators/ (Python)    │   ← SMA, AO, Fractals, ATR
└────┬──────────┬───────────┘
     │          │
     │ bars + indicators
     ▼          ▼
┌──────────┐  ┌─────────────────┐
│ Wave-Cls │  │ Market Filter   │  (rule)
└──────────┘  └─────────────────┘
     │ waves          │ pass/deny
     └──────┬─────────┘
            ▼
       LangGraph DAG
┌───────────────────────────────────────────────────────────────┐
│ Aggressive │ Neutral │ Conservative │ Macro │ Risk │ Reviewer│
└────┬────────┴────────┴────────┬─────┴───────┴────────┬────────┘
     │ YES/NO + conf           │ weights               │ feedback
     └────────────► Score Aggregator (rule)
                          │ final_decision
                          ▼
                 order_manager/ (Alpaca)
                          │
                 trade_lifecycle SM
                          ▼
                   reports/, logs/
3. Data Flow & Core Objects

# bars.py
class Bar(BaseModel):
    t: datetime; o: float; h: float; l: float; c: float; v: int

class MarketContext(BaseModel):
    symbol: str             # "AAPL"
    tf: Literal["1D"]       # timeframe
    bars: list[Bar]         # ≥40 bars
    indicators: dict        # ao, atr, fractals...
LLM agents exchange AgentDecision:

class AgentDecision(BaseModel):
    agent: str               # "neutral"
    decision: Literal["yes","no","hold"]
    confidence: float        # 0–1
    reason: str
    suggested_order: Order | None
4. Module-by-Module Detail

Folder / Module	Purpose	Key Classes / Fns
indicators/	Pure-Python maths	calc_fractals, calc_ao, calc_alligator, atr
wave_classifier/	Elliott 5-wave detection	WaveClassifier.detect(bars_by_tf)
agents/	LLM wrappers	AggressiveAgent, NeutralAgent, …
advisor_engine/	Aggregation, weighting	AdvisorEngine.score(decisions)
order_manager/	Alpaca REST	submit_order, cancel_order, flatten_position
risk_manager/	Capital & SL checks	check_risk(entry_price, sl)
langgraph/	YAML graph + runner	graph.yaml, run_graph()
reports/	Plots + markdown	chart_generator.py, markdown_reporter.py
cli/	CLI entrypoint	bot.py, argparse
data/	Logs & csv	orders.csv, trades.csv, review.csv
5. Agent Specs

Agent	Role	I/O	LLM Model
AggressiveAdvisor	Find earliest valid entry	MarketContext → AgentDecision	gpt-4o
NeutralAdvisor	Baseline yes/no	same	gpt-4o
ConservativeAdvisor	Require max confirmation	same	gpt-4o
WaveClassifier (non-LLM mock)	Detect wave 1-5 / A-C	bars_by_tf → wave dict	heuristic / future ML
MacroAdvisor	Check wave alignment; veto	waves + ctx → pass/hold	gpt-4o
MarketFilter	News, earnings, liquidity	rule → pass/deny	none
RiskAssessment	SL distance, R ÷ R	ctx → pass/deny	none
ReviewerAgent	Post-mortem scoring	trade record → feedback	gpt-4o (cheap)
Aggregation rule (weights):

Agent	Weight
Aggressive	0.2
Neutral	0.3
Conservative	0.2
Macro	0.1
MarketFilter	0.1
Risk	0.1
score = Σ(confidence × weight) → enter if ≥ 0.65.

6. LangGraph State Machines

6 . 1 Entry DAG (simplified YAML)
nodes:
  wave: WaveClassifier
  filter: MarketFilter
  advs:
    - AggressiveAdvisor
    - NeutralAdvisor
    - ConservativeAdvisor
  macro: MacroAdvisor
  risk: RiskAssessment
  scorer: AdvisorEngine

edges:
  wave -> filter
  filter.pass -> advs
  advs -> macro
  macro.pass -> risk
  risk.pass -> scorer
  scorer.enter -> place_order
6 . 2 Trade Lifecycle SM
(see diagram in requirements section)

7. Indicator & Wave Algorithms (core)

# fractals.py
def calc_fractals(bars: list[Bar]) -> list[tuple[int, str, float]]:
    peaks = []; troughs = []
    for i in range(2, len(bars)-2):
        window = bars[i-2:i+3]
        if bars[i].h == max(b.h for b in window):
            peaks.append((i, "up", bars[i].h))
        if bars[i].l == min(b.l for b in window):
            troughs.append((i, "down", bars[i].l))
    return peaks + troughs
# wave_classifier/mock.py
def detect(bars_by_tf: dict[str, list[Bar]]) -> dict:
    # heuristic: count consecutive fractal highs/lows
    waves = {}
    for tf, bars in bars_by_tf.items():
        count_up = sum(bars[i].h > bars[i-1].h for i in range(1, len(bars)))
        wave = min(5, (count_up // 3) + 1)  # toy logic
        waves[tf] = {"wave": wave, "type": "impulse", "confidence": 0.6}
    return waves
8. Order & Risk Management

def submit_trade(order: Order):
    if not risk_manager.check(order):
        return "DENIED"
    resp = alpaca.submit_order(...)
    log_order(resp)
    return resp.id
Stop-loss updated whenever a lower fractal (for longs) moves upward.

9. Reporting & Visualization

9 . 1 Chart Generator
# chart_generator.py
def plot_candles(df: pd.DataFrame, decisions: pd.DataFrame, waves: dict):
    ...
    ax2.plot(df.index, decisions.score, 'b-o')
    for wave_label, bar_idx in waves_labels:
        ax1.text(bar_idx, df.High[bar_idx]+1, wave_label)
    plt.savefig(f"charts/{symbol}_{date}.png")
9 . 2 Markdown Reporter
def write_md_report(trade_records, path):
    with open(path, "w") as f:
        f.write("# 📊 Daily Market Entry Advisor Report\\n")
        ...
10. CLI Interface (bot.py)

python bot.py \
  --symbols AAPL MSFT \
  --mode backtest \
  --start 2023-01-01 --end 2025-01-01 \
  --generate-plots
11. Logging

orders.csv — uuid, symbol, side, price, sl, status
trades.csv — entry/exit, PnL, wave phase
review.csv — agent decisions, reviewer feedback
12. Test Harness

pytest unit tests for indicators & wave detection
Integration tests that replay cached GPT responses
Backtest runner validating score threshold logic
13. Future Work

Add drawdown & PnL visualization
ML wave classifier replacement
Real-time sentiment filter
Multi-asset correlation risk module
End of Deep-Dive Design Document