import pandas as pd
from src.agents.signal_detection_agent import SignalDetectionAgent


def fake_llm(prompt: str) -> float:
    # simple deterministic stub returning high confidence
    return 0.8


def test_detect_signals():
    high = []
    low = []
    close = []
    for i in range(60):
        low.append(i)
        high.append(i + 5 if i % 5 == 2 else i + 2)
        close.append(i + 1)
    df = pd.DataFrame({'High': high, 'Low': low, 'Close': close})
    df.index = pd.date_range('2024-01-01', periods=len(df))
    agent = SignalDetectionAgent(df, llm_client=fake_llm)
    signals = agent.detect_signals()
    assert isinstance(signals, list)
    assert len(signals) > 0
    for s in signals:
        assert 'timestamp' in s and 'type' in s and 'confidence' in s
        assert s['confidence'] > 0

