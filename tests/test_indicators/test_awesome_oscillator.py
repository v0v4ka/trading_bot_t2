import pandas as pd
from src.indicators.awesome_oscillator import awesome_oscillator, ao_zero_crossings


def test_ao_and_crossings():
    df = pd.DataFrame(
        {"High": [i + 1 for i in range(40)], "Low": [i for i in range(40)]}
    )
    ao = awesome_oscillator(df)
    crossings = ao_zero_crossings(ao)
    assert len(ao) == len(df)
    assert crossings.sum() >= 0
