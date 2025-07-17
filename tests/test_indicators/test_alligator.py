import pandas as pd
from src.indicators.alligator import alligator


def test_alligator_shapes():
    df = pd.DataFrame({"Close": [float(i) for i in range(50)]})
    result = alligator(df)
    assert set(result.keys()) == {"jaw", "teeth", "lips"}
    assert all(len(result[k]) == len(df) for k in result)
