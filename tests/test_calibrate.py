import pandas as pd

import calibrate
import stock_scorer as sc


def test_vectorized_classify_matches_classify_setup():
    df = pd.DataFrame({"oversold": range(-5, 76)})
    params = dict(o_strong=sc.OVERSOLD_STRONG, o_high=sc.OVERSOLD_HIGH, o_low=sc.OVERSOLD_LOW)
    expected = [sc.classify_setup(o) for o in df["oversold"]]
    assert calibrate.classify(df, **params).tolist() == expected


def test_neighbor_mean_ignores_nan():
    import numpy as np
    grid = np.array([[1.0, np.nan], [3.0, 5.0]])
    assert calibrate.neighbor_mean(grid)[0, 0] == 3.0
