import numpy as np
import pandas as pd

import backtest
import calibrate
import stock_scorer as sc

from test_stock_scorer import geometric, make_hist


def test_vectorized_classify_matches_classify_setup():
    df = pd.DataFrame({"oversold": range(-5, 76)})
    params = dict(o_strong=sc.OVERSOLD_STRONG, o_high=sc.OVERSOLD_HIGH, o_low=sc.OVERSOLD_LOW)
    expected = [sc.classify_setup(o) for o in df["oversold"]]
    assert calibrate.classify(df, **params).tolist() == expected


def test_neighbor_mean_ignores_nan():
    grid = np.array([[1.0, np.nan], [3.0, 5.0]])
    assert calibrate.neighbor_mean(grid)[0, 0] == 3.0


def weekly_dates(n: int, per_week: int = 1) -> pd.Series:
    weeks = pd.date_range("2020-01-06", periods=n, freq="W-MON", tz="UTC")
    return pd.Series(np.repeat(weeks, per_week))


def test_robust_t_equals_naive_t_for_independent_weekly_values():
    rng = np.random.default_rng(0)
    values = pd.Series(rng.normal(0.5, 2.0, 200))
    naive = values.mean() / (values.std(ddof=0) / np.sqrt(len(values)))
    assert np.isclose(backtest.robust_t(values, weekly_dates(200), lags=0), naive)


def test_robust_t_counts_simultaneous_signals_once():
    """10 identische Signale pro Woche bringen keine zusaetzliche Information."""
    rng = np.random.default_rng(1)
    single = pd.Series(rng.normal(0.5, 2.0, 100))
    repeated = pd.Series(np.repeat(single.to_numpy(), 10))
    t_single = backtest.robust_t(single, weekly_dates(100), lags=0)
    t_repeated = backtest.robust_t(repeated, weekly_dates(100, per_week=10), lags=0)
    assert np.isclose(t_single, t_repeated)


def test_robust_t_has_same_sign_as_mean():
    values = pd.Series([3.0, 3.0, 3.0, -1.0, -1.0, 2.0])
    dates = pd.Series(weekly_dates(3, per_week=2).to_numpy())
    assert np.sign(backtest.robust_t(values, dates)) == np.sign(values.mean())


def test_signal_starts_marks_phase_begin_and_respects_cooldown():
    dates = pd.date_range("2024-01-01", periods=8, freq="7D", tz="UTC")
    #            Beginn      Unterbruch, < 30 Tage   Beginn nach > 30 Tagen
    oversold = [70, 70, 10, 70, 10, 10, 70, 70]
    df = pd.DataFrame({"symbol": "X", "date": dates, "oversold": oversold})
    starts = calibrate.signal_starts(df, 60, cooldown_days=30)
    assert starts.tolist() == [True, False, False, False, False, False, True, False]


def test_signal_starts_per_symbol():
    dates = pd.date_range("2024-01-01", periods=2, freq="7D", tz="UTC")
    df = pd.DataFrame({"symbol": ["A", "A", "B", "B"], "date": list(dates) * 2,
                       "oversold": [70, 70, 10, 70]})
    assert calibrate.signal_starts(df, 60).tolist() == [True, False, False, True]


def test_run_backtests_parallel_matches_serial():
    closes = np.concatenate([geometric(100, 0.001, 280), geometric(132, -0.01, 60)])
    hist = make_hist(closes)
    histories = {"A": hist, "B": hist.iloc[:300]}
    serial = pd.DataFrame(backtest.run_backtests(histories, step=10, jobs=1))
    parallel = pd.DataFrame(backtest.run_backtests(histories, step=10, jobs=2))
    pd.testing.assert_frame_equal(serial, parallel)
    assert set(serial["symbol"]) == {"A", "B"}
    assert len(serial) > 0
