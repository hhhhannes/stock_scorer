import datetime as dt

import numpy as np
import pandas as pd
import pytest

import stock_scorer as sc


def make_hist(closes, volume=1_000_000) -> pd.DataFrame:
    close = pd.Series(closes, dtype=float, index=pd.bdate_range("2024-01-01", periods=len(closes)))
    return pd.DataFrame({
        "Open": close,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": volume,
    })


def geometric(start: float, rate: float, n: int) -> np.ndarray:
    return start * (1 + rate) ** np.arange(n)


# --- Indikatoren -------------------------------------------------------------

def test_rsi_only_gains_is_100():
    assert sc.compute_rsi(pd.Series(np.arange(1, 51, dtype=float))) == pytest.approx(100)


def test_rsi_only_losses_is_0():
    assert sc.compute_rsi(pd.Series(np.arange(50, 0, -1, dtype=float))) == pytest.approx(0)


def test_rsi_flat_is_50():
    assert sc.compute_rsi(pd.Series([100.0] * 50)) == pytest.approx(50)


def test_rsi_alternating_is_near_50():
    closes = pd.Series([100.0 + (i % 2) for i in range(100)])
    assert 40 < sc.compute_rsi(closes) < 60


def test_rsi_too_short_falls_back_to_50():
    assert sc.compute_rsi(pd.Series([1.0, 2.0, 3.0])) == 50.0


def test_linear_regression_slope():
    values = pd.Series(np.arange(100, 120, dtype=float))
    assert sc.linear_regression_slope(values) == pytest.approx(1 / 109.5 * 100)


def test_bollinger_constant_series_collapses():
    lower, mid, upper = sc.compute_bollinger(pd.Series([10.0] * 30))
    assert lower == mid == upper == pytest.approx(10)


def test_atr_constant_range():
    close = pd.Series([100.0] * 40)
    hist = pd.DataFrame({"High": close + 1, "Low": close - 1, "Close": close})
    assert sc.compute_atr(hist) == pytest.approx(2)


# --- Scores ------------------------------------------------------------------

def test_trend_score_steady_uptrend_is_max():
    score, _ = sc.trend_score(make_hist(geometric(100, 0.005, 300)))
    assert score == sc.MAX_SCORE


def test_trend_score_steady_downtrend_is_low():
    score, _ = sc.trend_score(make_hist(geometric(100, -0.005, 300)))
    assert score < sc.TREND_SIDEWAYS


def test_oversold_score_downtrend_is_high():
    score, notes = sc.oversold_score(make_hist(geometric(100, -0.005, 300)))
    assert score >= sc.OVERSOLD_HIGH
    assert any("ueberverkauft" in n for n in notes)


def test_oversold_score_uptrend_is_low():
    score, _ = sc.oversold_score(make_hist(geometric(100, 0.005, 300)))
    assert score < sc.OVERSOLD_LOW


def test_oversold_note_has_no_double_negative():
    _, notes = sc.oversold_score(make_hist(geometric(100, -0.005, 300)))
    note = next(n for n in notes if "52W-Hoch" in n)
    assert not note.startswith("-")


RECOVERY_CLEAR = 35  # deutliche Wende-Signale (Recovery ist nur noch Kontext, keine Setup-Schwelle)


def test_recovery_score_during_decline_is_low():
    score, _ = sc.recovery_score(make_hist(geometric(100, -0.005, 300)))
    assert score < RECOVERY_CLEAR


def test_recovery_score_after_rebound_is_high():
    decline = geometric(100, -0.01, 100)
    rebound = decline[-1] * (1.02 ** np.arange(1, 9))
    score, _ = sc.recovery_score(make_hist(np.concatenate([decline, rebound])))
    assert score >= RECOVERY_CLEAR


@pytest.mark.parametrize("os_, expected", [
    (75, sc.SETUP_STRONG_OVERSOLD),
    (sc.OVERSOLD_STRONG, sc.SETUP_STRONG_OVERSOLD),
    (sc.OVERSOLD_STRONG - 1, sc.SETUP_PRESSED),
    (sc.OVERSOLD_HIGH, sc.SETUP_PRESSED),
    (sc.OVERSOLD_HIGH - 1, sc.SETUP_NEUTRAL),
    (sc.OVERSOLD_LOW, sc.SETUP_NEUTRAL),
    (sc.OVERSOLD_LOW - 1, sc.SETUP_NOT_CHEAP),
    (0, sc.SETUP_NOT_CHEAP),
])
def test_classify_setup(os_, expected):
    assert sc.classify_setup(os_) == expected


@pytest.mark.parametrize("trend, expected", [
    (75, sc.TREND_LABEL_UP),
    (sc.TREND_UP, sc.TREND_LABEL_UP),
    (sc.TREND_UP - 1, sc.TREND_LABEL_SIDEWAYS),
    (sc.TREND_SIDEWAYS, sc.TREND_LABEL_SIDEWAYS),
    (sc.TREND_SIDEWAYS - 1, sc.TREND_LABEL_DOWN),
    (-5, sc.TREND_LABEL_DOWN),
])
def test_classify_trend(trend, expected):
    assert sc.classify_trend(trend) == expected


# --- Fundamentaldaten & Termine ---------------------------------------------

def test_fundamental_score_empty():
    assert sc.fundamental_score({}) == (0, ["keine Fundamentaldaten verfuegbar"])


def test_fundamental_score_max():
    info = {"trailingPE": 20, "forwardPE": 15, "revenueGrowth": 0.2,
            "profitMargins": 0.2, "targetMeanPrice": 120, "currentPrice": 100}
    score, _ = sc.fundamental_score(info)
    assert score == sc.MAX_FUNDAMENTAL_SCORE


def test_earnings_info():
    today = dt.date(2026, 1, 10)
    assert sc.earnings_info(None, today)[1] is False
    assert sc.earnings_info(dt.date(2026, 1, 13), today)[1] is True
    assert sc.earnings_info(dt.date(2026, 2, 10), today)[1] is False


def test_next_earnings_date_from_dict():
    class FakeTicker:
        calendar = {"Earnings Date": [dt.date(2026, 3, 1), dt.date(2026, 3, 5)]}

    assert sc.next_earnings_date(FakeTicker()) == dt.date(2026, 3, 1)


def test_next_earnings_date_handles_errors():
    class FakeTicker:
        @property
        def calendar(self):
            raise RuntimeError("Yahoo down")

    assert sc.next_earnings_date(FakeTicker()) is None


# --- Ausgabe -----------------------------------------------------------------

def _result(symbol, os_, rec):
    return {"symbol": symbol, "name": symbol, "currency": "USD", "price": 1.0,
            "trend_score": 0, "oversold_score": os_, "recovery_score": rec,
            "fundamental_score": 0, "setup": sc.SETUP_NEUTRAL, "earnings_warning": False}


def test_sort_results_by_oversold_then_recovery_and_drops_errors():
    results = [_result("A", 30, 30), _result("B", 50, 0), _result("C", 30, 40),
               {"symbol": "X", "error": "kaputt"}]
    assert [r["symbol"] for r in sc.sort_results(results)] == ["B", "C", "A"]


def test_format_markdown_contains_table_and_errors():
    md = sc.format_markdown([_result("A", 10, 10), {"symbol": "X", "error": "kaputt"}])
    assert "| A |" in md
    assert "- X: kaputt" in md
