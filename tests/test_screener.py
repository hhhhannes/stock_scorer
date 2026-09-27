import datetime as dt

import numpy as np
import pandas as pd

import screener
import stock_scorer as sc

from test_stock_scorer import geometric, make_hist


def crash_hist(crash_days: int = 20) -> pd.DataFrame:
    """Ein Jahr leichter Anstieg, dann ein Absturz -> am Ende stark ueberverkauft."""
    return make_hist(np.concatenate([geometric(100, 0.001, 280),
                                     geometric(132, -0.02, crash_days)]))


def test_current_signal_finds_start_of_strong_phase():
    hist = crash_hist()
    start, score = screener.current_signal(hist)
    scores = sc.oversold_history(hist, days=screener.LOOKBACK_DAYS)
    assert score >= sc.OVERSOLD_STRONG
    assert scores[start] >= sc.OVERSOLD_STRONG
    assert scores[scores.index[scores.index.get_loc(start) - 1]] < sc.OVERSOLD_STRONG


def test_current_signal_none_when_not_oversold():
    assert screener.current_signal(make_hist(geometric(100, 0.001, 300))) is None


def test_find_new_signals_respects_cooldown():
    histories = {"X": crash_hist()}
    new = screener.find_new_signals(histories, {})
    assert [s["symbol"] for s in new] == ["X"]

    signal_date = new[0]["date"].date()
    assert screener.find_new_signals(histories, {"X": signal_date}) == []
    # Signal lange vor der Sperrfrist -> wieder melden
    old = signal_date - dt.timedelta(days=screener.COOLDOWN_DAYS + 1)
    assert len(screener.find_new_signals(histories, {"X": old})) == 1


def test_state_roundtrip_drops_expired_entries(tmp_path):
    path = tmp_path / "state.json"
    assert screener.load_state(path) == {}
    today = dt.date(2026, 9, 27)
    screener.save_state(path, {"A": dt.date(2026, 9, 20), "B": dt.date(2025, 1, 1)}, today=today)
    assert screener.load_state(path) == {"A": dt.date(2026, 9, 20)}


def test_format_message():
    new = [{"symbol": "NIO", "date": pd.Timestamp("2026-09-25"), "price": 3.58, "oversold": 65.0}]
    quotes = {"NIO": {"name": "NIO Inc.", "currency": "USD"}}
    assert screener.format_message(new, quotes).splitlines()[1] == (
        "• NIO NIO Inc.: Oversold 65, Kurs 3.58 USD (Signal 25.09.)")
