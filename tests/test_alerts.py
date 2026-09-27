import datetime as dt

import pytest

import alerts


def test_parse_rules_ignores_comments_and_blank_lines():
    rules = alerts.parse_rules("""
        # Kommentar
        aapl  price  <  300   # inline
        MSFT  OVERSOLD >= 60
    """)
    assert rules == [
        alerts.Rule("AAPL", "price", "<", 300.0),
        alerts.Rule("MSFT", "oversold", ">=", 60.0),
    ]


@pytest.mark.parametrize("line", [
    "AAPL price 300",
    "AAPL volume < 300",
    "AAPL price == 300",
    "AAPL price < abc",
    "AAPL setup < stark",
    "AAPL setup = super",
    "AAPL price = 300",
])
def test_parse_rules_rejects_invalid_lines(line):
    with pytest.raises(ValueError, match="Zeile 1"):
        alerts.parse_rules(line)


def _values(price=290.0, oversold=65):
    return {"AAPL": {"symbol": "AAPL", "name": "Apple Inc.", "currency": "USD",
                     "price": price, "change_pct": -2.5, "oversold_score": oversold}}


RULES = [alerts.Rule("AAPL", "price", "<", 300), alerts.Rule("AAPL", "oversold", ">=", 60)]


def test_evaluate_reports_newly_triggered_rules():
    active, messages = alerts.evaluate(RULES, _values(), previous=set())
    assert active == {"AAPL price < 300", "AAPL oversold >= 60"}
    assert messages == [
        "🔔 AAPL (Apple Inc.): Kurs 290.00 USD < 300",
        "🔔 AAPL (Apple Inc.): Oversold 65 >= 60",
    ]


def test_evaluate_does_not_repeat_already_reported_rules():
    active, messages = alerts.evaluate(RULES, _values(), previous={"AAPL price < 300"})
    assert "AAPL price < 300" in active
    assert messages == ["🔔 AAPL (Apple Inc.): Oversold 65 >= 60"]


def test_evaluate_rearms_rule_when_condition_no_longer_holds():
    active, messages = alerts.evaluate(RULES, _values(price=310, oversold=10),
                                       previous={"AAPL price < 300"})
    assert active == set()
    assert messages == []


def test_evaluate_keeps_state_when_data_missing():
    active, messages = alerts.evaluate(RULES, {}, previous={"AAPL price < 300"})
    assert active == {"AAPL price < 300"}
    assert messages == []


def test_change_message_shows_percent():
    rule = alerts.Rule("AAPL", "change", "<=", -2)
    _, messages = alerts.evaluate([rule], _values(), previous=set())
    assert messages == ["🔔 AAPL (Apple Inc.): Tagesaenderung -2.50% <= -2%"]


def test_state_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    assert alerts.load_state(path) == set()
    alerts.save_state(path, {"B", "A"})
    assert alerts.load_state(path) == {"A", "B"}


def test_parse_setup_and_earnings_rules():
    rules = alerts.parse_rules("NESN.SW setup = Stark\nAAPL earnings <= 7")
    assert rules == [
        alerts.Rule("NESN.SW", "setup", "=", "stark"),
        alerts.Rule("AAPL", "earnings", "<=", 7.0),
    ]
    assert rules[0].key == "NESN.SW setup = stark"


def test_setup_rule_matches_full_setup_label():
    rule = alerts.Rule("X", "setup", "=", "stark")
    values = {"X": {"setup": alerts.sc.SETUP_STRONG_OVERSOLD, "oversold_score": 65}}
    _, messages = alerts.evaluate([rule], values, previous=set())
    assert messages == [f"🔔 X: Setup {alerts.sc.SETUP_STRONG_OVERSOLD} (Oversold 65)"]

    values["X"]["setup"] = alerts.sc.SETUP_NEUTRAL
    assert alerts.evaluate([rule], values, previous=set()) == (set(), [])
    assert alerts.Rule("X", "setup", "!=", "stark").matches(alerts.sc.SETUP_NEUTRAL)


def test_earnings_rule():
    rule = alerts.Rule("AAPL", "earnings", "<=", 7)
    values = {"AAPL": {"earnings_days": 5, "earnings_date": dt.date(2026, 10, 2)}}
    _, messages = alerts.evaluate([rule], values, previous=set())
    assert messages == ["🔔 AAPL: Quartalszahlen in 5 Tagen (02.10.2026)"]
    # ohne bekannten Termin: keine Daten, keine Meldung
    assert alerts.evaluate([rule], {"AAPL": {"earnings_days": None}}, set()) == (set(), [])


def test_days_until():
    assert alerts.days_until(dt.date(2026, 10, 2), today=dt.date(2026, 9, 27)) == 5
    assert alerts.days_until(None) is None
