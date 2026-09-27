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
