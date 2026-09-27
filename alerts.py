"""
alerts.py - Preis- und Score-Alarme per Telegram
=================================================
Liest Regeln aus alerts.txt, prueft sie gegen aktuelle Kurse bzw. Scores und
schickt bei neu ausgeloesten Regeln eine Telegram-Nachricht.

Regel-Format (eine pro Zeile, '#' leitet Kommentare ein):
    SYMBOL  KENNZAHL  OPERATOR  SCHWELLE
    AAPL    price     <         300
    MSFT    change    <=        -3
    CFR.SW  oversold  >=        60
    NESN.SW setup     =         stark
    AAPL    earnings  <=        7

Kennzahlen: price, change (Tagesaenderung in %), trend, oversold, recovery, fundamental,
            earnings (Tage bis zu den naechsten Quartalszahlen)
            -> Operatoren: <, <=, >, >=
            setup (stark, gedrueckt, neutral, teuer)
            -> Operatoren: =, !=

Eine Regel meldet sich nur beim Ueberschreiten der Schwelle, nicht bei jedem
Lauf erneut. Erst wenn die Bedingung nicht mehr erfuellt ist, wird sie wieder
scharf. Der Zustand liegt in alerts_state.json.

Telegram-Zugang ueber Umgebungsvariablen TELEGRAM_BOT_TOKEN und TELEGRAM_CHAT_ID.
Fehlen sie, werden die Alarme nur ausgegeben (Trockenlauf).

Aufruf:
    python alerts.py [--rules alerts.txt] [--state alerts_state.json] [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import operator
import os
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import stock_scorer as sc

# Kennzahl -> (Anzeigename, Feld im Ergebnis-Dict)
METRICS = {
    "price": ("Kurs", "price"),
    "change": ("Tagesaenderung", "change_pct"),
    "trend": ("Trend", "trend_score"),
    "oversold": ("Oversold", "oversold_score"),
    "recovery": ("Recovery", "recovery_score"),
    "fundamental": ("Fundamental", "fundamental_score"),
    "earnings": ("Earnings", "earnings_days"),
    "setup": ("Setup", "setup"),
}
QUOTE_METRICS = {"price", "change"}  # brauchen nur den Kurs, keine volle Analyse

NUMERIC_OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}
SETUP_OPERATORS = {
    "=": operator.eq,
    "!=": operator.ne,
}
OPERATORS = {**NUMERIC_OPERATORS, **SETUP_OPERATORS}

# Kurzname in alerts.txt -> Setup-Bezeichnung aus stock_scorer
SETUP_VALUES = {
    "stark": sc.SETUP_STRONG_OVERSOLD,
    "gedrueckt": sc.SETUP_PRESSED,
    "neutral": sc.SETUP_NEUTRAL,
    "teuer": sc.SETUP_NOT_CHEAP,
}


@dataclass(frozen=True)
class Rule:
    symbol: str
    metric: str
    op: str
    threshold: float | str

    @property
    def threshold_text(self) -> str:
        if isinstance(self.threshold, str):
            return self.threshold
        return f"{self.threshold:g}"

    @property
    def key(self) -> str:
        return f"{self.symbol} {self.metric} {self.op} {self.threshold_text}"

    def matches(self, value: float | str) -> bool:
        threshold = SETUP_VALUES[self.threshold] if self.metric == "setup" else self.threshold
        return OPERATORS[self.op](value, threshold)


# ---------------------------------------------------------------------------
# Regeln einlesen
# ---------------------------------------------------------------------------

def parse_rules(text: str) -> list[Rule]:
    rules = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"Zeile {lineno}: erwartet 'SYMBOL KENNZAHL OPERATOR SCHWELLE': {raw!r}")
        symbol, metric, op, threshold = parts
        metric = metric.lower()
        if metric not in METRICS:
            raise ValueError(f"Zeile {lineno}: unbekannte Kennzahl {metric!r} "
                             f"(erlaubt: {', '.join(METRICS)})")
        allowed = SETUP_OPERATORS if metric == "setup" else NUMERIC_OPERATORS
        if op not in allowed:
            raise ValueError(f"Zeile {lineno}: Operator {op!r} passt nicht zu {metric!r} "
                             f"(erlaubt: {', '.join(allowed)})")
        if metric == "setup":
            if threshold.lower() not in SETUP_VALUES:
                raise ValueError(f"Zeile {lineno}: unbekanntes Setup {threshold!r} "
                                 f"(erlaubt: {', '.join(SETUP_VALUES)})")
            value = threshold.lower()
        else:
            try:
                value = float(threshold)
            except ValueError:
                raise ValueError(f"Zeile {lineno}: Schwelle ist keine Zahl: {threshold!r}") from None
        rules.append(Rule(symbol.upper(), metric, op, value))
    return rules


# ---------------------------------------------------------------------------
# Daten laden & Regeln auswerten
# ---------------------------------------------------------------------------

def load_values(rules: list[Rule]) -> dict[str, dict]:
    """Aktuelle Kennzahlen pro Symbol; volle Analyse nur fuer Symbole mit Score-Regeln."""
    symbols = list(dict.fromkeys(r.symbol for r in rules))
    score_symbols = list(dict.fromkeys(r.symbol for r in rules if r.metric not in QUOTE_METRICS))

    values: dict[str, dict] = {}
    for result in sc.analyze_many(score_symbols) if score_symbols else []:
        if "error" in result:
            print(f"{result['symbol']}: {result['error']}", file=sys.stderr)
        else:
            values[result["symbol"]] = {**result, "earnings_days": days_until(result["earnings_date"])}
    for sym, quote in sc.fetch_quotes(symbols).items():
        values[sym] = {**values.get(sym, {"symbol": sym}), **quote}
    return values


def days_until(date: dt.date | None, today: dt.date | None = None) -> int | None:
    if date is None:
        return None
    return (date - (today or dt.date.today())).days


def format_value(rule: Rule, data: dict) -> str:
    value = data[METRICS[rule.metric][1]]
    if rule.metric == "price":
        return f"{value:.2f} {data.get('currency', '')}".strip()
    if rule.metric == "change":
        return f"{value:+.2f}%"
    return f"{value:.0f}"


def alert_message(rule: Rule, data: dict) -> str:
    label = METRICS[rule.metric][0]
    name = data.get("name")
    head = f"{rule.symbol} ({name})" if name and name != rule.symbol else rule.symbol
    if rule.metric == "setup":
        return f"🔔 {head}: Setup {data['setup']} (Oversold {data['oversold_score']:.0f})"
    if rule.metric == "earnings":
        return (f"🔔 {head}: Quartalszahlen in {data['earnings_days']} Tagen "
                f"({data['earnings_date']:%d.%m.%Y})")
    unit = "%" if rule.metric == "change" else ""
    return f"🔔 {head}: {label} {format_value(rule, data)} {rule.op} {rule.threshold:g}{unit}"


def evaluate(rules: list[Rule], values: dict[str, dict],
             previous: set[str]) -> tuple[set[str], list[str]]:
    """
    Gibt (aktuell ausgeloeste Regeln, Nachrichten fuer neu ausgeloeste Regeln) zurueck.
    Regeln ohne Daten behalten ihren bisherigen Zustand, damit ein Ladefehler
    keine erneute Meldung ausloest.
    """
    active: set[str] = set()
    messages: list[str] = []
    for rule in rules:
        data = values.get(rule.symbol)
        field = METRICS[rule.metric][1]
        if data is None or data.get(field) is None:
            print(f"{rule.key}: keine Daten", file=sys.stderr)
            if rule.key in previous:
                active.add(rule.key)
            continue
        if rule.matches(data[field]):
            active.add(rule.key)
            if rule.key not in previous:
                messages.append(alert_message(rule, data))
    return active, messages


# ---------------------------------------------------------------------------
# Zustand & Telegram
# ---------------------------------------------------------------------------

def load_state(path: Path) -> set[str]:
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (FileNotFoundError, ValueError):
        return set()


def save_state(path: Path, active: set[str]):
    path.write_text(json.dumps(sorted(active), indent=2), encoding="utf-8")


def send_telegram(text: str, token: str, chat_id: str):
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    with urllib.request.urlopen(url, data=data, timeout=30) as resp:
        body = json.loads(resp.read())
    if not body.get("ok"):
        raise RuntimeError(f"Telegram-Fehler: {body}")


def main():
    parser = argparse.ArgumentParser(description="Preis- und Score-Alarme per Telegram")
    parser.add_argument("--rules", default="alerts.txt", type=Path, help="Regeldatei (Standard: alerts.txt)")
    parser.add_argument("--state", default="alerts_state.json", type=Path,
                        help="Zustandsdatei (Standard: alerts_state.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur ausgeben, nichts senden und keinen Zustand speichern")
    args = parser.parse_args()

    rules = parse_rules(args.rules.read_text(encoding="utf-8"))
    if not rules:
        print("Keine Regeln definiert.")
        return

    previous = load_state(args.state)
    active, messages = evaluate(rules, load_values(rules), previous)

    print(f"{len(rules)} Regeln geprueft, {len(active)} erfuellt, {len(messages)} neu.")
    for m in messages:
        print(m)

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if args.dry_run:
        return
    if messages:
        if token and chat_id:
            send_telegram("\n".join(messages), token, chat_id)
        else:
            print("TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID fehlen - nichts gesendet.", file=sys.stderr)
            return  # Zustand nicht speichern, damit die Meldung spaeter nachgeholt wird
    save_state(args.state, active)


if __name__ == "__main__":
    main()
