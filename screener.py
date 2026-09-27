#!/usr/bin/env python3
"""
screener.py - Taeglicher Screener 'Stark ueberverkauft'
=========================================================
Prueft alle Ticker aus universe.txt auf das Setup 'Stark ueberverkauft'
(das einzige Signal, das im Backtest stabil war) und meldet neue Signale
per Telegram.

Ein Signal ist der erste Tag einer Phase 'Stark ueberverkauft'. Pro Titel gilt
nach einem Signal eine Sperrfrist (COOLDOWN_DAYS), damit kurze Unterbrechungen
nicht als neues Signal zaehlen. Die zuletzt gemeldeten Signale liegen in
screener_state.json.

Telegram-Zugang ueber TELEGRAM_BOT_TOKEN und TELEGRAM_CHAT_ID (wie alerts.py).

Aufruf:
    python screener.py [--universe universe.txt] [--state screener_state.json] [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

import stock_scorer as sc
from alerts import send_telegram
from backtest import load_universe

HISTORY_PERIOD = "2y"      # genug fuer 252-Tage-Fenster plus Rueckblick
LOOKBACK_DAYS = 40         # so weit zurueck wird der Beginn einer Signal-Phase gesucht
COOLDOWN_DAYS = 30         # Kalendertage Sperrfrist pro Titel nach einem Signal


# ---------------------------------------------------------------------------
# Zustand: letztes gemeldetes Signal pro Titel
# ---------------------------------------------------------------------------

def load_state(path: Path) -> dict[str, dt.date]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}
    return {sym: dt.date.fromisoformat(d) for sym, d in raw.items()}


def save_state(path: Path, state: dict[str, dt.date], today: dt.date | None = None):
    """Speichert nur Eintraege, deren Sperrfrist noch laeuft."""
    cutoff = (today or dt.date.today()) - dt.timedelta(days=COOLDOWN_DAYS + LOOKBACK_DAYS)
    kept = {sym: d.isoformat() for sym, d in sorted(state.items()) if d >= cutoff}
    path.write_text(json.dumps(kept, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Signale finden
# ---------------------------------------------------------------------------

def current_signal(hist: pd.DataFrame) -> tuple[pd.Timestamp, float] | None:
    """Beginn (Datum, Score) der laufenden Phase 'Stark ueberverkauft', sonst None.

    Laeuft die Phase schon laenger als LOOKBACK_DAYS, ist sie kein neues Signal mehr.
    """
    if len(hist) < sc.TRADING_DAYS_PER_YEAR:
        return None
    scores = sc.oversold_history(hist, days=LOOKBACK_DAYS)
    strong = scores >= sc.OVERSOLD_STRONG
    if not strong.iloc[-1] or strong.all():
        return None
    start = strong[~strong].index[-1]
    start = strong.index[strong.index.get_loc(start) + 1]
    return start, float(scores[start])


def find_new_signals(histories: dict[str, pd.DataFrame], state: dict[str, dt.date]) -> list[dict]:
    new = []
    for sym, hist in histories.items():
        signal = current_signal(hist)
        if signal is None:
            continue
        date, score = signal
        last = state.get(sym)
        if last is not None and last >= date.date() - dt.timedelta(days=COOLDOWN_DAYS):
            continue
        new.append({"symbol": sym, "date": date, "price": float(hist["Close"][date]),
                    "oversold": score})
    return new


# ---------------------------------------------------------------------------
# Nachricht
# ---------------------------------------------------------------------------

def format_message(new: list[dict], quotes: dict[str, dict]) -> str:
    lines = [f"📉 Screener: neu stark ueberverkauft (Oversold >= {sc.OVERSOLD_STRONG})"]
    for s in new:
        q = quotes.get(s["symbol"], {})
        name = f" {q['name']}" if q.get("name") else ""
        lines.append(f"• {s['symbol']}{name}: Oversold {s['oversold']:.0f}, "
                     f"Kurs {s['price']:.2f} {q.get('currency', '')}".rstrip()
                     + f" (Signal {s['date']:%d.%m.})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Screener 'Stark ueberverkauft'")
    parser.add_argument("--universe", default="universe.txt", help="Ticker-Datei (Standard: universe.txt)")
    parser.add_argument("--state", default="screener_state.json", type=Path,
                        help="Zustandsdatei (Standard: screener_state.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur ausgeben, nichts senden und keinen Zustand speichern")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    sys.stdout.reconfigure(encoding="utf-8")

    symbols = load_universe(args.universe)
    print(f"Lade {HISTORY_PERIOD} Historie fuer {len(symbols)} Ticker ...")
    histories = {s: h for s, h in sc.fetch_histories(symbols, period=HISTORY_PERIOD).items()
                 if not h.empty}

    state = load_state(args.state)
    new = find_new_signals(histories, state)
    print(f"{len(histories)} Ticker geprueft, {len(new)} neue Signale.")
    if not new:
        return

    message = format_message(new, sc.fetch_quotes([s["symbol"] for s in new]))
    print("\n" + message)
    if args.dry_run:
        return

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        print("TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID fehlen - nichts gesendet.", file=sys.stderr)
        return  # Zustand nicht speichern, damit die Meldung spaeter nachgeholt wird
    send_telegram(message, token, chat_id)
    state.update({s["symbol"]: s["date"].date() for s in new})
    save_state(args.state, state)


if __name__ == "__main__":
    main()
