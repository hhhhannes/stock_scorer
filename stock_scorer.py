#!/usr/bin/env python3
"""
stock_scorer.py
================
Prueft Aktien auf drei Dimensionen:
  - Trend-Score:     wie stark/klar ist der aktuelle Kurstrend?
  - Oversold-Score:  wie weit/stark ist die Aktie gegenueber sich selbst gedrueckt?
  - Recovery-Score:  gibt es bereits Anzeichen, dass sich eine Drueckung dreht?

Installation (in einem virtuellen Environment):
    python -m venv .venv
    .venv\\Scripts\\activate          (Windows)   bzw.   source .venv/bin/activate
    pip install -r requirements-cli.txt

Benutzung:
    python stock_scorer.py AAPL MSFT NVDA
    python stock_scorer.py AAPL MSFT --markdown report.md
"""

from __future__ import annotations

import argparse
import sys
import datetime as dt
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate


# ---------------------------------------------------------------------------
# Konfiguration: Schwellen und Punkte
# ---------------------------------------------------------------------------

HISTORY_PERIOD = "1y"
MIN_HISTORY_DAYS = 30
TRADING_DAYS_PER_YEAR = 252

MAX_SCORE = 75              # Maximum fuer Trend, Oversold und Recovery
MAX_FUNDAMENTAL_SCORE = 40

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_WEAK = 40

# Trend-Einordnung (nur Kontext, fliesst nicht ins Setup ein)
TREND_UP = 45               # Trend-Score >= -> Aufwaerts
TREND_SIDEWAYS = 20         # Trend-Score >= -> Seitwaerts, darunter Abwaerts

# Setup-Einordnung: nur ueber den Oversold-Score. Im Backtest (96 Titel, 2017-2026,
# Train/Test-Split) war das das einzige Signal, das auch ausserhalb der Suchdaten hielt;
# Trend- und Recovery-Bedingungen haben es nicht verbessert (siehe README, calibrate.py).
OVERSOLD_STRONG = 60        # >= -> Stark ueberverkauft
OVERSOLD_HIGH = 45          # >= -> Gedrueckt
OVERSOLD_LOW = 20           # <  -> Nicht guenstig

EARNINGS_WARN_DAYS = 7

TREND_POINTS = {
    "above_sma50": 15,
    "above_sma200": 15,
    "sma50_above_sma200": 15,
    "sma50_slope_strong": 15,   # SMA50-Steigung 10T > 1%
    "sma50_slope": 7,           # SMA50-Steigung 10T > 0%
    "regression_strong": 15,    # Regression 20T > 0.15%/Tag
    "regression": 7,            # Regression 20T > 0
    "regression_down": -5,      # Regression 20T < -0.15%/Tag
}
SMA50_SLOPE_STRONG_PCT = 1.0
REGRESSION_STRONG_PCT = 0.15

OVERSOLD_POINTS = {
    "rsi_oversold": 25,
    "rsi_weak": 12,
    "range_bottom": 20,         # Position im 52W-Range < 20%
    "range_low": 10,            # Position im 52W-Range < 35%
    "far_below_sma50": 15,      # > 10% unter SMA50
    "below_sma50": 8,           # > 3% unter SMA50
    "below_bollinger": 15,
    "below_bollinger_mid": 5,
}

RECOVERY_POINTS = {
    "rebound_strong": 20,       # >= 1.5 ATR ueber dem 10T-Tief
    "rebound": 10,              # >= 0.75 ATR ueber dem 10T-Tief
    "sma10_recaptured": 15,
    "macd_cross": 20,
    "macd_above": 8,
    "higher_lows": 10,
    "volume_up_days": 10,
}
REBOUND_STRONG_ATR = 1.5
REBOUND_ATR = 0.75
VOLUME_RATIO_BULLISH = 1.3

TREND_LABEL_UP = "Aufwaerts"
TREND_LABEL_SIDEWAYS = "Seitwaerts"
TREND_LABEL_DOWN = "Abwaerts"

SETUP_STRONG_OVERSOLD = "Stark ueberverkauft"
SETUP_PRESSED = "Gedrueckt"
SETUP_NEUTRAL = "Neutral"
SETUP_NOT_CHEAP = "Nicht guenstig"


# ---------------------------------------------------------------------------
# Technische Indikatoren
# ---------------------------------------------------------------------------

def rsi_series(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI mit Wilder-Glaettung (wie TradingView & Co.)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    # Sonderfaelle: keine Verluste -> 100, weder Gewinne noch Verluste -> 50
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where((avg_gain != 0) | (avg_loss != 0), 50.0)
    return rsi


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> float:
    rsi = rsi_series(close, period)
    if rsi.empty or np.isnan(rsi.iloc[-1]):
        return 50.0
    return float(rsi.iloc[-1])


def compute_atr(hist: pd.DataFrame, period: int = 14) -> float:
    """Average True Range mit Wilder-Glaettung."""
    high, low, close = hist["High"], hist["Low"], hist["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return float(atr.iloc[-1])


def compute_macd(close: pd.Series, fast=12, slow=26, signal=9) -> tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def bollinger_bands(close: pd.Series, period=20, num_std=2) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Gibt (unteres Band, Mittellinie, oberes Band) als Serien zurueck."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return sma - num_std * std, sma, sma + num_std * std


def compute_bollinger(close: pd.Series, period=20, num_std=2) -> tuple[float, float, float]:
    lower, mid, upper = bollinger_bands(close, period, num_std)
    return float(lower.iloc[-1]), float(mid.iloc[-1]), float(upper.iloc[-1])


def linear_regression_slope(values: pd.Series) -> float:
    """Steigung einer linearen Regression, normalisiert auf % pro Tag relativ zum Mittelwert."""
    y = values.values
    x = np.arange(len(y))
    if len(y) < 2 or np.mean(y) == 0:
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return float(slope / np.mean(y) * 100)  # % Veraenderung pro Tag


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def trend_score(hist: pd.DataFrame) -> tuple[float, list[str]]:
    """Wie stark/klar ist der aktuelle Kurstrend (Richtung + Konsistenz)?"""
    p = TREND_POINTS
    close = hist["Close"]
    price = float(close.iloc[-1])
    score = 0
    notes = []

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean() if len(close) >= 200 else None

    # 1) Kurs vs. SMA50
    if not np.isnan(sma50.iloc[-1]):
        if price > sma50.iloc[-1]:
            score += p["above_sma50"]
            notes.append("Kurs > SMA50")
        else:
            notes.append("Kurs < SMA50")

    # 2) Kurs vs. SMA200
    if sma200 is not None and not np.isnan(sma200.iloc[-1]):
        if price > sma200.iloc[-1]:
            score += p["above_sma200"]
            notes.append("Kurs > SMA200")
        else:
            notes.append("Kurs < SMA200")

        # 3) SMA50 vs SMA200 (Golden/Death Cross Lage)
        if not np.isnan(sma50.iloc[-1]):
            if sma50.iloc[-1] > sma200.iloc[-1]:
                score += p["sma50_above_sma200"]
                notes.append("SMA50 > SMA200 (bullische Lage)")
            else:
                notes.append("SMA50 < SMA200 (baerische Lage)")
    else:
        notes.append("SMA200 nicht verfuegbar (zu wenig Historie)")

    # 4) Steigung der SMA50 (letzte 10 Tage) - zeigt ob Trend an Fahrt gewinnt
    if len(sma50.dropna()) > 10:
        sma50_slope_pct = (sma50.iloc[-1] / sma50.iloc[-11] - 1) * 100
        notes.append(f"SMA50-Steigung 10T: {sma50_slope_pct:+.1f}%")
        if sma50_slope_pct > SMA50_SLOPE_STRONG_PCT:
            score += p["sma50_slope_strong"]
        elif sma50_slope_pct > 0:
            score += p["sma50_slope"]

    # 5) Trendstaerke via linearer Regression der letzten 20 Handelstage
    if len(close) >= 20:
        slope_pct = linear_regression_slope(close.iloc[-20:])
        notes.append(f"Regressions-Steigung 20T: {slope_pct:+.2f}%/Tag")
        if slope_pct > REGRESSION_STRONG_PCT:
            score += p["regression_strong"]
        elif slope_pct > 0:
            score += p["regression"]
        elif slope_pct < -REGRESSION_STRONG_PCT:
            score += p["regression_down"]  # klarer Abwaertstrend

    return score, notes


def oversold_score(hist: pd.DataFrame) -> tuple[float, list[str]]:
    """Wie stark/weit ist die Aktie aktuell gedrueckt ('unter ihrem normalen Wert')?"""
    p = OVERSOLD_POINTS
    close = hist["Close"]
    price = float(close.iloc[-1])
    score = 0
    notes = []

    # RSI: klassischstes Ueberverkauft-Signal
    rsi = compute_rsi(close)
    if rsi < RSI_OVERSOLD:
        score += p["rsi_oversold"]
        notes.append(f"RSI14: {rsi:.0f} (ueberverkauft)")
    else:
        notes.append(f"RSI14: {rsi:.0f}")
        if rsi < RSI_WEAK:
            score += p["rsi_weak"]

    # Abstand zum 52-Wochen-Hoch/-Tief (aus Tageshochs/-tiefs)
    high_52w = float(hist["High"].iloc[-TRADING_DAYS_PER_YEAR:].max())
    low_52w = float(hist["Low"].iloc[-TRADING_DAYS_PER_YEAR:].min())
    range_52w = high_52w - low_52w
    if range_52w > 0:
        pct_below_high = (1 - price / high_52w) * 100
        pos_in_range = (price - low_52w) / range_52w * 100  # 0 = am Jahrestief, 100 = am Jahreshoch
        notes.append(f"{pct_below_high:.0f}% unter 52W-Hoch, Position im 52W-Range: {pos_in_range:.0f}%")
        if pos_in_range < 20:
            score += p["range_bottom"]
        elif pos_in_range < 35:
            score += p["range_low"]

    # Abstand zum SMA50
    sma50 = close.rolling(50).mean().iloc[-1]
    if not np.isnan(sma50):
        dist50 = (price / sma50 - 1) * 100
        notes.append(f"{dist50:+.1f}% vs. SMA50")
        if dist50 < -10:
            score += p["far_below_sma50"]
        elif dist50 < -3:
            score += p["below_sma50"]

    # Bollinger-Baender: Kurs nahe/unter dem unteren Band = statistisch guenstig
    if len(close) >= 20:
        lower, mid, _ = compute_bollinger(close)
        if price <= lower:
            score += p["below_bollinger"]
            notes.append("Kurs am/unter unterem Bollinger-Band")
        elif price < mid:
            score += p["below_bollinger_mid"]

    return score, notes


def recovery_score(hist: pd.DataFrame) -> tuple[float, list[str]]:
    """Gibt es bereits Anzeichen, dass sich eine gedrueckte Aktie zu drehen beginnt?

    Verwendet bewusst keinen RSI, damit der Score unabhaengig vom Oversold-Score ist.
    """
    p = RECOVERY_POINTS
    close = hist["Close"]
    price = float(close.iloc[-1])
    score = 0
    notes = []

    # Rebound vom 10-Tage-Tief, gemessen in ATR (volatilitaetsbereinigt)
    if len(hist) >= 15:
        atr = compute_atr(hist)
        low_10d = float(hist["Low"].iloc[-10:].min())
        if atr > 0 and not np.isnan(atr):
            rebound = (price - low_10d) / atr
            notes.append(f"Abstand zum 10T-Tief: {rebound:.1f} ATR")
            if rebound >= REBOUND_STRONG_ATR:
                score += p["rebound_strong"]
            elif rebound >= REBOUND_ATR:
                score += p["rebound"]

    # Kurs hat kurzfristigen gleitenden Durchschnitt (SMA10) zurueckerobert
    if len(close) >= 15:
        sma10 = close.rolling(10).mean()
        above = price > sma10.iloc[-1]
        crossed_above = above and close.iloc[-6] < sma10.iloc[-6]
        if crossed_above:
            score += p["sma10_recaptured"]
            notes.append("Kurs vs SMA10: ueber (frisch zurueckerobert)")
        else:
            notes.append(f"Kurs vs SMA10: {'ueber' if above else 'unter'}")

    # MACD-Line kreuzt Signal-Linie von unten nach oben (bullisches Signal)
    if len(close) >= 35:
        macd_line, signal_line = compute_macd(close)
        macd_now, sig_now = macd_line.iloc[-1], signal_line.iloc[-1]
        macd_prev, sig_prev = macd_line.iloc[-2], signal_line.iloc[-2]
        if macd_prev < sig_prev and macd_now > sig_now:
            score += p["macd_cross"]
            notes.append("MACD > Signal (frischer bullischer Crossover)")
        elif macd_now > sig_now:
            score += p["macd_above"]
            notes.append("MACD > Signal")
        else:
            notes.append("MACD < Signal")

    # Steigende Tiefs der letzten Tage (hoehere Lows = beginnender Aufwaertstrend)
    if len(hist) >= 10:
        lows = hist["Low"].iloc[-10:]
        first_min = lows.iloc[:5].min()
        if lows.iloc[-5:].min() > first_min and price > first_min:
            score += p["higher_lows"]
            notes.append("juengste Tiefs steigen")

    # Volumen an Aufwaertstagen vs. Abwaertstagen der letzten 20 Tage
    recent = hist.iloc[-21:]
    change = recent["Close"].diff()
    up_vol = recent.loc[change > 0, "Volume"]
    down_vol = recent.loc[change < 0, "Volume"]
    if not up_vol.empty and not down_vol.empty:
        vol_ratio = up_vol.mean() / max(down_vol.mean(), 1)
        notes.append(f"Volumen an Up-Tagen/Down-Tagen: {vol_ratio:.2f}x")
        if vol_ratio > VOLUME_RATIO_BULLISH:
            score += p["volume_up_days"]

    return score, notes


def volatility_info(hist: pd.DataFrame) -> tuple[float, float]:
    """Gibt (annualisierte Volatilitaet in %, ATR) zurueck."""
    returns = hist["Close"].pct_change().dropna()
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    return float(ann_vol), compute_atr(hist)


# ---------------------------------------------------------------------------
# Fundamentaldaten & Termine
# ---------------------------------------------------------------------------

def fundamental_score(info: dict) -> tuple[float, list[str]]:
    score = 0
    notes = []

    pe = info.get("trailingPE")
    fwd_pe = info.get("forwardPE")
    if pe:
        notes.append(f"KGV: {pe:.1f}")
        if pe < 25:
            score += 10
        elif pe < 40:
            score += 5
    if fwd_pe:
        notes.append(f"Forward-KGV: {fwd_pe:.1f}")
        if pe and fwd_pe < pe:
            score += 5
            notes.append("Forward-KGV < KGV (erwartetes Gewinnwachstum)")

    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        notes.append(f"Umsatzwachstum YoY: {rev_growth*100:+.1f}%")
        if rev_growth > 0.1:
            score += 10
        elif rev_growth > 0:
            score += 5

    profit_margin = info.get("profitMargins")
    if profit_margin is not None:
        notes.append(f"Gewinnmarge: {profit_margin*100:.1f}%")
        if profit_margin > 0.1:
            score += 5

    target = info.get("targetMeanPrice")
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if target and price:
        upside = (target / price - 1) * 100
        notes.append(f"Analysten-Kursziel: {upside:+.1f}% Upside")
        if upside > 10:
            score += 10
        elif upside > 0:
            score += 5

    return score, notes or ["keine Fundamentaldaten verfuegbar"]


def next_earnings_date(ticker_obj: "yf.Ticker") -> dt.date | None:
    try:
        cal = ticker_obj.calendar
    except Exception:
        return None

    raw = None
    if isinstance(cal, dict):
        dates = cal.get("Earnings Date")
        if dates:
            raw = dates[0] if isinstance(dates, (list, tuple)) else dates
    elif isinstance(cal, pd.DataFrame) and not cal.empty:
        raw = cal.iloc[0, 0]

    if raw is None:
        return None
    try:
        return pd.Timestamp(raw).date()
    except (ValueError, TypeError):
        return None


def earnings_info(earnings_date: dt.date | None, today: dt.date | None = None) -> tuple[str, bool]:
    """Gibt (Text, Warnung?) zurueck. Warnung, wenn Earnings in <= EARNINGS_WARN_DAYS Tagen."""
    if earnings_date is None:
        return "Earnings-Termin unbekannt - manuell pruefen!", False
    today = today or dt.date.today()
    days = (earnings_date - today).days
    if 0 <= days <= EARNINGS_WARN_DAYS:
        return f"ACHTUNG: Earnings in {days} Tagen ({earnings_date})", True
    return f"Naechste Earnings: {earnings_date}", False


# ---------------------------------------------------------------------------
# Einordnung
# ---------------------------------------------------------------------------

MARKET_STATE_LABELS = {
    "REGULAR": "🟢 offen",
    "PRE": "🟡 vorboerslich",
    "PREPRE": "🟡 vorboerslich",
    "POST": "🟡 nachboerslich",
    "POSTPOST": "🟡 nachboerslich",
    "CLOSED": "🔴 geschlossen",
}


def market_status(info: dict) -> str:
    """Handelsstatus der Boerse aus yfinance-'marketState'."""
    return MARKET_STATE_LABELS.get(info.get("marketState") or "", "unbekannt")


def classify_trend(trend: float) -> str:
    if trend >= TREND_UP:
        return TREND_LABEL_UP
    if trend >= TREND_SIDEWAYS:
        return TREND_LABEL_SIDEWAYS
    return TREND_LABEL_DOWN


def classify_setup(os_score: float) -> str:
    """Setup allein aus dem Oversold-Score (Trend und Recovery sind nur Kontext)."""
    if os_score >= OVERSOLD_STRONG:
        return SETUP_STRONG_OVERSOLD
    if os_score >= OVERSOLD_HIGH:
        return SETUP_PRESSED
    if os_score < OVERSOLD_LOW:
        return SETUP_NOT_CHEAP
    return SETUP_NEUTRAL


def sort_key(result: dict) -> tuple[float, float]:
    """Sortierung fuer Zusammenfassungen: Oversold, bei Gleichstand Recovery."""
    return result["oversold_score"], result["recovery_score"]


def sort_results(results: list[dict]) -> list[dict]:
    valid = [r for r in results if "error" not in r]
    return sorted(valid, key=sort_key, reverse=True)


# ---------------------------------------------------------------------------
# Daten laden & Hauptanalyse pro Ticker
# ---------------------------------------------------------------------------

def fetch_histories(symbols: list[str], period: str = HISTORY_PERIOD) -> dict[str, pd.DataFrame]:
    """Laedt die Kurshistorie aller Ticker mit einem einzigen Request."""
    data = yf.download(
        symbols, period=period, auto_adjust=True, group_by="ticker",
        progress=False, threads=True,
    )
    histories = {}
    for sym in symbols:
        if isinstance(data.columns, pd.MultiIndex):
            df = data[sym] if sym in data.columns.get_level_values(0) else pd.DataFrame()
        else:
            df = data
        histories[sym] = df.dropna(how="all")
    return histories


def analyze_ticker(symbol: str, hist: pd.DataFrame | None = None) -> dict:
    t = yf.Ticker(symbol)
    if hist is None:
        hist = t.history(period=HISTORY_PERIOD, auto_adjust=True)

    if hist.empty or len(hist) < MIN_HISTORY_DAYS:
        return {"symbol": symbol, "error": "Keine oder zu wenig Kursdaten gefunden"}

    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass

    trend, trend_notes = trend_score(hist)
    os_score, os_notes = oversold_score(hist)
    rec_score, rec_notes = recovery_score(hist)
    f_score, f_notes = fundamental_score(info)
    ann_vol, atr = volatility_info(hist)
    earnings_date = next_earnings_date(t)
    events, earnings_warning = earnings_info(earnings_date)
    trend_label = classify_trend(trend)
    setup = classify_setup(os_score)

    return {
        "symbol": symbol,
        "name": info.get("longName") or info.get("shortName") or symbol,
        "currency": info.get("currency") or "",
        "price": float(hist["Close"].iloc[-1]),
        "market_status": market_status(info),
        "change_pct": (float(hist["Close"].iloc[-1]) / float(hist["Close"].iloc[-2]) - 1) * 100,
        "trend_score": trend,
        "oversold_score": os_score,
        "recovery_score": rec_score,
        "fundamental_score": f_score,
        "trend_label": trend_label,
        "setup": setup,
        "verdict": f"Trend: {trend_label} | Setup: {setup}",
        "trend_notes": trend_notes,
        "oversold_notes": os_notes,
        "recovery_notes": rec_notes,
        "fundamental_notes": f_notes,
        "volatility": f"{ann_vol:.0f}% p.a. annualisiert, ATR14: {atr:.2f}",
        "earnings_date": earnings_date,
        "earnings_warning": earnings_warning,
        "events": events,
        "hist": hist,
    }


def analyze_many(symbols: list[str]) -> list[dict]:
    """Analysiert mehrere Ticker; Fehler pro Ticker landen im Ergebnis statt abzubrechen."""
    try:
        histories = fetch_histories(symbols)
    except Exception:
        histories = {}  # Fallback: jeder Ticker laedt einzeln

    results = []
    for sym in symbols:
        try:
            results.append(analyze_ticker(sym, histories.get(sym)))
        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})
    return results


def quote_from_info(info: dict) -> dict | None:
    """Aktueller Kurs, Tagesaenderung, Marktstatus und Kurszeit aus yfinance-'info'."""
    price = info.get("regularMarketPrice")
    prev_close = info.get("regularMarketPreviousClose")
    if not price or not prev_close:
        return None
    ts = info.get("regularMarketTime")
    return {
        "price": float(price),
        "change_pct": (float(price) / float(prev_close) - 1) * 100,
        "market_status": market_status(info),
        "quote_time": dt.datetime.fromtimestamp(ts, dt.timezone.utc) if ts else None,
    }


def fetch_quote(symbol: str) -> dict | None:
    try:
        return quote_from_info(yf.Ticker(symbol).info or {})
    except Exception:
        return None


def fetch_quotes(symbols: list[str]) -> dict[str, dict]:
    """Laedt aktuelle Kurse parallel (leichtgewichtig, fuer kurzen Cache gedacht)."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        quotes = dict(zip(symbols, pool.map(fetch_quote, symbols)))
    return {sym: q for sym, q in quotes.items() if q}


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------

def format_price(r: dict) -> str:
    return f"{r['price']:.2f} {r['currency']}".strip()


def summary_rows(results: list[dict]) -> list[list[str]]:
    return [
        [
            r["symbol"], r["name"], format_price(r),
            f"{r['trend_score']:.0f}", f"{r['oversold_score']:.0f}",
            f"{r['recovery_score']:.0f}", f"{r['fundamental_score']:.0f}",
            r["setup"] + (" (Earnings!)" if r["earnings_warning"] else ""),
        ]
        for r in sort_results(results)
    ]


SUMMARY_HEADERS = ["Symbol", "Name", "Kurs", "Trend", "Oversold", "Recovery", "Fundamental", "Setup"]


def print_report(results: list[dict]):
    print("\n" + "=" * 78)
    print(" AKTIEN-ANALYSE: TREND / OVERSOLD / RECOVERY")
    print("=" * 78)

    for r in results:
        if "error" in r:
            print(f"\n### {r['symbol']} -> FEHLER: {r['error']}")
            continue

        print(f"\n### {r['symbol']} ({r['name']})  |  Kurs: {format_price(r)}")
        print(f"  Trend-Score:     {r['trend_score']:.0f}/{MAX_SCORE}  ->  {'; '.join(r['trend_notes'])}")
        print(f"  Oversold-Score:  {r['oversold_score']:.0f}/{MAX_SCORE}  ->  {'; '.join(r['oversold_notes'])}")
        print(f"  Recovery-Score:  {r['recovery_score']:.0f}/{MAX_SCORE}  ->  {'; '.join(r['recovery_notes'])}")
        print(f"  Fundamental:     {r['fundamental_score']:.0f}/{MAX_FUNDAMENTAL_SCORE}  ->  "
              f"{'; '.join(r['fundamental_notes'])}")
        print(f"  Volatilitaet:    {r['volatility']}")
        print(f"  Termine:         {r['events']}")
        print(f"  ==> {r['verdict']}")

    print("\n" + "-" * 78)
    print(" ZUSAMMENFASSUNG (sortiert nach Oversold, dann Recovery)")
    print("-" * 78)
    print(tabulate(summary_rows(results), headers=SUMMARY_HEADERS, tablefmt="github"))
    print()


def format_markdown(results: list[dict]) -> str:
    """Zusammenfassung als Markdown (z.B. fuer $GITHUB_STEP_SUMMARY)."""
    lines = [
        f"## Stock Scorer - {dt.date.today():%d.%m.%Y}",
        "",
        "| " + " | ".join(SUMMARY_HEADERS) + " |",
        "|" + "---|" * len(SUMMARY_HEADERS),
    ]
    lines += ["| " + " | ".join(row) + " |" for row in summary_rows(results)]
    errors = [r for r in results if "error" in r]
    if errors:
        lines += ["", "**Fehler:**", ""]
        lines += [f"- {r['symbol']}: {r['error']}" for r in errors]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prueft Aktien auf Trend, Ueberverkauft-Zustand und Erholungsanzeichen.\n\n"
            f"SCORES (je 0-{MAX_SCORE}, Fundamental 0-{MAX_FUNDAMENTAL_SCORE}):\n"
            "  Trend:    Richtung/Konsistenz (SMA-Lage/-Steigung, Regression 20T)\n"
            "  Oversold: wie stark gedrueckt (RSI, 52W-Range, SMA-Abstand, Bollinger)\n"
            "  Recovery: erste Wende-Signale (Rebound vom 10T-Tief in ATR, SMA10,\n"
            "            MACD-Cross, steigende Tiefs, Volumen)\n\n"
            "ZEITSPANNEN (Handelstage):\n"
            "  Trend ~ 20-200T (1-10 Monate) | Oversold ~ 14-252T (2W-1J) | "
            "Recovery ~ 5-20T (kurzfristig)\n\n"
            "SETUP (allein aus dem Oversold-Score):\n"
            f"  >= {OVERSOLD_STRONG}  Stark ueberverkauft  im Backtest einziges stabiles Signal\n"
            "                              (Ueberrendite nach 20T, auch im Testzeitraum)\n"
            f"  >= {OVERSOLD_HIGH}  Gedrueckt            kein belegter Vorteil (nur Beobachtung)\n"
            f"  <  {OVERSOLD_LOW}  Nicht guenstig       kein Rabatt\n"
            "  sonst  Neutral\n\n"
            "Trend und Recovery sind Kontext: Im Backtest haben sie das Setup nicht\n"
            "verbessert. Ein Abwaertstrend war bei gedrueckten Titeln NICHT schlechter.\n"
            "Details und Grenzen (u.a. Survivorship-Bias): README.md\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("symbols", nargs="+", help="Ticker-Symbole, z.B. AAPL MSFT NVDA")
    parser.add_argument("--markdown", metavar="DATEI",
                        help="Zusammenfassung zusaetzlich als Markdown an DATEI anhaengen")
    args = parser.parse_args()

    # yfinance/pandas melden haeufig FutureWarnings, die fuer die Ausgabe irrelevant sind
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Umlaute (z.B. "Nestle S.A.") auch bei umgeleiteter Ausgabe (| tee, > datei) korrekt
    sys.stdout.reconfigure(encoding="utf-8")

    symbols = list(dict.fromkeys(s.upper() for s in args.symbols))
    print(f"Lade Daten fuer {', '.join(symbols)} ...")
    results = analyze_many(symbols)

    print_report(results)
    if args.markdown:
        with open(args.markdown, "a", encoding="utf-8") as f:
            f.write(format_markdown(results))


if __name__ == "__main__":
    main()
