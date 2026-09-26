#!/usr/bin/env python3
"""
stock_scorer.py
================
Prueft Aktien auf drei Dimensionen:
  - Trend-Score:     wie stark/klar ist der aktuelle Kurstrend?
  - Oversold-Score:  wie weit/stark ist die Aktie gegenueber sich selbst gedrueckt?
  - Recovery-Score:  gibt es bereits Anzeichen, dass sich eine Drueckung dreht?

Installation:
    pip install yfinance pandas numpy tabulate --break-system-packages

Benutzung:
    python stock_scorer.py AAPL MSFT NVDA
"""

import sys
import argparse
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Bitte zuerst installieren: pip install yfinance pandas numpy tabulate --break-system-packages")
    sys.exit(1)

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Technische Indikatoren
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty and not np.isnan(rsi.iloc[-1]) else 50.0


def compute_atr(hist: pd.DataFrame, period: int = 14) -> float:
    high, low, close = hist["High"], hist["Low"], hist["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return float(atr.iloc[-1])


def compute_macd(close: pd.Series, fast=12, slow=26, signal=9) -> tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_bollinger(close: pd.Series, period=20, num_std=2) -> tuple[float, float, float]:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return float(lower.iloc[-1]), float(sma.iloc[-1]), float(upper.iloc[-1])


def linear_regression_slope(values: pd.Series) -> float:
    """Steigung einer linearen Regression, normalisiert auf % pro Tag relativ zum Mittelwert."""
    y = values.values
    x = np.arange(len(y))
    if len(y) < 2 or np.mean(y) == 0:
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return float(slope / np.mean(y) * 100)  # % Veraenderung pro Tag


def trend_score(hist: pd.DataFrame) -> tuple[float, str]:
    """Wie stark/klar ist der aktuelle Kurstrend (Richtung + Konsistenz)?"""
    close = hist["Close"]
    price = float(close.iloc[-1])
    score = 0
    notes = []

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean() if len(close) >= 200 else None

    # 1) Kurs vs. SMA50
    if not np.isnan(sma50.iloc[-1]):
        if price > sma50.iloc[-1]:
            score += 15
            notes.append("Kurs > SMA50")
        else:
            notes.append("Kurs < SMA50")

    # 2) Kurs vs. SMA200
    if sma200 is not None and not np.isnan(sma200.iloc[-1]):
        if price > sma200.iloc[-1]:
            score += 15
            notes.append("Kurs > SMA200")
        else:
            notes.append("Kurs < SMA200")

        # 3) SMA50 vs SMA200 (Golden/Death Cross Lage)
        if not np.isnan(sma50.iloc[-1]):
            if sma50.iloc[-1] > sma200.iloc[-1]:
                score += 15
                notes.append("SMA50 > SMA200 (bullische Lage)")
            else:
                notes.append("SMA50 < SMA200 (baerische Lage)")
    else:
        notes.append("SMA200 nicht verfuegbar (zu wenig Historie)")

    # 4) Steigung der SMA50 (letzte 10 Tage) - zeigt ob Trend an Fahrt gewinnt
    if len(sma50.dropna()) > 10:
        sma50_now = sma50.iloc[-1]
        sma50_10d = sma50.iloc[-11]
        sma50_slope_pct = (sma50_now / sma50_10d - 1) * 100
        notes.append(f"SMA50-Steigung 10T: {sma50_slope_pct:+.1f}%")
        if sma50_slope_pct > 1:
            score += 15
        elif sma50_slope_pct > 0:
            score += 7

    # 5) Trendstaerke via linearer Regression der letzten 20 Handelstage
    if len(close) >= 20:
        slope_pct = linear_regression_slope(close.iloc[-20:])
        notes.append(f"Regressions-Steigung 20T: {slope_pct:+.2f}%/Tag")
        if slope_pct > 0.15:
            score += 15
        elif slope_pct > 0:
            score += 7
        elif slope_pct < -0.15:
            score -= 5  # klarer Abwaertstrend

    direction = "Aufwaertstrend" if score >= 45 else ("Seitwaerts/uneindeutig" if score >= 20 else "Abwaertstrend")
    notes.append(f"Einordnung: {direction}")

    return score, "; ".join(notes)



    """Wie stark/weit ist die Aktie aktuell gedrueckt ('unter ihrem normalen Wert')?"""
    close = hist["Close"]
    price = float(close.iloc[-1])
    score = 0
    notes = []

    # RSI: klassischstes Ueberverkauft-Signal
    rsi = compute_rsi(close)
    notes.append(f"RSI14: {rsi:.0f}")
    if rsi < 30:
        score += 25
        notes[-1] += " (ueberverkauft)"
    elif rsi < 40:
        score += 12

    # Abstand zum 52-Wochen-Hoch/-Tief
    high_52w = float(close.rolling(min(252, len(close))).max().iloc[-1])
    low_52w = float(close.rolling(min(252, len(close))).min().iloc[-1])
    range_52w = high_52w - low_52w
    if range_52w > 0:
        pct_from_high = (price / high_52w - 1) * 100
        pos_in_range = (price - low_52w) / range_52w * 100  # 0 = am Jahrestief, 100 = am Jahreshoch
        notes.append(f"{pct_from_high:.0f}% unter 52W-Hoch, Position im 52W-Range: {pos_in_range:.0f}%")
        if pos_in_range < 20:
            score += 20
        elif pos_in_range < 35:
            score += 10

    # Abstand zu gleitenden Durchschnitten (SMA50/SMA200)
    sma50 = close.rolling(50).mean().iloc[-1]
    if not np.isnan(sma50):
        dist50 = (price / sma50 - 1) * 100
        notes.append(f"{dist50:+.1f}% vs. SMA50")
        if dist50 < -10:
            score += 15
        elif dist50 < -3:
            score += 8

    # Bollinger-Baender: Kurs nahe/unter dem unteren Band = statistisch guenstig
    if len(close) >= 20:
        lower, mid, upper = compute_bollinger(close)
        if price <= lower:
            score += 15
            notes.append("Kurs am/unter unterem Bollinger-Band")
        elif price < mid:
            score += 5

    return score, "; ".join(notes)


def oversold_score(hist: pd.DataFrame) -> tuple[float, str]:
    """Wie stark/weit ist die Aktie aktuell gedrueckt ('unter ihrem normalen Wert')?"""
    close = hist["Close"]
    price = float(close.iloc[-1])
    score = 0
    notes = []

    # RSI: klassischstes Ueberverkauft-Signal
    rsi = compute_rsi(close)
    notes.append(f"RSI14: {rsi:.0f}")
    if rsi < 30:
        score += 25
        notes[-1] += " (ueberverkauft)"
    elif rsi < 40:
        score += 12

    # Abstand zum 52-Wochen-Hoch/-Tief
    high_52w = float(close.rolling(min(252, len(close))).max().iloc[-1])
    low_52w = float(close.rolling(min(252, len(close))).min().iloc[-1])
    range_52w = high_52w - low_52w
    if range_52w > 0:
        pct_from_high = (price / high_52w - 1) * 100
        pos_in_range = (price - low_52w) / range_52w * 100  # 0 = am Jahrestief, 100 = am Jahreshoch
        notes.append(f"{pct_from_high:.0f}% unter 52W-Hoch, Position im 52W-Range: {pos_in_range:.0f}%")
        if pos_in_range < 20:
            score += 20
        elif pos_in_range < 35:
            score += 10

    # Abstand zu gleitenden Durchschnitten (SMA50/SMA200)
    sma50 = close.rolling(50).mean().iloc[-1]
    if not np.isnan(sma50):
        dist50 = (price / sma50 - 1) * 100
        notes.append(f"{dist50:+.1f}% vs. SMA50")
        if dist50 < -10:
            score += 15
        elif dist50 < -3:
            score += 8

    # Bollinger-Baender: Kurs nahe/unter dem unteren Band = statistisch guenstig
    if len(close) >= 20:
        lower, mid, upper = compute_bollinger(close)
        if price <= lower:
            score += 15
            notes.append("Kurs am/unter unterem Bollinger-Band")
        elif price < mid:
            score += 5

    return score, "; ".join(notes)


def recovery_score(hist: pd.DataFrame) -> tuple[float, str]:
    """Gibt es bereits Anzeichen, dass sich eine gedrueckte Aktie zu drehen beginnt?"""
    close = hist["Close"]
    price = float(close.iloc[-1])
    score = 0
    notes = []

    # RSI dreht von unten nach oben (Momentum kehrt zurueck)
    rsi_now = compute_rsi(close)
    rsi_5d_ago = compute_rsi(close.iloc[:-5]) if len(close) > 30 else rsi_now
    notes.append(f"RSI: {rsi_5d_ago:.0f} -> {rsi_now:.0f}")
    if rsi_now > rsi_5d_ago and rsi_5d_ago < 40:
        score += 20
        notes[-1] += " (dreht nach oben aus ueberverkaufter Zone)"

    # Kurs hat kurzfristigen gleitenden Durchschnitt (SMA10) zurueckerobert
    if len(close) >= 15:
        sma10 = close.rolling(10).mean()
        crossed_above = price > sma10.iloc[-1] and close.iloc[-6] < sma10.iloc[-6]
        notes.append(f"Kurs vs SMA10: {'ueber' if price > sma10.iloc[-1] else 'unter'}")
        if crossed_above:
            score += 15
            notes[-1] += " (frisch zurueckerobert)"

    # MACD-Line kreuzt Signal-Linie von unten nach oben (bullisches Signal)
    if len(close) >= 35:
        macd_line, signal_line = compute_macd(close)
        macd_now, sig_now = macd_line.iloc[-1], signal_line.iloc[-1]
        macd_prev, sig_prev = macd_line.iloc[-2], signal_line.iloc[-2]
        bullish_cross = macd_prev < sig_prev and macd_now > sig_now
        notes.append(f"MACD {'> ' if macd_now > sig_now else '< '}Signal")
        if bullish_cross:
            score += 20
            notes[-1] += " (frischer bullischer Crossover)"
        elif macd_now > sig_now:
            score += 8

    # Steigende Tiefs der letzten Tage (h?here Lows = beginnender Aufwaertstrend)
    if len(close) >= 10:
        recent = close.iloc[-10:]
        higher_lows = recent.iloc[-1] > recent.iloc[:5].min() and recent.iloc[-5:].min() > recent.iloc[:5].min()
        if higher_lows:
            score += 10
            notes.append("juengste Tiefs steigen")

    # Volumen bei Aufwaertstagen steigt (Kaufinteresse kehrt zurueck)
    vol = hist["Volume"]
    if len(vol) >= 10:
        up_days = hist[close.diff() > 0].tail(5)
        down_days = hist[close.diff() < 0].tail(5)
        if not up_days.empty and not down_days.empty:
            vol_ratio = up_days["Volume"].mean() / max(down_days["Volume"].mean(), 1)
            notes.append(f"Volumen an Up-Tagen/Down-Tagen: {vol_ratio:.2f}x")
            if vol_ratio > 1.3:
                score += 10

    return score, "; ".join(notes)


def volatility_info(hist: pd.DataFrame) -> tuple[float, float]:
    """Gibt (annualisierte Volatilitaet in %, ATR) zurueck."""
    returns = hist["Close"].pct_change().dropna()
    ann_vol = returns.std() * np.sqrt(252) * 100
    atr = compute_atr(hist)
    return float(ann_vol), atr


# ---------------------------------------------------------------------------
# Fundamentaldaten
# ---------------------------------------------------------------------------

def fundamental_score(info: dict) -> tuple[float, str]:
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
        if fwd_pe and pe and fwd_pe < pe:
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

    return score, "; ".join(notes) if notes else "keine Fundamentaldaten verfuegbar"


def upcoming_events(ticker_obj: "yf.Ticker") -> str:
    try:
        cal = ticker_obj.calendar
        if isinstance(cal, dict) and cal.get("Earnings Date"):
            dates = cal["Earnings Date"]
            d = dates[0] if isinstance(dates, list) else dates
            return f"Naechste Earnings: {d}"
        if hasattr(cal, "empty") and not cal.empty:
            return f"Naechste Earnings: {cal.iloc[0, 0]}"
    except Exception:
        pass
    return "Earnings-Termin unbekannt - manuell pruefen!"


# ---------------------------------------------------------------------------
# Hauptanalyse pro Ticker
# ---------------------------------------------------------------------------

def analyze_ticker(symbol: str) -> dict:
    t = yf.Ticker(symbol)
    hist = t.history(period="1y", auto_adjust=True)

    if hist.empty or len(hist) < 30:
        return {"symbol": symbol, "error": "Keine oder zu wenig Kursdaten gefunden"}

    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass

    name = info.get("longName") or info.get("shortName") or symbol
    price = float(hist["Close"].iloc[-1])

    trend, trend_notes = trend_score(hist)
    os_score, os_notes = oversold_score(hist)
    rec_score, rec_notes = recovery_score(hist)
    f_score, f_notes = fundamental_score(info)
    ann_vol, atr = volatility_info(hist)
    events = upcoming_events(t)

    if os_score >= 45 and rec_score >= 35:
        setup = "Pullback+Wende" if trend >= 45 else ("Reversal-Versuch" if trend < 20 else "Erholung")
    elif os_score >= 45:
        setup = "Fallendes Messer" if trend < 20 else "Gedrueckt, keine Wende"
    elif os_score < 20:
        setup = "Nicht guenstig"
    else:
        setup = "Neutral"

    trend_label = "Aufwaerts" if trend >= 45 else ("Seitwaerts" if trend >= 20 else "Abwaerts")
    verdict = f"Trend: {trend_label} | Setup: {setup}"

    return {
        "symbol": symbol,
        "name": name,
        "price": price,
        "trend_score": trend,
        "oversold_score": os_score,
        "recovery_score": rec_score,
        "verdict": verdict,
        "trend_notes": trend_notes,
        "oversold_notes": os_notes,
        "recovery_notes": rec_notes,
        "fundamentals": f_notes,
        "volatility": f"{ann_vol:.0f}% p.a. annualisiert, ATR14: {atr:.2f}",
        "events": events,
        "hist": hist,
    }


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------

def print_report(results: list[dict]):
    print("\n" + "=" * 78)
    print(" AKTIEN-ANALYSE: TREND / OVERSOLD / RECOVERY")
    print("=" * 78)

    summary_rows = []
    for r in results:
        if "error" in r:
            print(f"\n### {r['symbol']} -> FEHLER: {r['error']}")
            continue

        print(f"\n### {r['symbol']} ({r['name']})  |  Kurs: {r['price']:.2f}")
        print(f"  Trend-Score:     {r['trend_score']:.0f}/75  ->  {r['trend_notes']}")
        print(f"  Oversold-Score:  {r['oversold_score']:.0f}/75  ->  {r['oversold_notes']}")
        print(f"  Recovery-Score:  {r['recovery_score']:.0f}/75  ->  {r['recovery_notes']}")
        print(f"  Fundamental:     {r['fundamentals']}")
        print(f"  Volatilitaet:    {r['volatility']}")
        print(f"  Termine:         {r['events']}")
        print(f"  ==> {r['verdict']}")

        summary_rows.append([
            r["symbol"], r["name"], f"{r['price']:.2f}",
            f"{r['trend_score']:.0f}", f"{r['oversold_score']:.0f}", f"{r['recovery_score']:.0f}",
            r["verdict"]
        ])

    print("\n" + "-" * 78)
    print(" ZUSAMMENFASSUNG (sortiert nach Oversold+Recovery kombiniert)")
    print("-" * 78)
    summary_rows.sort(key=lambda row: float(row[4]) + float(row[5]), reverse=True)
    headers = ["Symbol", "Name", "Kurs", "Trend", "Oversold", "Recovery", "Einschaetzung"]
    if HAS_TABULATE:
        print(tabulate(summary_rows, headers=headers, tablefmt="github"))
    else:
        print(headers)
        for row in summary_rows:
            print(row)
    print()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prueft Aktien auf Trend, Ueberverkauft-Zustand und Erholungsanzeichen.\n\n"
            "SCORES (je 0-75):\n"
            "  Trend:    Richtung/Konsistenz (SMA-Lage/-Steigung, Regression 20T)\n"
            "  Oversold: wie stark gedrueckt (RSI, 52W-Range, SMA-Abstand, Bollinger)\n"
            "  Recovery: erste Wende-Signale (RSI-Dreh, SMA10, MACD-Cross, Volumen)\n\n"
            "ZEITSPANNEN (Handelstage):\n"
            "  Trend ~ 20-200T (1-10 Monate) | Oversold ~ 14-252T (2W-1J) | "
            "Recovery ~ 5-10T (kurzfristig)\n\n"
            "SETUP-INTERPRETATION (Trend | Oversold | Recovery -> Lesart):\n"
            "  Hoch    | Hoch | Hoch     -> Pullback im Aufwaertstrend, Wende sichtbar\n"
            "                               (am haeufigsten gesuchtes Setup)\n"
            "  Hoch    | Niedrig | -      -> laeuft bereits, kein Rabatt mehr;\n"
            "                               manche kaufen trotzdem (Trend folgen)\n"
            "  Niedrig | Hoch | Hoch      -> Reversal-Versuch gegen den Haupttrend,\n"
            "                               riskanter (Wette gegen die Hauptrichtung)\n"
            "  Niedrig | Hoch | Niedrig   -> 'Fallendes Messer' - oft am gefaehrlichsten:\n"
            "                               gedrueckt, keine Wende bestaetigt, Trend faellt weiter\n"
            "  Niedrig | Niedrig | -      -> weder guenstig noch in Bewegung,\n"
            "                               meist wenig Grund zum Handeln\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("symbols", nargs="+", help="Ticker-Symbole, z.B. AAPL MSFT NVDA")
    args = parser.parse_args()

    results = []
    for sym in args.symbols:
        print(f"Lade Daten fuer {sym} ...")
        try:
            results.append(analyze_ticker(sym.upper()))
        except Exception as e:
            results.append({"symbol": sym.upper(), "error": str(e)})

    print_report(results)


if __name__ == "__main__":
    main()
