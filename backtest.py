#!/usr/bin/env python3
"""
backtest.py
===========
Prueft, ob die Setups des Stock Scorers historisch zu besseren Folge-Renditen
gefuehrt haben als der Durchschnitt aller Tage.

Fuer jeden Stichtag (alle --step Handelstage) werden die technischen Scores auf
Basis der letzten 252 Handelstage berechnet und die Rendite nach 5/10/20 Tagen
gemessen. Ausgewertet wird pro Setup: Anzahl, Rendite, Ueberrendite, Trefferquote
und ein robuster t-Wert (siehe robust_t).

Einschraenkungen:
  - nur technische Scores (Fundamentaldaten waeren Blick in die Zukunft)
  - Survivorship-Bias: heute existierende Ticker waren "Gewinner"
  - keine Gebuehren, Slippage oder Dividenden-Timing
  - wenige Ticker/Jahre -> kleine Stichproben, Ergebnisse vorsichtig lesen

Benutzung:
    python backtest.py AAPL MSFT NVDA GOOGL AMZN --years 5 --step 5
    python backtest.py --universe universe.txt --years 10 --csv backtest.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import stock_scorer as sc

HORIZONS = (5, 10, 20)
WINDOW = sc.TRADING_DAYS_PER_YEAR
# Ein 20-Tage-Horizont ueberlappt bei woechentlichen Stichtagen die naechsten ~4 Wochen
NW_LAGS = 4


def load_universe(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = (line.split("#")[0].strip() for line in f)
        return [line for line in lines if line]


def backtest_symbol(symbol: str, hist: pd.DataFrame, step: int) -> list[dict]:
    rows = []
    close = hist["Close"]
    for end in range(WINDOW, len(hist) - max(HORIZONS), step):
        window = hist.iloc[end - WINDOW:end]
        trend, _ = sc.trend_score(window)
        os_score, _ = sc.oversold_score(window)
        rec_score, _ = sc.recovery_score(window)
        setup = sc.classify_setup(os_score)

        last = end - 1
        row = {"symbol": symbol, "date": hist.index[last], "setup": setup,
               "trend": trend, "oversold": os_score, "recovery": rec_score}
        for h in HORIZONS:
            row[f"ret_{h}d"] = (close.iloc[last + h] / close.iloc[last] - 1) * 100
        rows.append(row)
    return rows


def _backtest_job(task: tuple[str, pd.DataFrame, int]) -> list[dict]:
    warnings.filterwarnings("ignore", category=FutureWarning)
    return backtest_symbol(*task)


def run_backtests(histories: dict[str, pd.DataFrame], step: int, jobs: int = 1) -> list[dict]:
    """Backtest aller Titel, bei jobs > 1 parallel auf mehreren Prozessen."""
    tasks = []
    for sym, hist in histories.items():
        if len(hist) < WINDOW + max(HORIZONS):
            print(f"  {sym}: zu wenig Historie, uebersprungen")
            continue
        print(f"  {sym}: {len(hist)} Handelstage")
        tasks.append((sym, hist, step))
    if jobs <= 1 or len(tasks) <= 1:
        results = list(map(_backtest_job, tasks))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_backtest_job, tasks))
    return [row for rows in results for row in rows]


def week_of(dates: pd.Series) -> pd.Series:
    """Kalenderwoche je Stichtag (Zeitzonen der Boersen werden ignoriert)."""
    return pd.to_datetime(dates, utc=True).dt.tz_localize(None).dt.to_period("W")


def add_excess_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Ueberrendite = Rendite minus Durchschnitt aller Titel in derselben Kalenderwoche.

    Entfernt den Markteffekt (z.B. Bullenmarkt), sodass nur zaehlt, ob ein Setup
    besser war als die anderen Titel zur selben Zeit.
    """
    df = df.copy()
    week = week_of(df["date"])
    for h in HORIZONS:
        col = f"ret_{h}d"
        df[f"excess_{h}d"] = df[col] - df.groupby(week)[col].transform("mean")
    return df


def robust_t(values: pd.Series, dates: pd.Series, lags: int = NW_LAGS) -> float:
    """t-Wert einer mittleren (Ueber-)Rendite, der Abhaengigkeiten zwischen Stichtagen
    beruecksichtigt:

    - Titel mit Stichtag in derselben Kalenderwoche sind korreliert (z.B. Maerz 2020)
      -> Abweichungen werden je Woche aufsummiert (Cluster).
    - Ueberlappende Horizonte korrelieren benachbarte Wochen -> Newey-West ueber
      'lags' Kalenderwochen.

    Deutlich vorsichtiger als der naive t-Wert, der alle Stichtage als unabhaengig
    behandelt. Bei einem Stichtag pro Woche und lags=0 sind beide gleich.
    """
    x = values.dropna()
    n = len(x)
    if n < 3:
        return np.nan
    mean = x.mean()
    sums = (x - mean).groupby(week_of(dates.loc[x.index])).sum()
    weeks = pd.period_range(sums.index.min(), sums.index.max(), freq="W")
    s = sums.reindex(weeks, fill_value=0.0).to_numpy(dtype=float)
    var = s @ s
    for k in range(1, min(lags, len(s) - 1) + 1):
        var += 2 * (1 - k / (lags + 1)) * (s[k:] @ s[:-k])
    if var <= 0:
        return np.nan
    return float(mean * n / np.sqrt(var))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    def stats(group: pd.DataFrame) -> pd.Series:
        s = {"Anzahl": len(group)}
        for h in HORIZONS:
            col = group[f"ret_{h}d"]
            s[f"Mittel {h}T %"] = col.mean()
            s[f"Treffer {h}T %"] = (col > 0).mean() * 100
            if f"excess_{h}d" in group:
                s[f"Ueber {h}T %"] = group[f"excess_{h}d"].mean()
        if "excess_20d" in group:
            s["t Ueber 20T"] = robust_t(group["excess_20d"], group["date"])
        return pd.Series(s)

    per_setup = df.groupby("setup").apply(stats, include_groups=False)
    baseline = stats(df).rename("ALLE (Vergleich)").to_frame().T
    if "t Ueber 20T" in baseline:
        baseline["t Ueber 20T"] = np.nan  # Ueberrendite aller Titel ist per Definition 0
    result = pd.concat([per_setup.sort_values("Anzahl", ascending=False), baseline])
    result["Anzahl"] = result["Anzahl"].astype(int)
    return result.round(2)


def main():
    parser = argparse.ArgumentParser(description="Backtest der Stock-Scorer-Setups")
    parser.add_argument("symbols", nargs="*", help="Ticker-Symbole, z.B. AAPL MSFT NVDA")
    parser.add_argument("--universe", metavar="DATEI",
                        help="Ticker aus Datei lesen (einer pro Zeile, # = Kommentar)")
    parser.add_argument("--years", type=int, default=5, help="Historie in Jahren (Standard: 5)")
    parser.add_argument("--step", type=int, default=5,
                        help="Abstand der Stichtage in Handelstagen (Standard: 5)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="Anzahl paralleler Prozesse (Standard: alle CPU-Kerne)")
    parser.add_argument("--csv", metavar="DATEI", help="Einzelergebnisse als CSV speichern")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    # Umlaute (z.B. "Nestle S.A.") auch bei umgeleiteter Ausgabe (| tee, > datei) korrekt
    sys.stdout.reconfigure(encoding="utf-8")

    symbols = list(args.symbols)
    if args.universe:
        symbols += load_universe(args.universe)
    symbols = list(dict.fromkeys(s.upper() for s in symbols))
    if not symbols:
        parser.error("Ticker-Symbole oder --universe angeben")
    print(f"Lade {args.years} Jahre Historie fuer {', '.join(symbols)} ...")
    histories = sc.fetch_histories(symbols, period=f"{args.years}y")

    rows = run_backtests(histories, args.step, args.jobs)
    if not rows:
        print("Keine auswertbaren Daten.")
        return

    df = add_excess_returns(pd.DataFrame(rows))
    if args.csv:
        df.to_csv(args.csv, index=False)

    print(f"\nBacktest: {len(df)} Stichtage, Horizonte {HORIZONS} Handelstage\n")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summarize(df))
    print("\nLesart: Ein Setup ist nur dann nuetzlich, wenn es klar und mit genug Stichproben"
          "\nueber der Zeile 'ALLE (Vergleich)' liegt. 'Ueber' = Ueberrendite gegenueber"
          "\nallen Titeln in derselben Woche (Markteffekt herausgerechnet)."
          "\n't Ueber 20T' beruecksichtigt gleichzeitige Signale und ueberlappende Horizonte;"
          "\nerst ab etwa 2 ist ein Effekt kaum noch mit Zufall zu erklaeren.")


if __name__ == "__main__":
    main()
