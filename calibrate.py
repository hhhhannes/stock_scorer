#!/usr/bin/env python3
"""
calibrate.py
============
Kalibriert und prueft die Setup-Schwellen des Stock Scorers anhand der
Einzelergebnisse von backtest.py (CSV mit Roh-Scores und Ueberrenditen).

Vorgehen:
  1. Zeitlicher Train/Test-Split (Schwellen werden NUR auf Train gesucht).
  2. Aussagekraft je Score: Ueberrendite nach Score-Klassen.
  3. Suche der Schwelle fuer 'Stark ueberverkauft' (OVERSOLD_STRONG).
     Zielgroesse: t-Wert der mittleren Ueberrendite, gemittelt mit den Nachbarn
     im Raster (bevorzugt stabile Plateaus statt Zufallsspitzen), mit
     Mindesthaeufigkeit.
  4. Kontext-Check: Verbessern Trend oder Recovery das Setup? (Train und Test)
  5. Pruefung aller Setups mit alten und neuen Schwellen auf Test und Urteil,
     ob der Vorschlag uebernommen werden sollte.

Benutzung:
    python backtest.py --universe universe.txt --years 10 --csv backtest.csv
    python calibrate.py backtest.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from itertools import product

import numpy as np
import pandas as pd

import stock_scorer as sc

TARGET = "excess_20d"
MIN_TEST_T = 1.0   # 'Stark ueberverkauft' muss auf Test mindestens diesen t-Wert erreichen
OVERSOLD_STRONG_GRID = [45, 50, 55, 60, 65, 70]


def t_stat(values: pd.Series) -> float:
    """t-Wert der mittleren Ueberrendite. Hinweis: ueberlappende 20T-Horizonte machen
    Beobachtungen abhaengig; der Wert dient nur zum Vergleichen, nicht als Signifikanztest."""
    n = len(values)
    if n < 2 or values.std() == 0:
        return np.nan
    return float(values.mean() / (values.std() / np.sqrt(n)))


def classify(df: pd.DataFrame, o_strong: float, o_high: float, o_low: float) -> pd.Series:
    """Vektorisierte Version von stock_scorer.classify_setup mit frei waehlbaren Schwellen."""
    return pd.Series(np.select(
        [df["oversold"] >= o_strong, df["oversold"] >= o_high, df["oversold"] < o_low],
        [sc.SETUP_STRONG_OVERSOLD, sc.SETUP_PRESSED, sc.SETUP_NOT_CHEAP],
        default=sc.SETUP_NEUTRAL,
    ), index=df.index)


def setup_table(df: pd.DataFrame, setups: pd.Series) -> pd.DataFrame:
    g = df.groupby(setups)
    table = pd.DataFrame({
        "Anzahl": g.size(),
        "Anteil %": g.size() / len(df) * 100,
        "Ueber 5T %": g["excess_5d"].mean(),
        "Ueber 20T %": g[TARGET].mean(),
        "Treffer 20T %": g[TARGET].apply(lambda s: (s > 0).mean() * 100),
        "t 20T": g[TARGET].apply(t_stat),
    })
    return table.sort_values("Anzahl", ascending=False).round(2)


def score_buckets(df: pd.DataFrame, col: str) -> pd.DataFrame:
    bins = [-np.inf, 0, 15, 30, 45, 60, np.inf]
    labels = ["<=0", "1-15", "16-30", "31-45", "46-60", ">60"]
    buckets = pd.cut(df[col], bins=bins, labels=labels)
    g = df.groupby(buckets, observed=True)[TARGET]
    return pd.DataFrame({"Anzahl": g.size(), "Ueber 20T %": g.mean(),
                         "t 20T": g.apply(t_stat)}).round(2)


def neighbor_mean(grid: np.ndarray) -> np.ndarray:
    """Mittelwert jeder Zelle mit ihren direkten Nachbarn (NaN wird ignoriert)."""
    padded = np.pad(grid, 1, constant_values=np.nan)
    stacks = []
    for offs in product((-1, 0, 1), repeat=grid.ndim):
        sl = tuple(slice(1 + o, 1 + o + n) for o, n in zip(offs, grid.shape))
        stacks.append(padded[sl])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # Zellen ganz ohne Nachbarn -> NaN
        return np.nanmean(np.stack(stacks), axis=0)


def search_strong(train: pd.DataFrame, min_share: float) -> tuple[int | None, pd.DataFrame]:
    """Sucht OVERSOLD_STRONG auf Train. Gibt None zurueck, wenn keine Schwelle genug
    Stichproben hat."""
    min_n = max(int(len(train) * min_share), 30)
    records = []
    for o in OVERSOLD_STRONG_GRID:
        vals = train.loc[train["oversold"] >= o, TARGET]
        records.append({"OVERSOLD_STRONG": o, "n": len(vals),
                        "Ueber 20T %": vals.mean() if len(vals) else np.nan,
                        "t": t_stat(vals) if len(vals) >= min_n else np.nan})
    table = pd.DataFrame(records)
    smooth = neighbor_mean(table["t"].to_numpy())
    smooth[table["t"].isna().to_numpy()] = np.nan  # nur Schwellen mit genug Stichproben
    table["t geglaettet"] = smooth
    if np.all(np.isnan(smooth)):
        return None, table.round(2)
    best = int(table.loc[np.nanargmax(smooth), "OVERSOLD_STRONG"])
    return best, table.round(2)


def context_check(df: pd.DataFrame, o_strong: float) -> pd.DataFrame:
    """Ueberrendite der stark ueberverkauften Titel, aufgeteilt nach Trend und Recovery."""
    strong = df[df["oversold"] >= o_strong]
    trend = strong["trend"].map(sc.classify_trend)
    recovery = pd.cut(strong["recovery"], [-np.inf, 15, 35, np.inf],
                      labels=["Recovery <15", "Recovery 15-34", "Recovery >=35"])
    parts = []
    for name, groups in (("Trend", trend), ("Recovery", recovery)):
        g = strong.groupby(groups, observed=True)[TARGET]
        part = pd.DataFrame({"n": g.size(), "Ueber 20T %": g.mean()})
        part.index = [f"{name}: {i}" for i in part.index]
        parts.append(part)
    return pd.concat(parts).round(2)


def main():
    parser = argparse.ArgumentParser(description="Kalibriert die Setup-Schwellen")
    parser.add_argument("csv", help="CSV aus backtest.py --csv")
    parser.add_argument("--test-share", type=float, default=0.3,
                        help="Anteil der juengsten Stichtage fuer den Test (Standard: 0.3)")
    parser.add_argument("--min-share", type=float, default=0.01,
                        help="Mindesthaeufigkeit fuer 'Stark ueberverkauft' auf Train "
                             "(Standard: 0.01 = 1%%)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    df = pd.read_csv(args.csv)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    split = df["date"].quantile(1 - args.test_share)
    train, test = df[df["date"] < split], df[df["date"] >= split]
    print(f"{len(df)} Stichtage, {df['symbol'].nunique()} Titel, "
          f"{df['date'].min():%Y-%m} bis {df['date'].max():%Y-%m}")
    print(f"Train: {len(train)} (bis {split:%Y-%m-%d}) | Test: {len(test)} (ab {split:%Y-%m-%d})")

    print("\n=== 1) Aussagekraft je Score (Ueberrendite 20T nach Score-Klasse) ===")
    for col in ("trend", "oversold", "recovery"):
        print(f"\n--- {col} ---  (links Train, rechts Test)")
        print(pd.concat([score_buckets(train, col), score_buckets(test, col)], axis=1,
                        keys=["Train", "Test"]))

    print("\n=== 2) Schwelle fuer 'Stark ueberverkauft' (Train) ===")
    o_strong, strong_table = search_strong(train, args.min_share)
    print(strong_table.to_string(index=False))
    if o_strong is None:
        print("\n!!! Keine Schwelle mit genug Stichproben - aktuelle Werte beibehalten.")
        return

    print(f"\n=== 3) Kontext bei Oversold >= {o_strong}: helfen Trend/Recovery? ===")
    print(pd.concat([context_check(train, o_strong), context_check(test, o_strong)], axis=1,
                    keys=["Train", "Test"]))

    old = dict(o_strong=sc.OVERSOLD_STRONG, o_high=sc.OVERSOLD_HIGH, o_low=sc.OVERSOLD_LOW)
    new = dict(old, o_strong=o_strong, o_high=min(sc.OVERSOLD_HIGH, o_strong))

    print("\n=== 4) Setups auf TEST (nicht fuer die Suche verwendet) ===")
    for label, params in (("AKTUELLE Schwellen", old), ("NEUE Schwellen", new)):
        print(f"\n--- {label}: {params} ---")
        print(setup_table(test, classify(test, **params)))

    strong_test = test.loc[test["oversold"] >= o_strong, TARGET]
    test_t = t_stat(strong_test)
    at_edge = o_strong in (OVERSOLD_STRONG_GRID[0], OVERSOLD_STRONG_GRID[-1])
    confirmed = not np.isnan(test_t) and test_t >= MIN_TEST_T
    if confirmed:
        print(f"\nBESTAETIGT: 'Stark ueberverkauft' hat auf Test t = {test_t:.2f} "
              f"(Ueberrendite {strong_test.mean():+.2f}%, n={len(strong_test)}).")
    else:
        print(f"\n!!! NICHT BESTAETIGT: 'Stark ueberverkauft' hat auf Test t = {test_t:.2f} "
              f"(Ueberrendite {strong_test.mean():+.2f}%, n={len(strong_test)}), "
              f"verlangt t >= {MIN_TEST_T}.\n!!! Die Schwellen unten NICHT uebernehmen.")
    if at_edge:
        print("!!! Optimum liegt am Rand des Suchrasters - Ergebnis unzuverlaessig.")

    print("\n=== Vorschlag fuer stock_scorer.py ===")
    print(f"OVERSOLD_STRONG = {new['o_strong']}")
    print(f"OVERSOLD_HIGH = {new['o_high']}")
    print(f"OVERSOLD_LOW = {new['o_low']}")


if __name__ == "__main__":
    main()
