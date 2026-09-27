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
     Zielgroesse: robuster t-Wert der mittleren Ueberrendite (backtest.robust_t),
     gemittelt mit den Nachbarn im Raster (bevorzugt stabile Plateaus statt
     Zufallsspitzen), mit Mindesthaeufigkeit.
  4. Kontext-Check: Verbessern Trend oder Recovery das Setup? (Train und Test)
  5. Stabilitaet: Ueberrendite von 'Stark ueberverkauft' je Kalenderjahr.
  6. Signale wie im Screener: nur der Beginn einer Phase, mit Sperrfrist.
  7. Pruefung der Setups mit alten und neuen Schwellen auf Test und Urteil,
     ob der Vorschlag uebernommen werden sollte.

Alle t-Werte sind robust: Titel mit gleichzeitigem Signal zaehlen zusammen nur
einmal pro Woche, ueberlappende 20-Tage-Horizonte werden per Newey-West
beruecksichtigt. Sie fallen deshalb deutlich kleiner aus als naive t-Werte.

Benutzung:
    python backtest.py --universe universe.txt --years 10 --csv backtest.csv
    python calibrate.py backtest.csv
    python calibrate.py backtest.csv --split-date 2026-09-26   # nur neue Daten als Test
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import warnings
from itertools import product

import numpy as np
import pandas as pd

import stock_scorer as sc
from backtest import robust_t
from screener import COOLDOWN_DAYS

TARGET = "excess_20d"
MIN_TEST_T = 1.0   # 'Stark ueberverkauft' muss auf Test mindestens diesen t-Wert erreichen
OVERSOLD_STRONG_GRID = [40, 45, 50, 55, 60, 65, 70]
FLAT_T_SPREAD = 0.5  # geglaettete t-Werte liegen enger beieinander -> Wahl ist Zufall


def classify(df: pd.DataFrame, o_strong: float, o_high: float, o_low: float) -> pd.Series:
    """Vektorisierte Version von stock_scorer.classify_setup mit frei waehlbaren Schwellen."""
    return pd.Series(np.select(
        [df["oversold"] >= o_strong, df["oversold"] >= o_high, df["oversold"] < o_low],
        [sc.SETUP_STRONG_OVERSOLD, sc.SETUP_PRESSED, sc.SETUP_NOT_CHEAP],
        default=sc.SETUP_NEUTRAL,
    ), index=df.index)


def group_stats(df: pd.DataFrame, groups) -> pd.DataFrame:
    """Anzahl, mittlere Ueberrendite 20T, Trefferquote und robuster t-Wert je Gruppe."""
    g = df.groupby(groups, observed=True)[TARGET]
    return pd.DataFrame({
        "n": g.size(),
        "Ueber 20T %": g.mean(),
        "Treffer %": g.apply(lambda s: (s > 0).mean() * 100),
        "t": g.apply(lambda s: robust_t(s, df["date"])),
    })


def setup_table(df: pd.DataFrame, setups: pd.Series) -> pd.DataFrame:
    table = group_stats(df, setups)
    table.insert(1, "Anteil %", table["n"] / len(df) * 100)
    table.insert(2, "Ueber 5T %", df.groupby(setups)["excess_5d"].mean())
    return table.sort_values("n", ascending=False).round(2)


def score_buckets(df: pd.DataFrame, col: str) -> pd.DataFrame:
    bins = [-np.inf, 0, 15, 30, 45, 60, np.inf]
    labels = ["<=0", "1-15", "16-30", "31-45", "46-60", ">60"]
    buckets = pd.cut(df[col], bins=bins, labels=labels)
    return group_stats(df, buckets).drop(columns="Treffer %").round(2)


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
                        "t": robust_t(vals, train["date"]) if len(vals) >= min_n else np.nan})
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
        part = group_stats(strong, groups).drop(columns="Treffer %")
        part.index = [f"{name}: {i}" for i in part.index]
        parts.append(part)
    return pd.concat(parts).round(2)


def yearly(df: pd.DataFrame, o_strong: float) -> pd.DataFrame:
    """Ueberrendite von 'Stark ueberverkauft' je Kalenderjahr: traegt das Signal
    ueber die Zeit oder nur in einzelnen Phasen (z.B. Maerz 2020)?"""
    strong = df[df["oversold"] >= o_strong]
    return group_stats(strong, strong["date"].dt.year.rename("Jahr")).round(2)


def signal_starts(df: pd.DataFrame, o_strong: float,
                  cooldown_days: int = COOLDOWN_DAYS) -> pd.Series:
    """Markiert Stichtage, die im Screener ein Signal waeren: erster Stichtag einer Phase
    Oversold >= o_strong, danach pro Titel 'cooldown_days' Kalendertage Sperrfrist.

    Ohne diese Filterung zaehlt eine lange Phase mehrfach und gleicht eher einem
    'Halten solange ueberverkauft' als dem tatsaechlichen Signal.
    """
    starts = pd.Series(False, index=df.index)
    cooldown = dt.timedelta(days=cooldown_days)
    for _, g in df.sort_values("date").groupby("symbol"):
        strong = g["oversold"] >= o_strong
        begins = strong & ~strong.shift(1, fill_value=False)
        last = None
        for idx, date in g.loc[begins, "date"].items():
            if last is None or date - last > cooldown:
                starts[idx] = True
                last = date
    return starts


def signal_table(parts: dict[str, pd.DataFrame], params: dict[str, float]) -> pd.DataFrame:
    rows = {}
    for label, o_strong in params.items():
        for part, df in parts.items():
            sig = df[signal_starts(df, o_strong)]
            vals = sig[TARGET]
            rows[(label, part)] = {"n": len(vals), "Ueber 20T %": vals.mean(),
                                   "Treffer %": (vals > 0).mean() * 100,
                                   "t": robust_t(vals, sig["date"])}
    return pd.DataFrame(rows).T.round(2)


def verdict(test: pd.DataFrame, current: int, proposed: int) -> bool:
    """Druckt das Urteil und gibt True zurueck, wenn der Vorschlag uebernommen werden sollte."""
    def test_stats(o):
        vals = test.loc[test["oversold"] >= o, TARGET]
        return vals, robust_t(vals, test["date"])

    new_vals, new_t = test_stats(proposed)
    cur_vals, cur_t = test_stats(current)
    print(f"Test, Oversold >= {proposed} (Vorschlag): Ueberrendite {new_vals.mean():+.2f}%, "
          f"t = {new_t:.2f}, n = {len(new_vals)}")
    if proposed != current:
        print(f"Test, Oversold >= {current} (aktuell):   Ueberrendite {cur_vals.mean():+.2f}%, "
              f"t = {cur_t:.2f}, n = {len(cur_vals)}")

    confirmed = not np.isnan(new_t) and new_t >= MIN_TEST_T
    at_edge = proposed in (OVERSOLD_STRONG_GRID[0], OVERSOLD_STRONG_GRID[-1])
    if not confirmed:
        print(f"\n!!! NICHT BESTAETIGT: Vorschlag erreicht auf Test nicht t >= {MIN_TEST_T}.")
    elif at_edge:
        print("\n!!! Optimum liegt am Rand des Suchrasters - Ergebnis unzuverlaessig.")
    elif proposed == current:
        print(f"\nBESTAETIGT: die aktuelle Schwelle {current} ist auf Train die beste "
              f"und haelt auf Test.")
        return False
    elif not np.isnan(cur_t) and cur_t > new_t:
        print("\n!!! Vorschlag ist auf Test schwaecher als die aktuelle Schwelle.")
    else:
        print("\nBESTAETIGT: Vorschlag ist auf Test mindestens so gut wie die aktuelle Schwelle.")
        return True
    print("!!! Aktuelle Werte beibehalten.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Kalibriert die Setup-Schwellen")
    parser.add_argument("csv", help="CSV aus backtest.py --csv")
    parser.add_argument("--test-share", type=float, default=0.3,
                        help="Anteil der juengsten Stichtage fuer den Test (Standard: 0.3)")
    parser.add_argument("--split-date", metavar="JJJJ-MM-TT",
                        help="Test = alle Stichtage ab diesem Datum (statt --test-share), "
                             "z.B. das Datum der letzten Kalibrierung")
    parser.add_argument("--min-share", type=float, default=0.01,
                        help="Mindesthaeufigkeit fuer 'Stark ueberverkauft' auf Train "
                             "(Standard: 0.01 = 1%%)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    df = pd.read_csv(args.csv)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    if args.split_date:
        split = pd.Timestamp(args.split_date, tz="UTC")
    else:
        split = df["date"].quantile(1 - args.test_share)
    train, test = df[df["date"] < split], df[df["date"] >= split]
    print(f"{len(df)} Stichtage, {df['symbol'].nunique()} Titel, "
          f"{df['date'].min():%Y-%m} bis {df['date'].max():%Y-%m}")
    print(f"Train: {len(train)} (bis {split:%Y-%m-%d}) | Test: {len(test)} (ab {split:%Y-%m-%d})")
    if train.empty or test.empty:
        print("\n!!! Train oder Test ist leer - Split anpassen.")
        return

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
    smooth = strong_table["t geglaettet"].dropna()
    if smooth.max() - smooth.min() < FLAT_T_SPREAD:
        print(f"\nHinweis: Raster ist flach (t geglaettet {smooth.min():.2f} bis "
              f"{smooth.max():.2f}) - auf Train ist keine Schwelle klar besser.")

    print(f"\n=== 3) Kontext bei Oversold >= {o_strong}: helfen Trend/Recovery? ===")
    print(pd.concat([context_check(train, o_strong), context_check(test, o_strong)], axis=1,
                    keys=["Train", "Test"]))

    print(f"\n=== 4) Stabilitaet: Oversold >= {sc.OVERSOLD_STRONG} (aktuell) je Jahr ===")
    years = yearly(df, sc.OVERSOLD_STRONG)
    print(years)
    positive = (years["Ueber 20T %"] > 0).sum()
    print(f"Positive Jahre: {positive} von {len(years)}")

    old = dict(o_strong=sc.OVERSOLD_STRONG, o_high=sc.OVERSOLD_HIGH, o_low=sc.OVERSOLD_LOW)
    new = dict(old, o_strong=o_strong, o_high=min(sc.OVERSOLD_HIGH, o_strong))

    print(f"\n=== 5) Signale wie im Screener (Phasenbeginn, {COOLDOWN_DAYS} Tage Sperrfrist) ===")
    thresholds = {f"aktuell >= {old['o_strong']}": old["o_strong"]}
    if new["o_strong"] != old["o_strong"]:
        thresholds[f"Vorschlag >= {new['o_strong']}"] = new["o_strong"]
    print(signal_table({"Train": train, "Test": test}, thresholds))

    print("\n=== 6) Setups auf TEST (nicht fuer die Suche verwendet) ===")
    for label, params in (("AKTUELLE Schwellen", old), ("NEUE Schwellen", new)):
        if label.startswith("NEUE") and params == old:
            continue
        print(f"\n--- {label}: {params} ---")
        print(setup_table(test, classify(test, **params)))

    print("\n=== 7) Urteil ===")
    adopt = verdict(test, old["o_strong"], new["o_strong"])

    print("\n=== Vorschlag fuer stock_scorer.py ===")
    params = new if adopt else old
    print(f"OVERSOLD_STRONG = {params['o_strong']}")
    print(f"OVERSOLD_HIGH = {params['o_high']}")
    print(f"OVERSOLD_LOW = {params['o_low']}")


if __name__ == "__main__":
    main()
