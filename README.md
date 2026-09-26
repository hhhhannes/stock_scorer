# Stock Scorer

Technisches Analyse-Werkzeug, das Aktien auf drei Dimensionen bewertet und daraus
ein **Chart-Setup** ableitet. Welche Signale tragen, wurde per Backtest geprüft –
das Setup beruht deshalb nur noch auf dem Oversold-Score, Trend und Recovery sind Kontext.

| Score | Frage | Zeithorizont |
|---|---|---|
| **Trend** (0–75) | Wie stark und klar ist der aktuelle Kurstrend? | 20–200 Handelstage |
| **Oversold** (0–75) | Wie stark ist die Aktie gegenüber ihrem eigenen Verlauf gedrückt? | 14–252 Handelstage |
| **Recovery** (0–75) | Gibt es erste Anzeichen, dass sich eine Drückung dreht? | 5–20 Handelstage |
| **Fundamental** (0–40) | Ergänzend: Bewertung, Wachstum, Marge, Analystenziel | – |

Das Projekt besteht aus einer Kommandozeile, einer Web-App (Streamlit), einem täglichen
GitHub-Workflow sowie Werkzeugen für Backtest und Kalibrierung.

> **Keine Anlageberatung.** Die Scores sind Heuristiken. Ob sie etwas taugen, lässt sich
> mit `backtest.py` und `calibrate.py` selbst überprüfen – siehe unten.

---

## Installation

Voraussetzung: Python 3.11 oder neuer.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux / macOS

python -m pip install -r requirements.txt  # alles inkl. Web-App
# oder nur Kommandozeile:
python -m pip install -r requirements-cli.txt
```

Virtuelle Umgebungen sind nicht verschiebbar. Wurde das Projekt samt `.venv` in einen
anderen Ordner kopiert oder heruntergeladen, `.venv` dort löschen und neu erstellen.

| Datei | Inhalt |
|---|---|
| `requirements-cli.txt` | Kern: yfinance (exakt gepinnt), pandas, numpy, tabulate |
| `requirements.txt` | Kern + Streamlit + Plotly (wird von Streamlit Cloud gelesen) |
| `requirements-dev.txt` | Kern + pytest |

---

## Benutzung

### Kommandozeile

```bash
python stock_scorer.py AAPL MSFT NVDA NESN.SW
python stock_scorer.py AAPL MSFT --markdown report.md    # Zusammenfassung zusätzlich als Markdown
python stock_scorer.py --help                            # Erklärung aller Scores und Setups
```

Ausgabe pro Titel: alle Scores mit Begründung, Volatilität, nächster Earnings-Termin und
das Setup. Am Ende eine Tabelle, sortiert nach Oversold (dann Recovery).

Ticker-Symbole folgen Yahoo Finance: `AAPL` (USA), `SAP.DE` (Xetra), `NESN.SW` (SIX).

### Web-App

```bash
streamlit run app.py
```

Ticker in der Seitenleiste eingeben → **Analysieren**. Pro Titel gibt es Score-Balken,
das Setup als farbiges Badge, Details und einen Kurschart mit SMA50/SMA200,
Bollinger-Bändern und Volumen. Daten werden eine Stunde zwischengespeichert
(**Daten neu laden** leert den Cache).

Kostenlos veröffentlichen: Repo auf GitHub pushen, auf
[share.streamlit.io](https://share.streamlit.io) verknüpfen, `app.py` wählen, *Deploy*.

### GitHub Actions

`.github/workflows/stock-scorer.yml` läuft Mo–Fr um 07:00 UTC (und manuell über
*Run workflow*):

1. Tests ausführen
2. Analyse für die Ticker im Workflow
3. Zusammenfassung direkt im Actions-Tab (Job Summary), vollständiger Report als Download

Die Ticker-Liste steht im Schritt *Script ausfuehren* des Workflows.

---

## Wie die Scores berechnet werden

Grundlage ist ein Jahr Tageskurse (dividenden-/splitbereinigt) von Yahoo Finance.
Alle Schwellen und Punktwerte stehen zentral am Anfang von `stock_scorer.py`.

### Trend (max. 75)

| Kriterium | Punkte |
|---|---|
| Kurs über SMA50 | 15 |
| Kurs über SMA200 | 15 |
| SMA50 über SMA200 (Golden-Cross-Lage) | 15 |
| Steigung SMA50 über 10 Tage > 1 % (> 0 %) | 15 (7) |
| Lineare Regression 20 Tage > 0.15 %/Tag (> 0) | 15 (7) |
| Regression 20 Tage < −0.15 %/Tag | −5 |

### Oversold (max. 75)

| Kriterium | Punkte |
|---|---|
| RSI14 < 30 (< 40) – mit Wilder-Glättung wie TradingView | 25 (12) |
| Position im 52-Wochen-Bereich (Hoch/Tief) < 20 % (< 35 %) | 20 (10) |
| Kurs > 10 % (> 3 %) unter SMA50 | 15 (8) |
| Kurs am/unter dem unteren Bollinger-Band (unter Mittellinie) | 15 (5) |

### Recovery (max. 75)

Verwendet bewusst **keinen RSI**, damit der Score unabhängig vom Oversold-Score ist.

| Kriterium | Punkte |
|---|---|
| Abstand zum 10-Tage-Tief ≥ 1.5 ATR (≥ 0.75 ATR) | 20 (10) |
| Kurs hat die SMA10 in den letzten 5 Tagen zurückerobert | 15 |
| Frischer MACD-Crossover über die Signallinie (MACD bereits darüber) | 20 (8) |
| Steigende Tiefs in den letzten 10 Tagen | 10 |
| Volumen an Auf-Tagen ≥ 1.3× Volumen an Ab-Tagen (letzte 20 Tage) | 10 |

### Fundamental (max. 40)

KGV < 25 (< 40): 10 (5) · Forward-KGV < KGV: 5 · Umsatzwachstum > 10 % (> 0): 10 (5) ·
Gewinnmarge > 10 %: 5 · Analysten-Kursziel > 10 % (> 0) über Kurs: 10 (5).
Fliesst **nicht** in das Setup ein, sondern dient als Zusatzinformation.

### Setups

Das Setup wird **allein aus dem Oversold-Score** abgeleitet
(Schwellen: `OVERSOLD_STRONG`, `OVERSOLD_HIGH`, `OVERSOLD_LOW`):

| Oversold | Setup | Lesart |
|---|---|---|
| ≥ 60 | **Stark überverkauft** | im Backtest das einzige Signal mit Überrendite nach 20 Tagen |
| 45–59 | **Gedrückt** | kein belegter Vorteil, nur Beobachtung |
| 20–44 | **Neutral** | – |
| < 20 | **Nicht günstig** | kein Rabatt (läuft evtl. bereits) |

**Trend und Recovery sind Kontext**, kein Teil des Setups. Sie werden angezeigt
(Trend als *Aufwärts / Seitwärts / Abwärts*), haben das Setup im Backtest aber nicht
verbessert – siehe [Ergebnis](#ergebnis-der-kalibrierung-stand-26092026).
Die Zusammenfassung ist nach Oversold sortiert, bei Gleichstand nach Recovery.

Zusätzlich warnt das Tool, wenn die nächsten **Earnings in ≤ 7 Tagen** anstehen –
kurzfristige Chart-Setups sind dann wenig aussagekräftig.

> Frühere Versionen kombinierten alle drei Scores zu Setups wie *Pullback+Wende* oder
> *Fallendes Messer*. Der Backtest hat diese Logik nicht bestätigt; sie wurde deshalb ersetzt.

---

## Backtest und Kalibrierung

### `backtest.py` – taugen die Setups etwas?

Berechnet für viele historische Stichtage die Scores (nur mit Daten bis zu diesem Tag)
und misst die Rendite nach 5, 10 und 20 Handelstagen. Neben der absoluten Rendite wird
die **Überrendite** ausgewiesen: Rendite minus Durchschnitt aller Titel in derselben Woche.
Damit ist der allgemeine Markttrend (z.B. ein Bullenmarkt) herausgerechnet.

```bash
python backtest.py AAPL MSFT NVDA --years 5
python backtest.py --universe universe.txt --years 10 --csv backtest.csv
```

`universe.txt` enthält rund 100 Titel aus USA, Deutschland/EU und der Schweiz –
bewusst auch Verlierer der letzten Jahre, damit die Auswertung nicht nur auf
Tech-Gewinnern beruht. Ein Lauf über 10 Jahre dauert einige Minuten.

### `calibrate.py` – Schwellen prüfen

```bash
python calibrate.py backtest.csv
```

Arbeitet nur mit der CSV aus dem Backtest (kein erneuter Download, dauert Sekunden).
Der Referenzlauf vom 26.09.2026 liegt in `data/`, das Ergebnis unten lässt sich damit
direkt nachvollziehen:

```bash
python calibrate.py data/backtest10y.csv
```

Die Spalte `setup` in dieser CSV stammt noch von der alten Setup-Logik; `calibrate.py`
verwendet nur die Roh-Scores (`trend`, `oversold`, `recovery`) und ist davon nicht betroffen.

Ablauf:

1. **Zeitlicher Train/Test-Split** (Standard: jüngste 30 % = Test). Schwellen werden nur
   auf Train gesucht.
2. **Aussagekraft je Score:** Überrendite nach Score-Klassen, getrennt für Train und Test.
3. **Suche nach `OVERSOLD_STRONG`** mit Mindesthäufigkeit. Bewertet wird der t-Wert der
   Überrendite, gemittelt mit den Nachbarn im Raster (stabiles Plateau statt Zufallsspitze).
4. **Kontext-Check:** Überrendite der stark überverkauften Titel nach Trend und Recovery –
   zeigt, ob sich eine zusätzliche Bedingung lohnen würde.
5. **Vergleich aktuelle vs. neue Schwellen auf Test** mit Urteil *BESTÄTIGT* oder
   *NICHT BESTÄTIGT* (t < 1) und Warnung, wenn das Optimum am Rand des Rasters liegt.

Der Vorschlag ist ein Hinweis, keine automatische Übernahme – die Konstanten stehen
am Anfang von `stock_scorer.py`.

### Ergebnis der Kalibrierung (Stand 26.09.2026)

96 Titel aus `universe.txt`, 2017–2026, 42 391 Stichtage. Train bis Jan. 2024, Test danach.

**Alte Logik (Kombination aus Trend, Oversold, Recovery) – verworfen:**

- *Pullback+Wende* ist in sich fast widersprüchlich: Ein hoher Trend-Score verlangt einen
  Kurs über SMA50, ein hoher Oversold-Score einen Kurs darunter. Das Setup kam praktisch
  nie vor. Die beste Kalibrierung auf Train hatte auf Test **−3.5 % Überrendite** (t = −3.95).
- *Fallendes Messer* war nicht schlechter als der Durchschnitt (Train und Test je +0.3 %).

**Neue Logik (nur Oversold):**

| Setup (Test, 2024–2026) | Anteil | Überrendite 20T | Trefferquote 20T |
|---|---|---|---|
| Stark überverkauft (≥ 60) | 2 % | **+1.9 %** (t = 2.7) | 56 % |
| Gedrückt (45–59) | 6 % | −0.1 % | 47 % |
| Neutral | 19 % | −0.3 % | 45 % |
| Nicht günstig | 73 % | 0.0 % | 47 % |

- Auf Train ist die Wahl der Schwelle **nicht eindeutig**: Zwischen 45 und 65 liegen die
  t-Werte alle bei 0.8–1.3. 60 hat auf Train die höchste mittlere Überrendite (+0.45 %).
- **Die Test-Zahl für 60 ist zu optimistisch:** Die Schwelle wurde gewählt, nachdem die
  Test-Daten bereits in einer Diagnose angesehen worden waren. Eine saubere Bestätigung
  gibt es erst mit neuen Daten (z.B. `calibrate.py` in 6–12 Monaten erneut laufen lassen).
- **Kontext bei gedrückten Titeln:** Ein Abwärtstrend war *nicht* schlechter. Titel ohne
  Wende-Signale (Recovery < 15) schnitten besser ab als solche mit beginnender Wende
  (Recovery 15–34, in Train und Test negativ).
- **Vorsicht Survivorship-Bias:** Er begünstigt gerade das Oversold-Signal. Titel, die sich
  nach einem Absturz *nicht* erholt haben und dekotiert wurden, fehlen in den Daten.

### Grenzen

- **Survivorship-Bias:** dekotierte Titel fehlen, weil Yahoo keine Daten liefert.
- Keine Gebühren, Slippage oder Steuern.
- Überlappende 20-Tage-Horizonte machen Stichtage voneinander abhängig; t-Werte dienen
  zum Vergleichen, nicht als Signifikanztest.
- Fundamentaldaten werden nicht getestet (historische Werte wären nicht verfügbar).

---

## Projektstruktur

```
stock_scorer.py      Indikatoren, Scores, Setup-Logik, Kommandozeile
app.py               Streamlit-Web-App
backtest.py          Historischer Test der Setups
calibrate.py         Kalibrierung der Setup-Schwellen
universe.txt         Ticker-Universum für Backtest/Kalibrierung
data/                Referenz-Backtest (10 Jahre, 96 Titel) als CSV + Log
tests/               pytest-Tests (Indikatoren, Scores, Setups, Kalibrierung)
.github/workflows/   Täglicher Lauf auf GitHub Actions
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
pytest
```

Die Tests laufen ohne Netzwerk mit synthetischen Kursreihen (z.B. RSI einer reinen
Aufwärtsreihe = 100, Setup-Einordnung an allen Schwellen).
