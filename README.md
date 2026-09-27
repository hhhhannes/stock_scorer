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
Bollinger-Bändern und Volumen. Violette Dreiecke im Chart markieren frühere Signale
*Stark überverkauft* mit der Rendite 20 Handelstage danach. Die Analyse wird eine
Stunde zwischengespeichert, Kurse eine Minute (**Daten neu laden** leert den Cache).

Kostenlos veröffentlichen: Repo auf GitHub pushen, auf
[share.streamlit.io](https://share.streamlit.io) verknüpfen, `app.py` wählen, *Deploy*.

### GitHub Actions

`.github/workflows/stock-scorer.yml` läuft Mo–Fr um 07:00 UTC (und manuell über
*Run workflow*):

1. Tests ausführen
2. Analyse für die Ticker im Workflow
3. Zusammenfassung direkt im Actions-Tab (Job Summary), vollständiger Report als Download

Die Ticker-Liste steht im Schritt *Script ausfuehren* des Workflows.

### Preis- und Score-Alarme (Telegram)

`alerts.py` prüft die Regeln in `alerts.txt` und schickt neu ausgelöste Alarme
per Telegram. `.github/workflows/alerts.yml` führt das Mo–Fr alle 30 Minuten
von 07:00 bis 21:30 UTC aus (GitHub startet geplante Läufe oft 5–30 Min. später).

```
# SYMBOL  KENNZAHL  OPERATOR  SCHWELLE
AAPL      price     <         300      # Kurs
NVDA      change    <=        -5       # Tagesänderung in %
MSFT      oversold  >=        60       # auch trend, recovery, fundamental
NESN.SW   setup     =         stark    # stark, gedrueckt, neutral, teuer (= oder !=)
AAPL      earnings  <=        7        # Tage bis zu den nächsten Quartalszahlen
```

Eine Regel meldet sich einmal und erst wieder, nachdem die Bedingung
zwischenzeitlich nicht mehr erfüllt war (Zustand im Actions-Cache).

Einrichtung:

1. In Telegram **@BotFather** anschreiben, `/newbot`, den **Token** notieren.
2. Dem neuen Bot eine beliebige Nachricht schicken, dann
   `https://api.telegram.org/bot<TOKEN>/getUpdates` im Browser öffnen und die
   Zahl bei `"chat":{"id":...}` notieren (**Chat-ID**).
3. Im GitHub-Repo unter *Settings → Secrets and variables → Actions* die Secrets
   `TELEGRAM_BOT_TOKEN` und `TELEGRAM_CHAT_ID` anlegen.
4. Im Actions-Tab den Workflow *Alarme* einmal mit *Run workflow* testen.

Lokal testen ohne zu senden: `python alerts.py --dry-run`

### Screener

`screener.py` prüft alle Titel aus `universe.txt` auf das Setup *Stark
überverkauft* – das einzige Signal, das im Backtest stabil war – und meldet
neue Signale per Telegram (gleiche Secrets wie die Alarme).
`.github/workflows/screener.yml` läuft Di–Sa um 05:30 UTC und wertet damit die
Schlusskurse Mo–Fr aus.

- Signal = erster Tag einer Phase *Stark überverkauft*
- Pro Titel 30 Tage Sperrfrist nach einem Signal (Zustand im Actions-Cache)

Lokal testen: `python screener.py --dry-run`

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
Tech-Gewinnern beruht. Die Titel werden parallel auf allen CPU-Kernen gerechnet
(`--jobs N` begrenzt das), ein Lauf über 10 Jahre dauert trotzdem einige Minuten.

Die Spalte `t Ueber 20T` ist ein **robuster t-Wert** der Überrendite. Er berücksichtigt,
dass Stichtage nicht unabhängig sind:

- Titel mit Signal in derselben Woche (z.B. März 2020) zählen als ein Cluster, nicht als
  viele unabhängige Beobachtungen.
- Überlappende 20-Tage-Horizonte korrelieren benachbarte Wochen (Newey-West, 4 Wochen).

Er fällt deshalb kleiner aus als ein naiver t-Wert. Faustregel: ab etwa 2 ist ein Effekt
kaum noch mit Zufall zu erklären.

### `calibrate.py` – Schwellen prüfen

```bash
python calibrate.py backtest.csv
python calibrate.py backtest.csv --split-date 2026-09-26   # Test = nur Daten ab Stichtag
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
   auf Train gesucht. Mit `--split-date` beginnt Test an einem festen Datum – z.B. dem der
   letzten Kalibrierung, damit nur Daten zählen, die bei der Wahl der Schwelle noch
   unbekannt waren.
2. **Aussagekraft je Score:** Überrendite nach Score-Klassen, getrennt für Train und Test.
3. **Suche nach `OVERSOLD_STRONG`** (Raster 40–70) mit Mindesthäufigkeit. Bewertet wird der
   robuste t-Wert der Überrendite, gemittelt mit den Nachbarn im Raster (stabiles Plateau
   statt Zufallsspitze). Hinweis, wenn das Raster flach ist, also keine Schwelle klar besser.
4. **Kontext-Check:** Überrendite der stark überverkauften Titel nach Trend und Recovery –
   zeigt, ob sich eine zusätzliche Bedingung lohnen würde.
5. **Stabilität je Jahr:** trägt das Signal über die Zeit oder nur in einzelnen Phasen?
6. **Signale wie im Screener:** nur der erste Stichtag einer Phase, danach 30 Tage
   Sperrfrist pro Titel. Eine lange Phase zählt damit einmal statt jede Woche.
7. **Urteil:** Der Vorschlag wird nur empfohlen, wenn er auf Test t ≥ 1 erreicht, nicht am
   Rand des Rasters liegt und auf Test mindestens so gut ist wie die aktuelle Schwelle.
   Sonst lautet der Vorschlag: aktuelle Werte beibehalten.

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

| Setup (Test, 2024–2026) | Anteil | Überrendite 20T | robuster t | Trefferquote 20T |
|---|---|---|---|---|
| Stark überverkauft (≥ 60) | 2 % | **+1.9 %** | 2.6 | 56 % |
| Gedrückt (45–59) | 6 % | −0.1 % | −0.3 | 47 % |
| Neutral | 19 % | −0.3 % | −1.0 | 45 % |
| Nicht günstig | 73 % | 0.0 % | 0.4 | 47 % |

- **Auf Train ist das Signal schwach:** +0.45 % Überrendite, robuster t = 0.7. Die Wahl der
  Schwelle ist nicht eindeutig: Zwischen 40 und 65 liegen die t-Werte alle bei 0.5–0.8.
  `calibrate.py` schlägt deshalb vor, 60 beizubehalten.
- **Je Jahr:** In 8 von 10 Jahren positiv, negativ waren 2020 (−1.3 %) und 2021 (−1.5 %).
- **Wie im Screener gezählt** (nur Phasenbeginn, 30 Tage Sperrfrist): Train +0.4 % (t = 0.7,
  439 Signale), Test +1.2 % (t = 1.4, 149 Signale). Auf Test also kleiner als in der
  Tabelle oben, die jede Woche einer Phase einzeln zählt.
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
- Überlappende Horizonte und gleichzeitige Signale werden im robusten t-Wert
  berücksichtigt. Weil die Schwellen auf denselben Daten gesucht und mehrfach angesehen
  wurden, ist er trotzdem kein strenger Signifikanztest.
- Fundamentaldaten werden nicht getestet (historische Werte wären nicht verfügbar).

---

## Projektstruktur

```
stock_scorer.py      Indikatoren, Scores, Setup-Logik, Kommandozeile
app.py               Streamlit-Web-App (Analyse, frühere Signale im Chart)
screener.py          Täglicher Screener 'Stark überverkauft' mit Telegram-Meldung
alerts.py            Preis-/Score-Alarme per Telegram (Regeln in alerts.txt)
backtest.py          Historischer Test der Setups
calibrate.py         Kalibrierung der Setup-Schwellen
universe.txt         Ticker-Universum für Backtest/Kalibrierung
data/                Referenz-Backtest (10 Jahre, 96 Titel) als CSV + Log
tests/               pytest-Tests (Indikatoren, Scores, Setups, Kalibrierung)
.github/workflows/   Täglicher Report, Screener und Alarme auf GitHub Actions
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
pytest
```

Die Tests laufen ohne Netzwerk mit synthetischen Kursreihen (z.B. RSI einer reinen
Aufwärtsreihe = 100, Setup-Einordnung an allen Schwellen).
