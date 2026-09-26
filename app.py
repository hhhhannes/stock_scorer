"""
app.py - Streamlit-Oberflaeche fuer den Stock Scorer
======================================================
Interaktive Web-App: Ticker eingeben, Trend-/Oversold-/Recovery-Scores
und einen Kursverlauf mit SMA50/SMA200/Bollinger-Baendern ansehen.

Installation:
    pip install -r requirements.txt

Lokal starten:
    streamlit run app.py

Deployment (kostenlos): https://share.streamlit.io (Streamlit Community Cloud)
    - Repo mit app.py + stock_scorer.py + requirements.txt + requirements-cli.txt pushen
    - Auf share.streamlit.io einloggen, Repo verknuepfen, "Deploy" klicken
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import stock_scorer as sc

CACHE_TTL_SECONDS = 3600

st.set_page_config(page_title="Stock Scorer", layout="wide")

st.title("📊 Stock Scorer: Trend / Oversold / Recovery")
st.caption(
    "Analyse-Werkzeug fuer kurzfristige Chart-Setups. "
    "Keine Anlageberatung, keine Kauf-/Verkaufsempfehlung."
)

# ---------------------------------------------------------------------------
# Eingabe
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Einstellungen")
    default_tickers = "AAPL, MSFT, NVDA, GOOGL, AMZN, AMAT, MU, CFR.SW, LISN.SW, ZAL.DE, GC=F, SI=F, EUNL.DE, TSLA, BTC-USD"
    ticker_input = st.text_area(
        "Ticker-Symbole (kommagetrennt)",
        value=default_tickers,
        height=80,
        help="z.B. AAPL, MSFT, NVDA oder ZAL.DE, NESN.SW fuer europaeische Boersen",
    )
    run_button = st.button("🔍 Analysieren", type="primary", width="stretch")
    if st.button("Daten neu laden", width="stretch",
                 help=f"Cache leeren (Daten werden sonst {CACHE_TTL_SECONDS // 60} Min. zwischengespeichert)"):
        st.cache_data.clear()

    st.divider()
    with st.expander("ℹ️ Was bedeuten die Scores?"):
        st.markdown(
            f"""
**Trend** (0-{sc.MAX_SCORE}): Richtung/Konsistenz des Kurstrends
(SMA-Lage, SMA-Steigung, Regression der letzten 20 Tage)

**Oversold** (0-{sc.MAX_SCORE}): wie stark eine Aktie gegenueber ihrem
eigenen Kursverlauf gedrueckt ist (RSI, 52W-Range, SMA-Abstand,
Bollinger-Baender)

**Recovery** (0-{sc.MAX_SCORE}): erste technische Anzeichen einer Trendwende
(Rebound vom 10-Tage-Tief in ATR, SMA10-Rueckeroberung, MACD-Crossover,
steigende Tiefs, Volumen an Auftagen)

**Fundamental** (0-{sc.MAX_FUNDAMENTAL_SCORE}): KGV, Wachstum, Marge, Analysten-Kursziel

**Setup** (allein aus dem Oversold-Score):
- *{sc.SETUP_STRONG_OVERSOLD}* (>= {sc.OVERSOLD_STRONG}): im Backtest einziges stabiles
  Signal - Ueberrendite nach 20 Tagen, auch im Testzeitraum
- *{sc.SETUP_PRESSED}* (>= {sc.OVERSOLD_HIGH}): kein belegter Vorteil, nur Beobachtung
- *{sc.SETUP_NEUTRAL}*: dazwischen
- *{sc.SETUP_NOT_CHEAP}* (< {sc.OVERSOLD_LOW}): kein Rabatt

Trend und Recovery sind **Kontext**: Im Backtest haben sie das Setup nicht
verbessert. Ein Abwaertstrend war bei gedrueckten Titeln nicht schlechter.
Vorsicht: Survivorship-Bias - dekotierte Titel fehlen in den Daten.
            """
        )

# ---------------------------------------------------------------------------
# Daten & Hilfsfunktionen fuer Darstellung
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_results(symbols: tuple[str, ...]) -> list[dict]:
    return sc.analyze_many(list(symbols))


SETUP_STYLE = {
    sc.SETUP_STRONG_OVERSOLD: st.success,
}


def score_bar(label: str, value: float, max_value: float = sc.MAX_SCORE):
    st.progress(min(max(value / max_value, 0.0), 1.0), text=f"{label}: {value:.0f}/{max_value:.0f}")


def setup_badge(result: dict):
    SETUP_STYLE.get(result["setup"], st.info)(result["verdict"])


def notes_markdown(notes: list[str]) -> str:
    return "\n".join(f"- {n}" for n in notes)


def price_chart(hist: pd.DataFrame, symbol: str):
    close = hist["Close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    lower_band, _, upper_band = sc.bollinger_bands(close)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    fig.add_trace(go.Scatter(x=hist.index, y=upper_band, line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=lower_band, line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(120,120,120,0.15)",
                             name="Bollinger-Band", hoverinfo="skip"), row=1, col=1)

    fig.add_trace(go.Scatter(x=hist.index, y=close, name="Kurs",
                             line=dict(color="#1f77b4", width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=sma50, name="SMA50",
                             line=dict(color="orange", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=sma200, name="SMA200",
                             line=dict(color="red", width=1.2, dash="dot")), row=1, col=1)

    colors = ["#2ca02c" if c >= o else "#d62728"
              for o, c in zip(hist["Open"], hist["Close"])]
    fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="Volumen",
                         marker_color=colors, showlegend=False), row=2, col=1)

    fig.update_layout(
        title=f"{symbol} - Kursverlauf (1 Jahr)",
        height=480, margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Kurs", row=1, col=1)
    fig.update_yaxes(title_text="Volumen", row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# Hauptbereich
# ---------------------------------------------------------------------------

if run_button:
    symbols = list(dict.fromkeys(s.strip().upper() for s in ticker_input.split(",") if s.strip()))

    if not symbols:
        st.warning("Bitte mindestens ein Ticker-Symbol eingeben.")
        st.stop()

    with st.spinner(f"Lade {', '.join(symbols)} ..."):
        results = load_results(tuple(symbols))

    valid = sc.sort_results(results)
    errored = [r for r in results if "error" in r]

    # --- Zusammenfassungstabelle ---
    if valid:
        st.subheader("Zusammenfassung")
        st.caption("Sortiert nach Oversold, dann Recovery")
        summary_df = pd.DataFrame([
            {
                "Symbol": r["symbol"],
                "Name": r["name"],
                "Kurs": round(r["price"], 2),
                "Waehrung": r["currency"],
                "Trend": r["trend_score"],
                "Oversold": r["oversold_score"],
                "Recovery": r["recovery_score"],
                "Fundamental": r["fundamental_score"],
                "Setup": r["setup"],
                "Earnings": "⚠️ bald" if r["earnings_warning"] else "",
            }
            for r in valid
        ])
        st.dataframe(summary_df, width="stretch", hide_index=True)

    for r in errored:
        st.error(f"{r['symbol']}: {r['error']}")

    st.divider()

    # --- Detailkarten pro Ticker ---
    for r in valid:
        st.subheader(f"{r['symbol']} — {r['name']}")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Kurs", sc.format_price(r))
            score_bar("Trend", r["trend_score"])
            score_bar("Oversold", r["oversold_score"])
            score_bar("Recovery", r["recovery_score"])
            score_bar("Fundamental", r["fundamental_score"], sc.MAX_FUNDAMENTAL_SCORE)
            setup_badge(r)
            if r["earnings_warning"]:
                st.warning(r["events"])

            with st.expander("Details"):
                st.markdown("**Trend:**\n" + notes_markdown(r["trend_notes"]))
                st.markdown("**Oversold:**\n" + notes_markdown(r["oversold_notes"]))
                st.markdown("**Recovery:**\n" + notes_markdown(r["recovery_notes"]))
                st.markdown("**Fundamental:**\n" + notes_markdown(r["fundamental_notes"]))
                st.markdown(f"**Volatilitaet:** {r['volatility']}")
                st.markdown(f"**Termine:** {r['events']}")

        with col2:
            st.plotly_chart(price_chart(r["hist"], r["symbol"]), width="stretch")

        st.divider()
else:
    st.info("Ticker links eingeben und auf **Analysieren** klicken.")
