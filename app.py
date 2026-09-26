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
    - Repo mit app.py + requirements_streamlit.txt + stock_scorer.py pushen
    - Auf share.streamlit.io einloggen, Repo verknuepfen, "Deploy" klicken
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stock_scorer import analyze_ticker

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
    default_tickers = "AAPL, MSFT, NVDA, GOOGL, AMZN"
    ticker_input = st.text_area(
        "Ticker-Symbole (kommagetrennt)",
        value=default_tickers,
        height=80,
        help="z.B. AAPL, MSFT, NVDA oder ZAL.DE, NESN.SW fuer europaeische Boersen",
    )
    run_button = st.button("🔍 Analysieren", type="primary", use_container_width=True)

    st.divider()
    with st.expander("ℹ️ Was bedeuten die Scores?"):
        st.markdown(
            """
**Trend** (0-75): Richtung/Konsistenz des Kurstrends
(SMA-Lage, SMA-Steigung, Regression der letzten 20 Tage)

**Oversold** (0-75): wie stark eine Aktie gegenueber ihrem
eigenen Kursverlauf gedrueckt ist (RSI, 52W-Range, SMA-Abstand,
Bollinger-Baender)

**Recovery** (0-75): erste technische Anzeichen einer Trendwende
(RSI-Drehung, SMA10-Rueckeroberung, MACD-Crossover, steigende
Tiefs, Volumen an Auftagen)

**Setups:**
- *Pullback+Wende*: Aufwaertstrend, gedrueckt, Wende sichtbar - am haeufigsten gesucht
- *Fallendes Messer*: Abwaertstrend, gedrueckt, keine Wende bestaetigt - riskant
- *Reversal-Versuch*: Wende gegen den Haupttrend - riskanter
- *Nicht guenstig*: laeuft bereits, kein Rabatt mehr
            """
        )

# ---------------------------------------------------------------------------
# Hilfsfunktionen fuer Darstellung
# ---------------------------------------------------------------------------

def score_bar(label: str, value: float, max_value: float = 75):
    st.progress(min(value / max_value, 1.0), text=f"{label}: {value:.0f}/{max_value:.0f}")


def setup_badge(verdict: str):
    if "Pullback+Wende" in verdict or "Erholung" in verdict:
        st.success(verdict)
    elif "Fallendes Messer" in verdict:
        st.error(verdict)
    elif "Reversal-Versuch" in verdict:
        st.warning(verdict)
    else:
        st.info(verdict)


def price_chart(hist: pd.DataFrame, symbol: str):
    close = hist["Close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20

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
    symbols = [s.strip().upper() for s in ticker_input.split(",") if s.strip()]

    if not symbols:
        st.warning("Bitte mindestens ein Ticker-Symbol eingeben.")
        st.stop()

    results = []
    progress = st.progress(0.0)
    for i, sym in enumerate(symbols):
        with st.spinner(f"Lade {sym} ..."):
            try:
                results.append(analyze_ticker(sym))
            except Exception as e:
                results.append({"symbol": sym, "error": str(e)})
        progress.progress((i + 1) / len(symbols))
    progress.empty()

    # --- Zusammenfassungstabelle ---
    valid = [r for r in results if "error" not in r]
    errored = [r for r in results if "error" in r]

    if valid:
        st.subheader("Zusammenfassung")
        summary_df = pd.DataFrame([
            {
                "Symbol": r["symbol"],
                "Name": r["name"],
                "Kurs": round(r["price"], 2),
                "Trend": r["trend_score"],
                "Oversold": r["oversold_score"],
                "Recovery": r["recovery_score"],
                "Einschaetzung": r["verdict"],
            }
            for r in valid
        ]).sort_values(by=["Oversold", "Recovery"], ascending=False)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if errored:
        for r in errored:
            st.error(f"{r['symbol']}: {r['error']}")

    st.divider()

    # --- Detailkarten pro Ticker ---
    for r in valid:
        st.subheader(f"{r['symbol']} — {r['name']}")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Kurs", f"{r['price']:.2f}")
            score_bar("Trend", r["trend_score"])
            score_bar("Oversold", r["oversold_score"])
            score_bar("Recovery", r["recovery_score"])
            setup_badge(r["verdict"])

            with st.expander("Details"):
                st.markdown(f"**Trend:** {r['trend_notes']}")
                st.markdown(f"**Oversold:** {r['oversold_notes']}")
                st.markdown(f"**Recovery:** {r['recovery_notes']}")
                st.markdown(f"**Fundamental:** {r['fundamentals']}")
                st.markdown(f"**Volatilitaet:** {r['volatility']}")
                st.markdown(f"**Termine:** {r['events']}")

        with col2:
            st.plotly_chart(price_chart(r["hist"], r["symbol"]), use_container_width=True)

        st.divider()
else:
    st.info("Ticker links eingeben und auf **Analysieren** klicken.")
