"""
IDX BREAKOUT & MOMENTUM SCREENER — WEB VERSION (Streamlit)
=============================================================
Versi web dari screener saham, tampilan lebih enak di browser.

Cara pakai:
  pip install streamlit yfinance pandas numpy plotly
  streamlit run idx_screener_web.py

Browser bakal otomatis kebuka di http://localhost:8501
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="IDX Screener", page_icon="📈", layout="wide")

# ==========================================================
# WATCHLIST DEFAULT
# ==========================================================
DEFAULT_WATCHLIST = [
    "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNVR", "ICBP",
    "GOTO", "BUMI", "ANTM", "MDKA", "ADRO", "PTBA", "ITMG", "INCO",
    "AMMN", "BRPT", "TPIA", "CUAN", "PANI", "AVIA", "MBMA", "SMGR",
    "KLBF", "CPIN", "JPFA", "EXCL", "ISAT", "MEDC"
]


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_batch(tickers_tuple, period="6mo"):
    """Ambil banyak ticker sekaligus dalam 1 request -> jauh lebih cepat & gak gampang rate-limited."""
    yf_tickers = " ".join(f"{t}.JK" for t in tickers_tuple)
    data = yf.download(yf_tickers, period=period, progress=False, auto_adjust=True,
                        group_by="ticker", threads=True)
    return data


def analyze_ticker(ticker, batch_data, params):
    try:
        if isinstance(batch_data.columns, pd.MultiIndex):
            df = batch_data[f"{ticker}.JK"].dropna(how="all")
        else:
            df = batch_data
        if df.empty or len(df) < 55 or "Close" not in df.columns:
            return None
    except Exception:
        return None

    close = df["Close"].dropna()
    volume = df["Volume"].dropna()
    if len(close) < 55:
        return None

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    vol_avg20 = volume.rolling(20).mean()
    rsi = calc_rsi(close)
    macd_line, signal_line, hist = calc_macd(close)
    high_52w = close.rolling(min(len(close), 252)).max()

    last_close = close.iloc[-1]
    last_sma20 = sma20.iloc[-1]
    last_sma50 = sma50.iloc[-1]
    last_vol = volume.iloc[-1]
    last_vol_avg = vol_avg20.iloc[-1]
    last_rsi = rsi.iloc[-1]
    last_hist = hist.iloc[-1]
    prev_hist = hist.iloc[-2]
    last_52w_high = high_52w.iloc[-1]

    if last_close < params["min_price"] or pd.isna(last_rsi):
        return None

    pct_from_52w_high = (last_52w_high - last_close) / last_52w_high * 100
    vol_ratio = last_vol / last_vol_avg if last_vol_avg > 0 else 0
    macd_crossover_up = (prev_hist < 0) and (last_hist > 0)

    above_sma20 = last_close > last_sma20
    above_sma50 = last_close > last_sma50
    golden_trend = last_sma20 > last_sma50
    vol_spike = vol_ratio >= params["vol_spike_mult"]
    rsi_ok = params["rsi_min"] <= last_rsi <= params["rsi_max"]
    near_high = pct_from_52w_high <= params["near_52w_high_pct"]

    score = sum([above_sma20, above_sma50, golden_trend, vol_spike, rsi_ok,
                 near_high or macd_crossover_up])

    return {
        "Ticker": ticker,
        "Close": round(last_close, 0),
        "RSI": round(last_rsi, 1),
        "Vol_Ratio": round(vol_ratio, 2),
        "Above_SMA20": above_sma20,
        "Above_SMA50": above_sma20 and above_sma50,
        "Golden_Trend": golden_trend,
        "MACD_Cross_Up": macd_crossover_up,
        "Pct_From_52wHigh": round(pct_from_52w_high, 1),
        "Score": score,
        "_close_series": close,
        "_sma20": sma20,
        "_sma50": sma50,
    }


def plot_candlestick(ticker, df, sma20, sma50):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name=ticker
    ))
    fig.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA20", line=dict(color="orange", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA50", line=dict(color="blue", width=1)))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10),
                       xaxis_rangeslider_visible=False, template="plotly_dark")
    return fig


# ==========================================================
# UI
# ==========================================================
st.title("📈 IDX Breakout & Momentum Screener")
st.caption("Screening saham IDX berdasarkan trend, volume, RSI & MACD. DYOR, bukan rekomendasi finansial.")

with st.sidebar:
    st.header("⚙️ Parameter")

    st.markdown("**Sumber Watchlist**")
    mode = st.radio(
        "Pilih sumber saham",
        ["Watchlist manual", "Upload Daftar Saham IDX (semua saham)"],
        label_visibility="collapsed"
    )

    watchlist = []
    if mode == "Watchlist manual":
        watchlist_input = st.text_area(
            "Watchlist (pisah koma)",
            value=", ".join(DEFAULT_WATCHLIST),
            height=120
        )
        watchlist = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
    else:
        st.caption(
            "Download dulu file-nya di idx.co.id/id/data-pasar/data-saham/daftar-saham "
            "(tombol 'Unduh'), lalu upload di sini."
        )
        uploaded = st.file_uploader("Upload file Daftar Saham (.xlsx/.csv)", type=["xlsx", "xls", "csv"])
        if uploaded is not None:
            try:
                if uploaded.name.endswith(".csv"):
                    raw = pd.read_csv(uploaded)
                else:
                    raw = pd.read_excel(uploaded)
                # Cari kolom yang isinya kode saham (biasanya nama kolom "Kode" / "Code")
                code_col = None
                for c in raw.columns:
                    if str(c).strip().lower() in ("kode", "code", "kode saham", "stock code"):
                        code_col = c
                        break
                if code_col is None:
                    code_col = raw.columns[0]
                watchlist = (
                    raw[code_col].astype(str).str.strip().str.upper().dropna().unique().tolist()
                )
                watchlist = [t for t in watchlist if t.isalpha() and 2 <= len(t) <= 5]
                st.success(f"✅ {len(watchlist)} kode saham terdeteksi dari file.")
            except Exception as e:
                st.error(f"Gagal baca file: {e}")

    if len(watchlist) > 100:
        st.warning(
            f"⚠️ {len(watchlist)} saham akan discreening. Ini bisa makan waktu "
            "beberapa menit dan pake batch download biar gak kena rate-limit Yahoo Finance."
        )
        batch_size = st.slider("Ukuran batch per request", 20, 100, 50, 10)
    else:
        batch_size = 50

    period = st.selectbox("Periode data", ["3mo", "6mo", "1y"], index=1)
    vol_spike_mult = st.slider("Volume spike minimal (x rata-rata)", 1.0, 3.0, 1.5, 0.1)
    rsi_min, rsi_max = st.slider("Range RSI sehat", 0, 100, (50, 75))
    near_52w_high_pct = st.slider("Maks % dari 52w high (breakout zone)", 1, 30, 10)
    min_price = st.number_input("Harga minimal (Rp)", value=50, step=10)
    run_button = st.button("🔍 Jalankan Screening", type="primary", use_container_width=True)

params = {
    "period": period,
    "vol_spike_mult": vol_spike_mult,
    "rsi_min": rsi_min,
    "rsi_max": rsi_max,
    "near_52w_high_pct": near_52w_high_pct,
    "min_price": min_price,
}

if run_button:
    if not watchlist:
        st.error("Watchlist kosong. Isi manual atau upload file dulu ya.")
        st.stop()

    results = []
    batches = [watchlist[i:i + batch_size] for i in range(0, len(watchlist), batch_size)]
    progress = st.progress(0, text="Memulai screening...")

    for bi, batch in enumerate(batches):
        progress.progress(
            (bi + 1) / len(batches),
            text=f"Batch {bi + 1}/{len(batches)} — mengambil data {len(batch)} saham..."
        )
        try:
            batch_data = fetch_batch(tuple(batch), params["period"])
        except Exception:
            continue
        for t in batch:
            r = analyze_ticker(t, batch_data, params)
            if r:
                results.append(r)
    progress.empty()

    if not results:
        st.error("Gak ada hasil. Cek watchlist atau koneksi internet.")
    else:
        df_results = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
        st.session_state["results"] = df_results
        st.session_state["last_run"] = datetime.now().strftime("%d %b %Y %H:%M")

if "results" in st.session_state:
    df_results = st.session_state["results"]
    st.success(f"✅ Screening selesai — {st.session_state['last_run']} | {len(df_results)} saham dianalisis")

    top = df_results[df_results["Score"] >= 4]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Dianalisis", len(df_results))
    col2.metric("Top Candidates (score ≥4)", len(top))
    col3.metric("Rata-rata Score", round(df_results["Score"].mean(), 1))

    st.subheader("🔥 Top Candidates")
    if not top.empty:
        for _, row in top.iterrows():
            tags = []
            if row["Golden_Trend"]:
                tags.append("🟢 Golden Trend")
            if row["Vol_Ratio"] >= params["vol_spike_mult"]:
                tags.append(f"📊 Vol Spike {row['Vol_Ratio']}x")
            if row["MACD_Cross_Up"]:
                tags.append("📈 MACD Cross↑")
            if row["Pct_From_52wHigh"] <= params["near_52w_high_pct"]:
                tags.append(f"🎯 {row['Pct_From_52wHigh']}% dari 52w High")
            with st.expander(f"**{row['Ticker']}** — Rp{int(row['Close'])} — Score {row['Score']}/6"):
                st.write(" • ".join(tags) if tags else "-")
                fig = plot_candlestick(row["Ticker"], pd.DataFrame({
                    "Open": row["_close_series"], "High": row["_close_series"],
                    "Low": row["_close_series"], "Close": row["_close_series"]
                }), row["_sma20"], row["_sma50"])
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Belum ada saham dengan score ≥4 saat ini.")

    st.subheader("📋 Semua Hasil")
    display_df = df_results.drop(columns=["_close_series", "_sma20", "_sma50"])
    try:
        styled_df = display_df.style.background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=6)
        st.dataframe(styled_df, use_container_width=True, height=500)
    except ImportError:
        # matplotlib belum keinstall, tampilin tabel biasa tanpa warna gradient
        st.dataframe(display_df, use_container_width=True, height=500)
        st.caption("💡 Install `matplotlib` (pip3 install matplotlib) biar tabel score-nya berwarna gradient.")

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download CSV", csv, "screener_result.csv", "text/csv")
else:
    st.info("👈 Atur parameter di sidebar, terus klik **Jalankan Screening** buat mulai.")

st.divider()
st.caption("⚠️ Ini murni technical screening otomatis, bukan rekomendasi beli/jual. Cross-check fundamental & berita sebelum entry. DYOR.")
