"""
IDX BREAKOUT & MOMENTUM SCREENER — WEB VERSION v2 (Streamlit)
================================================================
Upgrade: cache watchlist lengkap, filter sektor, Bollinger Bands, ATR,
dan data fundamental (PE/PBV/Market Cap) untuk top candidates.

Cara pakai:
  pip install streamlit yfinance pandas numpy plotly openpyxl matplotlib
  streamlit run idx_screener_web.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="IDX Screener", page_icon="📈", layout="wide")

CACHE_FILE = "idx_watchlist_cache.json"

DEFAULT_WATCHLIST = [
    "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNVR", "ICBP",
    "GOTO", "BUMI", "ANTM", "MDKA", "ADRO", "PTBA", "ITMG", "INCO",
    "AMMN", "BRPT", "TPIA", "CUAN", "PANI", "AVIA", "MBMA", "SMGR",
    "KLBF", "CPIN", "JPFA", "EXCL", "ISAT", "MEDC"
]


# ==========================================================
# CACHE HELPERS — biar upload gak perlu diulang tiap buka app
# ==========================================================
def save_cache(watchlist, sectors_map=None):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({
                "watchlist": watchlist,
                "sectors": sectors_map or {},
                "saved_at": datetime.now().strftime("%d %b %Y %H:%M")
            }, f)
    except Exception:
        pass


def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ==========================================================
# INDIKATOR
# ==========================================================
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


def calc_bollinger(series, period=20, num_std=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    percent_b = (series - lower) / (upper - lower)
    return upper, mid, lower, percent_b


def calc_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ==========================================================
# DATA FETCH
# ==========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_batch(tickers_tuple, period="6mo"):
    yf_tickers = " ".join(f"{t}.JK" for t in tickers_tuple)
    data = yf.download(yf_tickers, period=period, progress=False, auto_adjust=True,
                        group_by="ticker", threads=True)
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(ticker):
    """Cuma dipanggil buat top candidates -> gak berat."""
    try:
        info = yf.Ticker(f"{ticker}.JK").info
        return {
            "PE": info.get("trailingPE"),
            "PBV": info.get("priceToBook"),
            "MarketCap": info.get("marketCap"),
            "Sector": info.get("sector"),
        }
    except Exception:
        return {"PE": None, "PBV": None, "MarketCap": None, "Sector": None}


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
    high = df["High"].dropna()
    low = df["Low"].dropna()
    volume = df["Volume"].dropna()
    if len(close) < 55:
        return None

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    vol_avg20 = volume.rolling(20).mean()
    rsi = calc_rsi(close)
    macd_line, signal_line, hist = calc_macd(close)
    high_52w = close.rolling(min(len(close), 252)).max()
    bb_upper, bb_mid, bb_lower, percent_b = calc_bollinger(close)
    atr = calc_atr(high, low, close)

    last_close = close.iloc[-1]
    last_sma20 = sma20.iloc[-1]
    last_sma50 = sma50.iloc[-1]
    last_vol = volume.iloc[-1]
    last_vol_avg = vol_avg20.iloc[-1]
    last_rsi = rsi.iloc[-1]
    last_hist = hist.iloc[-1]
    prev_hist = hist.iloc[-2]
    last_52w_high = high_52w.iloc[-1]
    last_percent_b = percent_b.iloc[-1]
    last_atr = atr.iloc[-1]

    if last_close < params["min_price"] or pd.isna(last_rsi):
        return None

    pct_from_52w_high = (last_52w_high - last_close) / last_52w_high * 100
    vol_ratio = last_vol / last_vol_avg if last_vol_avg > 0 else 0
    macd_crossover_up = (prev_hist < 0) and (last_hist > 0)
    atr_pct = (last_atr / last_close * 100) if last_close else 0

    above_sma20 = last_close > last_sma20
    above_sma50 = last_close > last_sma50
    golden_trend = last_sma20 > last_sma50
    vol_spike = vol_ratio >= params["vol_spike_mult"]
    rsi_ok = params["rsi_min"] <= last_rsi <= params["rsi_max"]
    near_high = pct_from_52w_high <= params["near_52w_high_pct"]
    bb_breakout = last_percent_b >= 1.0
    bb_bounce = last_percent_b <= 0.1

    score = sum([above_sma20, above_sma50, golden_trend, vol_spike, rsi_ok,
                 near_high or macd_crossover_up, bb_breakout])

    return {
        "Ticker": ticker,
        "Close": round(last_close, 0),
        "RSI": round(last_rsi, 1),
        "Vol_Ratio": round(vol_ratio, 2),
        "ATR_Pct": round(atr_pct, 1),
        "Percent_B": round(last_percent_b, 2) if pd.notna(last_percent_b) else None,
        "Above_SMA20": above_sma20,
        "Above_SMA50": above_sma20 and above_sma50,
        "Golden_Trend": golden_trend,
        "MACD_Cross_Up": macd_crossover_up,
        "BB_Breakout": bb_breakout,
        "BB_Bounce": bb_bounce,
        "Pct_From_52wHigh": round(pct_from_52w_high, 1),
        "Score": score,
        "_close_series": close,
        "_sma20": sma20,
        "_sma50": sma50,
        "_bb_upper": bb_upper,
        "_bb_lower": bb_lower,
    }


def plot_candlestick(ticker, close, sma20, sma50, bb_upper, bb_lower):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, name="Close", line=dict(color="white", width=1.5)))
    fig.add_trace(go.Scatter(x=close.index, y=sma20, name="SMA20", line=dict(color="orange", width=1)))
    fig.add_trace(go.Scatter(x=close.index, y=sma50, name="SMA50", line=dict(color="blue", width=1)))
    fig.add_trace(go.Scatter(x=close.index, y=bb_upper, name="BB Upper", line=dict(color="gray", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=close.index, y=bb_lower, name="BB Lower", line=dict(color="gray", width=1, dash="dot"),
                              fill="tonexty", fillcolor="rgba(128,128,128,0.1)"))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark")
    return fig


# ==========================================================
# UI — SIDEBAR
# ==========================================================
cache = load_cache()

with st.sidebar:
    st.header("⚙️ Parameter")

    st.markdown("**Sumber Watchlist**")
    default_mode_idx = 1 if cache else 0
    mode = st.radio(
        "Pilih sumber saham",
        ["Watchlist manual", "Full IDX (upload/cache)"],
        index=default_mode_idx,
        label_visibility="collapsed"
    )

    watchlist = []
    sectors_map = {}

    if mode == "Watchlist manual":
        watchlist_input = st.text_area(
            "Watchlist (pisah koma)",
            value=", ".join(DEFAULT_WATCHLIST),
            height=120
        )
        watchlist = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
    else:
        if cache:
            st.success(f"✅ {len(cache['watchlist'])} saham tersimpan (terakhir update: {cache['saved_at']})")
            watchlist = cache["watchlist"]
            sectors_map = cache.get("sectors", {})
            if st.button("🗑️ Hapus cache & upload ulang"):
                os.remove(CACHE_FILE)
                st.rerun()
        else:
            st.caption(
                "Belum ada data tersimpan. Download dulu file-nya di "
                "idx.co.id/id/data-pasar/data-saham/daftar-saham (tombol 'Unduh'), lalu upload di sini. "
                "Sekali upload, otomatis kesimpen buat next time."
            )
            uploaded = st.file_uploader("Upload file Daftar Saham (.xlsx/.csv)", type=["xlsx", "xls", "csv"])
            if uploaded is not None:
                try:
                    raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
                    code_col = next((c for c in raw.columns if str(c).strip().lower() in
                                      ("kode", "code", "kode saham", "stock code")), raw.columns[0])
                    sector_col = next((c for c in raw.columns if "sektor" in str(c).strip().lower()
                                        or "sector" in str(c).strip().lower()), None)

                    tickers = raw[code_col].astype(str).str.strip().str.upper().dropna().tolist()
                    if sector_col:
                        for _, row in raw.iterrows():
                            code = str(row[code_col]).strip().upper()
                            sectors_map[code] = str(row[sector_col]).strip()

                    watchlist = [t for t in tickers if t.isalpha() and 2 <= len(t) <= 5]
                    save_cache(watchlist, sectors_map)
                    st.success(f"✅ {len(watchlist)} kode saham tersimpan permanen. Klik radio button lagi buat reload.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal baca file: {e}")

    # Filter sektor kalau ada datanya
    sector_filter = None
    if sectors_map:
        all_sectors = sorted(set(sectors_map.values()))
        sector_filter = st.multiselect("Filter sektor (kosongkan = semua)", all_sectors)
        if sector_filter:
            watchlist = [t for t in watchlist if sectors_map.get(t) in sector_filter]

    if len(watchlist) > 100:
        st.warning(f"⚠️ {len(watchlist)} saham akan discreening — bisa makan waktu beberapa menit.")
        batch_size = st.slider("Ukuran batch per request", 20, 100, 50, 10)
    else:
        batch_size = 50

    period = st.selectbox("Periode data", ["3mo", "6mo", "1y"], index=1)
    vol_spike_mult = st.slider("Volume spike minimal (x rata-rata)", 1.0, 3.0, 1.5, 0.1)
    rsi_min, rsi_max = st.slider("Range RSI sehat", 0, 100, (50, 75))
    near_52w_high_pct = st.slider("Maks % dari 52w high (breakout zone)", 1, 30, 10)
    min_price = st.number_input("Harga minimal (Rp)", value=50, step=10)
    fetch_fundamentals_toggle = st.checkbox("Ambil data fundamental (PE/PBV/Market Cap) untuk top candidates", value=True)
    run_button = st.button("🔍 Jalankan Screening", type="primary", use_container_width=True)

params = {
    "period": period, "vol_spike_mult": vol_spike_mult, "rsi_min": rsi_min,
    "rsi_max": rsi_max, "near_52w_high_pct": near_52w_high_pct, "min_price": min_price,
}

# ==========================================================
# MAIN
# ==========================================================
st.title("📈 IDX Breakout & Momentum Screener")
st.caption("Screening saham IDX: trend, volume, RSI, MACD, Bollinger Bands, ATR + fundamental untuk top candidates. DYOR.")

if run_button:
    if not watchlist:
        st.error("Watchlist kosong. Isi manual atau upload/cache dulu ya.")
        st.stop()

    results = []
    batches = [watchlist[i:i + batch_size] for i in range(0, len(watchlist), batch_size)]
    progress = st.progress(0, text="Memulai screening...")

    for bi, batch in enumerate(batches):
        progress.progress((bi + 1) / len(batches),
                           text=f"Batch {bi + 1}/{len(batches)} — {len(batch)} saham...")
        try:
            batch_data = fetch_batch(tuple(batch), params["period"])
        except Exception:
            continue
        for t in batch:
            r = analyze_ticker(t, batch_data, params)
            if r:
                r["Sector"] = sectors_map.get(t, "-")
                results.append(r)
    progress.empty()

    if not results:
        st.error("Gak ada hasil. Cek watchlist atau koneksi internet.")
    else:
        df_results = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)

        if fetch_fundamentals_toggle:
            top_tickers = df_results[df_results["Score"] >= 4]["Ticker"].tolist()[:15]
            fund_progress = st.progress(0, text="Mengambil data fundamental top candidates...")
            fund_data = {}
            for i, t in enumerate(top_tickers):
                fund_progress.progress((i + 1) / max(len(top_tickers), 1), text=f"Fundamental {t}...")
                fund_data[t] = fetch_fundamentals(t)
            fund_progress.empty()
            for t, f in fund_data.items():
                idx = df_results.index[df_results["Ticker"] == t]
                for k, v in f.items():
                    df_results.loc[idx, k] = v

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
            if row.get("BB_Breakout"):
                tags.append("🚀 BB Breakout")
            if row.get("BB_Bounce"):
                tags.append("🔄 BB Bounce (oversold)")
            if row["Pct_From_52wHigh"] <= params["near_52w_high_pct"]:
                tags.append(f"🎯 {row['Pct_From_52wHigh']}% dari 52w High")

            fund_str = ""
            if pd.notna(row.get("PE")):
                fund_str = f" | PE: {row['PE']:.1f} | PBV: {row.get('PBV', 0):.2f} | MCap: Rp{row.get('MarketCap', 0)/1e12:.1f}T"

            with st.expander(f"**{row['Ticker']}** — Rp{int(row['Close'])} — Score {row['Score']}/7 — ATR {row['ATR_Pct']}%{fund_str}"):
                st.write(" • ".join(tags) if tags else "-")
                if row.get("Sector") and row["Sector"] != "-":
                    st.caption(f"Sektor: {row['Sector']}")
                fig = plot_candlestick(row["Ticker"], row["_close_series"], row["_sma20"], row["_sma50"],
                                        row["_bb_upper"], row["_bb_lower"])
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Belum ada saham dengan score ≥4 saat ini.")

    st.subheader("📋 Semua Hasil")
    drop_cols = [c for c in ["_close_series", "_sma20", "_sma50", "_bb_upper", "_bb_lower"] if c in df_results.columns]
    display_df = df_results.drop(columns=drop_cols)
    try:
        styled_df = display_df.style.background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=7)
        st.dataframe(styled_df, use_container_width=True, height=500)
    except ImportError:
        st.dataframe(display_df, use_container_width=True, height=500)

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download CSV", csv, "screener_result.csv", "text/csv")
else:
    st.info("👈 Atur parameter di sidebar, terus klik **Jalankan Screening** buat mulai.")

st.divider()
st.caption("⚠️ Ini murni technical screening otomatis, bukan rekomendasi beli/jual. Cross-check fundamental & berita sebelum entry. DYOR.")
