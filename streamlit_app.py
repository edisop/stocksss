# streamlit_app.py
# Streamlit Cloud UI for your Modal-based "investment plans".
# - On startup: ensure there's a plan for today (>= 9am Australia/Melbourne). If none, create once.
# - Every run: show a dropdown of all plans (oldest -> latest), default to latest.
# - For the selected plan: Allows adjusting K, Amount, and Temp to simulate different allocations.
# - Bottom section: Historical timeline of all plans vs S&P 500 using ADJUSTED settings.
#
# Required secrets in Streamlit Cloud (Settings → Secrets):
#   MODAL_TOKEN_ID=...
#   MODAL_TOKEN_SECRET=...
# Optional (or set defaults below):
#   MODAL_APP_INFER=inmemory-latest-infer
#   MODAL_APP_PLANS=inmemory-latest-plans
#   CKPT_PATH=/vol/models/ckpts/biex_listmle_final.pt
#   TOP_K=20
#   INVEST_AMT=1000
#   TEMP=2.0
#   TIMEZONE=Australia/Melbourne
#   AUTO_CREATE_ON_START=1
#
# First deploy your Modal apps from your laptop:
#   modal deploy modal_infer_inmemory_latest.py
#   modal deploy modal_plans_store.py
#
from __future__ import annotations
import os, math, time
from datetime import datetime, date, timedelta
from typing import Dict, Any, List
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# --- Bridge Streamlit Secrets → Environment for Modal SDK ---
if hasattr(st, 'secrets'):
    tid = st.secrets.get('MODAL_TOKEN_ID')
    if tid:
        os.environ['MODAL_TOKEN_ID'] = str(tid)
    tsec = st.secrets.get('MODAL_TOKEN_SECRET')
    if tsec:
        os.environ['MODAL_TOKEN_SECRET'] = str(tsec)
    # Optional: forward other config as envs so os.getenv picks them up
    for k in (
        'MODAL_APP_INFER','MODAL_APP_PLANS','CKPT_PATH','TOP_K','INVEST_AMT','TEMP','TIMEZONE','AUTO_CREATE_ON_START'
    ):
        if k in st.secrets:
            os.environ[k] = str(st.secrets[k])

# Light price fetch
try:
    import yfinance as yf
except ImportError:
    st.error("`yfinance` is not installed. On Streamlit Cloud, add it to **requirements.txt**. Locally: `pip install yfinance`.")
    st.stop()

from modal import Function, App

# -----------------------------
# Config / Secrets
# -----------------------------
APP_INFER = os.getenv("MODAL_APP_INFER", "inmemory-latest-infer")
APP_PLANS = os.getenv("MODAL_APP_PLANS", "inmemory-latest-plans")

CKPT_PATH   = os.getenv("CKPT_PATH", "/vol/models/ckpts/biex_listmle_final.pt")
TOP_K       = int(os.getenv("TOP_K", "20"))
INVEST_AMT  = float(os.getenv("INVEST_AMT", "1000"))
TEMP        = float(os.getenv("TEMP", "2.0"))
TZ_NAME     = os.getenv("TIMEZONE", "Australia/Melbourne")
AUTO_CREATE = os.getenv("AUTO_CREATE_ON_START", "1") == "1"

TZ = ZoneInfo(TZ_NAME)

# -----------------------------
# Ticker universe (copied to keep Streamlit thin; Modal still does heavy work)
# -----------------------------
TICKERS_UNIVERSE = [
    'APTV','KEYS','LII','YUM','BX','MSCI','NKE','WELL','NVR','RJF','BF-B','GWW','NCLH','JCI','URI','SPGI','BLDR','INTU',
    'BXP','DOV','FTNT','MDT','LVS','STE','TROW','MAS','BG','ALB','CRM','HAL','PKG','VRTX','HRL','DLR','DAL','MLM','AON',
    'GE','NWS','AMZN','LUV','C','SBUX','AIG','SHW','KEY','AME','EBAY','DECK','MRK','L','EQIX','NTAP','COIN','VICI','LH',
    'EQT','PTC','KHC','KMB','IVZ','ABT','CINF','CMCSA','TRMB','RSG','EA','CNC','PFE','TPL','GPC','DIS','CTSH','O','CAT',
    'PPG','AKAM','RVTY','BR','OMC','ORLY','CFG','NVDA','WBD','FFIV','BEN','SYF','WDAY','CHTR','RCL','MNST','PYPL','PEP',
    'V','FDX','LW','GRMN','CPT','APH','LLY','UHS','WDC','DD','TMO','MTD','TDY','HIG','XOM','HLT','JBL','D','JBHT','CLX',
    'K','IRM','COO','OTIS','EOG','ES','HPQ','EMN','TYL','UPS','PSX','LMT','AVGO','DVN','CVS','FE','ECL','J','STT','AXON',
    'ELV','HOLX','GIS','MTB','AVB','NDAQ','CRWD','LYB','BA','DRI','MAR','MHK','GEHC','MGM','XEL','WY','AIZ','GILD','MET',
    'DUK','TKO','LYV','NDSN','MSFT','EXC','EW','VLTO','FSLR','BALL','CPB','GEV','CDW','EXPE','ITW','TXN','MA','WEC','ROST',
    'TRGP','PCG','EMR','HWM','ON','SPG','STLD','ENPH','TTD','MTCH','NEM','CPRT','HUBB','ANET','EPAM','ALLE','WTW','AMCR',
    'JKHY','CTVA','IR','CSCO','DGX','TFC','COF','VTRS','AAPL','NTRS','CAG','NEE','EL','CPAY','EXPD','ALGN','DLTR','IDXX',
    'LHX','RL','CF','CSGP','TXT','POOL','OXY','ADSK','HPE','PM','BRK-B','GPN','HON','NWSA','CNP','TDG','ADBE','IP','MMM',
    'APO','SNPS','RMD','KMX','GDDY','KMI','FDS','NFLX','VST','ABBV','EXE','MMC','BSX','OKE','PNW','KLAC','DE','DELL',
    'MOH','STX','FOX','VMC','MDLZ','CMI','SLB','MS','MCK','HAS','CMS','PCAR','FITB','WMB','GEN','WAT','CSX','WYNN','META',
    'DG','NXPI','SW','VRSN','EVRG','KVUE','MO','PGR','AMP','SYY','WRB','UDR','PAYX','BMY','HSY','ETN','PODD','DHR','PH',
    'MKC','INVH','TECH','HST','GOOGL','NUE','F','HII','APA','FAST','BDX','BIIB','LIN','MPC','HUM','IFF','SWKS','TSLA',
    'NRG','PNC','BBY','CTAS','ACN','T','CMG','MOS','AES','BAX','WFC','AVY','DTE','GOOG','CEG','PHM','IPG','LEN','TER',
    'WSM','AFL','MAA','SMCI','CCI','FOXA','GL','VTR','GNRC','BKNG','PEG','FI','PNR','SJM','UNH','MCHP','ADP','MCO','UNP',
    'ROP','AXP','CI','INTC','XYZ','EIX','PAYC','FIS','ISRG','ATO','DXCM','DOW','FICO','CHD','DPZ','KIM','PLD','ROL','KKR',
    'WST','DOC','KR','TRV','INCY','KDP','TGT','PSA','TJX','GM','EQR','MKTX','AMD','USB','AZO','DAY','BLK','AMAT','DVA',
    'EG','JPM','REGN','ADM','PPL','STZ','SRE','NOW','BAC','MU','TT','APD','CRL','VZ','AWK','ESS','VRSK','SBAC','ETR','FRT',
    'GD','MPWR','ORCL','HSIC','CDNS','PANW','FTV','LKQ','TSN','AMGN','PFG','ERIE','SOLV','WAB','BRO','TMUS','SWK','WMT',
    'RF','CTRA','KO','HCA','ARE','SNA','TAP','COST','GLW','LDOS','HD','MCD','DHI','FCX','NI','COP','LNT','REG','ADI','MRNA',
    'COR','CL','IBM','CZR','CCL','IQV','XYL','MSI','SYK','TPR','A','CBOE','JNJ','CAH','CARR','CME','ED','TSCO','TEL','BK',
    'WM','CVX','PRU','CBRE','CB','UAL','AEE','ROK','UBER','ODFL','CHRW','IEX','WBA','FANG','AOS','AMT','ZBRA','GS','ULTA',
    'PWR','DASH','ALL','LULU','IT','BKR','SO','RTX','LRCX','AJG','ICE','PG','QCOM','DDOG','ACGL','EFX','ABNB','NOC','NSC',
    'AEP','SCHW','EXR','PLTR','HBAN','TTWO','VLO','LOW'
]


# Modal functions (looked up by name; requires deployed apps)
build_payload = Function.from_name(APP_INFER, "build_payload_remote")
run_infer     = Function.from_name(APP_INFER, "run_infer_remote")

save_plan_fn  = Function.from_name(APP_PLANS, "save_plan")
list_plans_fn = Function.from_name(APP_PLANS, "list_plans")
get_plan_fn   = Function.from_name(APP_PLANS, "get_plan")

st.set_page_config(page_title="Daily Investment Plans", layout="wide")

st.title("📈 Daily Investment Plans (Modal × Streamlit)")
st.caption("Plans are generated by your Modal GPU inferencer and persisted in a Modal Volume. This dashboard fetches live prices to show P&L.")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=60)
def _live_prices(tickers: List[str]) -> Dict[str, float]:
    """Fetch live prices for a list of tickers. Tries FastInfo, then 1d history, then 5d history."""
    prices: Dict[str, float] = {}
    if not tickers:
        return prices
    
    # Chunking to avoid massive URL strings if list is huge
    all_tickers = list(set(tickers))
    
    try:
        # We fetch 5d to be safe over weekends/holidays
        df = yf.download(all_tickers, period="5d", interval="1d", progress=False, group_by="ticker", threads=True)
        # Handle MultiIndex vs Single Index
        if len(all_tickers) == 1:
            t = all_tickers[0]
            try:
                if not df.empty:
                    prices[t] = float(df["Close"].dropna().iloc[-1])
            except:
                pass
        else:
            if isinstance(df.columns, pd.MultiIndex):
                for t in all_tickers:
                    try:
                        if t in df.columns.levels[0]:
                            series = df[t]["Close"].dropna()
                            if not series.empty:
                                prices[t] = float(series.iloc[-1])
                    except:
                        pass
    except Exception:
        pass

    return prices

@st.cache_data(ttl=300)
def _get_spy_history():
    """Fetches 1 year of SPY history to use for benchmarking."""
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="2y", interval="1d")
        return hist["Close"]
    except:
        return pd.Series(dtype=float)

def mel_now():
    return datetime.now(TZ)

def need_today_plan(plans: List[Dict[str, Any]]) -> bool:
    """True if it's >= 9:00 in Melbourne and no plan exists for today's date."""
    now = mel_now()
    if now.hour < 9:
        return False
    # If any plan has created_at_melbourne date == today, consider done
    today = now.date()
    for p in plans:
        cam = p.get("created_at_melbourne")
        try:
            d = datetime.fromisoformat(cam).astimezone(TZ).date()
            if d == today:
                return False
        except Exception:
            pass
    return True

def create_plan_now() -> Dict[str, Any]:
    """Orchestrate: build payload -> run infer -> save plan (adds buy prices & shares)."""
    payload = build_payload.remote(for_date=None, tickers=TICKERS_UNIVERSE, max_candidates=500)
    result  = run_infer.remote(payload=payload, ckpt_path=CKPT_PATH, top_k=TOP_K, invest_amt=INVEST_AMT, temp=TEMP)
    # Persist as a plan (adds buy prices & shares on Modal side)
    saved   = save_plan_fn.remote(result)
    return saved

def recalculate_metrics(df_rows: pd.DataFrame, top_k: int, temp: float, invest_amt: float):
    """
    Recalculates weights, allocations, and shares based on simulation parameters.
    Assumes df_rows has 'score' (raw logits) and 'buy_price'.
    """
    # 1. Sort by score descending and take Top K
    df_sim = df_rows.sort_values(by="score", ascending=False).head(top_k).copy()
    
    # 2. Compute Softmax with new Temp
    # Softmax formula: exp(x/T) / sum(exp(x/T))
    # Stability fix: subtract max
    scores = df_sim["score"].values.astype(float)
    if temp <= 0: temp = 0.01 # prevent div by zero
    
    scaled_scores = scores / temp
    max_s = np.max(scaled_scores)
    exps = np.exp(scaled_scores - max_s)
    sum_exps = np.sum(exps)
    weights = exps / sum_exps
    
    df_sim["weight"] = weights
    
    # 3. Allocation and Shares
    df_sim["allocation"] = df_sim["weight"] * invest_amt
    # Guard against zero buy prices just in case
    df_sim["buy_price"] = pd.to_numeric(df_sim["buy_price"], errors="coerce").fillna(0.0)
    
    # If buy price is 0 or NaN, shares = 0
    df_sim["shares"] = np.where(
        df_sim["buy_price"] > 0, 
        df_sim["allocation"] / df_sim["buy_price"], 
        0.0
    )
    
    return df_sim

# -----------------------------
# Load plans and maybe auto-create
# -----------------------------
with st.spinner("Loading plans list..."):
    plans = list_plans_fn.remote()

if AUTO_CREATE and need_today_plan(plans):
    with st.spinner("Creating today's plan on Modal..."):
        try:
            saved = create_plan_now()
            # refresh list
            plans = list_plans_fn.remote()
            st.success("Today's plan created.")
        except Exception as e:
            st.error(f"Failed to auto-create plan: {e}")

if not plans:
    st.info("No plans yet. Click **Create Plan Now** to run inference and persist an initial plan.")
    if st.button("Create Plan Now", type="primary"):
        with st.spinner("Creating plan on Modal..."):
            saved = create_plan_now()
            plans = list_plans_fn.remote()
            st.success("Plan created.")
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")
if st.sidebar.button("Create Plan Now"):
    with st.spinner("Creating plan on Modal..."):
        saved = create_plan_now()
        plans = list_plans_fn.remote()
        st.success("Plan created.")

# Build select options
def _label(p):
    created = p.get("created_at_melbourne") or p.get("created_at_utc")
    date_str = p.get("date") or "n/a"
    return f"{p['plan_id']} — {date_str} — created {created}"

options = {_label(p): p["plan_id"] for p in plans}
labels = list(options.keys())
labels.sort()  # oldest->latest by label (plan_id contains UTC timestamp)
default_index = len(labels)-1 if labels else 0
choice = st.sidebar.selectbox("Select a plan", labels, index=default_index)
sel_plan_id = options[choice]

# Load the selected plan blob
with st.spinner(f"Fetching plan details..."):
    sel_plan = get_plan_fn.remote(sel_plan_id)

if sel_plan is None:
    st.error("Selected plan not found on storage.")
    st.stop()

# -------------------------------------------------------------------------
# SELECTED PLAN ADJUSTMENT & DISPLAY
# -------------------------------------------------------------------------
st.subheader(f"Plan Analysis: {sel_plan.get('date', 'n/a')}")

# Extract Original Params
orig_k = int(sel_plan.get("k", 20))
orig_amt = float(sel_plan.get("invest_amt", 1000.0))
orig_temp = float(sel_plan.get("temp", 1.0))

# -------------------
# ADJUSTABLE INPUTS
# -------------------
with st.expander("🛠️ Adjustment / Simulation Parameters", expanded=True):
    col_adj1, col_adj2, col_adj3 = st.columns(3)
    
    # Limit slider max to available rows in the data blob
    raw_rows = sel_plan.get("rows", [])
    max_available_rows = len(raw_rows)
    
    sim_k = col_adj1.slider("Top K (Subset)", min_value=1, max_value=max_available_rows, value=min(orig_k, max_available_rows))
    sim_amt = col_adj2.number_input("Investment Amount ($)", min_value=100.0, value=orig_amt, step=100.0)
    sim_temp = col_adj3.slider("Softmax Temperature", min_value=0.1, max_value=5.0, value=orig_temp, step=0.1)
    
    st.caption(f"Original Plan Settings: K={orig_k}, Amt=${orig_amt}, Temp={orig_temp}")

# -------------------
# RECALCULATE LOGIC
# -------------------
df_raw = pd.DataFrame(raw_rows)
if df_raw.empty:
    st.warning("No rows in plan.")
    st.stop()

# Perform simulation/recalculation locally
df_sim = recalculate_metrics(df_raw, sim_k, sim_temp, sim_amt)

# Fetch Live Prices
sim_tickers = df_sim["ticker"].dropna().unique().tolist()
live = _live_prices(sim_tickers)

# Compute P&L on Simulated Data
df_sim["current_price"] = df_sim["ticker"].map(live).astype(float)
df_sim["current_value"] = df_sim["shares"] * df_sim["current_price"]
df_sim["buy_value"] = df_sim["shares"] * df_sim["buy_price"]
df_sim["pnl_abs"] = df_sim["current_value"] - df_sim["buy_value"]
df_sim["pnl_pct"] = (df_sim["current_value"] / df_sim["buy_value"] - 1.0) * 100.0

totals = {
    "buy_value": float(np.nansum(df_sim["buy_value"])),
    "current_value": float(np.nansum(df_sim["current_value"])),
}
totals["pnl_abs"] = totals["current_value"] - totals["buy_value"]
totals["pnl_pct"] = (totals["current_value"] / totals["buy_value"] - 1.0) * 100.0 if totals["buy_value"] else np.nan

# -------------------
# DISPLAY TABLES
# -------------------
st.markdown("### Portfolio Performance (Based on Settings)")

kpi = st.columns(3)
def fmt_money(x): return f"${x:,.2f}"
def fmt_pct(x): return f"{x:.2f}%"

kpi[0].metric("Invested", fmt_money(totals["buy_value"]))
kpi[1].metric("Current Value", fmt_money(totals["current_value"]))
kpi[2].metric("P/L", fmt_money(totals["pnl_abs"]), fmt_pct(totals["pnl_pct"]))

view_cols = ["rank","ticker","score","weight","allocation","buy_price","shares","current_price","current_value","pnl_abs","pnl_pct"]
# Re-rank based on the sliced view
df_sim = df_sim.sort_values("weight", ascending=False).reset_index(drop=True)
df_sim["rank"] = df_sim.index + 1

df_view = df_sim.loc[:, [c for c in view_cols if c in df_sim.columns]].copy()

# Formatting for DataFrame
fmt_df = df_view.copy()
for col in ["allocation","buy_price","current_price","current_value","pnl_abs"]:
    if col in fmt_df:
        fmt_df[col] = fmt_df[col].map(lambda v: fmt_money(float(v)) if pd.notna(v) else "")
for col in ["score","weight","shares"]:
    if col in fmt_df:
        fmt_df[col] = fmt_df[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
if "pnl_pct" in fmt_df:
    fmt_df["pnl_pct"] = fmt_df["pnl_pct"].map(lambda v: fmt_pct(float(v)) if pd.notna(v) else "")

st.dataframe(fmt_df, use_container_width=True, hide_index=True)
st.caption(f"Last refreshed: {mel_now().isoformat()} ({TZ_NAME})")

st.divider()

# -------------------------------------------------------------------------
# HISTORICAL TIMELINE & BENCHMARK
# -------------------------------------------------------------------------
st.header("📅 Historical Performance vs S&P 500")
st.caption("Calculated using the **adjusted** Top-K, Amount, and Temp settings from above. If multiple plans exist for one day (e.g., from testing), only the **latest** one is used.")

if st.button("Generate Historical Timeline Analysis"):
    with st.spinner("Processing all historical plans with current simulation settings..."):
        
        # 1. Gather all plan data
        history_data = []
        all_plan_tickers = set()

        # Sort plans by date
        sorted_plans = sorted(plans, key=lambda x: x.get("created_at_utc", ""))
        
        # DEDUPLICATION LOGIC: Keep only the latest plan for each 'date' string
        deduped_map = {}
        for p in sorted_plans:
            d_str = p.get("date")
            if d_str:
                deduped_map[d_str] = p
        
        # Convert back to sorted list based on date string
        # Assuming YYYY-MM-DD format, sorting strings sorts dates correctly
        final_plans = sorted(deduped_map.values(), key=lambda x: x.get("date", ""))
        
        # Pre-fetch SPY history
        spy_hist = _get_spy_history() # Series with DatetimeIndex
        current_spy = spy_hist.iloc[-1] if not spy_hist.empty else 0.0

        # Create a progress bar
        prog_bar = st.progress(0)
        
        for i, p_meta in enumerate(final_plans):
            pid = p_meta["plan_id"]
            p_date_str = p_meta.get("date")
            
            # Fetch full plan
            blob = get_plan_fn.remote(pid)
            if not blob: continue
            
            rows = blob.get("rows", [])
            if not rows: continue
            
            d_rows = pd.DataFrame(rows)
            # Ensure necessary columns are numeric
            d_rows["buy_price"] = pd.to_numeric(d_rows["buy_price"], errors="coerce").fillna(0)
            d_rows["score"]     = pd.to_numeric(d_rows["score"], errors="coerce")

            # CRITICAL FIX: Recalculate metrics for history using current sliders
            # Instead of using stored "shares" and "allocation", we re-run logic.
            d_sim = recalculate_metrics(d_rows, sim_k, sim_temp, sim_amt)
            
            # Identify tickers needed for this plan (only the top K selected)
            plan_tickers = d_sim["ticker"].unique().tolist()
            all_plan_tickers.update(plan_tickers)
            
            history_data.append({
                "date": p_date_str,
                "created_at": p_meta.get("created_at_melbourne"),
                "df": d_sim,
                "invested": d_sim["shares"] * d_sim["buy_price"]
            })
            
            prog_bar.progress((i + 1) / len(final_plans))

        # 2. Fetch Live Prices for ALL unique tickers found across all history
        live_prices_all = _live_prices(list(all_plan_tickers))
        
        # 3. Compute Returns
        plot_points = []
        
        for item in history_data:
            df_h = item["df"]
            # Current value using live prices
            df_h["curr_p"] = df_h["ticker"].map(live_prices_all).astype(float)
            curr_val = np.nansum(df_h["shares"] * df_h["curr_p"])
            orig_val = np.nansum(item["invested"])
            
            if orig_val > 0:
                model_ret_pct = (curr_val / orig_val - 1.0) * 100.0
            else:
                model_ret_pct = 0.0
                
            # Benchmark Return
            spy_ret_pct = 0.0
            try:
                p_dt = pd.to_datetime(item["date"])
                idx_loc = spy_hist.index.get_indexer([p_dt], method='nearest')[0]
                if idx_loc >= 0:
                    spy_start_price = spy_hist.iloc[idx_loc]
                    spy_ret_pct = (current_spy / spy_start_price - 1.0) * 100.0
            except Exception as e:
                pass
                
            plot_points.append({
                "date": item["date"],
                "Model Return": model_ret_pct,
                "S&P 500 Return": spy_ret_pct,
                "Details": f"Date: {item['date']}<br>Invested: ${orig_val:,.0f}<br>Current: ${curr_val:,.0f}"
            })
            
        # 4. Plot
        if plot_points:
            df_plot = pd.DataFrame(plot_points)
            
            fig = go.Figure()
            
            # Bar chart for Model
            fig.add_trace(go.Bar(
                x=df_plot["date"],
                y=df_plot["Model Return"],
                name="Model Plan Return",
                marker_color='indianred',
                hovertemplate="%{text}<br>Model Return: %{y:.2f}%",
                text=df_plot["Details"],
                textposition='none' # Hide text on bars to prevent squashing
            ))
            
            # Line/Scatter for SPY
            fig.add_trace(go.Scatter(
                x=df_plot["date"],
                y=df_plot["S&P 500 Return"],
                name="S&P 500 (Benchmark)",
                mode='lines+markers',
                line=dict(color='royalblue', width=2),
                marker=dict(size=6),
                hovertemplate="SPY Return: %{y:.2f}%"
            ))
            
            fig.update_layout(
                title="Performance of Each Plan (Adjusted) vs S&P 500",
                xaxis_title="Plan Creation Date",
                yaxis_title="Total Return (%)",
                legend_title="Legend",
                hovermode="x unified",
                xaxis=dict(
                    tickformat="%b %d", # Concise date format (e.g. "Oct 12")
                    dtick="D1"          # Attempt to tick every day, plotly auto-hides if too crowded
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("No data points generated.")

st.markdown("#### Notes")
st.markdown("""
- A new plan is created **once per day** at ~9:00 in Australia/Melbourne time.
- **Buy prices** are snapshot on Modal when the plan was first created.
- **Top K, Temp, and Amount** used in the timeline above are the ones **currently selected** in the Simulation panel, applied retrospectively.
- Not financial advice.
""")
