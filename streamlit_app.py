# streamlit_app.py
# Streamlit Cloud UI for your Modal-based "investment plans".
# - On startup: ensure there's a plan for today (>= 9am Australia/Melbourne). If none, create once.
# - Every run: show a dropdown of all plans (oldest -> latest), default to latest.
# - For the selected plan: Allows adjusting K, Amount, Temp, AND Strategy (Softmax vs Uniform).
# - Bottom section: Historical timeline with Date Filter, Aggregate Stats, and Detailed Holdings (w/ Finnhub).
#   (Automatically generated, no button click required).
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
from typing import Dict, Any, List, Set
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
    st.error("`yfinance` is not installed. Add it to requirements.txt.")
    st.stop()

# Finnhub import
try:
    import finnhub
except ImportError:
    st.error("`finnhub-python` is not installed. Add it to requirements.txt.")
    st.stop()

from modal import Function, App

# -----------------------------
# Config / Secrets
# -----------------------------
APP_INFER = os.getenv("MODAL_APP_INFER", "inmemory-latest-infer")
APP_PLANS = os.getenv("MODAL_APP_PLANS", "inmemory-latest-plans")

CKPT_PATH   = os.getenv("CKPT_PATH", "/vol/models/ckpts/biex_listmle_final.pt")
# THIS REMAINS 20 for plan creation as requested
TOP_K       = int(os.getenv("TOP_K", "20")) 
INVEST_AMT  = float(os.getenv("INVEST_AMT", "1000"))
TEMP        = float(os.getenv("TEMP", "2.0"))
TZ_NAME     = os.getenv("TIMEZONE", "Australia/Melbourne")
AUTO_CREATE = os.getenv("AUTO_CREATE_ON_START", "1") == "1"
FINNHUB_KEY = "d2bg5c9r01qrj4ilmpb0d2bg5c9r01qrj4ilmpbg"

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
    """Fetches SPY history to use for benchmarking."""
    try:
        spy = yf.Ticker("SPY")
        # Fetch 5y to be safe
        hist = spy.history(period="5y", interval="1d")
        # IMPORTANT: Remove timezone info to match naive dates from plans
        if not hist.empty:
            hist.index = hist.index.tz_localize(None)
        return hist["Close"]
    except:
        return pd.Series(dtype=float)

@st.cache_data(ttl=3600)  # Cache longer to avoid hitting Finnhub rate limits
def _get_analyst_trends(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch analyst recommendation trends from Finnhub."""
    # Free tier rate limit is ~60 calls/minute. We need to be careful.
    results = {}
    if not tickers:
        return results
        
    finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)
    
    # Process unique tickers only
    unique_tickers = list(set(tickers))
    
    # We'll do a simple loop. 
    for t in unique_tickers:
        try:
            trends = finnhub_client.recommendation_trends(t)
            if trends and isinstance(trends, list):
                # SAFE SORT: Ensure we get the latest period "YYYY-MM-DD"
                trends.sort(key=lambda x: x.get("period", ""), reverse=True)
                latest = trends[0]
                results[t] = {
                    "strongBuy": latest.get("strongBuy", 0),
                    "buy": latest.get("buy", 0),
                    "hold": latest.get("hold", 0),
                    "sell": latest.get("sell", 0),
                    "strongSell": latest.get("strongSell", 0),
                    "period": latest.get("period", "")
                }
            # Respect rate limit gently
            time.sleep(0.1) 
        except Exception as e:
            pass
            
    return results

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

def recalculate_metrics(df_rows: pd.DataFrame, top_k: int, temp: float, invest_amt: float, strategy: str = "Softmax"):
    """
    Recalculates weights, allocations, and shares based on simulation parameters.
    Assumes df_rows has 'score' (raw logits) and 'buy_price'.
    strategy: "Softmax" (uses temp) or "Uniform" (1/k).
    """
    # 1. Sort by score descending and take Top K
    df_sim = df_rows.sort_values(by="score", ascending=False).head(top_k).copy()
    actual_k = len(df_sim)
    
    if strategy == "Uniform":
        # Equal weight = 1 / K
        if actual_k > 0:
            weight = 1.0 / actual_k
        else:
            weight = 0.0
        df_sim["weight"] = weight
        
    else:
        # Softmax formula: exp(x/T) / sum(exp(x/T))
        scores = df_sim["score"].values.astype(float)
        if temp <= 0: temp = 0.01 
        
        scaled_scores = scores / temp
        max_s = np.max(scaled_scores) if actual_k > 0 else 0
        exps = np.exp(scaled_scores - max_s)
        sum_exps = np.sum(exps)
        weights = exps / sum_exps if sum_exps > 0 else 0
        
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
raw_rows = sel_plan.get("rows", [])
max_available_rows = len(raw_rows)

# -------------------
# ADJUSTABLE INPUTS
# -------------------
with st.expander("🛠️ Adjustment / Simulation Parameters", expanded=True):
    # Strategy Mode
    sim_strategy = st.radio(
        "Allocation Strategy", 
        ["Softmax", "Uniform"], 
        index=1, # Default to Uniform as requested
        horizontal=True,
        help="Softmax uses score & temperature to weight stocks. Uniform gives equal money to all Top K stocks."
    )
    
    col_adj1, col_adj2, col_adj3 = st.columns(3)
    
    # Default to 5 as requested, capped by available rows
    sim_k = col_adj1.slider("Top K (Subset)", min_value=1, max_value=max_available_rows, value=min(5, max_available_rows))
    sim_amt = col_adj2.number_input("Investment Amount ($)", min_value=100.0, value=orig_amt, step=100.0)
    
    # Only show Temp slider if Softmax is chosen
    if sim_strategy == "Softmax":
        sim_temp = col_adj3.slider("Softmax Temperature", min_value=0.1, max_value=5.0, value=orig_temp, step=0.1)
    else:
        sim_temp = 1.0 # Default dummy value, unused in Uniform
        col_adj3.markdown("*(Temp ignored in Uniform mode)*")

# -------------------
# HIGHLIGHTING SETTINGS
# -------------------
with st.expander("🎨 Highlighting / Momentum Settings", expanded=False):
    st.caption("Customize the definition of 'Red' and 'Yellow' flags in the Detailed Holdings table.")
    
    hl_cols = st.columns(3)
    
    # User can adjust what "Top N" means for the highlighting logic
    hl_top_n = hl_cols[0].slider("Elite Rank Threshold (Top N)", min_value=1, max_value=max_available_rows, value=20, help="The fixed rank cut-off used to determine if a stock is 'Elite' (Red/Yellow logic).")
    
    # Red Lookback: How many past plans to check
    hl_red_lookback = hl_cols[1].slider("Red Warning Lookback (Days)", min_value=1, max_value=10, value=3, help="If a stock is not in the Top N of the last X plans, it turns RED.")
    
    # Yellow Lookback: How many *latest* plans count as 'Active'
    # Default 1 means "Latest Plan". 2 means "Latest or Yesterday".
    hl_active_lookback = hl_cols[2].slider("Active/White Window (Days)", min_value=1, max_value=hl_red_lookback, value=1, help="If a stock is in the Top N of these most recent Y plans, it stays WHITE (Active). If it falls out of this window but is still in the Red window, it turns YELLOW.")


# -------------------
# RECALCULATE LOGIC
# -------------------
df_raw = pd.DataFrame(raw_rows)
if df_raw.empty:
    st.warning("No rows in plan.")
    st.stop()

# Perform simulation/recalculation locally
df_sim = recalculate_metrics(df_raw, sim_k, sim_temp, sim_amt, strategy=sim_strategy)

# Fetch Live Prices
sim_tickers = df_sim["ticker"].dropna().unique().tolist()
live = _live_prices(sim_tickers)

# FETCH ANALYST TRENDS FOR SINGLE PLAN (NEW)
with st.spinner("Fetching Analyst Recommendation Trends..."):
    finnhub_single_data = _get_analyst_trends(sim_tickers)

# Compute P&L on Simulated Data
df_sim["current_price"] = df_sim["ticker"].map(live).astype(float)
df_sim["current_value"] = df_sim["shares"] * df_sim["current_price"]
df_sim["buy_value"] = df_sim["shares"] * df_sim["buy_price"]
df_sim["pnl_abs"] = df_sim["current_value"] - df_sim["buy_value"]
df_sim["pnl_pct"] = (df_sim["current_value"] / df_sim["buy_value"] - 1.0) * 100.0

# Add Finnhub Data columns
df_sim["Strong Buy"] = df_sim["ticker"].map(lambda t: finnhub_single_data.get(t, {}).get("strongBuy", ""))
df_sim["Buy"]        = df_sim["ticker"].map(lambda t: finnhub_single_data.get(t, {}).get("buy", ""))
df_sim["Hold"]       = df_sim["ticker"].map(lambda t: finnhub_single_data.get(t, {}).get("hold", ""))
df_sim["Sell"]       = df_sim["ticker"].map(lambda t: finnhub_single_data.get(t, {}).get("sell", ""))
df_sim["Strong Sell"]= df_sim["ticker"].map(lambda t: finnhub_single_data.get(t, {}).get("strongSell", ""))

totals = {
    "buy_value": float(np.nansum(df_sim["buy_value"])),
    "current_value": float(np.nansum(df_sim["current_value"])),
}
totals["pnl_abs"] = totals["current_value"] - totals["buy_value"]
totals["pnl_pct"] = (totals["current_value"] / totals["buy_value"] - 1.0) * 100.0 if totals["buy_value"] else np.nan

# -------------------
# DISPLAY TABLES
# -------------------
st.markdown(f"### Portfolio Performance ({sim_strategy} Mode)")

kpi = st.columns(3)
def fmt_money(x): return f"${x:,.2f}"
def fmt_pct(x): return f"{x:.2f}%"

kpi[0].metric("Invested", fmt_money(totals["buy_value"]))
kpi[1].metric("Current Value", fmt_money(totals["current_value"]))
kpi[2].metric("P/L", fmt_money(totals["pnl_abs"]), fmt_pct(totals["pnl_pct"]))

view_cols = [
    "rank","ticker","score","weight","allocation","buy_price","shares",
    "current_price","current_value","pnl_abs","pnl_pct",
    "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell" # Added these
]
# Re-rank based on the sliced view (sorting by weight still works for uniform, rank matches index)
df_sim = df_sim.sort_values(["weight", "score"], ascending=[False, False]).reset_index(drop=True)
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

# 0. Preparation & Deduplication
sorted_plans = sorted(plans, key=lambda x: x.get("created_at_utc", ""))
deduped_map = {}
for p in sorted_plans:
    d_str = p.get("date")
    if d_str:
        deduped_map[d_str] = p
# Final plans sorted by date
final_plans = sorted(deduped_map.values(), key=lambda x: x.get("date", ""))

if not final_plans:
    st.stop()

# 1. Date Range Slider
min_date = datetime.strptime(final_plans[0]["date"], "%Y-%m-%d").date()
max_date = datetime.strptime(final_plans[-1]["date"], "%Y-%m-%d").date()

# If only one date, slider crashes, so check
if min_date == max_date:
    start_date = min_date
else:
    start_date = st.slider(
        "Select Simulation Start Date:",
        min_value=min_date,
        max_value=max_date,
        value=min_date,
        format="YYYY-MM-DD"
    )

# Filter plans based on slider
active_plans = [p for p in final_plans if datetime.strptime(p["date"], "%Y-%m-%d").date() >= start_date]

st.caption(f"Simulating investment from **{start_date}** to **{max_date}** ({len(active_plans)} trading days).")
st.caption(f"Using Strategy: **{sim_strategy}**, K={sim_k}, Amt=${sim_amt}.")

# Automatically generated logic (No Button)
with st.spinner("Processing plans, calculating holdings, and fetching market data..."):
    
    # --- A. PRE-CALCULATE 'ACTIVE' UNIVERSES FOR HIGHLIGHTING ---
    # DEFINITION OF ELITE: Fixed Top N (hl_top_n) from Slider
    
    # 1. Broad Window (Red Logic): last 'hl_red_lookback' plans
    universe_broad_window: Set[str] = set()
    
    # 2. Active Window (Yellow/White Logic): last 'hl_active_lookback' plans
    universe_active_window: Set[str] = set()
    
    # We need the last X plans from the FULL list (final_plans), not just active_plans range
    total_count = len(final_plans)
    
    # Identify start indices for slices
    start_idx_broad = max(0, total_count - hl_red_lookback)
    start_idx_active = max(0, total_count - hl_active_lookback)
    
    broad_slice = final_plans[start_idx_broad:]
    active_slice = final_plans[start_idx_active:]
    
    # Helper to extract Top N set
    def get_top_n_set(plan_metas, n):
        s = set()
        for pm in plan_metas:
            blob = get_plan_fn.remote(pm["plan_id"])
            if not blob: continue
            raw = pd.DataFrame(blob.get("rows", []))
            if raw.empty: continue
            # Sort by raw score to get independent elite ranking
            raw_sorted = raw.sort_values(by="score", ascending=False).head(n)
            s.update(raw_sorted["ticker"].tolist())
        return s
        
    universe_broad_window = get_top_n_set(broad_slice, hl_top_n)
    universe_active_window = get_top_n_set(active_slice, hl_top_n)

    # --- B. PROCESS ACTIVE PLANS FOR TIMELINE & HOLDINGS ---
    
    history_data = []
    all_tickers_involved = set()
    
    # Accumulator for holdings: {ticker: {'shares': 0.0, 'invested': 0.0}}
    portfolio_holdings = {}

    spy_hist = _get_spy_history()
    current_spy = spy_hist.iloc[-1] if not spy_hist.empty else 0.0

    prog_bar = st.progress(0)
    
    for i, p_meta in enumerate(active_plans):
        pid = p_meta["plan_id"]
        blob = get_plan_fn.remote(pid)
        if not blob: continue
        
        rows = blob.get("rows", [])
        if not rows: continue
        
        d_rows = pd.DataFrame(rows)
        d_rows["buy_price"] = pd.to_numeric(d_rows["buy_price"], errors="coerce").fillna(0)
        d_rows["score"]     = pd.to_numeric(d_rows["score"], errors="coerce")

        # Apply Simulation Settings (With Strategy and user's chosen K)
        d_sim = recalculate_metrics(d_rows, sim_k, sim_temp, sim_amt, strategy=sim_strategy)
        
        plan_tickers = d_sim["ticker"].unique().tolist()
        all_tickers_involved.update(plan_tickers)
        
        # Update Portfolio Holdings
        for row in d_sim.itertuples():
            if row.shares > 0:
                if row.ticker not in portfolio_holdings:
                    portfolio_holdings[row.ticker] = {'shares': 0.0, 'invested': 0.0}
                portfolio_holdings[row.ticker]['shares'] += row.shares
                portfolio_holdings[row.ticker]['invested'] += (row.shares * row.buy_price)

        history_data.append({
            "date": p_meta.get("date"),
            "df": d_sim,
            "invested": d_sim["shares"] * d_sim["buy_price"]
        })
        
        prog_bar.progress((i + 1) / len(active_plans))

    # --- C. FETCH LIVE PRICES & FINNHUB DATA ---
    all_tickers_involved.update(portfolio_holdings.keys())
    live_prices_all = _live_prices(list(all_tickers_involved))
    finnhub_data_all = _get_analyst_trends(list(all_tickers_involved))
    
    # --- D. TIMELINE PLOT DATA ---
    plot_points = []
    agg_invested = 0.0
    agg_model_val = 0.0
    agg_spy_val = 0.0
    
    for item in history_data:
        df_h = item["df"]
        df_h["curr_p"] = df_h["ticker"].map(live_prices_all).astype(float)
        curr_val = np.nansum(df_h["shares"] * df_h["curr_p"])
        orig_val = np.nansum(item["invested"])
        
        agg_invested += orig_val
        agg_model_val += curr_val
        
        model_ret_pct = (curr_val / orig_val - 1.0) * 100.0 if orig_val > 0 else 0.0
        
        # Benchmark
        spy_ret_pct = 0.0
        spy_val_now = orig_val
        try:
            p_dt = pd.to_datetime(item["date"])
            idx_loc = spy_hist.index.get_indexer([p_dt], method='nearest')[0]
            if idx_loc >= 0:
                spy_start_price = spy_hist.iloc[idx_loc]
                spy_ret_pct = (current_spy / spy_start_price - 1.0) * 100.0
                spy_val_now = orig_val * (current_spy / spy_start_price)
        except:
            pass
        
        agg_spy_val += spy_val_now
        
        plot_points.append({
            "date": item["date"],
            "Model Return": model_ret_pct,
            "S&P 500 Return": spy_ret_pct,
            "Details": f"Date: {item['date']}<br>Invested: ${orig_val:,.0f}<br>Current: ${curr_val:,.0f}"
        })
        
    # --- E. PLOTTING ---
    if plot_points:
        df_plot = pd.DataFrame(plot_points)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_plot["date"], y=df_plot["Model Return"], name="Model Plan Return",
            marker_color='indianred', hovertemplate="%{text}<br>Model: %{y:.2f}%",
            text=df_plot["Details"], textposition='none'
        ))
        fig.add_trace(go.Scatter(
            x=df_plot["date"], y=df_plot["S&P 500 Return"], name="S&P 500",
            mode='lines+markers', line=dict(color='royalblue', width=2), marker=dict(size=6)
        ))
        fig.update_layout(
            title=f"Daily Plan Performance ({start_date} to {max_date}) - {sim_strategy} Mode",
            xaxis_title="Date", yaxis_title="Return (%)", hovermode="x unified",
            xaxis=dict(tickformat="%b %d", dtick="D1")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- F. AGGREGATE STATS ---
        st.markdown("### Aggregate Strategy Performance")
        agg_cols = st.columns(4)
        model_pnl = agg_model_val - agg_invested
        model_pct = (model_pnl / agg_invested * 100.0) if agg_invested > 0 else 0.0
        spy_pnl = agg_spy_val - agg_invested
        spy_pct = (spy_pnl / agg_invested * 100.0) if agg_invested > 0 else 0.0
        
        agg_cols[0].metric("Total Capital Invested", fmt_money(agg_invested))
        agg_cols[1].metric("Current Value (Model)", fmt_money(agg_model_val))
        agg_cols[2].metric("Total Profit (Model)", fmt_money(model_pnl), fmt_pct(model_pct))
        agg_cols[3].metric("Total Profit (S&P 500)", fmt_money(spy_pnl), fmt_pct(spy_pct))

    # --- G. DETAILED HOLDINGS TABLE ---
    st.divider()
    st.subheader("📦 Detailed Portfolio Holdings")
    st.caption(f"Aggregated holdings from the selected start date. Logic for highlighting (based on **Fixed Top {hl_top_n}**):")
    
    st.markdown(f"""
    - <span style='background-color: #ffcccc; padding: 2px 4px; border-radius: 4px; color: black;'>Red</span>: Stock is **NOT** in the Fixed Top {hl_top_n} of the last **{hl_red_lookback}** available plans.
    - <span style='background-color: #fff9c4; padding: 2px 4px; border-radius: 4px; color: black;'>Yellow</span>: Stock is **NOT** in the Fixed Top {hl_top_n} of the last **{hl_active_lookback}** plan(s) (but is in the last {hl_red_lookback}).
    - **White**: Stock is currently in the Fixed Top {hl_top_n} of the last **{hl_active_lookback}** plan(s).
    """, unsafe_allow_html=True)
    
    if portfolio_holdings:
        holdings_list = []
        for t, data in portfolio_holdings.items():
            curr_p = live_prices_all.get(t, 0.0)
            shares = data['shares']
            invested = data['invested']
            curr_val = shares * curr_p
            pnl = curr_val - invested
            pnl_pct = (pnl / invested * 100.0) if invested > 0 else 0.0
            
            # Analyst Data
            a_data = finnhub_data_all.get(t, {})
            
            # Determine Status for Coloring using the FIXED Top N sets
            if t not in universe_broad_window:
                status = "Sell/Drop (Red)"
            elif t not in universe_active_window:
                status = "Warning (Yellow)"
            else:
                status = "Active (Green)"
            
            holdings_list.append({
                "Ticker": t,
                "Total Shares": shares,
                "Total Invested": invested,
                "Current Price": curr_p,
                "Current Value": curr_val,
                "P&L ($)": pnl,
                "P&L (%)": pnl_pct,
                "Status": status,
                "Strong Buy": a_data.get("strongBuy", ""),
                "Buy": a_data.get("buy", ""),
                "Hold": a_data.get("hold", ""),
                "Sell": a_data.get("sell", ""),
                "Strong Sell": a_data.get("strongSell", "")
            })
        
        df_holdings = pd.DataFrame(holdings_list)
        
        # Styling function
        def highlight_status(row):
            s = row["Status"]
            if s == "Sell/Drop (Red)":
                return ['background-color: #ffcccc; color: black'] * len(row)
            elif s == "Warning (Yellow)":
                return ['background-color: #fff9c4; color: black'] * len(row)
            else:
                return [''] * len(row)

        # Apply formatting
        df_styled = df_holdings.style.apply(highlight_status, axis=1).format({
            "Total Shares": "{:.4f}",
            "Total Invested": "${:,.2f}",
            "Current Price": "${:,.2f}",
            "Current Value": "${:,.2f}",
            "P&L ($)": "${:,.2f}",
            "P&L (%)": "{:.2f}%"
        })
        
        st.dataframe(df_styled, use_container_width=True, hide_index=True)
    else:
        st.info("No holdings found for this period.")

st.markdown("#### Notes")
st.markdown("""
- A new plan is created **once per day** at ~9:00 in Australia/Melbourne time.
- **Top K, Temp, and Amount** are simulation parameters applied to historical raw scores.
- **Red/Yellow highlights** indicate momentum shifts based on the **Fixed Top N** rankings, separate from your investment simulation.
- Not financial advice.
""")
