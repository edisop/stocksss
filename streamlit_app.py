# streamlit_app.py
# Streamlit Cloud UI for your Modal-based "investment plans".
# - On startup: ensure there's a plan for today (>= 9am Australia/Melbourne). If none, create once.
# - Every run: show a dropdown of all plans (oldest -> latest), default to latest.
# - For the selected plan, fetch live prices via yfinance and compute P&L vs buy prices.
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
from datetime import datetime, date
from typing import Dict, Any, List
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import streamlit as st

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
    import streamlit as st
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
@st.cache_data(ttl=30)
def _live_prices(tickers: List[str]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    if not tickers:
        return prices
    # Try fast_info first
    for t in tickers:
        try:
            fi = yf.Ticker(t).fast_info
            p = fi.get("last_price") or fi.get("last_trade") or fi.get("last_close")
            if p:
                prices[t] = float(p)
        except Exception:
            pass
    # Fill using 1d/1m history
    missing = [t for t in tickers if t not in prices]
    if missing:
        try:
            df = yf.download(missing, period="1d", interval="1m", progress=False, group_by="ticker", threads=True)
            if isinstance(df.columns, pd.MultiIndex):
                for t in missing:
                    try:
                        last = float(df[t]["Close"].dropna().iloc[-1])
                        prices[t] = last
                    except Exception:
                        pass
        except Exception:
            pass
    # Final fill using 5d daily
    missing = [t for t in tickers if t not in prices]
    if missing:
        try:
            df = yf.download(missing, period="5d", interval="1d", progress=False, group_by="ticker", threads=True)
            if isinstance(df.columns, pd.MultiIndex):
                for t in missing:
                    try:
                        last = float(df[t]["Close"].dropna().iloc[-1])
                        prices[t] = last
                    except Exception:
                        pass
        except Exception:
            pass
    return prices

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

# -----------------------------
# Load plans and maybe auto-create
# -----------------------------
with st.spinner("Loading plans..."):
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
sel_plan = get_plan_fn.remote(sel_plan_id)
if sel_plan is None:
    st.error("Selected plan not found on storage.")
    st.stop()

st.subheader(f"Plan {sel_plan['plan_id']} — Date {sel_plan.get('date','n/a')}")
meta_cols = st.columns(4)
meta_cols[0].metric("K (top-n)", sel_plan.get("k"))
meta_cols[1].metric("Invest Amount", f"${sel_plan.get('invest_amt',0):,.2f}")
meta_cols[2].metric("Softmax Temp", f"{sel_plan.get('temp',0):.2f}")
meta_cols[3].metric("Created (Melbourne)", sel_plan.get("created_at_melbourne", "n/a"))

rows = sel_plan.get("rows", [])
df = pd.DataFrame(rows)

if df.empty:
    st.warning("No rows in plan.")
    st.stop()

tickers = df["ticker"].dropna().unique().tolist()
live = _live_prices(tickers)

# Compute current values and P&L
df["buy_price"] = pd.to_numeric(df["buy_price"], errors="coerce")
df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
df["allocation"] = pd.to_numeric(df["allocation"], errors="coerce")

df["current_price"] = df["ticker"].map(live).astype(float)
df["current_value"] = df["shares"] * df["current_price"]
df["buy_value"] = df["shares"] * df["buy_price"]
df["pnl_abs"] = df["current_value"] - df["buy_value"]
df["pnl_pct"] = (df["current_value"] / df["buy_value"] - 1.0) * 100.0

totals = {
    "buy_value": float(np.nansum(df["buy_value"])),
    "current_value": float(np.nansum(df["current_value"])),
}
totals["pnl_abs"] = totals["current_value"] - totals["buy_value"]
totals["pnl_pct"] = (totals["current_value"] / totals["buy_value"] - 1.0) * 100.0 if totals["buy_value"] else np.nan

# Display
st.markdown("### Live Portfolio View")
st.caption("Live prices via yfinance (cached ~30s). Buy prices fixed at plan creation on Modal.")

view_cols = ["rank","ticker","item_id","score","weight","allocation","buy_price","shares","current_price","current_value","pnl_abs","pnl_pct"]
df_view = df.loc[:, [c for c in view_cols if c in df.columns]].copy()

# Nice formatting
def fmt_money(x): 
    return "" if pd.isna(x) else f"${x:,.2f}"
def fmt_float(x, n=4):
    return "" if pd.isna(x) else f"{x:.{n}f}"
def fmt_pct(x):
    return "" if pd.isna(x) else f"{x:.2f}%"

fmt_df = df_view.copy()
for col in ["allocation","buy_price","current_price","current_value","pnl_abs"]:
    if col in fmt_df:
        fmt_df[col] = fmt_df[col].map(lambda v: fmt_money(float(v)) if pd.notna(v) else "")
for col in ["score","weight","shares"]:
    if col in fmt_df:
        fmt_df[col] = fmt_df[col].map(lambda v: fmt_float(float(v)) if pd.notna(v) else "")
if "pnl_pct" in fmt_df:
    fmt_df["pnl_pct"] = fmt_df["pnl_pct"].map(lambda v: fmt_pct(float(v)) if pd.notna(v) else "")

st.dataframe(fmt_df, use_container_width=True, hide_index=True)

# Totals
kpi = st.columns(3)
kpi[0].metric("Invested (buy_value)", fmt_money(totals["buy_value"]))
kpi[1].metric("Current Value", fmt_money(totals["current_value"]))
kpi[2].metric("P/L", fmt_money(totals["pnl_abs"]), fmt_pct(totals["pnl_pct"]))

st.caption(f"Last refreshed: {mel_now().isoformat()} ({TZ_NAME})")

st.divider()
st.markdown("#### Notes")
st.markdown("""
- A new plan is created **once per day** at ~9:00 in Australia/Melbourne time (if none exists for today).
- When a plan is created, **buy prices** are snapshot on Modal and share quantities are computed as `allocation / buy_price`.
- The table above uses **live prices** to compute current value and P/L. Numbers may differ slightly from brokerage fills.
- Tickers with missing prices are skipped or show blanks.
- Not financial advice.
""")
