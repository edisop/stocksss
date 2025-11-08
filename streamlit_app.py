# streamlit_app.py
# Streamlit UI that talks to the deployed Modal app *via the Modal Python SDK* (no HTTP endpoints).
# Optionally, on startup it can auto-deploy the Modal app by writing the embedded source to /tmp
# and running `modal deploy`. This keeps everything fully cloud.
#
# How it works:
#   - On first run, it can deploy the Modal app (toggle via AUTO_DEPLOY env var or sidebar button).
#   - Then it uses `modal.Function.from_name(APP_NAME, fn).remote(...)` to call:
#         - list_plans()
#         - get_plan_with_live(id=None|<plan_id>)
#   - The Modal app (GPU/CPU) does all heavy lifting and plan storage.
#
# Secrets you can set in Streamlit Cloud:
#   MODAL_TOKEN_ID, MODAL_TOKEN_SECRET   -> Modal API keys (or hardcode below)
#   MODAL_APP_NAME                       -> defaults to "inmemory-latest-infer"
#   AUTO_DEPLOY                          -> "1" to deploy the embedded modal app on start
#
# Hardcode tokens (optional; not recommended in public repos):
# MODAL_TOKEN_ID_HARDCODE = "YOUR_TOKEN_ID"
# MODAL_TOKEN_SECRET_HARDCODE = "YOUR_TOKEN_SECRET"

from __future__ import annotations
import os, subprocess, tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd

# ---- Optional: hardcode tokens here (fallback if env/secrets missing) ----
MODAL_TOKEN_ID_HARDCODE = os.getenv("MODAL_TOKEN_ID_HARDCODE", "ak-FEiBFqRU5O7Ka5uw5dAb1U")
MODAL_TOKEN_SECRET_HARDCODE = os.getenv("MODAL_TOKEN_SECRET_HARDCODE", "as-Rzw36potqKmoJ9RDugsM9O")

# ---- Modal SDK setup ----
import modal
# Prefer Streamlit secrets / env vars
_token_id = os.getenv("MODAL_TOKEN_ID") or MODAL_TOKEN_ID_HARDCODE
_token_secret = os.getenv("MODAL_TOKEN_SECRET") or MODAL_TOKEN_SECRET_HARDCODE
if _token_id and _token_secret:
    # Make sure the SDK sees the creds
    os.environ["MODAL_TOKEN_ID"] = _token_id
    os.environ["MODAL_TOKEN_SECRET"] = _token_secret

APP_NAME = os.getenv("MODAL_APP_NAME", "inmemory-latest-infer")

# ---- Embedded Modal app source (exact copy) ----
MODAL_APP_SOURCE = """# modal_infer_inmemory_latest.py
# Modalized "latest-day" inference + plan storage + web endpoints for Streamlit
# - CPU function builds texts in memory (no files)
# - GPU function loads your ckpt from a Modal Volume and scores items
# - "Plan" = top-k allocation with buy prices & quantities recorded at creation
# - Daily cron at 9am Australia/Melbourne creates a new plan
# - Web endpoints:
#     GET /list_plans              -> list available plans (oldest -> latest)
#     GET /get_plan_with_live      -> latest plan with live prices & P&L
#     GET /get_plan_with_live?id=PLAN_ID -> chosen plan with live prices & P&L
#
# Volumes expected:
#   - model-cache (v2): contains your checkpoint at /ckpts/...
#   - code-cache  (v1): contains /pkgs/interfusion_encoder-3.1 or .whl
#   - plans-cache (v1): this script will create JSON files in /vol/plans
#
# Deploy:
#   modal deploy modal_infer_inmemory_latest.py
#
# Local smoke test:
#   modal run modal_infer_inmemory_latest.py
#   (then curl the printed web URLs)
from __future__ import annotations

import os, re, math, time, json, warnings
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

import modal

# ============================================================
#                === CHANGE-ME DEFAULTS (edit here) ===
# ============================================================
DEFAULT_CKPT: str = "/vol/models/ckpts/biex_listmle_final.pt"
DEFAULT_INVEST_AMT: float = 1000.0
DEFAULT_TOP_K: int = 20
DEFAULT_TEMP: float = 2.0
DEFAULT_MAX_CANDIDATES: int = 500
DEFAULT_OVERRIDE_DATE: str = ""   # "" -> auto rollover with price coverage

# ============================================================
#                 Finnhub keys (hard-coded)
# ============================================================
DEFAULT_FINNHUB_KEYS = [
    "d1v4p9pr01qo0ln2h99gd1v4p9pr01qo0ln2h9a0",
    "d2bg239r01qrj4ilm3l0d2bg239r01qrj4ilm3lg",
    "d2bg2e9r01qrj4ilm6agd2bg2e9r01qrj4ilm6b0",
    "d2bg2nhr01qrj4ilm8igd2bg2nhr01qrj4ilm8j0",
    "d2bg3f1r01qrj4ilme4gd2bg3f1r01qrj4ilme50",
    "d2bg3m9r01qrj4ilmfu0d2bg3m9r01qrj4ilmfug",
    "d2bg3upr01qrj4ilmhqgd2bg3upr01qrj4ilmhr0",
    "d2bg451r01qrj4ilmja0d2bg451r01qrj4ilmjag",
    "d2bg4vhr01qrj4ilmmp0d2bg4vhr01qrj4ilmmpg",
    "d2bg5c9r01qrj4ilmpb0d2bg5c9r01qrj4ilmpbg",
]

# -----------------------------
# Modal app, images, volumes
# -----------------------------
app = modal.App("inmemory-latest-infer")

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # build helpers
        "pip>=25.0", "setuptools>=70", "wheel>=0.43",
        # data + utils
        "pandas>=2.2", "numpy>=1.26", "tqdm>=4.66",
        # finance / APIs
        "yfinance>=0.2.50", "finnhub-python>=2.4.18",
        # models
        "transformers>=4.46", "tokenizers>=0.20",
    )
)

gpu_image = base_image.pip_install("torch==2.5.*")

MODEL_VOL = modal.Volume.from_name("model-cache", create_if_missing=True, version=2)
CODE_VOL  = modal.Volume.from_name("code-cache",  create_if_missing=False, version=1)
PLANS_VOL = modal.Volume.from_name("plans-cache", create_if_missing=True, version=1)

# -----------------------------
# Static config
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
NEWS_1D_HOURS = int(os.getenv("NEWS_1D_HOURS", "48"))
NEWS_7D_DAYS  = 7
RECENCY_HALFLIFE_DAYS = float(os.getenv("RECENCY_HALFLIFE_DAYS", "2.0"))
TICKER_MENTION_BONUS  = float(os.getenv("TICKER_MENTION_BONUS", "0.15"))
PER_SOURCE_LIMIT      = int(os.getenv("PER_SOURCE_LIMIT", "50"))
UTC_ROLLOVER_HOUR     = int(os.getenv("UTC_ROLLOVER_HOUR", "16"))
PRICE_LOOKBACK_DAYS   = 120
SLEEP_BETWEEN_NEWS_TICKERS = float(os.getenv("SLEEP_BETWEEN_NEWS_TICKERS", "0.10"))

# ============================================================
#               Helpers
# ============================================================
import pandas as pd
import numpy as np

_CLEAN_TAIL_RE   = re.compile(r"\s*(—|-)\s*[A-Z][A-Za-z&.\s]+$")
_PARENS_TAIL_RE  = re.compile(r"\s*\([^()]*\)\s*$")
_WS_RE           = re.compile(r"\s+")

def _clean_headline(h: str) -> str:
    if not isinstance(h, str): return ""
    s = h.strip()
    s = _CLEAN_TAIL_RE.sub("", s)
    s = _PARENS_TAIL_RE.sub("", s)
    s = s.strip(" .:;,-")
    s = _WS_RE.sub(" ", s)
    return s

def _norm_key(h: str) -> str:
    s = _clean_headline(h).lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return _WS_RE.sub(" ", s).strip()

def _normalize_news_datetime(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns: return df if df is not None else pd.DataFrame()
    df = df.copy()
    def _p(v):
        try:
            if isinstance(v, (int, np.integer)) or (isinstance(v, str) and v.isdigit()):
                return pd.to_datetime(int(v), unit="s", utc=True)
            return pd.to_datetime(v, utc=True, errors="coerce")
        except Exception:
            return pd.NaT
    df[col] = [_p(v) for v in df[col]]
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    return df.dropna(subset=[col]).sort_values(col)

def _dedup_news(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df if df is not None else pd.DataFrame()
    keys = [k for k in ("id","headline","datetime","source") if k in df.columns]
    if "id" in keys:
        return df.drop_duplicates(subset=["id"])
    if {"headline","datetime"}.issubset(df.columns):
        return df.drop_duplicates(subset=["headline","datetime"])
    return df.drop_duplicates()

def _pct(x0: float, x1: float) -> Optional[float]:
    try: return (x1 / x0 - 1.0) * 100.0
    except Exception: return None

def _recency_score(dt: pd.Timestamp, ref: pd.Timestamp) -> float:
    days = max(0.0, (ref - dt).total_seconds()) / 86400.0
    lam = math.log(2.0) / max(1e-6, RECENCY_HALFLIFE_DAYS)
    return math.exp(-lam * days)

def _select_diverse_by_source(df: pd.DataFrame, k: int, ref_dt: pd.Timestamp, ticker: Optional[str] = None) -> pd.DataFrame:
    if df is None or df.empty or k <= 0:
        return df.iloc[0:0] if isinstance(df, pd.DataFrame) else pd.DataFrame()
    x = df.copy()
    x["recency"] = x["datetime"].map(lambda d: _recency_score(pd.to_datetime(d), ref_dt))
    x["mention_bonus"] = 0.0 if not ticker else x["headline"].str.contains(re.escape(ticker), case=False, na=False).astype(float) * TICKER_MENTION_BONUS
    x["score"] = x["recency"] + x["mention_bonus"]
    x = x.sort_values(["score","datetime"], ascending=[False, False])

    src = "source" if "source" in x.columns else None
    if not src:
        return x.head(k)

    buckets: Dict[str, List[pd.Series]] = {}
    for _, row in x.iterrows():
        sname = row[src] if isinstance(row[src], str) and row[src] else "?"
        buckets.setdefault(sname, []).append(row)

    cap = max(1, min(PER_SOURCE_LIMIT, int(math.ceil(k / max(1, len(buckets))))))
    out = []
    for rows in buckets.values():
        rows_sorted = sorted(rows, key=lambda r: (r["score"], r["datetime"]), reverse=True)
        out.extend(rows_sorted[:cap])
    if len(out) < k:
        used = set(map(id, out))
        remaining = []
        for rows in buckets.values():
            for r in rows:
                if id(r) not in used:
                    remaining.append(r)
        remaining.sort(key=lambda r: (r["score"], r["datetime"]), reverse=True)
        out.extend(remaining[:k-len(out)])
    sel = pd.DataFrame(out).drop_duplicates(subset=["norm"]) if out else x.iloc[0:0]
    return sel.head(k)

def _window_time(df: pd.DataFrame, end_date: str, hours_1d: int, days_7d: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return df.iloc[0:0], df.iloc[0:0]
    end_dt = pd.to_datetime(end_date).normalize() + pd.Timedelta(days=1)
    start_1d = end_dt - pd.Timedelta(hours=hours_1d)
    start_7d = end_dt - pd.Timedelta(days=days_7d)
    w1 = df.loc[(df["datetime"] >= start_1d) & (df["datetime"] < end_dt)].copy()
    w7 = df.loc[(df["datetime"] >= start_7d) & (df["datetime"] < end_dt)].copy()
    if w1.empty and w7.empty:
        return w1, w7
    w1 = w1.drop_duplicates(subset=["norm"]).sort_values("datetime")
    n1 = set(w1["norm"].tolist())
    w7_excl = w7.loc[~w7["norm"].isin(n1)].drop_duplicates(subset=["norm"]).sort_values("datetime")
    return w1, w7_excl

def _market_macro_block(for_date: str, px: Dict[str, pd.DataFrame]) -> str:
    out = "--- Macro Indicators ---\n"
    d = pd.to_datetime(for_date)
    g = px.get("^GSPC"); v = px.get("^VIX")
    if g is not None and not g.empty:
        s = g.loc[g.index <= d]
        if s.shape[0] >= 2:
            pct1 = _pct(float(s["Close"].iloc[-2]), float(s["Close"].iloc[-1]))
            if pct1 is not None: out += f"S&P 500 (1D Change): {pct1:.2f}%\n"
        if s.shape[0] >= 7:
            s7 = s.iloc[-7:]
            pct7 = _pct(float(s7["Close"].iloc[0]), float(s7["Close"].iloc[-1]))
            if pct7 is not None: out += f"S&P 500 (7D Change): {pct7:.2f}%\n"
    if v is not None and not v.empty:
        s2 = v.loc[v.index <= d]
        if not s2.empty:
            out += f"VIX Level (as of {s2.index[-1].date()}): {float(s2['Close'].iloc[-1]):.2f}\n"
    return out

def _stock_indicators(ticker: str, for_date: str, px: Dict[str, pd.DataFrame]) -> str:
    hdr = f"--- Ticker Indicators ---\n"
    p = px.get(ticker); g = px.get("^GSPC")
    d = pd.to_datetime(for_date)
    if p is None or p.empty:
        return hdr + "(no price coverage)\n"
    s = p.loc[p.index <= d]
    if s.shape[0] >= 1:
        last_close = float(s["Close"].iloc[-1])
        hdr += f"Last Close: {last_close:.2f}\n"
    if s.shape[0] >= 2:
        d1 = _pct(float(s["Close"].iloc[-2]), float(s["Close"].iloc[-1]))
        if d1 is not None: hdr += f"1D Change: {d1:.2f}%\n"
    if s.shape[0] >= 7:
        s7 = s.iloc[-7:]
        d7 = _pct(float(s7["Close"].iloc[0]), float(s7["Close"].iloc[-1]))
        if d7 is not None: hdr += f"7D Change: {d7:.2f}%\n"
    if s.shape[0] >= 20:
        s20 = s.iloc[-20:]
        d20 = _pct(float(s20["Close"].iloc[0]), float(s20["Close"].iloc[-1]))
        if d20 is not None: hdr += f"20D Change: {d20:.2f}%\n"
        rets = np.diff(np.log(s20["Close"].values))
        if rets.size >= 2:
            vol20 = float(np.std(rets, ddof=1) * math.sqrt(252) * 100.0)
            hdr += f"20D Vol (ann.): {vol20:.2f}%\n"
    if g is not None and not g.empty and s.shape[0] >= 61 and g.shape[0] >= 61:
        rt = np.diff(np.log(s["Close"].iloc[-61:].values))
        rs = np.diff(np.log(g["Close"].iloc[-61:].values))
        if rt.size == rs.size and rt.size >= 10 and np.var(rs) > 0:
            beta = float(np.cov(rt, rs)[0,1] / np.var(rs))
            hdr += f"Beta (60D vs SPX): {beta:.2f}\n"
    return hdr

def _format_lines(headlines: List[str]) -> str:
    lines = [f"- {str(h).strip()}" for h in headlines if str(h).strip()]
    return "\n".join(lines) if lines else "None"

def _flatten_prices_multi(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if df is None or df.empty: return out
    if isinstance(df.columns, pd.MultiIndex):
        for ticker in sorted(set(lvl for lvl in df.columns.get_level_values(0) if isinstance(lvl, str))):
            sub = df[ticker].copy()
            sub.index = pd.to_datetime(sub.index)
            if "Close" not in sub.columns:
                if "Adj Close" in sub.columns:
                    sub = sub.rename(columns={"Adj Close": "Close"})
                else:
                    num_cols = [c for c in sub.columns if np.issubdtype(sub[c].dtype, np.number)]
                    if num_cols:
                        sub = sub.rename(columns={num_cols[-1]: "Close"})
            out[ticker] = sub.sort_index()
    else:
        dd = df.copy()
        dd.index = pd.to_datetime(dd.index)
        if "Close" not in dd.columns and "Adj Close" in dd.columns:
            dd = dd.rename(columns={"Adj Close": "Close"})
        out["SINGLE"] = dd.sort_index()
    return out

def _effective_utc_date_for_infer() -> str:
    now_utc = pd.Timestamp.utcnow()
    use = now_utc.date() if now_utc.hour >= UTC_ROLLOVER_HOUR else (now_utc - pd.Timestamp(days=1)).date()
    return pd.Timestamp(use).strftime("%Y-%m-%d")

# ---------- Plan storage helpers (on /vol/plans) ----------
def _plans_dir() -> str:
    p = "/vol/plans"
    os.makedirs(p, exist_ok=True)
    return p

def _plan_path(plan_id: str) -> str:
    return os.path.join(_plans_dir(), f"{plan_id}.json")

def _list_plan_files() -> List[str]:
    d = _plans_dir()
    try:
        return sorted([f for f in os.listdir(d) if f.endswith(".json")])
    except FileNotFoundError:
        return []

def _load_plan(plan_id: str) -> Optional[Dict[str, Any]]:
    path = _plan_path(plan_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_plan(plan: Dict[str, Any]) -> None:
    path = _plan_path(plan["plan_id"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f, separators=(",", ":"), ensure_ascii=False)

def _list_plans_meta() -> List[Dict[str, Any]]:
    out = []
    for fn in _list_plan_files():
        try:
            with open(os.path.join(_plans_dir(), fn), "r", encoding="utf-8") as f:
                p = json.load(f)
            out.append({
                "plan_id": p.get("plan_id"),
                "date": p.get("date"),
                "created_at": p.get("created_at"),
                "label": f"{p.get('date')} (created {p.get('created_at')})",
            })
        except Exception:
            continue
    out = [x for x in out if x.get("plan_id")]
    out.sort(key=lambda r: r.get("created_at") or "")
    return out

def _latest_plan_id() -> Optional[str]:
    metas = _list_plans_meta()
    if not metas:
        return None
    return metas[-1]["plan_id"]

# ============================================================
#                Remote: CPU fetcher (build texts)
# ============================================================
@app.function(image=base_image, timeout=60*20)
def build_payload_remote(
    for_date: str | None,
    tickers: List[str],
    max_candidates: int = 500,
) -> Dict[str, Any]:
    import yfinance as yf
    import finnhub
    from tqdm import tqdm as _tqdm

    date_eff = (for_date or "").strip() or _effective_utc_date_for_infer()

    start = (pd.to_datetime(date_eff) - pd.Timedelta(days=PRICE_LOOKBACK_DAYS + 5)).strftime("%Y-%m-%d")
    end   = (pd.to_datetime(date_eff) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    all_syms = list(dict.fromkeys(tickers + ["^GSPC", "^VIX"]))
    data = yf.download(all_syms, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    px = _flatten_prices_multi(data)

    g = px.get("^GSPC")
    if g is not None and not g.empty:
        last_px_date = pd.to_datetime(g.index.max()).strftime("%Y-%m-%d")
        if pd.to_datetime(date_eff) > pd.to_datetime(last_px_date):
            date_eff = last_px_date

    key_list = list(DEFAULT_FINNHUB_KEYS)
    if not key_list:
        raise RuntimeError("DEFAULT_FINNHUB_KEYS is empty.")
    clients = [finnhub.Client(api_key=k) for k in key_list]
    idx = 0

    def _company_news(sym: str, start_date: str, end_date: str):
        nonlocal idx
        cli = clients[idx]
        idx = (idx + 1) % len(clients)
        try:
            return cli.company_news(sym, _from=start_date, to=end_date) or []
        except Exception:
            time.sleep(0.6)
            try:
                return clients[idx].company_news(sym, _from=start_date, to=end_date) or []
            except Exception:
                return []

    start_news = (pd.to_datetime(date_eff) - pd.Timedelta(days=NEWS_7D_DAYS)).strftime("%Y-%m-%d")

    news_by_ticker: Dict[str, pd.DataFrame] = {}
    for t in _tqdm(tickers, desc="company_news"):
        raw = _company_news(t, start_news, date_eff)
        rows = []
        for it in raw:
            rows.append({
                "id": it.get("id"),
                "datetime": it.get("datetime"),
                "headline": (it.get("headline") or "").strip(),
                "source": (it.get("source") or "").strip(),
                "ticker": t,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = _normalize_news_datetime(df, "datetime")
            df["headline"] = df["headline"].astype(str).map(_clean_headline)
            df["norm"] = df["headline"].map(_norm_key)
            df = _dedup_news(df).sort_values("datetime").reset_index(drop=True)
        else:
            df = pd.DataFrame(columns=["datetime","headline","norm","source","ticker"])
        news_by_ticker[t] = df
        time.sleep(SLEEP_BETWEEN_NEWS_TICKERS)

    frames = []
    for t in tickers:
        df = news_by_ticker.get(t)
        if df is not None and not df.empty:
            frames.append(df[["datetime","headline","norm","source"]])
    mkt_agg = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["datetime","headline","norm","source"])

    def _market_text(for_date: str, mkt_agg: pd.DataFrame, px: Dict[str, pd.DataFrame], k1: int = 100, k7: int = 100):
        hdr = f"CTX:MARKET | DATE:{for_date}\n" + _market_macro_block(for_date, px)
        if mkt_agg is None or mkt_agg.empty:
            return hdr + "NEWS_1D:\nNone\nNEWS_7D_EXCLUDING_1D:\nNone", 0, 0
        w1, w7_ex = _window_time(mkt_agg, for_date, NEWS_1D_HOURS, NEWS_7D_DAYS)
        ref_dt = pd.to_datetime(for_date) + pd.Timedelta(hours=23, minutes=59)
        m1 = _select_diverse_by_source(w1, k=k1, ref_dt=ref_dt)
        m7 = _select_diverse_by_source(w7_ex, k=k7, ref_dt=ref_dt)
        hdr += "NEWS_1D:\n" + _format_lines(m1["headline"].tolist())
        hdr += "\nNEWS_7D_EXCLUDING_1D:\n" + _format_lines(m7["headline"].tolist())
        return hdr, int(len(m1)), int(len(m7))

    def _stock_text(ticker: str, for_date: str, news_df: pd.DataFrame, px: Dict[str, pd.DataFrame], k1: int = 80, k7: int = 160):
        base = f"CTX:STOCK | TICKER:{ticker} | DATE:{for_date}\n" + _stock_indicators(ticker, for_date, px)
        if news_df is None or news_df.empty:
            return base + "NEWS_1D:\nNone\nNEWS_7D_EXCLUDING_1D:\nNone", 0, 0
        w1, w7_ex = _window_time(news_df, for_date, NEWS_1D_HOURS, NEWS_7D_DAYS)
        ref_dt = pd.to_datetime(for_date) + pd.Timedelta(hours=23, minutes=59)
        s1 = _select_diverse_by_source(w1, k=k1, ref_dt=ref_dt, ticker=ticker)
        s7 = _select_diverse_by_source(w7_ex, k=k7, ref_dt=ref_dt, ticker=ticker)
        out = base + "NEWS_1D:\n" + _format_lines(s1["headline"].tolist())
        out += "\nNEWS_7D_EXCLUDING_1D:\n" + _format_lines(s7["headline"].tolist())
        return out, int(len(s1)), int(len(s7))

    user_text, _, _ = _market_text(date_eff, mkt_agg, px)
    item_records = []
    for t in tickers[:max_candidates]:
        df_t = news_by_ticker.get(t, pd.DataFrame(columns=["datetime","headline","norm","source"]))
        itxt, _, _ = _stock_text(t, date_eff, df_t, px)
        item_records.append({"item_id": f"I_{t}", "item_text": itxt})

    payload = {
        "date_eff": date_eff,
        "user_record": [{"user_id": f"U_{date_eff}", "user_text": user_text}],
        "item_records": item_records,
    }
    return payload

# ============================================================
#               Remote: GPU inferencer (score)
# ============================================================
@app.function(
    image=gpu_image,
    gpu=["L4", "A10", "any"],
    volumes={"/vol/models": MODEL_VOL, "/vol/code": CODE_VOL},
    timeout=60*20,
)
def run_infer_remote(
    payload: Dict[str, Any],
    ckpt_path: str,
    top_k: int = 20,
    invest_amt: float = 1000.0,
    device_choice: str = "auto",
    temp: float = 2.0,
) -> Dict[str, Any]:
    def _ensure_interfusion_installed():
        try:
            import interfusion  # noqa: F401
            return
        except Exception:
            import sys, subprocess, os
            wheel_path = "/vol/code/pkgs/interfusion_encoder-3.1.whl"
            src_path   = "/vol/code/pkgs/interfusion_encoder-3.1"
            target = wheel_path if os.path.exists(wheel_path) else src_path
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", target])
    _ensure_interfusion_installed()

    import torch
    import torch.nn as nn
    import numpy as np

    from interfusion.models import NewsTextEncoder, BiEncoderWithFeatures
    from interfusion.trainer import (
        Standardizer,
        build_parsed_maps, build_cache,
        encode_news_pool_trainable,
        get_tqdm,
        _topk_softmax_np,
    )

    def _torch_load_weights_only_false(path: str):
        import inspect
        kw = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            kw["weights_only"] = False
        return torch.load(path, **kw)

    user_record = payload["user_record"]
    item_records = payload["item_records"]
    date_eff = payload["date_eff"]

    device = "cuda" if (device_choice == "auto" and torch.cuda.is_available()) else (device_choice if device_choice != "auto" else "cpu")

    ckpt = _torch_load_weights_only_false(ckpt_path)
    cfg = dict(ckpt.get("config", {}))
    cfg["device"] = device
    cfg.setdefault("online_encode_user_eval", bool(cfg.get("online_encode_user", True)))
    cfg.setdefault("online_encode_items_eval", bool(cfg.get("online_encode_items", False)))
    cfg["news_batch_size"] = int(cfg.get("news_batch_size", 128))

    enc_name    = cfg.get("headline_encoder_name", "nreimers/BERT-Tiny_L-2_H-128_A-2")
    enc_max_len = int(cfg.get("headline_max_length", 64))
    freeze_enc  = bool(cfg.get("freeze_headline_encoder", False))

    news_enc = NewsTextEncoder(enc_name, max_length=enc_max_len, trainable=not freeze_enc).to(device)
    if "news_encoder_state_dict" in ckpt:
        _missing, _unexpected = news_enc.load_state_dict(ckpt["news_encoder_state_dict"], strict=False)

    news_dim = int(news_enc.output_dim)
    model = BiEncoderWithFeatures(
        user_num_dim=3,
        item_num_dim=5,
        news_emb_dim=news_dim,
        proj_dim=int(cfg.get("proj_dim", 128)),
        d_model=int(cfg.get("d_model", 256)),
        normalize=bool(cfg.get("normalize_embeddings", True)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval(); news_enc.eval()

    u_std = Standardizer(); i_std = Standardizer()
    nm = ckpt.get("numeric_stats", {})
    if nm:
        if nm.get("user_mean") is not None: u_std.mean = np.asarray(nm["user_mean"], dtype=np.float32)
        if nm.get("user_std")  is not None: u_std.std  = np.asarray(nm["user_std"],  dtype=np.float32)
        if nm.get("item_mean") is not None: i_std.mean = np.asarray(nm["item_mean"], dtype=np.float32)
        if nm.get("item_std")  is not None: i_std.std  = np.asarray(nm["item_std"],  dtype=np.float32)

    tqdmm = get_tqdm(cfg)
    user_parsed, item_parsed = build_parsed_maps(user_record, item_records, u_std, i_std, tqdmm)

    online_user  = bool(cfg.get("online_encode_user_eval", False))
    online_items = bool(cfg.get("online_encode_items_eval", False))
    user_cache = {} if online_user  else build_cache(user_parsed, news_enc, int(cfg.get("news_batch_size",128)), tqdmm)
    item_cache = {} if online_items else build_cache(item_parsed, news_enc, int(cfg.get("news_batch_size",128)), tqdmm)

    uid = user_record[0]["user_id"]
    pe_u = user_parsed[uid]
    u_num = torch.from_numpy(pe_u.numeric).to(device).float().unsqueeze(0)

    if online_user:
        u_n1 = encode_news_pool_trainable(news_enc, pe_u.news1d_text)
        u_n7 = encode_news_pool_trainable(news_enc, pe_u.news7d_text)
    else:
        ce = user_cache[uid]
        u_n1 = torch.from_numpy(ce.news1d).to(device).float().unsqueeze(0)
        u_n7 = torch.from_numpy(ce.news7d).to(device).float().unsqueeze(0)

    ordered_ids = [r["item_id"] for r in item_records]
    it_nums, it_n1s, it_n7s = [], [], []
    for iid in ordered_ids:
        pe_i = item_parsed[iid]
        it_nums.append(pe_i.numeric)
        if online_items:
            it_n1s.append(encode_news_pool_trainable(news_enc, pe_i.news1d_text))
            it_n7s.append(encode_news_pool_trainable(news_enc, pe_i.news7d_text))
        else:
            ce = item_cache[iid]
            it_n1s.append(torch.from_numpy(ce.news1d).to(device).float().unsqueeze(0))
            it_n7s.append(torch.from_numpy(ce.news7d).to(device).float().unsqueeze(0))

    it_num_t = torch.from_numpy(np.stack(it_nums)).to(device).float()
    it_n1_t  = torch.cat(it_n1s, dim=0)
    it_n7_t  = torch.cat(it_n7s, dim=0)

    with torch.no_grad():
        if isinstance(model, nn.DataParallel):
            u_rep = model.module.encode_user(u_num.repeat(it_num_t.size(0),1), u_n1.repeat(it_num_t.size(0),1), u_n7.repeat(it_num_t.size(0),1))
            v_rep = model.module.encode_item(it_num_t, it_n1_t, it_n7_t)
        else:
            u_rep = model.encode_user(u_num.repeat(it_num_t.size(0),1), u_n1.repeat(it_num_t.size(0),1), u_n7.repeat(it_num_t.size(0),1))
            v_rep = model.encode_item(it_num_t, it_n1_t, it_n7_t)

        scores = (u_rep * v_rep).sum(dim=-1).detach().cpu().numpy()

    k = min(top_k, len(ordered_ids))
    weights = _topk_softmax_np(scores, k=k, temp=float(temp))
    order = np.argsort(scores)[::-1][:k]

    def _iid_to_ticker(iid: str) -> str:
        return iid.split("_", 1)[-1] if "_" in iid else iid

    rows = []
    total_w = 0.0
    total_d = 0.0
    for rank, idx in enumerate(order, start=1):
        iid = ordered_ids[idx]
        tk  = _iid_to_ticker(iid)
        sc  = float(scores[idx])
        wi  = float(weights[idx])
        alloc = float(invest_amt) * wi
        total_w += wi; total_d += alloc
        rows.append({
            "rank": rank,
            "ticker": tk,
            "item_id": iid,
            "score": sc,
            "weight": wi,
            "allocation": alloc
        })

    return {
        "date": date_eff,
        "k": k,
        "temp": float(temp),
        "invest_amt": float(invest_amt),
        "rows": rows,
        "sum_weights": total_w,
        "sum_allocation": total_d,
    }

# ============================================================
#     Plan creation (+ buy prices & quantities) on CPU
# ============================================================
@app.function(
    image=base_image,
    volumes={"/vol/plans": PLANS_VOL},
    timeout=60*25,
)
def create_investment_plan(
    ckpt: str = DEFAULT_CKPT,
    invest_amt: float = DEFAULT_INVEST_AMT,
    top_k: int = DEFAULT_TOP_K,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    date: str = DEFAULT_OVERRIDE_DATE,
    temp: float = DEFAULT_TEMP,
) -> Dict[str, Any]:
    #Create a new plan, compute buy prices & quantities, and persist it.
    import yfinance as yf

    payload = build_payload_remote.remote(
        for_date=date or None,
        tickers=TICKERS_UNIVERSE,
        max_candidates=int(max_candidates),
    )
    result = run_infer_remote.remote(
        payload=payload,
        ckpt_path=ckpt,
        top_k=int(top_k),
        invest_amt=float(invest_amt),
        temp=float(temp),
    )

    tickers = [r["ticker"] for r in result["rows"]]

    def _last_close_map(tickers: List[str]) -> Dict[str, Optional[float]]:
        if not tickers:
            return {}
        try:
            d1m = yf.download(tickers, period="1d", interval="1m", group_by="ticker", progress=False, threads=True, auto_adjust=False)
        except Exception:
            d1m = None
        if d1m is None or isinstance(d1m, pd.DataFrame) and d1m.empty:
            try:
                d1d = yf.download(tickers, period="1d", interval="1d", group_by="ticker", progress=False, threads=True, auto_adjust=False)
            except Exception:
                d1d = None
        else:
            d1d = None

        price_map: Dict[str, Optional[float]] = {t: None for t in tickers}

        def _extract(df: pd.DataFrame) -> Dict[str, float]:
            out = {}
            if df is None or df.empty:
                return out
            if isinstance(df.columns, pd.MultiIndex):
                for t in set(df.columns.get_level_values(0)):
                    if not isinstance(t, str):
                        continue
                    sub = df[t]
                    if "Close" in sub and not sub["Close"].empty:
                        out[t] = float(sub["Close"].iloc[-1])
            else:
                if "Close" in df and not df["Close"].empty:
                    out["SINGLE"] = float(df["Close"].iloc[-1])
            return out

        m1 = _extract(d1m) if d1m is not None else {}
        m2 = _extract(d1d) if d1d is not None else {}
        for t in tickers:
            if t in m1 and m1[t] is not None:
                price_map[t] = float(m1[t])
            elif t in m2 and m2[t] is not None:
                price_map[t] = float(m2[t])
            else:
                price_map[t] = None
        return price_map

    price_map = _last_close_map(tickers)

    rows_out = []
    for row in result["rows"]:
        tk = row["ticker"]
        alloc = float(row["allocation"])
        buy_price = price_map.get(tk)
        if buy_price is None or buy_price <= 0:
            quantity = None
        else:
            quantity = alloc / buy_price
        rows_out.append({
            **row,
            "buy_price": buy_price,
            "quantity": quantity,
        })

    now_utc = pd.Timestamp.utcnow()
    plan_id = f"{result['date']}_{now_utc.strftime('%Y%m%dT%H%M%SZ')}"
    plan = {
        "plan_id": plan_id,
        "created_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date": result["date"],
        "k": result["k"],
        "temp": result["temp"],
        "invest_amt": result["invest_amt"],
        "rows": rows_out,
        "sum_weights": result["sum_weights"],
        "sum_allocation": result["sum_allocation"],
    }
    _save_plan(plan)
    return plan

# ============================================================
#    Attach live prices & compute PnL for a stored plan
# ============================================================
@app.function(image=base_image, volumes={"/vol/plans": PLANS_VOL}, timeout=60*20)
def _attach_live_prices_and_pnl(plan: Dict[str, Any]) -> Dict[str, Any]:
    import yfinance as yf

    tickers = [r["ticker"] for r in plan.get("rows", [])]
    if not tickers:
        return {"plan": plan, "rows": [], "total_cost": 0.0, "total_value": 0.0, "total_pnl": 0.0, "total_pnl_pct": 0.0}

    def _live_price_map(tickers: List[str]) -> Dict[str, Optional[float]]:
        try:
            d1m = yf.download(tickers, period="1d", interval="1m", group_by="ticker", progress=False, threads=True, auto_adjust=False)
        except Exception:
            d1m = None
        if d1m is None or (isinstance(d1m, pd.DataFrame) and d1m.empty):
            try:
                d1d = yf.download(tickers, period="1d", interval="1d", group_by="ticker", progress=False, threads=True, auto_adjust=False)
            except Exception:
                d1d = None
        else:
            d1d = None

        price_map: Dict[str, Optional[float]] = {t: None for t in tickers}

        def _extract(df: pd.DataFrame) -> Dict[str, float]:
            out = {}
            if df is None or df.empty:
                return out
            if isinstance(df.columns, pd.MultiIndex):
                for t in set(df.columns.get_level_values(0)):
                    if not isinstance(t, str): continue
                    sub = df[t]
                    if "Close" in sub and not sub["Close"].empty:
                        out[t] = float(sub["Close"].iloc[-1])
            else:
                if "Close" in df and not df["Close"].empty:
                    out["SINGLE"] = float(df["Close"].iloc[-1])
            return out

        m1 = _extract(d1m) if d1m is not None else {}
        m2 = _extract(d1d) if d1d is not None else {}
        for t in tickers:
            if t in m1 and m1[t] is not None:
                price_map[t] = float(m1[t])
            elif t in m2 and m2[t] is not None:
                price_map[t] = float(m2[t])
            else:
                price_map[t] = None
        return price_map

    price_map = _live_price_map(tickers)

    rows = []
    total_cost = 0.0
    total_value = 0.0

    for r in plan.get("rows", []):
        tk = r["ticker"]
        buy_price = r.get("buy_price")
        qty = r.get("quantity")
        live_price = price_map.get(tk)

        if buy_price is None or qty is None or live_price is None:
            position_value = None
            pnl = None
            pnl_pct = None
            cost = 0.0
        else:
            cost = float(buy_price) * float(qty)
            position_value = float(live_price) * float(qty)
            pnl = position_value - cost
            pnl_pct = (position_value / cost - 1.0) * 100.0
            total_cost += cost
            total_value += position_value

        rows.append({
            **r,
            "live_price": live_price,
            "position_value": position_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        })

    total_pnl = total_value - total_cost
    total_pnl_pct = (total_value / total_cost - 1.0) * 100.0 if total_cost > 0 else 0.0

    return {
        "plan": plan,
        "rows": rows,
        "total_cost": total_cost,
        "total_value": total_value,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
    }

# ============================================================
#                    Web endpoints (HTTP)
# ============================================================
@app.function(image=base_image, volumes={"/vol/plans": PLANS_VOL})
@modal.web_endpoint(method="GET")
def list_plans() -> Dict[str, Any]:
    metas = _list_plans_meta()
    return {"plans": metas}

@app.function(image=base_image, volumes={"/vol/plans": PLANS_VOL})
@modal.web_endpoint(method="GET")
def get_plan_with_live(id: Optional[str] = None) -> Dict[str, Any]:
    #Return chosen plan with live prices & PnL. If none exist, create one.
    plan_id = id
    if not plan_id:
        pid = _latest_plan_id()
        if not pid:
            plan = create_investment_plan.remote()
            return _attach_live_prices_and_pnl.remote(plan)
        plan_id = pid

    plan = _load_plan(plan_id)
    if plan is None:
        plan = create_investment_plan.remote()
    return _attach_live_prices_and_pnl.remote(plan)

# ============================================================
#        Daily schedule at 9am Australia/Melbourne
# ============================================================
@app.function(
    schedule=modal.Cron("0 9 * * *", timezone="Australia/Melbourne"),
    image=base_image,
    volumes={"/vol/plans": PLANS_VOL},
    timeout=60*30,
)
def scheduled_daily_plan():
    return create_investment_plan.remote()

# ============================================================
#               Local entrypoint (manual trigger)
# ============================================================
@app.local_entrypoint()
def main():
    # Manual: create a plan and print its ID, and show the two web URLs
    plan = create_investment_plan.remote()
    metas = _list_plans_meta()
    print("Created plan_id:", plan["plan_id"])
    print("Total plans:", len(metas))
    print("List plans URL   : /list_plans")
    print("Get latest plan  : /get_plan_with_live")
    print("Get a specific   : /get_plan_with_live?id=<PLAN_ID>")
"""

def maybe_deploy_modal_app(auto: bool = False) -> str:
    """
    Writes the embedded Modal app to /tmp and deploys it using the Modal CLI.
    Returns the temp file path used for deployment.
    """
    tmp = Path(tempfile.gettempdir()) / "modal_infer_inmemory_latest.py"
    tmp.write_text(MODAL_APP_SOURCE, encoding="utf-8")
    if auto:
        try:
            subprocess.run(["modal", "deploy", str(tmp)], check=True, capture_output=True, text=True)
        except Exception as e:
            st.warning(f"Modal deploy failed (continuing anyway): {e}")
    return str(tmp)

def _sdk_app_ready() -> bool:
    """Check that the deployed app and functions exist (SDK lookup)."""
    try:
        modal.Function.from_name(APP_NAME, "list_plans")
        modal.Function.from_name(APP_NAME, "get_plan_with_live")
        return True
    except Exception:
        return False

@st.cache_data(ttl=60, show_spinner=False)
def _get_plans() -> List[Dict[str, Any]]:
    f = modal.Function.from_name(APP_NAME, "list_plans")
    data = f.remote()
    return data.get("plans", [])

@st.cache_data(ttl=30, show_spinner=False)
def _get_plan_with_live(plan_id: Optional[str]) -> Dict[str, Any]:
    f = modal.Function.from_name(APP_NAME, "get_plan_with_live")
    if plan_id:
        return f.remote(id=plan_id)
    return f.remote()

# --------------------------- UI ---------------------------
st.set_page_config(page_title="InterFusion Investment Plans", layout="wide")
st.title("Investment Plans (Modal GPU + Streamlit UI)")

st.sidebar.header("Modal control")
auto_deploy_default = os.getenv("AUTO_DEPLOY", "0") == "1"
deploy_now = st.sidebar.checkbox("Auto-deploy embedded Modal app on start", value=auto_deploy_default)
if st.sidebar.button("Deploy / Update Modal app now"):
    path = maybe_deploy_modal_app(auto=True)
    st.sidebar.success(f"Deployed from: {path}")
else:
    if deploy_now:
        maybe_deploy_modal_app(auto=True)

# Ensure app is deployed / functions resolvable
if not _sdk_app_ready():
    st.error("Couldn't find deployed Modal app. Use the sidebar to deploy it, and verify MODAL_TOKEN_ID / MODAL_TOKEN_SECRET.")
    st.stop()

# Load plans; create one if none exist
plans = _get_plans()
if not plans:
    st.info("No plans yet — creating one now on Modal...")
    _ = _get_plan_with_live(None)
    _get_plans.clear()
    plans = _get_plans()

if not plans:
    st.error("Still no plans after creation. Check Modal logs.")
    st.stop()

# Dropdown (oldest -> latest, default latest)
labels = [p["label"] for p in plans]
ids = [p["plan_id"] for p in plans]
default_idx = len(labels) - 1
sel_label = st.selectbox("Select investment plan", labels, index=default_idx)
sel_id = ids[labels.index(sel_label)]

# Fetch selected plan with live prices & PnL
data = _get_plan_with_live(sel_id)
if "error" in data:
    st.error(f"Modal returned an error: {data['error']}")
    st.stop()

plan = data["plan"]
rows = data.get("rows", [])
total_cost = data.get("total_cost", 0.0)
total_value = data.get("total_value", 0.0)
total_pnl = data.get("total_pnl", 0.0)
total_pnl_pct = data.get("total_pnl_pct", 0.0)

st.subheader(f"Plan for date {plan['date']}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Invest amount", f"${plan['invest_amt']:,.2f}")
c2.metric("Top-k", int(plan["k"]))
c3.metric("Current total value", f"${total_value:,.2f}")
c4.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl_pct:.2f}%")

st.markdown("---")

if not rows:
    st.warning("No rows in this plan.")
else:
    df = pd.DataFrame(rows)
    def m(x): return "—" if pd.isna(x) else f"${x:,.2f}"
    def q(x): return "—" if pd.isna(x) else f"{x:.4f}"
    df_disp = df.copy()
    for col, fmt in [
        ("allocation", m),
        ("buy_price", m),
        ("live_price", m),
        ("position_value", m),
        ("pnl", m),
        ("quantity", q),
    ]:
        if col in df_disp:
            df_disp[col] = df_disp[col].map(fmt)
    if "pnl_pct" in df_disp:
        df_disp["pnl_pct"] = df_disp["pnl_pct"].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}%")
    cols = ["rank","ticker","item_id","allocation","buy_price","quantity","live_price","position_value","pnl","pnl_pct","score","weight"]
    cols = [c for c in cols if c in df_disp.columns]
    st.dataframe(df_disp[cols], use_container_width=True, hide_index=True)

st.caption("Not financial advice. Plans created daily at 9am Australia/Melbourne by the Modal app.")
