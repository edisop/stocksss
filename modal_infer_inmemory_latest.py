# modal_infer_inmemory_latest.py
# Modalized "latest-day" inference:
# - CPU function builds texts in memory (no files)
# - GPU function loads your ckpt from a Modal Volume and scores items
# - Prints a nice top-k allocation table
#
# Requires:
#   - Modal CLI authed locally:  modal token new
#   - Volume "model-cache" (v2) containing your checkpoint at /ckpts/...
#   - Volume "code-cache"  (v1) containing /pkgs/interfusion_encoder-3.1
#
# Run (flags optional because we hard-code defaults below):
#   modal run modal_infer_inmemory_latest.py -- ^
#     --ckpt /vol/models/ckpts/biex_listmle_final.pt ^
#     --invest_amt 1000 ^
#     --top_k 20

from __future__ import annotations
import os, re, math, time, json, warnings
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

import modal

# ============================================================
#                === CHANGE-ME DEFAULTS (edit here) ===
#    These mirror your typical CLI values so you can just
#    `modal run` without passing flags every time.
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

# Base image with common deps (NO local file adds here)
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # build helpers (in case your uploaded package uses pyproject/setuptools)
        "pip>=25.0", "setuptools>=70", "wheel>=0.43",
        # data + utils
        "pandas>=2.2", "numpy>=1.26", "tqdm>=4.66",
        # finance / APIs
        "yfinance>=0.2.50", "finnhub-python>=2.4.18",
        # models
        "transformers>=4.46", "tokenizers>=0.20",
    )
)

gpu_image = (
    base_image
    .pip_install("torch==2.5.*")  # works on Modal GPU runners
)

# Persistent volumes
MODEL_VOL = modal.Volume.from_name("model-cache", create_if_missing=True, version=2)
CODE_VOL  = modal.Volume.from_name("code-cache",  create_if_missing=False, version=1)

# -----------------------------
# Static config (unchanged)
# -----------------------------
TICKERS_UNIVERSE = [
    # (same universe you pasted; kept as-is)
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
#               Helpers (unchanged from your script)
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
        # Only create entries for tickers that actually returned data
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
    use = now_utc.date() if now_utc.hour >= UTC_ROLLOVER_HOUR else (now_utc - pd.Timedelta(days=1)).date()
    return pd.Timestamp(use).strftime("%Y-%m-%d")

# ============================================================
#                Remote: CPU fetcher (build texts)
# ============================================================
@app.function(
    image=base_image,
    timeout=60 * 20,
)
def build_payload_remote(
    for_date: str | None,
    tickers: List[str],
    max_candidates: int = 500,
) -> Dict[str, Any]:
    import yfinance as yf
    import finnhub
    from tqdm import tqdm as _tqdm

    # resolve date with rollover
    date_eff = (for_date or "").strip() or _effective_utc_date_for_infer()

    # --- prices (tolerant to missing symbols) ---
    start = (pd.to_datetime(date_eff) - pd.Timedelta(days=PRICE_LOOKBACK_DAYS + 5)).strftime("%Y-%m-%d")
    end   = (pd.to_datetime(date_eff) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    all_syms = list(dict.fromkeys(tickers + ["^GSPC", "^VIX"]))
    # yfinance will warn on failures; we just flatten whatever arrives
    data = yf.download(all_syms, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    px = _flatten_prices_multi(data)

    # clamp to last ^GSPC date if needed (if market closed / weekend)
    g = px.get("^GSPC")
    if g is not None and not g.empty:
        last_px_date = pd.to_datetime(g.index.max()).strftime("%Y-%m-%d")
        if pd.to_datetime(date_eff) > pd.to_datetime(last_px_date):
            date_eff = last_px_date

    # --- headlines (7d window) ---
    key_list = list(DEFAULT_FINNHUB_KEYS)
    if not key_list:
        raise RuntimeError("DEFAULT_FINNHUB_KEYS is empty.")

    # simple round-robin client
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

    # --- build user/item texts (in memory) ---
    # MARKET aggregate across all tickers (FIX: use a LIST for column selection)
    frames = []
    for t in tickers:
        df = news_by_ticker.get(t)
        if df is not None and not df.empty:
            frames.append(df[["datetime","headline","norm","source"]])
    mkt_agg = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["datetime","headline","norm","source"])

    # market text
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

    # per-stock text
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
    gpu=["L4", "A10", "any"],   # prefer L4; fall back to A10/any
    volumes={"/vol/models": MODEL_VOL, "/vol/code": CODE_VOL},
    timeout=60 * 20,
)
def run_infer_remote(
    payload: Dict[str, Any],
    ckpt_path: str,
    top_k: int = 20,
    invest_amt: float = 1000.0,
    device_choice: str = "auto",
    temp: float = 2.0,
) -> Dict[str, Any]:
    # --- ensure your uploaded package is installed (from code-cache volume) ---
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

    # unpack payload
    user_record = payload["user_record"]
    item_records = payload["item_records"]
    date_eff = payload["date_eff"]

    # device
    device = "cuda" if (device_choice == "auto" and torch.cuda.is_available()) else (device_choice if device_choice != "auto" else "cpu")

    # load checkpoint + construct model
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

    # numeric scalers
    u_std = Standardizer(); i_std = Standardizer()
    nm = ckpt.get("numeric_stats", {})
    if nm:
        if nm.get("user_mean") is not None: u_std.mean = np.asarray(nm["user_mean"], dtype=np.float32)
        if nm.get("user_std")  is not None: u_std.std  = np.asarray(nm["user_std"],  dtype=np.float32)
        if nm.get("item_mean") is not None: i_std.mean = np.asarray(nm["item_mean"], dtype=np.float32)
        if nm.get("item_std")  is not None: i_std.std  = np.asarray(nm["item_std"],  dtype=np.float32)

    # parse & encode
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
#               Local entrypoint (CLI)
# ============================================================
@app.local_entrypoint()
def main(
    ckpt: str = DEFAULT_CKPT,
    invest_amt: float = DEFAULT_INVEST_AMT,
    top_k: int = DEFAULT_TOP_K,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    date: str = DEFAULT_OVERRIDE_DATE,  # YYYY-MM-DD; empty -> auto rollover
    temp: float = DEFAULT_TEMP,         # softmax temperature
):
    # 1) Build texts on CPU container (no files)
    payload = build_payload_remote.remote(
        for_date=date or None,
        tickers=TICKERS_UNIVERSE,
        max_candidates=int(max_candidates),
    )

    # 2) Score on GPU container
    result = run_infer_remote.remote(
        payload=payload,
        ckpt_path=ckpt,
        top_k=int(top_k),
        invest_amt=float(invest_amt),
        temp=float(temp),
    )

    # 3) Pretty print locally
    print(f"\n=== Inference — DATE={result['date']} — top-{result['k']} allocation (temp={result['temp']:.3f}, invest=${result['invest_amt']:.2f}) ===")
    print("{:>4s}  {:<10s}  {:<18s}  {:>11s}  {:>10s}  {:>12s}".format(
        "Rank","Ticker","ItemID","Score","Weight","Allocation"
    ))
    for r in result["rows"]:
        print("{:>4d}  {:<10s}  {:<18s}  {:>11.6f}  {:>9.6f}  ${:>11.2f}".format(
            r["rank"], r["ticker"], r["item_id"], r["score"], r["weight"], r["allocation"]
        ))
    print("--------------------------------------------------------------------------")
    print(f"Sum of weights (top-{result['k']}): {result['sum_weights']:.6f}")
    print(f"Total allocation       : ${result['sum_allocation']:.2f}")
    print("Note: weights are normalized over the top-k set (top-k softmax).")
