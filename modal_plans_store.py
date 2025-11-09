# modal_plans_store.py
# Modal "plans" microservice: stores and serves investment plans created by your inference app.
# - Uses a Modal Volume to persist JSON files between runs.
# - Exposes simple functions: save_plan, list_plans, get_plan, delete_plan
#
# Deploy once:
#   modal deploy modal_plans_store.py
#
# Then call from anywhere (e.g., Streamlit) with:
#   from modal import Function
#   save_plan = Function.from_name("inmemory-latest-plans", "save_plan")
#   save_plan.remote(plan_dict)

from __future__ import annotations
import os, json, time, math
from datetime import datetime, timezone
from typing import Dict, Any, List
from zoneinfo import ZoneInfo

import modal

APP_NAME = "inmemory-latest-plans"

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas>=2.2", "numpy>=1.26", "yfinance>=0.2.50")
)

PLANS_VOL = modal.Volume.from_name("plans-cache", create_if_missing=True, version=1)

app = modal.App(APP_NAME)

PLANS_DIR = "/vol/plans"
PLANS_INDEX = f"{PLANS_DIR}/plans.jsonl"   # one JSON per line (append-only)

def _now_iso_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

def _melbourne_ts() -> str:
    return datetime.now(ZoneInfo("Australia/Melbourne")).isoformat()

def _ensure_dir():
    os.makedirs(PLANS_DIR, exist_ok=True)
    if not os.path.exists(PLANS_INDEX):
        with open(PLANS_INDEX, "w", encoding="utf-8") as f:
            pass

def _load_index() -> List[Dict[str, Any]]:
    _ensure_dir()
    items: List[Dict[str, Any]] = []
    with open(PLANS_INDEX, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items

def _write_index_append(doc: Dict[str, Any]) -> None:
    _ensure_dir()
    with open(PLANS_INDEX, "a", encoding="utf-8") as f:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

def _save_plan_blob(plan_id: str, doc: Dict[str, Any]) -> None:
    _ensure_dir()
    p = f"{PLANS_DIR}/{plan_id}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

def _read_plan_blob(plan_id: str) -> Dict[str, Any] | None:
    p = f"{PLANS_DIR}/{plan_id}.json"
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _fetch_current_prices(tickers: List[str]) -> Dict[str, float]:
    # Robust yfinance fetch for spot prices.
    import pandas as pd
    import yfinance as yf
    prices: Dict[str, float] = {}
    if not tickers:
        return prices

    # Try fast path: Ticker(...).fast_info['last_price']
    for t in tickers:
        try:
            fi = yf.Ticker(t).fast_info
            p = fi.get("last_price") or fi.get("last_trade") or fi.get("last_close")
            if p:
                prices[t] = float(p)
        except Exception:
            pass

    # Fill gaps using history (1d/1m)
    missing = [t for t in tickers if t not in prices]
    if missing:
        try:
            df = yf.download(missing, period="1d", interval="1m", progress=False, group_by="ticker", threads=True)
            if hasattr(df, "columns") and len(df.columns) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    for t in missing:
                        try:
                            sub = df[t]
                            # prefer last non-NaN Close
                            last = float(sub["Close"].dropna().iloc[-1])
                            prices[t] = last
                        except Exception:
                            pass
                else:
                    # single frame
                    last = float(df["Close"].dropna().iloc[-1])
                    # Not sure which ticker; skip
            else:
                pass
        except Exception:
            pass

    # Last attempt: 5d daily
    missing = [t for t in tickers if t not in prices]
    if missing:
        try:
            df2 = yf.download(missing, period="5d", interval="1d", progress=False, group_by="ticker", threads=True)
            if isinstance(df2.columns, pd.MultiIndex):
                for t in missing:
                    try:
                        last = float(df2[t]["Close"].dropna().iloc[-1])
                        prices[t] = last
                    except Exception:
                        pass
            else:
                # single frame
                last = float(df2["Close"].dropna().iloc[-1])
                # Not sure which ticker; skip
        except Exception:
            pass

    return prices

@app.function(image=base_image, volumes={"/vol/plans": PLANS_VOL}, timeout=60*5)
def save_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enriches a fresh plan (from inference) with buy prices and share quantities,
    then persists it on the 'plans-cache' volume.
    Input plan must contain: date, k, temp, invest_amt, rows[{ticker, allocation, score, weight, item_id}].
    """
    import pandas as pd  # available via base_image

    _ensure_dir()

    # Assign plan_id and timestamps
    plan_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plan["plan_id"] = plan_id
    plan["created_at_utc"] = _now_iso_utc()
    plan["created_at_melbourne"] = _melbourne_ts()

    tickers = [r["ticker"] for r in plan.get("rows", []) if r.get("ticker")]
    prices = _fetch_current_prices(tickers)

    # Enrich rows with buy_price and shares
    for r in plan.get("rows", []):
        t = r.get("ticker")
        alloc = _safe_float(r.get("allocation"), 0.0) or 0.0
        buy = _safe_float(prices.get(t), None)
        r["buy_price"] = buy
        r["shares"] = (alloc / buy) if (buy and buy > 0) else None

    # Save blob + append index row
    _save_plan_blob(plan_id, plan)

    idx_row = {
        "plan_id": plan_id,
        "date": plan.get("date"),
        "k": plan.get("k"),
        "temp": plan.get("temp"),
        "invest_amt": plan.get("invest_amt"),
        "created_at_utc": plan["created_at_utc"],
        "created_at_melbourne": plan["created_at_melbourne"],
    }
    _write_index_append(idx_row)

    return plan

@app.function(image=base_image, volumes={"/vol/plans": PLANS_VOL}, timeout=60*5)
def list_plans() -> List[Dict[str, Any]]:
    """
    Returns a list of summaries sorted by creation time (oldest -> latest).
    """
    items = _load_index()
    # Stable sort by created_at_* then plan_id
    items.sort(key=lambda d: (d.get("created_at_utc",""), d.get("plan_id","")))
    return items

@app.function(image=base_image, volumes={"/vol/plans": PLANS_VOL}, timeout=60*5)
def get_plan(plan_id: str) -> Dict[str, Any] | None:
    return _read_plan_blob(plan_id)

@app.function(image=base_image, volumes={"/vol/plans": PLANS_VOL}, timeout=60*5)
def delete_plan(plan_id: str) -> bool:
    """
    Removes the plan json but keeps the index entry (append-only ledger).
    """
    p = f"{PLANS_DIR}/{plan_id}.json"
    try:
        if os.path.exists(p):
            os.remove(p)
        return True
    except Exception:
        return False