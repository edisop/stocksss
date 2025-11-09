# Daily Investment Plans — Modal × Streamlit (End‑to‑End Guide)

This guide walks you through deploying your **Modal** inference apps and wiring them into a **Streamlit Cloud** dashboard that visualizes **daily investment plans** and shows **live P&L** from Yahoo Finance.

> You’ll deploy Modal *once* from your laptop. After that, your Streamlit app can call the deployed functions by name (no heavy compute on Streamlit).

---

## Overview

- **Modal apps**
  - `modal_infer_inmemory_latest.py` — CPU builds texts (no files), GPU runs inference, returns top‑K allocation for a given day.
  - `modal_plans_store.py` — lightweight persistence API: `save_plan`, `get_plan`, `list_plans`. (Stores plans in a Modal Volume and snapshots buy prices / shares at creation time.)
- **Streamlit app**
  - `streamlit_app.py` — UI that **calls Modal** functions to create plans, and **fetches live prices** (every page load) to compute P&L.

**Flow**  
1. Streamlit (on app load / button) → calls Modal **CPU** → **GPU** → plan JSON.  
2. Streamlit asks Modal **plans API** to **save** the plan; Modal snapshots **buy prices** and computes **shares**.  
3. Streamlit **fetches live prices** (cached 30s) using `yfinance` and computes P&L versus the saved buy prices.

---

## Prerequisites

- Python 3.11+ on your laptop (for Modal CLI).
- A Modal account.
- A GitHub repo for the Streamlit app (public or private).

Install CLI locally:
```bash
pip install -U modal
modal token new
# Follow the prompt to log in. This stores tokens in your user profile.
```

> **Security**: Keep your Modal tokens out of your Git repo. In Streamlit Cloud you’ll use **Secrets**.

---

## 1) Prepare Modal Volumes (weights + code)

We use two volumes (created on-demand by your app code):

- `model-cache` (version 2): stores your checkpoint under **`/ckpts/…`**
- `code-cache`  (version 1): stores your local package under **`/pkgs/…`** (either a `.whl` or a source folder)

### 1.1 Upload your checkpoint to `model-cache`

**Windows (PowerShell):**
```powershell
# Example paths — change to your actual checkpoint file
$LOCAL_CKPT = "C:\Users\name\Desktop\Stock_data\Training\proj_az_leg711_batch1\biex_listmle_final.pt"
modal volume put model-cache "$LOCAL_CKPT" /ckpts/biex_listmle_final.pt
```

**macOS / Linux:**
```bash
LOCAL_CKPT="/path/to/biex_listmle_final.pt"
modal volume put model-cache "$LOCAL_CKPT" /ckpts/biex_listmle_final.pt
```

> The Modal app expects the checkpoint at: **`/vol/models/ckpts/biex_listmle_final.pt`**.

### 1.2 Upload your package to `code-cache`

You have two options:

**A) Wheel file**  
Build a wheel (recommended), then upload:
```bash
# Example wheel path; adjust to your build artifact
LOCAL_WHL="/path/to/dist/interfusion_encoder-3.1-py3-none-any.whl"
modal volume put code-cache "$LOCAL_WHL" /pkgs/interfusion_encoder-3.1.whl
```

**B) Source directory**  
If you don’t have a wheel, upload the source folder:
```bash
LOCAL_PKG_DIR="/path/to/interfusion_encoder-3.1"
modal volume put code-cache "$LOCAL_PKG_DIR" /pkgs/interfusion_encoder-3.1
```

> The GPU function installs from **`/vol/code/pkgs/interfusion_encoder-3.1.whl`** if present, otherwise from **`/vol/code/pkgs/interfusion_encoder-3.1`** (source).

---

## 2) Deploy your Modal apps (once, from laptop)

From the folder containing your Modal app files:

```bash
# Inference app (CPU+GPU)
modal deploy modal_infer_inmemory_latest.py

# Plans storage API
modal deploy modal_plans_store.py
```

**What deploy does**  
- Builds images (pip installs, etc.) if needed and registers functions under app names:
  - `inmemory-latest-infer` → `build_payload_remote`, `run_infer_remote`
  - `inmemory-latest-plans` → `save_plan`, `get_plan`, `list_plans`
- Doesn’t keep machines running — you pay per execution (unless you explicitly keep containers warm).

> **Updating code later?** Re-run the same `modal deploy` command to publish changes.

---

## 3) Prepare your Streamlit repo

Your repo should contain at least:

```
streamlit_app.py
requirements.txt
runtime.txt
```

**`requirements.txt`** (example)
```
streamlit>=1.36
modal>=0.62
yfinance>=0.2.50
pandas>=2.2
numpy>=1.26
requests>=2.31
python-dateutil>=2.8.2
```

**`runtime.txt`**
```
3.11
```

> The app uses `yfinance` for live prices; ensure it’s in `requirements.txt`.

**`streamlit_app.py`**  
- Reads Modal tokens from **Streamlit Secrets**
- Looks up deployed functions via `Function.from_name(APP, "function_name")`
- On load (if >= **09:00 Australia/Melbourne** and no plan created today), auto‑creates a plan.
- Button **Create Plan Now** forces creation anytime.
- Displays latest plan by default; dropdown lets you select older plans.
- Fetches **live prices** (cached 30s) to compute P&L vs **buy prices** snapped when the plan was created.

---

## 4) Configure Streamlit Cloud

1. Push your repo to GitHub (public or private).
2. In Streamlit Cloud:
   - **Create app** → Connect your repo/branch/path (e.g., `streamlit_app.py`).
   - Set **Python version** from `runtime.txt` (3.11).
3. In **Settings → Secrets**, add:
   ```
   MODAL_TOKEN_ID = "ak-..."
   MODAL_TOKEN_SECRET = "as-..."
   # Optional overrides (if you changed names/paths)
   MODAL_APP_INFER = "inmemory-latest-infer"
   MODAL_APP_PLANS = "inmemory-latest-plans"
   CKPT_PATH = "/vol/models/ckpts/biex_listmle_final.pt"
   TOP_K = "20"
   INVEST_AMT = "1000"
   TEMP = "2.0"
   TIMEZONE = "Australia/Melbourne"
   AUTO_CREATE_ON_START = "1"
   ```

> **Private repos** are fine — ensure Streamlit Cloud has permission to read private repos in your GitHub account.

---

## 5) Daily ops and behavior

- **Auto-create at 9am Melbourne**: When someone opens the app (or it refreshes) at/after 09:00 and there’s no plan for **today**, Streamlit asks Modal to create one.
- **Buy prices**: When saving a plan (`save_plan`), Modal snapshots **buy prices** via `yfinance` and computes **shares = allocation / buy_price`. The plan persists these fields.
- **Live prices**: Each view call uses `yfinance` from Streamlit (cached ~30s). It tries:
  1) `Ticker(t).fast_info['last_price' | 'last_trade' | 'last_close']`
  2) `download(period="1d", interval="1m")`
  3) `download(period="5d", interval="1d")`
- **P&L math**:
  - `current_value = shares * current_price`
  - `buy_value = shares * buy_price`
  - `pnl_abs = current_value - buy_value`
  - `pnl_pct = (current_value / buy_value - 1) * 100`

**Unattended daily plan (optional):**  
If you want creation to occur even if nobody opens the app, add a **Modal scheduled function** in a small app (e.g., `modal_daily_cron.py`) that calls your inference + save at 09:00 Australia/Melbourne, then `modal deploy` it. This still bills only during execution (unless you keep warm containers).

---

## 6) Cost expectations

- **Deploying** an app in Modal: $0 by itself (no idle billing).
- **Billing is per-execution**: CPU/GPU/memory only while containers are running.
- **Warm containers**: Only if you configure autoscaling to keep them warm (e.g., `min_containers > 0` or long `scaledown_window`) will you pay while idle. Otherwise, scale‑to‑zero.

**Volumes billing** (storage) is separate from compute; your `model-cache` and `code-cache` persist independently.

---

## 7) Troubleshooting

- **“Modal auth failed”** in Streamlit
  - Add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in **Streamlit Secrets**.
  - If you leaked or rotated tokens, run `modal token new` locally and update Secrets.

- **“Checkpoint not found” or path mismatch**
  - Verify you uploaded to **`/ckpts/biex_listmle_final.pt`** in the `model-cache` volume.
  - Confirm `CKPT_PATH=/vol/models/ckpts/biex_listmle_final.pt` in Secrets/env.

- **`need at least one array to stack`**
  - Means the item list was empty on GPU step. Ensure a non-empty ticker list flows into `build_payload_remote`. (The provided Streamlit app passes your universe explicitly.)

- **Missing `yfinance` or pandas import error**
  - Ensure `requirements.txt` includes `yfinance` and `pandas`, and Python `runtime.txt` is `3.11`.
  - Reboot the Streamlit app after editing requirements.

- **Yahoo price unavailable for a symbol**
  - Some tickers (e.g., delisted) may not return data. The app tolerates missing prices by leaving blanks.

- **Wheel not picked up from `code-cache`**
  - Either bump the file name/version (e.g., `interfusion_encoder-3.2.whl`) or force reinstall in your GPU function:
    ```python
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--force-reinstall", target])
    ```

- **Private GitHub repo issues**
  - Make sure Streamlit Cloud has **private repo access** to your account, or re-authorize the GitHub connection in Streamlit.

---

## 8) Updating things

- **Change inference code** → `modal deploy modal_infer_inmemory_latest.py`
- **Change plans API** → `modal deploy modal_plans_store.py`
- **Change weights or package in volumes** → just `modal volume put ...`; no redeploy required (unless you hard‑coded filenames).
- **Change Streamlit UI** → push to GitHub; Streamlit rebuilds automatically.

---

## 9) Quick sanity check (local)

You can run a one‑off job locally invoking Modal via CLI to confirm the inference path (no Streamlit needed):

```bash
modal run modal_infer_inmemory_latest.py -- \
  --ckpt /vol/models/ckpts/biex_listmle_final.pt \
  --invest_amt 1000 \
  --top_k 20
```

You should see a printed table with top‑K tickers and allocations for the effective date.

---

## 10) Environment variables (Streamlit)

- `MODAL_APP_INFER` (default `inmemory-latest-infer`)
- `MODAL_APP_PLANS` (default `inmemory-latest-plans`)
- `CKPT_PATH` (default `/vol/models/ckpts/biex_listmle_final.pt`)
- `TOP_K` (default `20`)
- `INVEST_AMT` (default `1000`)
- `TEMP` (default `2.0`)
- `TIMEZONE` (default `Australia/Melbourne`)
- `AUTO_CREATE_ON_START` (`1` to enable)

Set these in **Streamlit Secrets** or the environment.

---

## 11) Security checklist

- Keep Modal tokens **only** in Streamlit Secrets (or local OS keyring). Never commit tokens.
- If tokens were exposed, **rotate** them: `modal token new` → update Streamlit Secrets → revoke old token.
- Keep the GitHub repo **private** if you include infrastructure details you don’t want public.

---

## 12) File map (example)

```
your-repo/
├── streamlit_app.py              # UI (already implemented)
├── requirements.txt              # includes streamlit, modal, yfinance, pandas, numpy, etc.
├── runtime.txt                   # "3.11"
├── modal_infer_inmemory_latest.py# Modal CPU+GPU app (you deploy from laptop)
├── modal_plans_store.py          # Modal plan persistence API (you deploy from laptop)
└── .streamlit/
    └── secrets.toml              # (optional for local dev)
```
