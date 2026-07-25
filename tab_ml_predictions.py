"""
ML Predictions Tab - st.cache_data PERSISTENT CACHE  (v5)

Changes v5:
  - Replaced the separate "Quick View" selectbox panel with an inline
    Live Market Table that shows all predicted stocks at once:
      · Live price, day change %, intraday progress bar toward target,
        and a direct TradingView link — all in one unified view
      · No more picking stocks one-by-one
      · Removed the confusing "vs Pred Price" metric
      · "vs pred price" replaced with cleaner "Move Since Signal" label
      · Progress bar renders as a styled HTML column inside the table area
  - Caching system completely unchanged from v4

CACHE STRATEGY: UNCHANGED — all Supabase fetching methods identical to v4.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from db import get_supabase_client
from chart_utils import CHART_THEME, LAYOUT, AXIS_STYLE, COLORS, SIGNAL_COLORS, SIGNAL_BG, CONFUSION_COLORS

TAB_ID = "ml_predictions"

# Row cap used by _get_table_full — surfaced in the UI when a fetch hits it,
# so a truncated view doesn't silently look like a complete one.
_TABLE_ROW_CAP = 500

_DATE_COL = {
    "ml_explosion_predictions": "prediction_date",
    "ml_prediction_accuracy":   "prediction_date",
    "ml_missed_opportunities":  "detection_date",
    "ml_screening_logs":        "screening_date",
    "ml_feature_importance":   "training_date",
}

_SELECT = {
    "ml_explosion_predictions": (
        "prediction_date,symbol,exchange,signal,explosion_probability,"
        "current_price,target_price,target_gain_pct,target_price_low,target_price_high"
    ),
    "ml_prediction_accuracy": (
        "symbol,prediction_date,predicted_probability,predicted_signal,"
        "predicted_target_gain,became_winner,actual_gain_pct,actual_high_pct,"
        "actual_price,prediction_correct,gain_error_pct"
    ),
    "ml_missed_opportunities": (
        "symbol,detection_date,exchange,actual_gain_pct,actual_high_pct,"
        "actual_price,actual_volume,was_screened,screening_failure_reason,"
        "predicted_probability,predicted_signal"
    ),
    "ml_screening_logs": "*",
    "ml_feature_importance": "training_date,feature_name,importance,rank",
}

# Tables that are known/required — used for cache refresh and System Info.
_ALL_TABLES = [t for t in _DATE_COL.keys() if t != "ml_feature_importance"]
# Tables that may not exist yet in every deployment — fetched quietly, no
# warnings if missing, since they're an optional enhancement (see
# _get_table_optional below).
_OPTIONAL_TABLES = ["ml_feature_importance"]

# ── TradingView exchange prefix mapping ───────────────────────────────────────
_TV_EXCHANGE_MAP = {
    "NASDAQ":    "NASDAQ",
    "NYSE":      "NYSE",
    "AMEX":      "AMEX",
    "NYSE ARCA": "AMEX",
    "NYSEARCA":  "AMEX",
    "BATS":      "BATS",
    "OTC":       "OTC",
    "OTCMKTS":   "OTC",
    "OTCBB":     "OTC",
}

def _tv_url(symbol: str, exchange: str) -> str:
    prefix = _TV_EXCHANGE_MAP.get(str(exchange).strip().upper(), "NASDAQ")
    return f"https://www.tradingview.com/chart/?symbol={prefix}:{symbol.upper()}"


# ── Live price fetch (short TTL — totally separate from Supabase cache) ───────
@st.cache_data(ttl=60, show_spinner=False)
def _get_live_quote(symbol: str) -> dict | None:
    """
    Fetch a lightweight live quote via yfinance.fast_info.
    TTL=60s so data refreshes every minute without hammering Yahoo.
    Returns None gracefully if yfinance is not installed or the call fails.
    """
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).fast_info
        return {
            "last_price": getattr(info, "last_price",      None),
            "day_high":   getattr(info, "day_high",        None),
            "day_low":    getattr(info, "day_low",         None),
            "open":       getattr(info, "open",            None),
            "prev_close": getattr(info, "previous_close",  None),
            "volume":     getattr(info, "last_volume",     None),
        }
    except ImportError:
        return None
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def _get_bulk_live_quotes(symbols: tuple) -> dict:
    """
    Fetch live quotes for multiple symbols in one yfinance call.
    Returns dict[symbol -> quote_dict].  TTL=60s, same as single-symbol fetch.
    symbols must be a tuple (hashable) for st.cache_data to work.
    """
    result = {}
    try:
        import yfinance as yf
        tickers = yf.Tickers(" ".join(symbols))
        for sym in symbols:
            try:
                info = tickers.tickers[sym].fast_info
                result[sym] = {
                    "last_price": getattr(info, "last_price",     None),
                    "day_high":   getattr(info, "day_high",       None),
                    "day_low":    getattr(info, "day_low",        None),
                    "open":       getattr(info, "open",           None),
                    "prev_close": getattr(info, "previous_close", None),
                    "volume":     getattr(info, "last_volume",    None),
                }
            except Exception:
                result[sym] = None
    except ImportError:
        pass
    except Exception:
        pass
    return result


# ── Cached DB fetchers (UNCHANGED) ────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _get_table_full(table_name: str) -> pd.DataFrame:
    try:
        client     = get_supabase_client()
        date_col   = _DATE_COL.get(table_name)
        select_str = _SELECT.get(table_name, "*")
        query      = client.table(table_name).select(select_str)
        if date_col:
            query = query.order(date_col, desc=True)
        response = query.limit(_TABLE_ROW_CAP).execute()
        df = pd.DataFrame(response.data) if response.data else pd.DataFrame()
        # Tag whether this fetch hit the row cap, so callers can warn the user
        # instead of silently showing a truncated "Latest Predictions" view.
        df.attrs["truncated"] = len(df) >= _TABLE_ROW_CAP
        return df
    except Exception as e:
        st.warning(f"Could not load `{table_name}`: {e}")
        return pd.DataFrame()


def _get_table_optional(table_name: str) -> pd.DataFrame:
    """
    Like _get_table_full, but for tables that are an optional enhancement
    (e.g. feature importance) and may not exist in every deployment yet.
    Fails silently instead of showing a warning banner.
    """
    try:
        client     = get_supabase_client()
        date_col   = _DATE_COL.get(table_name)
        select_str = _SELECT.get(table_name, "*")
        query      = client.table(table_name).select(select_str)
        if date_col:
            query = query.order(date_col, desc=True)
        response = query.limit(_TABLE_ROW_CAP).execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _get_table_all(table_name: str) -> pd.DataFrame:
    """
    Like _get_table_full but paginates through the ENTIRE table instead of
    stopping at 500 rows. Used for charts that need full history (e.g. the
    Performance Trends tab) rather than just the most recent slice.
    """
    try:
        client     = get_supabase_client()
        date_col   = _DATE_COL.get(table_name)
        select_str = _SELECT.get(table_name, "*")
        page_size  = 1000
        offset     = 0
        frames     = []
        while True:
            query = client.table(table_name).select(select_str)
            if date_col:
                query = query.order(date_col, desc=False)
            response = query.range(offset, offset + page_size - 1).execute()
            rows = response.data or []
            if not rows:
                break
            frames.append(pd.DataFrame(rows))
            if len(rows) < page_size:
                break
            offset += page_size
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load full history for `{table_name}`: {e}")
        return pd.DataFrame()


# ── Statistics helpers (model diagnostics) ────────────────────────────────────
def _wilson_ci(k: int, n: int, z: float = 1.96):
    """
    Wilson score interval for a binomial proportion. Far better-behaved than
    a normal-approximation interval for small n or extreme proportions (both
    common here — a single day can have only a handful of positive calls).
    Returns (lower, upper) as fractions in [0, 1]. (nan, nan) if n == 0.
    """
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z*z / (2*n)) / denom
    half = (z * np.sqrt((phat*(1-phat) + z*z/(4*n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    """
    Minimal ROC curve + AUC computed from scratch (no sklearn dependency).
    Returns (fpr, tpr, auc). y_true is 0/1, y_score is a continuous score.
    """
    order = np.argsort(-y_score)
    y_true = y_true[order]
    P = y_true.sum()
    N = len(y_true) - P
    if P == 0 or N == 0:
        return None, None, np.nan
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    tpr = np.concatenate(([0.0], tps / P, [1.0]))
    fpr = np.concatenate(([0.0], fps / N, [1.0]))
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, auc


def _pr_curve(y_true: np.ndarray, y_score: np.ndarray):
    """
    Minimal precision-recall curve + average precision, computed from
    scratch. Returns (recall, precision, avg_precision).
    """
    order = np.argsort(-y_score)
    y_true = y_true[order]
    P = y_true.sum()
    if P == 0:
        return None, None, np.nan
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    precision = tps / (tps + fps)
    recall = tps / P
    # Average precision: step-wise area under PR curve (matches sklearn's
    # definition, weighting by the change in recall).
    recall_pad = np.concatenate(([0.0], recall))
    precision_pad = np.concatenate(([precision[0]], precision))
    ap = np.sum(np.diff(recall_pad) * precision_pad[1:])
    return recall, precision, ap


def _calibration_bins(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10):
    """
    Reliability-diagram bins: for each bin of predicted probability, the mean
    predicted probability vs. the empirical win rate, plus bin size. Uses
    quantile-based bins so each bin has similar sample size (fixed-width bins
    are misleading here since most scores cluster in a narrow range).
    """
    df = pd.DataFrame({"y": y_true, "p": y_score}).dropna()
    if len(df) < n_bins:
        n_bins = max(1, len(df) // 2) or 1
    if n_bins == 0 or df.empty:
        return pd.DataFrame(columns=["mean_pred", "empirical_rate", "count"])
    try:
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=["mean_pred", "empirical_rate", "count"])
    grouped = df.groupby("bin", observed=True).agg(
        mean_pred=("p", "mean"),
        empirical_rate=("y", "mean"),
        count=("y", "size"),
    ).reset_index(drop=True)
    return grouped


def _brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_score)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean((y_score[mask] - y_true[mask]) ** 2))


# ── Cache control (UNCHANGED) ─────────────────────────────────────────────────
def clear_all_cache():
    _get_table_full.clear()
    _get_live_quote.clear()
    _get_bulk_live_quotes.clear()


def refresh_cache():
    any_new = False
    for table in _ALL_TABLES:
        existing = _get_table_full(table)
        date_col = _DATE_COL.get(table)
        if existing.empty or not date_col or date_col not in existing.columns:
            continue
        latest_cached = existing[date_col].max()
        try:
            client     = get_supabase_client()
            select_str = _SELECT.get(table, "*")
            response   = (
                client.table(table)
                .select(select_str)
                .gt(date_col, latest_cached)
                .order(date_col, desc=True)
                .limit(500)
                .execute()
            )
            new_rows = pd.DataFrame(response.data) if response.data else pd.DataFrame()
        except Exception as e:
            st.warning(f"Could not check new rows for `{table}`: {e}")
            continue
        if not new_rows.empty:
            any_new = True

    if any_new:
        _get_table_full.clear()
        for table in _ALL_TABLES:
            _get_table_full(table)
        st.toast("✅ New data fetched and cached.")
    else:
        st.toast("✅ Cache is already up to date — no new data found.")


# ── Shared button helper (UNCHANGED) ─────────────────────────────────────────
def _render_cache_buttons(tab_id: str):
    confirm_key = f"{tab_id}_confirm_clear"
    col_r, col_c, _ = st.columns([1, 1, 5])
    with col_r:
        st.markdown('<div class="btn-refresh">', unsafe_allow_html=True)
        refresh = st.button("🔄 Refresh", key=f"{tab_id}_refresh", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_c:
        st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
        clear = st.button("🗑️ Clear Cache", key=f"{tab_id}_clear", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if clear:
        st.session_state[confirm_key] = True

    confirmed = False
    if st.session_state.get(confirm_key):
        st.markdown(
            '<div class="cache-warning">⚠️ This will wipe ALL cached data. '
            'Click <strong>Confirm Clear</strong> to proceed.</div>',
            unsafe_allow_html=True
        )
        cc1, cc2, _ = st.columns([1, 1, 5])
        with cc1:
            st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
            if st.button("✓ Confirm Clear", key=f"{tab_id}_confirm_yes", use_container_width=True):
                confirmed = True
                st.session_state[confirm_key] = False
            st.markdown('</div>', unsafe_allow_html=True)
        with cc2:
            if st.button("✕ Cancel", key=f"{tab_id}_confirm_no", use_container_width=True):
                st.session_state[confirm_key] = False
                st.rerun()
    return refresh, confirmed


# ── Main entry point ───────────────────────────────────────────────────────────
def render_ml_predictions_tab():
    refresh_clicked, clear_confirmed = _render_cache_buttons(TAB_ID)

    if clear_confirmed:
        clear_all_cache()
        st.rerun()
    if refresh_clicked:
        refresh_cache()
        st.rerun()

    st.subheader("ML Explosion Predictions")
    st.caption(
        "Autonomous system — screens 500–1500 stocks daily, generates predictions "
        "with target gains, and tracks accuracy."
    )

    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
        "Latest Predictions",
        "Predictions vs Actuals",
        "Missed Opportunities",
        "Performance Trends",
        "System Info",
    ])

    with subtab1: _render_latest_predictions()
    with subtab2: _render_predictions_vs_actuals()
    with subtab3: _render_missed_opportunities()
    with subtab4: _render_performance_trends()
    with subtab5: _render_system_info()


# ── Live Market Table — renders all stocks inline ─────────────────────────────
def _render_live_market_table(fdf: pd.DataFrame):
    """
    Dense two-line card layout per stock. Shows:
      Row 1: Ticker + exchange + signal badge + explosion probability
             | Signal price → Live price | Day change | Day high
      Row 2: Progress bar (day high vs target price) with target % and $ labels
             | Low / prev close sub-labels | Volume
    """
    if fdf.empty:
        return

    symbols = tuple(fdf["symbol"].tolist())

    with st.spinner(f"Fetching live quotes for {len(symbols)} stocks… (cached 60 s)"):
        quotes = _get_bulk_live_quotes(symbols)

    yf_available = bool(quotes)

    # ── Header strip ───────────────────────────────────────────────────────────
    col_hdr, col_ts = st.columns([5, 2])
    with col_hdr:
        st.markdown(
            '<span style="font-family:var(--font-body);font-size:0.58rem;font-weight:500;'
            'letter-spacing:0.2em;text-transform:uppercase;color:var(--text-2);">Live Market View</span>',
            unsafe_allow_html=True,
        )
    with col_ts:
        from datetime import datetime as _dt
        st.markdown(
            f'<div style="text-align:right;font-family:var(--font-body);font-size:0.58rem;'
            f'color:var(--text-3);">quotes cached 60 s · {_dt.now().strftime("%H:%M:%S")}</div>',
            unsafe_allow_html=True,
        )

    if not yf_available:
        st.warning("⚠️ Live quotes unavailable — install `yfinance` or check network.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Helpers ────────────────────────────────────────────────────────────────
    _sig_colors = {
        "STRONG BUY": ("var(--green-bright)", "rgba(16,185,129,0.12)", "rgba(16,185,129,0.25)"),
        "BUY":        ("var(--cyan)",          "rgba(0,212,255,0.08)",  "rgba(0,212,255,0.2)"),
        "HOLD":       ("var(--amber-bright)",  "rgba(245,158,11,0.08)", "rgba(245,158,11,0.2)"),
        "AVOID":      ("var(--red-bright)",    "rgba(239,68,68,0.08)",  "rgba(239,68,68,0.2)"),
    }
    _left_bar = {
        "STRONG BUY": "var(--green-bright)",
        "BUY":        "var(--cyan)",
        "HOLD":       "var(--amber-bright)",
        "AVOID":      "var(--red-bright)",
    }

    def _badge(signal: str) -> str:
        fg, bg, border = _sig_colors.get(signal, ("var(--text-2)", "var(--bg-3)", "var(--border)"))
        return (
            f'<span style="padding:3px 9px;border-radius:3px;background:{bg};'
            f'border:1px solid {border};color:{fg};font-family:var(--font-body);'
            f'font-size:0.65rem;font-weight:700;letter-spacing:0.08em;'
            f'text-transform:uppercase;white-space:nowrap;">{signal}</span>'
        )

    def _val(v, fmt="$.2f", fallback="—"):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return fallback
        if fmt.startswith("$"):
            return f"${v:{fmt[1:]}}"
        return f"{v:{fmt}}"

    def _chg_color(v):
        if v is None: return "var(--text-1)"
        return "var(--green-bright)" if v >= 0 else "var(--red-bright)"

    def _progress_bar(high_pct: float | None, target_pct: float | None,
                      target_price: float | None, pred_price: float | None) -> str:
        """
        Progress bar driven by day-high % gain vs target % gain.
        Shows: actual high $ | bar | target price $
        Below bar: day-high gain % and target gain % labels.
        """
        if target_pct is None or target_pct <= 0:
            tgt_str = _val(target_price, "$.2f")
            return (
                f'<div style="font-family:var(--font-body);font-size:0.72rem;color:var(--text-1);">'
                f'Target {tgt_str} · no target % set</div>'
            )

        progress  = min(max((high_pct or 0) / target_pct, 0.0), 1.0)
        hit       = progress >= 1.0
        bar_color = "var(--green-bright)" if hit else "var(--cyan)"
        lbl_color = "var(--green-bright)" if hit else "var(--text-0)"
        bar_w     = progress * 100

        high_str  = _val(
            (pred_price * (1 + (high_pct or 0) / 100)) if pred_price and high_pct is not None else None,
            "$.2f"
        )
        tgt_str   = _val(target_price, "$.2f")
        hit_label = "✓ TARGET HIT" if hit else f"{progress*100:.0f}%"

        return (
            # Top row: high $ ←bar→ target $
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<span style="font-family:var(--font-body);font-size:0.7rem;'
            f'color:var(--text-1);white-space:nowrap;min-width:56px;">H {high_str}</span>'
            f'<div style="flex:1;position:relative;">'
            f'<div style="height:6px;background:var(--bg-4);border-radius:3px;overflow:hidden;">'
            f'<div style="height:100%;width:{bar_w:.1f}%;background:{bar_color};'
            f'border-radius:3px;transition:width .4s ease;"></div>'
            f'</div></div>'
            f'<span style="font-family:var(--font-body);font-size:0.7rem;'
            f'color:var(--text-1);white-space:nowrap;min-width:56px;text-align:right;">{tgt_str} ↑</span>'
            f'</div>'
            # Bottom row: high gain % · progress label · target gain %
            f'<div style="display:flex;justify-content:space-between;margin-top:4px;">'
            f'<span style="font-family:var(--font-body);font-size:0.68rem;color:var(--text-1);">'
            f'High {("+" if (high_pct or 0) >= 0 else "")}{(high_pct or 0):.2f}%</span>'
            f'<span style="font-family:var(--font-body);font-size:0.7rem;'
            f'font-weight:700;color:{lbl_color};">{hit_label}</span>'
            f'<span style="font-family:var(--font-body);font-size:0.68rem;color:var(--text-1);">'
            f'Target +{target_pct:.1f}%</span>'
            f'</div>'
        )

    # ── Column header row ──────────────────────────────────────────────────────
    hdr_style = (
        "font-family:var(--font-body);font-size:0.6rem;letter-spacing:0.14em;"
        "text-transform:uppercase;color:var(--text-1);padding-bottom:6px;"
        "border-bottom:1px solid var(--border-mid);"
    )
    h = st.columns([2.6, 1.4, 1.1, 1.1, 1.3, 1.3, 4.2, 0.7])
    for col, lbl in zip(h, ["Stock", "Signal", "Prob", "Entry", "Live", "Day High", "→ Target", "TV"]):
        col.markdown(f'<div style="{hdr_style}">{lbl}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── One row per stock ──────────────────────────────────────────────────────
    for _, row in fdf.iterrows():
        sym        = row["symbol"]
        exchange   = row.get("exchange", "NASDAQ")
        signal     = str(row.get("signal", "—"))
        prob       = row.get("explosion_probability")   # 0–1 float
        pred_price = row.get("current_price")
        tgt_gain   = row.get("target_gain_pct")
        tgt_price  = row.get("target_price")
        tgt_low    = row.get("target_price_low")
        tgt_high   = row.get("target_price_high")
        tv_url     = _tv_url(sym, exchange)

        q          = quotes.get(sym) if quotes else None
        live       = q["last_price"] if q else None
        prev       = q["prev_close"] if q else None
        day_high   = q["day_high"]   if q else None
        day_low    = q["day_low"]    if q else None
        volume     = q["volume"]     if q else None

        day_chg    = (live - prev)     / prev     * 100 if live and prev     else None
        high_gain  = (day_high - pred_price) / pred_price * 100 if day_high and pred_price else None

        left_color = _left_bar.get(signal, "var(--border-mid)")

        # Volume formatting
        if volume:
            vol_str = f"{volume/1e6:.1f}M" if volume >= 1_000_000 else f"{volume/1e3:.0f}K"
        else:
            vol_str = "—"

        # Probability bar (inline mini)
        prob_pct = (prob * 100) if prob is not None else None
        prob_bar = ""
        if prob_pct is not None:
            p_color = (
                "var(--green-bright)" if prob_pct >= 70
                else "var(--cyan)" if prob_pct >= 50
                else "var(--amber-bright)"
            )
            prob_bar = (
                f'<div style="height:3px;background:var(--bg-4);border-radius:2px;'
                f'margin-top:4px;overflow:hidden;">'
                f'<div style="height:100%;width:{prob_pct:.0f}%;background:{p_color};'
                f'border-radius:2px;"></div></div>'
            )

        # Target price range sub-label
        tgt_range_str = ""
        if tgt_low and tgt_high:
            tgt_range_str = (
                f'<div style="font-family:var(--font-body);font-size:0.66rem;'
                f'color:var(--text-1);margin-top:3px;">${tgt_low:.2f} – ${tgt_high:.2f}</div>'
            )

        # Low of day sub-label
        low_str = (
            f'<div style="font-family:var(--font-body);font-size:0.68rem;'
            f'color:var(--text-1);margin-top:2px;">L ${day_low:.2f}</div>'
        ) if day_low else ""

        cols = st.columns([2.6, 1.4, 1.1, 1.1, 1.3, 1.3, 4.2, 0.7])

        # Col 0 — Ticker + exchange + volume
        with cols[0]:
            st.markdown(
                f'<div style="padding:7px 0 5px;border-left:2px solid {left_color};padding-left:10px;">'
                f'<span style="font-family:var(--font-body);font-size:0.95rem;'
                f'font-weight:700;color:var(--text-0);letter-spacing:0.04em;">{sym}</span>'
                f'<span style="font-family:var(--font-body);font-size:0.66rem;'
                f'color:var(--text-1);margin-left:7px;">{exchange}</span>'
                f'<div style="font-family:var(--font-body);font-size:0.68rem;'
                f'color:var(--text-1);margin-top:2px;">Vol {vol_str}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Col 1 — Signal badge
        with cols[1]:
            st.markdown(
                f'<div style="padding:8px 0 5px;">{_badge(signal)}</div>',
                unsafe_allow_html=True,
            )

        # Col 2 — Explosion probability + mini bar
        with cols[2]:
            prob_str = f"{prob_pct:.0f}%" if prob_pct is not None else "—"
            prob_color = (
                "var(--green-bright)" if (prob_pct or 0) >= 70
                else "var(--cyan)" if (prob_pct or 0) >= 50
                else "var(--amber-bright)"
            )
            st.markdown(
                f'<div style="padding:7px 0 5px;">'
                f'<span style="font-family:var(--font-body);font-size:0.9rem;'
                f'font-weight:700;color:{prob_color};">{prob_str}</span>'
                f'{prob_bar}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Col 3 — Entry / signal price
        with cols[3]:
            st.markdown(
                f'<div style="padding:7px 0 5px;">'
                f'<span style="font-family:var(--font-body);font-size:0.88rem;'
                f'color:var(--text-0);">{_val(pred_price, "$.2f")}</span>'
                f'<div style="font-family:var(--font-body);font-size:0.66rem;'
                f'color:var(--text-1);margin-top:2px;">entry</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Col 4 — Live price + day change
        with cols[4]:
            chg_color = _chg_color(day_chg)
            chg_str   = f'{day_chg:+.2f}%' if day_chg is not None else "—"
            live_str  = _val(live, "$.2f")
            st.markdown(
                f'<div style="padding:7px 0 5px;">'
                f'<span style="font-family:var(--font-body);font-size:0.9rem;'
                f'font-weight:700;color:var(--text-0);">{live_str}</span>'
                f'<div style="font-family:var(--font-body);font-size:0.72rem;'
                f'font-weight:600;color:{chg_color};margin-top:2px;">{chg_str}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Col 5 — Day high + low sub-label
        with cols[5]:
            high_chg_str = f'{high_gain:+.2f}%' if high_gain is not None else ""
            high_color   = _chg_color(high_gain)
            st.markdown(
                f'<div style="padding:7px 0 5px;">'
                f'<span style="font-family:var(--font-body);font-size:0.9rem;'
                f'font-weight:700;color:{high_color};">{_val(day_high, "$.2f")}</span>'
                f'<div style="font-family:var(--font-body);font-size:0.72rem;'
                f'color:var(--text-1);margin-top:2px;">{high_chg_str}</div>'
                f'{low_str}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Col 6 — Progress bar + target info
        with cols[6]:
            st.markdown(
                f'<div style="padding:7px 0 5px;">'
                f'{_progress_bar(high_gain, tgt_gain, tgt_price, pred_price)}'
                f'{tgt_range_str}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Col 7 — TradingView link
        with cols[7]:
            st.markdown(
                f'<div style="padding:6px 0 5px;text-align:center;">'
                f'<a href="{tv_url}" target="_blank" style="'
                f'display:inline-flex;align-items:center;justify-content:center;'
                f'width:30px;height:30px;background:transparent;'
                f'border:1px solid var(--border-mid);border-radius:var(--radius-sm);'
                f'color:var(--text-1);font-size:0.85rem;text-decoration:none;"'
                f'title="Open {sym} in TradingView"'
                f'onmouseover="this.style.borderColor=\'var(--cyan-border)\';'
                f'this.style.color=\'var(--cyan)\';this.style.background=\'var(--cyan-dim)\'"'
                f'onmouseout="this.style.borderColor=\'var(--border-mid)\';'
                f'this.style.color=\'var(--text-1)\';this.style.background=\'transparent\'">'
                f'↗</a></div>',
                unsafe_allow_html=True,
            )

        # Row divider
        st.markdown(
            '<div style="height:1px;background:var(--border-mid);margin:2px 0 4px;"></div>',
            unsafe_allow_html=True,
        )

    st.caption(
        f"{len(fdf)} stocks · Bar = intraday high % of the way to target gain · "
        f"Entry = price at time of prediction · Live data via Yahoo Finance · not financial advice"
    )


# ── Sub-tab 1 — Latest Predictions ────────────────────────────────────────────
def _render_latest_predictions():
    all_preds = _get_table_full("ml_explosion_predictions")

    if all_preds.empty:
        st.warning("No predictions available yet.")
        st.info("Run the screening workflow or wait for the scheduled run.")
        return

    dates = sorted(all_preds["prediction_date"].unique().tolist(), reverse=True)

    col_date, col_notice = st.columns([2, 3])
    with col_date:
        selected_date = st.selectbox(
            "Date",
            dates,
            format_func=lambda x: datetime.fromisoformat(x).strftime("%a %d %b %Y"),
            key=f"{TAB_ID}_pred_date",
        )
    with col_notice:
        pred_dt = datetime.fromisoformat(selected_date).date()
        st.markdown("<div style='padding-top:28px'>", unsafe_allow_html=True)
        if pred_dt >= datetime.now().date():
            st.success("Forward-looking predictions")
        else:
            st.info("Historical — see Predictions vs Actuals for outcomes")
        st.markdown("</div>", unsafe_allow_html=True)

    df = all_preds[all_preds["prediction_date"] == selected_date].copy()
    if df.empty:
        st.warning(f"No predictions for {selected_date}")
        return

    # Delta vs previous date
    prev_df = pd.DataFrame()
    date_idx = dates.index(selected_date)
    if date_idx + 1 < len(dates):
        prev_date = dates[date_idx + 1]
        prev_df   = all_preds[all_preds["prediction_date"] == prev_date].copy()

    # ── Summary metrics ────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Screened",
        len(df),
        delta=f"{len(df)-len(prev_df):+.0f} vs prev" if not prev_df.empty else None,
    )
    col2.metric("Strong Buy", int((df["signal"] == "STRONG BUY").sum()))
    col3.metric("Buy",        int((df["signal"] == "BUY").sum()))
    col4.metric(
        "Avg Probability",
        f"{df['explosion_probability'].mean() * 100:.1f}%",
        delta=(
            f"{(df['explosion_probability'].mean() - prev_df['explosion_probability'].mean()) * 100:+.1f}% vs prev"
            if not prev_df.empty else None
        ),
    )
    col5.metric(
        "Avg Target Gain",
        f"+{df['target_gain_pct'].mean():.1f}%",
        delta=(
            f"{df['target_gain_pct'].mean() - prev_df['target_gain_pct'].mean():+.1f}% vs prev"
            if not prev_df.empty else None
        ),
    )

    # ── Signal win rates (from accuracy data) ─────────────────────────────────
    acc_df = _get_table_full("ml_prediction_accuracy")
    if not acc_df.empty and "predicted_signal" in acc_df.columns:
        acc_copy = acc_df.copy()
        acc_copy["became_winner"] = acc_copy["became_winner"].astype(bool)
        sig_win = acc_copy.groupby("predicted_signal")["became_winner"].mean() * 100

        st.markdown("---")
        st.markdown("#### All-Time Win Rate by Signal")
        sig_order = ["STRONG BUY", "BUY", "HOLD", "AVOID"]
        wr_cols   = st.columns(len(sig_order))
        for i, sig in enumerate(sig_order):
            if sig in sig_win.index:
                wr    = sig_win[sig]
                with wr_cols[i]:
                    st.metric(sig.title(), f"{wr:.1f}%")
                    st.progress(int(min(wr, 100)))

    # ── Charts ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        fig = go.Figure(go.Histogram(
            x=df["explosion_probability"] * 100,
            nbinsx=20,
            marker_color=COLORS["primary"],
            opacity=0.85,
        ))
        fig.update_layout(
            title="Probability Distribution",
            xaxis_title="Probability (%)",
            yaxis_title="Count",
            height=300,
            showlegend=False,
            **LAYOUT,
        )
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        sc  = df["signal"].value_counts()
        fig = go.Figure(go.Pie(
            labels=sc.index,
            values=sc.values,
            marker=dict(colors=[SIGNAL_COLORS.get(s, "#999") for s in sc.index]),
            hole=0.5,
            textinfo="label+percent",
            textfont=dict(size=11, family="DM Mono, monospace"),
        ))
        fig.update_layout(title="Signal Breakdown", height=300, **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── Filters ────────────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Filter predictions", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sig_filter = st.multiselect(
                "Signal",
                ["STRONG BUY", "BUY", "HOLD", "AVOID"],
                default=["STRONG BUY", "BUY", "HOLD", "AVOID"],
                key=f"{TAB_ID}_sig_f",
            )
        with fc2:
            min_prob = st.slider("Min Probability %", 0, 100, 0, key=f"{TAB_ID}_prob_f")
        with fc3:
            min_tgt  = st.slider("Min Target Gain %", 0, 50, 0, key=f"{TAB_ID}_tgt_f")

    fdf = df[
        df["signal"].isin(sig_filter) &
        (df["explosion_probability"] >= min_prob / 100) &
        (df["target_gain_pct"] >= min_tgt)
    ].copy()

    # Sort by signal strength then probability
    signal_order = {"STRONG BUY": 0, "BUY": 1, "HOLD": 2, "AVOID": 3}
    fdf["_sig_rank"] = fdf["signal"].map(signal_order).fillna(9)
    fdf = fdf.sort_values(["_sig_rank", "explosion_probability"], ascending=[True, False]).drop(columns=["_sig_rank"])

    st.caption(f"{len(fdf)} stocks match current filters")

    if fdf.empty:
        st.warning("No stocks match the filters.")
        return

    # ── View switcher — tab-style, instant, no rerun ──────────────────────────
    view_tab_static, view_tab_live = st.tabs(["🗃 Predictions Table", "📡 Live Market View"])

    with view_tab_static:
        # ── Original static dataframe table (default) ──────────────────────────
        fdf_display = fdf.copy()
        fdf_display["explosion_probability"] = fdf_display["explosion_probability"] * 100

        def _highlight_sig(row):
            return [f"background-color: {SIGNAL_BG.get(row['signal'], '')}"] * len(row)

        display_cols = [
            c for c in [
                "symbol", "exchange", "signal", "explosion_probability",
                "current_price", "target_price", "target_gain_pct",
                "target_price_low", "target_price_high",
            ] if c in fdf_display.columns
        ]
        st.dataframe(
            fdf_display[display_cols].style.format(
                {
                    "explosion_probability": "{:.2f}%",
                    "current_price":         "${:.2f}",
                    "target_price":          "${:.2f}",
                    "target_price_low":      "${:.2f}",
                    "target_price_high":     "${:.2f}",
                    "target_gain_pct":       "+{:.2f}%",
                },
                na_rep="—",
            ).apply(_highlight_sig, axis=1),
            use_container_width=True,
            height=420,
        )
        st.download_button(
            "Download CSV",
            fdf[display_cols].to_csv(index=False),
            f"ml_predictions_{selected_date}.csv",
            "text/csv",
            key=f"{TAB_ID}_dl",
        )

    with view_tab_live:
        # ── Live Market Table — all stocks, inline ─────────────────────────────
        _render_live_market_table(fdf)


# ── Sub-tab 2 — Predictions vs Actuals ────────────────────────────────────────
def _render_predictions_vs_actuals():
    st.markdown("#### Prediction Accuracy Analysis")

    all_acc = _get_table_full("ml_prediction_accuracy")
    if all_acc.empty:
        st.warning("No accuracy data available yet.")
        return

    dates = sorted(all_acc["prediction_date"].unique().tolist(), reverse=True)
    selected_date = st.selectbox(
        "Date",
        dates,
        format_func=lambda x: datetime.fromisoformat(x).strftime("%a %d %b %Y"),
        key=f"{TAB_ID}_acc_date",
    )

    df = all_acc[all_acc["prediction_date"] == selected_date].copy()
    if df.empty:
        st.warning(f"No accuracy data for {selected_date}")
        return

    df["became_winner"]         = df["became_winner"].astype(bool)
    df["prediction_correct"]    = df["prediction_correct"].astype(bool)
    df["predicted_probability"] = pd.to_numeric(df["predicted_probability"], errors="coerce")
    df["actual_gain_pct"]       = pd.to_numeric(df["actual_gain_pct"],       errors="coerce")
    df["actual_high_pct"]       = pd.to_numeric(df["actual_high_pct"],       errors="coerce")
    df["gain_error_pct"]        = pd.to_numeric(df["gain_error_pct"],        errors="coerce")

    pos_mask          = df["predicted_signal"].isin(["STRONG BUY", "BUY"])
    total             = len(df)
    correct           = int(df["prediction_correct"].sum())
    actual_winners    = int(df["became_winner"].sum())
    predicted_winners = int(pos_mask.sum())
    tp = int((pos_mask &  df["became_winner"]).sum())
    fp = int((pos_mask & ~df["became_winner"]).sum())
    fn = int((~pos_mask &  df["became_winner"]).sum())
    tn = int((~pos_mask & ~df["became_winner"]).sum())

    accuracy_pct = correct           / total             * 100 if total             else 0
    precision    = tp                / predicted_winners * 100 if predicted_winners else 0
    recall       = tp                / actual_winners    * 100 if actual_winners    else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total",          total)
    col2.metric("Accuracy",       f"{accuracy_pct:.1f}%")
    col3.metric("Actual Winners", actual_winners)
    col4.metric("Pred. Winners",  predicted_winners)
    col5.metric("Precision",      f"{precision:.1f}%")
    col6.metric("Recall",         f"{recall:.1f}%")

    gain_populated = df["actual_gain_pct"].notna().sum()
    if gain_populated > 0:
        st.caption(f"actual_gain_pct populated for {gain_populated} / {total} symbols")
        col1, col2, col3, col4 = st.columns(4)
        winner_gains = df.loc[ df["became_winner"] & df["actual_gain_pct"].notna(), "actual_gain_pct"]
        non_gains    = df.loc[~df["became_winner"] & df["actual_gain_pct"].notna(), "actual_gain_pct"]
        err_df       = df.loc[df["gain_error_pct"].notna(), "gain_error_pct"]
        col1.metric("Avg Winner Gain",         f"{winner_gains.mean():.2f}%" if not winner_gains.empty else "—")
        col2.metric("Avg Non-Winner Gain",     f"{non_gains.mean():.2f}%"   if not non_gains.empty    else "—")
        col3.metric("Avg Gain Error",          f"{err_df.mean():.2f}%"      if not err_df.empty       else "—")
        col4.metric("Intraday High Populated", df["actual_high_pct"].notna().sum())

    st.markdown("---")

    col_cm, col_dist = st.columns(2)

    with col_cm:
        st.markdown("#### Confusion Matrix")
        fig = go.Figure(data=go.Heatmap(
            z=[[tp, fp], [fn, tn]],
            x=["Actually Exploded", "Didn't Explode"],
            y=["Predicted Explosion", "Predicted No Explosion"],
            text=[
                [f"<b>True Positive</b><br>{tp}", f"<b>False Positive</b><br>{fp}"],
                [f"<b>False Negative</b><br>{fn}", f"<b>True Negative</b><br>{tn}"],
            ],
            texttemplate="%{text}",
            textfont={"size": 13, "color": "white"},
            colorscale=[
                [0.0,  CONFUSION_COLORS["tn"]],
                [0.33, CONFUSION_COLORS["fn"]],
                [0.66, CONFUSION_COLORS["fp"]],
                [1.0,  CONFUSION_COLORS["tp"]],
            ],
            showscale=False,
        ))
        fig.update_layout(height=320, **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col_dist:
        if gain_populated > 0:
            st.markdown("#### Gain Distribution")
            fig = go.Figure()
            g_tp = df.loc[ pos_mask &  df["became_winner"] & df["actual_gain_pct"].notna(), "actual_gain_pct"]
            g_fp = df.loc[ pos_mask & ~df["became_winner"] & df["actual_gain_pct"].notna(), "actual_gain_pct"]
            if not g_tp.empty:
                fig.add_trace(go.Histogram(x=g_tp, nbinsx=20, name="True Positive",  marker_color=COLORS["secondary"], opacity=0.8))
            if not g_fp.empty:
                fig.add_trace(go.Histogram(x=g_fp, nbinsx=20, name="False Positive", marker_color=COLORS["red"],       opacity=0.8))
            fig.update_layout(
                barmode="overlay",
                xaxis_title="Actual Gain %",
                yaxis_title="Count",
                height=320,
                **LAYOUT,
            )
            fig.update_xaxes(**AXIS_STYLE)
            fig.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Gain distribution will appear once actual_gain_pct is populated.")

    st.markdown("---")
    st.markdown("#### Detailed Results")

    display_cols = [
        c for c in [
            "symbol", "predicted_signal", "predicted_probability",
            "predicted_target_gain", "became_winner", "actual_gain_pct",
            "actual_high_pct", "gain_error_pct", "prediction_correct",
        ] if c in df.columns
    ]

    def _highlight_correct(row):
        bg = "#10b98118" if row.get("prediction_correct") else "#ef444418"
        return [f"background-color: {bg}"] * len(row)

    fmt = {
        "predicted_probability": "{:.2%}",
        "predicted_target_gain": "{:.2f}%",
        "actual_gain_pct":       "{:.2f}%",
        "actual_high_pct":       "{:.2f}%",
        "gain_error_pct":        "{:.2f}%",
    }
    st.dataframe(
        df[display_cols]
        .sort_values("predicted_probability", ascending=False)
        .style.format({k: v for k, v in fmt.items() if k in display_cols}, na_rep="—")
        .apply(_highlight_correct, axis=1),
        use_container_width=True,
        height=500,
    )


# ── Sub-tab 3 — Missed Opportunities ──────────────────────────────────────────
def _render_missed_opportunities():
    st.markdown("#### Missed Opportunities — Recall Analysis")
    st.caption("Winners the model didn't predict.")

    all_missed = _get_table_full("ml_missed_opportunities")
    if all_missed.empty:
        st.warning("No missed opportunities data yet.")
        return

    all_missed = all_missed.copy()
    all_missed["actual_gain_pct"] = pd.to_numeric(all_missed["actual_gain_pct"], errors="coerce")
    all_missed["was_screened"]    = all_missed["was_screened"].astype(bool)

    dates = sorted(all_missed["detection_date"].unique().tolist(), reverse=True)
    selected_date = st.selectbox(
        "Date",
        dates,
        format_func=lambda x: datetime.fromisoformat(x).strftime("%a %d %b %Y"),
        key=f"{TAB_ID}_missed_date",
    )
    df = all_missed[all_missed["detection_date"] == selected_date].copy()
    if df.empty:
        st.info("No missed opportunities for this date.")
        return

    total_missed     = len(df)
    not_screened     = int((~df["was_screened"]).sum())
    screened_no_pred = int(df["was_screened"].sum())
    avg_gain         = df["actual_gain_pct"].mean()
    best_gain        = df["actual_gain_pct"].max()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Missed",            total_missed)
    col2.metric("Not Screened",            not_screened)
    col3.metric("Screened, Not Predicted", screened_no_pred)
    col4.metric("Avg Missed Gain",         f"+{avg_gain:.1f}%"  if pd.notna(avg_gain)  else "—")
    col5.metric("Best Missed Gain",        f"+{best_gain:.1f}%" if pd.notna(best_gain) else "—")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Why We Missed Them")
        reason_col = df["screening_failure_reason"].fillna(
            df["was_screened"].map({True: "screened_but_low_probability", False: "not_in_screener"})
        )
        reason_counts = reason_col.value_counts()
        fig = go.Figure(go.Bar(
            x=reason_counts.index.tolist(),
            y=reason_counts.values.tolist(),
            marker_color=COLORS["red"],
            opacity=0.85,
            text=reason_counts.values.tolist(),
            textposition="outside",
        ))
        fig.update_layout(xaxis_title="Reason", yaxis_title="Count", height=300, **LAYOUT)
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        gain_data = df["actual_gain_pct"].dropna()
        if not gain_data.empty:
            st.markdown("#### Distribution of Missed Gains")
            fig = go.Figure(go.Histogram(x=gain_data, nbinsx=20, marker_color=COLORS["amber"], opacity=0.85))
            fig.update_layout(xaxis_title="Actual Gain %", yaxis_title="Count", height=300, **LAYOUT)
            fig.update_xaxes(**AXIS_STYLE)
            fig.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Detail Table")
    display_cols = [
        c for c in [
            "symbol", "exchange", "actual_gain_pct", "actual_high_pct",
            "actual_price", "actual_volume", "was_screened",
            "screening_failure_reason", "predicted_probability",
        ] if c in df.columns
    ]
    st.dataframe(
        df[display_cols].sort_values("actual_gain_pct", ascending=False)
        .style.format(
            {
                "actual_gain_pct":       "+{:.2f}%",
                "actual_high_pct":       "+{:.2f}%",
                "actual_price":          "${:.2f}",
                "actual_volume":         "{:,.0f}",
                "predicted_probability": "{:.2%}",
            },
            na_rep="—",
        ),
        use_container_width=True,
        height=440,
    )


# ── Sub-tab 4 — Performance Trends ────────────────────────────────────────────
def _render_performance_trends():
    st.markdown("#### Model Performance Trends")

    all_acc = _get_table_all("ml_prediction_accuracy")
    if all_acc.empty:
        st.warning("No accuracy data available yet.")
        return

    all_acc = all_acc.copy()
    all_acc["became_winner"]         = all_acc["became_winner"].astype(bool)
    all_acc["prediction_correct"]    = all_acc["prediction_correct"].astype(bool)
    all_acc["actual_gain_pct"]       = pd.to_numeric(all_acc["actual_gain_pct"],       errors="coerce")
    all_acc["predicted_probability"] = pd.to_numeric(all_acc["predicted_probability"], errors="coerce")

    def _daily_agg(gdf):
        pos_mask    = gdf["predicted_signal"].isin(["STRONG BUY", "BUY"])
        total       = len(gdf)
        correct     = int(gdf["prediction_correct"].sum())
        tp          = int((pos_mask & gdf["became_winner"]).sum())
        pred_pos    = int(pos_mask.sum())
        actual_wins = int(gdf["became_winner"].sum())
        precision   = tp / pred_pos    * 100 if pred_pos    else 0.0
        recall      = tp / actual_wins * 100 if actual_wins else 0.0
        accuracy    = correct / total  * 100 if total       else 0.0
        avg_gain    = gdf.loc[gdf["became_winner"] & gdf["actual_gain_pct"].notna(), "actual_gain_pct"].mean()
        return pd.Series({
            "total": total, "correct": correct,
            "accuracy_pct": accuracy, "precision_pct": precision, "recall_pct": recall,
            "true_positives": tp, "predicted_pos": pred_pos,
            "actual_winners": actual_wins, "avg_winner_gain": avg_gain,
        })

    daily = (
        all_acc.groupby("prediction_date")
        .apply(_daily_agg)
        .reset_index()
        .sort_values("prediction_date")
    )

    def _trendline(x_ordinal: np.ndarray, y: np.ndarray):
        """Least-squares linear trend; returns fitted y-values, or None if
        there isn't enough valid data to fit one."""
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return None
        slope, intercept = np.polyfit(x_ordinal[mask], y[mask], 1)
        return slope * x_ordinal + intercept

    def _metric_panel(col, metric: str, color: str, name: str):
        with col:
            st.markdown(f"**{name}**")

            dates  = pd.to_datetime(daily["prediction_date"])
            x_ord  = dates.map(lambda d: d.toordinal()).to_numpy(dtype=float)
            y_vals = daily[metric].to_numpy(dtype=float)
            trend  = _trendline(x_ord, y_vals)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily["prediction_date"], y=y_vals,
                mode="lines+markers", name=name,
                line=dict(color=color, width=2),
                marker=dict(size=5),
            ))
            if trend is not None:
                fig.add_trace(go.Scatter(
                    x=daily["prediction_date"], y=trend,
                    mode="lines", name="Trend",
                    line=dict(color=color, width=1.5, dash="dash"),
                    opacity=0.55,
                ))
            fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.12)",
                          annotation_text="50% baseline", annotation_font_size=9)
            fig.update_layout(
                height=320, hovermode="x unified", showlegend=False,
                margin=dict(t=12, b=24, l=24, r=16),
                plot_bgcolor=LAYOUT["plot_bgcolor"], paper_bgcolor=LAYOUT["paper_bgcolor"],
                font=LAYOUT["font"],
            )
            fig.update_xaxes(**AXIS_STYLE)
            fig.update_yaxes(**AXIS_STYLE, title_text="%")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Accuracy / Precision / Recall Over Time")
    full_col = st.container()
    _metric_panel(full_col, "accuracy_pct",  COLORS["primary"],   "Accuracy")
    _metric_panel(full_col, "precision_pct", COLORS["secondary"], "Precision")
    _metric_panel(full_col, "recall_pct",    COLORS["amber"],     "Recall")

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily["prediction_date"], y=daily["true_positives"],
            name="True Positives", marker_color=COLORS["secondary"], opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            x=daily["prediction_date"],
            y=daily["predicted_pos"] - daily["true_positives"],
            name="False Positives", marker_color=COLORS["red"], opacity=0.85,
        ))
        fig.update_layout(barmode="stack", title="Predicted Positives Breakdown", height=300, **LAYOUT)
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Scatter(
            x=daily["prediction_date"], y=daily["avg_winner_gain"],
            mode="lines+markers", name="Avg Winner Gain",
            line=dict(color=COLORS["amber"], width=2),
            marker=dict(size=5),
            fill="tozeroy", fillcolor="rgba(251,191,36,0.06)",
        ))
        fig.update_layout(
            title="Avg Actual Gain (Winners Only)",
            xaxis_title="Date", yaxis_title="Gain %",
            height=300, **LAYOUT,
        )
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Average Gain Over Time by Signal")

    pos_signals = all_acc[all_acc["predicted_signal"].isin(["STRONG BUY", "BUY", "HOLD", "AVOID"])].copy()
    pos_signals["prediction_date"] = pd.to_datetime(pos_signals["prediction_date"])

    # Auto pick daily vs weekly bucketing based on how much history is available —
    # daily stays readable up to ~60 unique days, otherwise roll up to weekly.
    n_dates = pos_signals["prediction_date"].nunique()
    use_weekly = n_dates > 60

    if use_weekly:
        pos_signals["bucket"] = pos_signals["prediction_date"].dt.to_period("W").apply(lambda p: p.start_time)
        bucket_label = "Week Of"
    else:
        pos_signals["bucket"] = pos_signals["prediction_date"]
        bucket_label = "Date"

    gain_by_signal = (
        pos_signals[pos_signals["actual_gain_pct"].notna()]
        .groupby(["bucket", "predicted_signal"])["actual_gain_pct"]
        .mean()
        .reset_index()
        .sort_values("bucket")
    )

    period_note = "weekly" if use_weekly else "daily"
    st.caption(
        f"Showing {period_note} average actual gain per signal, across all {n_dates} "
        f"day(s) of available history."
    )

    fig = go.Figure()
    for signal in ["STRONG BUY", "BUY", "HOLD", "AVOID"]:
        sdf = gain_by_signal[gain_by_signal["predicted_signal"] == signal]
        if sdf.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sdf["bucket"], y=sdf["actual_gain_pct"],
            mode="lines+markers", name=signal,
            line=dict(color=SIGNAL_COLORS.get(signal, "#999"), width=2),
            marker=dict(size=6),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.12)")
    fig.update_layout(
        title=f"Average Gain ({period_note.capitalize()}) by Signal",
        xaxis_title=bucket_label, yaxis_title="Avg Gain %",
        height=380, hovermode="x unified", **LAYOUT,
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig, use_container_width=True)

    # ── Cumulative Gain Simulator ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Cumulative Gain Simulator")
    st.caption(
        "Simulates trading each individual signal as its own position — capital "
        "is split equally across every signal that fires on a given day, each "
        "position resolves on its own actual gain, and commission is charged "
        "per trade. Positions are closed out and capital is pooled back together "
        "at the end of each day before being redistributed the next day. This "
        "avoids inflating results by averaging gains before compounding, but "
        "still simplifies real trading (no slippage, no partial fills, "
        "equal-weight sizing only, and outcomes are based on this system's own "
        "historical gain data, not independently verified fills)."
    )

    min_date = pos_signals["prediction_date"].min().date()
    max_date = pos_signals["prediction_date"].max().date()

    sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
    with sim_c1:
        start_capital = st.number_input(
            "Starting capital ($)", min_value=1.0, value=10000.0, step=100.0,
            key="sim_start_capital",
        )
    with sim_c2:
        commission_fee = st.number_input(
            "Commission per trade ($)", min_value=0.0, value=0.0, step=0.5,
            key="sim_commission_fee",
        )
    with sim_c3:
        date_range = st.date_input(
            "Date range", value=(min_date, max_date),
            min_value=min_date, max_value=max_date,
            key="sim_date_range",
        )
    with sim_c4:
        sim_signals = st.multiselect(
            "Signal(s) to act on", ["STRONG BUY", "BUY", "HOLD", "AVOID"],
            default=["STRONG BUY", "BUY"], key="sim_signals",
        )

    max_positions = st.slider(
        "Max concurrent positions per day (caps how thin capital gets split)",
        min_value=1, max_value=25, value=10, key="sim_max_positions",
        help=(
            "If more signals fire on a given day than this, only this many "
            "(chosen by highest predicted probability, if available) are taken "
            "— capital isn't split across an unlimited number of positions."
        ),
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        sim_start, sim_end = date_range
    else:
        sim_start, sim_end = min_date, max_date

    if not sim_signals:
        st.info("Select at least one signal to run the simulation.")
    else:
        sim_df = pos_signals[
            (pos_signals["predicted_signal"].isin(sim_signals))
            & (pos_signals["prediction_date"].dt.date >= sim_start)
            & (pos_signals["prediction_date"].dt.date <= sim_end)
            & (pos_signals["actual_gain_pct"].notna())
        ].copy()

        if sim_df.empty:
            st.warning("No trades match the selected signals and date range.")
        else:
            if "predicted_probability" in sim_df.columns:
                sim_df = sim_df.sort_values("predicted_probability", ascending=False)

            records = []
            capital = start_capital
            for trade_date, gdf in sim_df.groupby("prediction_date", sort=True):
                trades = gdf.head(max_positions)
                n = len(trades)
                if n == 0:
                    continue
                position_size = capital / n
                day_end_capital = 0.0
                for gain in trades["actual_gain_pct"]:
                    resolved = position_size * (1 + gain / 100) - commission_fee
                    day_end_capital += max(resolved, 0.0)
                capital = day_end_capital
                records.append({
                    "prediction_date": trade_date,
                    "trades_taken": n,
                    "portfolio_value": capital,
                })

            sim_result = pd.DataFrame(records).sort_values("prediction_date")

            final_value  = sim_result["portfolio_value"].iloc[-1]
            total_return = (final_value - start_capital) / start_capital * 100
            n_trades     = int(sim_result["trades_taken"].sum())
            n_days       = len(sim_result)
            total_fees   = n_trades * commission_fee

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Final Portfolio Value", f"${final_value:,.2f}")
            m2.metric("Total Return", f"{total_return:+.2f}%")
            m3.metric("Trades Simulated", f"{n_trades} across {n_days} day(s)")
            m4.metric("Total Fees Paid", f"${total_fees:,.2f}")

            fig = go.Figure(go.Scatter(
                x=sim_result["prediction_date"], y=sim_result["portfolio_value"],
                mode="lines+markers", name="Portfolio Value",
                line=dict(color=COLORS["secondary"], width=2),
                marker=dict(size=5),
                fill="tozeroy", fillcolor="rgba(0,255,136,0.06)",
            ))
            fig.add_hline(
                y=start_capital, line_dash="dash", line_color="rgba(255,255,255,0.15)",
                annotation_text="Starting capital", annotation_font_size=10,
            )
            fig.update_layout(
                title=f"Cumulative Portfolio Value — Per-Trade Simulation ({', '.join(sim_signals)})",
                xaxis_title="Date", yaxis_title="Portfolio Value ($)",
                height=380, hovermode="x unified", **LAYOUT,
            )
            fig.update_xaxes(**AXIS_STYLE)
            fig.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Note: results still assume every simulated trade could actually "
                "be filled at the recorded actual gain, with no slippage, no "
                "market-impact cost, and unlimited liquidity — treat this as a "
                "best-case illustration of the model's signal quality, not a "
                "guarantee of real-world tradeable returns."
            )


# ── Sub-tab 5 — System Info ────────────────────────────────────────────────────
def _render_system_info():
    st.markdown("#### System Information")

    log_df = _get_table_full("ml_screening_logs")

    if not log_df.empty:
        date_col = _DATE_COL.get("ml_screening_logs", "screening_date")
        if date_col in log_df.columns:
            log_df = log_df.sort_values(date_col, ascending=False)

        log = log_df.iloc[0]

        def _safe(col, default="—"):
            return log[col] if col in log.index and pd.notna(log[col]) else default

        st.markdown("#### Latest Screening Run")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Date",             str(_safe("screening_date")))
        col2.metric("Stocks Attempted", str(_safe("total_symbols_attempted")))
        col3.metric("Predictions Made", str(_safe("total_predictions")))
        attempted = _safe("total_symbols_attempted")
        fetched   = _safe("symbols_fetched_successfully")
        if attempted not in ("—", 0) and fetched not in ("—",):
            col4.metric("Fetch Success Rate", f"{int(fetched)/int(attempted)*100:.1f}%")
        else:
            col4.metric("Successfully Fetched", str(fetched))

        funnel_steps = [
            ("total_symbols_attempted",      "Total Attempted"),
            ("symbols_fetched_successfully", "Fetched Successfully"),
            ("symbols_after_price_filter",   "After Price Filter"),
            ("symbols_after_volume_filter",  "After Volume Filter"),
            ("total_predictions",            "Final Predictions"),
        ]
        available_steps = [(col, lbl) for col, lbl in funnel_steps if col in log.index and pd.notna(log[col])]

        if available_steps:
            st.markdown("---")
            st.markdown("#### Screening Funnel")
            funnel_df = pd.DataFrame(available_steps, columns=["Field", "Stage"])
            funnel_df["Value"] = funnel_df["Field"].map(lambda c: int(log[c]))
            fig = go.Figure(go.Funnel(
                y=funnel_df["Stage"].tolist(),
                x=funnel_df["Value"].tolist(),
                textinfo="value+percent initial",
                marker=dict(color=COLORS["series"][:len(funnel_df)]),
            ))
            fig.update_layout(title="Screening Funnel", height=320, **LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        if len(log_df) > 1:
            st.markdown("---")
            st.markdown("#### Recent Screening Runs")
            show_cols = [c for c in log_df.columns if c != "id"]
            st.dataframe(log_df[show_cols], use_container_width=True, hide_index=True)

        extra_cols = [c for c in log.index if c not in [col for col, _ in funnel_steps] + ["id"]]
        if extra_cols:
            with st.expander("All log fields"):
                st.json({c: str(log[c]) for c in extra_cols if pd.notna(log[c])})

    else:
        st.info("No screening logs found yet.")

    st.markdown("---")
    st.markdown("#### Database Summary")

    preds_df  = _get_table_full("ml_explosion_predictions")
    acc_df    = _get_table_full("ml_prediction_accuracy")
    missed_df = _get_table_full("ml_missed_opportunities")

    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction Records",  len(preds_df))
    col2.metric("Accuracy Records",    len(acc_df))
    col3.metric("Missed Opp. Records", len(missed_df))

    if not acc_df.empty:
        acc_df = acc_df.copy()
        acc_df["became_winner"]      = acc_df["became_winner"].astype(bool)
        acc_df["prediction_correct"] = acc_df["prediction_correct"].astype(bool)
        col1, col2 = st.columns(2)
        col1.metric("Overall Accuracy (all dates)", f"{acc_df['prediction_correct'].mean()*100:.1f}%")
        col2.metric("Overall Winner Rate",          f"{acc_df['became_winner'].mean()*100:.1f}%")

    st.markdown("---")
    st.markdown("#### Automated Schedule (Estonia Time)")

    col_sched, col_model = st.columns(2)
    with col_sched:
        st.markdown("""
| Time | Job |
|------|-----|
| **15:00** | Stock screening & prediction |
| **23:30** | Daily winners collection |
| **05:30 +1** | Accuracy tracking + gain fetch |
| **Sun 09:00** | Weekly model retraining |
        """)

    with col_model:
        st.markdown("#### Model Details")
        st.markdown("""
| | |
|---|---|
| **Model** | XGBoost Classifier |
| **Features** | 97 technical indicators |
| **Target** | 20%+ single-day gain |
| **Window** | 90-day rolling |
| **Gain source** | yfinance (all predicted symbols) |
        """)

    st.warning(
        "Experimental system for research only. Not financial advice. "
        "Past performance does not guarantee future results."
    )
