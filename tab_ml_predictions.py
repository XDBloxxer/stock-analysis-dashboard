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

from db import get_supabase_client, run_with_retry, log_debug_error
from chart_utils import CHART_THEME, LAYOUT, AXIS_STYLE, COLORS, SIGNAL_COLORS, SIGNAL_BG, CONFUSION_COLORS
from dashboard_styles import render_section_header, render_empty_state, render_skeleton_rows, render_labeled_divider, ticker_copy_html, render_hero_metric, exchange_chip_html, radial_gauge_svg
from cache_ui import render_cache_buttons
from format_utils import fmt_compact

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
    "CBOE":      "BATS",
    "CBOE BZX":  "BATS",
    "BZX":       "BATS",
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
    except Exception as e:
        log_debug_error(f"_get_live_quote({symbol})", e)
        return None


def _fetch_one_quote(symbol: str) -> tuple[str, dict | None]:
    """Worker for the thread pool in _get_bulk_live_quotes — one symbol in, one quote out."""
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).fast_info
        return symbol, {
            "last_price": getattr(info, "last_price",     None),
            "day_high":   getattr(info, "day_high",       None),
            "day_low":    getattr(info, "day_low",        None),
            "open":       getattr(info, "open",           None),
            "prev_close": getattr(info, "previous_close", None),
            "volume":     getattr(info, "last_volume",    None),
        }
    except Exception as e:
        log_debug_error(f"_get_bulk_live_quotes({symbol})", e)
        return symbol, None


@st.cache_data(ttl=60, show_spinner=False)
def _get_bulk_live_quotes(symbols: tuple) -> dict:
    """
    Fetch live quotes for multiple symbols in parallel via a thread pool.
    Each yfinance `fast_info` call is a blocking HTTP request, so fetching
    them one at a time (as the previous `yf.Tickers(...)` loop effectively
    did — it still made N sequential calls under the hood) means the whole
    table waits on N round-trips in series. Fanning them out across threads
    turns that into ~1 round-trip's worth of wall-clock time regardless of
    how many symbols are on screen.
    Returns dict[symbol -> quote_dict]. TTL=60s, same as single-symbol fetch.
    symbols must be a tuple (hashable) for st.cache_data to work.
    """
    result = {}
    if not symbols:
        return result
    try:
        import yfinance as yf  # noqa: F401 — import check before spinning up threads
    except ImportError:
        return {sym: None for sym in symbols}

    max_workers = min(16, len(symbols))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one_quote, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                _, quote = future.result()
                result[sym] = quote
            except Exception as e:
                log_debug_error(f"_get_bulk_live_quotes future({sym})", e)
                result[sym] = None
    return result


def _fetch_one_sparkline(symbol: str) -> tuple[str, list | None]:
    """Worker for the thread pool in _get_bulk_sparklines — one symbol's
    today's intraday closes, downsampled to a short list for a mini chart."""
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period="1d", interval="15m")
        if hist.empty or "Close" not in hist.columns:
            return symbol, None
        closes = hist["Close"].dropna().tolist()
        return symbol, closes if len(closes) >= 2 else None
    except Exception as e:
        log_debug_error(f"_get_bulk_sparklines({symbol})", e)
        return symbol, None


@st.cache_data(ttl=300, show_spinner=False)
def _get_bulk_sparklines(symbols: tuple) -> dict:
    """
    Fetch today's intraday closes for multiple symbols in parallel, for the
    small inline sparkline next to Live price in the market table.
    TTL=300s (vs 60s for live quotes) — the day's overall shape barely
    changes minute to minute, so a slower refresh here avoids doubling the
    yfinance call volume of _get_bulk_live_quotes for little visual benefit.
    """
    result = {}
    if not symbols:
        return result
    try:
        import yfinance as yf  # noqa: F401 — import check before spinning up threads
    except ImportError:
        return {sym: None for sym in symbols}

    max_workers = min(16, len(symbols))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one_sparkline, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                _, closes = future.result()
                result[sym] = closes
            except Exception as e:
                log_debug_error(f"_get_bulk_sparklines future({sym})", e)
                result[sym] = None
    return result


def _sparkline_svg(values: list, color: str, width: int = 64, height: int = 22) -> str:
    """Tiny inline SVG polyline — no axes/labels, just the day's shape.

    Draws itself in on mount (see `.spark-draw` in dashboard_styles.py) via
    the SVG `pathLength` attribute, which normalizes the line's reported
    length to exactly 100 regardless of how many points it has or how far
    it spans — that lets one fixed `stroke-dasharray:100` CSS animation
    handle every sparkline without computing each polyline's real length in
    Python.
    """
    if not values or len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    n = len(values)
    pts = [
        f"{(i / (n - 1)) * width:.1f},{height - ((v - lo) / span) * height:.1f}"
        for i, v in enumerate(values)
    ]
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="display:block;overflow:visible;">'
        f'<polyline class="spark-draw" pathLength="100" points="{" ".join(pts)}" fill="none" stroke="{color}" '
        f'stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round" opacity="0.85"/>'
        f'</svg>'
    )


def _warn_if_truncated(df: pd.DataFrame, table_name: str) -> None:
    """
    Surface the `truncated` flag set by _get_table_full — without this, a
    fetch that silently hit the row cap looks identical to a complete one.
    """
    if not df.empty and df.attrs.get("truncated"):
        st.caption(
            f"⚠️ Showing the most recent {_TABLE_ROW_CAP} rows of `{table_name}` — "
            "older rows exist but aren't loaded in this view."
        )


# ── Cached DB fetchers ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _get_table_full(table_name: str) -> pd.DataFrame:
    try:
        client     = get_supabase_client()
        date_col   = _DATE_COL.get(table_name)
        select_str = _SELECT.get(table_name, "*")

        def _run():
            query = client.table(table_name).select(select_str)
            if date_col:
                query = query.order(date_col, desc=True)
            return query.limit(_TABLE_ROW_CAP).execute()

        response = run_with_retry(_run, source=f"_get_table_full({table_name})")
        df = pd.DataFrame(response.data) if response.data else pd.DataFrame()
        # Tag whether this fetch hit the row cap, so callers can warn the user
        # instead of silently showing a truncated "Latest Predictions" view.
        df.attrs["truncated"] = len(df) >= _TABLE_ROW_CAP
        return df
    except Exception as e:
        log_debug_error(f"_get_table_full({table_name})", e)
        st.warning(f"Could not load `{table_name}`: {e}")
        return pd.DataFrame()


def _get_table_optional(table_name: str) -> pd.DataFrame:
    """
    Like _get_table_full, but for tables that are an optional enhancement
    (e.g. feature importance) and may not exist in every deployment yet.
    Fails silently in the UI (still logged to the debug log) instead of
    showing a warning banner.
    """
    try:
        client     = get_supabase_client()
        date_col   = _DATE_COL.get(table_name)
        select_str = _SELECT.get(table_name, "*")

        def _run():
            query = client.table(table_name).select(select_str)
            if date_col:
                query = query.order(date_col, desc=True)
            return query.limit(_TABLE_ROW_CAP).execute()

        response = run_with_retry(_run, source=f"_get_table_optional({table_name})")
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        log_debug_error(f"_get_table_optional({table_name})", e)
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
            def _run(offset=offset):
                query = client.table(table_name).select(select_str)
                if date_col:
                    query = query.order(date_col, desc=False)
                return query.range(offset, offset + page_size - 1).execute()

            response = run_with_retry(_run, source=f"_get_table_all({table_name})")
            rows = response.data or []
            if not rows:
                break
            frames.append(pd.DataFrame(rows))
            if len(rows) < page_size:
                break
            offset += page_size
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception as e:
        log_debug_error(f"_get_table_all({table_name})", e)
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


# ── Main entry point ───────────────────────────────────────────────────────────
def render_ml_predictions_tab():
    refresh_clicked, clear_confirmed = render_cache_buttons(TAB_ID)

    if clear_confirmed:
        clear_all_cache()
        st.rerun()
    if refresh_clicked:
        refresh_cache()
        st.rerun()

    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
        "Today's Picks",
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

    # Skeleton rows are only shown the first time this tab renders in a
    # session. Showing/hiding them on *every* rerun (including reruns
    # triggered by a widget in a completely different tab, since Streamlit
    # re-executes the whole script top to bottom regardless of which tab is
    # visually active) repeatedly resizes this tab's content height, which
    # is a known Streamlit bug: it makes the page jump to this tab/section
    # (streamlit/streamlit#5069). Gating it behind session_state avoids that
    # height churn on steady-state reruns.
    _quotes_seen_key = f"{TAB_ID}_quotes_first_paint_done"
    if not st.session_state.get(_quotes_seen_key):
        _quotes_placeholder = st.empty()
        with _quotes_placeholder.container():
            render_skeleton_rows(4, height=64)
        quotes = _get_bulk_live_quotes(symbols)
        sparklines = _get_bulk_sparklines(symbols)
        _quotes_placeholder.empty()
        st.session_state[_quotes_seen_key] = True
    else:
        quotes = _get_bulk_live_quotes(symbols)
        sparklines = _get_bulk_sparklines(symbols)

    # Previous-render live prices, kept in session_state, so a tick can be
    # flashed green/red on the render where it actually changes rather than
    # just silently showing a different number than a moment ago.
    _prev_prices = st.session_state.setdefault("_live_price_flash_prev", {})

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
        "BUY":        ("var(--blue-bright)",  "rgba(56,189,248,0.10)", "rgba(56,189,248,0.25)"),
        "HOLD":       ("var(--amber-bright)", "rgba(245,158,11,0.08)", "rgba(245,158,11,0.2)"),
        "AVOID":      ("var(--red-bright)",   "rgba(239,68,68,0.08)",  "rgba(239,68,68,0.2)"),
    }
    _left_bar = {
        "STRONG BUY": "var(--green-bright)",
        "BUY":        "var(--blue-bright)",
        "HOLD":       "var(--amber-bright)",
        "AVOID":      "var(--red-bright)",
    }

    def _badge(signal: str) -> str:
        fg, bg, border = _sig_colors.get(signal, ("var(--text-2)", "var(--bg-3)", "var(--border)"))
        pulse_cls = " badge-pulse" if signal == "STRONG BUY" else ""
        return (
            f'<span class="{pulse_cls.strip()}" style="padding:3px 9px;border-radius:3px;background:{bg};'
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
        # Gradient by proximity to target, not a flat on/off color — early
        # progress reads as brass (matches the "not there yet" default),
        # the final stretch shifts to amber as a visual "getting close"
        # signal, and hitting/exceeding target goes full green. Three
        # fixed bands rather than a continuous interpolation, so the color
        # a user sees stays legible/nameable rather than a blurry gradient.
        if hit:
            bar_color, lbl_color = "var(--green-bright)", "var(--green-bright)"
        elif progress >= 0.75:
            bar_color, lbl_color = "var(--amber-bright)", "var(--amber-bright)"
        else:
            bar_color, lbl_color = "var(--cyan)", "var(--text-0)"
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
    h = st.columns([2.6, 1.4, 1.1, 1.1, 1.8, 1.3, 3.7, 0.7])
    for col, lbl in zip(h, ["Stock", "Signal", "Prob", "Entry", "Live", "Day High", "→ Target", "TV"]):
        col.markdown(f'<div style="{hdr_style}">{lbl}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Scopes the row-hover CSS (see .mkt-live-table-scope in dashboard_styles.py)
    # to just the data rows below, not the header row above.
    st.markdown('<div class="mkt-live-table-scope"></div>', unsafe_allow_html=True)

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
        vol_str = fmt_compact(volume) if volume else "—"

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
                f'<div class="bar-track" style="margin-top:4px;">'
                f'<div class="bar-fill" style="width:{prob_pct:.0f}%;background:{p_color};"></div>'
                f'</div>'
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

        cols = st.columns([2.6, 1.4, 1.1, 1.1, 1.8, 1.3, 3.7, 0.7])

        # Col 0 — Ticker + exchange + volume
        with cols[0]:
            sym_html = ticker_copy_html(
                sym, style="font-family:var(--font-body);font-size:0.95rem;"
                            "font-weight:700;color:var(--text-0);letter-spacing:0.04em;"
            )
            st.markdown(
                f'<div style="padding:7px 0 5px;border-left:2px solid {left_color};padding-left:10px;">'
                f'{sym_html}'
                f'<span style="margin-left:7px;">{exchange_chip_html(exchange)}</span>'
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

        # Col 4 — Live price + day change + intraday sparkline
        with cols[4]:
            chg_color = _chg_color(day_chg)
            chg_str   = f'{day_chg:+.2f}%' if day_chg is not None else "—"
            live_str  = _val(live, "$.2f")
            spark_vals = sparklines.get(sym) if sparklines else None
            spark_svg  = _sparkline_svg(spark_vals, chg_color) if spark_vals else ""

            # One-shot flash if the live price moved since the last render
            # of this symbol (see _live_price_flash_prev above).
            prev_live  = _prev_prices.get(sym)
            flash_cls  = ""
            if live is not None and prev_live is not None and live != prev_live:
                flash_cls = " price-flash-up" if live > prev_live else " price-flash-down"
            if live is not None:
                _prev_prices[sym] = live

            st.markdown(
                f'<div style="padding:7px 0 5px;">'
                f'<div style="display:flex;align-items:center;gap:8px;">'
                f'<span class="{flash_cls.strip()}" style="font-family:var(--font-body);font-size:0.9rem;'
                f'font-weight:700;color:var(--text-0);white-space:nowrap;padding:1px 4px;">{live_str}</span>'
                f'{spark_svg}'
                f'</div>'
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
        f"Sparkline = today's price shape (15-min bars, refreshed every 5 min) · "
        f"Entry = price at time of prediction · Live data via Yahoo Finance · not financial advice"
    )


# ── Sub-tab 1 — Latest Predictions ────────────────────────────────────────────
def _render_market_map(df: pd.DataFrame):
    """
    Treemap of the *entire* day's screened universe (not just the BUY/STRONG
    BUY picks shown below) — box size by model confidence, color by signal.
    The picks list answers "what should I look at"; this answers "what does
    today look like as a whole" at a glance, which a table of 50-200 rows
    can't do without scrolling.

    Sized by `explosion_probability` rather than volume/market-cap since
    this dataset doesn't carry either at the per-day universe level (volume
    is only fetched live for the shown picks, not the full screened set) —
    confidence is the one magnitude every row already has.
    """
    plot_df = df.copy()
    plot_df["explosion_probability"] = pd.to_numeric(plot_df["explosion_probability"], errors="coerce")
    plot_df = plot_df.dropna(subset=["explosion_probability", "signal", "symbol"])
    plot_df = plot_df[plot_df["explosion_probability"] > 0]
    if plot_df.empty or len(plot_df) < 2:
        return  # not enough rows for a treemap to say anything a table doesn't

    signals_present = [s for s in ["STRONG BUY", "BUY", "HOLD", "AVOID"] if s in plot_df["signal"].unique()]

    # Single treemap trace: the four signal names are their own top-level
    # nodes (parent=""), every symbol is a leaf under its signal. Group
    # nodes carry value=0 (branchvalues="remainder" means a parent's box
    # size comes purely from its children's sum, not its own value), so
    # group box size is entirely "how many + how confident" the symbols
    # under it are.
    labels  = signals_present + list(plot_df["symbol"])
    parents = [""] * len(signals_present) + list(plot_df["signal"])
    values  = [0] * len(signals_present) + list(plot_df["explosion_probability"])
    texts   = [""] * len(signals_present) + [f"{p*100:.0f}%" for p in plot_df["explosion_probability"]]
    colors  = [SIGNAL_BG.get(s, "rgba(134,149,171,0.13)") for s in signals_present] \
              + [SIGNAL_COLORS.get(s, "#8695ab") for s in plot_df["signal"]]

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="remainder",
        text=texts,
        texttemplate="<b>%{label}</b><br>%{text}",
        textfont=dict(family="DM Mono, monospace", size=12, color="#f8fafc"),
        marker=dict(colors=colors, line=dict(width=1.5, color="rgba(5,7,10,0.9)")),
        hovertemplate="<b>%{label}</b><br>Confidence: %{text}<br>Signal: %{parent}<extra></extra>",
        root_color="rgba(0,0,0,0)",
    ))

    fig.update_layout(
        margin=dict(t=8, b=8, l=8, r=8), height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="DM Mono, monospace"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_latest_predictions():
    render_section_header(1, "Today's Picks")
    st.caption("What the model is telling you to buy right now.")

    # Same reasoning as the live-quotes skeleton below: only paint it once
    # per session so hidden/other-tab reruns don't keep resizing this tab
    # and dragging the page's scroll position along with it.
    _picks_seen_key = f"{TAB_ID}_picks_first_paint_done"
    if not st.session_state.get(_picks_seen_key):
        _picks_placeholder = st.empty()
        with _picks_placeholder.container():
            render_skeleton_rows(4, height=64)
        all_preds = _get_table_full("ml_explosion_predictions")
        _picks_placeholder.empty()
        st.session_state[_picks_seen_key] = True
    else:
        all_preds = _get_table_full("ml_explosion_predictions")
    _warn_if_truncated(all_preds, "ml_explosion_predictions")

    if all_preds.empty:
        render_empty_state("No predictions available yet — run the screening workflow or wait for the scheduled run.")
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
        if pred_dt >= datetime.now().date():
            st.markdown("<div style='padding-top:28px'>", unsafe_allow_html=True)
            st.success("Forward-looking predictions")
            st.markdown("</div>", unsafe_allow_html=True)

    df = all_preds[all_preds["prediction_date"] == selected_date].copy()
    if df.empty:
        st.warning(f"No predictions for {selected_date}")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # TODAY'S PICKS — the thing partners actually open this dashboard for.
    # Everything else on this sub-tab is supporting detail, tucked away below.
    # ══════════════════════════════════════════════════════════════════════════
    picks = df[df["signal"].isin(["STRONG BUY", "BUY"])].copy()
    _signal_rank = {"STRONG BUY": 0, "BUY": 1}
    picks["_rank"] = picks["signal"].map(_signal_rank)
    picks = picks.sort_values(["_rank", "explosion_probability"], ascending=[True, False]).drop(columns=["_rank"])

    _MAX_PICKS_SHOWN = 10
    total_picks = len(picks)
    shown_picks = picks.head(_MAX_PICKS_SHOWN)

    if total_picks == 0:
        render_empty_state("No BUY / STRONG BUY signals for this date — the model isn't flagging anything actionable.")
    else:
        # ── Toast on newly-appeared STRONG BUY signals ──────────────────────
        # Tracked per prediction_date in session_state so a toast only fires
        # once per symbol/date combo (e.g. the first time this page renders
        # after a fresh screening run adds a new STRONG BUY), not on every
        # Streamlit rerun caused by an unrelated widget interaction.
        _seen_key = f"{TAB_ID}_seen_strong_buys"
        _seen = st.session_state.setdefault(_seen_key, set())
        _strong_buys_now = set(
            f"{selected_date}:{s}" for s in picks.loc[picks["signal"] == "STRONG BUY", "symbol"]
        )
        _new_strong_buys = _strong_buys_now - _seen
        if _new_strong_buys and _seen:  # skip the toast storm on first-ever load
            _new_syms = sorted(s.split(":", 1)[1] for s in _new_strong_buys)
            st.toast(
                f"New STRONG BUY: {', '.join(_new_syms[:3])}"
                + (f" +{len(_new_syms) - 3} more" if len(_new_syms) > 3 else ""),
                icon="🚀",
            )
        st.session_state[_seen_key] = _seen | _strong_buys_now

        _top = shown_picks.iloc[0]
        _top_gain = _top.get("target_gain_pct")
        render_hero_metric(
            "Top Pick Today",
            f"{_top['symbol']}",
            sub=(
                f"{_top['explosion_probability']*100:.1f}% confidence · {_top['signal']}"
                + (f" · target +{_top_gain:.1f}%" if pd.notna(_top_gain) else "")
            ),
            accent="green" if _top["signal"] == "STRONG BUY" else "cyan",
            glyph="★",
        )

        with st.expander("Market map — today's full screened universe", expanded=False):
            st.caption(
                "Every symbol screened today, grouped by signal and sized by "
                "model confidence — the shape of the whole day at a glance, "
                "not just the picks below."
            )
            _render_market_map(df)

        st.markdown(
            f"##### Top picks for today"
            + (f" (of {total_picks} flagged)" if total_picks > _MAX_PICKS_SHOWN else "")
        )

        _badge_class = {"STRONG BUY": "badge-green", "BUY": "badge-blue"}

        for _, row in shown_picks.iterrows():
            prob   = row["explosion_probability"] * 100
            gain   = row.get("target_gain_pct")
            price  = row.get("current_price")
            target = row.get("target_price")
            gain_str   = f"+{gain:.1f}%" if pd.notna(gain) else "—"
            price_str  = f"${price:.2f}" if pd.notna(price) else "—"
            target_str = f"${target:.2f}" if pd.notna(target) else "—"

            sym_html = ticker_copy_html(
                row['symbol'],
                style="font-family:var(--font-display);font-weight:700;font-size:1.05rem;color:var(--text-0);"
            )
            gauge_html = radial_gauge_svg(prob, size=56, stroke=5)
            st.markdown(
                f"""
<div class="data-card{' card-strong-buy' if row['signal'] == 'STRONG BUY' else ''}" style="display:flex;align-items:center;gap:18px;margin-bottom:8px;padding:14px 18px;">
    <div style="min-width:150px;">
        {sym_html}
        <span style="color:var(--text-2);font-size:0.72rem;margin-left:6px;">{row.get('exchange','')}</span>
        <div style="margin-top:4px;"><span class="badge {_badge_class.get(row['signal'],'badge-blue')}{' badge-pulse' if row['signal'] == 'STRONG BUY' else ''}">{row['signal']}</span></div>
    </div>
    <div style="display:flex;align-items:center;gap:10px;min-width:150px;">
        {gauge_html}
        <div>
            <div style="font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-2);">Confidence</div>
            <div style="font-family:var(--font-body);font-size:1.1rem;color:var(--text-0);font-weight:500;">{prob:.1f}%</div>
        </div>
    </div>
    <div style="flex:1;min-width:120px;">
        <div style="font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-2);">Entry Price</div>
        <div style="font-family:var(--font-body);font-size:1.1rem;color:var(--text-1);">{price_str}</div>
    </div>
    <div style="flex:1;min-width:120px;">
        <div style="font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-2);">Target</div>
        <div style="font-family:var(--font-body);font-size:1.1rem;color:var(--text-1);">{target_str}</div>
    </div>
    <div style="flex:1;min-width:120px;">
        <div style="font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-2);">Target Gain</div>
        <div class="num positive" style="font-size:1.1rem;">{gain_str}</div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        csv_cols = [c for c in [
            "symbol", "exchange", "signal", "explosion_probability",
            "current_price", "target_price", "target_gain_pct",
        ] if c in picks.columns]
        st.download_button(
            "Download today's picks (CSV)",
            picks[csv_cols].to_csv(index=False),
            f"todays_picks_{selected_date}.csv",
            "text/csv",
            key=f"{TAB_ID}_picks_dl",
        )

    render_labeled_divider("Filters")

    # ── Filters — light, always visible, drive both the live view and table ──
    with st.expander("Filter predictions", expanded=False):
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

    # NOTE: both explosion_probability and target_gain_pct can be NaN for
    # some rows (e.g. HOLD/AVOID signals, or a fresh screening run that
    # hasn't back-filled every field yet). `NaN >= threshold` is always
    # False in pandas, so at the default 0% thresholds those rows were
    # being silently dropped from the live view — while the picks list
    # above never applies either filter, which is why the same stocks were
    # visible there but not here. Treat a missing value as 0 so a row is
    # only excluded once the user actually raises that threshold above
    # zero, not just because the value is unknown.
    fdf = df[
        df["signal"].isin(sig_filter) &
        (df["explosion_probability"].fillna(0) >= min_prob / 100) &
        (df["target_gain_pct"].fillna(0) >= min_tgt)
    ].copy()

    # Sort by signal strength then probability
    signal_order = {"STRONG BUY": 0, "BUY": 1, "HOLD": 2, "AVOID": 3}
    fdf["_sig_rank"] = fdf["signal"].map(signal_order).fillna(9)
    fdf = fdf.sort_values(["_sig_rank", "explosion_probability"], ascending=[True, False]).drop(columns=["_sig_rank"])

    st.caption(f"{len(fdf)} stocks match current filters")

    if fdf.empty:
        render_empty_state("No stocks match the filters — try widening the signal, probability, or gain thresholds.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # Live Market View — surfaced immediately, not buried behind a table/expander
    # ══════════════════════════════════════════════════════════════════════════
    _render_live_market_table(fdf)


# ── Sub-tab 2 — Predictions vs Actuals ────────────────────────────────────────
def _render_predictions_vs_actuals():
    render_section_header(2, "Predictions vs Actuals")
    st.caption("How past predictions compared to what actually happened.")
    st.markdown("#### Prediction Accuracy Analysis")

    all_acc = _get_table_full("ml_prediction_accuracy")
    _warn_if_truncated(all_acc, "ml_prediction_accuracy")
    if all_acc.empty:
        render_empty_state("No accuracy data available yet.")
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
        render_empty_state(f"No accuracy data for {selected_date}.")
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

    # Animated fill bars under Precision/Recall — a plain st.metric only
    # shows the number, but these two rates are meant to be weighed against
    # each other (few false alarms vs. few missed winners), so a bar makes
    # the trade-off visible at a glance instead of requiring mental math.
    with col5:
        st.markdown(
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{precision:.0f}%;background:var(--cyan);"></div></div>',
            unsafe_allow_html=True,
        )
    with col6:
        st.markdown(
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{recall:.0f}%;background:var(--purple);"></div></div>',
            unsafe_allow_html=True,
        )

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

    render_labeled_divider("Confusion Matrix & Gain Distribution")

    col_cm, col_dist = st.columns(2)

    with col_cm:
        st.markdown("#### Confusion Matrix")
        # Fixed per-cell colors (not a value-driven colorscale — see note
        # above) so each label always renders with its own correct color no
        # matter which cell happens to have the highest/lowest count.
        _cm_cells = [
            ("True Positive",  tp, CONFUSION_COLORS["tp"], "#0b0f14"),   # bright green bg → dark text
            ("False Positive", fp, CONFUSION_COLORS["fp"], "#0b0f14"),   # bright red bg   → dark text
            ("False Negative", fn, CONFUSION_COLORS["fn"], "#0b0f14"),   # bright amber bg → dark text
            ("True Negative",  tn, CONFUSION_COLORS["tn"], "#cbd5e1"),   # dark slate bg   → light text
        ]
        _cm_html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">'
        for _label, _val, _bg, _fg in _cm_cells:
            _label_fg = "rgba(0,0,0,0.62)" if _fg == "#0b0f14" else "var(--text-2)"
            _cm_html += (
                f'<div style="background:{_bg};border-radius:var(--radius-sm);padding:16px 14px;text-align:center;">'
                f'<div style="font-family:var(--font-body);font-size:0.68rem;font-weight:600;'
                f'letter-spacing:0.08em;text-transform:uppercase;color:{_label_fg};margin-bottom:4px;">{_label}</div>'
                f'<div style="font-family:var(--font-display);font-size:1.6rem;font-weight:800;color:{_fg};">{_val}</div>'
                '</div>'
            )
        _cm_html += '</div>'
        st.markdown(
            '<div style="display:grid;grid-template-columns:1fr 1fr;font-family:var(--font-body);'
            'font-size:0.65rem;letter-spacing:0.06em;text-transform:uppercase;color:var(--text-2);'
            'text-align:center;margin-bottom:6px;">'
            '<div>Actually Exploded</div><div>Didn\'t Explode</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(_cm_html, unsafe_allow_html=True)

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

    render_labeled_divider("Detailed Results")

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
    render_section_header(3, "Missed Opportunities")
    st.markdown("#### Missed Opportunities — Recall Analysis")
    st.caption("Winners the model didn't predict.")

    all_missed = _get_table_full("ml_missed_opportunities")
    _warn_if_truncated(all_missed, "ml_missed_opportunities")
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
        render_empty_state("No missed opportunities for this date — nice.")
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

    render_labeled_divider("Why We Miss Winners")
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

    render_labeled_divider("Top Reasons We Miss Winners (All History)")
    st.caption(
        "Rolled up across every date on record, not just the one selected above — "
        "useful for spotting a persistent pattern (e.g. mostly a screener gap vs. "
        "mostly a probability-threshold issue) rather than a single day's noise."
    )

    all_reason_col = all_missed["screening_failure_reason"].fillna(
        all_missed["was_screened"].map({True: "screened_but_low_probability", False: "not_in_screener"})
    )
    all_reason_counts = all_reason_col.value_counts()

    agg_col_left, agg_col_right = st.columns([3, 2])
    with agg_col_left:
        fig = go.Figure(go.Bar(
            x=all_reason_counts.index.tolist(),
            y=all_reason_counts.values.tolist(),
            marker_color=COLORS["amber"],
            opacity=0.85,
            text=all_reason_counts.values.tolist(),
            textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="Reason", yaxis_title="Count (all history)",
            height=300, **LAYOUT,
        )
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig, use_container_width=True)

    with agg_col_right:
        not_screened_pct = (~all_missed["was_screened"]).mean() * 100
        st.metric(
            "Missed because never screened",
            f"{not_screened_pct:.0f}%",
            help=(
                "Of ALL missed winners on record, the % that never made it into "
                "the screener at all (vs. being screened but scored too low to "
                "flag as BUY/STRONG BUY)."
            ),
        )
        st.caption(
            "High and rising → look at the screening/universe-selection step. "
            "Low and mostly 'screened but low probability' → look at the model's "
            "probability threshold or feature set instead."
        )

    render_labeled_divider("Detail Table")
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
    render_section_header(4, "Performance Trends")
    st.caption("How the system as a whole has performed over time.")
    st.markdown("#### Model Performance Trends")

    all_acc = _get_table_all("ml_prediction_accuracy")
    if all_acc.empty:
        render_empty_state("No accuracy data available yet.")
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

    # ── Drift indicator ──────────────────────────────────────────────────
    # Pools raw counts (rather than averaging daily percentages) so the
    # recent window isn't skewed by a single thin day. Compares the most
    # recent DRIFT_WINDOW_DAYS of predictions against everything before
    # that window, and flags it if precision or recall has meaningfully
    # degraded (or improved).
    DRIFT_WINDOW_DAYS = 14
    DRIFT_MIN_SAMPLE  = 5     # min relevant calls needed on each side to trust the comparison
    DRIFT_ALERT_PTS   = 10    # point swing that triggers a warning/success banner

    dates_sorted = pd.to_datetime(all_acc["prediction_date"]).sort_values()
    if not dates_sorted.empty:
        cutoff = dates_sorted.max() - pd.Timedelta(days=DRIFT_WINDOW_DAYS)
        acc_dates = pd.to_datetime(all_acc["prediction_date"])
        recent_mask   = acc_dates > cutoff
        baseline_mask = ~recent_mask

        def _pooled_metric(mask, metric: str):
            """metric: 'precision' (of BUY/STRONG BUY calls, % that won) or
            'recall' (of actual winners, % the model flagged as BUY/STRONG BUY)."""
            gdf      = all_acc[mask]
            pos_mask = gdf["predicted_signal"].isin(["STRONG BUY", "BUY"])
            tp       = int((pos_mask & gdf["became_winner"]).sum())
            if metric == "precision":
                denom = int(pos_mask.sum())
            else:  # recall
                denom = int(gdf["became_winner"].sum())
            value = tp / denom * 100 if denom else np.nan
            return value, denom

        def _render_drift_row(metric: str, label: str, noun: str):
            recent_val, recent_n     = _pooled_metric(recent_mask, metric)
            baseline_val, baseline_n = _pooled_metric(baseline_mask, metric)

            if (
                baseline_n >= DRIFT_MIN_SAMPLE and recent_n >= DRIFT_MIN_SAMPLE
                and not np.isnan(recent_val) and not np.isnan(baseline_val)
            ):
                delta = recent_val - baseline_val
                if delta <= -DRIFT_ALERT_PTS:
                    st.warning(
                        f"⚠️ **{label} drift detected** — over the last {DRIFT_WINDOW_DAYS} days, "
                        f"{label.lower()} is **{recent_val:.1f}%** ({recent_n} {noun}) "
                        f"vs **{baseline_val:.1f}%** ({baseline_n} {noun}) before that — a "
                        f"{abs(delta):.1f} point drop. Worth a look before trusting today's signals as-is."
                    )
                elif delta >= DRIFT_ALERT_PTS:
                    st.success(
                        f"✅ **{label} trending up** — last {DRIFT_WINDOW_DAYS} days: "
                        f"**{recent_val:.1f}%** ({recent_n} {noun}) vs **{baseline_val:.1f}%** "
                        f"({baseline_n} {noun}) before that, a {delta:.1f} point improvement."
                    )
                else:
                    st.caption(
                        f"{label}, last {DRIFT_WINDOW_DAYS} days vs. prior history: "
                        f"{recent_val:.1f}% ({recent_n} {noun}) vs {baseline_val:.1f}% "
                        f"({baseline_n} {noun}) — no significant drift."
                    )
            else:
                st.caption(
                    f"Not enough {noun} yet on both sides of the last "
                    f"{DRIFT_WINDOW_DAYS} days to reliably check for {label.lower()} drift "
                    f"(need ≥{DRIFT_MIN_SAMPLE} {noun} each side)."
                )

        _render_drift_row("precision", "Precision", "BUY/STRONG BUY calls")
        _render_drift_row("recall",    "Recall",    "actual winners")

    def _trendline(x_ordinal: np.ndarray, y: np.ndarray):
        """Least-squares linear trend; returns fitted y-values, or None if
        there isn't enough valid data to fit one."""
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return None
        slope, intercept = np.polyfit(x_ordinal[mask], y[mask], 1)
        return slope * x_ordinal + intercept

    _METRIC_EXPLAINERS = {
        "accuracy_pct":  "Of all predictions made, the % that turned out correct (winner predicted as winner, non-winner predicted as non-winner).",
        "precision_pct": "Of the stocks the model flagged as BUY/STRONG BUY, the % that actually became winners. High precision = few false alarms.",
        "recall_pct":    "Of the stocks that actually became winners, the % the model successfully flagged as BUY/STRONG BUY. High recall = few missed winners.",
    }

    def _metric_panel(col, metric: str, color: str, name: str):
        with col:
            st.markdown(f"**{name}**")
            st.caption(_METRIC_EXPLAINERS[metric])

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

    # Recall uses COLORS['blue'] rather than COLORS['amber'] here — amber
    # sits too close to Accuracy's brass (COLORS['primary']) tone to tell
    # apart at a glance when the three panels are stacked together.
    st.markdown("##### Accuracy / Precision / Recall Over Time")
    full_col = st.container()
    _metric_panel(full_col, "accuracy_pct",  COLORS["primary"],   "Accuracy")
    _metric_panel(full_col, "precision_pct", COLORS["secondary"], "Precision")
    _metric_panel(full_col, "recall_pct",    COLORS["blue"],      "Recall")

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

    render_labeled_divider("Average Gain Over Time by Signal")

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


# ── Sub-tab 5 — System Info ────────────────────────────────────────────────────
def _render_system_info():
    render_section_header(5, "System Info")
    st.markdown("#### System Information")

    log_df = _get_table_full("ml_screening_logs")
    _warn_if_truncated(log_df, "ml_screening_logs")

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
            render_labeled_divider("Screening Funnel")
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
            render_labeled_divider("Recent Screening Runs")
            show_cols = [c for c in log_df.columns if c != "id"]
            st.dataframe(log_df[show_cols], use_container_width=True, hide_index=True)

        extra_cols = [c for c in log.index if c not in [col for col, _ in funnel_steps] + ["id"]]
        if extra_cols:
            with st.expander("All log fields"):
                st.json({c: str(log[c]) for c in extra_cols if pd.notna(log[c])})

    else:
        st.info("No screening logs found yet.")

    render_labeled_divider("Database Summary")

    preds_df  = _get_table_full("ml_explosion_predictions")
    acc_df    = _get_table_full("ml_prediction_accuracy")
    missed_df = _get_table_full("ml_missed_opportunities")

    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction Records",  len(preds_df))
    col2.metric("Accuracy Records",    len(acc_df))
    col3.metric("Missed Opp. Records", len(missed_df))
    if preds_df.attrs.get("truncated") or acc_df.attrs.get("truncated") or missed_df.attrs.get("truncated"):
        st.caption(
            f"⚠️ One or more counts above are capped at the most recent {_TABLE_ROW_CAP} "
            "rows — actual table sizes may be larger."
        )

    if not acc_df.empty:
        acc_df = acc_df.copy()
        acc_df["became_winner"]      = acc_df["became_winner"].astype(bool)
        acc_df["prediction_correct"] = acc_df["prediction_correct"].astype(bool)
        col1, col2 = st.columns(2)
        col1.metric("Overall Accuracy (all dates)", f"{acc_df['prediction_correct'].mean()*100:.1f}%")
        col2.metric("Overall Winner Rate",          f"{acc_df['became_winner'].mean()*100:.1f}%")

    render_labeled_divider("Automated Schedule (Estonia Time)")

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

    debug_errors = st.session_state.get("_debug_errors", [])
    if debug_errors:
        with st.expander(f"Debug Log ({len(debug_errors)} error(s) this session)"):
            st.caption(
                "Failures from Supabase/yfinance calls that were handled quietly "
                "elsewhere in the UI are logged here instead of being lost."
            )
            st.dataframe(
                pd.DataFrame(list(reversed(debug_errors))),
                use_container_width=True, hide_index=True,
            )
            if st.button("Clear debug log", key=f"{TAB_ID}_clear_debug_log"):
                st.session_state["_debug_errors"] = []
                st.rerun()
