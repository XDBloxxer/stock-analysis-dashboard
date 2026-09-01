"""
Backtesting Tab Module  (v5)

v5: Replaced the daily close-based approximation with real 5-minute intraday
sequencing as the only resolution method — every trade is walked against
actual yfinance bars to see whichever level (stop or target) was really
touched first. Nothing runs automatically: the date range is locked to the
last 60 days (yfinance's 5-min history limit), and results only appear
after pressing "🕵️ Interrogate the candles", so tweaking a slider never
kicks off a silent recompute or a downgrade to a rougher estimate.

v4: Added an optional side-by-side comparison mode (run two independently
configured simulations over the same capital/date range and overlay both
portfolio curves), plus a max-drawdown stat on every simulation result.

v3: Replaced the old strategy-creation / GitHub-Actions backtesting system
(which was no longer used) with the Cumulative Gain Simulator, moved here
in its entirety from the Performance Trends sub-tab of ML Predictions.
"""

import hashlib
import time
import datetime as _dt
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

from db import get_supabase_client, run_with_retry, log_debug_error
from datetime import date as _date
from chart_utils import LAYOUT, AXIS_STYLE, COLORS
from cache_ui import render_cache_buttons
from dashboard_styles import render_section_header, render_empty_state, render_skeleton_rows, render_labeled_divider, render_hero_metric
import user_state

TAB_ID = "backtesting"

_DATE_COL   = "prediction_date"
_TABLE_NAME = "ml_prediction_accuracy"
_SELECT = (
    "symbol,prediction_date,predicted_probability,predicted_signal,"
    "predicted_target_gain,became_winner,actual_gain_pct,actual_high_pct,"
    "actual_price,prediction_correct,gain_error_pct"
)

_ALL_SIGNALS = ["STRONG BUY", "BUY", "HOLD", "AVOID"]


def _info_card(text: str, muted: bool = False) -> None:
    """Render a note/caption inside a bordered card instead of bare floating text."""
    cls = "info-card muted" if muted else "info-card"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


# ── Cached DB fetcher (paginates the full table — used for full-history sim) ──
@st.cache_data(show_spinner=False)
def _get_table_all() -> pd.DataFrame:
    client    = get_supabase_client()
    page_size = 1000
    offset    = 0
    frames    = []
    try:
        while True:
            def _run(offset=offset):
                query = (
                    client.table(_TABLE_NAME)
                    .select(_SELECT)
                    .order(_DATE_COL, desc=False)
                )
                return query.range(offset, offset + page_size - 1).execute()

            response = run_with_retry(_run, source=f"_get_table_all({_TABLE_NAME})")
            rows = response.data or []
            if not rows:
                break
            frames.append(pd.DataFrame(rows))
            if len(rows) < page_size:
                break
            offset += page_size
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception as e:
        log_debug_error(f"_get_table_all({_TABLE_NAME})", e)
        st.warning(f"Could not load full history for `{_TABLE_NAME}`: {e}")
        return pd.DataFrame()


# ── Cache control ──────────────────────────────────────────────────────────
def clear_all_cache():
    _get_table_all.clear()


def refresh_cache():
    _get_table_all.clear()
    _get_table_all()
    st.toast("✅ Cache refreshed.")


# ── SPY buy-and-hold benchmark ───────────────────────────────────────────────
# Answers the question every backtest viewer actually has: "does this beat
# just buying the index and doing nothing?" Fetched via yfinance (already a
# project dependency for live quotes elsewhere) rather than sourced from
# Supabase, since the accuracy table only carries the screened universe's own
# symbols, never a benchmark. Cached for an hour — this is history, not a
# live quote, so it doesn't need the 60s live-quote cadence used elsewhere.
@st.cache_data(show_spinner=False, ttl=3600)
def _get_spy_benchmark(start: _date, end: _date) -> pd.Series | None:
    """Daily SPY close prices from `start` to `end` inclusive, normalized to
    a growth-of-$1 series (first value = 1.0), or None if the fetch failed
    (offline environment, yfinance hiccup, etc.) — callers should treat that
    as "benchmark unavailable" and just skip the overlay rather than error."""
    try:
        import yfinance as yf
        hist = yf.download(
            "SPY", start=start, end=end + pd.Timedelta(days=1),
            progress=False, auto_adjust=True,
        )
        if hist is None or hist.empty:
            return None
        closes = hist["Close"]
        if hasattr(closes, "columns"):  # yfinance sometimes returns a 1-col DataFrame
            closes = closes.iloc[:, 0]
        closes = closes.dropna()
        if closes.empty:
            return None
        return closes / closes.iloc[0]
    except Exception as e:
        log_debug_error("_get_spy_benchmark", e)
        return None


def _add_benchmark_trace(fig: go.Figure, sim_result: pd.DataFrame, start_capital: float):
    """Overlays a 'SPY buy & hold' line on `fig`, normalized to the same
    starting capital and date range as `sim_result` — no-op (with a small
    caption, handled by the caller) if the benchmark can't be fetched."""
    bench_start = sim_result["prediction_date"].min().date()
    bench_end   = sim_result["prediction_date"].max().date()
    growth = _get_spy_benchmark(bench_start, bench_end)
    if growth is None or growth.empty:
        return False
    bench_df = pd.DataFrame({
        "date": pd.to_datetime(growth.index),
        "value": growth.values * start_capital,
    })
    # Reindex the benchmark onto the sim's own trade dates (forward-filled)
    # so the two lines are directly comparable point-for-point in the hover
    # tooltip, rather than SPY's full daily calendar vs. the sim's
    # trade-day-only calendar.
    bench_df = bench_df.set_index("date").reindex(
        pd.to_datetime(sim_result["prediction_date"]), method="ffill"
    )
    fig.add_trace(go.Scatter(
        x=sim_result["prediction_date"], y=bench_df["value"],
        mode="lines", name="SPY Buy & Hold",
        line=dict(color="rgba(148,163,184,0.85)", width=1.6, dash="dot"),
        hovertemplate="SPY Buy & Hold<br>%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>",
    ))
    return True


# ── Precise Sequencing (5-min intraday walk-forward) ─────────────────────────
# The only resolution method: instead of guessing stop/target order from the
# day's resolved close-to-close gain, this walks actual 5-minute bars from
# yfinance for each trade and resolves whichever level — stop or target —
# is actually touched first.
#
# Hard limit: yfinance only serves ~60 days of 5-minute history, so the date
# range picker below is locked to that window — there's nothing to fall back
# to for older trades, so they're simply not offered as an option.
_FIVE_MIN_MAX_DAYS = 60
_FIVE_MIN_ET = "America/New_York"


def _five_min_cutoff_date() -> _date:
    """Oldest prediction_date still eligible for 5-min sequencing, as of today."""
    return _date.today() - _dt.timedelta(days=_FIVE_MIN_MAX_DAYS)


def _filter_sim_trades(pos_signals: pd.DataFrame, signals: list, sim_start, sim_end) -> pd.DataFrame:
    """The exact trade-matching filter used at the top of `_simulate`, factored
    out so the Precise Sequencing button can build its (symbol, date) work list
    from the identical set of trades the simulation itself will use — the two
    can never silently drift apart."""
    return pos_signals[
        (pos_signals["predicted_signal"].isin(signals))
        & (pos_signals["prediction_date"].dt.date >= sim_start)
        & (pos_signals["prediction_date"].dt.date <= sim_end)
        & (pos_signals["actual_gain_pct"].notna())
    ].copy()


def _fetch_5min_bars_one(symbol: str, start: _date, end: _date, retries: int = 2):
    """Worker: one symbol's 5-minute bars (incl. pre/post-market) spanning
    `start`..`end`, tz-normalized to America/New_York so wall-clock session
    boundaries (4:00 / 9:30 / 16:00 ET) can be compared directly against the
    index. Returns (symbol, DataFrame | None) — None on any failure or empty
    result, never raises, so one bad symbol can't take down the batch.

    Retries a couple of times with backoff before giving up — Yahoo's
    undocumented endpoint frequently rate-limits/blanks cloud-hosted
    requests, and a retry after a short pause often succeeds where the
    first attempt silently came back empty.

    Every failure path — exception OR an empty-but-no-exception response —
    is logged via log_debug_error so it's visible in the System Info tab's
    Debug Log. Previously only exceptions were logged, so a rate-limited
    "empty response, no error raised" case (the most common yfinance
    cloud-IP failure mode) left literally no trace of why a symbol dropped.
    """
    import yfinance as yf
    last_err = None
    for attempt in range(retries + 1):
        try:
            hist = yf.Ticker(symbol).history(
                start=start, end=end + _dt.timedelta(days=1),
                interval="5m", prepost=True, auto_adjust=True,
            )
            if hist is None or hist.empty:
                last_err = "empty response (no exception) — likely rate-limited or no data for this symbol"
                if attempt < retries:
                    time.sleep(0.6 * (attempt + 1))
                    continue
                log_debug_error(f"_fetch_5min_bars_one({symbol})", RuntimeError(last_err))
                return symbol, None
            if hist.index.tz is None:
                hist.index = hist.index.tz_localize(_FIVE_MIN_ET)
            else:
                hist.index = hist.index.tz_convert(_FIVE_MIN_ET)
            return symbol, hist
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
                continue
            log_debug_error(f"_fetch_5min_bars_one({symbol})", e)
            return symbol, None
    return symbol, None


def _fetch_5min_bars_batch(symbols: tuple, start: _date, end: _date) -> dict:
    """One yfinance call per symbol (not per trade) spanning that symbol's
    full needed date range, fanned out across a thread pool — keeps request
    count proportional to the number of distinct tickers touched, not the
    number of trades.

    Concurrency capped lower than before (4, was 8): hitting Yahoo's
    endpoint with many simultaneous connections from one cloud IP is a
    common trigger for the silent empty-response rate-limiting that was
    causing most trades to drop with no visible error."""
    result = {}
    if not symbols:
        return result
    max_workers = min(4, len(symbols))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_5min_bars_one, sym, start, end): sym for sym in symbols}
        for future in as_completed(futures):
            sym, hist = future.result()
            result[sym] = hist
    return result


def _resolve_trade_from_bars(bars: pd.DataFrame | None, trade_date: _date,
                              stop_loss_pct: float | None, target_pct: float | None) -> tuple[dict | None, str | None]:
    """
    Resolves one trade against real 5-minute bars. Returns (info, reason):
    info is None if there isn't enough data to resolve (a delisting, a data
    gap, etc. — the caller drops that trade from the simulation rather than
    guessing); reason is a short machine-stable string identifying *why*
    when info is None (used to build a drop-reason breakdown for the UI —
    "no bars at all" and "had bars but couldn't be resolved" look identical
    to the user otherwise, and are usually different underlying problems).

    Entry price: the last available bar's Close strictly before that day's
    4:00 AM ET pre-market open — i.e. the prior session's after-hours close
    if one exists, or the prior day's regular-session close if it doesn't.
    Matches how entry price is already defined elsewhere in this dashboard.

    Resolution: walks 5-minute bars from 9:30 AM to 4:00 PM ET on
    `trade_date` in order. The first bar whose Low reaches the stop level
    resolves the trade at exactly -stop_loss_pct; the first bar (that
    didn't already stop out) whose High reaches the target resolves it at
    exactly target_pct. If neither is ever touched, the trade resolves at
    the last session bar's Close. Stop is checked before target within the
    same bar — the conservative assumption when both could plausibly have
    happened inside one 5-minute candle.
    """
    if bars is None or bars.empty or "Close" not in bars.columns:
        return None, "no_bars_fetched"

    tz = bars.index.tz
    pre_market_open = pd.Timestamp.combine(trade_date, _dt.time(4, 0)).tz_localize(tz)
    market_open      = pd.Timestamp.combine(trade_date, _dt.time(9, 30)).tz_localize(tz)
    market_close      = pd.Timestamp.combine(trade_date, _dt.time(16, 0)).tz_localize(tz)

    pre_bars = bars[bars.index < pre_market_open]
    if pre_bars.empty:
        return None, "no_prior_bar_for_entry_price"
    entry_price = float(pre_bars["Close"].iloc[-1])
    if not entry_price or entry_price <= 0:
        return None, "invalid_entry_price"

    session = bars[(bars.index >= market_open) & (bars.index <= market_close)]
    if session.empty or "High" not in session.columns or "Low" not in session.columns:
        return None, "no_session_bars_for_trade_date"

    resolved_gain_pct = None
    method = None
    for _, row in session.iterrows():
        low_pct  = (float(row["Low"])  - entry_price) / entry_price * 100
        high_pct = (float(row["High"]) - entry_price) / entry_price * 100
        if stop_loss_pct is not None and low_pct <= -abs(stop_loss_pct):
            resolved_gain_pct = -abs(stop_loss_pct)
            method = "stop-loss hit (5-min)"
            break
        if target_pct is not None and high_pct >= target_pct:
            resolved_gain_pct = target_pct
            method = "target hit (5-min)"
            break

    if resolved_gain_pct is None:
        last_close = float(session["Close"].iloc[-1])
        resolved_gain_pct = (last_close - entry_price) / entry_price * 100
        method = "closed at session end (5-min)"

    return {
        "resolved_gain_pct": resolved_gain_pct,
        "resolution_method": method,
        "entry_price": entry_price,
    }, None


def _build_precise_map(eligible_trades: pd.DataFrame, use_stop_loss: bool,
                        stop_loss_pct: float, use_take_profit: bool) -> tuple[dict, dict]:
    """
    Fetches 5-min bars for every distinct symbol in `eligible_trades` (one
    batched call per symbol) and resolves each trade against them.

    Returns (precise_map, stats):
      precise_map: {(symbol, trade_date): {resolved_gain_pct, resolution_method, entry_price}}
                   only containing trades that were actually resolvable from
                   real bars — callers should drop any trade with no matching
                   key rather than guess at a resolution for it.
      stats: coverage counters for the UI (eligible / resolved / symbols_fetched).
    """
    if eligible_trades.empty:
        return {}, {"eligible": 0, "resolved": 0, "symbols_fetched": 0}

    symbols = tuple(sorted(eligible_trades["symbol"].dropna().unique()))
    dates = eligible_trades["prediction_date"].dt.date
    # A few extra days of lookback buffer so the very first eligible trade
    # can still find a prior after-hours/close bar to use as its entry,
    # even across a weekend or holiday.
    fetch_start = dates.min() - _dt.timedelta(days=5)
    fetch_end   = dates.max()

    bars_by_symbol = _fetch_5min_bars_batch(symbols, fetch_start, fetch_end)
    symbols_with_no_bars = sum(1 for sym in symbols if bars_by_symbol.get(sym) is None)

    precise_map = {}
    resolved = 0
    drop_reasons: dict[str, int] = {}
    for row in eligible_trades.itertuples():
        symbol = row.symbol
        trade_date = row.prediction_date.date()
        target_pct = None
        if use_take_profit:
            raw_target = getattr(row, "predicted_target_gain", None)
            if raw_target is not None and not pd.isna(raw_target):
                target_pct = float(raw_target)
        info, reason = _resolve_trade_from_bars(
            bars_by_symbol.get(symbol), trade_date,
            stop_loss_pct if use_stop_loss else None, target_pct,
        )
        if info is not None:
            precise_map[(symbol, trade_date)] = info
            resolved += 1
        else:
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

    stats = {
        "eligible": len(eligible_trades),
        "resolved": resolved,
        "symbols_fetched": len(symbols),
        "symbols_with_no_bars": symbols_with_no_bars,
        "drop_reasons": drop_reasons,
        "date_range": (fetch_start, fetch_end),
    }
    return precise_map, stats


def _run_config_key(symbols: tuple, sim_start: _date, sim_end: _date, signals: list,
                     start_capital: float, commission_fee: float, max_positions: int,
                     use_take_profit: bool, use_stop_loss: bool, stop_loss_pct: float,
                     weight_by_confidence: bool, slippage_bps: float,
                     max_deploy_pct: float) -> str:
    """Fingerprint of every widget that feeds a run, so a stale cached
    result (from before the user tweaked *anything*) is never silently
    re-rendered as if it were current — the button has to be pressed again."""
    raw = repr((
        symbols, sim_start, sim_end, tuple(sorted(signals)),
        round(start_capital, 2), round(commission_fee, 2), max_positions,
        use_take_profit, use_stop_loss,
        round(stop_loss_pct, 2) if use_stop_loss else None,
        weight_by_confidence, round(slippage_bps, 2), round(max_deploy_pct, 2),
    ))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _eligible_trades_and_key(
    pos_signals: pd.DataFrame, sim_signals: list, sim_start: _date, sim_end: _date,
    start_capital: float, commission_fee: float, max_positions: int,
    use_take_profit: bool, use_stop_loss: bool, stop_loss_pct: float,
    weight_by_confidence: bool, slippage_bps: float, max_deploy_pct: float,
) -> tuple[pd.DataFrame, str]:
    """Factored out so the submit-button `disabled=` check (computed before
    the form's submit button is drawn) and the actual run use the identical
    eligibility filter and fingerprint — see _render_precise_fetch_panel."""
    eligible_trades = _filter_sim_trades(pos_signals, sim_signals, sim_start, sim_end)
    symbols = tuple(sorted(eligible_trades["symbol"].dropna().unique()))
    current_key = _run_config_key(
        symbols, sim_start, sim_end, sim_signals,
        start_capital, commission_fee, max_positions, use_take_profit,
        use_stop_loss, stop_loss_pct, weight_by_confidence,
        slippage_bps, max_deploy_pct,
    )
    return eligible_trades, current_key


def _render_precise_fetch_panel(
    pos_signals: pd.DataFrame, sim_signals: list, sim_start: _date, sim_end: _date,
    start_capital: float, commission_fee: float, max_positions: int,
    use_take_profit: bool, use_stop_loss: bool, stop_loss_pct: float,
    weight_by_confidence: bool, slippage_bps: float, max_deploy_pct: float,
    submitted: bool,
    key_prefix: str = "sim",
) -> dict | None:
    """
    Runs the simulation, but only when `submitted` is True (the caller's
    `st.form_submit_button` was pressed this run). Fetching 5-min bars AND
    running `_simulate` only happens on that click. The entire result
    (precise map, portfolio series, stats, trade log) is computed once and
    cached in session state, fingerprinted against every widget that could
    change it. On any other rerun — including ones caused by tweaking a
    slider, checkbox, or anything else on this page (all of which now live
    inside an `st.form`, so they don't even trigger a rerun on their own) —
    this function does no fetching, no simulating, and returns exactly the
    same cached bundle it returned last time, so nothing on screen
    recomputes, reflows, or flashes a spinner until you click the button
    again.

    Returns the cached run bundle (dict with keys: config_key, map,
    precise_stats, sim_result, stats, trade_log), or None if nothing has
    been run yet — callers should treat None as "nothing to render yet".
    """
    eligible_trades, current_key = _eligible_trades_and_key(
        pos_signals, sim_signals, sim_start, sim_end,
        start_capital, commission_fee, max_positions, use_take_profit,
        use_stop_loss, stop_loss_pct, weight_by_confidence,
        slippage_bps, max_deploy_pct,
    )
    symbols = tuple(sorted(eligible_trades["symbol"].dropna().unique()))

    cache_bucket = f"{key_prefix}_run_cache"
    cached = st.session_state.get(cache_bucket)

    if submitted and not eligible_trades.empty:
        with st.spinner(f"Fetching 5-minute bars for {len(symbols)} symbol(s)..."):
            precise_map, precise_stats = _build_precise_map(
                eligible_trades, use_stop_loss, stop_loss_pct, use_take_profit,
            )
            sim_result, stats, trade_log = _simulate(
                pos_signals, sim_signals, sim_start, sim_end,
                start_capital, commission_fee, max_positions, use_take_profit,
                use_stop_loss=use_stop_loss, stop_loss_pct=stop_loss_pct,
                weight_by_confidence=weight_by_confidence,
                slippage_bps=slippage_bps, max_deploy_pct=max_deploy_pct,
                precise_map=precise_map,
            )
        st.session_state[cache_bucket] = {
            "config_key": current_key,
            "map": precise_map,
            "precise_stats": precise_stats,
            "sim_result": sim_result,
            "stats": stats,
            "trade_log": trade_log,
            "config": {
                "signals": sim_signals, "sim_start": sim_start, "sim_end": sim_end,
                "start_capital": start_capital, "commission_fee": commission_fee,
                "max_positions": max_positions, "use_take_profit": use_take_profit,
                "use_stop_loss": use_stop_loss, "stop_loss_pct": stop_loss_pct,
                "weight_by_confidence": weight_by_confidence,
                "slippage_bps": slippage_bps, "max_deploy_pct": max_deploy_pct,
            },
        }
        st.toast(f"✅ Resolved {precise_stats['resolved']} of {precise_stats['eligible']} trades from real 5-min bars.")
        cached = st.session_state[cache_bucket]

    # Caption reflects the *post-run* state (computed above, if this was a
    # submit run) rather than the state from before the click — otherwise a
    # first-ever click would show "not yet run" above results that just
    # rendered successfully underneath it.
    is_fresh = cached is not None and cached.get("config_key") == current_key
    if eligible_trades.empty:
        st.caption("No trades match this signal/date selection yet.")
    elif cached is None:
        st.caption("Set your conditions above, then click \"🕵️ Interrogate the candles\" to run the simulation.")
    elif is_fresh:
        precise_stats = cached["precise_stats"]
        dropped = precise_stats["eligible"] - precise_stats["resolved"]
        drop_note = f", {dropped} dropped (no bars)" if dropped else ""
        st.caption(
            f"✅ {precise_stats['resolved']} of {precise_stats['eligible']} trades resolved from real "
            f"5-min bars across {precise_stats['symbols_fetched']} symbol(s){drop_note}."
        )
        no_bars = precise_stats.get("symbols_with_no_bars", 0)
        reasons = precise_stats.get("drop_reasons") or {}
        if dropped:
            reason_labels = {
                "no_bars_fetched": "yfinance returned nothing for the symbol (rate-limited or no data)",
                "no_prior_bar_for_entry_price": "no bar before market open to price the entry",
                "invalid_entry_price": "entry price was zero/invalid",
                "no_session_bars_for_trade_date": "no bars during the trade's session",
            }
            reason_lines = "\n".join(
                f"- {reason_labels.get(r, r)}: {n}" for r, n in sorted(reasons.items(), key=lambda x: -x[1])
            )
            with st.expander(f"Why {dropped} trade(s) were dropped"):
                st.markdown(reason_lines)
                if no_bars:
                    st.caption(
                        f"{no_bars} of {precise_stats['symbols_fetched']} symbol(s) returned **zero** 5-min "
                        "bars at all. If that's most of them, this is very likely Yahoo Finance "
                        "rate-limiting/blocking requests from this server's IP rather than the symbols "
                        "genuinely lacking data — check the Debug Log in System Info for "
                        "\"empty response\" / rate-limit entries, and try again in a few minutes or with "
                        "a smaller symbol set."
                    )
    else:
        st.caption(
            "⏸️ Showing results from the last run — a setting has changed since then. "
            "Click \"🕵️ Interrogate the candles\" again to fetch and re-run with the current settings."
        )

    return cached


# ── Simulation core (shared by single-run and comparison modes) ─────────────
def _simulate(
    pos_signals: pd.DataFrame,
    signals: list,
    sim_start,
    sim_end,
    start_capital: float,
    commission_fee: float,
    max_positions: int,
    use_take_profit: bool,
    use_stop_loss: bool = False,
    stop_loss_pct: float = 8.0,
    weight_by_confidence: bool = False,
    slippage_bps: float = 0.0,
    max_deploy_pct: float = 100.0,
    precise_map: dict | None = None,
):
    """
    Runs the per-trade cumulative-gain simulation for one configuration.
    Returns (sim_result_df, stats_dict, trade_log_df) or (None, None, None)
    if no trades match.

    Every trade here is a same-day round trip (entry and exit resolve
    within `prediction_date`), so day-to-day capital reuse is valid — there
    is no multi-day holding-period overlap to account for.

    use_stop_loss: caps the loss on any trade at -stop_loss_pct.
    If `actual_low_pct` is present in `pos_signals` (an intraday-low column,
    fetched opportunistically — see _get_table_all), the stop is checked
    against whether price actually traded down to the stop level during the
    day: any trade whose low reached -stop_loss_pct or worse is resolved at
    exactly -stop_loss_pct, regardless of where it closed. This correctly
    catches "dipped through the stop, then recovered by close" trades that
    a close-only approximation would silently record as winners.
    If `actual_low_pct` is NOT available, this falls back to capping the
    day's resolved gain at -stop_loss_pct — which only ever helps trades
    that already closed negative and can never catch a trade that
    intraday-dipped below the stop and recovered to close positive. That
    fallback is therefore optimistically biased (understates how often and
    how often the stop would really trigger); this is called out in the UI.
    When both stop-loss and take-profit are enabled and low data isn't
    available to sequence the day, the stop is checked first (the more
    conservative assumption) before the take-profit target is applied.

    weight_by_confidence: position size is proportional to
    predicted_probability among the trades taken that day, instead of an
    equal split.

    slippage_bps: an estimated round-trip cost (entry + exit), in basis
    points of position size, applied on top of the flat commission_fee —
    approximates spread/market-impact cost that a flat per-trade commission
    doesn't capture, especially for thinner names.

    max_deploy_pct: the share of available capital actually put to work
    each day (0-100). The remainder is held as idle cash (0% return) and
    carried to the next day rather than force-deployed into every signal
    that happens to fire — a more realistic model than always being 100%
    invested regardless of conviction or number of signals.

    precise_map: optional {(symbol, trade_date): {resolved_gain_pct,
    resolution_method, entry_price}} from _build_precise_map, produced by
    the Precise Sequencing (5-min) button. Where a trade has an entry, its
    resolved_gain_pct/method OVERRIDE whatever the approximation logic
    above computed for it (real intraday sequencing beats a same-day-close
    guess). Trades with no matching entry — outside yfinance's ~60-day
    5-min window, or a fetch miss — silently keep their approximated
    resolution; nothing is blended, every trade is labeled either way via
    the `resolution_method` column added to the trade log below.
    """
    sim_df = _filter_sim_trades(pos_signals, signals, sim_start, sim_end)

    if sim_df.empty:
        return None, None, None

    if "predicted_probability" in sim_df.columns:
        sim_df = sim_df.sort_values("predicted_probability", ascending=False)

    has_low = "actual_low_pct" in sim_df.columns and sim_df["actual_low_pct"].notna().any()
    stopped_out = pd.Series(False, index=sim_df.index)

    if use_stop_loss and has_low:
        # Real check: did price actually trade down to the stop level
        # during the day? Applied before take-profit resolution below —
        # a trade that got stopped out never lives to hit its target.
        low = pd.to_numeric(sim_df["actual_low_pct"], errors="coerce")
        stopped_out = low.notna() & (low <= -abs(stop_loss_pct))

    if use_take_profit and "actual_high_pct" in sim_df.columns and "predicted_target_gain" in sim_df.columns:
        target = pd.to_numeric(sim_df["predicted_target_gain"], errors="coerce")
        high   = pd.to_numeric(sim_df["actual_high_pct"], errors="coerce")
        hit_target = target.notna() & high.notna() & (high >= target) & ~stopped_out
        sim_df["resolved_gain_pct"] = np.where(hit_target, target, sim_df["actual_gain_pct"])
    else:
        sim_df["resolved_gain_pct"] = sim_df["actual_gain_pct"]

    if use_stop_loss:
        if has_low:
            # Exact stop level for any trade whose low actually reached it;
            # everything else keeps its take-profit/close resolution above.
            sim_df["resolved_gain_pct"] = np.where(
                stopped_out, -abs(stop_loss_pct), sim_df["resolved_gain_pct"]
            )
        else:
            # Approximation fallback (no intraday-low data available) — see
            # docstring above for the optimistic-bias caveat.
            sim_df["resolved_gain_pct"] = np.maximum(sim_df["resolved_gain_pct"], -abs(stop_loss_pct))

    sim_df["resolution_method"] = (
        "approximated (daily, real low)" if has_low else "approximated (daily)"
    )

    if precise_map:
        for idx in sim_df.index:
            key = (sim_df.at[idx, "symbol"], sim_df.at[idx, "prediction_date"].date())
            info = precise_map.get(key)
            if info is not None:
                sim_df.at[idx, "resolved_gain_pct"]  = info["resolved_gain_pct"]
                sim_df.at[idx, "resolution_method"]  = info["resolution_method"]

    invested_frac = max(0.0, min(1.0, max_deploy_pct / 100.0))

    trade_rows = []
    records = []
    capital = start_capital
    for trade_date, gdf in sim_df.groupby("prediction_date", sort=True):
        trades = gdf.head(max_positions)
        n = len(trades)
        if n == 0:
            continue

        if weight_by_confidence and "predicted_probability" in trades.columns:
            probs = pd.to_numeric(trades["predicted_probability"], errors="coerce").fillna(0.0)
            if probs.sum() > 0:
                weights = probs / probs.sum()
            else:
                weights = pd.Series([1.0 / n] * n, index=trades.index)
        else:
            weights = pd.Series([1.0 / n] * n, index=trades.index)

        idle_cash = capital * (1 - invested_frac)
        day_end_capital = idle_cash
        for idx, gain in trades["resolved_gain_pct"].items():
            position_size = capital * weights.loc[idx] * invested_frac
            slippage_cost = position_size * (slippage_bps / 10000.0)
            pnl_dollars = position_size * (gain / 100) - commission_fee - slippage_cost
            resolved = max(position_size + pnl_dollars, 0.0)
            day_end_capital += resolved
            trade_rows.append({
                "prediction_date": trade_date,
                "symbol": trades.loc[idx, "symbol"] if "symbol" in trades.columns else None,
                "signal": trades.loc[idx, "predicted_signal"] if "predicted_signal" in trades.columns else None,
                "predicted_probability": trades.loc[idx].get("predicted_probability") if hasattr(trades.loc[idx], "get") else None,
                "position_size": position_size,
                "resolved_gain_pct": gain,
                "resolution_method": trades.loc[idx, "resolution_method"] if "resolution_method" in trades.columns else "approximated (daily)",
                "stopped_out": bool(stopped_out.loc[idx]) if idx in stopped_out.index else False,
                "pnl_dollars": pnl_dollars,
                "commission_fee": commission_fee,
                "slippage_cost": slippage_cost,
            })
        capital = day_end_capital
        records.append({
            "prediction_date": trade_date,
            "trades_taken": n,
            "portfolio_value": capital,
        })

    if not records:
        return None, None, None

    sim_result = pd.DataFrame(records).sort_values("prediction_date").reset_index(drop=True)
    trade_log  = pd.DataFrame(trade_rows).sort_values("prediction_date").reset_index(drop=True)

    final_value  = sim_result["portfolio_value"].iloc[-1]
    total_return = (final_value - start_capital) / start_capital * 100
    n_trades     = int(sim_result["trades_taken"].sum())
    n_days       = len(sim_result)
    total_fees      = n_trades * commission_fee
    total_slippage  = float(trade_log["slippage_cost"].sum()) if not trade_log.empty else 0.0
    # Win rate must be computed from the trades actually taken (trade_log),
    # not the full signal-matched set (sim_df) — sim_df still includes rows
    # that got cut by the max_positions-per-day cap (the lowest-confidence
    # signals, since sim_df is pre-sorted by predicted_probability desc
    # before the groupby/head()). Using sim_df here silently mixed in
    # never-executed trades and skewed the displayed win rate away from
    # what the simulated capital actually experienced.
    win_rate     = (trade_log["resolved_gain_pct"] > 0).mean() * 100 if not trade_log.empty else 0.0

    # Sharpe-like ratio on the day-over-day portfolio return series — "-like"
    # because these are simulated per-day compounding steps, not literal
    # daily market returns, so treat it as a rough risk-adjusted-return
    # signal rather than a textbook Sharpe ratio. Annualized using the
    # simulation's own observed trading cadence (return-observations per
    # calendar year over the actual date span) rather than a flat 252 —
    # a strategy that only trades a handful of days a year shouldn't get
    # scaled up as if it traded every session.
    daily_returns = sim_result["portfolio_value"].pct_change().dropna()
    if len(daily_returns) >= 2 and daily_returns.std() > 0:
        date_span_days = (sim_result["prediction_date"].iloc[-1] - sim_result["prediction_date"].iloc[0]).days
        if date_span_days > 0:
            periods_per_year = len(daily_returns) / date_span_days * 365.25
        else:
            periods_per_year = 252.0
        sharpe_like = (daily_returns.mean() / daily_returns.std()) * np.sqrt(periods_per_year)
    else:
        sharpe_like = None

    # Max drawdown: largest peak-to-trough decline in the simulated
    # portfolio value series (computable from day-by-day capital even
    # without intraday low data, since it's about the equity curve itself).
    running_peak = sim_result["portfolio_value"].cummax()
    drawdown_pct = (sim_result["portfolio_value"] - running_peak) / running_peak * 100
    max_drawdown = drawdown_pct.min() if not drawdown_pct.empty else 0.0

    n_trades_precise = 0
    if not trade_log.empty and "resolution_method" in trade_log.columns:
        n_trades_precise = int(trade_log["resolution_method"].str.endswith("(5-min)").sum())

    stats = {
        "final_value": final_value,
        "total_return": total_return,
        "n_trades": n_trades,
        "n_days": n_days,
        "total_fees": total_fees,
        "total_slippage": total_slippage,
        "win_rate": win_rate,
        "sharpe_like": sharpe_like,
        "max_drawdown": max_drawdown,
        "stop_loss_uses_real_low": bool(use_stop_loss and has_low),
        "n_trades_precise": n_trades_precise,
    }
    return sim_result, stats, trade_log


def _render_return_attribution(trade_log: pd.DataFrame, group_col: str, title: str, top_n: int = 15):
    """Bar chart of total $ P&L contribution grouped by `group_col`
    (signal type or symbol) — answers 'what actually drove the return',
    not just 'what was the return'."""
    if trade_log is None or trade_log.empty or group_col not in trade_log.columns:
        return
    attrib = (
        trade_log.groupby(group_col)["pnl_dollars"].sum()
        .sort_values(ascending=False)
    )
    if attrib.empty:
        return
    if len(attrib) > top_n:
        attrib = attrib.head(top_n)
    attrib = attrib.iloc[::-1]
    colors = [COLORS["secondary"] if v >= 0 else COLORS["red"] for v in attrib.values]
    fig = go.Figure(go.Bar(
        x=attrib.values, y=[str(v) for v in attrib.index],
        orientation="h", marker=dict(color=colors),
        hovertemplate="%{y}<br>P&L: $%{x:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=max(260, 26 * len(attrib)),
        margin=dict(t=44, b=24, l=90, r=16),
        **{k: v for k, v in LAYOUT.items() if k != "margin"},
    )
    fig.update_xaxes(title_text="P&L Contribution ($)", **AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig, use_container_width=True)


def _render_trade_log_download(trade_log: pd.DataFrame, key: str):
    if trade_log is None or trade_log.empty:
        return
    st.download_button(
        "⬇️ Download trade log (CSV)",
        trade_log.to_csv(index=False),
        file_name="backtest_trade_log.csv",
        mime="text/csv",
        key=key,
    )


def _run_monte_carlo(daily_returns: pd.Series, start_capital: float, n_sims: int = 500, seed: int = 42):
    """
    Bootstrap-resamples the simulation's own day-over-day portfolio returns
    (with replacement) into `n_sims` alternate equity-curve paths of the
    same length as the actual run. This reshuffles the order/combination of
    days that already happened — it does not invent new trades, edges, or
    signals — so it answers "how much did realized path order matter to
    this specific outcome", not "what if the strategy were different."

    Returns an (n_sims, n_days+1) array of portfolio values (column 0 is
    always start_capital), or None if there aren't enough daily
    observations (~5) to resample meaningfully.
    """
    returns = np.asarray(daily_returns.dropna(), dtype=float)
    if len(returns) < 5:
        return None
    rng = np.random.default_rng(seed)
    n_days = len(returns)
    sampled = rng.choice(returns, size=(n_sims, n_days), replace=True)
    growth = np.cumprod(1 + sampled, axis=1)
    paths = start_capital * np.hstack([np.ones((n_sims, 1)), growth])
    return paths


def _render_monte_carlo(sim_result: pd.DataFrame, start_capital: float, key_prefix: str,
                          show_divider: bool = True, chart_title: str | None = None):
    """Renders the Monte Carlo controls + fan chart + final-value histogram
    for one simulation result. Safe to call multiple times with different
    key_prefix values (e.g. once per side in comparison mode)."""
    if show_divider:
        render_labeled_divider("Monte Carlo Resampling")
        _info_card(
            "Resamples this run's own day-over-day returns with replacement to build alternate "
            "equity-curve paths of the same length — a distribution of plausible outcomes from the "
            "same edge and trade cadence, rather than the single deterministic path that actually "
            "occurred over this date range. It reshuffles day order/magnitude; it does not invent "
            "new trades or a different edge, and (like the deterministic run above) still assumes "
            "every trade fills at its resolved gain with no slippage.",
            muted=True,
        )

    run_mc = st.checkbox(
        "Run Monte Carlo resampling" if not chart_title else f"Run Monte Carlo resampling — {chart_title}",
        key=f"{key_prefix}_run_mc",
    )
    if not run_mc:
        return

    n_sims = st.slider(
        "Number of simulated paths", min_value=100, max_value=2000, value=500, step=100,
        key=f"{key_prefix}_mc_n_sims",
    )

    daily_returns = sim_result["portfolio_value"].pct_change()
    paths = _run_monte_carlo(daily_returns, start_capital, n_sims=n_sims)
    if paths is None:
        st.info("Not enough daily observations (need at least ~5 trading days) to run a meaningful resampling.")
        return

    pct = {p: np.percentile(paths, p, axis=0) for p in [5, 25, 50, 75, 95]}
    x_idx = list(range(paths.shape[1]))
    title = chart_title or f"Monte Carlo Fan Chart — {n_sims} resampled paths"

    fig = go.Figure()
    # Outer band (5th–95th pct) — two invisible-line traces with a fill
    # between them, the standard Plotly pattern for a shaded percentile band.
    fig.add_trace(go.Scatter(x=x_idx, y=pct[95], mode="lines", line=dict(width=0),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x_idx, y=pct[5], mode="lines", line=dict(width=0),
                              fill="tonexty", fillcolor="rgba(224,168,60,0.08)",
                              name="5th–95th percentile", hoverinfo="skip"))
    # Inner band (25th–75th pct), drawn darker so the interquartile range
    # reads as the "likely" zone against the wider tail band underneath.
    fig.add_trace(go.Scatter(x=x_idx, y=pct[75], mode="lines", line=dict(width=0),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x_idx, y=pct[25], mode="lines", line=dict(width=0),
                              fill="tonexty", fillcolor="rgba(224,168,60,0.20)",
                              name="25th–75th percentile", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x_idx, y=pct[50], mode="lines",
                              line=dict(color=COLORS["primary"], width=2.2), name="Median path"))
    fig.add_trace(go.Scatter(x=x_idx, y=[start_capital] * len(x_idx), mode="lines",
                              line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dash"),
                              name="Starting capital", hoverinfo="skip"))
    fig.update_layout(
        title=title, xaxis_title="Trading day #", yaxis_title="Portfolio Value ($)",
        height=380, hovermode="x unified", **LAYOUT,
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig, use_container_width=True)

    finals = paths[:, -1]
    prob_loss = float((finals < start_capital).mean() * 100)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Median Final Value", f"${np.median(finals):,.2f}")
    m2.metric("5th Percentile", f"${np.percentile(finals, 5):,.2f}",
              help="Worst-ish plausible outcome across resampled paths — 5% of paths finished below this.")
    m3.metric("95th Percentile", f"${np.percentile(finals, 95):,.2f}",
              help="Best-ish plausible outcome across resampled paths — 5% of paths finished above this.")
    m4.metric("P(Loss)", f"{prob_loss:.1f}%",
              help="Share of resampled paths that ended below starting capital.")

    hist_fig = go.Figure(go.Histogram(
        x=finals, nbinsx=40, marker=dict(color=COLORS["primary"]),
        hovertemplate="$%{x:,.0f}<br>Paths: %{y}<extra></extra>",
    ))
    hist_fig.add_vline(
        x=start_capital, line_dash="dash", line_color="rgba(255,255,255,0.3)",
        annotation_text="Starting capital", annotation_font_size=10,
    )
    hist_fig.update_layout(
        title="Distribution of Resampled Final Portfolio Values",
        xaxis_title="Final Portfolio Value ($)", yaxis_title="Paths", height=280, **LAYOUT,
    )
    hist_fig.update_xaxes(**AXIS_STYLE)
    hist_fig.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(hist_fig, use_container_width=True)


def _threshold_segments(x, y, threshold: float):
    """
    Split an (x, y) line into contiguous segments, each tagged 'above' or
    'below' a threshold value, with a linearly-interpolated point inserted
    exactly at each crossing so segments meet cleanly on the threshold line
    instead of jumping color mid-step.

    Returns a list of (x_seg: list, y_seg: list, side: 'above'|'below').
    """
    xs = list(x)
    ys = list(y)
    if not xs:
        return []

    def _side(v):
        return "above" if v >= threshold else "below"

    segments = []
    cur_x = [xs[0]]
    cur_y = [ys[0]]
    cur_side = _side(ys[0])

    for i in range(1, len(xs)):
        x0, y0 = xs[i - 1], ys[i - 1]
        x1, y1 = xs[i], ys[i]
        side1 = _side(y1)

        if side1 != cur_side:
            if y1 != y0:
                frac = (threshold - y0) / (y1 - y0)
                cross_x = x0 + (x1 - x0) * frac
            else:
                cross_x = x1
            cur_x.append(cross_x)
            cur_y.append(threshold)
            segments.append((cur_x, cur_y, cur_side))
            cur_x = [cross_x, x1]
            cur_y = [threshold, y1]
            cur_side = side1
        else:
            cur_x.append(x1)
            cur_y.append(y1)

    segments.append((cur_x, cur_y, cur_side))
    return segments


def _add_drawdown_shading(fig: go.Figure, x, y):
    """
    Shades the underwater/drawdown periods — the gap between the running
    peak (high-water mark) and the current value — in dim red beneath the
    portfolio line. A classic quant-terminal touch: makes it visually
    obvious at a glance how deep and how long each drawdown ran, rather
    than only reading it off the "Max Drawdown" stat as a single number.
    """
    y_arr  = np.asarray(y, dtype=float)
    peak   = np.maximum.accumulate(y_arr)

    fig.add_trace(go.Scatter(
        x=x, y=peak,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.14)", width=1, dash="dot"),
        name="Running Peak",
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_arr,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(239,68,68,0.09)",
        name="Drawdown",
        hoverinfo="skip",
        showlegend=True,
    ))


def _add_threshold_colored_trace(fig: go.Figure, x, y, threshold: float, name: str):
    """
    Adds a portfolio-value line to `fig` that is green where it's at/above
    `threshold` and red where it's below, each with a matching translucent
    fill down to zero. Shows a single combined legend entry for `name`.
    """
    green = "rgba(0,255,136,0.9)"
    green_fill = "rgba(0,255,136,0.10)"
    red = "rgba(255,77,77,0.9)"
    red_fill = "rgba(255,77,77,0.10)"

    segments = _threshold_segments(x, y, threshold)
    legend_shown = False
    for seg_x, seg_y, side in segments:
        is_above = side == "above"
        fig.add_trace(go.Scatter(
            x=seg_x, y=seg_y,
            mode="lines",
            line=dict(color=green if is_above else red, width=2),
            fill="tozeroy",
            fillcolor=green_fill if is_above else red_fill,
            name=name,
            legendgroup=name,
            showlegend=not legend_shown,
            hoverinfo="skip",
        ))
        legend_shown = True

    # Separate marker + hover trace on top, colored per actual data point
    # (not the interpolated crossing points) so hover values stay exact.
    marker_colors = [green if v >= threshold else red for v in y]
    fig.add_trace(go.Scatter(
        x=list(x), y=list(y),
        mode="markers",
        marker=dict(size=5, color=marker_colors),
        name=name,
        legendgroup=name,
        showlegend=False,
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>",
    ))


def _render_stats(stats: dict, key_prefix: str = ""):
    """Headline number rendered as a hero metric (matching Today's Picks /
    Daily Winners' visual weight) with the supporting numbers underneath as
    a regular metric grid — instead of every number competing at the same
    plain st.metric weight.

    The whole metric row is wrapped in an st.container keyed on a hash of
    `stats` itself (plus key_prefix, to keep the A/B panels from colliding
    with each other). NOTE: st.metric itself has never accepted a `key`
    argument (checked 1.37.0 through 1.62.0) — an earlier version of this
    function tried passing key= directly to st.metric() and that raised a
    TypeError in production. st.container DOES support key (since 1.31),
    so keying the container instead achieves the same goal: forcing
    Streamlit to treat a materially different result as a brand-new DOM
    subtree instead of reusing the previous one in place, which is a
    defensive fix for a class of Streamlit frontend desync where a metric
    widget's rendered value can lag behind the value Python actually
    computed for it (confirmed NOT a bug in this function's own logic —
    the debug expander above reads the exact same `stats` dict one line
    before this is called, in the same script run, so the two can never
    actually disagree in the backend)."""
    sharpe_display = f"{stats['sharpe_like']:.2f}" if stats["sharpe_like"] is not None else "N/A"

    stats_fingerprint = hashlib.md5(
        repr(sorted(stats.items(), key=lambda kv: kv[0])).encode("utf-8")
    ).hexdigest()[:10]
    widget_key = f"{key_prefix}_{stats_fingerprint}"

    accent = "green" if stats["total_return"] >= 0 else "red"
    render_hero_metric(
        "Total Return", f"{stats['total_return']:+.2f}%",
        sub=f"${stats['final_value']:,.2f} final value · {stats['n_trades']} trades over {stats['n_days']} day(s)",
        accent=accent,
    )

    with st.container(key=f"{widget_key}_metric_row"):
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Fees Paid", f"${stats['total_fees']:,.2f}")
        m2.metric("Est. Slippage Cost", f"${stats.get('total_slippage', 0.0):,.2f}",
                  help="Estimated spread/market-impact cost from the slippage (bps) setting, on top of commission.")
        m3.metric("Trade Win Rate", f"{stats['win_rate']:.1f}%",
                  help="% of individual simulated trades with a positive resolved gain.")
        m4.metric("Sharpe-like Ratio (annualized)", sharpe_display,
                  help="Mean ÷ std. dev. of day-over-day portfolio returns, annualized using this run's own observed trading frequency (not a flat 252). A rough risk-adjusted-return signal, not a textbook Sharpe ratio.")
        m5.metric("Max Drawdown", f"{stats['max_drawdown']:.1f}%",
                  help="Largest peak-to-trough decline in the simulated portfolio value over the run.")
    if stats.get("stop_loss_uses_real_low"):
        st.caption("✅ Stop-loss checked against actual intraday lows for this run.")
    if stats.get("n_trades_precise", 0) > 0:
        st.caption(
            f"🕵️ {stats['n_trades_precise']} of {stats['n_trades']} trades resolved via real "
            "5-minute intraday sequencing; the rest used the daily approximation (outside the "
            "~60-day 5-min data window, or no bars available)."
        )


def _config_controls(label: str, key_prefix: str, min_date, max_date, default_signals):
    st.markdown(f"**{label}**")
    c1, c2 = st.columns(2)
    with c1:
        signals = st.multiselect(
            "Signal(s) to act on", _ALL_SIGNALS,
            default=default_signals, key=f"{key_prefix}_signals",
        )
    with c2:
        default_exit = st.session_state.get("default_exit_mode", "Close at end of day")
        exit_options = ["Close at end of day", "Take-profit at target if hit intraday, else close"]
        exit_mode = st.selectbox(
            "Exit strategy", exit_options,
            index=exit_options.index(default_exit) if default_exit in exit_options else 0,
            key=f"{key_prefix}_exit_mode",
        )

    persisted_max_positions = st.session_state.get("max_positions", 10)
    max_positions = st.slider(
        "Max concurrent positions per day",
        min_value=1, max_value=25, value=persisted_max_positions, key=f"{key_prefix}_max_positions",
    )

    sl_c1, sl_c2 = st.columns([1, 2])
    with sl_c1:
        use_stop_loss = st.checkbox(
            "Stop-loss", key=f"{key_prefix}_use_stop_loss",
            help="Caps the resolved loss on any single trade at -X%. See note below the results for how this is approximated.",
        )
    with sl_c2:
        stop_loss_pct = st.slider(
            "Stop-loss %", min_value=1.0, max_value=30.0, value=8.0, step=0.5,
            key=f"{key_prefix}_stop_loss_pct", disabled=not use_stop_loss,
        )

    weight_by_confidence = st.checkbox(
        "Weight position size by model confidence (predicted_probability) instead of equal split",
        key=f"{key_prefix}_weight_by_confidence",
    )

    # Persist the primary ("sim") config's choices as the new defaults for
    # next session — the A/B comparison configs stay session-only since
    # they're explicitly meant to diverge from the default.
    if key_prefix == "sim":
        user_state.persist(max_positions=max_positions, default_exit_mode=exit_mode)

    return signals, exit_mode.startswith("Take-profit"), max_positions, use_stop_loss, stop_loss_pct, weight_by_confidence


# ── Main entry point ───────────────────────────────────────────────────────
def render_backtesting_tab():
    render_section_header(1, "Strategy Backtesting")
    st.markdown("Simulate cumulative portfolio performance from the model's historical signals")

    refresh_clicked, clear_confirmed = render_cache_buttons(
        TAB_ID,
        warning_message="⚠️ This will wipe ALL cached backtesting data. Click <strong>Confirm Clear</strong> to proceed.",
    )
    if clear_confirmed:
        clear_all_cache()
        st.rerun()
    if refresh_clicked:
        refresh_cache()
        st.rerun()

    # Only paint the skeleton once per session — see the matching comment in
    # tab_ml_predictions.py. Re-showing/hiding it on every rerun (including
    # ones triggered by widgets in other tabs) resizes this tab's content on
    # every interaction, which triggers Streamlit's known st.tabs
    # scroll-jump bug (streamlit/streamlit#5069).
    _bt_seen_key = f"{TAB_ID}_first_paint_done"
    if not st.session_state.get(_bt_seen_key):
        _bt_placeholder = st.empty()
        with _bt_placeholder.container():
            render_skeleton_rows(4, height=64)
        all_acc = _get_table_all()
        _bt_placeholder.empty()
        st.session_state[_bt_seen_key] = True
    else:
        all_acc = _get_table_all()
    if all_acc.empty:
        render_empty_state("No accuracy data available yet — check back once the model has a track record.")
        return

    all_acc = all_acc.copy()
    all_acc["became_winner"]         = all_acc["became_winner"].astype(bool)
    all_acc["prediction_correct"]    = all_acc["prediction_correct"].astype(bool)
    all_acc["actual_gain_pct"]       = pd.to_numeric(all_acc["actual_gain_pct"],       errors="coerce")
    all_acc["predicted_probability"] = pd.to_numeric(all_acc["predicted_probability"], errors="coerce")
    if "actual_low_pct" in all_acc.columns:
        all_acc["actual_low_pct"] = pd.to_numeric(all_acc["actual_low_pct"], errors="coerce")

    pos_signals = all_acc[all_acc["predicted_signal"].isin(_ALL_SIGNALS)].copy()
    pos_signals["prediction_date"] = pd.to_datetime(pos_signals["prediction_date"])

    if pos_signals.empty:
        st.warning("No signal data available yet.")
        return

    # ── Cumulative Gain Simulator ──────────────────────────────────────────
    st.markdown("#### Cumulative Gain Simulator")
    _info_card(
        "Simulates trading each individual signal as its own same-day position — "
        "capital is split across every signal that fires on a given day (equally, "
        "or by model confidence if selected), each position resolves within that "
        "day, and commission plus an estimated slippage cost are charged per "
        "trade. Positions are closed out and capital is pooled back together at "
        "the end of each day before being redistributed the next. This avoids "
        "inflating results by averaging gains before compounding, but still "
        "simplifies real trading (slippage here is a flat estimate, not modeled "
        "per-symbol from actual spreads; no partial fills; and outcomes are based "
        "on this system's own historical gain data, not independently verified "
        "fills).",
        muted=True,
    )

    data_min_date = pos_signals["prediction_date"].min().date()
    max_date = pos_signals["prediction_date"].max().date()
    # yfinance's 5-min bar history only goes back ~60 days, and the precise
    # sequencing simulation is now the only resolution method, so there's no
    # point letting the picker default to or select a range further back
    # than that — it would just get dropped ("no bars") when interrogated.
    lookback_floor = _dt.date.today() - _dt.timedelta(days=60)
    min_date = max(data_min_date, lookback_floor)

    # Everything the user can tune lives inside a single st.form now. Widgets
    # inside a form do NOT trigger a script rerun (no spinner, no scroll-jump,
    # no re-render of anything) when you type in a number field, drag a
    # slider, or toggle a checkbox — Streamlit only reruns the app when the
    # form's submit button is pressed. That's the actual fix for "why does
    # it do anything before I click Interrogate": previously every one of
    # these widgets lived directly on the page, and Streamlit reruns the
    # *entire* script on every single widget interaction by design — that
    # was never gated on the button, it just happened to not recompute
    # anything expensive most of the time. Combined with a known Streamlit
    # bug (streamlit/streamlit#5069) where a rerun triggered from inside a
    # non-first st.tabs() tab can reset scroll position, that's what was
    # producing the flash-to-loading / jump-to-"Today's Picks" behavior.
    compare_mode = st.checkbox(
        "Compare two configurations side-by-side",
        key="sim_compare_mode",
        help="Run two independently configured simulations over the same capital and date range, and overlay both portfolio curves.",
    )

    if not compare_mode:
        with st.form("sim_config_form", border=True):
            top_c1, top_c2, top_c3 = st.columns(3)
            with top_c1:
                start_capital = st.number_input(
                    "Starting capital ($)", min_value=1.0, value=10000.0, step=100.0,
                    key="sim_start_capital",
                )
            with top_c2:
                commission_fee = st.number_input(
                    "Commission per trade ($)", min_value=0.0,
                    value=float(st.session_state.get("commission_fee", 0.0)), step=0.5,
                    key="sim_commission_fee",
                )
            with top_c3:
                date_range = st.date_input(
                    "Date range", value=(min_date, max_date),
                    min_value=min_date, max_value=max_date,
                    key="sim_date_range",
                    help="Limited to the last 60 days: yfinance's 5-minute bar "
                         "history (used for precise stop/target sequencing) "
                         "doesn't go back further than that.",
                )
            cost_c1, cost_c2 = st.columns(2)
            with cost_c1:
                slippage_bps = st.number_input(
                    "Est. slippage (bps, round-trip)", min_value=0.0,
                    value=float(st.session_state.get("slippage_bps", 10.0)), step=1.0,
                    key="sim_slippage_bps",
                    help="Estimated spread/market-impact cost per trade, in basis points of position size — "
                         "on top of the flat commission. A flat dollar commission doesn't capture this, and it "
                         "matters most for thinner/less-liquid names. 10 bps (0.10%) is a reasonable starting "
                         "point for liquid large-caps; use more for small-caps.",
                )
            with cost_c2:
                max_deploy_pct = st.slider(
                    "Capital deployed per day (%)", min_value=10, max_value=100,
                    value=int(st.session_state.get("max_deploy_pct", 100)), step=5,
                    key="sim_max_deploy_pct",
                    help="Share of available capital actually put into that day's signals. The remainder is held "
                         "as idle cash (0% return) and carried to the next day, instead of always being 100% "
                         "invested regardless of conviction or how many signals fired.",
                )

            if isinstance(date_range, tuple) and len(date_range) == 2:
                sim_start, sim_end = date_range
            else:
                sim_start, sim_end = min_date, max_date

            sim_signals, use_take_profit, max_positions, use_stop_loss, stop_loss_pct, weight_by_confidence = _config_controls(
                "Configuration", "sim", min_date, max_date, default_signals=["STRONG BUY", "BUY"],
            )

            submitted = st.form_submit_button(
                "🕵️ Interrogate the candles", use_container_width=True,
            )

        # Widget values above are read from session_state and are current as
        # of *this* rerun regardless of whether that rerun was caused by the
        # submit button or something else on the page — but nothing in this
        # block runs expensive work unless `submitted` is True this run.
        user_state.persist(
            commission_fee=commission_fee, slippage_bps=slippage_bps,
            max_deploy_pct=max_deploy_pct,
        )

        if not sim_signals:
            st.info("Select at least one signal to run the simulation.")
            return

        run = _render_precise_fetch_panel(
            pos_signals, sim_signals, sim_start, sim_end,
            start_capital, commission_fee, max_positions, use_take_profit,
            use_stop_loss, stop_loss_pct, weight_by_confidence,
            slippage_bps, max_deploy_pct, submitted,
        )

        if run is None:
            st.info(
                "Set your conditions above, then click \"🕵️ Interrogate the candles\" "
                "to run the simulation. Nothing below will appear until you do."
            )
            return

        # Everything from here down is rendered from the frozen bundle
        # captured at the last button click — NOT from the live widgets
        # above, which may have moved on since then (see the "Showing
        # results from the last run" caption in that case).
        sim_result, stats, trade_log = run["sim_result"], run["stats"], run["trade_log"]
        run_cfg = run["config"]

        if sim_result is None:
            st.warning("No trades matched the selected signals and date range on the last run.")
            return

        with st.expander("🔧 Debug: parameters used in this run", expanded=False):
            st.code(
                f"signals={run_cfg['signals']}\n"
                f"date_range=({run_cfg['sim_start']} -> {run_cfg['sim_end']})\n"
                f"start_capital={run_cfg['start_capital']:,.2f}\n"
                f"commission_fee={run_cfg['commission_fee']:,.2f}\n"
                f"slippage_bps={run_cfg['slippage_bps']:g}\n"
                f"max_deploy_pct={run_cfg['max_deploy_pct']:g}\n"
                f"max_positions={run_cfg['max_positions']}\n"
                f"use_take_profit={run_cfg['use_take_profit']}\n"
                f"use_stop_loss={run_cfg['use_stop_loss']}\n"
                f"stop_loss_pct={run_cfg['stop_loss_pct']:g}\n"
                f"weight_by_confidence={run_cfg['weight_by_confidence']}\n"
                f"stop_loss_uses_real_low={stats.get('stop_loss_uses_real_low')}\n"
                f"n_trades={stats['n_trades']}\n"
                f"n_days={stats['n_days']}\n"
                f"total_fees={stats['total_fees']:,.2f}\n"
                f"total_slippage={stats.get('total_slippage', 0.0):,.2f}",
                language="text",
            )
            st.caption(
                "These reflect the settings as they were when you last clicked "
                "\"🕵️ Interrogate the candles\", not necessarily the widgets above right now — "
                "click it again to pick up any changes."
            )

        _render_stats(stats, key_prefix="sim")
        if run_cfg["use_stop_loss"]:
            if stats.get("stop_loss_uses_real_low"):
                _info_card(
                    f"Stop-loss ({run_cfg['stop_loss_pct']:.1f}%) is checked against each trade's actual intraday low — "
                    "any trade whose low reached the stop level is resolved at exactly -"
                    f"{run_cfg['stop_loss_pct']:.1f}%, even if it recovered by close. When a trade could have hit both "
                    "the stop and the take-profit target on the same day, the stop is assumed to trigger "
                    "first (the conservative read, since exact intraday sequencing isn't available).",
                    muted=True,
                )
            else:
                _info_card(
                    f"⚠️ Stop-loss ({run_cfg['stop_loss_pct']:.1f}%) is applied as an approximation on each trade's "
                    "resolved (close-based) gain, because this deployment's data doesn't carry an "
                    "intraday-low column. This can only cap trades that already closed worse than the "
                    "stop — it can NOT catch a trade that dipped through the stop intraday and recovered "
                    "to close positive, since that never shows up in the close-based gain. In other words, "
                    "this approximation is optimistically biased: a real stop-loss would very likely "
                    "trigger on more trades, and different trades, than this chart shows. Treat the "
                    "stop-loss results here as a soft upper bound, not a realistic simulation.",
                    muted=True,
                )

        show_benchmark = st.checkbox(
            "Show SPY buy & hold benchmark", value=True, key="sim_show_benchmark",
            help="Overlays what the same starting capital would be worth just buy-and-holding SPY over the same date range — fetched live via yfinance.",
        )

        fig = go.Figure()
        _add_drawdown_shading(
            fig, sim_result["prediction_date"], sim_result["portfolio_value"],
        )
        _add_threshold_colored_trace(
            fig, sim_result["prediction_date"], sim_result["portfolio_value"],
            threshold=run_cfg["start_capital"], name="Portfolio Value",
        )
        benchmark_shown = False
        if show_benchmark:
            benchmark_shown = _add_benchmark_trace(fig, sim_result, run_cfg["start_capital"])
        fig.add_hline(
            y=run_cfg["start_capital"], line_dash="dash", line_color="rgba(255,255,255,0.15)",
            annotation_text="Starting capital", annotation_font_size=10,
        )
        fig.update_layout(
            title=f"Cumulative Portfolio Value — Per-Trade Simulation ({', '.join(run_cfg['signals'])})",
            xaxis_title="Date", yaxis_title="Portfolio Value ($)",
            height=380, hovermode="x unified", **LAYOUT,
        )
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig, use_container_width=True)
        if show_benchmark and not benchmark_shown:
            st.caption("⚠️ SPY benchmark unavailable right now (data fetch failed) — showing strategy only.")

        render_labeled_divider("Return Attribution")
        attrib_by = st.radio(
            "Attribute return by", ["Signal type", "Symbol"], horizontal=True, key="sim_attrib_by",
        )
        _render_return_attribution(
            trade_log, "signal" if attrib_by == "Signal type" else "symbol",
            title=f"P&L Contribution by {attrib_by}",
        )

        render_labeled_divider("Trade Log")
        _render_trade_log_download(trade_log, key="sim_trade_log_dl")
        with st.expander(f"View all {len(trade_log)} simulated trades"):
            st.dataframe(trade_log, use_container_width=True, hide_index=True)

        _render_monte_carlo(sim_result, run_cfg["start_capital"], "sim")

        _info_card(
            "Note: results still assume every simulated trade could actually "
            "be filled at the resolved gain, with only a flat estimated "
            "slippage figure (not real per-symbol spread/market-impact "
            "data) and unlimited liquidity — treat this as an "
            "illustration of the model's signal quality under a "
            "reasonable cost assumption, not a guarantee of real-world "
            "tradeable returns.",
            muted=True,
        )

    else:
        with st.form("sim_compare_form", border=True):
            top_c1, top_c2, top_c3 = st.columns(3)
            with top_c1:
                start_capital = st.number_input(
                    "Starting capital ($)", min_value=1.0, value=10000.0, step=100.0,
                    key="sim_start_capital",
                )
            with top_c2:
                commission_fee = st.number_input(
                    "Commission per trade ($)", min_value=0.0,
                    value=float(st.session_state.get("commission_fee", 0.0)), step=0.5,
                    key="sim_commission_fee",
                )
            with top_c3:
                date_range = st.date_input(
                    "Date range", value=(min_date, max_date),
                    min_value=min_date, max_value=max_date,
                    key="sim_date_range",
                    help="Limited to the last 60 days: yfinance's 5-minute bar "
                         "history (used for precise stop/target sequencing) "
                         "doesn't go back further than that.",
                )
            cost_c1, cost_c2 = st.columns(2)
            with cost_c1:
                slippage_bps = st.number_input(
                    "Est. slippage (bps, round-trip)", min_value=0.0,
                    value=float(st.session_state.get("slippage_bps", 10.0)), step=1.0,
                    key="sim_slippage_bps",
                )
            with cost_c2:
                max_deploy_pct = st.slider(
                    "Capital deployed per day (%)", min_value=10, max_value=100,
                    value=int(st.session_state.get("max_deploy_pct", 100)), step=5,
                    key="sim_max_deploy_pct",
                )

            if isinstance(date_range, tuple) and len(date_range) == 2:
                sim_start, sim_end = date_range
            else:
                sim_start, sim_end = min_date, max_date

            col_a, col_b = st.columns(2)
            with col_a:
                signals_a, take_profit_a, max_pos_a, sl_a, sl_pct_a, wbc_a = _config_controls(
                    "Configuration A", "sim_a", min_date, max_date, default_signals=["STRONG BUY"],
                )
            with col_b:
                signals_b, take_profit_b, max_pos_b, sl_b, sl_pct_b, wbc_b = _config_controls(
                    "Configuration B", "sim_b", min_date, max_date, default_signals=["STRONG BUY", "BUY"],
                )

            submitted = st.form_submit_button("▶️ Run comparison", use_container_width=True)

        user_state.persist(
            commission_fee=commission_fee, slippage_bps=slippage_bps,
            max_deploy_pct=max_deploy_pct,
        )

        if not signals_a or not signals_b:
            st.info("Select at least one signal for both configurations to run the comparison.")
            return

        cmp_cache_key = repr((
            tuple(sorted(signals_a)), tuple(sorted(signals_b)), sim_start, sim_end,
            round(start_capital, 2), round(commission_fee, 2), max_pos_a, max_pos_b,
            take_profit_a, take_profit_b, sl_a, sl_b,
            round(sl_pct_a, 2) if sl_a else None, round(sl_pct_b, 2) if sl_b else None,
            wbc_a, wbc_b, round(slippage_bps, 2), round(max_deploy_pct, 2),
        ))
        cached_cmp = st.session_state.get("sim_compare_run_cache")
        if submitted:
            with st.spinner("Running both configurations..."):
                result_a, stats_a, trade_log_a = _simulate(
                    pos_signals, signals_a, sim_start, sim_end,
                    start_capital, commission_fee, max_pos_a, take_profit_a,
                    use_stop_loss=sl_a, stop_loss_pct=sl_pct_a, weight_by_confidence=wbc_a,
                    slippage_bps=slippage_bps, max_deploy_pct=max_deploy_pct,
                )
                result_b, stats_b, trade_log_b = _simulate(
                    pos_signals, signals_b, sim_start, sim_end,
                    start_capital, commission_fee, max_pos_b, take_profit_b,
                    use_stop_loss=sl_b, stop_loss_pct=sl_pct_b, weight_by_confidence=wbc_b,
                    slippage_bps=slippage_bps, max_deploy_pct=max_deploy_pct,
                )
            st.session_state["sim_compare_run_cache"] = {
                "key": cmp_cache_key,
                "result_a": result_a, "stats_a": stats_a, "trade_log_a": trade_log_a,
                "result_b": result_b, "stats_b": stats_b, "trade_log_b": trade_log_b,
            }
            cached_cmp = st.session_state["sim_compare_run_cache"]
        elif cached_cmp is not None and cached_cmp.get("key") != cmp_cache_key:
            st.caption(
                "⏸️ Showing results from the last run — a setting has changed since then. "
                "Click \"▶️ Run comparison\" again to re-run with the current settings."
            )

        if cached_cmp is None:
            st.info(
                "Set your conditions above, then click \"▶️ Run comparison\" to run both "
                "simulations. Nothing below will appear until you do."
            )
            return

        result_a, stats_a, trade_log_a = cached_cmp["result_a"], cached_cmp["stats_a"], cached_cmp["trade_log_a"]
        result_b, stats_b, trade_log_b = cached_cmp["result_b"], cached_cmp["stats_b"], cached_cmp["trade_log_b"]

        if result_a is None or result_b is None:
            st.warning("No trades match one or both configurations over the selected date range.")
            return

        with st.expander("🔧 Debug: parameters used in this run", expanded=False):
            st.code(
                f"shared: start_capital={start_capital:,.2f}, commission_fee={commission_fee:,.2f}, "
                f"slippage_bps={slippage_bps:g}, max_deploy_pct={max_deploy_pct:g}, "
                f"date_range=({sim_start} -> {sim_end})\n"
                f"A: signals={signals_a}, max_positions={max_pos_a}, take_profit={take_profit_a}, "
                f"stop_loss={sl_a} ({sl_pct_a:g}%), weight_by_confidence={wbc_a}, "
                f"n_trades={stats_a['n_trades']}, n_days={stats_a['n_days']}, "
                f"total_fees={stats_a['total_fees']:,.2f}, total_slippage={stats_a.get('total_slippage', 0.0):,.2f}\n"
                f"B: signals={signals_b}, max_positions={max_pos_b}, take_profit={take_profit_b}, "
                f"stop_loss={sl_b} ({sl_pct_b:g}%), weight_by_confidence={wbc_b}, "
                f"n_trades={stats_b['n_trades']}, n_days={stats_b['n_days']}, "
                f"total_fees={stats_b['total_fees']:,.2f}, total_slippage={stats_b.get('total_slippage', 0.0):,.2f}",
                language="text",
            )

        render_labeled_divider("Results")
        stat_a_col, stat_b_col = st.columns(2)
        with stat_a_col:
            st.markdown(f"**Configuration A** — {', '.join(signals_a)}")
            _render_stats(stats_a, key_prefix="sim_a")
        with stat_b_col:
            st.markdown(f"**Configuration B** — {', '.join(signals_b)}")
            _render_stats(stats_b, key_prefix="sim_b")

        show_benchmark_cmp = st.checkbox(
            "Show SPY buy & hold benchmark", value=True, key="sim_cmp_show_benchmark",
            help="Overlays what the same starting capital would be worth just buy-and-holding SPY over the same date range — fetched live via yfinance.",
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result_a["prediction_date"], y=result_a["portfolio_value"],
            mode="lines+markers", name=f"A: {', '.join(signals_a)}",
            line=dict(color=COLORS["primary"], width=2),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=result_b["prediction_date"], y=result_b["portfolio_value"],
            mode="lines+markers", name=f"B: {', '.join(signals_b)}",
            line=dict(color=COLORS["secondary"], width=2),
            marker=dict(size=5),
        ))
        benchmark_shown_cmp = False
        if show_benchmark_cmp:
            # Widest shared window of the two runs, so the benchmark line
            # doesn't stop short of whichever config's curve runs longer.
            longer_result = result_a if len(result_a) >= len(result_b) else result_b
            benchmark_shown_cmp = _add_benchmark_trace(fig, longer_result, start_capital)
        fig.add_hline(
            y=start_capital, line_dash="dash", line_color="rgba(255,255,255,0.15)",
            annotation_text="Starting capital", annotation_font_size=10,
        )
        fig.update_layout(
            title="Cumulative Portfolio Value — A vs B",
            xaxis_title="Date", yaxis_title="Portfolio Value ($)",
            height=420, hovermode="x unified", **LAYOUT,
        )
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig, use_container_width=True)
        if show_benchmark_cmp and not benchmark_shown_cmp:
            st.caption("⚠️ SPY benchmark unavailable right now (data fetch failed) — showing strategies only.")

        render_labeled_divider("Return Attribution")
        attrib_col_a, attrib_col_b = st.columns(2)
        with attrib_col_a:
            _render_return_attribution(trade_log_a, "signal", title="A — P&L by Signal Type")
        with attrib_col_b:
            _render_return_attribution(trade_log_b, "signal", title="B — P&L by Signal Type")

        render_labeled_divider("Trade Logs")
        dl_col_a, dl_col_b = st.columns(2)
        with dl_col_a:
            st.markdown("**Configuration A**")
            _render_trade_log_download(trade_log_a, key="sim_a_trade_log_dl")
        with dl_col_b:
            st.markdown("**Configuration B**")
            _render_trade_log_download(trade_log_b, key="sim_b_trade_log_dl")

        render_labeled_divider("Monte Carlo Resampling")
        _info_card(
            "Resamples each configuration's own day-over-day returns with replacement into alternate "
            "equity-curve paths of the same length — a distribution of plausible outcomes from each "
            "run's own edge and cadence, not a different strategy. Run independently per side below.",
            muted=True,
        )
        mc_col_a, mc_col_b = st.columns(2)
        with mc_col_a:
            _render_monte_carlo(result_a, start_capital, "sim_a", show_divider=False, chart_title="Configuration A")
        with mc_col_b:
            _render_monte_carlo(result_b, start_capital, "sim_b", show_divider=False, chart_title="Configuration B")

        _info_card(
            "Note: both runs share the same starting capital, commission, and "
            "date range so the comparison isolates the effect of signal "
            "selection, position cap, and exit strategy. Same best-case "
            "caveats apply as the single-configuration view above.",
            muted=True,
        )
