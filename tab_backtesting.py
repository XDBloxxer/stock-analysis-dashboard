"""
Backtesting Tab Module  (v4)

v4: Added an optional side-by-side comparison mode (run two independently
configured simulations over the same capital/date range and overlay both
portfolio curves), plus a max-drawdown stat on every simulation result.

v3: Replaced the old strategy-creation / GitHub-Actions backtesting system
(which was no longer used) with the Cumulative Gain Simulator, moved here
in its entirety from the Performance Trends sub-tab of ML Predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
    try:
        client    = get_supabase_client()
        page_size = 1000
        offset    = 0
        frames    = []
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
):
    """
    Runs the per-trade cumulative-gain simulation for one configuration.
    Returns (sim_result_df, stats_dict, trade_log_df) or (None, None, None)
    if no trades match.

    use_stop_loss: caps the resolved loss on any trade at -stop_loss_pct.
    NOTE: the underlying `ml_prediction_accuracy` table only carries the
    day's realized/high gain (actual_gain_pct, actual_high_pct) — there is
    no intraday-low column to check whether price actually touched the stop
    before recovering. So this is applied symmetrically to take-profit but
    as an approximation on the *resolved* gain (if the day's final gain
    would have been worse than -stop_loss_pct, the trade is capped there)
    rather than a true bar-by-bar stop simulation. Flagged in the UI.

    weight_by_confidence: position size is proportional to
    predicted_probability among the trades taken that day, instead of an
    equal split.
    """
    sim_df = pos_signals[
        (pos_signals["predicted_signal"].isin(signals))
        & (pos_signals["prediction_date"].dt.date >= sim_start)
        & (pos_signals["prediction_date"].dt.date <= sim_end)
        & (pos_signals["actual_gain_pct"].notna())
    ].copy()

    if sim_df.empty:
        return None, None, None

    if "predicted_probability" in sim_df.columns:
        sim_df = sim_df.sort_values("predicted_probability", ascending=False)

    if use_take_profit and "actual_high_pct" in sim_df.columns and "predicted_target_gain" in sim_df.columns:
        target = pd.to_numeric(sim_df["predicted_target_gain"], errors="coerce")
        high   = pd.to_numeric(sim_df["actual_high_pct"], errors="coerce")
        hit_target = target.notna() & high.notna() & (high >= target)
        sim_df["resolved_gain_pct"] = np.where(hit_target, target, sim_df["actual_gain_pct"])
    else:
        sim_df["resolved_gain_pct"] = sim_df["actual_gain_pct"]

    if use_stop_loss:
        sim_df["resolved_gain_pct"] = np.maximum(sim_df["resolved_gain_pct"], -abs(stop_loss_pct))

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

        day_end_capital = 0.0
        for idx, gain in trades["resolved_gain_pct"].items():
            position_size = capital * weights.loc[idx]
            pnl_dollars = position_size * (gain / 100) - commission_fee
            resolved = max(position_size + pnl_dollars, 0.0)
            day_end_capital += resolved
            trade_rows.append({
                "prediction_date": trade_date,
                "symbol": trades.loc[idx, "symbol"] if "symbol" in trades.columns else None,
                "signal": trades.loc[idx, "predicted_signal"] if "predicted_signal" in trades.columns else None,
                "predicted_probability": trades.loc[idx].get("predicted_probability") if hasattr(trades.loc[idx], "get") else None,
                "position_size": position_size,
                "resolved_gain_pct": gain,
                "pnl_dollars": pnl_dollars,
                "commission_fee": commission_fee,
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
    total_fees   = n_trades * commission_fee
    win_rate     = (sim_df["resolved_gain_pct"] > 0).mean() * 100

    # Sharpe-like ratio on the day-over-day portfolio return series — "-like"
    # because these are simulated per-day compounding steps, not literal
    # daily market returns, so treat it as a rough risk-adjusted-return
    # signal rather than a textbook Sharpe ratio.
    daily_returns = sim_result["portfolio_value"].pct_change().dropna()
    if len(daily_returns) >= 2 and daily_returns.std() > 0:
        sharpe_like = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_like = None

    # Max drawdown: largest peak-to-trough decline in the simulated
    # portfolio value series (computable from day-by-day capital even
    # without intraday low data, since it's about the equity curve itself).
    running_peak = sim_result["portfolio_value"].cummax()
    drawdown_pct = (sim_result["portfolio_value"] - running_peak) / running_peak * 100
    max_drawdown = drawdown_pct.min() if not drawdown_pct.empty else 0.0

    stats = {
        "final_value": final_value,
        "total_return": total_return,
        "n_trades": n_trades,
        "n_days": n_days,
        "total_fees": total_fees,
        "win_rate": win_rate,
        "sharpe_like": sharpe_like,
        "max_drawdown": max_drawdown,
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
    plain st.metric weight."""
    sharpe_display = f"{stats['sharpe_like']:.2f}" if stats["sharpe_like"] is not None else "N/A"

    accent = "green" if stats["total_return"] >= 0 else "red"
    render_hero_metric(
        "Total Return", f"{stats['total_return']:+.2f}%",
        sub=f"${stats['final_value']:,.2f} final value · {stats['n_trades']} trades over {stats['n_days']} day(s)",
        accent=accent,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Fees Paid", f"${stats['total_fees']:,.2f}")
    m2.metric("Trade Win Rate", f"{stats['win_rate']:.1f}%",
              help="% of individual simulated trades with a positive resolved gain.")
    m3.metric("Sharpe-like Ratio (annualized)", sharpe_display,
              help="Mean ÷ std. dev. of day-over-day portfolio returns, annualized by √252. A rough risk-adjusted-return signal, not a textbook Sharpe ratio.")
    m4.metric("Max Drawdown", f"{stats['max_drawdown']:.1f}%",
              help="Largest peak-to-trough decline in the simulated portfolio value over the run.")


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

    pos_signals = all_acc[all_acc["predicted_signal"].isin(_ALL_SIGNALS)].copy()
    pos_signals["prediction_date"] = pd.to_datetime(pos_signals["prediction_date"])

    if pos_signals.empty:
        st.warning("No signal data available yet.")
        return

    # ── Cumulative Gain Simulator ──────────────────────────────────────────
    st.markdown("#### Cumulative Gain Simulator")
    _info_card(
        "Simulates trading each individual signal as its own position — capital "
        "is split equally across every signal that fires on a given day, each "
        "position resolves on its own resolved gain, and commission is charged "
        "per trade. Positions are closed out and capital is pooled back together "
        "at the end of each day before being redistributed the next day. This "
        "avoids inflating results by averaging gains before compounding, but "
        "still simplifies real trading (no slippage, no partial fills, "
        "equal-weight sizing only, and outcomes are based on this system's own "
        "historical gain data, not independently verified fills).",
        muted=True,
    )

    min_date = pos_signals["prediction_date"].min().date()
    max_date = pos_signals["prediction_date"].max().date()

    with st.container(border=True):
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
            user_state.persist(commission_fee=commission_fee)
        with top_c3:
            date_range = st.date_input(
                "Date range", value=(min_date, max_date),
                min_value=min_date, max_value=max_date,
                key="sim_date_range",
            )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        sim_start, sim_end = date_range
    else:
        sim_start, sim_end = min_date, max_date

    compare_mode = st.checkbox(
        "Compare two configurations side-by-side",
        key="sim_compare_mode",
        help="Run two independently configured simulations over the same capital and date range, and overlay both portfolio curves.",
    )

    if not compare_mode:
        with st.container(border=True):
            sim_signals, use_take_profit, max_positions, use_stop_loss, stop_loss_pct, weight_by_confidence = _config_controls(
                "Configuration", "sim", min_date, max_date, default_signals=["STRONG BUY", "BUY"],
            )

        if not sim_signals:
            st.info("Select at least one signal to run the simulation.")
            return

        sim_result, stats, trade_log = _simulate(
            pos_signals, sim_signals, sim_start, sim_end,
            start_capital, commission_fee, max_positions, use_take_profit,
            use_stop_loss=use_stop_loss, stop_loss_pct=stop_loss_pct,
            weight_by_confidence=weight_by_confidence,
        )

        if sim_result is None:
            st.warning("No trades match the selected signals and date range.")
            return

        _render_stats(stats)
        if use_stop_loss:
            _info_card(
                f"Stop-loss ({stop_loss_pct:.1f}%) is applied to each trade's resolved gain, since the "
                "underlying data doesn't carry an intraday-low column to check whether price actually "
                "touched the stop mid-day before recovering. Treat it as a same-day damage cap, not a "
                "true bar-by-bar stop simulation.",
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
            threshold=start_capital, name="Portfolio Value",
        )
        benchmark_shown = False
        if show_benchmark:
            benchmark_shown = _add_benchmark_trace(fig, sim_result, start_capital)
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

        _info_card(
            "Note: results still assume every simulated trade could actually "
            "be filled at the resolved gain, with no slippage, no "
            "market-impact cost, and unlimited liquidity — treat this as a "
            "best-case illustration of the model's signal quality, not a "
            "guarantee of real-world tradeable returns.",
            muted=True,
        )

    else:
        col_a, col_b = st.columns(2)
        with col_a:
            with st.container(border=True):
                signals_a, take_profit_a, max_pos_a, sl_a, sl_pct_a, wbc_a = _config_controls(
                    "Configuration A", "sim_a", min_date, max_date, default_signals=["STRONG BUY"],
                )
        with col_b:
            with st.container(border=True):
                signals_b, take_profit_b, max_pos_b, sl_b, sl_pct_b, wbc_b = _config_controls(
                    "Configuration B", "sim_b", min_date, max_date, default_signals=["STRONG BUY", "BUY"],
                )

        if not signals_a or not signals_b:
            st.info("Select at least one signal for both configurations to run the comparison.")
            return

        result_a, stats_a, trade_log_a = _simulate(
            pos_signals, signals_a, sim_start, sim_end,
            start_capital, commission_fee, max_pos_a, take_profit_a,
            use_stop_loss=sl_a, stop_loss_pct=sl_pct_a, weight_by_confidence=wbc_a,
        )
        result_b, stats_b, trade_log_b = _simulate(
            pos_signals, signals_b, sim_start, sim_end,
            start_capital, commission_fee, max_pos_b, take_profit_b,
            use_stop_loss=sl_b, stop_loss_pct=sl_pct_b, weight_by_confidence=wbc_b,
        )

        if result_a is None or result_b is None:
            st.warning("No trades match one or both configurations over the selected date range.")
            return

        render_labeled_divider("Results")
        stat_a_col, stat_b_col = st.columns(2)
        with stat_a_col:
            st.markdown(f"**Configuration A** — {', '.join(signals_a)}")
            _render_stats(stats_a)
        with stat_b_col:
            st.markdown(f"**Configuration B** — {', '.join(signals_b)}")
            _render_stats(stats_b)

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

        _info_card(
            "Note: both runs share the same starting capital, commission, and "
            "date range so the comparison isolates the effect of signal "
            "selection, position cap, and exit strategy. Same best-case "
            "caveats apply as the single-configuration view above.",
            muted=True,
        )
