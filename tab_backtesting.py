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
from chart_utils import LAYOUT, AXIS_STYLE, COLORS
from cache_ui import render_cache_buttons
from dashboard_styles import render_section_header, render_empty_state, render_skeleton_rows, render_labeled_divider

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
):
    """
    Runs the per-trade cumulative-gain simulation for one configuration.
    Returns (sim_result_df, stats_dict) or (None, None) if no trades match.
    """
    sim_df = pos_signals[
        (pos_signals["predicted_signal"].isin(signals))
        & (pos_signals["prediction_date"].dt.date >= sim_start)
        & (pos_signals["prediction_date"].dt.date <= sim_end)
        & (pos_signals["actual_gain_pct"].notna())
    ].copy()

    if sim_df.empty:
        return None, None

    if "predicted_probability" in sim_df.columns:
        sim_df = sim_df.sort_values("predicted_probability", ascending=False)

    if use_take_profit and "actual_high_pct" in sim_df.columns and "predicted_target_gain" in sim_df.columns:
        target = pd.to_numeric(sim_df["predicted_target_gain"], errors="coerce")
        high   = pd.to_numeric(sim_df["actual_high_pct"], errors="coerce")
        hit_target = target.notna() & high.notna() & (high >= target)
        sim_df["resolved_gain_pct"] = np.where(hit_target, target, sim_df["actual_gain_pct"])
    else:
        sim_df["resolved_gain_pct"] = sim_df["actual_gain_pct"]

    records = []
    capital = start_capital
    for trade_date, gdf in sim_df.groupby("prediction_date", sort=True):
        trades = gdf.head(max_positions)
        n = len(trades)
        if n == 0:
            continue
        position_size = capital / n
        day_end_capital = 0.0
        for gain in trades["resolved_gain_pct"]:
            resolved = position_size * (1 + gain / 100) - commission_fee
            day_end_capital += max(resolved, 0.0)
        capital = day_end_capital
        records.append({
            "prediction_date": trade_date,
            "trades_taken": n,
            "portfolio_value": capital,
        })

    if not records:
        return None, None

    sim_result = pd.DataFrame(records).sort_values("prediction_date").reset_index(drop=True)

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
    return sim_result, stats


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
    sharpe_display = f"{stats['sharpe_like']:.2f}" if stats["sharpe_like"] is not None else "N/A"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Portfolio Value", f"${stats['final_value']:,.2f}")
    m2.metric("Total Return", f"{stats['total_return']:+.2f}%")
    m3.metric("Trades Simulated", f"{stats['n_trades']} across {stats['n_days']} day(s)")
    m4.metric("Total Fees Paid", f"${stats['total_fees']:,.2f}")

    m5, m6, m7 = st.columns(3)
    m5.metric("Trade Win Rate", f"{stats['win_rate']:.1f}%",
              help="% of individual simulated trades with a positive resolved gain.")
    m6.metric("Sharpe-like Ratio (annualized)", sharpe_display,
              help="Mean ÷ std. dev. of day-over-day portfolio returns, annualized by √252. A rough risk-adjusted-return signal, not a textbook Sharpe ratio.")
    m7.metric("Max Drawdown", f"{stats['max_drawdown']:.1f}%",
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
        exit_mode = st.selectbox(
            "Exit strategy",
            ["Close at end of day", "Take-profit at target (if hit intraday)"],
            key=f"{key_prefix}_exit_mode",
        )
    max_positions = st.slider(
        "Max concurrent positions per day",
        min_value=1, max_value=25, value=10, key=f"{key_prefix}_max_positions",
    )
    return signals, exit_mode.startswith("Take-profit"), max_positions


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

    _bt_placeholder = st.empty()
    with _bt_placeholder.container():
        render_skeleton_rows(4, height=64)
    all_acc = _get_table_all()
    _bt_placeholder.empty()
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
                "Commission per trade ($)", min_value=0.0, value=0.0, step=0.5,
                key="sim_commission_fee",
            )
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
            sim_signals, use_take_profit, max_positions = _config_controls(
                "Configuration", "sim", min_date, max_date, default_signals=["STRONG BUY", "BUY"],
            )

        if not sim_signals:
            st.info("Select at least one signal to run the simulation.")
            return

        sim_result, stats = _simulate(
            pos_signals, sim_signals, sim_start, sim_end,
            start_capital, commission_fee, max_positions, use_take_profit,
        )

        if sim_result is None:
            st.warning("No trades match the selected signals and date range.")
            return

        _render_stats(stats)

        fig = go.Figure()
        _add_drawdown_shading(
            fig, sim_result["prediction_date"], sim_result["portfolio_value"],
        )
        _add_threshold_colored_trace(
            fig, sim_result["prediction_date"], sim_result["portfolio_value"],
            threshold=start_capital, name="Portfolio Value",
        )
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
                signals_a, take_profit_a, max_pos_a = _config_controls(
                    "Configuration A", "sim_a", min_date, max_date, default_signals=["STRONG BUY"],
                )
        with col_b:
            with st.container(border=True):
                signals_b, take_profit_b, max_pos_b = _config_controls(
                    "Configuration B", "sim_b", min_date, max_date, default_signals=["STRONG BUY", "BUY"],
                )

        if not signals_a or not signals_b:
            st.info("Select at least one signal for both configurations to run the comparison.")
            return

        result_a, stats_a = _simulate(
            pos_signals, signals_a, sim_start, sim_end,
            start_capital, commission_fee, max_pos_a, take_profit_a,
        )
        result_b, stats_b = _simulate(
            pos_signals, signals_b, sim_start, sim_end,
            start_capital, commission_fee, max_pos_b, take_profit_b,
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

        _info_card(
            "Note: both runs share the same starting capital, commission, and "
            "date range so the comparison isolates the effect of signal "
            "selection, position cap, and exit strategy. Same best-case "
            "caveats apply as the single-configuration view above.",
            muted=True,
        )
