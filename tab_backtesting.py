"""
Backtesting Tab Module  (v3)

v3: Replaced the old strategy-creation / GitHub-Actions backtesting system
(which was no longer used) with the Cumulative Gain Simulator, moved here
in its entirety from the Performance Trends sub-tab of ML Predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from db import get_supabase_client
from chart_utils import CHART_THEME, LAYOUT, AXIS_STYLE, COLORS

TAB_ID = "backtesting"

_DATE_COL   = "prediction_date"
_TABLE_NAME = "ml_prediction_accuracy"
_SELECT = (
    "symbol,prediction_date,predicted_probability,predicted_signal,"
    "predicted_target_gain,became_winner,actual_gain_pct,actual_high_pct,"
    "actual_price,prediction_correct,gain_error_pct"
)


# ── Cached DB fetcher (paginates the full table — used for full-history sim) ──
@st.cache_data(show_spinner=False)
def _get_table_all() -> pd.DataFrame:
    try:
        client    = get_supabase_client()
        page_size = 1000
        offset    = 0
        frames    = []
        while True:
            query = (
                client.table(_TABLE_NAME)
                .select(_SELECT)
                .order(_DATE_COL, desc=False)
            )
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
        st.warning(f"Could not load full history for `{_TABLE_NAME}`: {e}")
        return pd.DataFrame()


# ── Cache control ──────────────────────────────────────────────────────────
def clear_all_cache():
    _get_table_all.clear()


def refresh_cache():
    _get_table_all.clear()
    _get_table_all()
    st.toast("✅ Cache refreshed.")


# ── Shared button helper ───────────────────────────────────────────────────
def _render_cache_buttons(tab_id: str):
    confirm_key = f"{tab_id}_confirm_clear"
    col_r, col_c, _ = st.columns([1, 1, 5])
    with col_r:
        st.markdown('<div class="btn-refresh">', unsafe_allow_html=True)
        refresh = st.button("🔄 Refresh", key=f"{tab_id}_refresh_top", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_c:
        st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
        clear = st.button("🗑️ Clear Cache", key=f"{tab_id}_clear_top", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if clear:
        st.session_state[confirm_key] = True

    confirmed = False
    if st.session_state.get(confirm_key):
        st.markdown(
            '<div class="cache-warning">⚠️ This will wipe ALL cached backtesting data. '
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


# ── Main entry point ───────────────────────────────────────────────────────
def render_backtesting_tab():
    st.subheader("Strategy Backtesting")
    st.markdown("Simulate cumulative portfolio performance from the model's historical signals")

    refresh_clicked, clear_confirmed = _render_cache_buttons(TAB_ID)
    if clear_confirmed:
        clear_all_cache()
        st.rerun()
    if refresh_clicked:
        refresh_cache()
        st.rerun()

    all_acc = _get_table_all()
    if all_acc.empty:
        st.warning("No accuracy data available yet.")
        return

    all_acc = all_acc.copy()
    all_acc["became_winner"]         = all_acc["became_winner"].astype(bool)
    all_acc["prediction_correct"]    = all_acc["prediction_correct"].astype(bool)
    all_acc["actual_gain_pct"]       = pd.to_numeric(all_acc["actual_gain_pct"],       errors="coerce")
    all_acc["predicted_probability"] = pd.to_numeric(all_acc["predicted_probability"], errors="coerce")

    pos_signals = all_acc[all_acc["predicted_signal"].isin(["STRONG BUY", "BUY", "HOLD", "AVOID"])].copy()
    pos_signals["prediction_date"] = pd.to_datetime(pos_signals["prediction_date"])

    if pos_signals.empty:
        st.warning("No signal data available yet.")
        return

    # ── Cumulative Gain Simulator ──────────────────────────────────────────
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
