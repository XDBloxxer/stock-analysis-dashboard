"""
ML Predictions Tab - st.cache_data PERSISTENT CACHE  (v4)

Changes v4:
  - Added TradingView quick-view panel to Latest Predictions sub-tab:
      • Selectbox below the predictions table lets user pick any predicted symbol
      • Live price panel fetches real-time data via yfinance (TTL=60s cache,
        completely separate from the persistent Supabase cache)
      • Shows: live price, day high %, intraday change, prediction price,
        target gain, and a one-click TradingView button that opens in new tab
      • Exchange → TradingView prefix mapping handles NASDAQ / NYSE / AMEX /
        NYSE ARCA / BATS / OTC variants automatically
      • yfinance import is guarded — if not installed the panel shows a warning
        instead of crashing the whole tab

CACHE STRATEGY: UNCHANGED from v3 — all Supabase fetching methods identical.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

from db import get_supabase_client
from chart_utils import CHART_THEME, LAYOUT, AXIS_STYLE, COLORS, SIGNAL_COLORS, SIGNAL_BG, CONFUSION_COLORS

TAB_ID = "ml_predictions"

_DATE_COL = {
    "ml_explosion_predictions": "prediction_date",
    "ml_prediction_accuracy":   "prediction_date",
    "ml_missed_opportunities":  "detection_date",
    "ml_screening_logs":        "screening_date",
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
}

_ALL_TABLES = list(_DATE_COL.keys())

# ── TradingView exchange prefix mapping ───────────────────────────────────────
# Maps the exchange values stored in your DB to the TradingView symbol prefix.
# TradingView URLs look like: https://www.tradingview.com/chart/?symbol=NASDAQ:AAPL
_TV_EXCHANGE_MAP = {
    "NASDAQ":    "NASDAQ",
    "NYSE":      "NYSE",
    "AMEX":      "AMEX",
    "NYSE ARCA": "AMEX",   # TradingView uses AMEX for NYSE Arca ETFs
    "NYSEARCA":  "AMEX",
    "BATS":      "BATS",
    "OTC":       "OTC",
    "OTCMKTS":   "OTC",
    "OTCBB":     "OTC",
}

def _tv_url(symbol: str, exchange: str) -> str:
    """Build a TradingView chart URL for the given symbol + exchange."""
    prefix = _TV_EXCHANGE_MAP.get(str(exchange).strip().upper(), "NASDAQ")
    return f"https://www.tradingview.com/chart/?symbol={prefix}:{symbol.upper()}"


# ── Live price fetch (short TTL — totally separate from Supabase cache) ───────
@st.cache_data(ttl=60, show_spinner=False)
def _get_live_quote(symbol: str) -> dict | None:
    """
    Fetch a lightweight live quote via yfinance.fast_info.
    TTL=60s so data refreshes every minute without hammering Yahoo.
    Returns None (gracefully) if yfinance is not installed or the call fails.
    """
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).fast_info
        return {
            "last_price":   getattr(info, "last_price",   None),
            "day_high":     getattr(info, "day_high",     None),
            "day_low":      getattr(info, "day_low",      None),
            "open":         getattr(info, "open",         None),
            "prev_close":   getattr(info, "previous_close", None),
            "volume":       getattr(info, "last_volume",  None),
        }
    except ImportError:
        return None
    except Exception:
        return None


def _render_tradingview_panel(symbol: str, exchange: str, pred_row: pd.Series):
    """
    Render the live price panel + TradingView button for a selected symbol.
    Sits entirely inside _render_latest_predictions — no cache side-effects.
    """
    tv_url = _tv_url(symbol, exchange)

    # ── Live quote ────────────────────────────────────────────────────────────
    with st.spinner(f"Fetching live quote for {symbol}…"):
        quote = _get_live_quote(symbol)

    st.markdown(
        f'<div style="background:var(--bg-2);border:1px solid var(--cyan-border);'
        f'border-left:3px solid var(--cyan);border-radius:var(--radius);'
        f'padding:16px 20px 14px;margin-bottom:4px;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">'
        f'<div>'
        f'<span class="ticker">{symbol}</span>'
        f'&nbsp;&nbsp;'
        f'<span style="font-family:var(--font-body);font-size:0.62rem;color:var(--text-2);'
        f'letter-spacing:.1em;text-transform:uppercase;">{exchange}</span>'
        f'</div>'
        f'<a href="{tv_url}" target="_blank" style="'
        f'display:inline-flex;align-items:center;gap:6px;'
        f'padding:6px 14px;background:transparent;'
        f'border:1px solid var(--cyan-border);border-radius:var(--radius-sm);'
        f'color:var(--cyan);font-family:var(--font-body);font-size:0.65rem;'
        f'font-weight:500;letter-spacing:.1em;text-transform:uppercase;'
        f'text-decoration:none;transition:background .15s;"'
        f'onmouseover="this.style.background=\'var(--cyan-dim)\'"'
        f'onmouseout="this.style.background=\'transparent\'">'
        f'📈 Open in TradingView</a>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Prediction baseline values
    pred_price      = pred_row.get("current_price")
    target_gain_pct = pred_row.get("target_gain_pct")
    target_price    = pred_row.get("target_price")
    signal          = pred_row.get("signal", "—")
    signal_color    = SIGNAL_COLORS.get(str(signal), COLORS["primary"])

    if quote and quote.get("last_price") is not None:
        live  = quote["last_price"]
        high  = quote["day_high"]
        low   = quote["day_low"]
        prev  = quote["prev_close"]
        vol   = quote["volume"]

        # Day change vs previous close
        day_chg_pct = (live - prev) / prev * 100 if prev else None
        # Change vs prediction price (how much it moved since yesterday's screening)
        vs_pred_pct = (live - pred_price) / pred_price * 100 if pred_price else None
        # How far today's high has gone toward the target
        high_vs_pred = (high - pred_price) / pred_price * 100 if (high and pred_price) else None

        def _color(v):
            if v is None: return "var(--text-2)"
            return "var(--green-bright)" if v >= 0 else "var(--red-bright)"

        def _fmt(v, suffix="%", plus=True):
            if v is None: return "—"
            return f"{'+' if plus and v>=0 else ''}{v:.2f}{suffix}"

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Live Price",     f"${live:.2f}",
                    delta=_fmt(day_chg_pct) if day_chg_pct is not None else None)
        col2.metric("Day High",       f"${high:.2f}" if high else "—",
                    delta=_fmt(high_vs_pred) + " vs pred" if high_vs_pred is not None else None)
        col3.metric("Day Low",        f"${low:.2f}" if low else "—")
        col4.metric("vs Pred Price",  _fmt(vs_pred_pct) if vs_pred_pct is not None else "—",
                    delta=f"pred ${pred_price:.2f}" if pred_price else None)
        col5.metric("Target Gain",    _fmt(target_gain_pct) if target_gain_pct is not None else "—",
                    delta=f"@ ${target_price:.2f}" if target_price else None)

        # Progress bar: how far today's high has travelled toward the target gain
        if high_vs_pred is not None and target_gain_pct and target_gain_pct > 0:
            progress = min(max(high_vs_pred / target_gain_pct, 0.0), 1.0)
            bar_color = "var(--green-bright)" if progress >= 1.0 else "var(--cyan)"
            st.markdown(
                f'<div style="margin-top:10px;">'
                f'<div style="font-family:var(--font-body);font-size:0.58rem;'
                f'color:var(--text-2);letter-spacing:.12em;text-transform:uppercase;'
                f'margin-bottom:5px;">Intraday progress toward target '
                f'<span style="color:{bar_color}">{progress*100:.0f}%</span></div>'
                f'<div style="height:4px;background:var(--bg-4);border-radius:2px;overflow:hidden;">'
                f'<div style="height:100%;width:{progress*100:.1f}%;'
                f'background:{bar_color};border-radius:2px;'
                f'transition:width .3s ease;"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        if vol:
            vol_str = f"{vol/1e6:.2f}M" if vol >= 1_000_000 else f"{vol/1e3:.0f}K"
            st.caption(f"Volume: {vol_str}  ·  Prev close: ${prev:.2f}" if prev else f"Volume: {vol_str}")

    else:
        # yfinance unavailable or call failed — still show prediction data + TV link
        st.markdown(
            '<div style="color:var(--text-2);font-family:var(--font-body);font-size:0.78rem;">'
            '⚠️ Live quote unavailable — install <code>yfinance</code> or check network. '
            'Showing prediction data only.</div>',
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("Signal",       signal)
        col2.metric("Pred Price",   f"${pred_price:.2f}" if pred_price else "—")
        col3.metric("Target Gain",  f"+{target_gain_pct:.2f}%" if target_gain_pct else "—")

    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Live quote cached for 60 s · data via Yahoo Finance · not financial advice")


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
        response = query.limit(500).execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load `{table_name}`: {e}")
        return pd.DataFrame()


# ── Cache control (UNCHANGED) ─────────────────────────────────────────────────
def clear_all_cache():
    _get_table_full.clear()
    _get_live_quote.clear()


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
                color = SIGNAL_COLORS.get(sig, COLORS["primary"])
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

    # ── Filters + table ────────────────────────────────────────────────────────
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

    st.caption(f"{len(fdf)} stocks match current filters")

    if fdf.empty:
        st.warning("No stocks match the filters.")
        return

    fdf["explosion_probability"] = fdf["explosion_probability"] * 100

    def _highlight_sig(row):
        return [f"background-color: {SIGNAL_BG.get(row['signal'], '')}"] * len(row)

    display_cols = [
        c for c in [
            "symbol", "exchange", "signal", "explosion_probability",
            "current_price", "target_price", "target_gain_pct",
            "target_price_low", "target_price_high",
        ] if c in fdf.columns
    ]
    st.dataframe(
        fdf[display_cols].style.format(
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

    # ── TradingView / Live Quote Panel ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Quick View — Live Price & TradingView")
    st.caption(
        "Select any predicted stock to see its live price and open its chart "
        "in TradingView. Live quotes refresh every 60 s."
    )

    # Build symbol list sorted by signal strength then probability
    signal_order = {"STRONG BUY": 0, "BUY": 1, "HOLD": 2, "AVOID": 3}
    sorted_fdf = fdf.copy()
    sorted_fdf["_sig_rank"] = sorted_fdf["signal"].map(signal_order).fillna(9)
    sorted_fdf = sorted_fdf.sort_values(["_sig_rank", "explosion_probability"], ascending=[True, False])
    symbol_options = sorted_fdf["symbol"].tolist()

    if not symbol_options:
        st.info("No symbols to display — adjust filters above.")
        return

    # Format options to show signal + probability inline
    def _sym_label(sym):
        row = sorted_fdf[sorted_fdf["symbol"] == sym].iloc[0]
        sig  = row.get("signal", "")
        prob = row.get("explosion_probability", 0)
        return f"{sym}  ·  {sig}  ·  {prob:.1f}%"

    selected_sym = st.selectbox(
        "Pick a stock:",
        options=symbol_options,
        format_func=_sym_label,
        key=f"{TAB_ID}_tv_symbol",
    )

    if selected_sym:
        pred_row_matches = sorted_fdf[sorted_fdf["symbol"] == selected_sym]
        if not pred_row_matches.empty:
            pred_row = pred_row_matches.iloc[0]
            exchange = pred_row.get("exchange", "NASDAQ")
            _render_tradingview_panel(selected_sym, exchange, pred_row)


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
        col1.metric("Avg Winner Gain",        f"{winner_gains.mean():.2f}%" if not winner_gains.empty else "—")
        col2.metric("Avg Non-Winner Gain",    f"{non_gains.mean():.2f}%"   if not non_gains.empty    else "—")
        col3.metric("Avg Gain Error",         f"{err_df.mean():.2f}%"      if not err_df.empty       else "—")
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
        fig.update_layout(
            xaxis_title="Reason",
            yaxis_title="Count",
            height=300,
            **LAYOUT,
        )
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        gain_data = df["actual_gain_pct"].dropna()
        if not gain_data.empty:
            st.markdown("#### Distribution of Missed Gains")
            fig = go.Figure(go.Histogram(
                x=gain_data, nbinsx=20,
                marker_color=COLORS["amber"], opacity=0.85,
            ))
            fig.update_layout(
                xaxis_title="Actual Gain %",
                yaxis_title="Count",
                height=300,
                **LAYOUT,
            )
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

    all_acc = _get_table_full("ml_prediction_accuracy")
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

    fig = go.Figure()
    for metric, color, name in [
        ("accuracy_pct",  COLORS["primary"],   "Accuracy"),
        ("precision_pct", COLORS["secondary"], "Precision"),
        ("recall_pct",    COLORS["amber"],     "Recall"),
    ]:
        fig.add_trace(go.Scatter(
            x=daily["prediction_date"], y=daily[metric],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2),
            marker=dict(size=5),
        ))
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.12)",
                  annotation_text="50% baseline", annotation_font_size=10)
    fig.update_layout(
        title="Accuracy / Precision / Recall Over Time",
        xaxis_title="Date", yaxis_title="%",
        height=360, hovermode="x unified",
        **LAYOUT,
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily["prediction_date"], y=daily["true_positives"],
            name="True Positives",
            marker_color=COLORS["secondary"], opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            x=daily["prediction_date"],
            y=daily["predicted_pos"] - daily["true_positives"],
            name="False Positives",
            marker_color=COLORS["red"], opacity=0.85,
        ))
        fig.update_layout(
            barmode="stack",
            title="Predicted Positives Breakdown",
            height=300, **LAYOUT,
        )
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
    st.markdown("#### Performance by Signal Type")

    pos_signals = all_acc[all_acc["predicted_signal"].isin(["STRONG BUY", "BUY", "HOLD", "AVOID"])]

    def _sig_agg(gdf):
        total    = len(gdf)
        hits     = int(gdf["became_winner"].sum())
        avg_prob = gdf["predicted_probability"].mean() * 100
        avg_gain = gdf.loc[ gdf["became_winner"] & gdf["actual_gain_pct"].notna(), "actual_gain_pct"].mean()
        avg_loss = gdf.loc[~gdf["became_winner"] & gdf["actual_gain_pct"].notna(), "actual_gain_pct"].mean()
        return pd.Series({
            "total_predictions": total, "became_winner": hits,
            "success_rate_pct":  hits / total * 100 if total else 0,
            "avg_probability":   avg_prob,
            "avg_gain_correct":  avg_gain,
            "avg_gain_wrong":    avg_loss,
        })

    sig_perf = pos_signals.groupby("predicted_signal").apply(_sig_agg).reset_index()

    col_bar, col_tbl = st.columns([2, 3])

    with col_bar:
        fig = go.Figure(go.Bar(
            x=sig_perf["predicted_signal"],
            y=sig_perf["success_rate_pct"],
            text=sig_perf["success_rate_pct"].map(lambda x: f"{x:.1f}%"),
            textposition="outside",
            marker_color=[SIGNAL_COLORS.get(s, "#999") for s in sig_perf["predicted_signal"]],
            opacity=0.9,
        ))
        fig.update_layout(
            title="Success Rate by Signal",
            xaxis_title="Signal", yaxis_title="Success Rate (%)",
            height=300, **LAYOUT,
        )
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig, use_container_width=True)

    with col_tbl:
        st.dataframe(
            sig_perf.style.format(
                {
                    "success_rate_pct": "{:.2f}%",
                    "avg_probability":  "{:.2f}%",
                    "avg_gain_correct": "+{:.2f}%",
                    "avg_gain_wrong":   "{:.2f}%",
                },
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
            height=280,
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
