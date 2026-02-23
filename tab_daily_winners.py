"""
Daily Winners Tab Module - st.cache_data PERSISTENT CACHE

CACHE STRATEGY:
  - Data is stored in @st.cache_data on the SERVER PROCESS, not in session_state.
  - This means the cache survives browser tab closes, page refreshes, and
    re-opening the page — as long as the Streamlit server process is running.
  - On first load: fetches all dates + table data for every date.
  - Refresh button: fetches only dates newer than the latest cached date,
    fetches table data for those new dates only, then busts the relevant
    cached functions so they re-run with the expanded date list.
  - Clear Cache: calls the per-function .clear() to wipe everything → full
    re-fetch on next render.
  - Switching dates or tabs: zero egress (reads from cache_data).

HOW INVALIDATION WORKS:
  - _get_all_dates() is cached. To force a re-fetch, call _get_all_dates.clear().
  - _get_table_for_date(table, date) is cached per (table, date) pair.
    Once fetched, a specific date is NEVER re-fetched (stable historical data).
  - On Refresh: we fetch new dates, store them via a cache-busting sentinel in
    session_state, then clear only _get_all_dates so it re-runs; individual
    date caches are left untouched.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
from supabase import create_client, Client

TAB_ID = "daily_winners"

CHART_THEME = {
    'plot_bgcolor': 'rgba(26, 29, 41, 0.6)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'font': dict(color='#e8eaf0'),
    'title_font': dict(size=18, color='#ffffff'),
    'xaxis': dict(gridcolor='rgba(255, 255, 255, 0.1)', color='#b8bcc8'),
    'yaxis': dict(gridcolor='rgba(255, 255, 255, 0.1)', color='#b8bcc8'),
}

COLORS = {
    'primary': '#667eea', 'success': '#10b981',
    'danger': '#ef4444',  'warning': '#f59e0b', 'info': '#3b82f6'
}

DATE_TABLES = [
    "daily_winners",
    "winners_market_open",
    "winners_market_close",
    "winners_day_prior_open",
    "winners_day_prior_close",
]

# Timepoint labels in chronological order
TIMEPOINTS = [
    ("T-1 Open",    "day_prior_open"),
    ("T-1 Close",   "day_prior_close"),
    ("Day Open",    "market_open"),
    ("Day Close",   "market_close"),
]

# Columns that are boolean flags — excluded from the indicator timeline
BOOL_FIELDS = {
    "ema20_above_ema50", "ema50_above_ema200", "price_above_ema20",
    "ema10_above_ema20", "sma50_above_sma200",
    "doji", "hammer", "bullish_engulfing", "gap_up", "gap_down",
}

# Columns that are metadata / not indicators
META_FIELDS = {
    "id", "symbol", "exchange", "detection_date", "created_at",
    "updated_at", "name", "description",
}


# ── Supabase client ────────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase_client():
    supabase_url = os.environ.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url")
    supabase_key = os.environ.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("key")
    if not supabase_url or not supabase_key:
        st.error("❌ Supabase credentials not configured!")
        st.stop()
    return create_client(supabase_url, supabase_key)


# ── Cached DB fetchers ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _get_all_dates() -> list[str]:
    try:
        client = get_supabase_client()
        response = client.table("daily_winners").select("detection_date").limit(500).execute()
        if response.data:
            return sorted(set(row["detection_date"] for row in response.data), reverse=True)
        return []
    except Exception as e:
        st.error(f"Error loading dates: {e}")
        return []


@st.cache_data(show_spinner=False)
def _get_table_for_date(table_name: str, date: str) -> pd.DataFrame:
    try:
        client = get_supabase_client()
        response = (
            client.table(table_name)
            .select("*")
            .eq("detection_date", date)
            .limit(200)
            .execute()
        )
        if not response.data:
            return pd.DataFrame()

        df = pd.DataFrame(response.data)
        df = df.dropna(how='all')

        rename_map = {
            'macd.macd':    'macd_value',
            'macd.signal':  'macd_signal',
            'macd_diff':    'macd_diff',
            'stoch.k':      'stoch_k',
            'stoch.d':      'stoch_d',
            'stoch.k[1]':   'stoch_k1',
            'stoch.d[1]':   'stoch_d1',
            'w.r':          'w_r',
            'rsi[1]':       'rsi_1',
            'rsi[2]':       'rsi_2',
            'mom[1]':       'mom_1',
            'adx+di':       'adx_plus_di',
            'adx-di':       'adx_minus_di',
            'bb.upper':     'bb_upper',
            'bb.lower':     'bb_lower',
            'bb.middle':    'bb_middle',
            'gap_%':        'gap_pct',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].str.strip().str.upper()

        return df
    except Exception as e:
        st.warning(f"Could not load {table_name} for {date}: {e}")
        return pd.DataFrame()


# ── Cache control ──────────────────────────────────────────────────────────────

def clear_all_cache():
    _get_all_dates.clear()
    _get_table_for_date.clear()


def refresh_cache():
    existing_dates = _get_all_dates()

    if not existing_dates:
        _get_all_dates.clear()
        return

    latest_cached = max(existing_dates)

    try:
        client = get_supabase_client()
        response = (
            client.table("daily_winners")
            .select("detection_date")
            .gt("detection_date", latest_cached)
            .limit(500)
            .execute()
        )
        new_dates = []
        if response.data:
            new_dates = sorted(
                set(row["detection_date"] for row in response.data), reverse=True
            )
    except Exception as e:
        st.error(f"Error checking for new dates: {e}")
        return

    if not new_dates:
        st.toast("✅ Cache is already up to date — no new dates found.")
        return

    _get_all_dates.clear()

    for date in new_dates:
        for table in DATE_TABLES:
            _get_table_for_date(table, date)

    st.toast(f"✅ Loaded {len(new_dates)} new date(s): {', '.join(new_dates)}")


# ── Helper: extract a symbol's row from a snapshot df ────────────────────────
def _extract_symbol_row(df: pd.DataFrame, symbol: str):
    if df.empty or 'symbol' not in df.columns:
        return None
    d = df.copy()
    d['symbol'] = d['symbol'].str.strip().str.upper()
    m = d[d['symbol'] == symbol]
    return m.iloc[0] if not m.empty else None


# ── Indicator Timeline chart ──────────────────────────────────────────────────
def render_indicator_timeline(symbol: str,
                               open_df: pd.DataFrame,
                               close_df: pd.DataFrame,
                               prior_open_df: pd.DataFrame,
                               prior_close_df: pd.DataFrame):
    """
    Builds a dynamic indicator timeline using ONLY the already-cached dataframes
    (zero extra Supabase calls). Shows indicator values across the 4 timepoints
    plus a change-summary table.
    """
    symbol = symbol.strip().upper()

    # Map timepoint key → dataframe
    df_map = {
        "day_prior_open":  prior_open_df,
        "day_prior_close": prior_close_df,
        "market_open":     open_df,
        "market_close":    close_df,
    }

    # Extract rows for this symbol from each timepoint
    rows = {}
    for label, key in TIMEPOINTS:
        row = _extract_symbol_row(df_map[key], symbol)
        rows[label] = row

    available_timepoints = [lbl for lbl, _ in TIMEPOINTS if rows[lbl] is not None]

    if len(available_timepoints) < 2:
        st.info("Need at least 2 timepoints with data to display the indicator timeline.")
        return

    # Collect all numeric indicator columns available across any timepoint
    all_numeric_cols = set()
    for lbl in available_timepoints:
        row = rows[lbl]
        if row is not None:
            for col in row.index:
                if col in META_FIELDS or col in BOOL_FIELDS:
                    continue
                try:
                    val = row[col]
                    if pd.notna(val) and isinstance(val, (int, float)):
                        all_numeric_cols.add(col)
                except Exception:
                    pass

    if not all_numeric_cols:
        st.info("No numeric indicator data found for this symbol.")
        return

    sorted_cols = sorted(all_numeric_cols)

    # ── Build a summary dataframe: one row per indicator, one col per timepoint
    summary_data = {}
    for lbl in available_timepoints:
        row = rows[lbl]
        col_vals = {}
        for col in sorted_cols:
            if row is not None and col in row.index and pd.notna(row[col]):
                try:
                    col_vals[col] = float(row[col])
                except Exception:
                    col_vals[col] = None
            else:
                col_vals[col] = None
        summary_data[lbl] = col_vals

    summary_df = pd.DataFrame(summary_data, index=sorted_cols)
    # Only keep indicators with at least 2 non-null values (so we can chart them)
    summary_df = summary_df[summary_df.notna().sum(axis=1) >= 2]

    # Compute change metrics
    first_valid = summary_df.apply(lambda r: r.dropna().iloc[0]  if r.dropna().size else None, axis=1)
    last_valid  = summary_df.apply(lambda r: r.dropna().iloc[-1] if r.dropna().size else None, axis=1)

    summary_df['Δ Abs']  = (last_valid - first_valid).round(4)
    summary_df['Δ %']    = ((last_valid - first_valid) / first_valid.abs().replace(0, float('nan')) * 100).round(2)
    summary_df['Max']    = summary_df[available_timepoints].max(axis=1).round(4)
    summary_df['Min']    = summary_df[available_timepoints].min(axis=1).round(4)
    summary_df['Range']  = (summary_df['Max'] - summary_df['Min']).round(4)

    # ── Section header
    st.markdown("---")
    st.markdown("### 📉 Indicator Timeline")
    st.caption("Track how any indicator evolves across T-1 Open → T-1 Close → Day Open → Day Close. Uses only already-loaded data — zero extra database calls.")

    # ── Indicator multi-select
    # Smart defaults: pick a handful of interesting ones if present
    default_candidates = ['rsi', 'macd_value', 'adx', 'volume', 'close', 'stoch_k']
    defaults = [c for c in default_candidates if c in summary_df.index][:4]
    if not defaults:
        defaults = list(summary_df.index[:3])

    selected_indicators = st.multiselect(
        "Select indicators to chart:",
        options=list(summary_df.index),
        default=defaults,
        key=f"indicator_timeline_select_{symbol}",
        help="You can select any combination of numeric indicators from the database."
    )

    if not selected_indicators:
        st.info("Select at least one indicator above to see the chart.")
    else:
        # ── Chart: separate y-axes if scales differ wildly
        # Use a subplot per indicator if scales are very different, else overlay
        indicator_values = {}
        for ind in selected_indicators:
            vals = []
            lbls = []
            for lbl in available_timepoints:
                v = summary_df.loc[ind, lbl] if lbl in summary_df.columns else None
                if pd.notna(v):
                    vals.append(v)
                    lbls.append(lbl)
            if vals:
                indicator_values[ind] = (lbls, vals)

        if indicator_values:
            # Check if we should use dual-axis (scales differ by >10x)
            ranges = []
            for ind, (lbls, vals) in indicator_values.items():
                r = max(vals) - min(vals) if vals else 0
                ranges.append(abs(max(vals)) if max(abs(v) for v in vals) > 0 else 1)

            use_subplots = len(indicator_values) > 1 and (max(ranges) / max(min(ranges), 0.001) > 20)

            PALETTE = ['#667eea', '#10b981', '#f59e0b', '#ef4444', '#3b82f6',
                       '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#84cc16']

            if use_subplots:
                n = len(indicator_values)
                fig = make_subplots(
                    rows=n, cols=1,
                    subplot_titles=list(indicator_values.keys()),
                    vertical_spacing=0.08,
                    shared_xaxes=True,
                )
                for i, (ind, (lbls, vals)) in enumerate(indicator_values.items(), 1):
                    color = PALETTE[(i - 1) % len(PALETTE)]
                    fig.add_trace(
                        go.Scatter(
                            x=lbls, y=vals,
                            mode='lines+markers',
                            name=ind,
                            line=dict(color=color, width=2.5),
                            marker=dict(size=9, color=color,
                                        line=dict(color='white', width=1.5)),
                            hovertemplate=f'<b>{ind}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>',
                        ),
                        row=i, col=1,
                    )
                    fig.update_yaxes(gridcolor='rgba(255,255,255,0.08)',
                                     color='#b8bcc8', row=i, col=1)

                fig.update_layout(
                    height=max(280 * n, 400),
                    showlegend=False,
                    hovermode='x unified',
                    **{k: v for k, v in CHART_THEME.items()
                       if k not in ('xaxis', 'yaxis')},
                )
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.08)', color='#b8bcc8')

            else:
                fig = go.Figure()
                for i, (ind, (lbls, vals)) in enumerate(indicator_values.items()):
                    color = PALETTE[i % len(PALETTE)]
                    fig.add_trace(go.Scatter(
                        x=lbls, y=vals,
                        mode='lines+markers',
                        name=ind,
                        line=dict(color=color, width=2.5),
                        marker=dict(size=9, color=color,
                                    line=dict(color='white', width=1.5)),
                        hovertemplate=f'<b>{ind}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>',
                    ))

                fig.update_layout(
                    height=420,
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                xanchor='right', x=1),
                    **{k: v for k, v in CHART_THEME.items()
                       if k not in ('xaxis', 'yaxis')},
                )
                fig.update_xaxes(**CHART_THEME['xaxis'])
                fig.update_yaxes(**CHART_THEME['yaxis'])

            st.plotly_chart(fig, use_container_width=True)

    # ── Summary / change table ────────────────────────────────────────────────
    st.markdown("#### 📊 Indicator Change Summary")
    st.caption("Shows how every numeric indicator shifted across the 4 timepoints. Sorted by absolute % change by default.")

    col_sort, col_filter, col_top = st.columns([2, 2, 1])
    with col_sort:
        sort_by = st.selectbox(
            "Sort by:",
            ["Δ % (abs)", "Δ Abs (abs)", "Range", "Indicator name"],
            key=f"summary_sort_{symbol}",
        )
    with col_filter:
        filter_text = st.text_input(
            "Filter indicators:",
            placeholder="e.g. rsi, macd, stoch…",
            key=f"summary_filter_{symbol}",
        )
    with col_top:
        show_top = st.number_input("Show top N:", min_value=5, max_value=len(summary_df),
                                   value=min(30, len(summary_df)),
                                   key=f"summary_top_{symbol}")

    display_summary = summary_df.copy()

    if filter_text.strip():
        terms = [t.strip().lower() for t in filter_text.split(',') if t.strip()]
        mask  = display_summary.index.to_series().apply(
            lambda x: any(t in x.lower() for t in terms)
        )
        display_summary = display_summary[mask]

    if sort_by == "Δ % (abs)":
        display_summary = display_summary.reindex(
            display_summary['Δ %'].abs().sort_values(ascending=False).index
        )
    elif sort_by == "Δ Abs (abs)":
        display_summary = display_summary.reindex(
            display_summary['Δ Abs'].abs().sort_values(ascending=False).index
        )
    elif sort_by == "Range":
        display_summary = display_summary.sort_values('Range', ascending=False)
    else:
        display_summary = display_summary.sort_index()

    display_summary = display_summary.head(int(show_top))

    # Round timepoint columns
    for col in available_timepoints:
        if col in display_summary.columns:
            display_summary[col] = display_summary[col].round(4)

    # Styling: colour Δ% column green/red
    def _style_delta(val):
        if pd.isna(val):
            return ''
        if val > 0:
            intensity = min(int(abs(val) / 5 * 180), 180)
            return f'color: #10b981; font-weight: 600'
        elif val < 0:
            return f'color: #ef4444; font-weight: 600'
        return ''

    def _style_row_bg(row):
        delta = row.get('Δ %', 0)
        if pd.isna(delta) or delta == 0:
            return [''] * len(row)
        if delta > 10:
            bg = 'background-color: rgba(16,185,129,0.07)'
        elif delta < -10:
            bg = 'background-color: rgba(239,68,68,0.07)'
        else:
            bg = ''
        return [bg] * len(row)

    show_cols = available_timepoints + ['Δ Abs', 'Δ %', 'Range']
    styled = (
        display_summary[show_cols]
        .style
        .format({
            **{c: '{:.4f}' for c in available_timepoints},
            'Δ Abs': '{:+.4f}',
            'Δ %':   '{:+.2f}%',
            'Range': '{:.4f}',
        }, na_rep='—')
        .applymap(_style_delta, subset=['Δ %'])
        .apply(_style_row_bg, axis=1)
    )

    st.dataframe(styled, use_container_width=True, height=500)

    # ── Top movers highlight ──────────────────────────────────────────────────
    st.markdown("#### 🏆 Biggest Movers")
    movers_df = summary_df[['Δ %', 'Δ Abs']].copy()
    movers_df['Δ % abs_sort'] = movers_df['Δ %'].abs()
    movers_df = movers_df[movers_df['Δ % abs_sort'].notna()].sort_values('Δ % abs_sort', ascending=False)

    top_movers = movers_df.head(10)

    if not top_movers.empty:
        colors = ['#10b981' if v >= 0 else '#ef4444' for v in top_movers['Δ %']]
        fig_bar = go.Figure(go.Bar(
            x=top_movers.index.tolist(),
            y=top_movers['Δ %'].tolist(),
            marker_color=colors,
            text=[f"{v:+.2f}%" for v in top_movers['Δ %']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Δ: %{y:+.2f}%<extra></extra>',
        ))
        fig_bar.add_hline(y=0, line_color='rgba(255,255,255,0.2)', line_width=1)
        fig_bar.update_layout(
            title=f"Top 10 Indicators by % Change (T-1 Open → Day Close)",
            xaxis_title="Indicator",
            yaxis_title="% Change",
            height=360,
            showlegend=False,
            **{k: v for k, v in CHART_THEME.items() if k not in ('xaxis', 'yaxis')},
        )
        fig_bar.update_xaxes(**CHART_THEME['xaxis'])
        fig_bar.update_yaxes(**CHART_THEME['yaxis'])
        st.plotly_chart(fig_bar, use_container_width=True)

        # Biggest absolute movers
        movers_abs = movers_df.sort_values('Δ Abs', key=lambda s: s.abs(), ascending=False).head(5)
        col1, col2, col3, col4, col5 = st.columns(5)
        for col_widget, (idx, row) in zip([col1, col2, col3, col4, col5], movers_abs.iterrows()):
            delta_pct = row['Δ %']
            col_widget.metric(
                label=idx,
                value=f"{row['Δ Abs']:+.4f}",
                delta=f"{delta_pct:+.2f}%" if pd.notna(delta_pct) else None,
            )


# ── UI helpers ─────────────────────────────────────────────────────────────────
def render_indicator_snapshot(data_row, title, snapshot_type):
    st.markdown(f"**{title}**")

    price_field = 'open' if snapshot_type in ['market_open', 'day_prior_open'] else 'close'
    price_label = 'Opening Price' if price_field == 'open' else 'Closing Price'

    indicator_groups = {
        "Price & OHLC": {
            'fields': [price_field, 'volume'],
            'labels': {price_field: price_label, 'volume': 'Volume'},
        },
        "Momentum": {
            'fields': [
                "rsi", "rsi_1", "rsi_2",
                "stoch_k", "stoch_d", "stoch_k1", "stoch_d1",
                "mom", "mom_1", "roc", "w_r", "ao", "uo",
                "tsi", "kama", "cci20",
                "dpo", "kst", "kst_signal",
                "vortex_pos", "vortex_neg", "mass_index",
            ],
            'labels': {
                "rsi_1": "RSI [1]", "rsi_2": "RSI [2]",
                "stoch_k": "Stoch K", "stoch_d": "Stoch D",
                "stoch_k1": "Stoch K [1]", "stoch_d1": "Stoch D [1]",
                "mom_1": "Mom [1]", "w_r": "W %R", "ao": "AO",
                "uo": "UO", "cci20": "CCI 20", "dpo": "DPO",
                "kst": "KST", "kst_signal": "KST Signal",
                "vortex_pos": "Vortex +", "vortex_neg": "Vortex −",
                "mass_index": "Mass Index",
            },
        },
        "Trend": {
            'fields': [
                "macd_value", "macd_signal", "macd_diff",
                "adx", "adx_plus_di", "adx_minus_di",
                "aroon_up", "aroon_down", "aroon_indicator",
                "psar", "psar_up", "psar_down",
                "ema5", "ema10", "ema20", "ema50", "ema100", "ema200",
                "sma5", "sma10", "sma20", "sma50", "sma100", "sma200",
                "vwap",
            ],
            'labels': {
                "macd_value": "MACD", "macd_signal": "MACD Signal",
                "macd_diff": "MACD Diff",
                "adx_plus_di": "ADX +DI", "adx_minus_di": "ADX −DI",
                "aroon_indicator": "Aroon Ind.",
                "psar_up": "PSAR Up", "psar_down": "PSAR Down",
            },
        },
        "Volatility": {
            'fields': [
                "atr", "atr_pct", "bb_upper", "bb_lower", "bb_middle",
                "bb_width", "bbpower",
                "keltner_upper", "keltner_lower", "keltner_middle",
                "donchian_upper", "donchian_lower", "donchian_middle",
                "volatility_10d", "volatility_20d", "volatility_30d",
            ],
            'labels': {
                "atr_pct": "ATR %",
                "bb_upper": "BB Upper", "bb_lower": "BB Lower",
                "bb_middle": "BB Middle", "bb_width": "BB Width",
                "bbpower": "BB Power",
                "keltner_upper": "Keltner Upper",
                "keltner_lower": "Keltner Lower",
                "keltner_middle": "Keltner Mid",
                "donchian_upper": "Donchian Upper",
                "donchian_lower": "Donchian Lower",
                "donchian_middle": "Donchian Mid",
                "volatility_10d": "Volatility 10d",
                "volatility_20d": "Volatility 20d",
                "volatility_30d": "Volatility 30d",
            },
        },
        "Volume": {
            'fields': [
                "obv", "cmf", "force_index", "eom", "eom_signal",
                "vpt", "nvi",
                "volume_sma5", "volume_sma10", "volume_sma20", "volume_ratio",
            ],
            'labels': {
                "obv": "OBV", "cmf": "CMF", "force_index": "Force Index",
                "eom": "EOM", "eom_signal": "EOM Signal",
                "vpt": "VPT", "nvi": "NVI",
                "volume_sma5": "Vol SMA 5", "volume_sma10": "Vol SMA 10",
                "volume_sma20": "Vol SMA 20", "volume_ratio": "Vol Ratio",
            },
        },
        "Price Context": {
            'fields': [
                "price_change_1d", "price_change_2d", "price_change_3d",
                "price_change_5d", "price_change_10d", "price_change_20d",
                "price_change_30d",
                "high_52w", "low_52w", "price_vs_high_52w", "price_vs_low_52w",
                "gap_pct",
            ],
            'labels': {
                "price_change_1d": "Change 1d %", "price_change_2d": "Change 2d %",
                "price_change_3d": "Change 3d %", "price_change_5d": "Change 5d %",
                "price_change_10d": "Change 10d %", "price_change_20d": "Change 20d %",
                "price_change_30d": "Change 30d %",
                "high_52w": "52W High", "low_52w": "52W Low",
                "price_vs_high_52w": "vs 52W High %", "price_vs_low_52w": "vs 52W Low %",
                "gap_pct": "Gap %",
            },
        },
        "Signals": {
            'fields': [
                "ema20_above_ema50", "ema50_above_ema200", "price_above_ema20",
                "ema10_above_ema20", "sma50_above_sma200",
                "doji", "hammer", "bullish_engulfing",
                "gap_up", "gap_down",
            ],
            'labels': {
                "ema20_above_ema50": "EMA20 > EMA50",
                "ema50_above_ema200": "EMA50 > EMA200",
                "price_above_ema20": "Price > EMA20",
                "ema10_above_ema20": "EMA10 > EMA20",
                "sma50_above_sma200": "SMA50 > SMA200",
                "doji": "Doji", "hammer": "Hammer",
                "bullish_engulfing": "Bull Engulf.",
                "gap_up": "Gap Up", "gap_down": "Gap Down",
            },
        },
    }

    bool_fields = {
        "ema20_above_ema50", "ema50_above_ema200", "price_above_ema20",
        "ema10_above_ema20", "sma50_above_sma200",
        "doji", "hammer", "bullish_engulfing", "gap_up", "gap_down",
    }

    tabs = st.tabs(list(indicator_groups.keys()))
    for i, (group_name, group_info) in enumerate(indicator_groups.items()):
        with tabs[i]:
            available = [
                f for f in group_info['fields']
                if f in data_row.index and (
                    f in bool_fields or pd.notna(data_row[f])
                )
            ]
            if not available:
                st.info(f"No {group_name} indicators available")
                continue

            cols = st.columns(min(4, len(available)))
            for j, field in enumerate(available):
                with cols[j % 4]:
                    value = data_row[field]
                    label = group_info['labels'].get(
                        field, field.replace(".", " ").replace("_", " ").upper()
                    )
                    if field in bool_fields:
                        display_val = "✅ Yes" if value else "❌ No"
                    elif field == "volume":
                        display_val = f"{value/1e6:.1f}M" if value > 1_000_000 else f"{value/1e3:.1f}K"
                    elif field in ("volume_sma5", "volume_sma10", "volume_sma20",
                                   "obv", "force_index", "vpt", "nvi", "eom", "eom_signal"):
                        if abs(value) >= 1_000_000:
                            display_val = f"{value/1e6:.2f}M"
                        elif abs(value) >= 1_000:
                            display_val = f"{value/1e3:.1f}K"
                        else:
                            display_val = f"{value:.2f}"
                    elif abs(value) >= 1000:
                        display_val = f"{value:.2f}"
                    elif abs(value) >= 1:
                        display_val = f"{value:.3f}"
                    else:
                        display_val = f"{value:.4f}"
                    st.metric(label, display_val)


def render_price_journey(symbol, open_df, close_df, prior_open_df, prior_close_df):
    symbol = symbol.strip().upper()

    def _row(df):
        if df.empty or 'symbol' not in df.columns:
            return None
        d = df.copy()
        d['symbol'] = d['symbol'].str.strip().str.upper()
        m = d[d['symbol'] == symbol]
        return m.iloc[0] if not m.empty else None

    snapshots = [
        ("T-1 Open",     _row(prior_open_df)),
        ("T-1 Close",    _row(prior_close_df)),
        ("Market Open",  _row(open_df)),
        ("Market Close", _row(close_df)),
    ]
    available = [(lbl, row) for lbl, row in snapshots if row is not None]

    if len(available) < 2:
        st.info(f"Need at least 2 timepoints for chart. Found {len(available)}.")
        return

    labels = [lbl for lbl, _ in available]
    opens, highs, lows, closes = [], [], [], []
    for lbl, row in available:
        o = float(row['open'])  if pd.notna(row.get('open'))  else None
        c = float(row['close']) if pd.notna(row.get('close')) else None
        h = float(row['high'])  if pd.notna(row.get('high'))  else None
        l = float(row['low'])   if pd.notna(row.get('low'))   else None
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=labels,
        open=opens, high=highs, low=lows, close=closes,
        name="OHLC",
        increasing=dict(line=dict(color='#10b981'), fillcolor='rgba(16,185,129,0.3)'),
        decreasing=dict(line=dict(color='#ef4444'), fillcolor='rgba(239,68,68,0.3)'),
        whiskerwidth=0.5,
    ))

    close_vals = [c for c in closes if c is not None]
    close_lbls = [labels[i] for i, c in enumerate(closes) if c is not None]
    fig.add_trace(go.Scatter(
        x=close_lbls, y=close_vals,
        mode='lines+markers',
        name='Close',
        line=dict(color='#667eea', width=2, dash='dot'),
        marker=dict(size=7, color='#667eea'),
        showlegend=True,
    ))

    fig.update_layout(
        title=f"<b>Price Journey — {symbol}</b>",
        yaxis_title="Price ($)",
        height=420,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        **CHART_THEME,
    )
    fig.update_xaxes(**CHART_THEME['xaxis'])
    fig.update_yaxes(**CHART_THEME['yaxis'])
    st.plotly_chart(fig, use_container_width=True)


def render_stock_history(symbol: str):
    try:
        from tab_ml_predictions import _get_table_full as _ml_get
    except ImportError:
        st.info("ML predictions module not available.")
        return

    preds_df = _ml_get("ml_explosion_predictions")
    acc_df   = _ml_get("ml_prediction_accuracy")

    has_preds = not preds_df.empty and 'symbol' in preds_df.columns
    has_acc   = not acc_df.empty   and 'symbol' in acc_df.columns

    if not has_preds and not has_acc:
        st.info("No ML prediction history available. Visit the ML Predictions tab first to load the data.")
        return

    sym_preds = preds_df[preds_df['symbol'].str.upper() == symbol].copy() if has_preds else pd.DataFrame()
    sym_acc   = acc_df[acc_df['symbol'].str.upper() == symbol].copy()     if has_acc  else pd.DataFrame()

    if sym_preds.empty and sym_acc.empty:
        st.info(f"**{symbol}** has never appeared in ML predictions.")
        return

    total_appearances = len(sym_preds) if not sym_preds.empty else len(sym_acc)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Times Predicted", total_appearances)

    if not sym_acc.empty:
        sym_acc['became_winner']      = sym_acc['became_winner'].astype(bool)
        sym_acc['prediction_correct'] = sym_acc['prediction_correct'].astype(bool)
        sym_acc['actual_gain_pct']    = pd.to_numeric(sym_acc['actual_gain_pct'], errors='coerce')

        overall_win_rate = sym_acc['became_winner'].mean() * 100
        correct_gains    = sym_acc.loc[ sym_acc['became_winner'] & sym_acc['actual_gain_pct'].notna(), 'actual_gain_pct']
        incorrect_gains  = sym_acc.loc[~sym_acc['became_winner'] & sym_acc['actual_gain_pct'].notna(), 'actual_gain_pct']

        col2.metric("Win Rate",        f"{overall_win_rate:.1f}%")
        col3.metric("Avg Gain (Win)",  f"+{correct_gains.mean():.2f}%"  if not correct_gains.empty  else "—")
        col4.metric("Avg Gain (Loss)", f"{incorrect_gains.mean():.2f}%" if not incorrect_gains.empty else "—")

        if 'predicted_signal' in sym_acc.columns:
            st.markdown("##### Win Rate by Signal Type")
            sig_grp = (
                sym_acc.groupby('predicted_signal')
                .agg(
                    times=('became_winner', 'count'),
                    wins=('became_winner', 'sum'),
                    avg_gain=('actual_gain_pct', 'mean'),
                )
                .reset_index()
            )
            sig_grp['win_rate'] = sig_grp['wins'] / sig_grp['times'] * 100

            _sig_colors = {
                "STRONG BUY": "#10b981", "BUY": "#667eea",
                "HOLD": "#f59e0b", "AVOID": "#ef4444",
            }
            fig = go.Figure(go.Bar(
                x=sig_grp['predicted_signal'],
                y=sig_grp['win_rate'],
                text=sig_grp.apply(lambda r: f"{r['win_rate']:.0f}% ({int(r['wins'])}/{int(r['times'])})", axis=1),
                textposition='outside',
                marker_color=[_sig_colors.get(s, '#999') for s in sig_grp['predicted_signal']],
            ))
            fig.update_layout(height=280, yaxis_title="Win Rate %", yaxis_range=[0, 115],
                              showlegend=False, **CHART_THEME)
            fig.update_xaxes(**CHART_THEME['xaxis'])
            fig.update_yaxes(**CHART_THEME['yaxis'])
            st.plotly_chart(fig, use_container_width=True)

        gain_data = sym_acc['actual_gain_pct'].dropna()
        if not gain_data.empty:
            st.markdown("##### Actual Gain Distribution")
            fig = go.Figure()
            w = sym_acc.loc[ sym_acc['became_winner'] & sym_acc['actual_gain_pct'].notna(), 'actual_gain_pct']
            l = sym_acc.loc[~sym_acc['became_winner'] & sym_acc['actual_gain_pct'].notna(), 'actual_gain_pct']
            if not w.empty:
                fig.add_trace(go.Histogram(x=w, name='Winner',     marker_color='#10b981', opacity=0.75, nbinsx=15))
            if not l.empty:
                fig.add_trace(go.Histogram(x=l, name='Non-Winner', marker_color='#ef4444', opacity=0.75, nbinsx=15))
            fig.update_layout(barmode='overlay', height=260,
                              xaxis_title='Actual Gain %', yaxis_title='Count', **CHART_THEME)
            fig.update_xaxes(**CHART_THEME['xaxis'])
            fig.update_yaxes(**CHART_THEME['yaxis'])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Every Prediction & Outcome")
        show_cols = [c for c in [
            'prediction_date', 'predicted_signal', 'predicted_probability',
            'predicted_target_gain', 'became_winner', 'actual_gain_pct',
            'actual_high_pct', 'prediction_correct',
        ] if c in sym_acc.columns]

        _SIG_BG = {
            "STRONG BUY": "#10b98133", "BUY": "#667eea33",
            "HOLD": "#f59e0b33",       "AVOID": "#ef444433",
        }

        def _row_color(row):
            if 'predicted_signal' in row.index:
                bg = _SIG_BG.get(row['predicted_signal'], '')
                return [f'background-color: {bg}'] * len(row)
            return [''] * len(row)

        fmt = {
            'predicted_probability': '{:.2%}',
            'predicted_target_gain': '{:.2f}%',
            'actual_gain_pct':       '{:.2f}%',
            'actual_high_pct':       '{:.2f}%',
        }
        st.dataframe(
            sym_acc[show_cols]
            .sort_values('prediction_date', ascending=False)
            .style.format({k: v for k, v in fmt.items() if k in show_cols}, na_rep='—')
            .apply(_row_color, axis=1),
            use_container_width=True,
            height=300,
        )
    elif not sym_preds.empty:
        st.markdown("##### Prediction History (no accuracy data loaded yet)")
        show_cols = [c for c in [
            'prediction_date', 'signal', 'explosion_probability', 'target_gain_pct',
        ] if c in sym_preds.columns]
        st.dataframe(
            sym_preds[show_cols].sort_values('prediction_date', ascending=False),
            use_container_width=True, height=300,
        )


# ── Main entry point ───────────────────────────────────────────────────────────
def render_daily_winners_tab():

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col2:
        refresh_clicked = st.button("🔄 Refresh Data", use_container_width=True,
                                    key="daily_winners_refresh")
    with col3:
        clear_clicked = st.button("🗑️ Clear Cache", use_container_width=True,
                                  key="daily_winners_clear_cache")

    if clear_clicked:
        clear_all_cache()
        st.rerun()

    if refresh_clicked:
        refresh_cache()
        st.rerun()

    available_dates = _get_all_dates()

    if not available_dates:
        with st.spinner("Loading daily winners data…"):
            available_dates = _get_all_dates()

    if not available_dates:
        st.warning("No daily winners data available yet.")
        st.info("Run `python daily_winners_main.py` to collect data.")
        return

    with col1:
        selected_date = st.selectbox(
            "Select Date:", available_dates,
            format_func=lambda x: datetime.fromisoformat(x).strftime("%A, %B %d, %Y"),
            key="daily_winners_date",
        )
    with col4:
        st.metric("Day of Week", datetime.fromisoformat(selected_date).strftime("%A"))

    winners_df         = _get_table_for_date("daily_winners",           selected_date)
    market_open_df     = _get_table_for_date("winners_market_open",     selected_date)
    market_close_df    = _get_table_for_date("winners_market_close",    selected_date)
    day_prior_open_df  = _get_table_for_date("winners_day_prior_open",  selected_date)
    day_prior_close_df = _get_table_for_date("winners_day_prior_close", selected_date)

    if winners_df.empty:
        st.warning(f"No winners data found for {selected_date}.")
        return

    if 'symbol' in winners_df.columns:
        winners_df = winners_df.copy()
        winners_df['symbol'] = winners_df['symbol'].str.strip().str.upper()

    st.subheader(f"Top {len(winners_df)} Winners — {selected_date}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Winners",  len(winners_df))
    col2.metric("Avg Change",     f"{winners_df['change_pct'].mean():.2f}%")
    col3.metric("Best Performer", f"{winners_df['change_pct'].max():.2f}%")
    col4.metric("Avg Price",      f"${winners_df['price'].mean():.2f}")
    col5.metric("Avg Volume",     f"{winners_df['volume'].mean()/1e6:.1f}M")

    st.markdown("### Winners List")

    display_df = (
        winners_df[['symbol', 'exchange', 'price', 'change_pct', 'volume']]
        .copy()
        .sort_values('change_pct', ascending=False)
        .reset_index(drop=True)
    )
    display_df.index   = display_df.index + 1
    display_df.columns = ['Symbol', 'Exchange', 'Price ($)', 'Change (%)', 'Volume']

    st.dataframe(
        display_df.style
        .format({'Price ($)': '${:.2f}', 'Change (%)': '{:+.2f}%', 'Volume': '{:,.0f}'})
        .background_gradient(subset=['Change (%)'], cmap='PiYG'),
        use_container_width=True, height=400,
    )

    st.markdown("---")
    st.subheader("Detailed Stock Analysis")

    symbols = sorted(winners_df['symbol'].unique())
    selected_symbol = st.selectbox("Select a stock to analyse:", symbols,
                                   key="daily_winners_symbol")

    if not selected_symbol:
        return

    selected_symbol = selected_symbol.strip().upper()
    winner_info = winners_df[winners_df['symbol'] == selected_symbol].iloc[0]

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Symbol", selected_symbol)
    col2.metric("Open",   f"${winner_info['open']:.2f}")
    col3.metric("Close",  f"${winner_info['price']:.2f}")
    col4.metric("Change", f"{winner_info['change_pct']:+.2f}%",
                delta=f"{winner_info['change_pct']:.2f}%")
    col5.metric("Volume", f"{winner_info['volume']/1e6:.1f}M")
    col6.metric("High",   f"${winner_info['high']:.2f}"  if 'high' in winner_info.index and pd.notna(winner_info['high'])  else "—")
    col7.metric("Low",    f"${winner_info['low']:.2f}"   if 'low'  in winner_info.index and pd.notna(winner_info['low'])   else "—")

    st.markdown("---")
    st.markdown("### Price Journey")
    st.caption("Candlestick chart across T-1 Open → T-1 Close → Market Open → Market Close")
    render_price_journey(
        selected_symbol,
        market_open_df, market_close_df,
        day_prior_open_df, day_prior_close_df,
    )

    # ── NEW: Indicator Timeline ───────────────────────────────────────────────
    render_indicator_timeline(
        selected_symbol,
        market_open_df, market_close_df,
        day_prior_open_df, day_prior_close_df,
    )

    st.markdown("---")
    st.markdown("### Technical Indicator Snapshots")

    snapshot_tabs = st.tabs([
        "📈 Day Prior Open (T-1 9:30 AM)",
        "📊 Day Prior Close (T-1 4:00 PM)",
        "🌅 Market Open (9:30 AM)",
        "🌆 Market Close (4:00 PM)",
    ])

    def _show_snapshot(df: pd.DataFrame, label: str, snap_type: str):
        if df.empty or 'symbol' not in df.columns:
            st.warning(f"No {label} data available.")
            return
        df = df.copy()
        df['symbol'] = df['symbol'].str.strip().str.upper()
        rows = df[df['symbol'] == selected_symbol]
        if rows.empty:
            st.warning(f"No {label} data for {selected_symbol}.")
        else:
            render_indicator_snapshot(rows.iloc[0], label, snap_type)

    with snapshot_tabs[0]:
        _show_snapshot(day_prior_open_df,  "Day Prior Open — T-1 9:30 AM",  'day_prior_open')
    with snapshot_tabs[1]:
        _show_snapshot(day_prior_close_df, "Day Prior Close — T-1 4:00 PM", 'day_prior_close')
    with snapshot_tabs[2]:
        _show_snapshot(market_open_df,     "Market Open — 9:30 AM",         'market_open')
    with snapshot_tabs[3]:
        _show_snapshot(market_close_df,    "Market Close — 4:00 PM",        'market_close')

    st.markdown("---")
    with st.expander(f"📊 Full ML History for {selected_symbol}", expanded=False):
        render_stock_history(selected_symbol)
