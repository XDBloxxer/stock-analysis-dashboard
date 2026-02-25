"""
Daily Winners Tab Module - st.cache_data PERSISTENT CACHE  (v2)

NEW in v2:
  - Cross-date symbol search bar (top of tab, skips date picker entirely)
  - Delta comparisons on all summary metric cards (vs previous date)
  - Indicator Timeline promoted ABOVE snapshot tabs (most useful feature first)
  - Skeleton loading placeholders while data loads
  - Refresh (cyan) / Clear Cache (red danger) button differentiation
  - Cache-clear confirmation guard (shows warning, requires second click)
  - Step-chart replaces misleading 4-bar candlestick for price journey
  - Indicator preset chips (Momentum / Trend / Volume / Custom)
  - Default show_top = 15 instead of 40
  - Consolidated get_supabase_client from db.py

CACHE STRATEGY: Unchanged — @st.cache_data on SERVER PROCESS, not session_state.
All fetching methods are identical to v1.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

from db import get_supabase_client
from chart_utils import CHART_THEME, LAYOUT, AXIS_STYLE, AXIS_STYLE_SM, COLORS

TAB_ID = "daily_winners"

DATE_TABLES = [
    "daily_winners",
    "winners_market_open",
    "winners_market_close",
    "winners_day_prior_open",
    "winners_day_prior_close",
]

TIMEPOINTS = [
    ("T-1 Open",    "day_prior_open"),
    ("T-1 Close",   "day_prior_close"),
    ("Day Open",    "market_open"),
    ("Day Close",   "market_close"),
]

BOOL_FIELDS = {
    "ema20_above_ema50", "ema50_above_ema200", "price_above_ema20",
    "ema10_above_ema20", "sma50_above_sma200",
    "doji", "hammer", "bullish_engulfing", "gap_up", "gap_down",
}

META_FIELDS = {
    "id", "symbol", "exchange", "detection_date", "created_at",
    "updated_at", "name", "description",
}

# Indicator presets for the multiselect chips
INDICATOR_PRESETS = {
    "Momentum":   ["rsi", "stoch_k", "stoch_d", "mom", "w_r"],
    "Trend":      ["macd_value", "macd_diff", "adx", "ema20", "ema50"],
    "Volume":     ["obv", "cmf", "volume", "volume_ratio"],
    "Volatility": ["atr", "bb_width", "bbpower"],
}

_BULLISH_ON_RISE = {
    'rsi', 'rsi_1', 'rsi_2', 'macd_value', 'macd_diff', 'adx',
    'stoch_k', 'stoch_d', 'stoch_k1', 'stoch_d1', 'mom', 'mom_1',
    'roc', 'ao', 'uo', 'tsi', 'cci20', 'obv', 'cmf', 'vpt',
    'aroon_up', 'aroon_indicator', 'vortex_pos', 'close', 'open',
    'volume', 'volume_ratio', 'kst', 'dpo', 'ema5', 'ema10', 'ema20',
    'ema50', 'ema100', 'ema200', 'sma5', 'sma10', 'sma20', 'sma50',
    'sma100', 'sma200', 'vwap', 'adx_plus_di', 'price_change_1d',
    'price_change_2d', 'price_change_3d', 'price_change_5d',
}
_BULLISH_ON_FALL = {
    'w_r', 'aroon_down', 'vortex_neg', 'adx_minus_di', 'price_vs_high_52w',
}
_INDICATOR_THRESHOLDS = {
    'rsi':     {'oversold': 30, 'overbought': 70},
    'rsi_1':   {'oversold': 30, 'overbought': 70},
    'rsi_2':   {'oversold': 30, 'overbought': 70},
    'stoch_k': {'oversold': 20, 'overbought': 80},
    'stoch_d': {'oversold': 20, 'overbought': 80},
    'w_r':     {'oversold': -80, 'overbought': -20},
    'adx':     {'weak': 20, 'strong': 40},
    'cci20':   {'oversold': -100, 'overbought': 100},
    'cmf':     {'negative': 0},
    'macd_diff':  {'zero': 0},
    'macd_value': {'zero': 0},
}

PALETTE = COLORS['series']


# ── Cached DB fetchers (UNCHANGED from v1) ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def _get_all_dates() -> list[str]:
    try:
        client   = get_supabase_client()
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
        client   = get_supabase_client()
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


# ── Cache control (UNCHANGED logic) ───────────────────────────────────────────
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
        client   = get_supabase_client()
        response = (
            client.table("daily_winners")
            .select("detection_date")
            .gt("detection_date", latest_cached)
            .limit(500)
            .execute()
        )
        new_dates = []
        if response.data:
            new_dates = sorted(set(row["detection_date"] for row in response.data), reverse=True)
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


# ── Helper ────────────────────────────────────────────────────────────────────
def _extract_symbol_row(df: pd.DataFrame, symbol: str):
    if df.empty or 'symbol' not in df.columns:
        return None
    d = df.copy()
    d['symbol'] = d['symbol'].str.strip().str.upper()
    m = d[d['symbol'] == symbol]
    return m.iloc[0] if not m.empty else None


def _skeleton_metrics(n: int = 5):
    """Render n skeleton metric placeholders while data loads."""
    cols = st.columns(n)
    for col in cols:
        with col:
            st.markdown(
                '<div class="skeleton" style="height:82px; margin-bottom:4px;"></div>',
                unsafe_allow_html=True
            )


def _render_cache_buttons(tab_id: str):
    """
    Renders Refresh (cyan) + Clear Cache (red danger) buttons.
    Clear cache requires a second click (session_state confirmation guard).
    Returns (refresh_clicked, clear_confirmed).
    """
    confirm_key = f"{tab_id}_confirm_clear"

    col_r, col_c, col_spacer = st.columns([1, 1, 4])
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
            '<div class="cache-warning">⚠️ This will wipe ALL cached data and force a full re-fetch. '
            'Click <strong>Confirm Clear</strong> to proceed, or <strong>Cancel</strong>.</div>',
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


# ── Prediction signal helpers (UNCHANGED) ─────────────────────────────────────
def _prediction_signal(indicator: str, val_first, val_last) -> tuple[str, str]:
    if pd.isna(val_first) or pd.isna(val_last):
        return '—', '—'
    delta = val_last - val_first
    if delta == 0:
        return '➡️ Flat', '➡️'
    rising = delta > 0
    ind    = indicator.lower()
    if ind in _BULLISH_ON_RISE:
        return ('📈 Bullish', '🟢') if rising else ('📉 Bearish', '🔴')
    elif ind in _BULLISH_ON_FALL:
        return ('📈 Bullish', '🟢') if not rising else ('📉 Bearish', '🔴')
    else:
        return ('↑ Rising', '🔵') if rising else ('↓ Falling', '🟠')


def _momentum_label(val_first, val_last, available_vals) -> str:
    if len(available_vals) < 3:
        return '—'
    mid = available_vals[1] if len(available_vals) >= 3 else None
    if mid is None:
        return '—'
    first_half  = mid - val_first
    second_half = val_last - mid
    if first_half > 0 and second_half > first_half:   return '⚡ Accelerating ↑'
    elif first_half > 0 and second_half < 0:           return '🔄 Reversed ↓'
    elif first_half > 0 and 0 <= second_half <= first_half: return '🐢 Slowing ↑'
    elif first_half < 0 and second_half < first_half:  return '⚡ Accelerating ↓'
    elif first_half < 0 and second_half > 0:           return '🔄 Reversed ↑'
    elif first_half < 0 and first_half <= second_half < 0: return '🐢 Slowing ↓'
    return '➡️ Steady'


# ── Price Journey — step chart replaces misleading 4-bar candlestick ──────────
def render_price_journey(symbol: str, open_df, close_df, prior_open_df, prior_close_df):
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
    closes, highs, lows = [], [], []
    for _, row in available:
        c = float(row['close']) if pd.notna(row.get('close')) else None
        h = float(row['high'])  if pd.notna(row.get('high'))  else None
        l = float(row['low'])   if pd.notna(row.get('low'))   else None
        closes.append(c)
        highs.append(h)
        lows.append(l)

    # Color segments by direction
    seg_colors = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        curr = closes[i]
        if prev is None or curr is None:
            seg_colors.append(COLORS['primary'])
        elif curr >= prev:
            seg_colors.append(COLORS['secondary'])
        else:
            seg_colors.append(COLORS['red'])

    fig = go.Figure()

    # High-low range bars
    h_vals = [h for h in highs if h is not None]
    l_vals = [l for l in lows if l is not None]
    if h_vals and l_vals:
        for i, (lbl, h, l) in enumerate(zip(labels, highs, lows)):
            if h is not None and l is not None:
                fig.add_shape(
                    type="line",
                    x0=lbl, x1=lbl, y0=l, y1=h,
                    line=dict(color='rgba(0,212,255,0.25)', width=8),
                    layer='below',
                )

    # Step-connected close line
    fig.add_trace(go.Scatter(
        x=labels, y=closes,
        mode='lines+markers',
        name='Close Price',
        line=dict(color=COLORS['primary'], width=2.5, shape='spline'),
        marker=dict(
            size=11,
            color=closes,
            colorscale=[[0, COLORS['red']], [0.5, COLORS['primary']], [1, COLORS['secondary']]],
            line=dict(color='white', width=1.5),
            showscale=False,
        ),
        hovertemplate='<b>%{x}</b><br>Close: $%{y:.2f}<extra></extra>',
    ))

    # Direction annotations
    for i in range(1, len(closes)):
        if closes[i] is not None and closes[i - 1] is not None:
            delta = closes[i] - closes[i - 1]
            pct   = delta / closes[i - 1] * 100
            color = COLORS['secondary'] if delta >= 0 else COLORS['red']
            sign  = '+' if delta >= 0 else ''
            fig.add_annotation(
                x=labels[i], y=closes[i],
                text=f"{sign}{pct:.2f}%",
                showarrow=False,
                yshift=18,
                font=dict(size=9, color=color, family='JetBrains Mono'),
            )

    first_c = next((c for c in closes if c is not None), None)
    last_c  = next((c for c in reversed(closes) if c is not None), None)
    if first_c and last_c:
        total_pct = (last_c - first_c) / first_c * 100
        title_color = COLORS['secondary'] if total_pct >= 0 else COLORS['red']
        title_suffix = f"  <span style='color:{title_color};font-size:12px'>{'+' if total_pct>=0 else ''}{total_pct:.2f}% T-1 Open→Close</span>"
    else:
        title_suffix = ""

    fig.update_layout(
        title=dict(
            text=f"<b>Price Journey — {symbol}</b>{title_suffix}",
            font=dict(size=14, color='#e2ecf8'),
        ),
        yaxis_title="Price ($)",
        height=360,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        showlegend=False,
        **{k: v for k, v in CHART_THEME.items() if k not in ('xaxis', 'yaxis', 'title_font')},
    )
    fig.update_xaxes(**CHART_THEME['xaxis'])
    fig.update_yaxes(**CHART_THEME['yaxis'])
    st.plotly_chart(fig, use_container_width=True)


# ── Indicator Timeline ─────────────────────────────────────────────────────────
def render_indicator_timeline(symbol: str, open_df, close_df, prior_open_df, prior_close_df):
    symbol = symbol.strip().upper()
    df_map = {
        "day_prior_open":  prior_open_df,
        "day_prior_close": prior_close_df,
        "market_open":     open_df,
        "market_close":    close_df,
    }
    rows = {}
    for label, key in TIMEPOINTS:
        rows[label] = _extract_symbol_row(df_map[key], symbol)

    available_timepoints = [lbl for lbl, _ in TIMEPOINTS if rows[lbl] is not None]
    if len(available_timepoints) < 2:
        st.info("Need at least 2 timepoints with data to display the indicator timeline.")
        return

    # Collect all numeric indicator columns
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

    # Build summary dataframe
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
    summary_df = summary_df[summary_df.notna().sum(axis=1) >= 2]

    first_valid = summary_df.apply(lambda r: r.dropna().iloc[0]  if r.dropna().size else None, axis=1)
    last_valid  = summary_df.apply(lambda r: r.dropna().iloc[-1] if r.dropna().size else None, axis=1)

    summary_df['Δ Abs']  = (last_valid - first_valid).round(4)
    summary_df['Δ %']    = ((last_valid - first_valid) / first_valid.abs().replace(0, float('nan')) * 100).round(2)
    summary_df['Max']    = summary_df[available_timepoints].max(axis=1).round(4)
    summary_df['Min']    = summary_df[available_timepoints].min(axis=1).round(4)
    summary_df['Range']  = (summary_df['Max'] - summary_df['Min']).round(4)

    signals, momentums = [], []
    for ind in summary_df.index:
        fv = first_valid[ind]
        lv = last_valid[ind]
        sig_text, sig_dot = _prediction_signal(ind, fv, lv)
        signals.append(f"{sig_dot} {sig_text}")
        av = [summary_df.loc[ind, tp] for tp in available_timepoints
              if tp in summary_df.columns and pd.notna(summary_df.loc[ind, tp])]
        momentums.append(_momentum_label(fv, lv, av) if len(av) >= 2 else '—')

    summary_df['Signal']   = signals
    summary_df['Momentum'] = momentums

    st.markdown("---")
    st.markdown("### 📉 Indicator Timeline")

    tp_first = available_timepoints[0]
    tp_last  = available_timepoints[-1]
    st.caption(
        f"Δ = **{tp_last}** minus **{tp_first}** — the full observed window. "
        f"Zero extra database calls; uses already-loaded snapshots."
    )

    # ── Preset chips ──────────────────────────────────────────────────────────
    preset_key = f"preset_{symbol}"
    if preset_key not in st.session_state:
        st.session_state[preset_key] = "Momentum"

    st.markdown("**Quick presets:**")
    preset_cols = st.columns(len(INDICATOR_PRESETS) + 1)
    for i, preset_name in enumerate(INDICATOR_PRESETS):
        with preset_cols[i]:
            is_active = st.session_state[preset_key] == preset_name
            label = f"{'✓ ' if is_active else ''}{preset_name}"
            if st.button(label, key=f"preset_btn_{symbol}_{preset_name}", use_container_width=True):
                st.session_state[preset_key] = preset_name
                st.rerun()
    with preset_cols[-1]:
        is_custom = st.session_state[preset_key] == "Custom"
        if st.button(f"{'✓ ' if is_custom else ''}Custom", key=f"preset_btn_{symbol}_custom", use_container_width=True):
            st.session_state[preset_key] = "Custom"
            st.rerun()

    # Determine defaults from preset
    active_preset = st.session_state[preset_key]
    if active_preset in INDICATOR_PRESETS:
        preset_defaults = [c for c in INDICATOR_PRESETS[active_preset] if c in summary_df.index]
        if not preset_defaults:
            preset_defaults = list(summary_df.index[:3])
    else:
        preset_defaults = list(summary_df.index[:3])

    selected_indicators = st.multiselect(
        "Select indicators to chart:",
        options=list(summary_df.index),
        default=preset_defaults,
        key=f"indicator_timeline_select_{symbol}",
        help="Each selected indicator gets its own mini-chart panel."
    )

    if not selected_indicators:
        st.info("Select at least one indicator above to see the chart.")
    else:
        indicator_values = {}
        for ind in selected_indicators:
            vals, lbls = [], []
            for lbl in available_timepoints:
                v = summary_df.loc[ind, lbl] if lbl in summary_df.columns else None
                if pd.notna(v):
                    vals.append(float(v))
                    lbls.append(lbl)
            if vals:
                indicator_values[ind] = (lbls, vals)

        if indicator_values:
            inds  = list(indicator_values.keys())
            n     = len(inds)
            pairs = [(inds[i], inds[i + 1] if i + 1 < n else None) for i in range(0, n, 2)]

            for left_ind, right_ind in pairs:
                col_left, col_right = st.columns(2)
                for col_widget, ind in [(col_left, left_ind), (col_right, right_ind)]:
                    if ind is None:
                        continue
                    lbls, vals = indicator_values[ind]
                    color = PALETTE[inds.index(ind) % len(PALETTE)]

                    delta_val = summary_df.loc[ind, 'Δ %']   if ind in summary_df.index else None
                    delta_abs = summary_df.loc[ind, 'Δ Abs'] if ind in summary_df.index else None
                    delta_str = (
                        f"Δ {delta_abs:+.4f} ({delta_val:+.2f}%)"
                        if pd.notna(delta_val) and pd.notna(delta_abs) else ""
                    )

                    fig = go.Figure()

                    if ind in _INDICATOR_THRESHOLDS:
                        for tname, tval in _INDICATOR_THRESHOLDS[ind].items():
                            fig.add_hline(
                                y=tval,
                                line_dash='dot',
                                line_color='rgba(255,255,255,0.2)',
                                annotation_text=f" {tname}={tval}",
                                annotation_font_color='rgba(255,255,255,0.35)',
                                annotation_font_size=9,
                            )

                    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                    fig.add_trace(go.Scatter(
                        x=lbls, y=vals,
                        mode='lines+markers',
                        name=ind,
                        line=dict(color=color, width=2.5),
                        marker=dict(size=10, color=color, line=dict(color='white', width=1.5)),
                        fill='tozeroy',
                        fillcolor=f'rgba({r},{g},{b},0.07)',
                        hovertemplate=f'<b>{ind}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>',
                    ))

                    title_color = (
                        COLORS['secondary'] if (pd.notna(delta_val) and delta_val > 0)
                        else (COLORS['red'] if (pd.notna(delta_val) and delta_val < 0) else '#e8eaf0')
                    )
                    fig.update_layout(
                        title=dict(
                            text=f"<b>{ind}</b>   <span style='font-size:11px;color:{title_color}'>{delta_str}</span>",
                            font=dict(size=13, color='#e2ecf8'),
                        ),
                        height=250,
                        showlegend=False,
                        hovermode='x unified',
                        margin=dict(t=38, b=28, l=38, r=10),
                        **{k: v for k, v in CHART_THEME.items() if k not in ('xaxis', 'yaxis', 'title_font')},
                    )
                    fig.update_xaxes(**AXIS_STYLE_SM)
                    fig.update_yaxes(**AXIS_STYLE_SM)
                    with col_widget:
                        st.plotly_chart(fig, use_container_width=True)

    # ── Prediction table ───────────────────────────────────────────────────────
    st.markdown("#### 🔮 Indicator Prediction Table")
    st.caption(
        f"**Δ = {tp_last} minus {tp_first}** (T-1 Open to Day Close). "
        f"**Signal** = whether direction is historically bullish/bearish. "
        f"**Momentum** = shape of move across all snapshots."
    )

    col_sort, col_filter, col_top = st.columns([2, 2, 1])
    with col_sort:
        sort_by = st.selectbox(
            "Sort by:",
            ["Δ % (abs)", "Bullish first", "Bearish first", "Δ Abs (abs)", "Range", "Indicator name"],
            key=f"summary_sort_{symbol}",
        )
    with col_filter:
        filter_text = st.text_input(
            "Filter indicators:",
            placeholder="e.g. rsi, macd, stoch…",
            key=f"summary_filter_{symbol}",
        )
    with col_top:
        # Default 15 instead of 40
        show_top = st.number_input(
            "Show top N:", min_value=5, max_value=len(summary_df),
            value=min(15, len(summary_df)),
            key=f"summary_top_{symbol}",
        )

    display_summary = summary_df.copy()
    if filter_text.strip():
        terms = [t.strip().lower() for t in filter_text.split(',') if t.strip()]
        mask  = display_summary.index.to_series().apply(lambda x: any(t in x.lower() for t in terms))
        display_summary = display_summary[mask]

    if sort_by == "Δ % (abs)":
        display_summary = display_summary.reindex(display_summary['Δ %'].abs().sort_values(ascending=False).index)
    elif sort_by == "Bullish first":
        display_summary = display_summary.reindex(
            display_summary['Signal'].apply(lambda s: 0 if 'Bullish' in str(s) else (1 if 'Bearish' in str(s) else 2)).sort_values().index)
    elif sort_by == "Bearish first":
        display_summary = display_summary.reindex(
            display_summary['Signal'].apply(lambda s: 0 if 'Bearish' in str(s) else (1 if 'Bullish' in str(s) else 2)).sort_values().index)
    elif sort_by == "Δ Abs (abs)":
        display_summary = display_summary.reindex(display_summary['Δ Abs'].abs().sort_values(ascending=False).index)
    elif sort_by == "Range":
        display_summary = display_summary.sort_values('Range', ascending=False)
    else:
        display_summary = display_summary.sort_index()

    display_summary = display_summary.head(int(show_top))
    for col in available_timepoints:
        if col in display_summary.columns:
            display_summary[col] = display_summary[col].round(4)

    def _style_signal(val):
        if 'Bullish' in str(val):  return 'color: #10b981; font-weight: 600'
        elif 'Bearish' in str(val): return 'color: #ef4444; font-weight: 600'
        elif 'Rising' in str(val):  return 'color: #3b82f6; font-weight: 600'
        elif 'Falling' in str(val): return 'color: #f59e0b; font-weight: 600'
        return 'color: #b8bcc8'

    def _style_delta_pct(val):
        if pd.isna(val): return ''
        return 'color: #10b981; font-weight: 600' if val > 0 else ('color: #ef4444; font-weight: 600' if val < 0 else '')

    def _style_momentum(val):
        if 'Accel' in str(val) and '↑' in str(val): return 'color: #10b981'
        elif 'Accel' in str(val):                    return 'color: #ef4444'
        elif 'Revers' in str(val):                   return 'color: #f59e0b'
        return 'color: #b8bcc8'

    show_cols = available_timepoints + ['Δ Abs', 'Δ %', 'Signal', 'Momentum', 'Range']
    fmt = {
        **{c: '{:.4f}' for c in available_timepoints},
        'Δ Abs': '{:+.4f}', 'Δ %': '{:+.2f}%', 'Range': '{:.4f}',
    }
    styled = (
        display_summary[show_cols]
        .style
        .format({k: v for k, v in fmt.items() if k in show_cols}, na_rep='—')
        .applymap(_style_signal,    subset=['Signal'])
        .applymap(_style_delta_pct, subset=['Δ %'])
        .applymap(_style_momentum,  subset=['Momentum'])
    )
    st.dataframe(styled, use_container_width=True, height=460)

    if len(summary_df) > int(show_top):
        remaining = len(summary_df) - int(show_top)
        st.caption(f"Showing top {int(show_top)} of {len(summary_df)} indicators. Increase 'Show top N' to see {remaining} more.")

    # ── Top movers bar chart ────────────────────────────────────────────────────
    st.markdown("#### 🏆 Biggest Movers (T-1 Open to Day Close)")
    movers_df = summary_df[['Δ %', 'Δ Abs', 'Signal']].copy()
    movers_df['_abs'] = movers_df['Δ %'].abs()
    movers_df = movers_df[movers_df['_abs'].notna()].sort_values('_abs', ascending=False).head(10)

    if not movers_df.empty:
        bar_colors = []
        for _, r in movers_df.iterrows():
            if 'Bullish' in str(r['Signal']): bar_colors.append(COLORS['secondary'])
            elif 'Bearish' in str(r['Signal']): bar_colors.append(COLORS['red'])
            else: bar_colors.append(COLORS['primary'])

        fig_bar = go.Figure(go.Bar(
            x=movers_df.index.tolist(),
            y=movers_df['Δ %'].tolist(),
            marker_color=bar_colors,
            text=[f"{v:+.2f}%" for v in movers_df['Δ %']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Δ: %{y:+.2f}%<extra></extra>',
        ))
        fig_bar.add_hline(y=0, line_color='rgba(255,255,255,0.12)', line_width=1)
        fig_bar.update_layout(
            xaxis_title="Indicator", yaxis_title="Delta % (T-1 Open to Day Close)",
            height=320, showlegend=False, **LAYOUT,
        )
        fig_bar.update_xaxes(**AXIS_STYLE)
        fig_bar.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_bar, use_container_width=True)

        movers_abs = summary_df[['Δ Abs', 'Δ %', 'Signal']].copy()
        movers_abs['_abs'] = movers_abs['Δ Abs'].abs()
        movers_abs = movers_abs[movers_abs['_abs'].notna()].sort_values('_abs', ascending=False).head(5)

        metric_cols = st.columns(5)
        for col_widget, (idx, row) in zip(metric_cols, movers_abs.iterrows()):
            delta_pct = row['Δ %']
            col_widget.metric(
                label=idx,
                value=f"{row['Δ Abs']:+.4f}",
                delta=f"{delta_pct:+.2f}%" if pd.notna(delta_pct) else None,
            )


# ── Indicator snapshot tabs (UNCHANGED) ───────────────────────────────────────
def render_indicator_snapshot(data_row, title, snapshot_type):
    st.markdown(f"**{title}**")
    price_field = 'open' if snapshot_type in ['market_open', 'day_prior_open'] else 'close'
    price_label = 'Opening Price' if price_field == 'open' else 'Closing Price'

    indicator_groups = {
        "Price & OHLC": {'fields': [price_field, 'volume'], 'labels': {price_field: price_label, 'volume': 'Volume'}},
        "Momentum": {
            'fields': ["rsi","rsi_1","rsi_2","stoch_k","stoch_d","stoch_k1","stoch_d1","mom","mom_1","roc","w_r","ao","uo","tsi","kama","cci20","dpo","kst","kst_signal","vortex_pos","vortex_neg","mass_index"],
            'labels': {"rsi_1":"RSI [1]","rsi_2":"RSI [2]","stoch_k":"Stoch K","stoch_d":"Stoch D","stoch_k1":"Stoch K [1]","stoch_d1":"Stoch D [1]","mom_1":"Mom [1]","w_r":"W %R","ao":"AO","uo":"UO","cci20":"CCI 20","dpo":"DPO","kst":"KST","kst_signal":"KST Signal","vortex_pos":"Vortex +","vortex_neg":"Vortex −","mass_index":"Mass Index"},
        },
        "Trend": {
            'fields': ["macd_value","macd_signal","macd_diff","adx","adx_plus_di","adx_minus_di","aroon_up","aroon_down","aroon_indicator","psar","psar_up","psar_down","ema5","ema10","ema20","ema50","ema100","ema200","sma5","sma10","sma20","sma50","sma100","sma200","vwap"],
            'labels': {"macd_value":"MACD","macd_signal":"MACD Signal","macd_diff":"MACD Diff","adx_plus_di":"ADX +DI","adx_minus_di":"ADX −DI","aroon_indicator":"Aroon Ind.","psar_up":"PSAR Up","psar_down":"PSAR Down"},
        },
        "Volatility": {
            'fields': ["atr","atr_pct","bb_upper","bb_lower","bb_middle","bb_width","bbpower","keltner_upper","keltner_lower","keltner_middle","donchian_upper","donchian_lower","donchian_middle","volatility_10d","volatility_20d","volatility_30d"],
            'labels': {"atr_pct":"ATR %","bb_upper":"BB Upper","bb_lower":"BB Lower","bb_middle":"BB Middle","bb_width":"BB Width","bbpower":"BB Power","keltner_upper":"Keltner Upper","keltner_lower":"Keltner Lower","keltner_middle":"Keltner Mid","donchian_upper":"Donchian Upper","donchian_lower":"Donchian Lower","donchian_middle":"Donchian Mid","volatility_10d":"Volatility 10d","volatility_20d":"Volatility 20d","volatility_30d":"Volatility 30d"},
        },
        "Volume": {
            'fields': ["obv","cmf","force_index","eom","eom_signal","vpt","nvi","volume_sma5","volume_sma10","volume_sma20","volume_ratio"],
            'labels': {"obv":"OBV","cmf":"CMF","force_index":"Force Index","eom":"EOM","eom_signal":"EOM Signal","vpt":"VPT","nvi":"NVI","volume_sma5":"Vol SMA 5","volume_sma10":"Vol SMA 10","volume_sma20":"Vol SMA 20","volume_ratio":"Vol Ratio"},
        },
        "Price Context": {
            'fields': ["price_change_1d","price_change_2d","price_change_3d","price_change_5d","price_change_10d","price_change_20d","price_change_30d","high_52w","low_52w","price_vs_high_52w","price_vs_low_52w","gap_pct"],
            'labels': {"price_change_1d":"Change 1d %","price_change_2d":"Change 2d %","price_change_3d":"Change 3d %","price_change_5d":"Change 5d %","price_change_10d":"Change 10d %","price_change_20d":"Change 20d %","price_change_30d":"Change 30d %","high_52w":"52W High","low_52w":"52W Low","price_vs_high_52w":"vs 52W High %","price_vs_low_52w":"vs 52W Low %","gap_pct":"Gap %"},
        },
        "Signals": {
            'fields': ["ema20_above_ema50","ema50_above_ema200","price_above_ema20","ema10_above_ema20","sma50_above_sma200","doji","hammer","bullish_engulfing","gap_up","gap_down"],
            'labels': {"ema20_above_ema50":"EMA20 > EMA50","ema50_above_ema200":"EMA50 > EMA200","price_above_ema20":"Price > EMA20","ema10_above_ema20":"EMA10 > EMA20","sma50_above_sma200":"SMA50 > SMA200","doji":"Doji","hammer":"Hammer","bullish_engulfing":"Bull Engulf.","gap_up":"Gap Up","gap_down":"Gap Down"},
        },
    }
    bool_fields = {"ema20_above_ema50","ema50_above_ema200","price_above_ema20","ema10_above_ema20","sma50_above_sma200","doji","hammer","bullish_engulfing","gap_up","gap_down"}

    tabs = st.tabs(list(indicator_groups.keys()))
    for i, (group_name, group_info) in enumerate(indicator_groups.items()):
        with tabs[i]:
            available = [f for f in group_info['fields'] if f in data_row.index and (f in bool_fields or pd.notna(data_row[f]))]
            if not available:
                st.info(f"No {group_name} indicators available")
                continue
            cols = st.columns(min(4, len(available)))
            for j, field in enumerate(available):
                with cols[j % 4]:
                    value = data_row[field]
                    label = group_info['labels'].get(field, field.replace(".", " ").replace("_", " ").upper())
                    if field in bool_fields:
                        display_val = "✅ Yes" if value else "❌ No"
                    elif field == "volume":
                        display_val = f"{value/1e6:.1f}M" if value > 1_000_000 else f"{value/1e3:.1f}K"
                    elif field in ("volume_sma5","volume_sma10","volume_sma20","obv","force_index","vpt","nvi","eom","eom_signal"):
                        if abs(value) >= 1_000_000: display_val = f"{value/1e6:.2f}M"
                        elif abs(value) >= 1_000:   display_val = f"{value/1e3:.1f}K"
                        else:                        display_val = f"{value:.2f}"
                    elif abs(value) >= 1000: display_val = f"{value:.2f}"
                    elif abs(value) >= 1:    display_val = f"{value:.3f}"
                    else:                    display_val = f"{value:.4f}"
                    st.metric(label, display_val)


# ── Stock history (from ML predictions) — UNCHANGED ───────────────────────────
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

        show_cols = [c for c in ['prediction_date','predicted_signal','predicted_probability','predicted_target_gain','became_winner','actual_gain_pct','actual_high_pct','prediction_correct'] if c in sym_acc.columns]
        fmt = {'predicted_probability':'{:.2%}','predicted_target_gain':'{:.2f}%','actual_gain_pct':'{:.2f}%','actual_high_pct':'{:.2f}%'}
        st.dataframe(
            sym_acc[show_cols].sort_values('prediction_date', ascending=False)
            .style.format({k: v for k, v in fmt.items() if k in show_cols}, na_rep='—'),
            use_container_width=True, height=280,
        )


# ── Cross-date symbol search ───────────────────────────────────────────────────
def render_symbol_search(available_dates: list[str]):
    """
    Global symbol search across all cached dates.
    Shows every date a symbol appeared, with its change %, price, volume.
    Zero extra DB calls — uses already-cached _get_table_for_date().
    """
    st.markdown("### 🔍 Symbol Search  <span style='font-size:0.7rem;color:#3a5070;font-family:JetBrains Mono'>cross-date · no extra DB calls</span>", unsafe_allow_html=True)

    st.markdown('<div class="search-bar">', unsafe_allow_html=True)
    search_sym = st.text_input(
        "Symbol search",
        placeholder="Type a ticker, e.g. AAPL, TSLA, NVDA…",
        key="global_symbol_search",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if not search_sym.strip():
        st.caption("Enter a symbol above to search across all loaded dates without any extra database queries.")
        return

    target = search_sym.strip().upper()
    results = []

    for date in available_dates:
        df = _get_table_for_date("daily_winners", date)
        if df.empty or 'symbol' not in df.columns:
            continue
        df['symbol'] = df['symbol'].str.strip().str.upper()
        match = df[df['symbol'] == target]
        if not match.empty:
            row = match.iloc[0]
            results.append({
                'Date':       date,
                'Price ($)':  row.get('price', None),
                'Change (%)': row.get('change_pct', None),
                'Open ($)':   row.get('open', None),
                'Volume':     row.get('volume', None),
            })

    if not results:
        st.warning(f"**{target}** not found in any loaded date. It may not have been a winner on those days.")
        return

    res_df = pd.DataFrame(results).sort_values('Date', ascending=False)
    n       = len(res_df)
    avg_chg = res_df['Change (%)'].mean()
    best    = res_df['Change (%)'].max()

    st.markdown(f"""
    <div class="search-result-card">
        <span class="ticker">{target}</span>
        &nbsp;
        <span style="color:#6e8aaa;font-size:0.75rem;">appeared on <strong style="color:#e2ecf8">{n}</strong> winner day(s)
        &nbsp;·&nbsp; avg change <strong style="color:{'#00ff88' if avg_chg>0 else '#ff3860'}">{avg_chg:+.2f}%</strong>
        &nbsp;·&nbsp; best <strong style="color:#00ff88">{best:+.2f}%</strong></span>
    </div>
    """, unsafe_allow_html=True)

    # Appearance frequency mini-chart
    if n > 1:
        fig = go.Figure(go.Bar(
            x=res_df['Date'].tolist(),
            y=res_df['Change (%)'].tolist(),
            marker_color=[COLORS['secondary'] if v >= 0 else COLORS['red'] for v in res_df['Change (%)']],
            text=[f"{v:+.2f}%" for v in res_df['Change (%)']],
            textposition='outside',
            hovertemplate='%{x}<br>Change: %{y:+.2f}%<extra></extra>',
        ))
        fig.update_layout(
            height=220, showlegend=False, yaxis_title="Change %",
            margin=dict(t=20, b=30, l=30, r=10), **LAYOUT,
        )
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig, use_container_width=True)

    # Full results table
    st.dataframe(
        res_df.style.format({
            'Price ($)':  '${:.2f}',
            'Open ($)':   '${:.2f}',
            'Change (%)': '{:+.2f}%',
            'Volume':     '{:,.0f}',
        }, na_rep='—')
        .background_gradient(subset=['Change (%)'], cmap='RdYlGn'),
        use_container_width=True,
        hide_index=True,
    )


# ── Main entry point ───────────────────────────────────────────────────────────
def render_daily_winners_tab():

    # ── Cache control buttons ─────────────────────────────────────────────────
    refresh_clicked, clear_confirmed = _render_cache_buttons(TAB_ID)

    if clear_confirmed:
        clear_all_cache()
        st.rerun()
    if refresh_clicked:
        refresh_cache()
        st.rerun()

    # ── Load dates ────────────────────────────────────────────────────────────
    available_dates = _get_all_dates()

    if not available_dates:
        _skeleton_metrics(5)
        st.warning("No daily winners data available yet.")
        st.info("Run `python daily_winners_main.py` to collect data.")
        return

    # ── Symbol search (cross-date, zero egress) ────────────────────────────────
    with st.expander("🔍 Cross-Date Symbol Search", expanded=False):
        render_symbol_search(available_dates)

    st.markdown("---")

    # ── Date selector + day-of-week metric ────────────────────────────────────
    col1, col2 = st.columns([4, 1])
    with col1:
        selected_date = st.selectbox(
            "Select Date:", available_dates,
            format_func=lambda x: datetime.fromisoformat(x).strftime("%A, %B %d, %Y"),
            key="daily_winners_date",
        )
    with col2:
        st.metric("Day of Week", datetime.fromisoformat(selected_date).strftime("%A"))

    # ── Load all 5 tables for selected date ───────────────────────────────────
    winners_df         = _get_table_for_date("daily_winners",           selected_date)
    market_open_df     = _get_table_for_date("winners_market_open",     selected_date)
    market_close_df    = _get_table_for_date("winners_market_close",    selected_date)
    day_prior_open_df  = _get_table_for_date("winners_day_prior_open",  selected_date)
    day_prior_close_df = _get_table_for_date("winners_day_prior_close", selected_date)

    if winners_df.empty:
        _skeleton_metrics(5)
        st.warning(f"No winners data found for {selected_date}.")
        return

    if 'symbol' in winners_df.columns:
        winners_df = winners_df.copy()
        winners_df['symbol'] = winners_df['symbol'].str.strip().str.upper()

    # ── Summary metrics with delta vs previous date ────────────────────────────
    st.subheader(f"Top {len(winners_df)} Winners — {selected_date}")

    # Compute deltas from previous date if available
    prev_winners_df = pd.DataFrame()
    date_idx = available_dates.index(selected_date)
    if date_idx + 1 < len(available_dates):
        prev_date = available_dates[date_idx + 1]
        prev_winners_df = _get_table_for_date("daily_winners", prev_date)

    def _delta(current_val, prev_df, col, fmt=".2f"):
        if prev_df.empty or col not in prev_df.columns:
            return None
        prev_val = prev_df[col].mean() if col in ['change_pct', 'price', 'volume'] else len(prev_df)
        diff = current_val - prev_val
        return f"{diff:+{fmt}}"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Total Winners", len(winners_df),
        delta=_delta(len(winners_df), prev_winners_df, '__count__', '.0f') if not prev_winners_df.empty else None,
    )
    col2.metric(
        "Avg Change", f"{winners_df['change_pct'].mean():.2f}%",
        delta=f"{winners_df['change_pct'].mean() - prev_winners_df['change_pct'].mean():+.2f}% vs prev" if not prev_winners_df.empty else None,
    )
    col3.metric("Best Performer", f"{winners_df['change_pct'].max():.2f}%")
    col4.metric(
        "Avg Price", f"${winners_df['price'].mean():.2f}",
        delta=f"${winners_df['price'].mean() - prev_winners_df['price'].mean():+.2f} vs prev" if not prev_winners_df.empty else None,
    )
    col5.metric(
        "Avg Volume", f"{winners_df['volume'].mean()/1e6:.1f}M",
        delta=f"{(winners_df['volume'].mean() - prev_winners_df['volume'].mean())/1e6:+.1f}M vs prev" if not prev_winners_df.empty else None,
    )

    # ── Winners list ──────────────────────────────────────────────────────────
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
    selected_symbol = st.selectbox("Select a stock to analyse:", symbols, key="daily_winners_symbol")
    if not selected_symbol:
        return

    selected_symbol = selected_symbol.strip().upper()
    winner_info     = winners_df[winners_df['symbol'] == selected_symbol].iloc[0]

    # ── Stock detail metrics — small variant ──────────────────────────────────
    st.markdown('<div class="metrics-sm">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Symbol", selected_symbol)
    col2.metric("Open",   f"${winner_info['open']:.2f}")
    col3.metric("Close",  f"${winner_info['price']:.2f}")
    col4.metric("Change", f"{winner_info['change_pct']:+.2f}%", delta=f"{winner_info['change_pct']:.2f}%")
    col5.metric("Volume", f"{winner_info['volume']/1e6:.1f}M")
    col6.metric("High",   f"${winner_info['high']:.2f}"  if 'high' in winner_info.index and pd.notna(winner_info.get('high'))  else "—")
    col7.metric("Low",    f"${winner_info['low']:.2f}"   if 'low'  in winner_info.index and pd.notna(winner_info.get('low'))   else "—")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Price Journey ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Price Journey")
    st.caption("Close price step-chart — T-1 Open to T-1 Close to Market Open to Market Close. High/low shown as range bars.")
    render_price_journey(
        selected_symbol,
        market_open_df, market_close_df,
        day_prior_open_df, day_prior_close_df,
    )

    # ── Indicator Timeline PROMOTED above snapshot tabs ────────────────────────
    render_indicator_timeline(
        selected_symbol,
        market_open_df, market_close_df,
        day_prior_open_df, day_prior_close_df,
    )

    # ── Technical Indicator Snapshots ─────────────────────────────────────────
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

    # ── ML History ────────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander(f"📊 Full ML History for {selected_symbol}", expanded=False):
        render_stock_history(selected_symbol)
