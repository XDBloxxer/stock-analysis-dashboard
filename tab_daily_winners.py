"""
Daily Winners Tab Module - APPEND-ONLY SESSION-STATE CACHE

CACHE STRATEGY:
  - All fetched data lives in st.session_state under tab-specific keys.
  - On first load: fetches everything available.
  - Refresh button: fetches only the date list, finds dates newer than the
    latest already cached date, fetches table data for those new dates only,
    then appends them. Existing cached dates are never re-fetched.
  - Clear Cache: wipes all session-state keys for this tab → full re-fetch
    on next render.
  - Switching dates or tabs: zero egress (reads from session state).
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
from supabase import create_client, Client

# ── Constants ──────────────────────────────────────────────────────────────────
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

# Tables that need indicator data for a given date
DATE_TABLES = [
    "daily_winners",
    "winners_market_open",
    "winners_market_close",
    "winners_day_prior_open",
    "winners_day_prior_close",
]


# ── Supabase client ────────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase_client():
    supabase_url = os.environ.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url")
    supabase_key = os.environ.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("key")
    if not supabase_url or not supabase_key:
        st.error("❌ Supabase credentials not configured!")
        st.stop()
    return create_client(supabase_url, supabase_key)


# ── Raw DB fetchers (no caching — session state handles that) ──────────────────
def _fetch_all_dates() -> list[str]:
    """Fetch the full list of available detection_dates."""
    try:
        client = get_supabase_client()
        response = client.table("daily_winners").select("detection_date").limit(500).execute()
        if response.data:
            return sorted(set(row["detection_date"] for row in response.data), reverse=True)
        return []
    except Exception as e:
        st.error(f"Error loading dates: {e}")
        return []


def _fetch_dates_newer_than(after_date: str) -> list[str]:
    """Fetch only detection_dates strictly newer than after_date."""
    try:
        client = get_supabase_client()
        response = (
            client.table("daily_winners")
            .select("detection_date")
            .gt("detection_date", after_date)
            .limit(500)
            .execute()
        )
        if response.data:
            return sorted(set(row["detection_date"] for row in response.data), reverse=True)
        return []
    except Exception as e:
        st.error(f"Error loading new dates: {e}")
        return []


def _fetch_table_for_date(table_name: str, date: str) -> pd.DataFrame:
    """Fetch all rows for a given table + date."""
    try:
        client = get_supabase_client()

        # Always fetch every column — no indicators left behind
        query = client.table(table_name).select("*")

        response = query.eq("detection_date", date).limit(200).execute()

        if not response.data:
            return pd.DataFrame()

        df = pd.DataFrame(response.data)
        df = df.dropna(how='all')  # only drop fully-empty rows; keep all columns

        # Rename dotted / bracketed column names to safe Python identifiers
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


# ── Session-state cache helpers ────────────────────────────────────────────────
def _ss(key: str):
    """Shorthand getter."""
    return st.session_state.get(f"{TAB_ID}__{key}")

def _ss_set(key: str, value):
    st.session_state[f"{TAB_ID}__{key}"] = value

def _ss_has(key: str) -> bool:
    return f"{TAB_ID}__{key}" in st.session_state

def _cached_dates() -> list:
    val = _ss("dates")
    return val if val is not None else []

def _cached_table(table_name: str, date: str) -> pd.DataFrame:
    val = _ss(f"{table_name}__{date}")
    return val if isinstance(val, pd.DataFrame) else pd.DataFrame()

def _store_table(table_name: str, date: str, df: pd.DataFrame):
    _ss_set(f"{table_name}__{date}", df)

def _date_is_cached(date: str) -> bool:
    # A date is fully cached if the primary table has been stored for it
    return _ss_has(f"daily_winners__{date}")

def _clear_tab_cache():
    keys = [k for k in st.session_state if k.startswith(f"{TAB_ID}__")]
    for k in keys:
        del st.session_state[k]


# ── Cache population helpers ───────────────────────────────────────────────────
def _fetch_and_store_date(date: str):
    """Fetch all tables for a single date and store them in session state."""
    for table in DATE_TABLES:
        df = _fetch_table_for_date(table, date)
        _store_table(table, date, df)


def _initialise_cache():
    """
    Called on very first render (nothing cached yet).
    Fetches full date list + all table data for every date.
    """
    all_dates = _fetch_all_dates()
    _ss_set("dates", all_dates)
    for date in all_dates:
        _fetch_and_store_date(date)


def _refresh_cache():
    """
    Called when Refresh button is pressed.
    Fetches only dates newer than the most recent cached date.
    Appends new dates; never re-fetches existing ones.
    """
    existing_dates = _cached_dates()

    if not existing_dates:
        # Nothing cached yet → do a full init instead
        _initialise_cache()
        return

    latest_cached = max(existing_dates)   # e.g. "2025-02-19"
    new_dates = _fetch_dates_newer_than(latest_cached)

    if not new_dates:
        st.toast("✅ Cache is already up to date — no new dates found.")
        return

    for date in new_dates:
        _fetch_and_store_date(date)

    # Merge date lists (deduplicated, sorted descending)
    merged = sorted(set(existing_dates) | set(new_dates), reverse=True)
    _ss_set("dates", merged)
    st.toast(f"✅ Loaded {len(new_dates)} new date(s): {', '.join(new_dates)}")


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
            # Boolean fields: show if column exists even if value is 0
            # Numeric fields: show if column exists and value is not NaN
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
                        # Large absolute numbers
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
    """
    4-candle OHLC chart across the two trading days, with a line
    connecting the close prices to show the price journey.
    """
    symbol = symbol.strip().upper()

    def _row(df):
        if df.empty or 'symbol' not in df.columns:
            return None
        d = df.copy()
        d['symbol'] = d['symbol'].str.strip().str.upper()
        m = d[d['symbol'] == symbol]
        return m.iloc[0] if not m.empty else None

    # Collect the 4 timepoints in chronological order
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
    # For each timepoint use open/close depending on snapshot type;
    # high and low come from the row directly
    opens, highs, lows, closes = [], [], [], []
    for lbl, row in available:
        # open-type snapshots: the "price" is the open field
        # close-type snapshots: the "price" is the close field
        is_open_snap = "Open" in lbl
        o = float(row['open'])  if pd.notna(row.get('open'))  else None
        c = float(row['close']) if pd.notna(row.get('close')) else None
        h = float(row['high'])  if pd.notna(row.get('high'))  else None
        l = float(row['low'])   if pd.notna(row.get('low'))   else None
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)

    fig = go.Figure()

    # Candlestick bars
    fig.add_trace(go.Candlestick(
        x=labels,
        open=opens, high=highs, low=lows, close=closes,
        name="OHLC",
        increasing=dict(line=dict(color='#10b981'), fillcolor='rgba(16,185,129,0.3)'),
        decreasing=dict(line=dict(color='#ef4444'), fillcolor='rgba(239,68,68,0.3)'),
        whiskerwidth=0.5,
    ))

    # Line connecting the close prices
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
    """
    Unified historical view for a symbol, pulling from cached ML prediction
    and accuracy data. Zero extra DB calls — reads from session state.
    """
    # Pull from ml_predictions tab cache (stored under 'ml_predictions__...' keys)
    def _ml_cache(table: str) -> pd.DataFrame:
        key = f"ml_predictions__{table}__*__None__500"
        # Try the exact key pattern used by tab_ml_predictions
        for k, v in st.session_state.items():
            if k.startswith(f"ml_predictions__{table}") and isinstance(v, pd.DataFrame):
                return v
        return pd.DataFrame()

    preds_df = _ml_cache("ml_explosion_predictions")
    acc_df   = _ml_cache("ml_prediction_accuracy")

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

    # ── Appearance count ───────────────────────────────────────────────────
    total_appearances = len(sym_preds) if not sym_preds.empty else len(sym_acc)
    date_col_p = 'prediction_date' if 'prediction_date' in (sym_preds.columns if not sym_preds.empty else []) else None

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
        col3.metric("Avg Gain (Win)",  f"+{correct_gains.mean():.2f}%"   if not correct_gains.empty  else "—")
        col4.metric("Avg Gain (Loss)", f"{incorrect_gains.mean():.2f}%"  if not incorrect_gains.empty else "—")

        # ── Win rate by signal type ────────────────────────────────────────
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
            fig.update_layout(
                height=280, yaxis_title="Win Rate %",
                yaxis_range=[0, 115],
                showlegend=False,
                **CHART_THEME,
            )
            fig.update_xaxes(**CHART_THEME['xaxis'])
            fig.update_yaxes(**CHART_THEME['yaxis'])
            st.plotly_chart(fig, use_container_width=True)

        # ── Gain distribution: wins vs losses ─────────────────────────────
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
            fig.update_layout(
                barmode='overlay', height=260,
                xaxis_title='Actual Gain %', yaxis_title='Count',
                **CHART_THEME,
            )
            fig.update_xaxes(**CHART_THEME['xaxis'])
            fig.update_yaxes(**CHART_THEME['yaxis'])
            st.plotly_chart(fig, use_container_width=True)

        # ── Full prediction history table ──────────────────────────────────
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
        # Accuracy data not loaded yet — show predictions only
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

    # ── Controls ───────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col2:
        refresh_clicked = st.button("🔄 Refresh Data", use_container_width=True,
                                    key="daily_winners_refresh")
    with col3:
        clear_clicked = st.button("🗑️ Clear Cache", use_container_width=True,
                                  key="daily_winners_clear_cache")

    # ── Cache actions ──────────────────────────────────────────────────────
    if clear_clicked:
        _clear_tab_cache()
        st.rerun()

    if refresh_clicked:
        if not _cached_dates():
            _initialise_cache()
        else:
            _refresh_cache()
        st.rerun()

    # ── Initialise on very first render ───────────────────────────────────
    if not _cached_dates():
        with st.spinner("Loading daily winners data for the first time…"):
            _initialise_cache()

    available_dates = _cached_dates()

    if not available_dates:
        st.warning("No daily winners data available yet.")
        st.info("Run `python daily_winners_main.py` to collect data.")
        return

    # ── Date selector ──────────────────────────────────────────────────────
    with col1:
        selected_date = st.selectbox(
            "Select Date:", available_dates,
            format_func=lambda x: datetime.fromisoformat(x).strftime("%A, %B %d, %Y"),
            key="daily_winners_date",
        )
    with col4:
        st.metric("Day of Week", datetime.fromisoformat(selected_date).strftime("%A"))

    # ── Read data from cache (zero egress) ────────────────────────────────
    winners_df        = _cached_table("daily_winners",          selected_date)
    market_open_df    = _cached_table("winners_market_open",    selected_date)
    market_close_df   = _cached_table("winners_market_close",   selected_date)
    day_prior_open_df = _cached_table("winners_day_prior_open",  selected_date)
    day_prior_close_df= _cached_table("winners_day_prior_close", selected_date)

    if winners_df.empty:
        st.warning(f"No winners data found for {selected_date}.")
        return

    if 'symbol' in winners_df.columns:
        winners_df['symbol'] = winners_df['symbol'].str.strip().str.upper()

    # ── Summary metrics ────────────────────────────────────────────────────
    st.subheader(f"Top {len(winners_df)} Winners — {selected_date}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Winners",  len(winners_df))
    col2.metric("Avg Change",     f"{winners_df['change_pct'].mean():.2f}%")
    col3.metric("Best Performer", f"{winners_df['change_pct'].max():.2f}%")
    col4.metric("Avg Price",      f"${winners_df['price'].mean():.2f}")
    col5.metric("Avg Volume",     f"{winners_df['volume'].mean()/1e6:.1f}M")

    # ── Winners list table ─────────────────────────────────────────────────
    st.markdown("### Winners List")

    display_df = (
        winners_df[['symbol', 'exchange', 'price', 'change_pct', 'volume']]
        .copy()
        .sort_values('change_pct', ascending=False)
        .reset_index(drop=True)
    )
    display_df.index  = display_df.index + 1
    display_df.columns = ['Symbol', 'Exchange', 'Price ($)', 'Change (%)', 'Volume']

    st.dataframe(
        display_df.style
        .format({'Price ($)': '${:.2f}', 'Change (%)': '{:+.2f}%', 'Volume': '{:,.0f}'})
        .background_gradient(subset=['Change (%)'], cmap='PiYG'),
        use_container_width=True, height=400,
    )

    st.markdown("---")

    # ── Detailed stock analysis ────────────────────────────────────────────
    st.subheader("Detailed Stock Analysis")

    symbols = sorted(winners_df['symbol'].unique())
    selected_symbol = st.selectbox("Select a stock to analyse:", symbols,
                                   key="daily_winners_symbol")

    if not selected_symbol:
        return

    selected_symbol = selected_symbol.strip().upper()
    winner_info = winners_df[winners_df['symbol'] == selected_symbol].iloc[0]

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Symbol", selected_symbol)
    col2.metric("Close",  f"${winner_info['price']:.2f}")
    col3.metric("Change", f"{winner_info['change_pct']:+.2f}%",
                delta=f"{winner_info['change_pct']:.2f}%")
    col4.metric("Volume", f"{winner_info['volume']/1e6:.1f}M")
    col5.metric("High",   f"${winner_info['high']:.2f}"  if 'high' in winner_info.index and pd.notna(winner_info['high'])  else "—")
    col6.metric("Low",    f"${winner_info['low']:.2f}"   if 'low'  in winner_info.index and pd.notna(winner_info['low'])   else "—")

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
    st.markdown("### Price Journey")
    st.caption("Candlestick chart across T-1 Open → T-1 Close → Market Open → Market Close")
    render_price_journey(
        selected_symbol,
        market_open_df, market_close_df,
        day_prior_open_df, day_prior_close_df,
    )

    st.markdown("---")
    with st.expander(f"📊 Full ML History for {selected_symbol}", expanded=False):
        render_stock_history(selected_symbol)
