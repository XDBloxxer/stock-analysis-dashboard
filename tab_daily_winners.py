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
from plotly.subplots import make_subplots
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

        if table_name == "daily_winners":
            query = client.table(table_name).select(
                "symbol,exchange,price,change_pct,volume,detection_date"
            )
        elif table_name in [
            "winners_market_open", "winners_market_close",
            "winners_day_prior_open", "winners_day_prior_close",
        ]:
            query = client.table(table_name).select(
                'symbol,exchange,detection_date,snapshot_type,snapshot_time,'
                'open,high,low,close,volume,'
                'rsi,"macd.macd","macd.signal",adx,ema20,sma20,atr,bb_width,'
                '"stoch.k","stoch.d","w.r",ao,cci20'
            )
        else:
            query = client.table(table_name).select("*")

        response = query.eq("detection_date", date).limit(200).execute()

        if not response.data:
            return pd.DataFrame()

        df = pd.DataFrame(response.data)
        df = df.dropna(how='all').dropna(axis=1, how='all')

        rename_map = {
            'macd.macd': 'macd_value', 'macd.signal': 'macd_signal',
            'stoch.k': 'stoch_k', 'stoch.d': 'stoch_d', 'w.r': 'w_r',
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
        "Price & Volume": {
            'fields': [price_field, 'volume'],
            'labels': {price_field: price_label, 'volume': 'Volume'},
        },
        "Momentum": {
            'fields': ["rsi", "rsi[1]", "rsi[2]", "stoch_k", "stoch_d", "mom", "w_r", "roc", "tsi", "kama"],
            'labels': {},
        },
        "Trend": {
            'fields': ["macd_value", "macd_signal", "adx", "adx+di", "adx-di",
                       "ema20", "ema50", "ema200", "sma20", "sma50", "sma200",
                       "aroon_up", "aroon_down", "psar"],
            'labels': {},
        },
        "Volatility": {
            'fields': ["atr", "atr_pct", "bb.upper", "bb.lower", "bb_width",
                       "volatility_20d", "keltner_upper", "keltner_lower",
                       "donchian_upper", "donchian_lower"],
            'labels': {},
        },
        "Volume Indicators": {
            'fields': ["obv", "cmf", "force_index", "vpt", "volume_sma20", "volume_ratio"],
            'labels': {},
        },
        "Other": {
            'fields': ["cci20", "ao", "uo", "vwap", "high_52w", "low_52w", "gap_%"],
            'labels': {},
        },
    }

    tabs = st.tabs(list(indicator_groups.keys()))
    for i, (group_name, group_info) in enumerate(indicator_groups.items()):
        with tabs[i]:
            available = [
                f for f in group_info['fields']
                if f in data_row.index and pd.notna(data_row[f])
            ]
            if not available:
                st.info(f"No {group_name} indicators available")
                continue

            cols = st.columns(min(4, len(available)))
            for j, field in enumerate(available):
                with cols[j % 4]:
                    value = data_row[field]
                    if field == "volume":
                        display_val = f"{value/1e6:.1f}M" if value > 1_000_000 else f"{value/1e3:.1f}K"
                    elif abs(value) >= 1000:
                        display_val = f"{value:.0f}"
                    elif abs(value) >= 10:
                        display_val = f"{value:.2f}"
                    else:
                        display_val = f"{value:.3f}"
                    label = group_info['labels'].get(
                        field, field.replace(".", " ").replace("_", " ").upper()
                    )
                    st.metric(label, display_val)


def render_indicator_evolution(symbol, open_df, close_df, prior_open_df, prior_close_df):
    symbol = symbol.strip().upper()

    def _row(df):
        if df.empty or 'symbol' not in df.columns:
            return None
        df = df.copy()
        df['symbol'] = df['symbol'].str.strip().str.upper()
        matches = df[df['symbol'] == symbol]
        return matches.iloc[0] if not matches.empty else None

    timepoints = []
    for label, df in [
        ("T-1 Open",     prior_open_df),
        ("T-1 Close",    prior_close_df),
        ("Market Open",  open_df),
        ("Market Close", close_df),
    ]:
        row = _row(df)
        if row is not None:
            timepoints.append((label, row))

    if len(timepoints) < 2:
        st.info(f"Need at least 2 time points for comparison. Found {len(timepoints)}.")
        return

    common_indicators = [
        "rsi", "macd_value", "adx", "volume", "close", "atr",
        "bb_width", "stoch_k", "ema20", "ema50", "sma20",
        "volatility_20d", "volume_ratio",
    ]
    available_indicators = [
        ind for ind in common_indicators
        if all(ind in data.index and pd.notna(data[ind]) for _, data in timepoints)
    ]

    if not available_indicators:
        st.warning("No common indicators found across all available time points.")
        return

    selected_indicators = st.multiselect(
        "Select indicators to plot:", available_indicators,
        default=available_indicators[:4],
        key=f"indicator_evolution_select_{symbol}",
    )
    if not selected_indicators:
        return

    rows = (len(selected_indicators) + 1) // 2
    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=selected_indicators,
        vertical_spacing=0.15, horizontal_spacing=0.1,
    )

    for idx, indicator in enumerate(selected_indicators):
        r, c = idx // 2 + 1, idx % 2 + 1
        fig.add_trace(go.Scatter(
            x=[name for name, _ in timepoints],
            y=[data[indicator] for _, data in timepoints],
            mode='lines+markers', name=indicator, showlegend=False,
            marker=dict(size=10, color='#6366f1'),
            line=dict(width=3, color='#6366f1'),
        ), row=r, col=c)
        fig.update_xaxes(title_text="", row=r, col=c, **CHART_THEME['xaxis'])
        fig.update_yaxes(title_text=indicator, row=r, col=c, **CHART_THEME['yaxis'])

    fig.update_layout(
        height=300 * rows,
        title_text=f"<b>Indicator Evolution for {symbol}</b>",
        showlegend=False,
        **CHART_THEME,
    )
    st.plotly_chart(fig, use_container_width=True)


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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Symbol", selected_symbol)
    col2.metric("Price",  f"${winner_info['price']:.2f}")
    col3.metric("Change", f"{winner_info['change_pct']:+.2f}%",
                delta=f"{winner_info['change_pct']:.2f}%")
    col4.metric("Volume", f"{winner_info['volume']/1e6:.1f}M")

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
    st.markdown("### Indicator Evolution")
    st.info("📊 Compare indicators across 4 timepoints: T-1 Open → T-1 Close → Market Open → Market Close")
    render_indicator_evolution(
        selected_symbol,
        market_open_df, market_close_df,
        day_prior_open_df, day_prior_close_df,
    )
