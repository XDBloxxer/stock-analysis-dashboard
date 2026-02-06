"""
Daily Winners Tab Module - UPDATED for new schema
Now reads from winners_day_prior_open and winners_day_prior_close tables
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
from supabase import create_client, Client

# Chart theme
CHART_THEME = {
    'plot_bgcolor': 'rgba(26, 29, 41, 0.6)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'font': dict(color='#e8eaf0'),
    'title_font': dict(size=18, color='#ffffff'),
    'xaxis': dict(gridcolor='rgba(255, 255, 255, 0.1)', color='#b8bcc8'),
    'yaxis': dict(gridcolor='rgba(255, 255, 255, 0.1)', color='#b8bcc8'),
}

COLORS = {
    'primary': '#667eea',
    'success': '#10b981',
    'danger': '#ef4444',
    'warning': '#f59e0b',
    'info': '#3b82f6'
}

@st.cache_resource
def get_supabase_client():
    supabase_url = os.environ.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url")
    supabase_key = os.environ.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("key")
    
    if not supabase_url or not supabase_key:
        st.error("❌ Supabase credentials not configured!")
        st.stop()
    
    return create_client(supabase_url, supabase_key)

@st.cache_data(ttl=300)
def load_supabase_data(table_name: str, filters: dict = None, _refresh_key: int = 0):
    try:
        client = get_supabase_client()
        query = client.table(table_name).select("*")
        
        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)
        
        response = query.execute()
        
        if not response.data:
            return pd.DataFrame()
        
        df = pd.DataFrame(response.data)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].str.strip().str.upper()
        
        return df
    except Exception as e:
        st.warning(f"Could not load from {table_name}: {str(e)}")
        return pd.DataFrame()

def render_indicator_snapshot(data_row, title):
    st.markdown(f"**{title}**")

    indicator_groups = {
        "Price & Volume": ["close", "open", "high", "low", "volume"],
        "Momentum": ["rsi", "rsi[1]", "rsi[2]", "stoch.k", "stoch.d", "mom", "w.r", "roc", "tsi", "kama"],
        "Trend": ["macd.macd", "macd.signal", "adx", "adx+di", "adx-di", "ema20", "ema50", "ema200", 
                 "sma20", "sma50", "sma200", "aroon_up", "aroon_down", "psar"],
        "Volatility": ["atr", "atr_pct", "bb.upper", "bb.lower", "bb_width", "volatility_20d", 
                      "keltner_upper", "keltner_lower", "donchian_upper", "donchian_lower"],
        "Volume Indicators": ["obv", "cmf", "force_index", "vpt", "volume_sma20", "volume_ratio"],
        "Other": ["cci20", "ao", "uo", "vwap", "high_52w", "low_52w", "gap_%"]
    }

    tabs = st.tabs(list(indicator_groups.keys()))

    for i, (group_name, indicators) in enumerate(indicator_groups.items()):
        with tabs[i]:
            available = [
                ind for ind in indicators
                if ind in data_row.index and pd.notna(data_row[ind])
            ]

            if not available:
                st.info(f"No {group_name} indicators available")
                continue

            cols = st.columns(min(4, len(available)))

            for j, indicator in enumerate(available):
                with cols[j % 4]:
                    value = data_row[indicator]

                    if indicator == "volume":
                        display_val = f"{value/1e6:.1f}M"
                    elif abs(value) >= 1000:
                        display_val = f"{value:.0f}"
                    elif abs(value) >= 10:
                        display_val = f"{value:.2f}"
                    else:
                        display_val = f"{value:.3f}"

                    label = indicator.replace(".", " ").replace("_", " ").upper()
                    st.metric(label, display_val)

def render_indicator_evolution(symbol, open_df, close_df, prior_open_df, prior_close_df):
    """Enhanced to show 4 timepoints: T-1 Open, T-1 Close, T0 Open, T0 Close"""
    symbol = symbol.strip().upper()
    
    # Clean up all dataframes
    for df in [open_df, close_df, prior_open_df, prior_close_df]:
        if not df.empty and 'symbol' in df.columns:
            df['symbol'] = df['symbol'].str.strip().str.upper()
    
    # Extract data for this symbol
    open_data = open_df[open_df['symbol'] == symbol].iloc[0] if not open_df.empty and symbol in open_df['symbol'].values else None
    close_data = close_df[close_df['symbol'] == symbol].iloc[0] if not close_df.empty and symbol in close_df['symbol'].values else None
    prior_open_data = prior_open_df[prior_open_df['symbol'] == symbol].iloc[0] if not prior_open_df.empty and symbol in prior_open_df['symbol'].values else None
    prior_close_data = prior_close_df[prior_close_df['symbol'] == symbol].iloc[0] if not prior_close_df.empty and symbol in prior_close_df['symbol'].values else None
    
    # Build timepoints list
    timepoints = []
    
    if prior_open_data is not None:
        timepoints.append(("T-1 Open", prior_open_data))
    if prior_close_data is not None:
        timepoints.append(("T-1 Close", prior_close_data))
    if open_data is not None:
        timepoints.append(("Market Open", open_data))
    if close_data is not None:
        timepoints.append(("Market Close", close_data))
    
    if len(timepoints) < 2:
        st.info(f"Need at least 2 time points for comparison. Found {len(timepoints)} timepoint(s).")
        if len(timepoints) == 1:
            st.write(f"Available: {timepoints[0][0]}")
        return
    
    # Find common indicators across all timepoints
    common_indicators = ["rsi", "macd.macd", "adx", "volume", "close", "atr", "bb_width", "stoch.k", 
                        "ema20", "ema50", "sma20", "volatility_20d", "volume_ratio"]
    available_indicators = []
    
    for ind in common_indicators:
        if all(ind in data.index and pd.notna(data[ind]) for _, data in timepoints):
            available_indicators.append(ind)
    
    if not available_indicators:
        st.warning("No common indicators across all time points")
        # Show what's available at each timepoint
        with st.expander("Debug: Available indicators per timepoint"):
            for name, data in timepoints:
                st.write(f"**{name}:** {len([c for c in common_indicators if c in data.index and pd.notna(data[c])])} indicators")
        return
    
    selected_indicators = st.multiselect(
        "Select indicators to plot:",
        available_indicators,
        default=available_indicators[:4] if len(available_indicators) >= 4 else available_indicators,
        key=f"indicator_evolution_select_{symbol}"
    )
    
    if not selected_indicators:
        return
    
    rows = (len(selected_indicators) + 1) // 2
    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=selected_indicators,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    for idx, indicator in enumerate(selected_indicators):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        times = [name for name, _ in timepoints]
        values = [data[indicator] for _, data in timepoints]
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode='lines+markers',
                name=indicator,
                marker=dict(size=10, color='#6366f1'),
                line=dict(width=3, color='#6366f1'),
                showlegend=False
            ),
            row=row,
            col=col
        )
        
        fig.update_xaxes(title_text="", row=row, col=col, **CHART_THEME['xaxis'])
        fig.update_yaxes(title_text=indicator, row=row, col=col, **CHART_THEME['yaxis'])
    
    fig.update_layout(
        height=300 * rows,
        title_text=f"<b>Indicator Evolution for {symbol}</b>",
        showlegend=False,
        **CHART_THEME
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_daily_winners_tab():
    if 'daily_winners_refresh_counter' not in st.session_state:
        st.session_state.daily_winners_refresh_counter = 0
    
    with st.spinner("Loading available dates..."):
        client = get_supabase_client()
        try:
            response = client.table("daily_winners").select("detection_date").execute()
            if response.data:
                available_dates = sorted(list(set(row["detection_date"] for row in response.data)), reverse=True)
            else:
                available_dates = []
        except Exception as e:
            st.error(f"Error loading dates: {e}")
            available_dates = []
    
    if not available_dates:
        st.warning("No daily winners data available yet")
        st.info("Run `python daily_winners_main.py` to collect data")
        return
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_date = st.selectbox(
            "Select Date:",
            available_dates,
            format_func=lambda x: datetime.fromisoformat(x).strftime("%A, %B %d, %Y"),
            key="daily_winners_date"
        )
    
    with col2:
        if st.button("Refresh Data", use_container_width=True, key="daily_winners_refresh"):
            st.cache_data.clear()
            st.session_state.daily_winners_refresh_counter += 1
            st.rerun()
    
    with col3:
        date_obj = datetime.fromisoformat(selected_date)
        st.metric("Day of Week", date_obj.strftime("%A"))
    
    refresh_key = st.session_state.daily_winners_refresh_counter
    with st.spinner(f"Loading data for {selected_date}..."):
        winners_df = load_supabase_data("daily_winners", {"detection_date": selected_date}, refresh_key)
        market_open_df = load_supabase_data("winners_market_open", {"detection_date": selected_date}, refresh_key)
        market_close_df = load_supabase_data("winners_market_close", {"detection_date": selected_date}, refresh_key)
        day_prior_open_df = load_supabase_data("winners_day_prior_open", {"detection_date": selected_date}, refresh_key)
        day_prior_close_df = load_supabase_data("winners_day_prior_close", {"detection_date": selected_date}, refresh_key)
    
    if winners_df.empty:
        st.warning(f"No winners data found for {selected_date}")
        return
    
    if 'symbol' in winners_df.columns:
        winners_df['symbol'] = winners_df['symbol'].str.strip().str.upper()
    
    st.subheader(f"Top {len(winners_df)} Winners - {selected_date}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Winners", len(winners_df))
    with col2:
        st.metric("Avg Change", f"{winners_df['change_pct'].mean():.2f}%")
    with col3:
        st.metric("Best Performer", f"{winners_df['change_pct'].max():.2f}%")
    with col4:
        st.metric("Avg Price", f"${winners_df['price'].mean():.2f}")
    with col5:
        st.metric("Avg Volume", f"{winners_df['volume'].mean()/1e6:.1f}M")
    
    st.markdown("### Winners List")
    
    display_df = winners_df[['symbol', 'exchange', 'price', 'change_pct', 'volume']].copy()
    display_df = display_df.sort_values('change_pct', ascending=False).reset_index(drop=True)
    display_df.index = display_df.index + 1
    display_df.columns = ['Symbol', 'Exchange', 'Price ($)', 'Change (%)', 'Volume']
    
    st.dataframe(
        display_df.style.format({
            'Price ($)': '${:.2f}',
            'Change (%)': '{:+.2f}%',
            'Volume': '{:,.0f}'
        }).background_gradient(subset=['Change (%)'], cmap='RdYlGn'),
        use_container_width=True,
        height=400
    )
    
    st.markdown("---")
    
    # Enhanced debug info
    with st.expander("📊 Data Availability Summary"):
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Winners", len(winners_df))
        with col2:
            st.metric("Market Open", len(market_open_df))
        with col3:
            st.metric("Market Close", len(market_close_df))
        with col4:
            st.metric("Prior Open", len(day_prior_open_df))
        with col5:
            st.metric("Prior Close", len(day_prior_close_df))
        
        if not winners_df.empty:
            st.write("**Winners symbols:**", ', '.join(sorted(winners_df['symbol'].unique())))
        
        # Show which symbols have data in each table
        all_symbols = set(winners_df['symbol'].unique())
        
        if not market_open_df.empty and 'symbol' in market_open_df.columns:
            open_symbols = set(market_open_df['symbol'].unique())
            missing_open = all_symbols - open_symbols
            if missing_open:
                st.warning(f"Market Open missing: {', '.join(sorted(missing_open))}")
        
        if not market_close_df.empty and 'symbol' in market_close_df.columns:
            close_symbols = set(market_close_df['symbol'].unique())
            missing_close = all_symbols - close_symbols
            if missing_close:
                st.warning(f"Market Close missing: {', '.join(sorted(missing_close))}")
        
        if not day_prior_open_df.empty and 'symbol' in day_prior_open_df.columns:
            prior_open_symbols = set(day_prior_open_df['symbol'].unique())
            missing_prior_open = all_symbols - prior_open_symbols
            if missing_prior_open:
                st.warning(f"Prior Open missing: {', '.join(sorted(missing_prior_open))}")
        
        if not day_prior_close_df.empty and 'symbol' in day_prior_close_df.columns:
            prior_close_symbols = set(day_prior_close_df['symbol'].unique())
            missing_prior_close = all_symbols - prior_close_symbols
            if missing_prior_close:
                st.warning(f"Prior Close missing: {', '.join(sorted(missing_prior_close))}")
    
    st.subheader("Detailed Stock Analysis")
    
    symbols = sorted(winners_df['symbol'].unique())
    selected_symbol = st.selectbox("Select a stock to analyze:", symbols, key="daily_winners_symbol")
    
    if selected_symbol:
        selected_symbol = selected_symbol.strip().upper()
        
        winner_info = winners_df[winners_df['symbol'] == selected_symbol].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Symbol", selected_symbol)
        with col2:
            st.metric("Price", f"${winner_info['price']:.2f}")
        with col3:
            st.metric("Change", f"{winner_info['change_pct']:+.2f}%", delta=f"{winner_info['change_pct']:.2f}%")
        with col4:
            st.metric("Volume", f"{winner_info['volume']/1e6:.1f}M")
        
        st.markdown("---")
        
        st.markdown("### Technical Indicator Snapshots")
        
        # Create 4 tabs instead of 3
        snapshot_tabs = st.tabs([
            "Day Prior Open (T-1 9:30 AM)", 
            "Day Prior Close (T-1 4:00 PM)",
            "Market Open (9:30 AM)", 
            "Market Close (4:00 PM)"
        ])
        
        with snapshot_tabs[0]:
            if not day_prior_open_df.empty and 'symbol' in day_prior_open_df.columns:
                day_prior_open_df['symbol'] = day_prior_open_df['symbol'].str.strip().str.upper()
                symbol_data = day_prior_open_df[day_prior_open_df['symbol'] == selected_symbol]
                
                if not symbol_data.empty:
                    render_indicator_snapshot(symbol_data.iloc[0], "Day Prior Open - T-1 9:30 AM")
                else:
                    st.warning(f"No day prior open data for {selected_symbol}")
            else:
                st.warning("No day prior open data available")
        
        with snapshot_tabs[1]:
            if not day_prior_close_df.empty and 'symbol' in day_prior_close_df.columns:
                day_prior_close_df['symbol'] = day_prior_close_df['symbol'].str.strip().str.upper()
                symbol_data = day_prior_close_df[day_prior_close_df['symbol'] == selected_symbol]
                
                if not symbol_data.empty:
                    render_indicator_snapshot(symbol_data.iloc[0], "Day Prior Close - T-1 4:00 PM")
                else:
                    st.warning(f"No day prior close data for {selected_symbol}")
            else:
                st.warning("No day prior close data available")
        
        with snapshot_tabs[2]:
            if not market_open_df.empty and 'symbol' in market_open_df.columns:
                market_open_df['symbol'] = market_open_df['symbol'].str.strip().str.upper()
                symbol_open = market_open_df[market_open_df['symbol'] == selected_symbol]
                
                if not symbol_open.empty:
                    render_indicator_snapshot(symbol_open.iloc[0], "Market Open - 9:30 AM")
                else:
                    st.warning(f"No market open data for {selected_symbol}")
            else:
                st.warning("No market open data available")
        
        with snapshot_tabs[3]:
            if not market_close_df.empty and 'symbol' in market_close_df.columns:
                market_close_df['symbol'] = market_close_df['symbol'].str.strip().str.upper()
                symbol_close = market_close_df[market_close_df['symbol'] == selected_symbol]
                
                if not symbol_close.empty:
                    render_indicator_snapshot(symbol_close.iloc[0], "Market Close - 4:00 PM")
                else:
                    st.warning(f"No market close data for {selected_symbol}")
            else:
                st.warning("No market close data available")
        
        st.markdown("---")
        
        st.markdown("### Indicator Evolution")
        st.info("Compare how indicators changed across 4 timepoints: T-1 Open → T-1 Close → Market Open → Market Close")
        render_indicator_evolution(selected_symbol, market_open_df, market_close_df, day_prior_open_df, day_prior_close_df)
