"""
Daily Winners Tab Module
Handles all Daily Winners functionality
FIXED: Better symbol matching and debugging for indicator snapshots
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
from supabase import create_client, Client


# ============================================================================
# SUPABASE CONNECTION
# ============================================================================

@st.cache_resource
def get_supabase_client():
    """Initialize Supabase client"""
    supabase_url = os.environ.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url")
    supabase_key = os.environ.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("key")
    
    if not supabase_url or not supabase_key:
        st.error("❌ Supabase credentials not configured!")
        st.stop()
    
    return create_client(supabase_url, supabase_key)


@st.cache_data(ttl=300)
def load_supabase_data(table_name: str, filters: dict = None, _refresh_key: int = 0):
    """Load data from Supabase with optional filters"""
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
        
        # Normalize symbol column if it exists (strip whitespace, uppercase)
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].str.strip().str.upper()
        
        return df
    except Exception as e:
        st.warning(f"⚠️ Could not load from {table_name}: {str(e)}")
        return pd.DataFrame()


# ============================================================================
# RENDERING FUNCTIONS
# ============================================================================

def render_indicator_snapshot(data_row, title):
    """Render a single indicator snapshot"""
    st.markdown(f"**{title}**")

    indicator_groups = {
        "Price & Volume": ["close", "open", "high", "low", "volume"],
        "Momentum": ["rsi", "stoch.k", "stoch.d", "mom", "w.r"],
        "Trend": ["macd.macd", "macd.signal", "adx", "ema20", "ema50", "sma20", "sma50"],
        "Volatility": ["atr", "bb.upper", "bb.lower", "bb_width", "volatility_20d"],
        "Other": ["cci20", "ao", "uo", "vwap"]
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


def render_indicator_evolution(symbol, open_df, close_df, prior_df):
    """Show how indicators evolved from T-1 to market open to market close"""
    
    # Normalize symbol for comparison
    symbol = symbol.strip().upper()
    
    if open_df.empty or 'symbol' not in open_df.columns:
        open_df = pd.DataFrame()
    if close_df.empty or 'symbol' not in close_df.columns:
        close_df = pd.DataFrame()
    if prior_df.empty or 'symbol' not in prior_df.columns:
        prior_df = pd.DataFrame()
    
    # Normalize symbols in dataframes
    if not open_df.empty:
        open_df['symbol'] = open_df['symbol'].str.strip().str.upper()
    if not close_df.empty:
        close_df['symbol'] = close_df['symbol'].str.strip().str.upper()
    if not prior_df.empty:
        prior_df['symbol'] = prior_df['symbol'].str.strip().str.upper()
    
    open_data = open_df[open_df['symbol'] == symbol] if not open_df.empty else pd.DataFrame()
    close_data = close_df[close_df['symbol'] == symbol] if not close_df.empty else pd.DataFrame()
    prior_data = prior_df[prior_df['symbol'] == symbol] if not prior_df.empty else pd.DataFrame()
    
    if open_data.empty and close_data.empty and prior_data.empty:
        st.warning("No indicator data available for comparison")
        return
    
    timepoints = []
    
    if not prior_data.empty:
        timepoints.append(("T-1 Close", prior_data.iloc[0]))
    if not open_data.empty:
        timepoints.append(("Market Open", open_data.iloc[0]))
    if not close_data.empty:
        timepoints.append(("Market Close", close_data.iloc[0]))
    
    if len(timepoints) < 2:
        st.info("Need at least 2 time points for comparison")
        return
    
    common_indicators = ["rsi", "macd.macd", "adx", "volume", "close", "atr", "bb_width", "stoch.k"]
    available_indicators = []
    
    for ind in common_indicators:
        if all(ind in data.index and pd.notna(data[ind]) for _, data in timepoints):
            available_indicators.append(ind)
    
    if not available_indicators:
        st.warning("No common indicators across all time points")
        return
    
    selected_indicators = st.multiselect(
        "Select indicators to plot:",
        available_indicators,
        default=available_indicators[:4] if len(available_indicators) >= 4 else available_indicators,
        key="indicator_evolution_select"
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
                marker=dict(size=10),
                line=dict(width=3),
                showlegend=False
            ),
            row=row,
            col=col
        )
        
        fig.update_xaxes(title_text="", row=row, col=col)
        fig.update_yaxes(title_text=indicator, row=row, col=col)
    
    fig.update_layout(
        height=300 * rows,
        title_text=f"<b>Indicator Evolution for {symbol}</b>",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# MAIN TAB FUNCTION
# ============================================================================

def render_daily_winners_tab():
    """Main Daily Winners tab rendering function"""
    
    # Initialize refresh counter in session state
    if 'daily_winners_refresh_counter' not in st.session_state:
        st.session_state.daily_winners_refresh_counter = 0
    
    # Load available dates
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
        st.warning("📭 No daily winners data available yet.")
        return
    
    # Date selector
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
            st.session_state.daily_winners_refresh_counter += 1
            st.rerun()
    
    with col3:
        date_obj = datetime.fromisoformat(selected_date)
        st.metric("Day of Week", date_obj.strftime("%A"))
    
    # Load data for selected date with refresh key
    refresh_key = st.session_state.daily_winners_refresh_counter
    with st.spinner(f"Loading data for {selected_date}..."):
        winners_df = load_supabase_data("daily_winners", {"detection_date": selected_date}, refresh_key)
        market_open_df = load_supabase_data("winners_market_open", {"detection_date": selected_date}, refresh_key)
        market_close_df = load_supabase_data("winners_market_close", {"detection_date": selected_date}, refresh_key)
        day_prior_df = load_supabase_data("winners_day_prior", {"detection_date": selected_date}, refresh_key)
    
    if winners_df.empty:
        st.warning(f"No winners data found for {selected_date}")
        return
    
    # Normalize symbols in winners_df
    if 'symbol' in winners_df.columns:
        winners_df['symbol'] = winners_df['symbol'].str.strip().str.upper()
    
    # Display winners summary
    st.subheader(f"Top {len(winners_df)} Winners - {selected_date}")
    
    # Summary metrics
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
    
    # Winners table
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
    
    # DEBUG INFO (expandable)
    with st.expander("🔍 Debug Info - Data Availability"):
        st.write(f"**Winners table:** {len(winners_df)} rows")
        st.write(f"**Market Open table:** {len(market_open_df)} rows")
        st.write(f"**Market Close table:** {len(market_close_df)} rows")
        st.write(f"**Day Prior table:** {len(day_prior_df)} rows")
        
        if not winners_df.empty:
            st.write(f"**Winners symbols:** {', '.join(sorted(winners_df['symbol'].unique()[:10]))}")
        
        if not market_open_df.empty and 'symbol' in market_open_df.columns:
            st.write(f"**Market Open symbols:** {', '.join(sorted(market_open_df['symbol'].unique()[:10]))}")
        
        if not market_close_df.empty and 'symbol' in market_close_df.columns:
            st.write(f"**Market Close symbols:** {', '.join(sorted(market_close_df['symbol'].unique()[:10]))}")
        
        if not day_prior_df.empty and 'symbol' in day_prior_df.columns:
            st.write(f"**Day Prior symbols:** {', '.join(sorted(day_prior_df['symbol'].unique()[:10]))}")
    
    # Stock selector for detailed analysis
    st.subheader("Detailed Stock Analysis")
    
    symbols = sorted(winners_df['symbol'].unique())
    selected_symbol = st.selectbox("Select a stock to analyze:", symbols, key="daily_winners_symbol")
    
    if selected_symbol:
        # Normalize selected symbol
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
        
        snapshot_tabs = st.tabs(["Market Open (9:30 AM)", "Market Close (4:00 PM)", "Day Prior (T-1)"])
        
        with snapshot_tabs[0]:
            if not market_open_df.empty and 'symbol' in market_open_df.columns:
                # Normalize symbols in dataframe
                market_open_df['symbol'] = market_open_df['symbol'].str.strip().str.upper()
                symbol_open = market_open_df[market_open_df['symbol'] == selected_symbol]
                
                if not symbol_open.empty:
                    render_indicator_snapshot(symbol_open.iloc[0], "Market Open - 9:30 AM")
                else:
                    st.warning(f"No market open data for {selected_symbol}")
                    st.write(f"**Available symbols in market_open:** {', '.join(sorted(market_open_df['symbol'].unique()[:5]))}")
            else:
                st.warning("No market open data available")
        
        with snapshot_tabs[1]:
            if not market_close_df.empty and 'symbol' in market_close_df.columns:
                # Normalize symbols in dataframe
                market_close_df['symbol'] = market_close_df['symbol'].str.strip().str.upper()
                symbol_close = market_close_df[market_close_df['symbol'] == selected_symbol]
                
                if not symbol_close.empty:
                    render_indicator_snapshot(symbol_close.iloc[0], "Market Close - 4:00 PM")
                else:
                    st.warning(f"No market close data for {selected_symbol}")
                    st.write(f"**Available symbols in market_close:** {', '.join(sorted(market_close_df['symbol'].unique()[:5]))}")
            else:
                st.warning("No market close data available")
        
        with snapshot_tabs[2]:
            if not day_prior_df.empty and 'symbol' in day_prior_df.columns:
                # Normalize symbols in dataframe
                day_prior_df['symbol'] = day_prior_df['symbol'].str.strip().str.upper()
                symbol_prior = day_prior_df[day_prior_df['symbol'] == selected_symbol]
                
                if not symbol_prior.empty:
                    render_indicator_snapshot(symbol_prior.iloc[0], "Day Prior (T-1) - 4:00 PM")
                else:
                    st.warning(f"No day prior data for {selected_symbol}")
                    st.write(f"**Available symbols in day_prior:** {', '.join(sorted(day_prior_df['symbol'].unique()[:5]))}")
            else:
                st.warning("No day prior data available")
        
        st.markdown("---")
        
        st.markdown("### Indicator Evolution")
        render_indicator_evolution(selected_symbol, market_open_df, market_close_df, day_prior_df)
