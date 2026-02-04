"""
Daily Winners Tracker Dashboard
Standalone dashboard for tracking top daily stock winners
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# Supabase
from supabase import create_client, Client

# Page config
st.set_page_config(
    page_title="Daily Winners Tracker",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem 2rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    h1 {
        color: #2d3748;
        font-weight: 700;
    }
    h2 {
        color: #4a5568;
        font-weight: 600;
        margin-top: 2rem;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


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
        st.info("Set SUPABASE_URL and SUPABASE_KEY in Streamlit secrets or environment variables")
        st.stop()
    
    return create_client(supabase_url, supabase_key)


@st.cache_data(ttl=300)
def load_supabase_data(table_name: str, filters: dict = None):
    """Load data from Supabase with optional filters"""
    try:
        client = get_supabase_client()
        
        query = client.table(table_name).select("*")
        
        # Apply filters
        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)
        
        response = query.execute()
        
        if not response.data:
            return pd.DataFrame()
        
        df = pd.DataFrame(response.data)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        return df
    except Exception as e:
        st.warning(f"⚠️ Could not load from {table_name}: {str(e)}")
        return pd.DataFrame()


# ============================================================================
# INDICATOR SNAPSHOT RENDERING
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

                    # Format value
                    if indicator == "volume":
                        display_val = f"{value/1e6:.1f}M"
                    elif abs(value) >= 1000:
                        display_val = f"{value:.0f}"
                    elif abs(value) >= 10:
                        display_val = f"{value:.2f}"
                    else:
                        display_val = f"{value:.3f}"

                    # Clean label
                    label = indicator.replace(".", " ").replace("_", " ").upper()

                    st.metric(label, display_val)


def render_indicator_evolution(symbol, open_df, close_df, prior_df):
    """Show how indicators evolved from T-1 to market open to market close"""
    
    # Check if dataframes are empty or don't have the symbol column
    if open_df.empty or 'symbol' not in open_df.columns:
        open_df = pd.DataFrame()
    if close_df.empty or 'symbol' not in close_df.columns:
        close_df = pd.DataFrame()
    if prior_df.empty or 'symbol' not in prior_df.columns:
        prior_df = pd.DataFrame()
    
    # Get data for this symbol
    open_data = open_df[open_df['symbol'] == symbol] if not open_df.empty else pd.DataFrame()
    close_data = close_df[close_df['symbol'] == symbol] if not close_df.empty else pd.DataFrame()
    prior_data = prior_df[prior_df['symbol'] == symbol] if not prior_df.empty else pd.DataFrame()
    
    if open_data.empty and close_data.empty and prior_data.empty:
        st.warning("No indicator data available for comparison")
        return
    
    # Combine into time series
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
    
    # Select indicators to plot
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
        default=available_indicators[:4] if len(available_indicators) >= 4 else available_indicators
    )
    
    if not selected_indicators:
        return
    
    # Create subplots
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
        
        # Extract values
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
        
        # Update axes
        fig.update_xaxes(title_text="", row=row, col=col)
        fig.update_yaxes(title_text=indicator, row=row, col=col)
    
    fig.update_layout(
        height=300 * rows,
        title_text=f"<b>Indicator Evolution for {symbol}</b>",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    """Main daily winners dashboard"""
    
    st.title("🏆 Daily Winners Tracker")
    st.markdown("Track top 10 daily winners and their technical indicators at market open, close, and T-1")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Data Source:**")
        st.markdown("`daily_winners_main.py`")
        
        st.markdown("---")
        st.markdown("**Other Dashboards:**")
        st.markdown("📈 [Spike/Grinder Analysis](dashboard_spike_grinder.py)")
    
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
        st.warning("📭 No daily winners data available yet. Run the daily winners tracker first.")
        st.code("python daily_winners_main.py --verbose", language="bash")
        return
    
    # Date selector
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_date = st.selectbox(
            "📅 Select Date:",
            available_dates,
            format_func=lambda x: datetime.fromisoformat(x).strftime("%A, %B %d, %Y")
        )
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        date_obj = datetime.fromisoformat(selected_date)
        st.metric("Day of Week", date_obj.strftime("%A"))
    
    # Load data for selected date
    with st.spinner(f"Loading data for {selected_date}..."):
        winners_df = load_supabase_data("daily_winners", {"detection_date": selected_date})
        market_open_df = load_supabase_data("winners_market_open", {"detection_date": selected_date})
        market_close_df = load_supabase_data("winners_market_close", {"detection_date": selected_date})
        day_prior_df = load_supabase_data("winners_day_prior", {"detection_date": selected_date})
    
    if winners_df.empty:
        st.warning(f"No winners data found for {selected_date}")
        return
    
    # Display winners summary
    st.subheader(f"📊 Top {len(winners_df)} Winners - {selected_date}")
    
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
    st.markdown("### 🎯 Winners List")
    
    # Format the display dataframe
    display_df = winners_df[['symbol', 'exchange', 'price', 'change_pct', 'volume']].copy()
    display_df = display_df.sort_values('change_pct', ascending=False).reset_index(drop=True)
    display_df.index = display_df.index + 1  # Start from 1
    display_df.columns = ['Symbol', 'Exchange', 'Price ($)', 'Change (%)', 'Volume']
    
    # Style the dataframe
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
    
    # Stock selector for detailed analysis
    st.subheader("🔍 Detailed Stock Analysis")
    
    symbols = sorted(winners_df['symbol'].unique())
    selected_symbol = st.selectbox("Select a stock to analyze:", symbols)
    
    if selected_symbol:
        # Get data for this symbol
        winner_info = winners_df[winners_df['symbol'] == selected_symbol].iloc[0]
        
        # Symbol header
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
        
        # Indicator snapshots
        st.markdown("### 📸 Technical Indicator Snapshots")
        
        snapshot_tabs = st.tabs(["🌅 Market Open (9:30 AM)", "🌆 Market Close (4:00 PM)", "⏮️ Day Prior (T-1)"])
        
        # Market Open
        with snapshot_tabs[0]:
            if not market_open_df.empty and 'symbol' in market_open_df.columns:
                symbol_open = market_open_df[market_open_df['symbol'] == selected_symbol]
                if not symbol_open.empty:
                    render_indicator_snapshot(symbol_open.iloc[0], "Market Open - 9:30 AM")
                else:
                    st.warning(f"No market open data for {selected_symbol}")
            else:
                st.warning("No market open data available")
        
        # Market Close
        with snapshot_tabs[1]:
            if not market_close_df.empty and 'symbol' in market_close_df.columns:
                symbol_close = market_close_df[market_close_df['symbol'] == selected_symbol]
                if not symbol_close.empty:
                    render_indicator_snapshot(symbol_close.iloc[0], "Market Close - 4:00 PM")
                else:
                    st.warning(f"No market close data for {selected_symbol}")
            else:
                st.warning("No market close data available")
        
        # Day Prior
        with snapshot_tabs[2]:
            if not day_prior_df.empty and 'symbol' in day_prior_df.columns:
                symbol_prior = day_prior_df[day_prior_df['symbol'] == selected_symbol]
                if not symbol_prior.empty:
                    render_indicator_snapshot(symbol_prior.iloc[0], "Day Prior (T-1) - 4:00 PM")
                else:
                    st.warning(f"No day prior data for {selected_symbol}")
            else:
                st.warning("No day prior data available")
        
        st.markdown("---")
        
        # Comparison chart across time points
        st.markdown("### 📈 Indicator Evolution")
        render_indicator_evolution(selected_symbol, market_open_df, market_close_df, day_prior_df)


if __name__ == "__main__":
    main()
