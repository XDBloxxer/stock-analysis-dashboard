"""
Stock Event Analysis Dashboard - Enhanced Version
- Daily Winners Tracker (new)
- Spike/Grinder Analysis (existing, cleaned up)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import numpy as np
from scipy import stats

# Supabase
from supabase import create_client, Client

# Page config
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📊",
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
# DAILY WINNERS TAB
# ============================================================================

def render_daily_winners_tab():
    """Render the Daily Winners analysis tab"""
    st.header("🏆 Daily Winners Tracker")
    st.markdown("Track top 10 daily winners and their technical indicators at market open, close, and T-1")
    
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
            if not market_open_df.empty:
                symbol_open = market_open_df[market_open_df['symbol'] == selected_symbol]
                if not symbol_open.empty:
                    render_indicator_snapshot(symbol_open.iloc[0], "Market Open - 9:30 AM")
                else:
                    st.warning(f"No market open data for {selected_symbol}")
            else:
                st.warning("No market open data available")
        
        # Market Close
        with snapshot_tabs[1]:
            if not market_close_df.empty:
                symbol_close = market_close_df[market_close_df['symbol'] == selected_symbol]
                if not symbol_close.empty:
                    render_indicator_snapshot(symbol_close.iloc[0], "Market Close - 4:00 PM")
                else:
                    st.warning(f"No market close data for {selected_symbol}")
            else:
                st.warning("No market close data available")
        
        # Day Prior
        with snapshot_tabs[2]:
            if not day_prior_df.empty:
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


def render_indicator_snapshot(data_row, title):
    """Render a single indicator snapshot"""
    st.markdown(f"**{title}**")
    
    # Define indicator groups
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
            # Filter to available indicators
            available = [ind for ind in indicators if ind in data_row.index and pd.notna(data_row[ind])]
            
            if not available:
                st.info(f"No {group_name} indicators available")
                continue
            
            # Create columns for metrics
            cols = st.columns(min(4, len(available)))
            
            for j, indicator in enumerate(available):
                with cols[j % 4]:
                    value = data_row[ind]
                    
                    # Format value
                    if indicator == 'volume':
                        display_val = f"{value/1e6:.1f}M"
                    elif abs(value) > 1000:
                        display_val = f"{value:.0f}"
                    elif abs(value) > 10:
                        display_val = f"{value:.2f}"
                    else:
                        display_val = f"{value:.3f}"
                    
                    st.metric(indicator.upper(), display_val)


def render_indicator_evolution(symbol, open_df, close_df, prior_df):
    """Show how indicators evolved from T-1 to market open to market close"""
    
    # Get data for this symbol
    open_data = open_df[open_df['symbol'] == symbol]
    close_data = close_df[close_df['symbol'] == symbol]
    prior_data = prior_df[prior_df['symbol'] == symbol]
    
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
# SPIKE/GRINDER ANALYSIS TAB
# ============================================================================

# Indicator categories (from original)
INDICATOR_CATEGORIES = {
    "Momentum": {
        "indicators": ["rsi", "rsi[1]", "mom", "mom[1]", "stoch.k", "stoch.d", "stoch.k[1]", "stoch.d[1]"],
        "default_enabled": True,
    },
    "Trend": {
        "indicators": ["macd.macd", "macd.signal", "macd_diff", "adx", "adx+di", "adx-di", 
                      "ema5", "ema10", "ema20", "ema50", "ema100", "ema200",
                      "sma5", "sma10", "sma20", "sma50", "sma100", "sma200"],
        "default_enabled": True,
    },
    "Volatility": {
        "indicators": ["atr", "bb.upper", "bb.lower", "bb.middle", "bb_width", "bbpower", "volatility_20d"],
        "default_enabled": True,
    },
    "Volume": {
        "indicators": ["volume", "volume_sma5", "volume_sma20", "volume_ratio"],
        "default_enabled": True,
    },
}

TIME_LAG_ORDER = ["T-1", "T-3", "T-5", "T-10", "T-30"]


def get_indicator_category(indicator):
    """Get category for an indicator"""
    indicator_lower = indicator.lower()
    for category, info in INDICATOR_CATEGORIES.items():
        if indicator_lower in [ind.lower() for ind in info["indicators"]]:
            return category
    return "Other"


def filter_indicators_by_categories(df, enabled_categories):
    """Filter dataframe to only include indicators from enabled categories"""
    if 'indicator' not in df.columns:
        return df
    
    enabled_indicators = []
    for category, info in INDICATOR_CATEGORIES.items():
        if category in enabled_categories:
            enabled_indicators.extend([ind.lower() for ind in info["indicators"]])
    
    df_filtered = df[df['indicator'].str.lower().isin(enabled_indicators)].copy()
    return df_filtered


def create_category_selector(key_prefix):
    """Create a category selector widget"""
    categories = list(INDICATOR_CATEGORIES.keys())
    default_selected = [cat for cat, info in INDICATOR_CATEGORIES.items() if info["default_enabled"]]
    
    selected = st.multiselect(
        "Filter by Category:",
        options=categories,
        default=default_selected,
        key=f"{key_prefix}_categories"
    )
    
    return selected


def create_top_differences_chart(analysis_df, top_n=10, enabled_categories=None):
    """Create chart showing top indicator differences"""
    if analysis_df.empty or 'difference' not in analysis_df.columns:
        return None
    
    df = analysis_df.copy()
    
    if enabled_categories:
        df = filter_indicators_by_categories(df, enabled_categories)
    
    if df.empty:
        return None
    
    # Group by indicator
    df_grouped = df.groupby('indicator').agg({
        'difference': 'mean',
        'avg_spikers': 'mean',
        'avg_grinders': 'mean'
    }).reset_index()
    
    # Get top N by absolute difference
    df_grouped['abs_diff'] = df_grouped['difference'].abs()
    df_top = df_grouped.nlargest(top_n, 'abs_diff').sort_values('difference')
    
    if df_top.empty:
        return None
    
    # Color code
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_top['difference']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_top['indicator'],
        x=df_top['difference'],
        orientation='h',
        marker_color=colors,
        text=[f'{x:+.2f}' for x in df_top['difference']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"<b>Top {top_n} Differentiating Indicators</b><br><sub>Green = Spikers Higher | Red = Grinders Higher</sub>",
        xaxis_title="Difference (Spikers - Grinders)",
        yaxis_title="",
        height=max(400, top_n * 35),
        showlegend=False
    )
    
    return fig


def render_spike_grinder_analysis_tab():
    """Render the Spike/Grinder analysis tab"""
    st.header("📈 Spike vs Grinder Pattern Analysis")
    
    # Load data
    with st.spinner("Loading analysis data..."):
        candidates_df = load_supabase_data("candidates")
        analysis_df = load_supabase_data("analysis")
        summary_df = load_supabase_data("summary_stats")
    
    if analysis_df.empty and candidates_df.empty:
        st.warning("⚠️ No analysis data available yet. Run the main analysis pipeline first.")
        st.code("python main.py --all --verbose", language="bash")
        return
    
    # Validate analysis data
    if not analysis_df.empty:
        required_cols = ['indicator', 'time_lag', 'avg_spikers', 'avg_grinders']
        missing_cols = [col for col in required_cols if col not in analysis_df.columns]
        
        if missing_cols:
            st.error(f"❌ Analysis data is missing required columns: {', '.join(missing_cols)}")
            return
        
        analysis_df = analysis_df[analysis_df['indicator'].notna()]
        for col in ['avg_spikers', 'avg_grinders', 'difference']:
            if col in analysis_df.columns:
                analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
    
    # Summary metrics
    if not summary_df.empty:
        st.subheader("📊 Summary Statistics")
        summary_dict = dict(zip(summary_df['metric'], summary_df['value']))
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Events", f"{summary_dict.get('total_events', 0):.0f}")
        with col2:
            st.metric("Spikers", f"{summary_dict.get('total_spikers', 0):.0f}")
        with col3:
            st.metric("Grinders", f"{summary_dict.get('total_grinders', 0):.0f}")
        with col4:
            st.metric("Avg Spiker Move", f"{summary_dict.get('avg_spiker_change_pct', 0):.1f}%")
        with col5:
            st.metric("Avg Grinder Move", f"{summary_dict.get('avg_grinder_change_pct', 0):.1f}%")
        
        st.markdown("---")
    
    # Create sub-tabs
    subtabs = st.tabs(["📊 Key Insights", "🔍 Custom Analysis", "📋 Raw Data"])
    
    # Key Insights
    with subtabs[0]:
        if not analysis_df.empty:
            st.subheader("Top Differentiating Indicators")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                enabled_cat = create_category_selector("insights")
            with col2:
                top_n = st.slider("Top N:", 5, 20, 10, key='insights_top_n')
            
            fig = create_top_differences_chart(analysis_df, top_n, enabled_cat)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # Custom Analysis
    with subtabs[1]:
        st.subheader("Build Custom Visualizations")
        
        if not analysis_df.empty:
            # Chart type selector
            chart_type = st.selectbox(
                "Select chart type:",
                ["Time Lag Comparison", "Indicator Distribution", "Scatter Plot"]
            )
            
            if chart_type == "Time Lag Comparison":
                # Select indicators
                available_indicators = sorted(analysis_df['indicator'].unique())
                selected_indicators = st.multiselect(
                    "Select indicators:",
                    available_indicators,
                    default=available_indicators[:3] if len(available_indicators) >= 3 else available_indicators
                )
                
                if selected_indicators:
                    df_filtered = analysis_df[analysis_df['indicator'].isin(selected_indicators)]
                    
                    # Sort by time lag
                    df_filtered['time_lag_order'] = df_filtered['time_lag'].map(
                        {lag: i for i, lag in enumerate(TIME_LAG_ORDER)}
                    )
                    df_filtered = df_filtered.sort_values(['indicator', 'time_lag_order'])
                    
                    fig = px.line(
                        df_filtered,
                        x='time_lag',
                        y='difference',
                        color='indicator',
                        markers=True,
                        title="Indicator Differences Across Time Lags"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Indicator Distribution":
                # Load raw data for a specific time lag
                available_lags = sorted(analysis_df['time_lag'].unique())
                selected_lag = st.selectbox("Select time lag:", available_lags)
                
                raw_df = load_supabase_data("raw_data", {"time_lag": selected_lag})
                
                if not raw_df.empty and 'event_type' in raw_df.columns:
                    exclude_cols = ['id', 'symbol', 'event_date', 'event_type', 'exchange', 'time_lag', 'created_at']
                    available_ind = [col for col in raw_df.columns if col not in exclude_cols]
                    
                    selected_ind = st.selectbox("Select indicator:", sorted(available_ind))
                    
                    if selected_ind in raw_df.columns:
                        spikers = raw_df[raw_df['event_type'] == 'Spiker'][selected_ind].dropna()
                        grinders = raw_df[raw_df['event_type'] == 'Grinder'][selected_ind].dropna()
                        
                        fig = go.Figure()
                        
                        if len(spikers) > 0:
                            fig.add_trace(go.Histogram(x=spikers, name='Spikers', opacity=0.7, nbinsx=30))
                        if len(grinders) > 0:
                            fig.add_trace(go.Histogram(x=grinders, name='Grinders', opacity=0.7, nbinsx=30))
                        
                        fig.update_layout(
                            title=f"Distribution: {selected_ind} at {selected_lag}",
                            barmode='overlay',
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Stats
                        if len(spikers) > 0 and len(grinders) > 0:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Spiker Mean", f"{spikers.mean():.3f}")
                            with col2:
                                st.metric("Grinder Mean", f"{grinders.mean():.3f}")
            
            elif chart_type == "Scatter Plot":
                st.info("Select two indicators to plot against each other")
                
                raw_lag = st.selectbox("Select time lag:", sorted(analysis_df['time_lag'].unique()), key='scatter_lag')
                raw_df = load_supabase_data("raw_data", {"time_lag": raw_lag})
                
                if not raw_df.empty:
                    exclude_cols = ['id', 'symbol', 'event_date', 'event_type', 'exchange', 'time_lag', 'created_at']
                    available_ind = [col for col in raw_df.columns if col not in exclude_cols]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        x_ind = st.selectbox("X-axis:", sorted(available_ind), key='scatter_x')
                    with col2:
                        y_ind = st.selectbox("Y-axis:", sorted(available_ind), key='scatter_y', 
                                           index=min(1, len(available_ind)-1))
                    
                    if x_ind in raw_df.columns and y_ind in raw_df.columns and 'event_type' in raw_df.columns:
                        fig = px.scatter(
                            raw_df,
                            x=x_ind,
                            y=y_ind,
                            color='event_type',
                            color_discrete_map={'Spiker': '#e74c3c', 'Grinder': '#3498db'},
                            title=f"{y_ind} vs {x_ind} at {raw_lag}",
                            opacity=0.6
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
    
    # Raw Data
    with subtabs[2]:
        data_tabs = st.tabs(["Candidates", "Analysis Results", "Raw Indicator Data"])
        
        with data_tabs[0]:
            if not candidates_df.empty:
                st.dataframe(candidates_df, use_container_width=True, height=500)
                st.download_button("📥 Download", candidates_df.to_csv(index=False), "candidates.csv", "text/csv")
        
        with data_tabs[1]:
            if not analysis_df.empty:
                st.dataframe(analysis_df, use_container_width=True, height=500)
                st.download_button("📥 Download", analysis_df.to_csv(index=False), "analysis.csv", "text/csv")
        
        with data_tabs[2]:
            lag_selector = st.selectbox("Select time lag:", TIME_LAG_ORDER, key='raw_data_lag')
            raw_df = load_supabase_data("raw_data", {"time_lag": lag_selector})
            if not raw_df.empty:
                st.dataframe(raw_df, use_container_width=True, height=500)
                st.download_button("📥 Download", raw_df.to_csv(index=False), f"raw_data_{lag_selector}.csv", "text/csv")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application"""
    
    st.title("📊 Stock Analysis Dashboard")
    st.markdown("Comprehensive stock pattern analysis and daily winners tracking")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if st.button("🔄 Refresh All Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Data Sources:**")
        st.markdown("- Daily Winners: `daily_winners_main.py`")
        st.markdown("- Spike/Grinder: `main.py`")
    
    # Main tabs
    main_tabs = st.tabs(["🏆 Daily Winners", "📈 Spike/Grinder Analysis"])
    
    with main_tabs[0]:
        render_daily_winners_tab()
    
    with main_tabs[1]:
        render_spike_grinder_analysis_tab()


if __name__ == "__main__":
    main()
