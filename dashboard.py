"""
Stock Event Analysis Dashboard - Improved Version
Clear, actionable visualizations for discovering what distinguishes spikers from grinders
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import numpy as np
from scipy import stats

# Supabase
from supabase import create_client, Client

# Page config
st.set_page_config(
    page_title="Stock Pattern Analysis",
    page_icon="🎯",
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
# INDICATOR CATEGORIZATION
# ============================================================================

INDICATOR_CATEGORIES = {
    "Momentum": {
        "indicators": ["rsi", "rsi[1]", "mom", "mom[1]", "stoch.k", "stoch.d", "stoch.k[1]", "stoch.d[1]"],
        "default_enabled": True,
        "description": "Rate of price change indicators (RSI, Stochastic, Momentum)"
    },
    "Trend": {
        "indicators": ["macd.macd", "macd.signal", "macd_diff", "adx", "adx+di", "adx-di", 
                      "ema5", "ema10", "ema20", "ema50", "ema100", "ema200",
                      "sma5", "sma10", "sma20", "sma50", "sma100", "sma200",
                      "ema20_above_ema50", "ema50_above_ema200", "price_above_ema20", "ema10_above_ema20"],
        "default_enabled": True,
        "description": "Trend direction and strength (MACD, ADX, Moving Averages)"
    },
    "Volatility": {
        "indicators": ["atr", "bb.upper", "bb.lower", "bb.middle", "bb_width", "bbpower", "volatility_20d"],
        "default_enabled": True,
        "description": "Price volatility and range (ATR, Bollinger Bands)"
    },
    "Volume": {
        "indicators": ["volume", "volume_sma5", "volume_sma20", "volume_ratio"],
        "default_enabled": True,
        "description": "Trading volume patterns"
    },
    "Oscillators": {
        "indicators": ["w.r", "cci20", "ao", "uo"],
        "default_enabled": True,
        "description": "Overbought/oversold indicators (Williams %R, CCI, AO, UO)"
    },
    "Price Levels": {
        "indicators": ["close", "open", "high", "low", "vwap",
                      "high_52w", "low_52w", "price_vs_high_52w", "price_vs_low_52w"],
        "default_enabled": True,
        "description": "Absolute and relative price levels"
    },
    "Price Changes (Contaminating)": {
        "indicators": ["gap_%", "gap_up", "gap_down",
                      "price_change_1d", "price_change_3d", "price_change_5d", 
                      "price_change_10d", "price_change_20d"],
        "default_enabled": False,
        "description": "Recent price changes (may contaminate pre-event analysis)"
    }
}

# Time lag order
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
    
    # Filter rows
    df_filtered = df[df['indicator'].str.lower().isin(enabled_indicators)].copy()
    
    return df_filtered


def sort_time_lags(time_lags):
    """Sort time lags in correct order"""
    ordered = []
    for lag in TIME_LAG_ORDER:
        if lag in time_lags:
            ordered.append(lag)
    # Add any other lags not in the standard order
    for lag in time_lags:
        if lag not in ordered:
            ordered.append(lag)
    return ordered


def create_category_selector(key_prefix, include_all_option=True):
    """Create a category selector widget"""
    categories = list(INDICATOR_CATEGORIES.keys())
    
    if include_all_option:
        options = ["All Categories"] + categories
        default_selected = ["All Categories"] + [
            cat for cat, info in INDICATOR_CATEGORIES.items() 
            if info["default_enabled"]
        ]
    else:
        options = categories
        default_selected = [
            cat for cat, info in INDICATOR_CATEGORIES.items() 
            if info["default_enabled"]
        ]
    
    selected = st.multiselect(
        "Filter by Category:",
        options=options,
        default=default_selected,
        key=f"{key_prefix}_categories",
        help="Select which indicator categories to include in this chart"
    )
    
    # If "All Categories" is selected, return all
    if "All Categories" in selected:
        return categories
    
    return selected


# ============================================================================
# DATA LOADING
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
def load_supabase_data(table_name: str, time_lag: str = None):
    """Load data from Supabase"""
    try:
        client = get_supabase_client()
        
        query = client.table(table_name).select("*")
        
        if time_lag:
            query = query.eq("time_lag", time_lag)
        
        response = query.execute()
        
        if not response.data:
            return pd.DataFrame()
        
        df = pd.DataFrame(response.data)
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        return df
    except Exception as e:
        st.warning(f"⚠️ Could not load from {table_name}: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_all_time_lags():
    """Load all time lag data from Supabase"""
    time_lags = {}
    
    try:
        client = get_supabase_client()
        response = client.table("raw_data").select("time_lag").execute()
        
        if response.data:
            unique_lags = list(set(row["time_lag"] for row in response.data))
            unique_lags = sort_time_lags(unique_lags)
            
            for lag in unique_lags:
                df = load_supabase_data("raw_data", time_lag=lag)
                if not df.empty:
                    time_lags[lag] = df
    except Exception as e:
        st.warning(f"Could not load time lags: {str(e)}")
    
    return time_lags


def validate_analysis_df(df):
    """Validate that analysis dataframe has required columns"""
    if df.empty:
        return df
    
    required_cols = ['indicator', 'time_lag', 'avg_spikers', 'avg_grinders']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Analysis data is missing required columns: {', '.join(missing_cols)}")
        return pd.DataFrame()
    
    # Remove rows where indicator is empty/null
    df = df[df['indicator'].notna() & (df['indicator'].astype(str).str.strip() != '')]
    
    # Ensure numeric columns are actually numeric
    for col in ['avg_spikers', 'avg_grinders', 'difference', 'ratio']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows where both averages are null
    df = df[df['avg_spikers'].notna() | df['avg_grinders'].notna()]
    
    return df


# ============================================================================
# NEW VISUALIZATION FUNCTIONS (Based on Sample)
# ============================================================================

def create_top_spikers_outperform(analysis_df, top_n=10, enabled_categories=None):
    """Chart 1: Top indicators where spikers outperform (vertical bar)"""
    if analysis_df.empty or 'difference' not in analysis_df.columns:
        return None
    
    df = analysis_df.copy()
    
    # Filter by categories
    if enabled_categories:
        df = filter_indicators_by_categories(df, enabled_categories)
    
    if df.empty:
        return None
    
    # Group by indicator and get average difference
    df_grouped = df.groupby('indicator').agg({
        'difference': 'mean',
        'avg_spikers': 'mean',
        'avg_grinders': 'mean'
    }).reset_index()
    
    # Get top N where spikers outperform
    df_top = df_grouped.nlargest(top_n, 'difference')
    
    if df_top.empty:
        return None
    
    # Add category
    df_top['category'] = df_top['indicator'].apply(get_indicator_category)
    
    # Create hover text
    hover_texts = []
    for _, row in df_top.iterrows():
        hover_texts.append(
            f"<b>{row['indicator']}</b> ({row['category']})<br>" +
            f"Avg Difference: {row['difference']:.2f}<br>" +
            f"Spiker Avg: {row['avg_spikers']:.2f}<br>" +
            f"Grinder Avg: {row['avg_grinders']:.2f}"
        )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_top['indicator'],
        y=df_top['difference'],
        marker_color='steelblue',
        marker_line_color='black',
        marker_line_width=1,
        text=[f'{x:.1f}' for x in df_top['difference']],
        textposition='outside',
        hovertext=hover_texts,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=f"<b>Top {top_n} Indicators Where Spikers Outperform Grinders</b><br>" +
              "<sub>Average difference across all time periods</sub>",
        xaxis_title="Indicator",
        yaxis_title="Avg Difference (Spikers - Grinders)",
        height=500,
        showlegend=False,
        xaxis={'tickangle': 45}
    )
    
    return fig


def create_top_grinders_outperform(analysis_df, top_n=10, enabled_categories=None):
    """Chart 2: Top indicators where grinders outperform (horizontal bar)"""
    if analysis_df.empty or 'difference' not in analysis_df.columns:
        return None
    
    df = analysis_df.copy()
    
    # Filter by categories
    if enabled_categories:
        df = filter_indicators_by_categories(df, enabled_categories)
    
    if df.empty:
        return None
    
    # Group by indicator and get average difference
    df_grouped = df.groupby('indicator').agg({
        'difference': 'mean',
        'avg_spikers': 'mean',
        'avg_grinders': 'mean'
    }).reset_index()
    
    # Get top N where grinders outperform (most negative)
    df_top = df_grouped.nsmallest(top_n, 'difference').sort_values('difference')
    
    if df_top.empty:
        return None
    
    # Add category
    df_top['category'] = df_top['indicator'].apply(get_indicator_category)
    
    # Create hover text
    hover_texts = []
    for _, row in df_top.iterrows():
        hover_texts.append(
            f"<b>{row['indicator']}</b> ({row['category']})<br>" +
            f"Avg Difference: {row['difference']:.2f}<br>" +
            f"Spiker Avg: {row['avg_spikers']:.2f}<br>" +
            f"Grinder Avg: {row['avg_grinders']:.2f}"
        )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_top['indicator'],
        x=df_top['difference'],
        orientation='h',
        marker_color='coral',
        marker_line_color='black',
        marker_line_width=1,
        text=[f'{x:.1f}' for x in df_top['difference']],
        textposition='outside',
        hovertext=hover_texts,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=f"<b>Top {top_n} Indicators Where Grinders Outperform Spikers</b><br>" +
              "<sub>Average difference across all time periods</sub>",
        xaxis_title="Avg Difference (Spikers - Grinders)",
        yaxis_title="",
        height=max(400, top_n * 35),
        showlegend=False
    )
    
    return fig


def create_time_lag_line_chart(analysis_df, indicators=None, enabled_categories=None):
    """Chart 3: Average values across time lags - line chart"""
    if analysis_df.empty:
        return None
    
    df = analysis_df.copy()
    
    # Filter by categories
    if enabled_categories:
        df = filter_indicators_by_categories(df, enabled_categories)
    
    if df.empty:
        return None
    
    # If specific indicators provided, filter to those
    if indicators:
        df = df[df['indicator'].isin(indicators)]
    
    # Group by time lag
    df_grouped = df.groupby('time_lag').agg({
        'avg_spikers': 'mean',
        'avg_grinders': 'mean'
    }).reset_index()
    
    # Sort by time lag
    df_grouped['time_lag_order'] = df_grouped['time_lag'].map(
        {lag: i for i, lag in enumerate(TIME_LAG_ORDER)}
    )
    df_grouped = df_grouped.sort_values('time_lag_order')
    
    if df_grouped.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_grouped['time_lag'],
        y=df_grouped['avg_spikers'],
        mode='lines+markers',
        name='Spikers',
        line=dict(color='blue', width=2),
        marker=dict(size=10, symbol='circle')
    ))
    
    fig.add_trace(go.Scatter(
        x=df_grouped['time_lag'],
        y=df_grouped['avg_grinders'],
        mode='lines+markers',
        name='Grinders',
        line=dict(color='orange', width=2),
        marker=dict(size=10, symbol='square')
    ))
    
    fig.update_layout(
        title="<b>Average Indicator Values Across Time Lags</b><br>" +
              "<sub>Spikers vs Grinders comparison</sub>",
        xaxis_title="Time Lag",
        yaxis_title="Average Value",
        height=500,
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig


def create_leading_indicators_t1(analysis_df, top_n=10, enabled_categories=None):
    """Chart 4: Leading indicators at T-1 with color coding"""
    if analysis_df.empty or 'time_lag' not in analysis_df.columns:
        return None
    
    df = analysis_df[analysis_df['time_lag'] == 'T-1'].copy()
    
    # Filter by categories
    if enabled_categories:
        df = filter_indicators_by_categories(df, enabled_categories)
    
    if df.empty or 'difference' not in df.columns:
        return None
    
    # Sort by absolute difference and take top N
    df['abs_diff'] = df['difference'].abs()
    df_top = df.nlargest(top_n, 'abs_diff').sort_values('difference', ascending=False)
    
    if df_top.empty:
        return None
    
    # Color code: green if positive (spikers higher), red if negative
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_top['difference']]
    
    # Add category
    df_top['category'] = df_top['indicator'].apply(get_indicator_category)
    
    # Create hover text
    hover_texts = []
    for _, row in df_top.iterrows():
        hover_texts.append(
            f"<b>{row['indicator']}</b> ({row['category']})<br>" +
            f"Difference: {row['difference']:.2f}<br>" +
            f"Spiker Avg: {row['avg_spikers']:.2f}<br>" +
            f"Grinder Avg: {row['avg_grinders']:.2f}"
        )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_top['indicator'],
        y=df_top['difference'],
        marker_color=colors,
        marker_line_color='black',
        marker_line_width=1,
        text=[f'{x:+.2f}' for x in df_top['difference']],
        textposition='outside',
        hovertext=hover_texts,
        hoverinfo='text'
    ))
    
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    fig.update_layout(
        title=f"<b>Top {top_n} Leading Indicators at T-1 (Day Before Event)</b><br>" +
              "<sub>Green = Spikers Higher | Red = Grinders Higher</sub>",
        xaxis_title="Indicator",
        yaxis_title="Difference (Spikers - Grinders)",
        height=500,
        showlegend=False,
        xaxis={'tickangle': 45}
    )
    
    return fig


def create_price_change_comparison(analysis_df, indicator='price_change_20d', enabled_categories=None):
    """Chart 5: Grouped bar chart comparing specific indicator across time lags"""
    if analysis_df.empty or indicator not in analysis_df['indicator'].values:
        return None
    
    df = analysis_df[analysis_df['indicator'] == indicator].copy()
    
    if df.empty:
        return None
    
    # Sort by time lag
    df['time_lag_order'] = df['time_lag'].map(
        {lag: i for i, lag in enumerate(TIME_LAG_ORDER)}
    )
    df = df.sort_values('time_lag_order')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Spikers',
        x=df['time_lag'],
        y=df['avg_spikers'],
        marker_color='steelblue',
        text=[f'{x:.1f}' for x in df['avg_spikers']],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Grinders',
        x=df['time_lag'],
        y=df['avg_grinders'],
        marker_color='coral',
        text=[f'{x:.1f}' for x in df['avg_grinders']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"<b>{indicator}: Spikers vs Grinders by Time Lag</b>",
        xaxis_title="Time Lag",
        yaxis_title=f"{indicator} Value",
        barmode='group',
        height=500,
        legend=dict(x=0.02, y=0.98)
    )
    
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig


def create_best_predictive_indicators(analysis_df, top_n=10, enabled_categories=None):
    """Chart 6: Best predictive indicators (horizontal bars)"""
    if analysis_df.empty or 'difference' not in analysis_df.columns:
        return None
    
    df = analysis_df.copy()
    
    # Filter by categories
    if enabled_categories:
        df = filter_indicators_by_categories(df, enabled_categories)
    
    if df.empty:
        return None
    
    # Calculate average metrics per indicator
    df_grouped = df.groupby('indicator').agg({
        'difference': 'mean',
        'ratio': 'mean',
        'avg_spikers': 'mean',
        'avg_grinders': 'mean'
    }).reset_index()
    
    # Get top N by average difference
    df_top = df_grouped.nlargest(top_n, 'difference').sort_values('difference')
    
    if df_top.empty:
        return None
    
    # Add category
    df_top['category'] = df_top['indicator'].apply(get_indicator_category)
    
    # Create hover text
    hover_texts = []
    for _, row in df_top.iterrows():
        hover_texts.append(
            f"<b>{row['indicator']}</b> ({row['category']})<br>" +
            f"Avg Difference: {row['difference']:.2f}<br>" +
            f"Avg Ratio: {row['ratio']:.2f}<br>" +
            f"Spiker Avg: {row['avg_spikers']:.2f}<br>" +
            f"Grinder Avg: {row['avg_grinders']:.2f}"
        )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_top['indicator'],
        x=df_top['difference'],
        orientation='h',
        marker_color='teal',
        marker_line_color='black',
        marker_line_width=1,
        text=[f'{x:.2f}' for x in df_top['difference']],
        textposition='outside',
        hovertext=hover_texts,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=f"<b>Top {top_n} Predictive Indicators for Spikers</b><br>" +
              "<sub>Averaged across all time lags</sub>",
        xaxis_title="Average Difference",
        yaxis_title="",
        height=max(400, top_n * 35),
        showlegend=False
    )
    
    return fig


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    """Main dashboard"""
    
    st.title("🎯 Stock Pattern Analysis Dashboard")
    st.markdown("**Discover what distinguishes explosive movers from steady grinders**")
    
    # Sidebar - Global settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("**💡 Tip:** Each chart has its own category filters")
    
    # Load data
    with st.spinner("Loading data from Supabase..."):
        candidates_df = load_supabase_data("candidates")
        analysis_df_raw = load_supabase_data("analysis")
        summary_df = load_supabase_data("summary_stats")
        raw_data_dict = load_all_time_lags()
    
    # Validate analysis data
    analysis_df = validate_analysis_df(analysis_df_raw)
    
    if analysis_df.empty and candidates_df.empty:
        st.warning("⚠️ No data found in Supabase")
        return
    
    # Summary metrics
    st.header("📈 Summary Statistics")
    
    if not summary_df.empty:
        summary_dict = dict(zip(summary_df['metric'], summary_df['value']))
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
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
        with col6:
            spiker_price = summary_dict.get('avg_spiker_price', 0)
            grinder_price = summary_dict.get('avg_grinder_price', 0)
            st.metric("Avg Price", f"${(spiker_price + grinder_price) / 2:.2f}")
    
    st.markdown("---")
    
    # Main visualizations
    if not analysis_df.empty:
        
        # Tab layout
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Key Insights",
            "🔍 Deep Dive",
            "📈 Time Analysis",
            "📋 Data Explorer"
        ])
        
        # TAB 1: Key Insights (New clear visualizations)
        with tab1:
            st.header("Key Insights: What Distinguishes Spikers from Grinders")
            
            # Chart 1: Top Spikers Outperform
            st.subheader("1️⃣ Where Spikers Excel")
            col1, col2 = st.columns([3, 1])
            with col1:
                enabled_cat_1 = create_category_selector("chart1")
            with col2:
                top_n_1 = st.slider("Top N:", 5, 20, 10, key='top_n_1')
            
            fig1 = create_top_spikers_outperform(analysis_df, top_n_1, enabled_cat_1)
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown("---")
            
            # Chart 2: Top Grinders Outperform
            st.subheader("2️⃣ Where Grinders Excel")
            col1, col2 = st.columns([3, 1])
            with col1:
                enabled_cat_2 = create_category_selector("chart2")
            with col2:
                top_n_2 = st.slider("Top N:", 5, 20, 10, key='top_n_2')
            
            fig2 = create_top_grinders_outperform(analysis_df, top_n_2, enabled_cat_2)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("---")
            
            # Chart 4: Leading Indicators at T-1
            st.subheader("3️⃣ Most Predictive Indicators (Day Before Event)")
            col1, col2 = st.columns([3, 1])
            with col1:
                enabled_cat_4 = create_category_selector("chart4")
            with col2:
                top_n_4 = st.slider("Top N:", 5, 20, 10, key='top_n_4')
            
            fig4 = create_leading_indicators_t1(analysis_df, top_n_4, enabled_cat_4)
            if fig4:
                st.plotly_chart(fig4, use_container_width=True)
            
            st.markdown("---")
            
            # Chart 6: Best Overall Predictive
            st.subheader("4️⃣ Best Overall Predictive Indicators")
            col1, col2 = st.columns([3, 1])
            with col1:
                enabled_cat_6 = create_category_selector("chart6")
            with col2:
                top_n_6 = st.slider("Top N:", 5, 15, 10, key='top_n_6')
            
            fig6 = create_best_predictive_indicators(analysis_df, top_n_6, enabled_cat_6)
            if fig6:
                st.plotly_chart(fig6, use_container_width=True)
        
        # TAB 2: Deep Dive
        with tab2:
            st.header("Deep Dive Analysis")
            
            # Time lag comparison
            st.subheader("Indicator Trends Across Time")
            enabled_cat_3 = create_category_selector("chart3")
            
            fig3 = create_time_lag_line_chart(analysis_df, enabled_categories=enabled_cat_3)
            if fig3:
                st.plotly_chart(fig3, use_container_width=True)
            
            st.markdown("---")
            
            # Specific indicator comparison
            st.subheader("Specific Indicator Deep Dive")
            
            # Get unique indicators
            unique_indicators = sorted(analysis_df['indicator'].unique())
            
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_indicator = st.selectbox(
                    "Select Indicator:",
                    unique_indicators,
                    index=unique_indicators.index('price_change_20d') if 'price_change_20d' in unique_indicators else 0,
                    key='specific_ind'
                )
            
            fig5 = create_price_change_comparison(analysis_df, selected_indicator)
            if fig5:
                st.plotly_chart(fig5, use_container_width=True)
            
            # Show detailed stats for selected indicator
            ind_data = analysis_df[analysis_df['indicator'] == selected_indicator]
            if not ind_data.empty:
                st.subheader(f"Statistics for {selected_indicator}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Spiker Value", f"{ind_data['avg_spikers'].mean():.2f}")
                with col2:
                    st.metric("Avg Grinder Value", f"{ind_data['avg_grinders'].mean():.2f}")
                with col3:
                    st.metric("Avg Difference", f"{ind_data['difference'].mean():.2f}")
                with col4:
                    if 'ratio' in ind_data.columns:
                        st.metric("Avg Ratio", f"{ind_data['ratio'].mean():.2f}")
        
        # TAB 3: Time Analysis
        with tab3:
            st.header("Time-Based Pattern Analysis")
            
            available_lags = sort_time_lags(analysis_df['time_lag'].unique().tolist())
            
            selected_lag = st.selectbox("Select Time Period:", available_lags, key='time_lag_sel')
            
            # Show distributions for this time lag
            if selected_lag in raw_data_dict:
                st.subheader(f"Indicator Distributions at {selected_lag}")
                
                lag_df = raw_data_dict[selected_lag]
                
                # Category filter
                enabled_cat_dist = create_category_selector("dist_time")
                
                # Get available indicators
                exclude_cols = ['id', 'symbol', 'event_date', 'event_type', 'exchange', 'time_lag', 'created_at', 'updated_at']
                available_indicators = [col for col in lag_df.columns if col not in exclude_cols]
                
                # Filter by enabled categories
                if enabled_cat_dist:
                    enabled_indicators = []
                    for category, info in INDICATOR_CATEGORIES.items():
                        if category in enabled_cat_dist:
                            enabled_indicators.extend([ind.lower() for ind in info["indicators"]])
                    available_indicators = [ind for ind in available_indicators if ind.lower() in enabled_indicators]
                
                if available_indicators:
                    selected_ind_dist = st.selectbox("Select Indicator:", sorted(available_indicators), key='ind_dist_time')
                    
                    # Create distribution plot
                    if selected_ind_dist in lag_df.columns and 'event_type' in lag_df.columns:
                        spikers = lag_df[lag_df['event_type'] == 'Spiker'][selected_ind_dist].dropna()
                        grinders = lag_df[lag_df['event_type'] == 'Grinder'][selected_ind_dist].dropna()
                        
                        fig = go.Figure()
                        
                        if len(spikers) > 0:
                            fig.add_trace(go.Histogram(
                                x=spikers,
                                name='Spikers',
                                opacity=0.7,
                                marker_color='#e74c3c',
                                nbinsx=30
                            ))
                        
                        if len(grinders) > 0:
                            fig.add_trace(go.Histogram(
                                x=grinders,
                                name='Grinders',
                                opacity=0.7,
                                marker_color='#3498db',
                                nbinsx=30
                            ))
                        
                        fig.update_layout(
                            title=f"<b>Distribution: {selected_ind_dist} at {selected_lag}</b>",
                            xaxis_title=selected_ind_dist,
                            yaxis_title="Count",
                            barmode='overlay',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics
                        if len(spikers) > 0 and len(grinders) > 0:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Spiker Mean", f"{spikers.mean():.3f}")
                                st.metric("Spiker Std Dev", f"{spikers.std():.3f}")
                            
                            with col2:
                                st.metric("Grinder Mean", f"{grinders.mean():.3f}")
                                st.metric("Grinder Std Dev", f"{grinders.std():.3f}")
                            
                            with col3:
                                t_stat, p_value = stats.ttest_ind(spikers, grinders, equal_var=False)
                                st.metric("T-Statistic", f"{t_stat:.3f}")
                                st.metric("P-Value", f"{p_value:.4f}")
                                
                                if p_value < 0.05:
                                    st.success("✓ Statistically significant")
                                else:
                                    st.info("Not significant")
        
        # TAB 4: Data Explorer
        with tab4:
            st.header("Raw Data Explorer")
            
            subtab1, subtab2, subtab3 = st.tabs(["Candidates", "Analysis", "Raw Data"])
            
            with subtab1:
                if not candidates_df.empty:
                    st.dataframe(candidates_df, use_container_width=True, height=600)
                    st.download_button(
                        "📥 Download CSV",
                        candidates_df.to_csv(index=False),
                        "candidates.csv",
                        "text/csv"
                    )
            
            with subtab2:
                if not analysis_df.empty:
                    st.dataframe(analysis_df, use_container_width=True, height=600)
                    st.download_button(
                        "📥 Download CSV",
                        analysis_df.to_csv(index=False),
                        "analysis.csv",
                        "text/csv"
                    )
            
            with subtab3:
                if raw_data_dict:
                    raw_lag = st.selectbox("Select time lag:", list(raw_data_dict.keys()), key='raw_lag_exp')
                    if raw_lag in raw_data_dict:
                        st.dataframe(raw_data_dict[raw_lag], use_container_width=True, height=600)
                        st.download_button(
                            f"📥 Download {raw_lag} CSV",
                            raw_data_dict[raw_lag].to_csv(index=False),
                            f"raw_data_{raw_lag}.csv",
                            "text/csv"
                        )


if __name__ == "__main__":
    main()
