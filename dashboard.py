"""
Stock Event Analysis Dashboard - Enhanced Version
Comprehensive analysis with customizable charts, heatmaps, and per-graph filtering
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

# Google Sheets (fallback)
import gspread
from google.oauth2.service_account import Credentials
from pathlib import Path

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
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_percentage_difference_chart(analysis_df, time_lag='T-1', top_n=20, enabled_categories=None):
    """Create horizontal bar chart showing percentage differences"""
    if analysis_df.empty or 'time_lag' not in analysis_df.columns:
        return None
    
    df = analysis_df[analysis_df['time_lag'] == time_lag].copy()
    
    # Filter by categories
    if enabled_categories:
        df = filter_indicators_by_categories(df, enabled_categories)
    
    if df.empty:
        return None
    
    # Calculate percentage difference
    df['pct_difference'] = np.where(
        df['avg_grinders'].abs() > 0.01,
        ((df['avg_spikers'] - df['avg_grinders']) / df['avg_grinders'].abs()) * 100,
        0
    )
    
    # Remove infinite and NaN values
    df = df[np.isfinite(df['pct_difference'])]
    
    if df.empty:
        return None
    
    # Sort and take top N
    df['abs_pct_diff'] = df['pct_difference'].abs()
    actual_n = min(top_n, len(df))
    df = df.nlargest(actual_n, 'abs_pct_diff').sort_values('pct_difference')
    
    # Add category info
    df['category'] = df['indicator'].apply(get_indicator_category)
    
    # Color by direction
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in df['pct_difference']]
    
    # Create hover text
    hover_texts = []
    for _, row in df.iterrows():
        hover_texts.append(
            f"<b>{row['indicator']}</b> ({row['category']})<br>" +
            f"% Difference: {row['pct_difference']:.1f}%<br><br>" +
            f"Spiker Avg: {row['avg_spikers']:.3f}<br>" +
            f"Grinder Avg: {row['avg_grinders']:.3f}<br>"
        )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['indicator'],
        x=df['pct_difference'],
        orientation='h',
        marker_color=colors,
        text=[f'{x:+.1f}%' for x in df['pct_difference']],
        textposition='outside',
        hovertext=hover_texts,
        hoverinfo='text'
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2)
    
    max_abs = df['pct_difference'].abs().max()
    x_range = [-max_abs * 1.15, max_abs * 1.15]
    
    fig.update_layout(
        title=f"<b>Top {actual_n} Discriminating Indicators ({time_lag})</b><br>" +
              "<sub>Red = Spikers Higher | Blue = Grinders Higher</sub>",
        xaxis_title="Percentage Difference (%)",
        yaxis_title="",
        height=max(500, actual_n * 30),
        showlegend=False,
        xaxis=dict(range=x_range, zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    )
    
    return fig


def create_absolute_difference_chart(analysis_df, time_lag='T-1', top_n=20, enabled_categories=None):
    """Create chart showing absolute value differences"""
    if analysis_df.empty or 'time_lag' not in analysis_df.columns:
        return None
    
    df = analysis_df[analysis_df['time_lag'] == time_lag].copy()
    
    # Filter by categories
    if enabled_categories:
        df = filter_indicators_by_categories(df, enabled_categories)
    
    if df.empty or 'difference' not in df.columns:
        return None
    
    # Remove NaN
    df = df[df['difference'].notna()]
    
    if df.empty:
        return None
    
    # Sort by absolute difference
    df['abs_diff'] = df['difference'].abs()
    actual_n = min(top_n, len(df))
    df = df.nlargest(actual_n, 'abs_diff').sort_values('difference')
    
    # Add category
    df['category'] = df['indicator'].apply(get_indicator_category)
    
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in df['difference']]
    
    hover_texts = []
    for _, row in df.iterrows():
        hover_texts.append(
            f"<b>{row['indicator']}</b> ({row['category']})<br>" +
            f"Difference: {row['difference']:.3f}<br><br>" +
            f"Spiker Avg: {row['avg_spikers']:.3f}<br>" +
            f"Grinder Avg: {row['avg_grinders']:.3f}<br>"
        )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['indicator'],
        x=df['difference'],
        orientation='h',
        marker_color=colors,
        text=[f'{x:+.3f}' for x in df['difference']],
        textposition='outside',
        hovertext=hover_texts,
        hoverinfo='text'
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2)
    
    max_abs = df['difference'].abs().max()
    x_range = [-max_abs * 1.15, max_abs * 1.15]
    
    fig.update_layout(
        title=f"<b>Top {actual_n} by Absolute Difference ({time_lag})</b><br>" +
              "<sub>Red = Spikers Higher | Blue = Grinders Higher</sub>",
        xaxis_title="Absolute Difference",
        yaxis_title="",
        height=max(500, actual_n * 30),
        showlegend=False,
        xaxis=dict(range=x_range)
    )
    
    return fig


def create_heatmap_across_time_lags(analysis_df, metric='pct_difference', top_n=20, enabled_categories=None):
    """Create heatmap showing indicator behavior across all time lags"""
    if analysis_df.empty:
        return None
    
    # Filter by categories
    if enabled_categories:
        df = filter_indicators_by_categories(analysis_df, enabled_categories)
    else:
        df = analysis_df.copy()
    
    if df.empty:
        return None
    
    # Calculate metric
    if metric == 'pct_difference':
        df['value'] = np.where(
            df['avg_grinders'].abs() > 0.01,
            ((df['avg_spikers'] - df['avg_grinders']) / df['avg_grinders'].abs()) * 100,
            0
        )
        title = "Percentage Difference Heatmap Across Time Lags"
        colorbar_title = "% Diff"
    elif metric == 'absolute_difference':
        if 'difference' not in df.columns:
            return None
        df['value'] = df['difference']
        title = "Absolute Difference Heatmap Across Time Lags"
        colorbar_title = "Abs Diff"
    elif metric == 'ratio':
        if 'ratio' not in df.columns:
            return None
        df['value'] = df['ratio']
        title = "Ratio Heatmap Across Time Lags (Spiker/Grinder)"
        colorbar_title = "Ratio"
    
    # Remove infinite values
    df = df[np.isfinite(df['value'])]
    
    if df.empty:
        return None
    
    # Find top indicators by average absolute value across lags
    indicator_scores = df.groupby('indicator')['value'].apply(lambda x: x.abs().mean())
    top_indicators = indicator_scores.nlargest(top_n).index.tolist()
    
    df = df[df['indicator'].isin(top_indicators)]
    
    # Pivot for heatmap
    pivot_df = df.pivot(index='indicator', columns='time_lag', values='value')
    
    # Sort columns in correct time lag order
    available_lags = [lag for lag in TIME_LAG_ORDER if lag in pivot_df.columns]
    other_lags = [lag for lag in pivot_df.columns if lag not in TIME_LAG_ORDER]
    pivot_df = pivot_df[available_lags + other_lags]
    
    # Sort by average absolute value
    pivot_df['_sort'] = pivot_df.abs().mean(axis=1)
    pivot_df = pivot_df.sort_values('_sort', ascending=False).drop(columns=['_sort'])
    
    if pivot_df.empty:
        return None
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='RdBu_r',
        zmid=0,
        text=pivot_df.values,
        texttemplate='%{text:.1f}',
        textfont={"size": 10},
        colorbar=dict(title=colorbar_title),
        hovertemplate='<b>%{y}</b><br>%{x}<br>Value: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"<b>{title}</b><br><sub>Top {len(pivot_df)} indicators</sub>",
        xaxis_title="Time Lag",
        yaxis_title="Indicator",
        height=max(600, len(pivot_df) * 25),
        xaxis={'side': 'top'}
    )
    
    return fig


def create_scatter_comparison(analysis_df, time_lag='T-1', enabled_categories=None):
    """Create scatter plot comparing spiker vs grinder averages"""
    if analysis_df.empty or 'time_lag' not in analysis_df.columns:
        return None
    
    df = analysis_df[analysis_df['time_lag'] == time_lag].copy()
    
    # Filter by categories
    if enabled_categories:
        df = filter_indicators_by_categories(df, enabled_categories)
    
    if df.empty:
        return None
    
    # Remove rows where either value is null
    df = df[df['avg_spikers'].notna() & df['avg_grinders'].notna()]
    
    if df.empty:
        return None
    
    # Add category
    df['category'] = df['indicator'].apply(get_indicator_category)
    
    # Create scatter
    fig = px.scatter(
        df,
        x='avg_grinders',
        y='avg_spikers',
        hover_data=['indicator', 'category'],
        color='category',
        title=f"<b>Spiker vs Grinder Averages ({time_lag})</b>",
        labels={
            'avg_grinders': 'Grinder Average',
            'avg_spikers': 'Spiker Average'
        }
    )
    
    # Add diagonal line (where spiker = grinder)
    min_val = min(df['avg_grinders'].min(), df['avg_spikers'].min())
    max_val = max(df['avg_grinders'].max(), df['avg_spikers'].max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        height=600,
        xaxis_title="Grinder Average",
        yaxis_title="Spiker Average"
    )
    
    return fig


def create_category_summary(analysis_df, time_lag='T-1', enabled_categories=None):
    """Create summary chart by category"""
    if analysis_df.empty or 'time_lag' not in analysis_df.columns:
        return None
    
    df = analysis_df[analysis_df['time_lag'] == time_lag].copy()
    
    # Filter by categories
    if enabled_categories:
        df = filter_indicators_by_categories(df, enabled_categories)
    
    if df.empty:
        return None
    
    # Add category
    df['category'] = df['indicator'].apply(get_indicator_category)
    
    # Calculate average percentage difference per category
    df['pct_difference'] = np.where(
        df['avg_grinders'].abs() > 0.01,
        ((df['avg_spikers'] - df['avg_grinders']) / df['avg_grinders'].abs()) * 100,
        0
    )
    
    df = df[np.isfinite(df['pct_difference'])]
    
    if df.empty:
        return None
    
    category_summary = df.groupby('category').agg({
        'pct_difference': ['mean', 'std', 'count'],
        'indicator': 'count'
    }).reset_index()
    
    category_summary.columns = ['category', 'avg_pct_diff', 'std_pct_diff', 'count', 'num_indicators']
    category_summary = category_summary.sort_values('avg_pct_diff', key=abs, ascending=False)
    
    # Create bar chart
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in category_summary['avg_pct_diff']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=category_summary['category'],
        y=category_summary['avg_pct_diff'],
        error_y=dict(type='data', array=category_summary['std_pct_diff']),
        marker_color=colors,
        text=[f'{x:+.1f}%' for x in category_summary['avg_pct_diff']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Avg % Diff: %{y:.1f}%<br>Std Dev: %{error_y.array:.1f}%<br>Indicators: %{customdata}<extra></extra>',
        customdata=category_summary['num_indicators']
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=2)
    
    fig.update_layout(
        title=f"<b>Average % Difference by Indicator Category ({time_lag})</b><br>" +
              "<sub>Error bars show standard deviation | Red = Spikers Higher | Blue = Grinders Higher</sub>",
        xaxis_title="Category",
        yaxis_title="Average % Difference",
        height=500,
        showlegend=False
    )
    
    return fig


def create_distribution_comparison(raw_data_df, indicator, time_lag='T-1'):
    """Create distribution comparison for a specific indicator"""
    if raw_data_df.empty or indicator not in raw_data_df.columns:
        return None
    
    df = raw_data_df.copy()
    
    # Remove nulls
    df = df[df[indicator].notna()]
    
    if df.empty or 'event_type' not in df.columns:
        return None
    
    spikers = df[df['event_type'] == 'Spiker'][indicator]
    grinders = df[df['event_type'] == 'Grinder'][indicator]
    
    if len(spikers) == 0 and len(grinders) == 0:
        return None
    
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
        title=f"<b>Distribution Comparison: {indicator} ({time_lag})</b>",
        xaxis_title=indicator,
        yaxis_title="Count",
        barmode='overlay',
        height=500,
        legend=dict(x=0.8, y=0.95)
    )
    
    return fig


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    """Main dashboard"""
    
    st.title("🎯 Stock Pattern Analysis Dashboard")
    st.markdown("**Discover what distinguishes explosive movers from steady grinders**")
    
    # Sidebar - Global settings only
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
    if not analysis_df.empty and 'time_lag' in analysis_df.columns:
        available_lags = sort_time_lags(analysis_df['time_lag'].unique().tolist())
        
        # Tab layout
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Key Discriminators",
            "🔥 Heatmaps",
            "📊 Distributions",
            "📈 Category Analysis",
            "📋 Data Explorer"
        ])
        
        # TAB 1: Key Discriminators
        with tab1:
            st.header("Top Discriminating Indicators")
            
            # Controls
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                selected_lag_disc = st.selectbox("Time Period:", available_lags, index=0, key='disc_lag')
            with col2:
                top_n_disc = st.slider("Number of indicators:", 10, 50, 20, key='disc_n')
            with col3:
                chart_type = st.selectbox("Metric:", ["% Difference", "Absolute Difference"], key='disc_metric')
            
            # Category filter for this chart
            enabled_categories_disc = create_category_selector("disc")
            
            if chart_type == "% Difference":
                fig = create_percentage_difference_chart(
                    analysis_df, selected_lag_disc, top_n_disc, enabled_categories_disc
                )
            else:
                fig = create_absolute_difference_chart(
                    analysis_df, selected_lag_disc, top_n_disc, enabled_categories_disc
                )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for selected filters")
            
            # Scatter plot
            st.subheader("Spiker vs Grinder Comparison")
            enabled_categories_scatter = create_category_selector("scatter")
            
            fig_scatter = create_scatter_comparison(analysis_df, selected_lag_disc, enabled_categories_scatter)
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # TAB 2: Heatmaps
        with tab2:
            st.header("Indicator Behavior Across Time Lags")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                heatmap_metric = st.selectbox(
                    "Metric:",
                    ["% Difference", "Absolute Difference", "Ratio"],
                    key='heatmap_metric'
                )
            with col2:
                heatmap_n = st.slider("Number of indicators:", 10, 50, 25, key='heatmap_n')
            
            # Category filter for heatmap
            enabled_categories_heatmap = create_category_selector("heatmap")
            
            metric_map = {
                "% Difference": "pct_difference",
                "Absolute Difference": "absolute_difference",
                "Ratio": "ratio"
            }
            
            fig_heatmap = create_heatmap_across_time_lags(
                analysis_df,
                metric=metric_map[heatmap_metric],
                top_n=heatmap_n,
                enabled_categories=enabled_categories_heatmap
            )
            
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("No data available for selected filters")
        
        # TAB 3: Distributions
        with tab3:
            st.header("Indicator Distribution Comparison")
            
            if raw_data_dict:
                col1, col2 = st.columns([1, 1])
                with col1:
                    dist_lag = st.selectbox("Time Period:", list(raw_data_dict.keys()), key='dist_lag')
                
                # Category filter for distributions
                enabled_categories_dist = create_category_selector("dist")
                
                with col2:
                    # Get available indicators for selected lag
                    if dist_lag in raw_data_dict:
                        lag_df = raw_data_dict[dist_lag]
                        exclude_cols = ['id', 'symbol', 'event_date', 'event_type', 'exchange', 'time_lag', 'created_at', 'updated_at']
                        available_indicators = [col for col in lag_df.columns if col not in exclude_cols]
                        
                        # Filter by enabled categories
                        if enabled_categories_dist:
                            enabled_indicators = []
                            for category, info in INDICATOR_CATEGORIES.items():
                                if category in enabled_categories_dist:
                                    enabled_indicators.extend([ind.lower() for ind in info["indicators"]])
                            available_indicators = [ind for ind in available_indicators if ind.lower() in enabled_indicators]
                        
                        if available_indicators:
                            dist_indicator = st.selectbox("Indicator:", sorted(available_indicators), key='dist_ind')
                            
                            fig_dist = create_distribution_comparison(lag_df, dist_indicator, dist_lag)
                            if fig_dist:
                                st.plotly_chart(fig_dist, use_container_width=True)
                                
                                # Show statistics
                                spikers = lag_df[lag_df['event_type'] == 'Spiker'][dist_indicator].dropna()
                                grinders = lag_df[lag_df['event_type'] == 'Grinder'][dist_indicator].dropna()
                                
                                if len(spikers) > 0 and len(grinders) > 0:
                                    st.subheader("Statistical Summary")
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Spiker Mean", f"{spikers.mean():.3f}")
                                        st.metric("Spiker Median", f"{spikers.median():.3f}")
                                    
                                    with col2:
                                        st.metric("Grinder Mean", f"{grinders.mean():.3f}")
                                        st.metric("Grinder Median", f"{grinders.median():.3f}")
                                    
                                    with col3:
                                        # T-test
                                        t_stat, p_value = stats.ttest_ind(spikers, grinders, equal_var=False)
                                        st.metric("T-Statistic", f"{t_stat:.3f}")
                                        st.metric("P-Value", f"{p_value:.4f}")
                                        
                                        if p_value < 0.05:
                                            st.success("✓ Statistically significant (p < 0.05)")
                                        else:
                                            st.info("Not statistically significant")
                        else:
                            st.info("No indicators available for selected categories")
            else:
                st.info("Raw data not available")
        
        # TAB 4: Category Analysis
        with tab4:
            st.header("Analysis by Indicator Category")
            
            cat_lag = st.selectbox("Time Period:", available_lags, key='cat_lag')
            
            # Category filter for category analysis
            enabled_categories_cat = create_category_selector("cat")
            
            fig_cat = create_category_summary(analysis_df, cat_lag, enabled_categories_cat)
            if fig_cat:
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("No data available for selected filters")
        
        # TAB 5: Data Explorer
        with tab5:
            st.header("Raw Data Explorer")
            
            subtab1, subtab2, subtab3 = st.tabs(["Candidates", "Analysis", "Raw Data"])
            
            with subtab1:
                if not candidates_df.empty:
                    st.dataframe(candidates_df, use_container_width=True, height=600)
                    st.download_button(
                        "📥 Download Candidates CSV",
                        candidates_df.to_csv(index=False),
                        "candidates.csv",
                        "text/csv"
                    )
            
            with subtab2:
                if not analysis_df.empty:
                    st.dataframe(analysis_df, use_container_width=True, height=600)
                    st.download_button(
                        "📥 Download Analysis CSV",
                        analysis_df.to_csv(index=False),
                        "analysis.csv",
                        "text/csv"
                    )
            
            with subtab3:
                if raw_data_dict:
                    raw_lag = st.selectbox("Select time lag:", list(raw_data_dict.keys()), key='raw_lag')
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
