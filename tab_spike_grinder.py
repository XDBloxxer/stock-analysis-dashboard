"""
Spike/Grinder Analysis Tab Module
Handles all Spike/Grinder analysis functionality
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        return df
    except Exception as e:
        st.warning(f"⚠️ Could not load from {table_name}: {str(e)}")
        return pd.DataFrame()


# ============================================================================
# CONSTANTS
# ============================================================================

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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
    
    df_grouped = df.groupby('indicator').agg({
        'difference': 'mean',
        'avg_spikers': 'mean',
        'avg_grinders': 'mean'
    }).reset_index()
    
    df_grouped['abs_diff'] = df_grouped['difference'].abs()
    df_top = df_grouped.nlargest(top_n, 'abs_diff').sort_values('difference')
    
    if df_top.empty:
        return None
    
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


def create_heatmap(analysis_df, selected_indicators, value_type='difference'):
    """Create heatmap showing indicator values across time lags"""
    if analysis_df.empty or 'indicator' not in analysis_df.columns:
        return None
    
    # Filter to selected indicators
    df = analysis_df[analysis_df['indicator'].isin(selected_indicators)].copy()
    
    if df.empty:
        return None
    
    # Pivot the data
    pivot_df = df.pivot_table(
        values=value_type,
        index='indicator',
        columns='time_lag',
        aggfunc='mean'
    )
    
    # Reorder columns by time lag
    available_lags = [lag for lag in TIME_LAG_ORDER if lag in pivot_df.columns]
    pivot_df = pivot_df[available_lags]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='RdYlGn' if value_type == 'difference' else 'Viridis',
        colorbar=dict(title=value_type.replace('_', ' ').title()),
        text=pivot_df.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    title_map = {
        'difference': 'Difference (Spikers - Grinders)',
        'avg_spikers': 'Average Spiker Values',
        'avg_grinders': 'Average Grinder Values'
    }
    
    fig.update_layout(
        title=f"<b>{title_map.get(value_type, value_type)}</b> Across Time Lags",
        xaxis_title="Time Lag",
        yaxis_title="Indicator",
        height=max(400, len(selected_indicators) * 40)
    )
    
    return fig


def create_correlation_matrix(analysis_df, selected_lag):
    """Create correlation matrix for indicators at a specific time lag"""
    # Load raw data for the selected lag
    raw_df = load_supabase_data("raw_data", {"time_lag": selected_lag})
    
    if raw_df.empty or 'event_type' not in raw_df.columns:
        return None
    
    # Get numeric columns (exclude metadata)
    exclude_cols = ['id', 'symbol', 'event_date', 'event_type', 'exchange', 'time_lag', 'created_at']
    numeric_cols = [col for col in raw_df.columns if col not in exclude_cols]
    
    if len(numeric_cols) < 2:
        return None
    
    # Calculate correlation matrix
    corr_df = raw_df[numeric_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title="Correlation"),
        text=corr_df.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f"<b>Indicator Correlation Matrix</b> at {selected_lag}",
        xaxis_title="",
        yaxis_title="",
        height=max(600, len(numeric_cols) * 30),
        width=max(600, len(numeric_cols) * 30)
    )
    
    return fig


def create_box_plot(analysis_df, selected_indicators):
    """Create box plot comparing spikers vs grinders for selected indicators"""
    if analysis_df.empty:
        return None
    
    # We need raw data for box plots
    all_data = []
    
    for lag in TIME_LAG_ORDER:
        raw_df = load_supabase_data("raw_data", {"time_lag": lag})
        if not raw_df.empty and 'event_type' in raw_df.columns:
            raw_df['time_lag'] = lag
            all_data.append(raw_df)
    
    if not all_data:
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create subplots for each indicator
    from plotly.subplots import make_subplots
    
    n_indicators = len(selected_indicators)
    rows = (n_indicators + 1) // 2
    
    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=selected_indicators,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    for idx, indicator in enumerate(selected_indicators):
        if indicator not in combined_df.columns:
            continue
        
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        # Spikers
        spiker_data = combined_df[combined_df['event_type'] == 'Spiker'][indicator].dropna()
        fig.add_trace(
            go.Box(y=spiker_data, name='Spikers', marker_color='#e74c3c', showlegend=(idx == 0)),
            row=row,
            col=col
        )
        
        # Grinders
        grinder_data = combined_df[combined_df['event_type'] == 'Grinder'][indicator].dropna()
        fig.add_trace(
            go.Box(y=grinder_data, name='Grinders', marker_color='#3498db', showlegend=(idx == 0)),
            row=row,
            col=col
        )
    
    fig.update_layout(
        title="<b>Distribution Comparison: Spikers vs Grinders</b>",
        height=300 * rows,
        showlegend=True
    )
    
    return fig


def create_time_series_comparison(analysis_df, selected_indicators):
    """Create line chart showing how differences evolve across time lags"""
    if analysis_df.empty:
        return None
    
    df = analysis_df[analysis_df['indicator'].isin(selected_indicators)].copy()
    
    if df.empty:
        return None
    
    # Sort by time lag
    df['time_lag_order'] = df['time_lag'].map(
        {lag: i for i, lag in enumerate(TIME_LAG_ORDER)}
    )
    df = df.sort_values(['indicator', 'time_lag_order'])
    
    fig = go.Figure()
    
    for indicator in selected_indicators:
        ind_data = df[df['indicator'] == indicator]
        
        fig.add_trace(go.Scatter(
            x=ind_data['time_lag'],
            y=ind_data['difference'],
            mode='lines+markers',
            name=indicator,
            marker=dict(size=8),
            line=dict(width=2)
        ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="<b>Indicator Differences Across Time Lags</b>",
        xaxis_title="Time Lag",
        yaxis_title="Difference (Spikers - Grinders)",
        height=500,
        hovermode='x unified'
    )
    
    return fig


# ============================================================================
# MAIN TAB FUNCTION
# ============================================================================

def render_spike_grinder_tab():
    """Main Spike/Grinder analysis tab rendering function"""
    
    # Initialize refresh counter in session state
    if 'spike_grinder_refresh_counter' not in st.session_state:
        st.session_state.spike_grinder_refresh_counter = 0
    
    # Load data with refresh key
    refresh_key = st.session_state.spike_grinder_refresh_counter
    
    with st.spinner("Loading analysis data..."):
        candidates_df = load_supabase_data("candidates", None, refresh_key)
        analysis_df = load_supabase_data("analysis", None, refresh_key)
        summary_df = load_supabase_data("summary_stats", None, refresh_key)
    
    if analysis_df.empty and candidates_df.empty:
        st.warning("⚠️ No analysis data available yet.")
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
        col_header1, col_header2 = st.columns([4, 1])
        
        with col_header1:
            st.subheader("Summary Statistics")
        with col_header2:
            if st.button("Refresh Data", use_container_width=True, key="spike_grinder_refresh"):
                st.session_state.spike_grinder_refresh_counter += 1
                st.rerun()
        
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
    subtabs = st.tabs(["Key Insights", "Custom Analysis", "Raw Data"])
    
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
        else:
            st.info("No analysis data available for insights")
    
    # Custom Analysis
    with subtabs[1]:
        st.subheader("Build Custom Visualizations")
        
        if not analysis_df.empty:
            chart_type = st.selectbox(
                "Select chart type:",
                ["Heatmap", "Time Series", "Box Plot", "Correlation Matrix", "Indicator Distribution", "Scatter Plot"],
                key="spike_grinder_chart_type"
            )
            
            if chart_type == "Heatmap":
                st.markdown("**Create a heatmap showing indicator values across time lags**")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    available_indicators = sorted(analysis_df['indicator'].unique())
                    selected_indicators = st.multiselect(
                        "Select indicators:",
                        available_indicators,
                        default=available_indicators[:10] if len(available_indicators) >= 10 else available_indicators,
                        key="heatmap_indicators"
                    )
                
                with col2:
                    value_type = st.selectbox(
                        "Value type:",
                        ["difference", "avg_spikers", "avg_grinders"],
                        format_func=lambda x: {
                            "difference": "Difference",
                            "avg_spikers": "Spiker Avg",
                            "avg_grinders": "Grinder Avg"
                        }[x],
                        key="heatmap_value_type"
                    )
                
                if selected_indicators:
                    fig = create_heatmap(analysis_df, selected_indicators, value_type)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not create heatmap with selected indicators")
            
            elif chart_type == "Time Series":
                st.markdown("**Show how indicator differences evolve across time lags**")
                
                available_indicators = sorted(analysis_df['indicator'].unique())
                selected_indicators = st.multiselect(
                    "Select indicators:",
                    available_indicators,
                    default=available_indicators[:5] if len(available_indicators) >= 5 else available_indicators,
                    key="timeseries_indicators"
                )
                
                if selected_indicators:
                    fig = create_time_series_comparison(analysis_df, selected_indicators)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Box Plot":
                st.markdown("**Compare distributions of spikers vs grinders**")
                
                available_indicators = sorted(analysis_df['indicator'].unique())
                selected_indicators = st.multiselect(
                    "Select indicators:",
                    available_indicators,
                    default=available_indicators[:4] if len(available_indicators) >= 4 else available_indicators,
                    key="boxplot_indicators"
                )
                
                if selected_indicators:
                    with st.spinner("Loading raw data for box plots..."):
                        fig = create_box_plot(analysis_df, selected_indicators)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Could not create box plots - raw data may be unavailable")
            
            elif chart_type == "Correlation Matrix":
                st.markdown("**Explore correlations between indicators at a specific time lag**")
                
                available_lags = sorted(analysis_df['time_lag'].unique())
                selected_lag = st.selectbox("Select time lag:", available_lags, key="corr_lag")
                
                with st.spinner("Calculating correlations..."):
                    fig = create_correlation_matrix(analysis_df, selected_lag)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not create correlation matrix - data may be unavailable")
            
            elif chart_type == "Indicator Distribution":
                available_lags = sorted(analysis_df['time_lag'].unique())
                selected_lag = st.selectbox("Select time lag:", available_lags, key="dist_lag")
                
                raw_df = load_supabase_data("raw_data", {"time_lag": selected_lag}, refresh_key)
                
                if not raw_df.empty and 'event_type' in raw_df.columns:
                    exclude_cols = ['id', 'symbol', 'event_date', 'event_type', 'exchange', 'time_lag', 'created_at']
                    available_ind = [col for col in raw_df.columns if col not in exclude_cols]
                    
                    selected_ind = st.selectbox("Select indicator:", sorted(available_ind), key="dist_indicator")
                    
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
                        
                        if len(spikers) > 0 and len(grinders) > 0:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Spiker Mean", f"{spikers.mean():.3f}")
                            with col2:
                                st.metric("Grinder Mean", f"{grinders.mean():.3f}")
                else:
                    st.info("No raw data available for this time lag")
            
            elif chart_type == "Scatter Plot":
                st.info("Select two indicators to plot against each other")
                
                raw_lag = st.selectbox("Select time lag:", sorted(analysis_df['time_lag'].unique()), key='scatter_lag')
                raw_df = load_supabase_data("raw_data", {"time_lag": raw_lag}, refresh_key)
                
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
                else:
                    st.info("No raw data available for this time lag")
        else:
            st.info("No analysis data available for custom visualizations")
    
    # Raw Data
    with subtabs[2]:
        data_tabs = st.tabs(["Candidates", "Analysis Results", "Raw Indicator Data"])
        
        with data_tabs[0]:
            if not candidates_df.empty:
                st.dataframe(candidates_df, use_container_width=True, height=500)
                st.download_button("Download CSV", candidates_df.to_csv(index=False), "candidates.csv", "text/csv", key="download_candidates")
            else:
                st.info("No candidates data available")
        
        with data_tabs[1]:
            if not analysis_df.empty:
                st.dataframe(analysis_df, use_container_width=True, height=500)
                st.download_button("Download CSV", analysis_df.to_csv(index=False), "analysis.csv", "text/csv", key="download_analysis")
            else:
                st.info("No analysis data available")
        
        with data_tabs[2]:
            if not analysis_df.empty and 'time_lag' in analysis_df.columns:
                lag_selector = st.selectbox("Select time lag:", TIME_LAG_ORDER, key='raw_data_lag')
                raw_df = load_supabase_data("raw_data", {"time_lag": lag_selector})
                if not raw_df.empty:
                    st.dataframe(raw_df, use_container_width=True, height=500)
                    st.download_button("Download CSV", raw_df.to_csv(index=False), f"raw_data_{lag_selector}.csv", "text/csv", key="download_raw")
                else:
                    st.info(f"No raw data available for {lag_selector}")
            else:
                st.info("No raw data available")
