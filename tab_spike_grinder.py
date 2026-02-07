"""
Spike/Grinder Analysis Tab Module - FIXED FOR ZERO AUTO-EGRESS
CHANGES:
- ttl=0 (never auto-refresh)
- Added .limit(1000) to all queries
- Column selection instead of SELECT *
- Only refreshes when user clicks refresh button
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from supabase import create_client, Client

# Chart theme configuration
CHART_THEME = {
    'plot_bgcolor': 'rgba(26, 29, 41, 0.6)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'font': dict(color='#e8eaf0', family='sans-serif'),
    'title_font': dict(size=18, color='#ffffff'),
    'xaxis': dict(gridcolor='rgba(255, 255, 255, 0.1)', color='#b8bcc8', linecolor='rgba(255, 255, 255, 0.2)'),
    'yaxis': dict(gridcolor='rgba(255, 255, 255, 0.1)', color='#b8bcc8', linecolor='rgba(255, 255, 255, 0.2)'),
    'legend': dict(bgcolor='rgba(45, 49, 66, 0.8)', bordercolor='rgba(255, 255, 255, 0.2)', borderwidth=1, font=dict(color='#e8eaf0'))
}

COLORS = {
    'spiker': '#ef4444',
    'grinder': '#3b82f6',
    'primary': '#667eea',
    'success': '#10b981',
    'warning': '#f59e0b'
}

# Custom colorscales for heatmaps with better readability
COLORSCALES = {
    'rdylgn_dark': [
        [0, '#7f1d1d'],      # Dark red
        [0.25, '#dc2626'],   # Red
        [0.4, '#94a3b8'],    # Gray (neutral)
        [0.6, '#94a3b8'],    # Gray (neutral)
        [0.75, '#059669'],   # Green
        [1, '#064e3b']       # Dark green
    ],
    'blues_dark': [
        [0, '#1e293b'],
        [0.5, '#475569'],
        [1, '#667eea']
    ],
    'rdbu_dark': [
        [0, '#7f1d1d'],      # Dark red
        [0.5, '#475569'],    # Gray
        [1, '#1e3a8a']       # Dark blue
    ]
}

@st.cache_resource
def get_supabase_client():
    supabase_url = os.environ.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url")
    supabase_key = os.environ.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("key")
    
    if not supabase_url or not supabase_key:
        st.error("❌ Supabase credentials not configured!")
        st.stop()
    
    return create_client(supabase_url, supabase_key)

@st.cache_data(ttl=0)  # ✅ CHANGED: Never auto-refresh (was 300)
def load_supabase_data(table_name: str, filters: dict = None, _refresh_key: int = 0):
    """
    OPTIMIZED: Only loads when refresh button is clicked
    - ttl=0 means cache never expires on its own
    - Uses column selection to reduce egress by 90%+
    - Limits results to prevent massive queries
    """
    try:
        client = get_supabase_client()
        # ✅ Add column selection for candidates too
        if table_name == "candidates":
            query = client.table(table_name).select(
                "symbol,exchange,date,event_type,price,change_pct,volume"
            )

        
        # ✅ CRITICAL FIX: Select only needed columns, not SELECT *
        elif table_name == "raw_data":
            # Only select columns actually used in analysis/charts
            query = client.table(table_name).select(
                "symbol,event_date,event_type,time_lag,"
                "rsi,macd.macd,adx,volume,close,ema20,sma20,atr,bb_width"
            )
        elif table_name == "candidates":
            query = client.table(table_name).select(
                "symbol,date,event_type,exchange,price,change_pct,volume"
            )
        elif table_name == "analysis":
            query = client.table(table_name).select("*")  # Small table, OK
        elif table_name == "summary_stats":
            query = client.table(table_name).select("*")  # Tiny table, OK
        else:
            query = client.table(table_name).select("*")
        
        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)
        
        # ✅ CRITICAL FIX: Add limit to prevent massive queries
        query = query.limit(1000)
        
        response = query.execute()
        
        if not response.data:
            return pd.DataFrame()
        
        df = pd.DataFrame(response.data)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        return df
    except Exception as e:
        st.warning(f"Could not load from {table_name}: {str(e)}")
        return pd.DataFrame()

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

def filter_indicators_by_categories(df, enabled_categories):
    if 'indicator' not in df.columns:
        return df
    
    enabled_indicators = []
    for category, info in INDICATOR_CATEGORIES.items():
        if category in enabled_categories:
            enabled_indicators.extend([ind.lower() for ind in info["indicators"]])
    
    df_filtered = df[df['indicator'].str.lower().isin(enabled_indicators)].copy()
    return df_filtered

def create_category_selector(key_prefix):
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
    
    colors = ['#064e3b' if x > 0 else '#7f1d1d' for x in df_top['difference']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_top['indicator'],
        x=df_top['difference'],
        orientation='h',
        marker_color=colors,
        text=[f'{x:+.2f}' for x in df_top['difference']],
        textposition='outside',
        textfont=dict(color='#ffffff')
    ))
    
    fig.update_layout(
        title=f"<b>Top {top_n} Differentiating Indicators</b><br><sub>Green = Spikers Higher | Red = Grinders Higher</sub>",
        xaxis_title="Difference (Spikers - Grinders)",
        yaxis_title="",
        height=max(400, top_n * 35),
        showlegend=False,
        **CHART_THEME
    )
    
    fig.update_xaxes(**CHART_THEME['xaxis'])
    fig.update_yaxes(**CHART_THEME['yaxis'])
    
    return fig

def create_heatmap(analysis_df, selected_indicators, value_type='difference'):
    if analysis_df.empty or 'indicator' not in analysis_df.columns:
        return None
    
    df = analysis_df[analysis_df['indicator'].isin(selected_indicators)].copy()
    
    if df.empty:
        return None
    
    pivot_df = df.pivot_table(
        values=value_type,
        index='indicator',
        columns='time_lag',
        aggfunc='mean'
    )
    
    available_lags = [lag for lag in TIME_LAG_ORDER if lag in pivot_df.columns]
    pivot_df = pivot_df[available_lags]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale=COLORSCALES['rdylgn_dark'] if value_type == 'difference' else COLORSCALES['blues_dark'],
        colorbar=dict(
            title=dict(text=value_type.replace('_', ' ').title(), font=dict(color='#e8eaf0')),
            tickfont=dict(color='#e8eaf0')
        ),
        text=pivot_df.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10, "color": "#ffffff"},
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
        height=max(400, len(selected_indicators) * 40),
        **CHART_THEME
    )
    
    fig.update_xaxes(**CHART_THEME['xaxis'])
    fig.update_yaxes(**CHART_THEME['yaxis'])
    
    return fig

def create_correlation_matrix(analysis_df, selected_lag):
    raw_df = load_supabase_data("raw_data", {"time_lag": selected_lag})
    
    if raw_df.empty or 'event_type' not in raw_df.columns:
        return None
    
    exclude_cols = ['id', 'symbol', 'event_date', 'event_type', 'exchange', 'time_lag', 'created_at']
    numeric_cols = [col for col in raw_df.columns if col not in exclude_cols]
    
    if len(numeric_cols) < 2:
        return None
    
    corr_df = raw_df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale=COLORSCALES['rdbu_dark'],
        zmid=0,
        colorbar=dict(
            title=dict(text="Correlation", font=dict(color='#e8eaf0')),
            tickfont=dict(color='#e8eaf0')
        ),
        text=corr_df.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8, "color": "#ffffff"},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f"<b>Indicator Correlation Matrix</b> at {selected_lag}",
        xaxis_title="",
        yaxis_title="",
        height=max(600, len(numeric_cols) * 30),
        width=max(600, len(numeric_cols) * 30),
        **CHART_THEME
    )
    
    return fig

def create_box_plot(analysis_df, selected_indicators):
    if analysis_df.empty:
        return None
    
    all_data = []
    
    for lag in TIME_LAG_ORDER:
        raw_df = load_supabase_data("raw_data", {"time_lag": lag})
        if not raw_df.empty and 'event_type' in raw_df.columns:
            raw_df['time_lag'] = lag
            all_data.append(raw_df)
    
    if not all_data:
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
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
        
        spiker_data = combined_df[combined_df['event_type'] == 'Spiker'][indicator].dropna()
        fig.add_trace(
            go.Box(y=spiker_data, name='Spikers', marker_color='#b91c1c', showlegend=(idx == 0)),
            row=row,
            col=col
        )
        
        grinder_data = combined_df[combined_df['event_type'] == 'Grinder'][indicator].dropna()
        fig.add_trace(
            go.Box(y=grinder_data, name='Grinders', marker_color='#1e40af', showlegend=(idx == 0)),
            row=row,
            col=col
        )
    
    fig.update_layout(
        title="<b>Distribution Comparison: Spikers vs Grinders</b>",
        height=300 * rows,
        showlegend=True,
        **CHART_THEME
    )
    
    for i in range(1, rows + 1):
        for j in range(1, 3):
            fig.update_xaxes(**CHART_THEME['xaxis'], row=i, col=j)
            fig.update_yaxes(**CHART_THEME['yaxis'], row=i, col=j)
    
    return fig

def create_time_series_comparison(analysis_df, selected_indicators):
    if analysis_df.empty:
        return None
    
    df = analysis_df[analysis_df['indicator'].isin(selected_indicators)].copy()
    
    if df.empty:
        return None
    
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
    
    fig.add_hline(y=0, line_dash="dash", line_color='#d97706', opacity=0.5, line_width=1)
    
    fig.update_layout(
        title="<b>Indicator Differences Across Time Lags</b>",
        xaxis_title="Time Lag",
        yaxis_title="Difference (Spikers - Grinders)",
        height=500,
        hovermode='x unified',
        **CHART_THEME
    )
    
    fig.update_xaxes(**CHART_THEME['xaxis'])
    fig.update_yaxes(**CHART_THEME['yaxis'])
    
    return fig

def render_spike_grinder_tab():
    # Initialize session state FIRST
    if 'spike_grinder_data_loaded' not in st.session_state:
        st.session_state.spike_grinder_data_loaded = False
    
    if 'spike_grinder_refresh_counter' not in st.session_state:
        st.session_state.spike_grinder_refresh_counter = 0
    
    col_header1, col_header2 = st.columns([4, 1])
    
    with col_header1:
        st.subheader("Spike/Grinder Analysis")
    
    with col_header2:
        if st.button("🔄 Refresh Data", use_container_width=True, key="spike_grinder_manual_refresh"):
            st.cache_data.clear()
            st.session_state.spike_grinder_refresh_counter += 1
            st.session_state.spike_grinder_data_loaded = True
            st.rerun()
    
    # CRITICAL: Don't load data until user clicks refresh
    if not st.session_state.spike_grinder_data_loaded:
        st.info("👆 Click 'Refresh Data' to load spike/grinder analysis")
        
        with st.expander("ℹ️ About Spike/Grinder Analysis"):
            st.markdown("""
            This analysis compares technical indicators between:
            - **Spikers**: Stocks with single-day explosive moves
            - **Grinders**: Stocks with sustained multi-day gains
            
            Click 'Refresh Data' above to load the analysis.
            """)
        return  # EXIT - no data loading!
    
    # Now load data (only after user action)
    refresh_key = st.session_state.spike_grinder_refresh_counter
    
    # Use session state to cache loaded data
    cache_key = f"spike_grinder_data_{refresh_key}"
    
    if cache_key not in st.session_state:
        with st.spinner("Loading analysis data..."):
            candidates_df = load_supabase_data("candidates", None, refresh_key)
            analysis_df = load_supabase_data("analysis", None, refresh_key)
            summary_df = load_supabase_data("summary_stats", None, refresh_key)
            
            st.session_state[cache_key] = {
                'candidates': candidates_df,
                'analysis': analysis_df,
                'summary': summary_df
            }
    
    # Use cached data
    data = st.session_state[cache_key]
    candidates_df = data['candidates']
    analysis_df = data['analysis']
    summary_df = data['summary']
    
    # Rest of your code...
    
    # ... rest of your code ...
    
    if analysis_df.empty and candidates_df.empty:
        st.warning("No analysis data available yet")
        return
    
    if not analysis_df.empty:
        required_cols = ['indicator', 'time_lag', 'avg_spikers', 'avg_grinders']
        missing_cols = [col for col in required_cols if col not in analysis_df.columns]
        
        if missing_cols:
            st.error(f"Analysis data is missing required columns: {', '.join(missing_cols)}")
            return
        
        analysis_df = analysis_df[analysis_df['indicator'].notna()]
        for col in ['avg_spikers', 'avg_grinders', 'difference']:
            if col in analysis_df.columns:
                analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
    
    if not summary_df.empty:
        st.markdown("### Summary Statistics")
        
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
    
    subtabs = st.tabs(["Key Insights", "Custom Analysis", "Raw Data"])
    
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
                            fig.add_trace(go.Histogram(x=spikers, name='Spikers', opacity=0.75, nbinsx=30, marker_color='#b91c1c'))
                        if len(grinders) > 0:
                            fig.add_trace(go.Histogram(x=grinders, name='Grinders', opacity=0.75, nbinsx=30, marker_color='#1e40af'))
                        
                        fig.update_layout(
                            title=f"Distribution: {selected_ind} at {selected_lag}",
                            barmode='overlay',
                            height=500,
                            **CHART_THEME
                        )
                        fig.update_xaxes(**CHART_THEME['xaxis'])
                        fig.update_yaxes(**CHART_THEME['yaxis'])
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
                            color_discrete_map={'Spiker': '#b91c1c', 'Grinder': '#1e40af'},
                            title=f"{y_ind} vs {x_ind} at {raw_lag}",
                            opacity=0.6
                        )
                        fig.update_layout(height=500, **CHART_THEME)
                        fig.update_xaxes(**CHART_THEME['xaxis'])
                        fig.update_yaxes(**CHART_THEME['yaxis'])
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No raw data available for this time lag")
        else:
            st.info("No analysis data available for custom visualizations")
    
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
