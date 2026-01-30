"""
Stock Event Analysis Dashboard - Pattern Discovery & Insights
Enhanced version with better pattern visualization and analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from pathlib import Path
import numpy as np
from scipy import stats
import json

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
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .pattern-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
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
    .metric-good { color: #48bb78; font-weight: 600; }
    .metric-bad { color: #f56565; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_google_sheets_client():
    """Initialize Google Sheets client"""
    try:
        credentials_dict = st.secrets["google_sheets_credentials"]
        credentials = Credentials.from_service_account_info(
            credentials_dict,
            scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        )
    except:
        credentials_path = "credentials/google_sheets_credentials.json"
        if Path(credentials_path).exists():
            credentials = Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
            )
        else:
            st.error("❌ Google Sheets credentials not found!")
            st.stop()
    
    return gspread.authorize(credentials)


@st.cache_data(ttl=300)
def load_sheet_data(spreadsheet_id, sheet_name):
    """Load data from Google Sheets with smart parsing"""
    try:
        client = get_google_sheets_client()
        spreadsheet = client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        data = worksheet.get_all_records()
        
        if not data:
            return pd.DataFrame()
        
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading {sheet_name}: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_all_time_lags(spreadsheet_id):
    """Load all time lag sheets"""
    time_lags = {}
    for lag in [1, 3, 5, 10, 30]:
        sheet_name = f"Raw Data_T-{lag}"
        df = load_sheet_data(spreadsheet_id, sheet_name)
        if not df.empty:
            time_lags[f"T-{lag}"] = df
    return time_lags


def create_top_indicators_comparison(analysis_df, time_lag='T-1', top_n=20, use_percentage=False):
    """Create comprehensive comparison of top indicators showing actual differences"""
    if analysis_df.empty or 'Time_Lag' not in analysis_df.columns:
        return None
    
    # Filter for specific time lag
    df = analysis_df[analysis_df['Time_Lag'] == time_lag].copy()
    
    if df.empty:
        return None
    
    # Calculate absolute difference and sort
    df['Abs_Difference'] = df['Difference'].abs()
    
    # Calculate percentage difference for better cross-indicator comparison
    df['Pct_Difference'] = np.where(
        df['AVG_Grinders'].abs() > 0,
        (df['Difference'] / df['AVG_Grinders'].abs()) * 100,
        0
    )
    
    df = df.nlargest(top_n, 'Abs_Difference')
    
    # Create side-by-side comparison
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'<b>{"Percentage Difference" if use_percentage else "Average Values Comparison"} ({time_lag})</b>',
            f'<b>{"Relative Difference (%)" if use_percentage else "Absolute Difference"}</b>'
        ),
        column_widths=[0.6, 0.4],
        horizontal_spacing=0.12
    )
    
    # Sort by difference for better visualization
    df = df.sort_values('Difference')
    
    if use_percentage:
        # Show percentage difference from grinder baseline
        spiker_pct = np.where(
            df['AVG_Grinders'].abs() > 0,
            ((df['AVG_Spikers'] - df['AVG_Grinders']) / df['AVG_Grinders'].abs()) * 100,
            0
        )
        
        fig.add_trace(
            go.Bar(
                name='% Difference',
                y=df['Indicator'],
                x=spiker_pct,
                orientation='h',
                marker_color=['#e74c3c' if x > 0 else '#3498db' for x in spiker_pct],
                text=[f'{x:.1f}%' for x in spiker_pct],
                textposition='outside',
                textfont=dict(size=10)
            ),
            row=1, col=1
        )
        
        # Right chart: Relative importance score
        importance = df['Abs_Difference'] / df['AVG_Grinders'].abs() * 100
        colors = ['#27ae60' if x > 0 else '#e67e22' for x in df['Difference']]
        
        fig.add_trace(
            go.Bar(
                y=df['Indicator'],
                x=importance,
                orientation='h',
                marker_color=colors,
                text=[f'{x:.1f}%' for x in importance],
                textposition='outside',
                showlegend=False,
                textfont=dict(size=10)
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="% Difference from Grinder Baseline", row=1, col=1)
        fig.update_xaxes(title_text="Importance Score (%)", row=1, col=2)
        
    else:
        # Left chart: Side-by-side bars with formatted values
        spiker_text = []
        grinder_text = []
        for s, g in zip(df['AVG_Spikers'], df['AVG_Grinders']):
            spiker_text.append(format_number(s))
            grinder_text.append(format_number(g))
        
        fig.add_trace(
            go.Bar(
                name='Spikers',
                y=df['Indicator'],
                x=df['AVG_Spikers'],
                orientation='h',
                marker_color='#e74c3c',
                text=spiker_text,
                textposition='outside',
                textfont=dict(size=10)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                name='Grinders',
                y=df['Indicator'],
                x=df['AVG_Grinders'],
                orientation='h',
                marker_color='#3498db',
                text=grinder_text,
                textposition='outside',
                textfont=dict(size=10)
            ),
            row=1, col=1
        )
        
        # Right chart: Difference bars (colored by direction) with formatted values
        colors = ['#27ae60' if x > 0 else '#e67e22' for x in df['Difference']]
        diff_text = [format_number(x) for x in df['Difference']]
        
        fig.add_trace(
            go.Bar(
                y=df['Indicator'],
                x=df['Difference'],
                orientation='h',
                marker_color=colors,
                text=diff_text,
                textposition='outside',
                showlegend=False,
                textfont=dict(size=10)
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Value", row=1, col=1)
        fig.update_xaxes(title_text="Difference", row=1, col=2)
    
    # Add zero line to difference chart
    fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=2)
    
    fig.update_yaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="", row=1, col=2, showticklabels=False)
    
    fig.update_layout(
        height=max(600, top_n * 25),
        showlegend=True if not use_percentage else False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        barmode='group',
        title_text=f"<b>Top {top_n} Most Discriminating Indicators at {time_lag}</b><br>" +
                   "<sub>Green = Spikers higher, Orange = Grinders higher</sub>",
        title_x=0.5,
        title_font_size=16
    )
    
    return fig


def format_number(num):
    """Format large numbers with K/M/B suffixes"""
    if pd.isna(num):
        return "N/A"
    
    num = float(num)
    
    if abs(num) >= 1_000_000_000:
        return f'{num/1_000_000_000:.2f}B'
    elif abs(num) >= 1_000_000:
        return f'{num/1_000_000:.2f}M'
    elif abs(num) >= 1_000:
        return f'{num/1_000:.2f}K'
    else:
        return f'{num:.2f}'


def create_pattern_strength_chart(analysis_df, time_lag='T-1', top_n=15):
    """Create chart showing pattern strength and consistency"""
    if analysis_df.empty or 'Time_Lag' not in analysis_df.columns:
        return None
    
    df = analysis_df[analysis_df['Time_Lag'] == time_lag].copy()
    
    if df.empty:
        return None
    
    # Calculate relative difference (percentage)
    df['Pct_Difference'] = np.where(
        df['AVG_Grinders'].abs() > 0,
        (df['Difference'] / df['AVG_Grinders'].abs()) * 100,
        0
    )
    
    # Calculate strength score combining absolute and relative
    df['Strength'] = df['Difference'].abs() + df['Pct_Difference'].abs() * 0.5
    df = df.nlargest(top_n, 'Strength')
    df = df.sort_values('Strength')
    
    fig = go.Figure()
    
    # Pattern strength bar
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in df['Difference']]
    
    fig.add_trace(go.Bar(
        y=df['Indicator'],
        x=df['Strength'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='#2c3e50', width=1)
        ),
        text=df['Strength'].round(1),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>' +
                      'Strength Score: %{x:.1f}<br>' +
                      'Difference: %{customdata[0]:.2f}<br>' +
                      'Pct Diff: %{customdata[1]:.1f}%<br>' +
                      '<extra></extra>',
        customdata=np.column_stack((df['Difference'], df['Pct_Difference']))
    ))
    
    fig.update_layout(
        title=f"<b>Pattern Strength Score ({time_lag})</b><br>" +
              "<sub>Higher score = stronger discrimination between Spikers and Grinders</sub>",
        xaxis_title="Strength Score",
        yaxis_title="",
        height=max(500, top_n * 30),
        showlegend=False,
        title_x=0.5
    )
    
    return fig


def create_time_evolution_heatmap(analysis_df, top_n=25):
    """Create enhanced heatmap showing how patterns evolve over time"""
    if analysis_df.empty or 'Time_Lag' not in analysis_df.columns:
        return None
    
    # Get top indicators from T-1 (most recent)
    t1_data = analysis_df[analysis_df['Time_Lag'] == 'T-1'].copy()
    if t1_data.empty:
        return None
    
    t1_data['Abs_Diff'] = t1_data['Difference'].abs()
    top_indicators = t1_data.nlargest(top_n, 'Abs_Diff')['Indicator'].tolist()
    
    # Get all available time lags
    time_lags = sorted(analysis_df['Time_Lag'].unique(), 
                      key=lambda x: int(x.split('-')[1]))
    
    # Build heatmap data
    heatmap_data = []
    annotations = []
    
    for indicator in top_indicators:
        row_data = []
        for lag in time_lags:
            lag_value = analysis_df[
                (analysis_df['Time_Lag'] == lag) & 
                (analysis_df['Indicator'] == indicator)
            ]
            if not lag_value.empty:
                value = lag_value.iloc[0]['Difference']
                row_data.append(value)
            else:
                row_data.append(0)
        heatmap_data.append(row_data)
    
    # Calculate color scale range
    all_values = np.array(heatmap_data).flatten()
    max_abs_val = np.max(np.abs(all_values))
    
    # Create annotations with values
    for i, row in enumerate(heatmap_data):
        for j, val in enumerate(row):
            # Format large numbers with K/M suffix
            if abs(val) >= 1000000:
                text = f'{val/1000000:.1f}M'
            elif abs(val) >= 1000:
                text = f'{val/1000:.1f}K'
            else:
                text = f'{val:.1f}'
            
            annotations.append(
                dict(
                    x=j, y=i,
                    text=text,
                    showarrow=False,
                    font=dict(size=9, color='white' if abs(val) > np.std(all_values) else 'black')
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=time_lags,
        y=top_indicators,
        colorscale=[
            [0, '#3498db'],      # Blue (Grinders higher)
            [0.5, '#ecf0f1'],    # Light gray (neutral)
            [1, '#e74c3c']       # Red (Spikers higher)
        ],
        zmid=0,
        colorbar=dict(
            title=dict(text="Difference<br>(Spiker - Grinder)", side="right")
        ),
        hovertemplate='<b>%{y}</b><br>%{x}<br>Difference: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="<b>Indicator Evolution Across Time Lags</b><br>" +
              "<sub>Red = Spikers higher | Blue = Grinders higher | Track consistency across time</sub>",
        xaxis_title="Days Before Event",
        yaxis_title="",
        height=max(700, top_n * 25),
        annotations=annotations,
        title_x=0.5
    )
    
    return fig


def create_dual_distribution_comparison(raw_data_dict, indicator, time_lags=['T-1', 'T-3', 'T-5']):
    """Create comprehensive distribution comparison across time lags"""
    if not raw_data_dict:
        return None
    
    # Filter to available time lags
    available_lags = [lag for lag in time_lags if lag in raw_data_dict]
    
    if not available_lags:
        return None
    
    fig = make_subplots(
        rows=len(available_lags), cols=1,
        subplot_titles=[f'<b>{lag}</b>' for lag in available_lags],
        vertical_spacing=0.08,
        row_heights=[1.0/len(available_lags)] * len(available_lags)
    )
    
    for idx, lag in enumerate(available_lags, 1):
        df = raw_data_dict[lag]
        
        if indicator not in df.columns or 'Event_Type' not in df.columns:
            continue
        
        # Get data for each type
        spikers = df[df['Event_Type'] == 'Spiker'][indicator].dropna()
        grinders = df[df['Event_Type'] == 'Grinder'][indicator].dropna()
        
        if len(spikers) == 0 or len(grinders) == 0:
            continue
        
        # Add violin plots
        fig.add_trace(
            go.Violin(
                x=spikers,
                name='Spikers',
                line_color='#e74c3c',
                fillcolor='rgba(231, 76, 60, 0.5)',
                showlegend=(idx == 1),
                box_visible=True,
                meanline_visible=True,
                orientation='h'
            ),
            row=idx, col=1
        )
        
        fig.add_trace(
            go.Violin(
                x=grinders,
                name='Grinders',
                line_color='#3498db',
                fillcolor='rgba(52, 152, 219, 0.5)',
                showlegend=(idx == 1),
                box_visible=True,
                meanline_visible=True,
                orientation='h'
            ),
            row=idx, col=1
        )
        
        # Update axes for this subplot
        fig.update_xaxes(title_text=indicator if idx == len(available_lags) else "", row=idx, col=1)
        fig.update_yaxes(showticklabels=True, row=idx, col=1)
    
    fig.update_layout(
        title=f"<b>{indicator} Distribution: Spikers vs Grinders</b><br>" +
              "<sub>Comparing distributions across different time lags</sub>",
        height=300 * len(available_lags),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title_x=0.5
    )
    
    return fig


def calculate_statistical_significance(raw_data_dict, indicator):
    """Calculate statistical tests across all time lags"""
    results = []
    
    for lag, df in raw_data_dict.items():
        if indicator not in df.columns or 'Event_Type' not in df.columns:
            continue
        
        spikers = df[df['Event_Type'] == 'Spiker'][indicator].dropna()
        grinders = df[df['Event_Type'] == 'Grinder'][indicator].dropna()
        
        if len(spikers) > 1 and len(grinders) > 1:
            # T-test
            t_stat, p_value = stats.ttest_ind(spikers, grinders)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(spikers)-1)*spikers.std()**2 + 
                                 (len(grinders)-1)*grinders.std()**2) / 
                                (len(spikers) + len(grinders) - 2))
            cohens_d = (spikers.mean() - grinders.mean()) / pooled_std if pooled_std > 0 else 0
            
            results.append({
                'Time_Lag': lag,
                'Spiker_Mean': spikers.mean(),
                'Grinder_Mean': grinders.mean(),
                'Difference': spikers.mean() - grinders.mean(),
                'P_Value': p_value,
                'Significant': '✅ Yes' if p_value < 0.05 else '❌ No',
                'Effect_Size': cohens_d,
                'Spiker_N': len(spikers),
                'Grinder_N': len(grinders)
            })
    
    return pd.DataFrame(results)


def generate_actionable_insights(analysis_df, threshold_strength=2.0):
    """Generate clear, actionable trading insights"""
    insights = []
    
    if analysis_df.empty or 'Time_Lag' not in analysis_df.columns:
        return insights
    
    # Focus on T-1 for immediate actionability
    t1_data = analysis_df[analysis_df['Time_Lag'] == 'T-1'].copy()
    
    if t1_data.empty:
        return insights
    
    # Calculate strength
    t1_data['Abs_Diff'] = t1_data['Difference'].abs()
    t1_data['Pct_Diff'] = np.where(
        t1_data['AVG_Grinders'].abs() > 0,
        (t1_data['Difference'] / t1_data['AVG_Grinders'].abs()) * 100,
        0
    )
    t1_data['Strength'] = t1_data['Abs_Diff'] + t1_data['Pct_Diff'].abs() * 0.5
    
    # Get strong patterns
    strong_patterns = t1_data[t1_data['Strength'] > threshold_strength].nlargest(10, 'Strength')
    
    for _, row in strong_patterns.iterrows():
        direction = "HIGHER" if row['Difference'] > 0 else "LOWER"
        magnitude = abs(row['Difference'])
        pct_diff = abs(row['Pct_Diff'])
        
        # Determine confidence level
        if pct_diff > 50 and magnitude > 5:
            confidence = "STRONG"
        elif pct_diff > 30 and magnitude > 2:
            confidence = "MODERATE"
        else:
            confidence = "WEAK"
        
        insights.append({
            'indicator': row['Indicator'],
            'direction': direction,
            'spiker_avg': row['AVG_Spikers'],
            'grinder_avg': row['AVG_Grinders'],
            'difference': magnitude,
            'pct_difference': pct_diff,
            'confidence': confidence,
            'strength': row['Strength']
        })
    
    return insights


def create_insight_cards(insights):
    """Create visual insight cards"""
    if not insights:
        st.info("📊 Run analysis to generate insights")
        return
    
    for insight in insights[:5]:
        confidence_colors = {
            'STRONG': '#27ae60',
            'MODERATE': '#f39c12',
            'WEAK': '#95a5a6'
        }
        
        color = confidence_colors.get(insight['confidence'], '#95a5a6')
        
        # Format the numbers appropriately
        spiker_formatted = format_number(insight['spiker_avg'])
        grinder_formatted = format_number(insight['grinder_avg'])
        diff_formatted = format_number(insight['difference'])
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                    color: white; padding: 20px; border-radius: 10px; margin: 10px 0;">
            <h3 style="margin: 0 0 10px 0;">🎯 {insight['indicator']}</h3>
            <p style="font-size: 1.1em; margin: 5px 0;">
                <b>Pattern:</b> Spikers show {insight['direction']} values
            </p>
            <p style="margin: 5px 0;">
                <b>Spikers avg:</b> {spiker_formatted} | <b>Grinders avg:</b> {grinder_formatted}
            </p>
            <p style="margin: 5px 0;">
                <b>Difference:</b> {diff_formatted} ({insight['pct_difference']:.1f}% relative)
            </p>
            <p style="margin: 5px 0;">
                <b>Confidence:</b> {insight['confidence']} | <b>Strength Score:</b> {insight['strength']:.1f}
            </p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main dashboard"""
    
    # Header
    st.title("🎯 Stock Pattern Analysis Dashboard")
    st.markdown("**Discover predictive patterns in explosive stock movements**")
    
    # Configuration
    with st.expander("⚙️ Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            spreadsheet_id = st.text_input(
                "Google Sheets ID",
                value="1rgJQiGk6eKc1Eks9cyn0X9ukkA3C6nk44QDMamPGwC0"
            )
        with col2:
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
    
    # Load data
    with st.spinner("Loading data..."):
        candidates_df = load_sheet_data(spreadsheet_id, "Candidates")
        analysis_df = load_sheet_data(spreadsheet_id, "Analysis")
        summary_df = load_sheet_data(spreadsheet_id, "Summary Stats")
        raw_data_dict = load_all_time_lags(spreadsheet_id)
    
    if analysis_df.empty and candidates_df.empty:
        st.warning("⚠️ No data found. Run the analysis pipeline first: `python main.py --all`")
        return
    
    # Summary metrics
    if not summary_df.empty:
        summary_dict = dict(zip(summary_df['Metric'], summary_df['Value']))
        
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
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Key Patterns", 
        "📊 Deep Dive Analysis", 
        "🔍 Statistical Testing",
        "📋 Raw Data"
    ])
    
    # TAB 1: KEY PATTERNS
    with tab1:
        st.header("🎯 Most Predictive Patterns")
        
        if not analysis_df.empty and 'Time_Lag' in analysis_df.columns:
            # Time lag selector
            available_lags = sorted(analysis_df['Time_Lag'].unique(), 
                                   key=lambda x: int(x.split('-')[1]))
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                selected_lag = st.selectbox("Time Period:", available_lags, index=0)
            with col2:
                top_n = st.slider("Number of indicators to show:", 10, 50, 20)
            with col3:
                view_mode = st.radio("View:", ["Absolute", "Percentage"], horizontal=True)
            
            st.info("💡 **Tip:** Use **Percentage view** to compare indicators with different scales (e.g., Volume vs RSI). " +
                   "Use **Absolute view** to see raw differences.")
            
            # Main comparison chart
            use_pct = (view_mode == "Percentage")
            fig1 = create_top_indicators_comparison(analysis_df, selected_lag, top_n, use_pct)
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown("---")
            
            # Pattern strength chart
            st.subheader("📊 Pattern Strength Analysis")
            fig2 = create_pattern_strength_chart(analysis_df, selected_lag, min(top_n, 15))
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("---")
            
            # Actionable insights
            st.subheader("💡 Actionable Trading Insights")
            insights = generate_actionable_insights(analysis_df)
            create_insight_cards(insights)
            
            # Add summary table
            if insights:
                st.subheader("📋 Quick Reference Table")
                
                summary_data = []
                for insight in insights[:10]:
                    summary_data.append({
                        'Indicator': insight['indicator'],
                        'Direction': '↑ Spikers' if insight['direction'] == 'HIGHER' else '↓ Spikers',
                        'Spiker Avg': format_number(insight['spiker_avg']),
                        'Grinder Avg': format_number(insight['grinder_avg']),
                        'Difference': format_number(insight['difference']),
                        '% Diff': f"{insight['pct_difference']:.1f}%",
                        'Confidence': insight['confidence']
                    })
                
                summary_df = pd.DataFrame(summary_data)
                
                # Style the dataframe
                def highlight_confidence(val):
                    if val == 'STRONG':
                        return 'background-color: #27ae60; color: white'
                    elif val == 'MODERATE':
                        return 'background-color: #f39c12; color: white'
                    else:
                        return 'background-color: #95a5a6; color: white'
                
                styled_df = summary_df.style.applymap(
                    highlight_confidence, 
                    subset=['Confidence']
                )
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                st.caption("💡 **How to use:** Focus on STRONG confidence indicators with high % Diff. " +
                          "These show the most reliable differences between Spikers and Grinders.")
            
        else:
            st.info("Analysis data not available. Run the pipeline to generate insights.")
    
    # TAB 2: DEEP DIVE
    with tab2:
        st.header("📊 Deep Dive Analysis")
        
        if not analysis_df.empty and 'Time_Lag' in analysis_df.columns:
            # Time evolution heatmap
            st.subheader("⏱️ Pattern Evolution Over Time")
            
            top_n_heatmap = st.slider("Indicators in heatmap:", 15, 40, 25, key='heatmap_n')
            fig_heatmap = create_time_evolution_heatmap(analysis_df, top_n_heatmap)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
                st.info("💡 **How to read:** Red = Spikers higher, Blue = Grinders higher. " +
                       "Consistent colors across time lags indicate reliable patterns.")
            
            st.markdown("---")
            
            # Distribution analysis
            st.subheader("📈 Distribution Analysis Across Time")
            
            if raw_data_dict:
                # Get available indicators
                first_lag_df = list(raw_data_dict.values())[0]
                metadata_cols = {'Symbol', 'Event_Date', 'Event_Type', 'Exchange', 'Time_Lag'}
                indicator_cols = [col for col in first_lag_df.columns if col not in metadata_cols]
                
                if indicator_cols:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        selected_indicator = st.selectbox(
                            "Select indicator:",
                            sorted(indicator_cols),
                            key='dist_indicator'
                        )
                    
                    with col2:
                        lag_options = list(raw_data_dict.keys())
                        selected_lags = st.multiselect(
                            "Time lags to compare:",
                            lag_options,
                            default=lag_options[:min(3, len(lag_options))]
                        )
                    
                    if selected_indicator and selected_lags:
                        # Distribution comparison
                        fig_dist = create_dual_distribution_comparison(
                            raw_data_dict, selected_indicator, selected_lags
                        )
                        if fig_dist:
                            st.plotly_chart(fig_dist, use_container_width=True)
                        
                        # Statistical summary
                        st.subheader("📊 Statistical Summary")
                        stats_df = calculate_statistical_significance(raw_data_dict, selected_indicator)
                        if not stats_df.empty:
                            # Format for display
                            display_stats = stats_df.copy()
                            display_stats['Spiker_Mean'] = display_stats['Spiker_Mean'].round(2)
                            display_stats['Grinder_Mean'] = display_stats['Grinder_Mean'].round(2)
                            display_stats['Difference'] = display_stats['Difference'].round(2)
                            display_stats['P_Value'] = display_stats['P_Value'].round(4)
                            display_stats['Effect_Size'] = display_stats['Effect_Size'].round(3)
                            
                            st.dataframe(display_stats, use_container_width=True, hide_index=True)
                            
                            st.caption("**P-Value < 0.05** indicates statistically significant difference. " +
                                     "**Effect Size** (Cohen's d): 0.2=small, 0.5=medium, 0.8=large")
        else:
            st.info("Load analysis data to view deep dive analytics")
    
    # TAB 3: STATISTICAL TESTING
    with tab3:
        st.header("🔍 Statistical Testing & Validation")
        
        if raw_data_dict and analysis_df is not None and not analysis_df.empty:
            st.subheader("📊 Cross-Indicator Comparison")
            
            # Get indicators
            first_lag_df = list(raw_data_dict.values())[0]
            metadata_cols = {'Symbol', 'Event_Date', 'Event_Type', 'Exchange', 'Time_Lag'}
            indicator_cols = [col for col in first_lag_df.columns if col not in metadata_cols]
            
            if len(indicator_cols) >= 2:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    indicator1 = st.selectbox("X-axis:", sorted(indicator_cols), key='stat_x')
                with col2:
                    indicator2 = st.selectbox("Y-axis:", sorted(indicator_cols), 
                                            index=min(1, len(indicator_cols)-1), key='stat_y')
                with col3:
                    lag_for_scatter = st.selectbox("Time lag:", list(raw_data_dict.keys()), key='stat_lag')
                
                if indicator1 and indicator2 and lag_for_scatter in raw_data_dict:
                    df = raw_data_dict[lag_for_scatter]
                    
                    if (indicator1 in df.columns and indicator2 in df.columns and 
                        'Event_Type' in df.columns):
                        
                        # Create scatter plot
                        fig = px.scatter(
                            df,
                            x=indicator1,
                            y=indicator2,
                            color='Event_Type',
                            color_discrete_map={'Spiker': '#e74c3c', 'Grinder': '#3498db'},
                            hover_data=['Symbol'] if 'Symbol' in df.columns else None,
                            title=f"{indicator1} vs {indicator2} ({lag_for_scatter})",
                            opacity=0.7,
                            marginal_x="box",
                            marginal_y="box"
                        )
                        
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate separation metrics
                        spikers = df[df['Event_Type'] == 'Spiker'][[indicator1, indicator2]].dropna()
                        grinders = df[df['Event_Type'] == 'Grinder'][[indicator1, indicator2]].dropna()
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(f"{indicator1} Difference", 
                                    f"{(spikers[indicator1].mean() - grinders[indicator1].mean()):.2f}")
                        with col2:
                            st.metric(f"{indicator2} Difference", 
                                    f"{(spikers[indicator2].mean() - grinders[indicator2].mean()):.2f}")
                        with col3:
                            # Calculate correlation difference
                            if len(spikers) > 2 and len(grinders) > 2:
                                corr_spike = spikers[indicator1].corr(spikers[indicator2])
                                corr_grind = grinders[indicator1].corr(grinders[indicator2])
                                st.metric("Correlation Difference", f"{(corr_spike - corr_grind):.3f}")
            
            st.markdown("---")
            st.subheader("🔬 Comprehensive Statistical Tests")
            
            # Select indicator for comprehensive testing
            test_indicator = st.selectbox(
                "Select indicator for full statistical analysis:",
                sorted(indicator_cols),
                key='comprehensive_test'
            )
            
            if test_indicator:
                stats_df = calculate_statistical_significance(raw_data_dict, test_indicator)
                
                if not stats_df.empty:
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    
                    # Summary interpretation
                    significant_count = (stats_df['P_Value'] < 0.05).sum()
                    total_count = len(stats_df)
                    
                    if significant_count / total_count >= 0.8:
                        st.success(f"✅ **Strong Pattern:** Statistically significant in {significant_count}/{total_count} time lags")
                    elif significant_count / total_count >= 0.5:
                        st.warning(f"⚠️ **Moderate Pattern:** Significant in {significant_count}/{total_count} time lags")
                    else:
                        st.info(f"ℹ️ **Weak Pattern:** Significant in only {significant_count}/{total_count} time lags")
        else:
            st.info("Load raw data to perform statistical testing")
    
    # TAB 4: RAW DATA
    with tab4:
        st.header("📋 Data Explorer")
        
        data_view = st.radio(
            "Select data view:",
            ["Candidates", "Analysis Results", "Raw Data (By Time Lag)"],
            horizontal=True
        )
        
        if data_view == "Candidates" and not candidates_df.empty:
            st.subheader("Candidate Events")
            st.dataframe(candidates_df, use_container_width=True, height=400)
            
            csv = candidates_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "candidates.csv", "text/csv")
        
        elif data_view == "Analysis Results" and not analysis_df.empty:
            st.subheader("Analysis Results")
            
            # Filter by time lag
            if 'Time_Lag' in analysis_df.columns:
                selected_view_lag = st.selectbox(
                    "Filter by time lag:",
                    ['All'] + sorted(analysis_df['Time_Lag'].unique())
                )
                
                if selected_view_lag != 'All':
                    display_df = analysis_df[analysis_df['Time_Lag'] == selected_view_lag]
                else:
                    display_df = analysis_df
            else:
                display_df = analysis_df
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            csv = display_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "analysis.csv", "text/csv")
        
        elif data_view == "Raw Data (By Time Lag)" and raw_data_dict:
            st.subheader("Raw Indicator Data")
            
            selected_raw_lag = st.selectbox("Select time lag:", list(raw_data_dict.keys()))
            
            if selected_raw_lag in raw_data_dict:
                raw_df = raw_data_dict[selected_raw_lag]
                st.info(f"Shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
                st.dataframe(raw_df, use_container_width=True, height=400)
                
                csv = raw_df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, f"raw_data_{selected_raw_lag}.csv", "text/csv")
        else:
            st.info("No data available for selected view")


if __name__ == "__main__":
    main()
