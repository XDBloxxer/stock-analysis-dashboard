"""
Stock Event Analysis Dashboard - Pattern Discovery & Insights
Focused on finding actionable patterns that predict Spikers vs Grinders
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


def calculate_predictive_power(analysis_df):
    """Calculate which indicators have the strongest predictive power"""
    if analysis_df.empty or 'Difference' not in analysis_df.columns:
        return pd.DataFrame()
    
    df = analysis_df.copy()
    
    # Calculate absolute difference and percentage difference
    df['Abs_Difference'] = df['Difference'].abs()
    df['Pct_Difference'] = np.where(
        df['AVG_Grinders'] != 0,
        (df['Difference'] / df['AVG_Grinders'].abs()) * 100,
        0
    )
    
    # Calculate a composite score combining absolute and relative difference
    df['Predictive_Score'] = (df['Abs_Difference'] * 0.6 + df['Pct_Difference'].abs() * 0.4)
    
    # Sort by predictive score
    df = df.sort_values('Predictive_Score', ascending=False)
    
    return df


def find_indicator_patterns(raw_data_df, analysis_df):
    """Find patterns: which indicators consistently separate Spikers from Grinders"""
    patterns = []
    
    if analysis_df.empty or raw_data_df.empty:
        return pd.DataFrame()
    
    # Get top discriminating indicators
    top_indicators = calculate_predictive_power(analysis_df).head(20)
    
    for _, row in top_indicators.iterrows():
        indicator = row['Indicator']
        
        if indicator not in raw_data_df.columns:
            continue
        
        # Check if Spikers consistently higher or lower
        direction = "higher" if row['Difference'] > 0 else "lower"
        magnitude = abs(row['Difference'])
        
        # Calculate consistency - what % of Spikers follow this pattern
        if 'Event_Type' in raw_data_df.columns:
            spikers = raw_data_df[raw_data_df['Event_Type'] == 'Spiker'][indicator].dropna()
            grinders = raw_data_df[raw_data_df['Event_Type'] == 'Grinder'][indicator].dropna()
            
            if len(spikers) > 0 and len(grinders) > 0:
                spiker_median = spikers.median()
                grinder_median = grinders.median()
                
                if direction == "higher":
                    consistency = (spikers > grinder_median).sum() / len(spikers) * 100
                else:
                    consistency = (spikers < grinder_median).sum() / len(spikers) * 100
                
                patterns.append({
                    'Indicator': indicator,
                    'Pattern': f"Spikers have {direction} values",
                    'Avg_Spiker': row['AVG_Spikers'],
                    'Avg_Grinder': row['AVG_Grinders'],
                    'Difference': magnitude,
                    'Consistency': consistency,
                    'Predictive_Score': row['Predictive_Score']
                })
    
    return pd.DataFrame(patterns)


def create_top_indicators_chart(analysis_df, n=15):
    """Create interactive chart of top predictive indicators"""
    if analysis_df.empty:
        return None
    
    df = calculate_predictive_power(analysis_df).head(n)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Values', 'Difference Magnitude'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Left: Comparison
    fig.add_trace(
        go.Bar(name='Spikers', x=df['Indicator'], y=df['AVG_Spikers'],
               marker_color='#e74c3c', text=df['AVG_Spikers'].round(2),
               textposition='auto'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Grinders', x=df['Indicator'], y=df['AVG_Grinders'],
               marker_color='#3498db', text=df['AVG_Grinders'].round(2),
               textposition='auto'),
        row=1, col=1
    )
    
    # Right: Differences
    colors = ['#27ae60' if x > 0 else '#e67e22' for x in df['Difference']]
    fig.add_trace(
        go.Bar(x=df['Indicator'], y=df['Difference'].abs(),
               marker_color=colors, name='Absolute Difference',
               text=df['Difference'].abs().round(2), textposition='auto',
               showlegend=False),
        row=1, col=2
    )
    
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(
        height=500,
        title_text=f"Top {n} Most Predictive Indicators (T-1)",
        showlegend=True,
        barmode='group'
    )
    
    return fig


def create_pattern_heatmap(analysis_df, time_lags=['T-1', 'T-3', 'T-5', 'T-10', 'T-30']):
    """Create heatmap showing indicator patterns across time lags"""
    if analysis_df.empty or 'Time_Lag' not in analysis_df.columns:
        return None
    
    # Get top indicators from T-1
    t1_data = analysis_df[analysis_df['Time_Lag'] == 'T-1']
    top_indicators = calculate_predictive_power(t1_data).head(15)['Indicator'].tolist()
    
    # Build heatmap data
    heatmap_data = []
    y_labels = []
    
    for indicator in top_indicators:
        row = []
        for lag in time_lags:
            lag_data = analysis_df[(analysis_df['Time_Lag'] == lag) & (analysis_df['Indicator'] == indicator)]
            if not lag_data.empty:
                row.append(lag_data.iloc[0]['Difference'])
            else:
                row.append(0)
        heatmap_data.append(row)
        y_labels.append(indicator)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=time_lags,
        y=y_labels,
        colorscale='RdBu',
        zmid=0,
        text=[[f'{val:.2f}' for val in row] for row in heatmap_data],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Difference<br>(Spiker - Grinder)")
    ))
    
    fig.update_layout(
        title="Indicator Behavior Across Time Lags",
        xaxis_title="Days Before Event",
        yaxis_title="Indicator",
        height=600
    )
    
    return fig


def create_prediction_scatter(raw_data_df, indicator1, indicator2):
    """Create scatter plot to visualize separation between Spikers and Grinders"""
    if raw_data_df.empty or indicator1 not in raw_data_df.columns or indicator2 not in raw_data_df.columns:
        return None
    
    if 'Event_Type' not in raw_data_df.columns:
        return None
    
    fig = px.scatter(
        raw_data_df,
        x=indicator1,
        y=indicator2,
        color='Event_Type',
        color_discrete_map={'Spiker': '#e74c3c', 'Grinder': '#3498db'},
        hover_data=['Symbol'] if 'Symbol' in raw_data_df.columns else None,
        title=f"{indicator1} vs {indicator2}",
        opacity=0.6,
        marginal_x="box",
        marginal_y="box"
    )
    
    fig.update_layout(height=500)
    
    return fig


def create_distribution_violin(raw_data_df, indicator):
    """Create violin plot showing distribution differences"""
    if raw_data_df.empty or indicator not in raw_data_df.columns:
        return None
    
    if 'Event_Type' not in raw_data_df.columns:
        return None
    
    fig = go.Figure()
    
    for event_type, color in [('Spiker', '#e74c3c'), ('Grinder', '#3498db')]:
        data = raw_data_df[raw_data_df['Event_Type'] == event_type][indicator].dropna()
        
        fig.add_trace(go.Violin(
            y=data,
            name=event_type,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            opacity=0.6,
            x0=event_type
        ))
    
    fig.update_layout(
        title=f"{indicator} - Distribution Comparison",
        yaxis_title="Value",
        showlegend=True,
        height=400
    )
    
    return fig


def generate_trading_insights(patterns_df, raw_data_df):
    """Generate actionable trading insights from patterns"""
    insights = []
    
    if patterns_df.empty:
        return insights
    
    # Top 5 most consistent patterns
    top_patterns = patterns_df.nlargest(5, 'Consistency')
    
    for _, pattern in top_patterns.iterrows():
        if pattern['Consistency'] > 70:  # High consistency
            insight = {
                'type': 'strong',
                'indicator': pattern['Indicator'],
                'rule': pattern['Pattern'],
                'consistency': pattern['Consistency'],
                'difference': pattern['Difference']
            }
            insights.append(insight)
        elif pattern['Consistency'] > 55:  # Moderate consistency
            insight = {
                'type': 'moderate',
                'indicator': pattern['Indicator'],
                'rule': pattern['Pattern'],
                'consistency': pattern['Consistency'],
                'difference': pattern['Difference']
            }
            insights.append(insight)
    
    return insights


def main():
    """Main dashboard"""
    
    # Header
    st.title("🎯 Stock Pattern Analysis Dashboard")
    st.markdown("**Discover patterns that predict explosive stock movements**")
    
    # Configuration in expander
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
        raw_data_df = load_sheet_data(spreadsheet_id, "Raw Data_T-1")
        analysis_df = load_sheet_data(spreadsheet_id, "Analysis")
        summary_df = load_sheet_data(spreadsheet_id, "Summary Stats")
    
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
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Key Patterns", "📊 Deep Dive", "🔍 Custom Analysis", "📋 Data"])
    
    # TAB 1: KEY PATTERNS
    with tab1:
        st.header("🎯 Most Predictive Patterns")
        
        if not analysis_df.empty:
            # Calculate patterns
            patterns_df = find_indicator_patterns(raw_data_df, analysis_df)
            
            if not patterns_df.empty:
                # Top indicators chart
                fig = create_top_indicators_chart(analysis_df, n=15)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Pattern cards
                st.subheader("💡 Actionable Insights")
                
                insights = generate_trading_insights(patterns_df, raw_data_df)
                
                if insights:
                    for insight in insights[:5]:  # Top 5 insights
                        if insight['type'] == 'strong':
                            st.markdown(f"""
                            <div class="insight-box">
                                <h3>🎯 Strong Pattern: {insight['indicator']}</h3>
                                <p style="font-size: 1.1em;">{insight['rule']}</p>
                                <p><strong>Consistency: {insight['consistency']:.1f}%</strong> | Difference: {insight['difference']:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="pattern-card">
                                <h4>📌 {insight['indicator']}</h4>
                                <p>{insight['rule']}</p>
                                <p><strong>Consistency: {insight['consistency']:.1f}%</strong> | Difference: {insight['difference']:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Detailed table
                st.subheader("📊 Pattern Details")
                
                display_df = patterns_df.head(15)[['Indicator', 'Pattern', 'Avg_Spiker', 'Avg_Grinder', 'Difference', 'Consistency']].copy()
                display_df['Avg_Spiker'] = display_df['Avg_Spiker'].round(2)
                display_df['Avg_Grinder'] = display_df['Avg_Grinder'].round(2)
                display_df['Difference'] = display_df['Difference'].round(2)
                display_df['Consistency'] = display_df['Consistency'].round(1)
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("Run analysis to discover patterns")
        else:
            st.info("No analysis data available")
    
    # TAB 2: DEEP DIVE
    with tab2:
        st.header("📊 Deep Dive Analysis")
        
        if not analysis_df.empty:
            # Time lag analysis
            st.subheader("⏱️ Pattern Evolution Across Time")
            
            if 'Time_Lag' in analysis_df.columns:
                fig = create_pattern_heatmap(analysis_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("💡 **Interpretation**: Red = Spikers higher, Blue = Grinders higher. Look for indicators that stay red/blue across all time lags for strongest patterns.")
            
            # Distribution analysis
            st.subheader("📈 Indicator Distributions")
            
            if not raw_data_df.empty:
                metadata_cols = ['Symbol', 'Event_Date', 'Event_Type', 'Exchange']
                indicator_cols = [col for col in raw_data_df.columns if col not in metadata_cols]
                
                if indicator_cols:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        selected_indicator = st.selectbox(
                            "Select indicator to analyze:",
                            sorted(indicator_cols),
                            key='dist_indicator'
                        )
                    
                    if selected_indicator:
                        fig = create_distribution_violin(raw_data_df, selected_indicator)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics
                        if 'Event_Type' in raw_data_df.columns:
                            spikers = raw_data_df[raw_data_df['Event_Type'] == 'Spiker'][selected_indicator].dropna()
                            grinders = raw_data_df[raw_data_df['Event_Type'] == 'Grinder'][selected_indicator].dropna()
                            
                            if len(spikers) > 0 and len(grinders) > 0:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Spiker Mean", f"{spikers.mean():.2f}")
                                    st.metric("Spiker Median", f"{spikers.median():.2f}")
                                
                                with col2:
                                    st.metric("Grinder Mean", f"{grinders.mean():.2f}")
                                    st.metric("Grinder Median", f"{grinders.median():.2f}")
                                
                                with col3:
                                    diff = spikers.mean() - grinders.mean()
                                    st.metric("Difference", f"{diff:.2f}", delta=f"{(diff/grinders.mean()*100):.1f}%")
                                    
                                    # T-test
                                    if len(spikers) > 1 and len(grinders) > 1:
                                        t_stat, p_value = stats.ttest_ind(spikers, grinders)
                                        significance = "✅ Significant" if p_value < 0.05 else "⚠️ Not significant"
                                        st.metric("Statistical Test", significance)
    
    # TAB 3: CUSTOM ANALYSIS
    with tab3:
        st.header("🔍 Custom Analysis")
        
        if not raw_data_df.empty:
            metadata_cols = ['Symbol', 'Event_Date', 'Event_Type', 'Exchange']
            indicator_cols = [col for col in raw_data_df.columns if col not in metadata_cols]
            
            if len(indicator_cols) >= 2:
                st.subheader("Compare Two Indicators")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    indicator1 = st.selectbox("X-axis indicator:", sorted(indicator_cols), key='x_ind')
                with col2:
                    indicator2 = st.selectbox("Y-axis indicator:", sorted(indicator_cols), index=1, key='y_ind')
                
                if indicator1 and indicator2:
                    fig = create_prediction_scatter(raw_data_df, indicator1, indicator2)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info("💡 Look for clear separation between red (Spikers) and blue (Grinders) points. Good separation = good predictive power.")
            
            # Correlation analysis
            st.subheader("🔗 Indicator Correlations")
            
            numeric_df = raw_data_df.select_dtypes(include=[np.number])
            
            if not numeric_df.empty and len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 8}
                ))
                
                fig.update_layout(
                    title="Indicator Correlation Matrix",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: DATA
    with tab4:
        st.header("📋 Data Explorer")
        
        data_view = st.radio("Select data to view:", ["Candidates", "Raw Data", "Analysis"], horizontal=True)
        
        if data_view == "Candidates" and not candidates_df.empty:
            st.subheader("Candidate Events")
            st.dataframe(candidates_df, use_container_width=True, height=400)
            
            csv = candidates_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "candidates.csv", "text/csv")
        
        elif data_view == "Raw Data" and not raw_data_df.empty:
            st.subheader("Raw Indicator Data (T-1)")
            st.info(f"Shape: {raw_data_df.shape[0]} rows × {raw_data_df.shape[1]} columns")
            st.dataframe(raw_data_df, use_container_width=True, height=400)
            
            csv = raw_data_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "raw_data.csv", "text/csv")
        
        elif data_view == "Analysis" and not analysis_df.empty:
            st.subheader("Analysis Results")
            st.dataframe(analysis_df, use_container_width=True, height=400)
            
            csv = analysis_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "analysis.csv", "text/csv")


if __name__ == "__main__":
    main()
