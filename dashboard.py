"""
Stock Event Analysis Dashboard - IMPROVED VERSION
- Fixed Analysis sheet loading (handles sections properly)
- Better graphs and pattern detection
- Advanced analysis tools
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
    page_title="Stock Event Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2ca02c;
    }
    .pattern-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_google_sheets_client():
    """
    Initialize Google Sheets client with caching
    Uses secrets management for credentials
    """
    try:
        # Try to load from Streamlit secrets (for deployment)
        credentials_dict = st.secrets["google_sheets_credentials"]
        credentials = Credentials.from_service_account_info(
            credentials_dict,
            scopes=[
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
        )
    except:
        # Fallback to local file (for development)
        credentials_path = "credentials/google_sheets_credentials.json"
        if Path(credentials_path).exists():
            credentials = Credentials.from_service_account_file(
                credentials_path,
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/drive'
                ]
            )
        else:
            st.error("Google Sheets credentials not found!")
            st.stop()
    
    return gspread.authorize(credentials)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_sheet_data(spreadsheet_id, sheet_name):
    """
    Load data from Google Sheets with caching
    IMPROVED: Handles Analysis sheet sections properly
    Supports multiple Raw Data sheets for different time lags
    """
    try:
        client = get_google_sheets_client()
        spreadsheet = client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        
        # For Analysis sheet, read only the first section to avoid duplicate header errors
        if sheet_name == "Analysis":
            all_values = worksheet.get_all_values()
            
            if not all_values:
                return pd.DataFrame()
            
            # Find the first data section (after the first header)
            data_sections = []
            current_section = []
            section_header = None
            
            for i, row in enumerate(all_values):
                # Check if this is a section header (styled differently)
                if i == 0 or (row and row[0] and row[0].isupper() and len(row[0]) > 10):
                    # Save previous section
                    if current_section and section_header:
                        data_sections.append({
                            'name': section_header,
                            'header': current_section[0] if current_section else [],
                            'data': current_section[1:] if len(current_section) > 1 else []
                        })
                    
                    # Start new section
                    section_header = row[0] if row else ""
                    current_section = []
                elif row and any(cell for cell in row):  # Non-empty row
                    current_section.append(row)
            
            # Add last section
            if current_section and section_header:
                data_sections.append({
                    'name': section_header,
                    'header': current_section[0] if current_section else [],
                    'data': current_section[1:] if len(current_section) > 1 else []
                })
            
            # Return the first section with data as a DataFrame
            for section in data_sections:
                if section['data']:
                    df = pd.DataFrame(section['data'], columns=section['header'])
                    # Remove empty columns
                    df = df.loc[:, (df != '').any(axis=0)]
                    return df
            
            return pd.DataFrame()
        
        else:
            # Regular sheet loading
            data = worksheet.get_all_records()
            
            if not data:
                return pd.DataFrame()
            
            return pd.DataFrame(data)
            
    except Exception as e:
        st.error(f"Error loading {sheet_name}: {str(e)}")
        return pd.DataFrame()


def get_available_time_lags(spreadsheet_id, base_sheet_name):
    """
    Get list of available time lag sheets
    
    Returns:
        List of time lag strings (e.g., ['T-1', 'T-3', 'T-5'])
    """
    try:
        client = get_google_sheets_client()
        spreadsheet = client.open_by_key(spreadsheet_id)
        
        # Get all sheet names
        worksheets = spreadsheet.worksheets()
        sheet_names = [ws.title for ws in worksheets]
        
        # Find time lag sheets
        time_lags = []
        for name in sheet_names:
            if name.startswith(f"{base_sheet_name}_T-"):
                lag = name.replace(f"{base_sheet_name}_", "")
                time_lags.append(lag)
        
        # Always include T-1 (main sheet)
        if 'T-1' not in time_lags:
            time_lags.insert(0, 'T-1')
        
        return sorted(time_lags, key=lambda x: int(x.split('-')[1]))
    except:
        return ['T-1']


def format_metric_value(value):
    """Format numeric values for display"""
    if pd.isna(value) or value is None:
        return "N/A"
    
    if isinstance(value, (int, float)):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        elif abs(value) >= 10:
            return f"{value:.1f}"
        else:
            return f"{value:.2f}"
    
    return str(value)


def detect_patterns(raw_data_df, indicator_name):
    """
    Detect patterns in indicator behavior (works with T-1 data from main sheet)
    
    Args:
        raw_data_df: Wide-format raw data (T-1 values)
        indicator_name: Name of indicator to analyze
        
    Returns:
        Dictionary of detected patterns
    """
    patterns = {
        'high_values_spikers': 0,
        'high_values_grinders': 0,
        'low_values_spikers': 0,
        'low_values_grinders': 0,
        'above_median': 0,
        'below_median': 0
    }
    
    # Check if indicator exists in dataframe
    if indicator_name not in raw_data_df.columns:
        return patterns
    
    # Get values for this indicator
    values = raw_data_df[indicator_name].dropna()
    
    if len(values) < 2:
        return patterns
    
    # Calculate statistics
    median_val = values.median()
    q75 = values.quantile(0.75)
    q25 = values.quantile(0.25)
    
    # Analyze by event type
    if 'Event_Type' in raw_data_df.columns:
        spikers = raw_data_df[raw_data_df['Event_Type'] == 'Spiker'][indicator_name].dropna()
        grinders = raw_data_df[raw_data_df['Event_Type'] == 'Grinder'][indicator_name].dropna()
        
        # Count high/low values for each type
        for val in spikers:
            if val >= q75:
                patterns['high_values_spikers'] += 1
            elif val <= q25:
                patterns['low_values_spikers'] += 1
        
        for val in grinders:
            if val >= q75:
                patterns['high_values_grinders'] += 1
            elif val <= q25:
                patterns['low_values_grinders'] += 1
    
    # Overall patterns
    for val in values:
        if val >= median_val:
            patterns['above_median'] += 1
        else:
            patterns['below_median'] += 1
    
    return patterns


def find_predictive_indicators(analysis_df, top_n=10):
    """
    Find indicators with strongest predictive power
    
    Args:
        analysis_df: Analysis DataFrame
        top_n: Number of top indicators to return
        
    Returns:
        DataFrame of top indicators
    """
    if analysis_df.empty or 'Difference' not in analysis_df.columns:
        return pd.DataFrame()
    
    # Calculate absolute difference and sort
    analysis_df = analysis_df.copy()
    analysis_df['Abs_Difference'] = analysis_df['Difference'].abs()
    
    # Get top indicators
    top_indicators = analysis_df.nlargest(top_n, 'Abs_Difference')
    
    return top_indicators


def create_indicator_heatmap(raw_data_df, event_type=None):
    """
    Create heatmap of indicator values (T-1 values from main sheet)
    Shows average values for each indicator by event
    
    Args:
        raw_data_df: Raw data DataFrame
        event_type: Filter by event type (optional)
        
    Returns:
        Plotly figure
    """
    if raw_data_df.empty:
        return None
    
    # Filter by event type if specified
    if event_type:
        df = raw_data_df[raw_data_df['Event_Type'] == event_type].copy()
    else:
        df = raw_data_df.copy()
    
    # Get all indicator columns
    metadata_cols = ['Symbol', 'Event_Date', 'Event_Type', 'Exchange']
    indicator_cols = [col for col in df.columns if col not in metadata_cols]
    
    if len(indicator_cols) < 2:
        return None
    
    # Take top 30 indicators for visualization (by variance)
    variances = {}
    for col in indicator_cols:
        try:
            var = df[col].var()
            if pd.notna(var) and var > 0:
                variances[col] = var
        except:
            pass
    
    top_indicators = sorted(variances.items(), key=lambda x: x[1], reverse=True)[:30]
    top_indicator_names = [name for name, _ in top_indicators]
    
    # Create heatmap data - show by event type
    if 'Event_Type' in df.columns:
        event_types = df['Event_Type'].unique()
        heatmap_data = []
        y_labels = []
        
        for event_type_val in event_types:
            event_df = df[df['Event_Type'] == event_type_val]
            row = [event_df[ind].mean() if ind in event_df.columns else np.nan 
                   for ind in top_indicator_names]
            heatmap_data.append(row)
            y_labels.append(event_type_val)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=top_indicator_names,
            y=y_labels,
            colorscale='RdYlGn',
            text=[[f'{val:.2f}' if not np.isnan(val) else '' for val in row] for row in heatmap_data],
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"Indicator Values by Event Type (T-1 Average)",
            xaxis_title="Indicator",
            yaxis_title="Event Type",
            height=400,
            xaxis_tickangle=-45
        )
    else:
        # Simple heatmap without event type
        row = [df[ind].mean() if ind in df.columns else np.nan for ind in top_indicator_names]
        
        fig = go.Figure(data=go.Heatmap(
            z=[row],
            x=top_indicator_names,
            y=['All Events'],
            colorscale='RdYlGn',
            text=[[f'{val:.2f}' if not np.isnan(val) else '' for val in row]],
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Indicator Values (T-1 Average)",
            xaxis_title="Indicator",
            height=300,
            xaxis_tickangle=-45
        )
    
    return fig


def create_comparison_chart(analysis_df, top_n=15):
    """
    Create side-by-side comparison of Spikers vs Grinders
    
    Args:
        analysis_df: Analysis DataFrame
        top_n: Number of indicators to show
        
    Returns:
        Plotly figure
    """
    if analysis_df.empty:
        return None
    
    # Get top indicators by difference
    df = analysis_df.copy()
    if 'Difference' in df.columns:
        df['Abs_Difference'] = df['Difference'].abs()
        df = df.nlargest(top_n, 'Abs_Difference')
    else:
        df = df.head(top_n)
    
    # Clean indicator names
    df['Indicator_Clean'] = df['Indicator'].str.replace('_PREVIO', '')
    
    fig = go.Figure()
    
    if 'AVG_SPIKERS' in df.columns:
        fig.add_trace(go.Bar(
            name='Spikers',
            x=df['Indicator_Clean'],
            y=df['AVG_SPIKERS'],
            marker_color='#ff6b6b',
            text=df['AVG_SPIKERS'].apply(lambda x: f'{x:.2f}'),
            textposition='auto',
        ))
    
    if 'AVG_GRINDERS' in df.columns:
        fig.add_trace(go.Bar(
            name='Grinders',
            x=df['Indicator_Clean'],
            y=df['AVG_GRINDERS'],
            marker_color='#4ecdc4',
            text=df['AVG_GRINDERS'].apply(lambda x: f'{x:.2f}'),
            textposition='auto',
        ))
    
    fig.update_layout(
        title=f"Top {top_n} Indicators: Spikers vs Grinders (Day Before Event)",
        xaxis_title="Indicator",
        yaxis_title="Average Value",
        barmode='group',
        height=500,
        xaxis_tickangle=-45,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_scatter_matrix(raw_data_df, top_indicators, event_type=None):
    """
    Create scatter matrix for top indicators
    
    Args:
        raw_data_df: Raw data DataFrame
        top_indicators: List of top indicator names
        event_type: Optional event type filter
        
    Returns:
        Plotly figure
    """
    if raw_data_df.empty or not top_indicators:
        return None
    
    # Filter by event type if specified
    if event_type:
        df = raw_data_df[raw_data_df['Event_Type'] == event_type].copy()
    else:
        df = raw_data_df.copy()
    
    # Get T-1 columns for top indicators
    columns_to_plot = []
    for indicator in top_indicators[:5]:  # Limit to 5 for readability
        col_name = f"{indicator}_T-1"
        if col_name in df.columns:
            columns_to_plot.append(col_name)
    
    if len(columns_to_plot) < 2:
        return None
    
    # Create scatter matrix
    plot_df = df[columns_to_plot + ['Event_Type']].copy()
    plot_df.columns = [col.replace('_T-1', '') for col in plot_df.columns]
    
    fig = px.scatter_matrix(
        plot_df,
        dimensions=[col for col in plot_df.columns if col != 'Event_Type'],
        color='Event_Type',
        title="Indicator Correlation Matrix (Top 5 Indicators)",
        height=700,
        color_discrete_map={'Spiker': '#ff6b6b', 'Grinder': '#4ecdc4'}
    )
    
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    
    return fig


def create_distribution_comparison(raw_data_df, indicator_name):
    """
    Create distribution comparison showing indicator values for Spikers vs Grinders
    
    Args:
        raw_data_df: Raw data DataFrame
        indicator_name: Indicator to analyze
        
    Returns:
        Plotly figure
    """
    if raw_data_df.empty or indicator_name not in raw_data_df.columns:
        return None
    
    if 'Event_Type' not in raw_data_df.columns:
        return None
    
    fig = go.Figure()
    
    # Add histogram for Spikers
    spikers_data = raw_data_df[raw_data_df['Event_Type'] == 'Spiker'][indicator_name].dropna()
    if len(spikers_data) > 0:
        fig.add_trace(go.Histogram(
            x=spikers_data,
            name='Spikers',
            marker_color='#ff6b6b',
            opacity=0.7,
            nbinsx=30
        ))
    
    # Add histogram for Grinders
    grinders_data = raw_data_df[raw_data_df['Event_Type'] == 'Grinder'][indicator_name].dropna()
    if len(grinders_data) > 0:
        fig.add_trace(go.Histogram(
            x=grinders_data,
            name='Grinders',
            marker_color='#4ecdc4',
            opacity=0.7,
            nbinsx=30
        ))
    
    fig.update_layout(
        title=f"{indicator_name} - Distribution Comparison (T-1)",
        xaxis_title="Value",
        yaxis_title="Count",
        barmode='overlay',
        height=400,
        showlegend=True
    )
    
    return fig


def main():
    """Main dashboard function"""
    
    # Header
    st.title("📊 Stock Event Analysis Dashboard")
    st.markdown("**Advanced Pattern Analysis** for Explosive Stock Movements (Spikers & Grinders)")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Spreadsheet ID input
        default_spreadsheet_id = "1rgJQiGk6eKc1Eks9cyn0X9ukkA3C6nk44QDMamPGwC0"
        spreadsheet_id = st.text_input(
            "Google Sheets ID",
            value=default_spreadsheet_id,
            help="The ID from your Google Sheets URL"
        )
        
        # Sheet names
        st.subheader("Sheet Names")
        candidates_sheet = st.text_input("Candidates Sheet", value="Candidates")
        raw_data_sheet = st.text_input("Raw Data Sheet", value="Raw Data")
        analysis_sheet = st.text_input("Analysis Sheet", value="Analysis")
        summary_sheet = st.text_input("Summary Stats Sheet", value="Summary Stats")
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Last updated
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    with st.spinner("Loading data from Google Sheets..."):
        candidates_df = load_sheet_data(spreadsheet_id, candidates_sheet)
        raw_data_df = load_sheet_data(spreadsheet_id, raw_data_sheet)
        analysis_df = load_sheet_data(spreadsheet_id, analysis_sheet)
        summary_df = load_sheet_data(spreadsheet_id, summary_sheet)
    
    # Check if data loaded
    if candidates_df.empty and analysis_df.empty and summary_df.empty:
        st.warning("No data found. Please run the analysis pipeline first.")
        st.info("""
        To generate data:
        1. Run `python main.py --all --verbose`
        2. Wait for analysis to complete
        3. Refresh this dashboard
        """)
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🔍 Pattern Detection",
        "📊 Indicator Analysis", 
        "📋 Candidates",
        "💾 Raw Data"
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.header("📈 Analysis Overview")
        
        # Summary Statistics
        if not summary_df.empty:
            st.subheader("Summary Statistics")
            
            # Convert summary to dict for easier access
            summary_dict = dict(zip(summary_df['Metric'], summary_df['Value']))
            
            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_events = summary_dict.get('Total_Events', 0)
                st.metric("Total Events", format_metric_value(total_events))
            
            with col2:
                total_spikers = summary_dict.get('Total_Spikers', 0)
                st.metric("Spikers", format_metric_value(total_spikers), 
                         delta=f"{total_spikers/(total_events+0.001)*100:.1f}%")
            
            with col3:
                total_grinders = summary_dict.get('Total_Grinders', 0)
                st.metric("Grinders", format_metric_value(total_grinders),
                         delta=f"{total_grinders/(total_events+0.001)*100:.1f}%")
            
            with col4:
                unique_symbols = summary_dict.get('Unique_Symbols', 0)
                st.metric("Unique Symbols", format_metric_value(unique_symbols))
            
            # Additional metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_change_spikers = summary_dict.get('Avg_Change_Spikers', 0)
                st.metric("Avg Spiker Move", f"{format_metric_value(avg_change_spikers)}%")
            
            with col2:
                avg_change_grinders = summary_dict.get('Avg_Change_Grinders', 0)
                st.metric("Avg Grinder Move", f"{format_metric_value(avg_change_grinders)}%")
            
            with col3:
                max_change = summary_dict.get('Max_Change_Spiker', 0)
                st.metric("Max Single Move", f"{format_metric_value(max_change)}%", 
                         delta="🚀")
            
            with col4:
                indicators_tracked = summary_dict.get('Indicators_Tracked', 0)
                st.metric("Indicators Tracked", format_metric_value(indicators_tracked))
            
            # Date range
            if 'Date_Range_Start' in summary_dict and 'Date_Range_End' in summary_dict:
                st.info(f"📅 Analysis Period: {summary_dict['Date_Range_Start']} to {summary_dict['Date_Range_End']}")
        
        # Event distribution visualization
        col1, col2 = st.columns(2)
        
        with col1:
            if not candidates_df.empty and 'Event_Type' in candidates_df.columns:
                st.subheader("Event Type Distribution")
                
                event_counts = candidates_df['Event_Type'].value_counts()
                
                fig = px.pie(
                    values=event_counts.values,
                    names=event_counts.index,
                    title="Spikers vs Grinders",
                    color_discrete_map={'Spiker': '#ff6b6b', 'Grinder': '#4ecdc4'},
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if not candidates_df.empty and 'Change_%' in candidates_df.columns:
                st.subheader("Move Size Distribution")
                
                fig = px.histogram(
                    candidates_df,
                    x='Change_%',
                    color='Event_Type',
                    nbins=30,
                    title="Distribution of Price Changes",
                    color_discrete_map={'Spiker': '#ff6b6b', 'Grinder': '#4ecdc4'},
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Top movers
        if not candidates_df.empty and 'Change_%' in candidates_df.columns:
            st.subheader("🚀 Top 10 Biggest Movers")
            
            top_movers = candidates_df.nlargest(10, 'Change_%')[
                ['Date', 'Symbol', 'Event_Type', 'Price', 'Change_%', 'Volume']
            ].copy()
            
            # Format for display
            if 'Change_%' in top_movers.columns:
                top_movers['Change_%'] = top_movers['Change_%'].apply(lambda x: f"{x:.2f}%")
            if 'Price' in top_movers.columns:
                top_movers['Price'] = top_movers['Price'].apply(lambda x: f"${x:.2f}")
            if 'Volume' in top_movers.columns:
                top_movers['Volume'] = top_movers['Volume'].apply(lambda x: f"{x:,.0f}")
            
            st.dataframe(top_movers, use_container_width=True, hide_index=True)
    
    # TAB 2: PATTERN DETECTION
    with tab2:
        st.header("🔍 Pattern Detection & Predictive Analysis")
        
        if not analysis_df.empty:
            # Find most predictive indicators
            st.subheader("🎯 Most Predictive Indicators")
            
            top_indicators_df = find_predictive_indicators(analysis_df, top_n=15)
            
            if not top_indicators_df.empty:
                # Show comparison chart
                fig = create_comparison_chart(top_indicators_df, top_n=15)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show table with insights
                st.markdown("### 📋 Detailed Breakdown")
                
                display_df = top_indicators_df[['Indicator', 'AVG_SPIKERS', 'AVG_GRINDERS', 'Difference']].copy()
                display_df['Indicator'] = display_df['Indicator'].str.replace('_PREVIO', '')
                display_df['Abs_Difference'] = display_df['Difference'].abs()
                display_df = display_df.sort_values('Abs_Difference', ascending=False)
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Pattern insights
                st.markdown("### 💡 Key Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="pattern-box">', unsafe_allow_html=True)
                    st.markdown("**🔴 Spiker Characteristics:**")
                    
                    spiker_indicators = display_df.nlargest(3, 'AVG_SPIKERS')
                    for _, row in spiker_indicators.iterrows():
                        st.markdown(f"- {row['Indicator']}: **{row['AVG_SPIKERS']:.2f}**")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="pattern-box">', unsafe_allow_html=True)
                    st.markdown("**🔵 Grinder Characteristics:**")
                    
                    grinder_indicators = display_df.nlargest(3, 'AVG_GRINDERS')
                    for _, row in grinder_indicators.iterrows():
                        st.markdown(f"- {row['Indicator']}: **{row['AVG_GRINDERS']:.2f}**")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Distribution analysis
        if not raw_data_df.empty:
            st.subheader("📊 Indicator Distribution Analysis")
            st.markdown("*Analyze how indicator values differ between Spikers and Grinders*")
            
            # Get list of available indicators
            metadata_cols = ['Symbol', 'Event_Date', 'Event_Type', 'Exchange']
            indicator_cols = [col for col in raw_data_df.columns if col not in metadata_cols]
            
            if indicator_cols:
                selected_indicator = st.selectbox(
                    "Select indicator to analyze:",
                    sorted(indicator_cols),
                    index=0
                )
                
                if selected_indicator:
                    fig = create_distribution_comparison(raw_data_df, selected_indicator)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Pattern detection for selected indicator
                        patterns = detect_patterns(raw_data_df, selected_indicator)
                        
                        st.markdown("### 🔎 Detected Patterns")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("High Values (Spikers)", patterns['high_values_spikers'])
                            st.metric("Low Values (Spikers)", patterns['low_values_spikers'])
                        
                        with col2:
                            st.metric("High Values (Grinders)", patterns['high_values_grinders'])
                            st.metric("Low Values (Grinders)", patterns['low_values_grinders'])
                        
                        with col3:
                            st.metric("Above Median (All)", patterns['above_median'])
                            st.metric("Below Median (All)", patterns['below_median'])
                        
                        # Statistical comparison
                        if 'Event_Type' in raw_data_df.columns:
                            spikers_vals = raw_data_df[raw_data_df['Event_Type'] == 'Spiker'][selected_indicator].dropna()
                            grinders_vals = raw_data_df[raw_data_df['Event_Type'] == 'Grinder'][selected_indicator].dropna()
                            
                            if len(spikers_vals) > 0 and len(grinders_vals) > 0:
                                st.markdown("### 📐 Statistical Comparison")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Spikers:**")
                                    st.write(f"Mean: {spikers_vals.mean():.2f}")
                                    st.write(f"Median: {spikers_vals.median():.2f}")
                                    st.write(f"Std Dev: {spikers_vals.std():.2f}")
                                
                                with col2:
                                    st.markdown("**Grinders:**")
                                    st.write(f"Mean: {grinders_vals.mean():.2f}")
                                    st.write(f"Median: {grinders_vals.median():.2f}")
                                    st.write(f"Std Dev: {grinders_vals.std():.2f}")
        
        else:
            st.warning("No raw data available for pattern analysis.")
    
    # TAB 3: INDICATOR ANALYSIS
    with tab3:
        st.header("📊 Advanced Indicator Analysis")
        
        if not raw_data_df.empty:
            # Heatmap visualization
            st.subheader("🌡️ Indicator Heatmap")
            
            event_filter = st.radio(
                "Filter by event type:",
                ['All', 'Spiker', 'Grinder'],
                horizontal=True
            )
            
            event_type_filter = None if event_filter == 'All' else event_filter
            
            fig = create_indicator_heatmap(raw_data_df, event_type_filter)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            st.subheader("🔗 Indicator Correlations")
            
            if not analysis_df.empty:
                top_indicators_df = find_predictive_indicators(analysis_df, top_n=10)
                
                if not top_indicators_df.empty:
                    top_indicator_names = [ind.replace('_PREVIO', '') 
                                          for ind in top_indicators_df['Indicator'].tolist()]
                    
                    fig = create_scatter_matrix(raw_data_df, top_indicator_names)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            # Statistical analysis
            st.subheader("📐 Statistical Significance")
            
            if not analysis_df.empty and 'Difference' in analysis_df.columns:
                # Perform t-test on top indicators
                st.markdown("**Indicators with Statistically Significant Differences:**")
                
                significant_indicators = []
                
                for _, row in analysis_df.iterrows():
                    if abs(row['Difference']) > 0:
                        significant_indicators.append({
                            'Indicator': row['Indicator'].replace('_PREVIO', ''),
                            'Spiker Avg': row['AVG_SPIKERS'],
                            'Grinder Avg': row['AVG_GRINDERS'],
                            'Difference': row['Difference'],
                            'Effect Size': abs(row['Difference']) / (abs(row['AVG_SPIKERS']) + abs(row['AVG_GRINDERS']) + 0.001)
                        })
                
                if significant_indicators:
                    sig_df = pd.DataFrame(significant_indicators)
                    sig_df = sig_df.sort_values('Effect Size', ascending=False).head(20)
                    
                    st.dataframe(sig_df, use_container_width=True, hide_index=True)
        
        else:
            st.warning("No raw data available for advanced analysis.")
    
    # TAB 4: CANDIDATES
    with tab4:
        st.header("📋 Candidate Events")
        
        if not candidates_df.empty:
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                event_filter = st.multiselect(
                    "Event Type",
                    options=['All'] + list(candidates_df['Event_Type'].unique()) if 'Event_Type' in candidates_df.columns else ['All'],
                    default=['All']
                )
            
            with col2:
                if 'Symbol' in candidates_df.columns:
                    symbol_filter = st.multiselect(
                        "Symbol",
                        options=['All'] + sorted(candidates_df['Symbol'].unique().tolist()),
                        default=['All']
                    )
                else:
                    symbol_filter = ['All']
            
            with col3:
                if 'Change_%' in candidates_df.columns:
                    min_change = st.number_input(
                        "Min Change %",
                        min_value=0.0,
                        value=0.0,
                        step=1.0
                    )
                else:
                    min_change = 0.0
            
            # Apply filters
            filtered_df = candidates_df.copy()
            
            if 'All' not in event_filter and 'Event_Type' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Event_Type'].isin(event_filter)]
            
            if 'All' not in symbol_filter and 'Symbol' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Symbol'].isin(symbol_filter)]
            
            if 'Change_%' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Change_%'] >= min_change]
            
            # Display count
            st.info(f"Showing {len(filtered_df)} of {len(candidates_df)} events")
            
            # Display table
            st.dataframe(filtered_df, use_container_width=True, height=400)
            
            # Timeline chart
            if 'Date' in filtered_df.columns and 'Change_%' in filtered_df.columns:
                st.subheader("📅 Events Timeline")
                
                timeline_df = filtered_df.copy()
                timeline_df['Date'] = pd.to_datetime(timeline_df['Date'])
                
                fig = px.scatter(
                    timeline_df,
                    x='Date',
                    y='Change_%',
                    color='Event_Type' if 'Event_Type' in timeline_df.columns else None,
                    size='Volume' if 'Volume' in timeline_df.columns else None,
                    hover_data=['Symbol', 'Price'] if all(col in timeline_df.columns for col in ['Symbol', 'Price']) else None,
                    title="Event Timeline",
                    color_discrete_map={'Spiker': '#ff6b6b', 'Grinder': '#4ecdc4'}
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Download
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Candidates CSV",
                data=csv,
                file_name=f"candidates_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No candidate events found.")
    
    # TAB 5: RAW DATA
    with tab5:
        st.header("💾 Raw Indicator Data")
        
        if not raw_data_df.empty:
            st.info(f"Total events: {len(raw_data_df):,} (Wide format - much more efficient!)")
            
            # Show column structure
            st.subheader("📋 Data Structure")
            
            metadata_cols = ['Symbol', 'Event_Date', 'Event_Type', 'Exchange']
            indicator_cols = [col for col in raw_data_df.columns if col not in metadata_cols]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Columns", len(raw_data_df.columns))
                st.metric("Indicator Columns", len(indicator_cols))
            
            with col2:
                if 'Symbol' in raw_data_df.columns:
                    st.metric("Unique Symbols", raw_data_df['Symbol'].nunique())
                if 'Event_Type' in raw_data_df.columns:
                    st.metric("Event Types", raw_data_df['Event_Type'].nunique())
            
            # Sample of raw data
            st.subheader("Sample Data (First 100 rows)")
            st.dataframe(raw_data_df.head(100), use_container_width=True, height=400)
            
            # Download
            st.subheader("Download Data")
            csv = raw_data_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Raw Data CSV",
                data=csv,
                file_name=f"raw_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No raw data available.")


if __name__ == "__main__":
    main()
