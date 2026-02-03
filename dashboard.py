"""
Stock Event Analysis Dashboard - Supabase Version
Reads from Supabase (primary) with Google Sheets fallback
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


@st.cache_resource
def get_google_sheets_client():
    """Initialize Google Sheets client (fallback)"""
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
            return None
    
    return gspread.authorize(credentials)


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
def load_sheet_data(spreadsheet_id, sheet_name):
    """Load data from Google Sheets (fallback)"""
    try:
        client = get_google_sheets_client()
        if not client:
            return pd.DataFrame()
        
        spreadsheet = client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        data = worksheet.get_all_records()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        return df
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_all_time_lags():
    """Load all time lag data from Supabase"""
    time_lags = {}
    
    # Get unique time lags
    try:
        client = get_supabase_client()
        response = client.table("raw_data").select("time_lag").execute()
        
        if response.data:
            unique_lags = list(set(row["time_lag"] for row in response.data))
            
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
    
    # Check with lowercase (Supabase uses lowercase)
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Analysis data is missing required columns: {', '.join(missing_cols)}")
        return pd.DataFrame()
    
    # Remove rows where indicator is empty/null
    df = df[df['indicator'].notna() & (df['indicator'].astype(str).str.strip() != '')]
    
    # Ensure numeric columns are actually numeric
    for col in ['avg_spikers', 'avg_grinders']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows where both averages are null or both are 0
    df = df[
        (df['avg_spikers'].notna() | df['avg_grinders'].notna()) &
        ~((df['avg_spikers'] == 0) & (df['avg_grinders'] == 0))
    ]
    
    return df


# ... (rest of the visualization functions remain the same, just update column names to lowercase)
# I'll include a few key ones here:

def create_top_discriminators_chart(analysis_df, time_lag='T-1', top_n=15):
    """Create chart showing top discriminating indicators"""
    if analysis_df.empty or 'time_lag' not in analysis_df.columns:
        return None
    
    df = analysis_df[analysis_df['time_lag'] == time_lag].copy()
    
    if df.empty:
        return None
    
    # Calculate percentage difference
    df['pct_difference'] = np.where(
        df['avg_grinders'].abs() > 0,
        ((df['avg_spikers'] - df['avg_grinders']) / df['avg_grinders'].abs()) * 100,
        0
    )
    
    # Remove infinite and NaN values
    df = df[np.isfinite(df['pct_difference'])]
    
    if df.empty:
        return None
    
    # Cap extreme outliers
    if len(df) > 1:
        cap_value = np.percentile(df['pct_difference'].abs(), 99)
    else:
        cap_value = df['pct_difference'].abs().max()
    
    df['pct_difference_display'] = df['pct_difference'].clip(-cap_value, cap_value)
    df['is_capped'] = (df['pct_difference'].abs() > cap_value)
    
    # Sort by absolute percentage difference
    df['abs_pct_diff'] = df['pct_difference'].abs()
    
    actual_n = min(top_n, len(df))
    df = df.nlargest(actual_n, 'abs_pct_diff')
    df = df.sort_values('pct_difference_display')
    
    if df.empty:
        return None
    
    # Create color based on direction
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in df['pct_difference']]
    
    # Create hover text
    hover_texts = []
    for _, row in df.iterrows():
        capped_text = " (CAPPED - actual: {:.1f}%)".format(row['pct_difference']) if row['is_capped'] else ""
        hover_texts.append(
            f"<b>{row['indicator']}</b><br>" +
            f"Percentage Difference: {row['pct_difference_display']:.1f}%{capped_text}<br>" +
            f"<br>" +
            f"Spiker Avg: {row['avg_spikers']:.2f}<br>" +
            f"Grinder Avg: {row['avg_grinders']:.2f}<br>"
        )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['indicator'],
        x=df['pct_difference_display'],
        orientation='h',
        marker_color=colors,
        text=[f'{x:+.1f}%{"*" if c else ""}' for x, c in zip(df['pct_difference_display'], df['is_capped'])],
        textposition='outside',
        hovertext=hover_texts,
        hoverinfo='text'
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2)
    
    # Calculate symmetric x-axis range
    max_abs_display = df['pct_difference_display'].abs().max()
    x_range = [-max_abs_display * 1.15, max_abs_display * 1.15]
    
    fig.update_layout(
        title=f"<b>Top {actual_n} Discriminating Indicators ({time_lag})</b><br>" +
              "<sub>Red = Spikers higher | Blue = Grinders higher</sub>",
        xaxis_title="Percentage Difference (%)",
        yaxis_title="",
        height=max(500, actual_n * 30),
        showlegend=False,
        title_x=0.5,
        xaxis=dict(range=x_range, zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    )
    
    return fig


def main():
    """Main dashboard"""
    
    st.title("🎯 Stock Pattern Analysis Dashboard")
    st.markdown("**Discover what distinguishes explosive movers from steady grinders**")
    
    # Data source selector
    data_source = st.radio(
        "Data Source:",
        ["Supabase (Primary)", "Google Sheets (Fallback)"],
        horizontal=True
    )
    
    use_supabase = "Supabase" in data_source
    
    # Configuration
    with st.expander("⚙️ Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if not use_supabase:
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
        if use_supabase:
            candidates_df = load_supabase_data("candidates")
            analysis_df_raw = load_supabase_data("analysis")
            summary_df = load_supabase_data("summary_stats")
            raw_data_dict = load_all_time_lags()
        else:
            candidates_df = load_sheet_data(spreadsheet_id, "Candidates")
            analysis_df_raw = load_sheet_data(spreadsheet_id, "Analysis")
            summary_df = load_sheet_data(spreadsheet_id, "Summary Stats")
            raw_data_dict = {}  # Would need to implement for sheets
    
    # Validate analysis data
    analysis_df = validate_analysis_df(analysis_df_raw)
    
    if analysis_df.empty and candidates_df.empty:
        st.warning(f"⚠️ No data found in {data_source}")
        return
    
    # Summary metrics
    if not summary_df.empty:
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
    
    # Show key discriminators
    if not analysis_df.empty and 'time_lag' in analysis_df.columns:
        st.header("🎯 Most Discriminating Indicators")
        
        available_lags = sorted(analysis_df['time_lag'].unique())
        
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_lag = st.selectbox("Time Period:", available_lags, index=0)
        with col2:
            top_n = st.slider("Number of indicators:", 10, 50, 20)
        
        fig = create_top_discriminators_chart(analysis_df, selected_lag, top_n)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Data preview
    st.markdown("---")
    st.header("📋 Data Preview")
    
    tab1, tab2, tab3 = st.tabs(["Candidates", "Analysis", "Raw Data"])
    
    with tab1:
        if not candidates_df.empty:
            st.dataframe(candidates_df.head(100), use_container_width=True)
    
    with tab2:
        if not analysis_df.empty:
            st.dataframe(analysis_df.head(100), use_container_width=True)
    
    with tab3:
        if raw_data_dict:
            selected_lag = st.selectbox("Select time lag:", list(raw_data_dict.keys()), key='raw_lag')
            if selected_lag in raw_data_dict:
                st.dataframe(raw_data_dict[selected_lag].head(100), use_container_width=True)


if __name__ == "__main__":
    main()
