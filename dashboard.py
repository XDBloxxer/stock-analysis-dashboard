"""
Stock Event Analysis Dashboard
Displays results from Google Sheets analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from pathlib import Path
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
    """
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


def main():
    """Main dashboard function"""
    
    # Header
    st.title("📊 Stock Event Analysis Dashboard")
    st.markdown("Real-time analysis of explosive stock movements (Spikers & Grinders)")
    
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview", 
        "🔍 Detailed Analysis", 
        "📊 Candidates", 
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
                st.metric("Spikers", format_metric_value(total_spikers))
            
            with col3:
                total_grinders = summary_dict.get('Total_Grinders', 0)
                st.metric("Grinders", format_metric_value(total_grinders))
            
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
                st.metric("Max Single Move", f"{format_metric_value(max_change)}%")
            
            with col4:
                indicators_tracked = summary_dict.get('Indicators_Tracked', 0)
                st.metric("Indicators Tracked", format_metric_value(indicators_tracked))
            
            # Date range
            if 'Date_Range_Start' in summary_dict and 'Date_Range_End' in summary_dict:
                st.info(f"📅 Analysis Period: {summary_dict['Date_Range_Start']} to {summary_dict['Date_Range_End']}")
        
        # Event distribution chart
        if not candidates_df.empty and 'Event_Type' in candidates_df.columns:
            st.subheader("Event Type Distribution")
            
            event_counts = candidates_df['Event_Type'].value_counts()
            
            fig = px.pie(
                values=event_counts.values,
                names=event_counts.index,
                title="Spikers vs Grinders",
                color_discrete_sequence=['#1f77b4', '#2ca02c']
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
    
    # TAB 2: DETAILED ANALYSIS
    with tab2:
        st.header("🔍 Detailed Analysis")
        
        if not analysis_df.empty:
            # Check if we have the summary section
            if 'Indicator' in analysis_df.columns:
                st.subheader("📊 Pre-Move Indicator Averages")
                
                # Filter to summary rows (those with "_PREVIO" suffix)
                summary_rows = analysis_df[analysis_df['Indicator'].str.contains('_PREVIO', na=False)].copy()
                
                if not summary_rows.empty:
                    # Clean indicator names
                    summary_rows['Indicator_Clean'] = summary_rows['Indicator'].str.replace('_PREVIO', '')
                    
                    # Sort by absolute difference
                    if 'Difference' in summary_rows.columns:
                        summary_rows['Abs_Diff'] = summary_rows['Difference'].abs()
                        summary_rows = summary_rows.sort_values('Abs_Diff', ascending=False)
                    
                    # Display top indicators with biggest differences
                    st.markdown("**Indicators with Biggest Differences Between Spikers & Grinders:**")
                    
                    display_cols = ['Indicator_Clean', 'AVG_SPIKERS', 'AVG_GRINDERS', 'Difference']
                    available_cols = [col for col in display_cols if col in summary_rows.columns]
                    
                    st.dataframe(
                        summary_rows[available_cols].head(20),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Visualization: Compare Spikers vs Grinders
                    st.subheader("📈 Spikers vs Grinders - Top 15 Indicators")
                    
                    top_15 = summary_rows.head(15)
                    
                    fig = go.Figure()
                    
                    if 'AVG_SPIKERS' in top_15.columns:
                        fig.add_trace(go.Bar(
                            name='Spikers',
                            x=top_15['Indicator_Clean'],
                            y=top_15['AVG_SPIKERS'],
                            marker_color='#1f77b4'
                        ))
                    
                    if 'AVG_GRINDERS' in top_15.columns:
                        fig.add_trace(go.Bar(
                            name='Grinders',
                            x=top_15['Indicator_Clean'],
                            y=top_15['AVG_GRINDERS'],
                            marker_color='#2ca02c'
                        ))
                    
                    fig.update_layout(
                        title="Average Indicator Values (Day Before Event)",
                        xaxis_title="Indicator",
                        yaxis_title="Average Value",
                        barmode='group',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Full analysis table
            st.subheader("📋 Complete Analysis Data")
            st.dataframe(analysis_df, use_container_width=True, height=400)
            
            # Download button
            csv = analysis_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Analysis CSV",
                data=csv,
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No analysis data available yet.")
    
    # TAB 3: CANDIDATES
    with tab3:
        st.header("📊 Candidate Events")
        
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
                    color_discrete_sequence=['#1f77b4', '#2ca02c']
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
    
    # TAB 4: RAW DATA
    with tab4:
        st.header("💾 Raw Indicator Data")
        
        if not raw_data_df.empty:
            st.info(f"Total data points: {len(raw_data_df):,}")
            
            # Sample of raw data
            st.subheader("Sample Data (First 1000 rows)")
            st.dataframe(raw_data_df.head(1000), use_container_width=True, height=400)
            
            # Statistics
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Symbol' in raw_data_df.columns:
                    st.metric("Unique Symbols", raw_data_df['Symbol'].nunique())
            
            with col2:
                if 'Indicator_Name' in raw_data_df.columns:
                    st.metric("Unique Indicators", raw_data_df['Indicator_Name'].nunique())
            
            # Indicator breakdown
            if 'Indicator_Name' in raw_data_df.columns:
                st.subheader("Indicator Coverage")
                
                indicator_counts = raw_data_df['Indicator_Name'].value_counts()
                
                fig = px.bar(
                    x=indicator_counts.index[:20],
                    y=indicator_counts.values[:20],
                    title="Top 20 Indicators (by data points)",
                    labels={'x': 'Indicator', 'y': 'Count'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
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
