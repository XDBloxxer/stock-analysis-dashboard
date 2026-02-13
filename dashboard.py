"""
Main Dashboard - 3 TABS
Communicates with tradingview-analysis repo via GitHub Actions + Supabase
"""

import streamlit as st
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .css-1d391kg {
        background-color: #1a1d29;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #e8eaf0;
    }
    div[data-testid="stMetricLabel"] {
        color: #b8bac5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1d29;
        border-radius: 4px;
        color: #b8bac5;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2d3142;
        color: #e8eaf0;
    }
    .refresh-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main dashboard with 3 tabs"""
    
    st.title("📊 Stock Analysis Dashboard")
    
    # Check environment variables
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        st.error("⚠️ Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_KEY environment variables.")
        st.stop()
    
    # Optional: GitHub token for triggering workflows
    github_token = os.getenv("GITHUB_TOKEN")
    github_repo = os.getenv("GITHUB_REPO", "your-username/tradingview-analysis")
    
    # Store in session state for tabs to access
    st.session_state.github_token = github_token
    st.session_state.github_repo = github_repo
    
    try:
        from tab_daily_winners import render_daily_winners_tab
        from tab_ml_predictions import render_ml_predictions_tab
        from tab_backtesting import render_backtesting_tab
    except ImportError as e:
        st.error(f"Error importing tab modules: {e}")
        st.info("Make sure all dashboard files are in the same directory")
        st.stop()
    
    # Create 3 tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Daily Winners",
        "🤖 ML Predictions",
        "📈 Strategy Backtesting"
    ])
    
    with tab1:
        render_daily_winners_tab()
    
    with tab2:
        render_ml_predictions_tab()
    
    with tab3:
        render_backtesting_tab()


if __name__ == "__main__":
    main()
