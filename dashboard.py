"""
Stock Analysis Dashboard
Professional multi-tab dashboard with modular components
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, professional CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        font-weight: 500;
        font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    h1 {
        color: #1a202c;
        font-weight: 600;
        margin-bottom: 2rem;
    }
    h2 {
        color: #2d3748;
        font-weight: 600;
        margin-top: 2rem;
    }
    h3 {
        color: #4a5568;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    """Main dashboard with tabs"""
    
    st.title("Stock Analysis Dashboard")
    
    # Import the tab modules
    try:
        from tab_daily_winners import render_daily_winners_tab
        from tab_spike_grinder import render_spike_grinder_tab
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Daily Winners", "Spike/Grinder Analysis", "Backtesting"])
        
        with tab1:
            render_daily_winners_tab()
        
        with tab2:
            render_spike_grinder_tab()
        
        with tab3:
            render_backtesting_tab()  # NEW
            
    except ImportError as e:
        st.error(f"Error importing tab modules: {e}")
        st.info("Make sure tab_daily_winners.py and tab_spike_grinder.py are in the same directory as dashboard.py")


if __name__ == "__main__":
    main()
