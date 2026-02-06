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

# Modern professional CSS with dark theme
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 1rem 2rem;
        background: linear-gradient(135deg, #1a1d29 0%, #2d3142 100%);
    }
    
    /* Streamlit default backgrounds */
    .stApp {
        background: linear-gradient(135deg, #1a1d29 0%, #2d3142 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: rgba(255, 255, 255, 0.05);
        padding: 0.5rem;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
        background-color: transparent;
        border-radius: 8px;
        color: #b8bcc8;
        font-weight: 500;
        font-size: 15px;
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric cards */
    .stMetric {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stMetric label {
        color: #b8bcc8 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
    }
    
    /* Headers */
    h1 {
        color: #ffffff;
        font-weight: 700;
        margin-bottom: 2rem;
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        color: #e8eaf0;
        font-weight: 600;
        margin-top: 2rem;
        font-size: 1.8rem;
    }
    
    h3 {
        color: #d1d5db;
        font-weight: 500;
        font-size: 1.3rem;
        margin-top: 1.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Select boxes and inputs */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div,
    .stNumberInput > div > div,
    .stTextArea > div > div {
        background-color: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #ffffff;
    }
    
    .stSelectbox label,
    .stMultiSelect label,
    .stTextInput label,
    .stNumberInput label,
    .stTextArea label,
    .stSlider label {
        color: #b8bcc8 !important;
        font-weight: 500;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        overflow: hidden;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #e8eaf0;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(255, 255, 255, 0.12);
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    [data-testid="stMarkdownContainer"] p {
        color: #d1d5db;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
        margin: 2rem 0;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Date input */
    .stDateInput > div > div {
        background-color: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(245, 87, 108, 0.4);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Plotly charts - dark background */
    .js-plotly-plot {
        background-color: transparent !important;
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
        from tab_backtesting import render_backtesting_tab
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Daily Winners", "Spike/Grinder Analysis", "Backtesting"])
        
        with tab1:
            render_daily_winners_tab()
        
        with tab2:
            render_spike_grinder_tab()
        
        with tab3:
            render_backtesting_tab()
            
    except ImportError as e:
        st.error(f"Error importing tab modules: {e}")
        st.info("Make sure tab_daily_winners.py and tab_spike_grinder.py are in the same directory as dashboard.py")


if __name__ == "__main__":
    main()
