"""
Stock Analysis Dashboard Hub
Landing page with links to specialized dashboards
"""

import streamlit as st
from datetime import datetime
import os

# Supabase
from supabase import create_client, Client

# Page config
st.set_page_config(
    page_title="Stock Analysis Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem 3rem;
    }
    .dashboard-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .dashboard-card h2 {
        color: white;
        margin: 0 0 1rem 0;
    }
    .dashboard-card p {
        color: rgba(255,255,255,0.9);
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .status-active {
        background: #10b981;
        color: white;
    }
    .status-pending {
        background: #f59e0b;
        color: white;
    }
    h1 {
        color: #2d3748;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #718096;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_supabase_client():
    """Initialize Supabase client"""
    try:
        supabase_url = os.environ.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url")
        supabase_key = os.environ.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("key")
        
        if not supabase_url or not supabase_key:
            return None
        
        return create_client(supabase_url, supabase_key)
    except:
        return None


def check_data_availability(table_name):
    """Check if data exists in a table"""
    try:
        client = get_supabase_client()
        if not client:
            return False
        
        response = client.table(table_name).select("id").limit(1).execute()
        return len(response.data) > 0
    except:
        return False


def main():
    """Main hub page"""
    
    st.title("📊 Stock Analysis Dashboard Hub")
    st.markdown('<p class="subtitle">Choose a specialized dashboard below to begin your analysis</p>', 
                unsafe_allow_html=True)
    
    # Check data availability
    with st.spinner("Checking data availability..."):
        daily_winners_available = check_data_availability("daily_winners")
        spike_grinder_available = check_data_availability("candidates") or check_data_availability("analysis")
    
    # Dashboard cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="dashboard-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h2>🏆 Daily Winners Tracker</h2>
            <p>Track the top 10 daily stock winners with detailed technical indicator snapshots captured at market open, market close, and the previous trading day.</p>
            <strong>Features:</strong>
            <ul>
                <li>Top performer identification</li>
                <li>Multi-timepoint indicator analysis</li>
                <li>Indicator evolution visualization</li>
                <li>Performance metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        status = "status-active" if daily_winners_available else "status-pending"
        status_text = "✓ Data Available" if daily_winners_available else "⏳ No Data Yet"
        st.markdown(f'<span class="status-badge {status}">{status_text}</span>', unsafe_allow_html=True)
        
        if st.button("🚀 Open Daily Winners", use_container_width=True, type="primary"):
            st.info("📝 To run this dashboard separately:\n\n`streamlit run dashboard_daily_winners.py`")
        
        if not daily_winners_available:
            st.caption("💡 Run `python daily_winners_main.py` to populate data")
    
    with col2:
        st.markdown("""
        <div class="dashboard-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h2>📈 Spike vs Grinder Analysis</h2>
            <p>Analyze technical patterns that differentiate rapid spike events from steady grinder movements across multiple time lags.</p>
            <strong>Features:</strong>
            <ul>
                <li>Pattern differentiation analysis</li>
                <li>Time lag comparisons</li>
                <li>Indicator distribution analysis</li>
                <li>Custom visualization builder</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        status = "status-active" if spike_grinder_available else "status-pending"
        status_text = "✓ Data Available" if spike_grinder_available else "⏳ No Data Yet"
        st.markdown(f'<span class="status-badge {status}">{status_text}</span>', unsafe_allow_html=True)
        
        if st.button("🚀 Open Spike/Grinder", use_container_width=True, type="primary"):
            st.info("📝 To run this dashboard separately:\n\n`streamlit run dashboard_spike_grinder.py`")
        
        if not spike_grinder_available:
            st.caption("💡 Run `python main.py --all` to populate data")
    
    st.markdown("---")
    
    # System info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("System Status", "🟢 Online")
    
    with col2:
        client = get_supabase_client()
        db_status = "🟢 Connected" if client else "🔴 Not Connected"
        st.metric("Database", db_status)
    
    with col3:
        st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))
    
    # Instructions
    with st.expander("📚 Quick Start Guide"):
        st.markdown("""
        ### Running Individual Dashboards
        
        You can run each dashboard independently:
        
        ```bash
        # Daily Winners Dashboard
        streamlit run dashboard_daily_winners.py
        
        # Spike/Grinder Analysis Dashboard
        streamlit run dashboard_spike_grinder.py
        
        # This Hub Page
        streamlit run dashboard.py
        ```
        
        ### Data Population
        
        Before using the dashboards, make sure to populate the database:
        
        ```bash
        # For Daily Winners data
        python daily_winners_main.py --verbose
        
        # For Spike/Grinder analysis data
        python main.py --all --verbose
        ```
        
        ### Benefits of Separate Dashboards
        
        - **Independence**: Each dashboard works even if the other has no data
        - **Performance**: Faster loading times with focused functionality
        - **Reliability**: One dashboard's errors won't affect the other
        - **Flexibility**: Deploy or share dashboards individually
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #718096;">Built with Streamlit • Data powered by Supabase</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
