"""
Main Dashboard - 3 TABS
Communicates with tradingview-analysis repo via GitHub Actions + Supabase
"""

import streamlit as st
import os
from datetime import datetime
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from dashboard_styles import DASHBOARD_CSS
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


def main():
    """Main dashboard with 3 tabs"""

    # ── Header ────────────────────────────────────────────────────────────────
    col_logo, col_title, col_status = st.columns([1, 5, 2])

    with col_logo:
        st.markdown("""
        <div style="
            width: 48px; height: 48px;
            border: 1px solid rgba(0,212,255,0.3);
            border-radius: 6px;
            display: flex; align-items: center; justify-content: center;
            background: rgba(0,212,255,0.05);
            font-size: 22px;
            margin-top: 4px;
        ">📊</div>
        """, unsafe_allow_html=True)

    with col_title:
        now = datetime.now()
        st.markdown(f"""
        <div>
            <h1 style="margin:0; padding:0; line-height:1.1;">STOCK ANALYSIS</h1>
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.65rem;
                letter-spacing: 0.2em;
                color: #3a5070;
                text-transform: uppercase;
                margin-top: 2px;
            ">Market Intelligence Terminal &nbsp;·&nbsp; {now.strftime("%A %d %b %Y &nbsp;·&nbsp; %H:%M")} EST</div>
        </div>
        """, unsafe_allow_html=True)

    with col_status:
        market_hour = now.hour
        # Rough EST market hours (14:30–21:00 UTC)
        market_open = 9 <= market_hour <= 16
        status_dot  = "live" if market_open else "idle"
        status_text = "MARKET OPEN" if market_open else "MARKET CLOSED"
        status_color = "#00ff88" if market_open else "#3a5070"
        st.markdown(f"""
        <div style="
            text-align: right;
            padding-top: 8px;
            font-family: 'JetBrains Mono', monospace;
        ">
            <div style="
                display: inline-flex;
                align-items: center;
                gap: 7px;
                background: rgba(0,0,0,0.3);
                border: 1px solid rgba(0,212,255,0.1);
                border-radius: 4px;
                padding: 6px 14px;
            ">
                <span class="status-dot {status_dot}"></span>
                <span style="font-size: 0.65rem; letter-spacing: 0.12em; color: {status_color};">{status_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 6px;'></div>", unsafe_allow_html=True)

    # ── Credential check ──────────────────────────────────────────────────────
    if not st.secrets.get("supabase", {}).get("url") or not st.secrets.get("supabase", {}).get("key"):
        st.error("⚠️ Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_KEY.")
        st.stop()

    github_token = st.secrets.get("secrets", {}).get("G_TOKEN")
    github_repo  = st.secrets.get("secrets", {}).get("GITHUB_REPO_NAME", "XDBloxxer/tradingview-analysis")
    st.session_state.github_token = github_token
    st.session_state.github_repo  = github_repo

    try:
        from tab_daily_winners   import render_daily_winners_tab
        from tab_ml_predictions  import render_ml_predictions_tab
        from tab_backtesting     import render_backtesting_tab
    except ImportError as e:
        st.error(f"Error importing tab modules: {e}")
        st.info("Make sure all dashboard files are in the same directory")
        st.stop()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "◈  Daily Winners",
        "◈  ML Predictions",
        "◈  Strategy Backtesting",
    ])

    with tab1:
        render_daily_winners_tab()
    with tab2:
        render_ml_predictions_tab()
    with tab3:
        render_backtesting_tab()


if __name__ == "__main__":
    main()
