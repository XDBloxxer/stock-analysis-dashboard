"""
Main Dashboard - 3 TABS
Communicates with tradingview-analysis repo via GitHub Actions + Supabase

Changes v2:
  - Fixed EST timezone (uses pytz, not naive datetime.now())
  - Improved market status logic (correct 9:30–16:00 ET window)
  - More restrained status dot (slower pulse via CSS)
  - Cache clear button now styled as danger/red
  - Refresh button styled as cyan accent
"""

import streamlit as st
import os
from datetime import datetime
from pathlib import Path

try:
    import pytz
    _EST = pytz.timezone("US/Eastern")
    def _now_et():
        return datetime.now(_EST)
except ImportError:
    # Fallback if pytz not installed — user should add pytz to requirements.txt
    def _now_et():
        return datetime.utcnow()

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
            border: 1px solid rgba(0,212,255,0.25);
            border-radius: 6px;
            display: flex; align-items: center; justify-content: center;
            background: rgba(0,212,255,0.04);
            font-size: 22px;
            margin-top: 4px;
        ">📊</div>
        """, unsafe_allow_html=True)

    with col_title:
        now = _now_et()
        st.markdown(f"""
        <div>
            <h1 style="margin:0; padding:0; line-height:1.1;">STOCK ANALYSIS</h1>
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.63rem;
                letter-spacing: 0.2em;
                color: #3a5070;
                text-transform: uppercase;
                margin-top: 2px;
            ">Market Intelligence Terminal &nbsp;·&nbsp; {now.strftime("%A %d %b %Y &nbsp;·&nbsp; %H:%M")} ET</div>
        </div>
        """, unsafe_allow_html=True)

    with col_status:
        now_et = _now_et()
        hour   = now_et.hour
        minute = now_et.minute
        # Market open 9:30–16:00 ET, Mon–Fri
        is_weekday    = now_et.weekday() < 5
        after_open    = (hour > 9) or (hour == 9 and minute >= 30)
        before_close  = hour < 16
        market_open   = is_weekday and after_open and before_close

        # Pre-market 4:00–9:30
        pre_market = is_weekday and (
            (hour >= 4 and hour < 9) or (hour == 9 and minute < 30)
        )
        # After-hours 16:00–20:00
        after_hours = is_weekday and (16 <= hour < 20)

        if market_open:
            status_dot, status_text, status_color = "live",    "MARKET OPEN",   "#00ff88"
        elif pre_market:
            status_dot, status_text, status_color = "warning", "PRE-MARKET",    "#ffb800"
        elif after_hours:
            status_dot, status_text, status_color = "warning", "AFTER HOURS",   "#ffb800"
        else:
            status_dot, status_text, status_color = "idle",    "MARKET CLOSED", "#3a5070"

        st.markdown(f"""
        <div style="text-align: right; padding-top: 8px; font-family: 'JetBrains Mono', monospace;">
            <div style="
                display: inline-flex;
                align-items: center;
                gap: 7px;
                background: rgba(0,0,0,0.3);
                border: 1px solid rgba(0,212,255,0.08);
                border-radius: 4px;
                padding: 6px 13px;
            ">
                <span class="status-dot {status_dot}"></span>
                <span style="font-size: 0.63rem; letter-spacing: 0.12em; color: {status_color};">{status_text}</span>
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
