"""
Main Dashboard - 3 TABS
Communicates with tradingview-analysis repo via GitHub Actions + Supabase

Changes v3:
  - Refined Terminal Noir aesthetic (dashboard_styles.py v4)
  - Cleaner header layout — less inline HTML noise
  - All caching logic unchanged
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
    col_title, col_status = st.columns([5, 2])

    now = _now_et()
    with col_title:
        st.markdown(f"""
        <div style="display:flex; align-items:baseline; gap:16px; padding:4px 0 2px;">
            <h1 style="margin:0; padding:0;">Stock Analysis</h1>
            <span style="
                font-family: 'DM Mono', monospace;
                font-size: 0.6rem;
                letter-spacing: 0.22em;
                color: var(--text-2);
                text-transform: uppercase;
                padding-bottom: 2px;
            ">{now.strftime("%a %d %b %Y · %H:%M")} ET</span>
        </div>
        """, unsafe_allow_html=True)

    with col_status:
        now_et = _now_et()
        hour, minute = now_et.hour, now_et.minute
        is_weekday   = now_et.weekday() < 5
        after_open   = (hour > 9) or (hour == 9 and minute >= 30)
        before_close = hour < 16
        market_open  = is_weekday and after_open and before_close
        pre_market   = is_weekday and ((hour >= 4 and hour < 9) or (hour == 9 and minute < 30))
        after_hours  = is_weekday and (16 <= hour < 20)

        if market_open:
            dot_cls, label, color = "live",    "Market Open",   "var(--green)"
        elif pre_market:
            dot_cls, label, color = "warning", "Pre-Market",    "var(--amber)"
        elif after_hours:
            dot_cls, label, color = "warning", "After Hours",   "var(--amber)"
        else:
            dot_cls, label, color = "idle",    "Market Closed", "var(--text-2)"

        st.markdown(f"""
        <div style="display:flex; justify-content:flex-end; padding-top:8px;">
            <div style="
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: var(--bg-2);
                border: 1px solid var(--border);
                border-radius: var(--radius-sm);
                padding: 6px 14px;
            ">
                <span class="status-dot {dot_cls}"></span>
                <span style="font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:0.14em; color:{color}; text-transform:uppercase;">{label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

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
        "Daily Winners",
        "ML Predictions",
        "Strategy Backtesting",
    ])

    with tab1:
        render_daily_winners_tab()
    with tab2:
        render_ml_predictions_tab()
    with tab3:
        render_backtesting_tab()


if __name__ == "__main__":
    main()
