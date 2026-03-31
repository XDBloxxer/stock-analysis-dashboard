"""
Main Dashboard - 3 TABS
"""

import streamlit as st
from datetime import datetime

try:
    import pytz
    _EST = pytz.timezone("US/Eastern")
    def _now_et():
        return datetime.now(_EST)
except ImportError:
    def _now_et():
        return datetime.utcnow()

st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from dashboard_styles import DASHBOARD_CSS
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


def main():
    now    = _now_et()
    hour   = now.hour
    minute = now.minute
    is_weekday   = now.weekday() < 5
    after_open   = (hour > 9) or (hour == 9 and minute >= 30)
    before_close = hour < 16
    market_open  = is_weekday and after_open and before_close
    pre_market   = is_weekday and ((4 <= hour < 9) or (hour == 9 and minute < 30))
    after_hours  = is_weekday and (16 <= hour < 20)

    if market_open:
        dot_cls, label, color = "live",    "Open",        "var(--green-bright)"
    elif pre_market:
        dot_cls, label, color = "warning", "Pre-Market",  "var(--amber-bright)"
    elif after_hours:
        dot_cls, label, color = "warning", "After Hours", "var(--amber-bright)"
    else:
        dot_cls, label, color = "idle",    "Closed",      "var(--text-2)"

    date_str = now.strftime("%a %d %b %Y")
    time_str = now.strftime("%H:%M")

    # ── Header — use columns to avoid Streamlit's nested-div rendering bug ────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(
            '<div style="font-family:\'DM Mono\',monospace;font-size:0.52rem;letter-spacing:0.35em;'
            'color:var(--cyan);text-transform:uppercase;opacity:0.65;margin-bottom:6px;">'
            'Market Intelligence Terminal</div>'
            '<h1 style="margin:0;line-height:0.9;letter-spacing:0.06em;">Stock Analysis</h1>',
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:flex-end;gap:16px;padding-top:6px;">'
            '<div style="text-align:right;">'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.58rem;letter-spacing:0.12em;color:var(--text-2);text-transform:uppercase;">{date_str}</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:1.05rem;font-weight:300;color:var(--text-1);">{time_str} <span style="font-size:0.6rem;color:var(--text-2);">ET</span></div>'
            '</div>'
            '<div style="width:1px;height:32px;background:var(--border-mid);flex-shrink:0;"></div>'
            f'<div style="display:flex;align-items:center;gap:8px;padding:7px 14px;background:var(--bg-2);border:1px solid var(--border-mid);border-radius:var(--radius-sm);white-space:nowrap;">'
            f'<span class="status-dot {dot_cls}"></span>'
            f'<span style="font-family:\'DM Mono\',monospace;font-size:0.62rem;letter-spacing:0.18em;text-transform:uppercase;color:{color};">{label}</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr style="margin:16px 0 22px;">', unsafe_allow_html=True)

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
        st.stop()

    tab1, tab2, tab3 = st.tabs([
        "Daily Winners",
        "ML Predictions",
        "Strategy Backtesting",
    ])

    with tab1: render_daily_winners_tab()
    with tab2: render_ml_predictions_tab()
    with tab3: render_backtesting_tab()


if __name__ == "__main__":
    main()
