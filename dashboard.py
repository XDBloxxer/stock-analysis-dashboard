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

    st.markdown(f"""
    <div style="
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        padding-bottom: 20px;
        border-bottom: 1px solid var(--border-mid);
        margin-bottom: 22px;
    ">
        <div>
            <div style="
                font-family: 'DM Mono', monospace;
                font-size: 0.52rem;
                letter-spacing: 0.35em;
                color: var(--cyan);
                text-transform: uppercase;
                margin-bottom: 7px;
                opacity: 0.65;
            ">Market Intelligence Terminal</div>
            <h1 style="margin:0; line-height:0.9; letter-spacing:0.06em;">Stock Analysis</h1>
        </div>

        <div style="display:flex; align-items:center; gap:20px; padding-bottom:3px;">
            <div style="text-align:right; line-height:1.7;">
                <div style="font-family:'DM Mono',monospace; font-size:0.58rem; letter-spacing:0.12em; color:var(--text-2); text-transform:uppercase;">{now.strftime("%a %d %b %Y")}</div>
                <div style="font-family:'DM Mono',monospace; font-size:1.1rem; font-weight:300; letter-spacing:0.06em; color:var(--text-1);">{now.strftime("%H:%M")}<span style="color:var(--text-2);font-size:0.65rem;"> ET</span></div>
            </div>
            <div style="width:1px; height:36px; background:var(--border-mid);"></div>
            <div style="display:flex; align-items:center; gap:8px; padding:8px 16px; background:var(--bg-2); border:1px solid var(--border-mid); border-radius:var(--radius-sm);">
                <span class="status-dot {dot_cls}"></span>
                <span style="font-family:'DM Mono',monospace; font-size:0.62rem; letter-spacing:0.18em; text-transform:uppercase; color:{color};">{label}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
