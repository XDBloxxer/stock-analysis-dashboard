"""
Main Dashboard - 3 TABS
"""

import streamlit as st
from datetime import datetime, date, timedelta

try:
    import pytz
    _EST = pytz.timezone("US/Eastern")
    def _now_et():
        return datetime.now(_EST)
except ImportError:
    def _now_et():
        return datetime.utcnow()


# ── NYSE market holiday calendar ──────────────────────────────────────────────
# Fixed-date + floating-date US market holidays, computed rather than hardcoded
# per-year so the schedule stays correct without an annual edit. Observed-date
# shifting (Sat -> preceding Fri, Sun -> following Mon) is applied for the
# fixed-date holidays, matching NYSE convention.
def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """weekday: Monday=0 ... Sunday=6. n=1 for first, n=3 for third, etc."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    d += timedelta(days=offset + 7 * (n - 1))
    return d


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher) for Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed(d: date) -> date:
    """NYSE convention: Saturday holidays observed the preceding Friday,
    Sunday holidays observed the following Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nyse_holidays(year: int) -> set[date]:
    good_friday = _easter_sunday(year) - timedelta(days=2)
    return {
        _observed(date(year, 1, 1)),                          # New Year's Day
        _nth_weekday_of_month(year, 1, 0, 3),                 # MLK Day (3rd Mon Jan)
        _nth_weekday_of_month(year, 2, 0, 3),                 # Presidents' Day (3rd Mon Feb)
        good_friday,                                          # Good Friday
        _last_weekday_of_month(year, 5, 0),                   # Memorial Day (last Mon May)
        _observed(date(year, 6, 19)),                         # Juneteenth
        _observed(date(year, 7, 4)),                          # Independence Day
        _nth_weekday_of_month(year, 9, 0, 1),                 # Labor Day (1st Mon Sep)
        _nth_weekday_of_month(year, 11, 3, 4),                # Thanksgiving (4th Thu Nov)
        _observed(date(year, 12, 25)),                        # Christmas
    }


def _is_market_holiday(d: date) -> bool:
    return d in _nyse_holidays(d.year)

st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from dashboard_styles import DASHBOARD_CSS, COMPACT_CSS
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


def main():
    now    = _now_et()
    hour   = now.hour
    minute = now.minute
    is_weekday    = now.weekday() < 5
    is_holiday    = _is_market_holiday(now.date())
    is_trading_day = is_weekday and not is_holiday
    after_open   = (hour > 9) or (hour == 9 and minute >= 30)
    before_close = hour < 16
    market_open  = is_trading_day and after_open and before_close
    pre_market   = is_trading_day and ((4 <= hour < 9) or (hour == 9 and minute < 30))
    after_hours  = is_trading_day and (16 <= hour < 20)

    if market_open:
        dot_cls, label, color = "live",    "Open",        "var(--green-bright)"
    elif pre_market:
        dot_cls, label, color = "warning", "Pre-Market",  "var(--amber-bright)"
    elif after_hours:
        dot_cls, label, color = "warning", "After Hours", "var(--amber-bright)"
    elif is_weekday and is_holiday:
        dot_cls, label, color = "idle",    "Holiday",     "var(--text-2)"
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

    col_hr, col_toggle = st.columns([6, 1])
    with col_hr:
        st.markdown('<hr style="margin:16px 0 22px;">', unsafe_allow_html=True)
    with col_toggle:
        compact = st.checkbox(
            "Compact", value=st.session_state.get("compact_mode", False),
            key="compact_mode", help="Tighter spacing — more rows/charts visible per screen.",
        )
    if compact:
        st.markdown(COMPACT_CSS, unsafe_allow_html=True)

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
        "Today's Picks",
        "Daily Winners",
        "Strategy Backtesting",
    ])

    with tab1: render_ml_predictions_tab()
    with tab2: render_daily_winners_tab()
    with tab3: render_backtesting_tab()


if __name__ == "__main__":
    main()
