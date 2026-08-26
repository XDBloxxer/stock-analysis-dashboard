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


# ── NYSE early-close (half-day) calendar ──────────────────────────────────────
# Unlike full holidays, early closes (day-after-Thanksgiving, Christmas Eve,
# the occasional pre-July-4th half day) are NOT on a fixed formula — NYSE
# announces them by press release a couple years out, so this has to be a
# literal table that's manually re-verified/extended periodically. Sourced
# from NYSE Group's official 2025/2026/2027 holiday & early-closing calendar
# (nyse.com/trade/hours-calendars). All times ET.
EARLY_CLOSE_DATES = {
    date(2025, 7, 3):  (13, 0),   # day before observed Independence Day
    date(2025, 11, 28): (13, 0),  # day after Thanksgiving
    date(2025, 12, 24): (13, 0),  # Christmas Eve
    date(2026, 11, 27): (13, 0),  # day after Thanksgiving
    date(2026, 12, 24): (13, 0),  # Christmas Eve
    date(2027, 11, 26): (13, 0),  # day after Thanksgiving
}
# The table above is only verified through this date. NYSE typically
# announces the next year's calendar in the back half of the prior year, so
# start warning a few months before we run off the end of known coverage.
EARLY_CLOSE_TABLE_THROUGH = date(2027, 12, 31)
EARLY_CLOSE_WARN_WINDOW_DAYS = 90


def _early_close_time(d: date):
    """Returns (hour, minute) of an early close on `d`, or None for a normal
    session day."""
    return EARLY_CLOSE_DATES.get(d)


def _early_close_table_stale(d: date) -> bool:
    """True once we're within the warning window of (or past) the last date
    the early-close table is verified for."""
    return (EARLY_CLOSE_TABLE_THROUGH - d).days <= EARLY_CLOSE_WARN_WINDOW_DAYS

_FAVICON = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Ccircle cx='16' cy='16' r='13' fill='%2305070a' stroke='%23e0a83c' stroke-width='2'/%3E"
    "%3Ccircle cx='16' cy='16' r='5.5' fill='%23e0a83c'/%3E"
    "%3C/svg%3E"
)

st.set_page_config(
    page_title="Market Intelligence Terminal",
    page_icon=_FAVICON,  # custom SVG dot matching the exact brass accent hex,
                         # rather than a generic 🟡/📊 emoji glyph
    layout="wide",
    initial_sidebar_state="collapsed"
)

from dashboard_styles import DASHBOARD_CSS, COMPACT_CSS, inject_count_up_script, inject_mouse_glow_script, inject_live_clock_script, inject_price_scramble_script, render_boot_sequence, render_signature_footer
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


def main():
    import user_state
    user_state.load_into_session_state()
    render_boot_sequence()
    inject_count_up_script()
    inject_mouse_glow_script()
    inject_live_clock_script()
    inject_price_scramble_script()
    now    = _now_et()
    hour   = now.hour
    minute = now.minute
    is_weekday    = now.weekday() < 5
    is_holiday    = _is_market_holiday(now.date())
    is_trading_day = is_weekday and not is_holiday
    early_close   = _early_close_time(now.date()) if is_trading_day else None
    close_hour, close_minute = early_close if early_close else (16, 0)
    after_open   = (hour > 9) or (hour == 9 and minute >= 30)
    before_close = (hour < close_hour) or (hour == close_hour and minute < close_minute)
    market_open  = is_trading_day and after_open and before_close
    pre_market   = is_trading_day and ((4 <= hour < 9) or (hour == 9 and minute < 30))
    after_hours_start_min = close_hour * 60 + close_minute
    now_min = hour * 60 + minute
    after_hours  = is_trading_day and (after_hours_start_min <= now_min < 20 * 60)

    if market_open:
        dot_cls, label, color = "live",    "Open",        "var(--green-bright)"
        if early_close:
            label = "Open (Early Close)"
    elif pre_market:
        dot_cls, label, color = "warning", "Pre-Market",  "var(--amber-bright)"
    elif after_hours:
        dot_cls, label, color = "warning", "After Hours", "var(--amber-bright)"
    elif is_weekday and is_holiday:
        dot_cls, label, color = "idle",    "Holiday",     "var(--text-2)"
    else:
        dot_cls, label, color = "idle",    "Closed",      "var(--text-2)"

    date_str = now.strftime("%a %d %b %Y")
    time_str = now.strftime("%H:%M:%S")

    stale_table = _early_close_table_stale(now.date())

    # Next-session countdown label + target, mirrored client-side in JS
    # (see inject_live_clock_script) so this is just the first-paint value
    # before the per-second script takes over.
    if market_open:
        _next_dt = now.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0)
        countdown_label = "Closes in"
    elif pre_market:
        _next_dt = now.replace(hour=9, minute=30, second=0, microsecond=0)
        countdown_label = "Opens in"
    elif after_hours:
        _next_dt = now.replace(hour=20, minute=0, second=0, microsecond=0)
        countdown_label = "Ends in"
    else:
        _next_dt = now.replace(hour=4, minute=0, second=0, microsecond=0)
        if _next_dt <= now:
            _next_dt += timedelta(days=1)
        while not (_next_dt.weekday() < 5 and not _is_market_holiday(_next_dt.date())):
            _next_dt += timedelta(days=1)
        countdown_label = "Pre-market in"
    _remaining = max(_next_dt - now, timedelta(0))
    _h, _rem_s = divmod(int(_remaining.total_seconds()), 3600)
    _m, _s = divmod(_rem_s, 60)
    _d = _remaining.days
    countdown_str = f"{_d}d {_h:02d}:{_m:02d}:{_s:02d}" if _d else f"{_h:02d}:{_m:02d}:{_s:02d}"

    # First-paint urgency class — mirrors the thresholds inject_live_clock_script
    # applies client-side every second, so there's no color flash/jump the
    # moment the JS ticker takes over from this initial server render.
    _remaining_s = _remaining.total_seconds()
    if countdown_label == "Pre-market in":
        countdown_cls = ""
    elif _remaining_s <= 5 * 60:
        countdown_cls = " cd-critical"
    elif _remaining_s <= 15 * 60:
        countdown_cls = " cd-warn"
    else:
        countdown_cls = ""

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
            f'<div id="mit-date" style="font-family:\'DM Mono\',monospace;font-size:0.58rem;letter-spacing:0.12em;color:var(--text-2);text-transform:uppercase;">{date_str}</div>'
            f'<div id="mit-time" style="font-family:\'DM Mono\',monospace;font-size:1.6rem;font-weight:300;color:var(--text-1);letter-spacing:0.02em;">{time_str} <span style="font-size:0.75rem;color:var(--text-2);">ET</span></div>'
            f'<div id="mit-countdown" class="{countdown_cls.strip()}" style="font-family:\'DM Mono\',monospace;font-size:0.78rem;letter-spacing:0.04em;color:var(--text-2);margin-top:4px;">{countdown_label} {countdown_str}</div>'
            '</div>'
            '<div style="width:1px;height:32px;background:var(--border-mid);flex-shrink:0;"></div>'
            f'<div style="display:flex;align-items:center;gap:8px;padding:7px 14px;background:var(--bg-2);border:1px solid var(--border-mid);border-radius:var(--radius-sm);white-space:nowrap;">'
            f'<span id="mit-status-dot" class="status-dot {dot_cls}"></span>'
            f'<span id="mit-status-label" style="font-family:\'DM Mono\',monospace;font-size:0.62rem;letter-spacing:0.18em;text-transform:uppercase;color:{color};">{label}</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Early-close table staleness warning — only shown once we're inside the
    # warning window of EARLY_CLOSE_TABLE_THROUGH (see dashboard.py's
    # EARLY_CLOSE_DATES table). Kept in sync client-side too, via
    # #mit-stale-warning, since inject_live_clock_script owns the ticking
    # clock and could in principle cross midnight into the warning window
    # without a Streamlit rerun.
    st.markdown(
        f'<div id="mit-stale-warning" style="display:{"flex" if stale_table else "none"};align-items:center;gap:8px;'
        'margin:-10px 0 14px;padding:6px 12px;background:rgba(251,191,36,0.08);'
        'border:1px solid rgba(251,191,36,0.28);border-radius:var(--radius-sm);width:fit-content;'
        'margin-left:auto;">'
        '<span style="font-size:0.75rem;color:var(--amber-bright);">⚠</span>'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:0.58rem;letter-spacing:0.03em;color:var(--text-2);">'
        f'Early-close calendar verified only through {EARLY_CLOSE_TABLE_THROUGH.strftime("%b %Y")} — update EARLY_CLOSE_DATES</span>'
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
    if compact != user_state._read_store().get("compact_mode", False):
        user_state.persist(compact_mode=compact)
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

    # ── Command center — single at-a-glance read before drilling into a tab ──
    # IMPORTANT: this block's structure must render identically (same number
    # of elements, every run) regardless of data/errors — st.tabs() below
    # loses track of which tab is active if the DOM structure preceding it
    # shifts between reruns (streamlit/streamlit#5069), and since Streamlit
    # reruns the whole script top-to-bottom for ANY widget interaction in
    # ANY tab, an unstable block here was resetting users to the first tab
    # every time they touched a control elsewhere. So: no conditional
    # try/except that can skip rendering, and the strip itself always
    # renders the same three columns every run.
    from command_center import render_command_center
    render_command_center(label, dot_cls)
    st.markdown('<hr style="margin:18px 0 22px;">', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "Today's Picks",
        "Daily Winners",
        "Strategy Backtesting",
    ])

    with tab1: render_ml_predictions_tab()
    with tab2: render_daily_winners_tab()
    with tab3: render_backtesting_tab()

    st.markdown(
        '<div class="app-footer">'
        'Data provided for informational purposes only — not financial advice. '
        'Signals and backtests are model-generated and may lag real market conditions.'
        '</div>',
        unsafe_allow_html=True,
    )
    render_signature_footer()


if __name__ == "__main__":
    main()
