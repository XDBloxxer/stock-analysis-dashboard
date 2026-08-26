"""
command_center.py — Global watchlist bar.

Previously also rendered a 3-up hero-metric "command center" strip (top
signal, backtest edge, market status) above the tabs — dropped per
feedback since it largely repeated numbers already visible inside the
tabs themselves. Kept the watchlist bar, since starring a symbol in one
tab (Today's Picks live table, Daily Winners symbol search) and seeing it
flagged everywhere else was the actual cross-tab gap being solved.
"""

import streamlit as st

from db import log_debug_error
import user_state


def render_watchlist_bar():
    """Compact chip strip of every starred symbol — visible regardless of
    which tab is active, since starring itself happens inside individual
    tabs.

    Always renders the same single wrapper <div>, whether or not there are
    any starred symbols — an empty watchlist collapses to zero height via
    CSS rather than skipping the element entirely, so this block's element
    count never changes between reruns. That matters because it sits just
    above st.tabs() in dashboard.py, and Streamlit can lose track of which
    tab is active if the DOM structure preceding the tabs shifts between
    reruns (streamlit/streamlit#5069) — which happens on every widget
    interaction anywhere in the app, since Streamlit reruns the whole
    script top-to-bottom regardless of which tab is visible.
    """
    try:
        wl = user_state.get_watchlist()
    except Exception as e:
        log_debug_error("command_center.render_watchlist_bar", e)
        wl = []

    if wl:
        chips = "".join(
            '<span style="display:inline-flex;align-items:center;gap:4px;padding:4px 10px;'
            'margin:0 6px 6px 0;background:var(--bg-2);border:1px solid var(--border-mid);'
            'border-radius:999px;font-family:\'DM Mono\',monospace;font-size:0.68rem;'
            f'letter-spacing:0.04em;color:var(--text-0);">★ {sym}</span>'
            for sym in sorted(wl)
        )
        label_html = (
            '<span style="font-family:\'DM Mono\',monospace;font-size:0.58rem;letter-spacing:0.18em;'
            'text-transform:uppercase;color:var(--text-2);margin-right:10px;white-space:nowrap;">Watchlist</span>'
        )
        inner = f'{label_html}{chips}'
        wrap_style = "display:flex;align-items:center;flex-wrap:wrap;margin:2px 0 4px;"
    else:
        # Same wrapper element, just visually collapsed — keeps the DOM
        # shape identical to the non-empty case instead of omitting it.
        inner = ""
        wrap_style = "display:block;height:0;margin:0;overflow:hidden;"

    st.markdown(f'<div style="{wrap_style}">{inner}</div>', unsafe_allow_html=True)
