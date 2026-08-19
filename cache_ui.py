"""
cache_ui.py — Shared "Refresh / Clear Cache" button pair with a confirmation
guard on Clear, used identically across tab_daily_winners.py,
tab_ml_predictions.py, and tab_backtesting.py.

Previously this was copy-pasted (with tiny message-wording drift) into all
three tab modules. Centralized here so there's one implementation to
maintain instead of three that quietly diverge over time.

v2: tucked behind a collapsed expander instead of sitting as a standing
button row on every tab — these are occasional maintenance actions, not
something that should compete for attention with the data above it. Also
drops the emoji icons in favor of plain text, matching the rest of the
"instrument panel" styling instead of reading as a generic AI-generated
button pair.
"""

import streamlit as st


def render_cache_buttons(tab_id: str, warning_message: str = None):
    """
    Renders a Refresh / Clear Cache control pair for the given tab_id,
    tucked inside a small collapsed-by-default expander. On Clear, shows a
    confirmation prompt (Confirm / Cancel) before actually reporting a
    confirmed clear back to the caller.

    Returns (refresh_clicked: bool, clear_confirmed: bool).
    """
    if warning_message is None:
        warning_message = (
            "This clears ALL cached data. Select <strong>Confirm</strong> to proceed."
        )

    confirm_key = f"{tab_id}_confirm_clear"

    st.markdown('<div class="cache-controls-expander">', unsafe_allow_html=True)
    with st.expander("Data controls", expanded=False):
        col_r, col_c, _ = st.columns([1, 1, 5])
        with col_r:
            st.markdown('<div class="btn-utility util-refresh">', unsafe_allow_html=True)
            refresh = st.button("Refresh", key=f"{tab_id}_refresh_top", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_c:
            st.markdown('<div class="btn-utility util-danger">', unsafe_allow_html=True)
            clear = st.button("Clear cache", key=f"{tab_id}_clear_top", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if clear:
            st.session_state[confirm_key] = True

        confirmed = False
        if st.session_state.get(confirm_key):
            st.markdown(
                f'<div class="cache-warning">{warning_message}</div>',
                unsafe_allow_html=True,
            )
            cc1, cc2, _ = st.columns([1, 1, 5])
            with cc1:
                st.markdown('<div class="btn-utility util-confirm">', unsafe_allow_html=True)
                if st.button("Confirm", key=f"{tab_id}_confirm_yes", use_container_width=True):
                    confirmed = True
                    st.session_state[confirm_key] = False
                st.markdown('</div>', unsafe_allow_html=True)
            with cc2:
                st.markdown('<div class="btn-utility util-cancel">', unsafe_allow_html=True)
                if st.button("Cancel", key=f"{tab_id}_confirm_no", use_container_width=True):
                    st.session_state[confirm_key] = False
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    return refresh, confirmed
