"""
cache_ui.py — Shared "Refresh / Clear Cache" button pair with a confirmation
guard on Clear, used identically across tab_daily_winners.py,
tab_ml_predictions.py, and tab_backtesting.py.

Previously this was copy-pasted (with tiny message-wording drift) into all
three tab modules. Centralized here so there's one implementation to
maintain instead of three that quietly diverge over time.
"""

import streamlit as st


def render_cache_buttons(tab_id: str, warning_message: str = None):
    """
    Renders a Refresh / Clear Cache button pair for the given tab_id.
    On Clear, shows a confirmation prompt (Confirm / Cancel) before
    actually reporting a confirmed clear back to the caller.

    Returns (refresh_clicked: bool, clear_confirmed: bool).
    """
    if warning_message is None:
        warning_message = (
            "⚠️ This will wipe ALL cached data. Click <strong>Confirm Clear</strong> "
            "to proceed."
        )

    confirm_key = f"{tab_id}_confirm_clear"

    col_r, col_c, _ = st.columns([1, 1, 5])
    with col_r:
        st.markdown('<div class="btn-refresh">', unsafe_allow_html=True)
        refresh = st.button("🔄 Refresh", key=f"{tab_id}_refresh_top", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_c:
        st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
        clear = st.button("🗑️ Clear Cache", key=f"{tab_id}_clear_top", use_container_width=True)
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
            st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
            if st.button("✓ Confirm Clear", key=f"{tab_id}_confirm_yes", use_container_width=True):
                confirmed = True
                st.session_state[confirm_key] = False
            st.markdown('</div>', unsafe_allow_html=True)
        with cc2:
            if st.button("✕ Cancel", key=f"{tab_id}_confirm_no", use_container_width=True):
                st.session_state[confirm_key] = False
                st.rerun()

    return refresh, confirmed
