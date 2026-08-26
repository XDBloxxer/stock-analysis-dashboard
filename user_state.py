"""
user_state.py — Small local persistence layer for cross-session settings
and the global watchlist.

Streamlit's st.session_state resets every browser session, so anything a
user configures (max positions, commission fee, default exit strategy,
compact mode, starred symbols) was being lost the moment the tab closed.
This module persists a small JSON file to disk next to the app and loads
it back into session_state at the start of the next session.

This is deliberately a single shared local file, not a per-user database
table — appropriate for a single-operator/small-team dashboard, not a
multi-tenant one. If the filesystem turns out to be read-only (some
hosting environments mount the app read-only), writes just fail silently
and the app falls back to session-only behavior — exactly today's
behavior — rather than crashing.
"""

import json
import os
import streamlit as st

_STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".user_settings.json")

_DEFAULTS = {
    "compact_mode": False,
    "max_positions": 10,
    "commission_fee": 0.0,
    "slippage_bps": 10.0,
    "max_deploy_pct": 100,
    "default_exit_mode": "Close at end of day",
    "watchlist": [],
}


def _read_store() -> dict:
    try:
        with open(_STORE_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _write_store(data: dict) -> bool:
    try:
        with open(_STORE_PATH, "w") as f:
            json.dump(data, f)
        return True
    except Exception:
        return False


def load_into_session_state():
    """Call once near the top of the app, before any widget that reads
    these keys is instantiated. Only fills in keys not already present in
    session_state, so it never clobbers an in-progress interaction — just
    seeds first-paint defaults from the last session's saved values."""
    if st.session_state.get("_user_settings_loaded"):
        return
    stored = _read_store()
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = stored.get(key, default)
    st.session_state["_user_settings_loaded"] = True


def persist(**kwargs) -> None:
    """Persist one or more settings to disk immediately and mirror them
    into session_state, e.g. persist(compact_mode=True, max_positions=8).
    Cheap enough to call on every rerun for a couple of scalar settings —
    this isn't a hot loop."""
    stored = _read_store()
    stored.update(kwargs)
    for k, v in kwargs.items():
        st.session_state[k] = v
    _write_store(stored)


# ── Global watchlist ───────────────────────────────────────────────────────
def get_watchlist() -> list:
    load_into_session_state()
    return list(st.session_state.get("watchlist", []))


def is_watchlisted(symbol: str) -> bool:
    if not symbol:
        return False
    return symbol.strip().upper() in get_watchlist()


def toggle_watchlist(symbol: str) -> bool:
    """Adds/removes `symbol` from the global watchlist (shared across every
    tab — Today's Picks, Daily Winners, etc.). Returns the new state
    (True = now on the watchlist)."""
    symbol = symbol.strip().upper()
    if not symbol:
        return False
    wl = get_watchlist()
    if symbol in wl:
        wl.remove(symbol)
        now_on = False
    else:
        wl.append(symbol)
        now_on = True
    persist(watchlist=wl)
    return now_on


def star_button(symbol: str, key: str, size_label: bool = False) -> None:
    """Renders a small ★/☆ toggle button for `symbol` and flips its
    watchlist membership on click. `key` must be unique per call site
    (usually f"{TAB_ID}_star_{symbol}_{extra_disambiguator}")."""
    starred = is_watchlisted(symbol)
    label = ("★ Watchlisted" if size_label else "★") if starred else ("☆ Watchlist" if size_label else "☆")
    if st.button(label, key=key, help=f"{'Remove' if starred else 'Add'} {symbol} {'from' if starred else 'to'} the global watchlist"):
        toggle_watchlist(symbol)
        st.rerun()
