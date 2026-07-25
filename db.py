"""
db.py — Single Supabase client, shared across all tabs.
Import get_supabase_client() from here instead of defining per-tab.
"""

import time
import streamlit as st
import os
from supabase import create_client, Client


@st.cache_resource
def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url")
    key = os.environ.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("key")
    if not url or not key:
        st.error("❌ Supabase credentials not configured!")
        st.stop()
    return create_client(url, key)


def log_debug_error(source: str, error: Exception) -> None:
    """
    Append a (source, error, timestamp) entry to st.session_state so failures
    that are otherwise swallowed (to keep the UI clean) are still visible
    somewhere — surfaced in the System Info tab's "Debug Log" expander.
    """
    errors = st.session_state.setdefault("_debug_errors", [])
    errors.append({
        "time":   time.strftime("%H:%M:%S"),
        "source": source,
        "error":  str(error),
    })
    # Keep the log bounded so a bad connection doesn't grow this unbounded.
    if len(errors) > 50:
        del errors[:-50]


def run_with_retry(fn, *, retries: int = 3, backoff: float = 0.4, source: str = "query"):
    """
    Run a zero-arg callable (typically a Supabase `.execute()` call wrapped
    in a lambda) with a small retry/backoff loop, since transient network
    blips shouldn't surface as a hard failure to the user.

    Re-raises the last exception if every attempt fails, after logging each
    failed attempt via log_debug_error.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            log_debug_error(f"{source} (attempt {attempt}/{retries})", e)
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise last_exc
