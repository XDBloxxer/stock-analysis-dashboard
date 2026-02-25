"""
db.py — Single Supabase client, shared across all tabs.
Import get_supabase_client() from here instead of defining per-tab.
"""

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
