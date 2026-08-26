"""
command_center.py — Landing summary strip + global watchlist bar.

Renders a slim row above the three tabs so there's a single at-a-glance
read (today's top signal, recent backtest edge, market status) before
drilling into any one tab — using the same hero-metric styling the tabs
already use for their own headline numbers, so it reads as part of the
same instrument panel rather than a bolted-on widget.

Deliberately runs its own tiny, cheap Supabase queries (top-1 / small
aggregate rows) instead of importing the heavier per-tab caching layers in
tab_ml_predictions.py / tab_backtesting.py, which are built to pull much
larger slices for their own tabs.
"""

import streamlit as st
import pandas as pd
from datetime import date, timedelta

from db import get_supabase_client, run_with_retry, log_debug_error
from dashboard_styles import render_hero_metric
import user_state

_TTL = 300  # 5 min — this is a summary glance, not a live feed


@st.cache_data(show_spinner=False, ttl=_TTL)
def _get_top_signal():
    """Highest-probability signal from the most recent prediction date."""
    try:
        client = get_supabase_client()

        def _run():
            return (
                client.table("ml_explosion_predictions")
                .select("prediction_date,symbol,signal,explosion_probability,target_gain_pct")
                .order("prediction_date", desc=True)
                .order("explosion_probability", desc=True)
                .limit(1)
                .execute()
            )

        response = run_with_retry(_run, source="command_center._get_top_signal")
        rows = response.data or []
        return rows[0] if rows else None
    except Exception as e:
        log_debug_error("command_center._get_top_signal", e)
        return None


@st.cache_data(show_spinner=False, ttl=_TTL)
def _get_recent_edge(lookback_days: int = 30):
    """Win rate + avg gain of resolved predictions over the trailing
    `lookback_days` — a cheap proxy for 'current backtest edge' without
    running a full simulation just for the summary strip."""
    try:
        client = get_supabase_client()
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        def _run():
            return (
                client.table("ml_prediction_accuracy")
                .select("prediction_date,became_winner,actual_gain_pct")
                .gte("prediction_date", cutoff)
                .execute()
            )

        response = run_with_retry(_run, source="command_center._get_recent_edge")
        rows = response.data or []
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["actual_gain_pct"] = pd.to_numeric(df["actual_gain_pct"], errors="coerce")
        return {
            "win_rate": float(df["became_winner"].astype(bool).mean() * 100),
            "avg_gain": float(df["actual_gain_pct"].mean()),
            "n": int(len(df)),
        }
    except Exception as e:
        log_debug_error("command_center._get_recent_edge", e)
        return None


def render_command_center(market_label: str, market_dot_cls: str):
    """Slim 3-up hero-metric row: today's top signal, recent backtest edge,
    market status — followed by the global watchlist bar if it's non-empty."""
    top = _get_top_signal()
    edge = _get_recent_edge()

    c1, c2, c3 = st.columns(3)
    with c1:
        if top and top.get("symbol"):
            prob = top.get("explosion_probability")
            prob_str = f"{prob * 100:.0f}% conf." if prob is not None else ""
            sub = " · ".join(s for s in [top.get("signal", "—"), prob_str] if s)
            accent = "green" if top.get("signal") == "STRONG BUY" else "cyan"
            render_hero_metric("Today's Top Signal", top["symbol"], sub=sub, accent=accent)
        else:
            render_hero_metric("Today's Top Signal", "—", sub="no predictions yet", accent="cyan")
    with c2:
        if edge:
            accent = "green" if edge["win_rate"] >= 50 else "amber"
            render_hero_metric(
                "30D Backtest Edge",
                f"{edge['win_rate']:.0f}% win rate",
                sub=f"avg {edge['avg_gain']:+.1f}% · {edge['n']} resolved",
                accent=accent,
            )
        else:
            render_hero_metric("30D Backtest Edge", "—", sub="not enough resolved trades yet", accent="cyan")
    with c3:
        dot_accent = {"live": "green", "warning": "amber"}.get(market_dot_cls, "cyan")
        render_hero_metric("Market Status", market_label, sub="see clock above for countdown", accent=dot_accent)

    render_watchlist_bar()


def render_watchlist_bar():
    """Compact chip strip of every starred symbol — visible regardless of
    which tab is active, since starring itself happens inside individual
    tabs (Today's Picks live table, Daily Winners symbol search)."""
    wl = user_state.get_watchlist()
    if not wl:
        return
    chips = "".join(
        '<span style="display:inline-flex;align-items:center;gap:4px;padding:4px 10px;'
        'margin:0 6px 6px 0;background:var(--bg-2);border:1px solid var(--border-mid);'
        'border-radius:999px;font-family:\'DM Mono\',monospace;font-size:0.68rem;'
        f'letter-spacing:0.04em;color:var(--text-0);">★ {sym}</span>'
        for sym in sorted(wl)
    )
    st.markdown(
        '<div style="display:flex;align-items:center;flex-wrap:wrap;margin:14px 0 4px;">'
        '<span style="font-family:\'DM Mono\',monospace;font-size:0.58rem;letter-spacing:0.18em;'
        'text-transform:uppercase;color:var(--text-2);margin-right:10px;white-space:nowrap;">Watchlist</span>'
        f'{chips}'
        '</div>',
        unsafe_allow_html=True,
    )
