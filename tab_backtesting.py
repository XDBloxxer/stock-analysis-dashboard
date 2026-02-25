"""
Backtesting Tab Module - st.cache_data PERSISTENT CACHE  (v2)

NEW in v2:
  - Imports get_supabase_client from db.py (single source)
  - Uses CHART_THEME/LAYOUT/AXIS_STYLE from chart_utils.py
  - Cumulative P&L promoted as the HERO chart (not match count)
  - Match count chart clearly labelled as "(count, not $)"
  - Refresh (cyan) / Clear Cache (red danger) buttons with confirmation guard
  - Live form validation feedback (checkmarks per field, submit disabled until valid)
  - Strategy creation: real-time condition preview
  - Manage Strategies: Run button styled green (btn-action)
  - Flatten inner tabs: View Results / Create / Manage all accessible at same level

CACHE STRATEGY: UNCHANGED — all fetching methods identical to v1.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import os
import requests

from db import get_supabase_client
from chart_utils import CHART_THEME, LAYOUT, AXIS_STYLE, COLORS

TAB_ID = "backtesting"


# ── Cached DB fetchers (UNCHANGED) ────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _get_all_strategies() -> pd.DataFrame:
    try:
        client   = get_supabase_client()
        response = (
            client.table("backtest_strategies")
            .select("*")
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )
        if not response.data:
            return pd.DataFrame()
        df = pd.DataFrame(response.data)
        if 'indicator_criteria' in df.columns:
            df['indicator_criteria'] = df['indicator_criteria'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        return df
    except Exception as e:
        st.error(f"Error loading strategies: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _get_strategy_results(strategy_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        client = get_supabase_client()
        daily_resp = (
            client.table("backtest_results")
            .select("*")
            .eq("strategy_id", strategy_id)
            .order("test_date", desc=False)
            .limit(1000)
            .execute()
        )
        trades_resp = (
            client.table("backtest_trades")
            .select("*")
            .eq("strategy_id", strategy_id)
            .order("signal_date", desc=True)
            .limit(1000)
            .execute()
        )
        daily_df  = pd.DataFrame(daily_resp.data)  if daily_resp.data  else pd.DataFrame()
        trades_df = pd.DataFrame(trades_resp.data) if trades_resp.data else pd.DataFrame()
        return daily_df, trades_df
    except Exception as e:
        st.error(f"Error loading strategy results: {e}")
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(show_spinner=False)
def _get_date_range() -> tuple:
    try:
        client = get_supabase_client()
        min_r  = client.table("historical_market_data").select("date").order("date", desc=False).limit(1).execute()
        max_r  = client.table("historical_market_data").select("date").order("date", desc=True).limit(1).execute()
        if min_r.data and max_r.data:
            return (
                datetime.fromisoformat(min_r.data[0]['date']).date(),
                datetime.fromisoformat(max_r.data[0]['date']).date(),
            )
        return None, None
    except Exception as e:
        st.warning(f"Could not get date range: {e}")
        return None, None


# ── Cache control (UNCHANGED logic) ───────────────────────────────────────────
def clear_all_cache():
    _get_all_strategies.clear()
    _get_strategy_results.clear()
    _get_date_range.clear()


def refresh_cache():
    existing = _get_all_strategies()
    if existing.empty:
        _get_all_strategies.clear()
        return
    latest_created = existing['created_at'].max()
    try:
        client   = get_supabase_client()
        response = (
            client.table("backtest_strategies")
            .select("id,created_at")
            .gt("created_at", latest_created)
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )
        new_ids = [row['id'] for row in response.data] if response.data else []
    except Exception as e:
        st.error(f"Error checking for new strategies: {e}")
        return
    if not new_ids:
        st.toast("✅ Cache is already up to date — no new strategies found.")
        return
    _get_all_strategies.clear()
    _get_all_strategies()
    for sid in new_ids:
        _get_strategy_results(sid)
    st.toast(f"✅ Loaded {len(new_ids)} new strategy(ies).")


# ── Shared button helper ───────────────────────────────────────────────────────
def _render_cache_buttons(tab_id: str):
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
            '<div class="cache-warning">⚠️ This will wipe ALL cached backtesting data. '
            'Click <strong>Confirm Clear</strong> to proceed.</div>',
            unsafe_allow_html=True
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


# ── Chart helpers (using shared theme) ────────────────────────────────────────
def create_performance_chart(trades_df: pd.DataFrame):
    """Match count chart — clearly labelled as count, not P&L."""
    if trades_df.empty:
        return None

    date_col = next((c for c in ['signal_date', 'date', 'trade_date'] if c in trades_df.columns), None)
    if date_col is None:
        return None

    trades_df = trades_df.copy()
    trades_df[date_col] = pd.to_datetime(trades_df[date_col]).dt.date

    daily_stats = trades_df.groupby(date_col).agg({'matched_criteria': 'sum'}).reset_index()

    if 'hit_target' in trades_df.columns:
        tp = trades_df[trades_df['hit_target'] == True].groupby(date_col).size()
        daily_stats = daily_stats.merge(tp.to_frame('true_positives'), left_on=date_col, right_index=True, how='left')
        daily_stats['true_positives'] = daily_stats['true_positives'].fillna(0).astype(int)
    else:
        daily_stats['true_positives'] = 0

    daily_stats.columns = [date_col, 'total_matches', 'true_positives']
    daily_stats['false_positives'] = daily_stats['total_matches'] - daily_stats['true_positives']

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        subplot_titles=("Cumulative Signal Matches (count, not $)", "Daily TP / FP Breakdown"),
        vertical_spacing=0.12,
    )
    ds = daily_stats.sort_values(date_col)
    ds['cumulative_matches'] = ds['total_matches'].cumsum()

    fig.add_trace(go.Scatter(
        x=ds[date_col], y=ds['cumulative_matches'],
        mode='lines', name='Cumulative Matches',
        line=dict(color=COLORS['primary'], width=2),
        fill='tozeroy', fillcolor='rgba(0,212,255,0.08)',
    ), row=1, col=1)

    fig.add_trace(go.Bar(x=daily_stats[date_col], y=daily_stats['true_positives'],
                         name='True Positives',  marker_color=COLORS['secondary'], opacity=0.85), row=2, col=1)
    fig.add_trace(go.Bar(x=daily_stats[date_col], y=daily_stats['false_positives'],
                         name='False Positives', marker_color=COLORS['red'],       opacity=0.85), row=2, col=1)

    fig.update_layout(height=560, showlegend=True, barmode='stack', **LAYOUT)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


def create_confusion_matrix(trades_df: pd.DataFrame):
    if trades_df.empty:
        return None

    tp = len(trades_df[(trades_df['matched_criteria'] == True)  & (trades_df['hit_target'] == True)])
    fp = len(trades_df[(trades_df['matched_criteria'] == True)  & (trades_df['hit_target'] == False)])
    fn = len(trades_df[(trades_df['matched_criteria'] == False) & (trades_df['hit_target'] == True)])
    tn = len(trades_df[(trades_df['matched_criteria'] == False) & (trades_df['hit_target'] == False)])

    from chart_utils import CONFUSION_COLORS
    fig = go.Figure(data=go.Heatmap(
        z=[[tp, fp], [fn, tn]],
        x=['Target Hit', 'Target Missed'],
        y=['Matched Criteria', 'Missed Criteria'],
        text=[[f"True Positive<br>{tp}", f"False Positive<br>{fp}"],
              [f"False Negative<br>{fn}", f"True Negative<br>{tn}"]],
        texttemplate='%{text}', textfont={"size": 13, "color": "#ffffff"},
        colorscale=[
            [0.0,  CONFUSION_COLORS['tn']],
            [0.33, CONFUSION_COLORS['fn']],
            [0.66, CONFUSION_COLORS['fp']],
            [1.0,  CONFUSION_COLORS['tp']],
        ],
        showscale=False,
    ))
    fig.update_layout(title="<b>Confusion Matrix</b>", height=380, **LAYOUT)
    return fig


def create_gain_distribution(trades_df: pd.DataFrame):
    if trades_df.empty or 'actual_gain_pct' not in trades_df.columns:
        return None

    matched   = trades_df[(trades_df['matched_criteria'] == True)  & trades_df['actual_gain_pct'].notna()]['actual_gain_pct']
    unmatched = trades_df[(trades_df['matched_criteria'] == False) & trades_df['actual_gain_pct'].notna()]['actual_gain_pct']

    fig = go.Figure()
    if not matched.empty:
        fig.add_trace(go.Histogram(x=matched,   name='Matched Criteria',  opacity=0.8,  marker_color=COLORS['secondary'], nbinsx=30))
    if not unmatched.empty:
        fig.add_trace(go.Histogram(x=unmatched, name='Missed Criteria',   opacity=0.75, marker_color=COLORS['amber'],     nbinsx=30))

    fig.update_layout(title="<b>Gain Distribution</b>", xaxis_title="Gain %", yaxis_title="Frequency", barmode='overlay', height=380, **LAYOUT)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


def create_exit_analysis_chart(trades_df: pd.DataFrame):
    if trades_df.empty:
        return None
    required = ['actual_gain_pct', 'matched_criteria']
    if not all(c in trades_df.columns for c in required):
        return None
    high_col = next((c for c in ['max_possible_gain_pct', 'high_pct', 'exit_high'] if c in trades_df.columns), None)
    low_col  = next((c for c in ['max_drawdown_pct', 'low_pct', 'exit_low']        if c in trades_df.columns), None)
    if high_col is None or low_col is None:
        return None

    matched = trades_df[trades_df['matched_criteria'] == True].copy()
    if matched.empty:
        return None

    date_col = next((c for c in ['signal_date', 'date', 'trade_date'] if c in matched.columns), None)
    matched  = matched.sort_values(date_col) if date_col else matched.reset_index(drop=True)
    matched['trade_num'] = range(len(matched))
    hover = matched['symbol'] if 'symbol' in matched.columns else None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=matched['trade_num'], y=matched['actual_gain_pct'],
        mode='markers', name='Actual Gain (Close)',
        marker=dict(size=8, color=COLORS['primary'], opacity=0.85),
        text=hover, hovertemplate='%{text}<br>Gain: %{y:.2f}%<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=matched['trade_num'], y=matched[high_col],
        mode='markers', name='Max Possible Gain',
        marker=dict(size=6, color=COLORS['secondary'], symbol='triangle-up'),
        hovertemplate='Max: %{y:.2f}%<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=matched['trade_num'], y=matched[low_col],
        mode='markers', name='Max Drawdown',
        marker=dict(size=6, color=COLORS['red'], symbol='triangle-down'),
        hovertemplate='Drawdown: %{y:.2f}%<extra></extra>',
    ))
    fig.update_layout(title="<b>Exit Analysis: Actual vs Intraday Range</b>", xaxis_title="Trade Number", yaxis_title="Gain %", height=380, hovermode='closest', **LAYOUT)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


def create_cumulative_pnl(trades_df: pd.DataFrame):
    """Hero P&L chart — the first chart users see."""
    if trades_df.empty:
        return None
    if 'actual_gain_pct' not in trades_df.columns or 'matched_criteria' not in trades_df.columns:
        return None

    matched = trades_df[trades_df['matched_criteria'] == True].copy()
    if matched.empty:
        return None

    matched = matched.sort_values('signal_date')
    matched['cumulative_pnl'] = matched['actual_gain_pct'].cumsum()

    # Color area green/red based on positive/negative
    final_pnl   = matched['cumulative_pnl'].iloc[-1]
    area_color  = 'rgba(0,255,136,0.08)' if final_pnl >= 0 else 'rgba(255,56,96,0.08)'
    line_color  = COLORS['secondary']    if final_pnl >= 0 else COLORS['red']

    fig = go.Figure(go.Scatter(
        x=matched['signal_date'], y=matched['cumulative_pnl'],
        mode='lines', name='Cumulative P&L (%)',
        line=dict(color=line_color, width=2.5),
        fill='tozeroy', fillcolor=area_color,
        hovertemplate='%{x}<br>P&L: %{y:+.2f}%<extra></extra>',
    ))
    fig.add_hline(y=0, line_color='rgba(255,255,255,0.12)', line_width=1)
    fig.update_layout(
        title=f"<b>Cumulative P&L</b>  <span style='color:{line_color}'>{final_pnl:+.2f}%</span>",
        xaxis_title="Date", yaxis_title="Cumulative Gain %",
        height=380, hovermode='x unified', **LAYOUT,
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


# ── Save / run helpers (UNCHANGED) ────────────────────────────────────────────
def save_strategy(name, description, start_date, end_date, indicator_criteria, target_min_gain_pct):
    try:
        client   = get_supabase_client()
        strategy = {
            'name': name, 'description': description,
            'start_date': str(start_date), 'end_date': str(end_date),
            'indicator_criteria': json.dumps(indicator_criteria),
            'target_min_gain_pct': target_min_gain_pct,
            'run_status': 'pending',
            'created_at': datetime.now().isoformat(),
        }
        response = client.table("backtest_strategies").insert(strategy).execute()
        if response.data:
            _get_all_strategies.clear()
            return response.data[0]['id']
        return None
    except Exception as e:
        st.error(f"Error saving strategy: {e}")
        return None


def run_backtest_via_github(strategy_id: int) -> bool:
    try:
        github_token = os.environ.get("GITHUB_TOKEN") or st.secrets.get("G_TOKEN")
        repo_owner   = os.environ.get("GITHUB_REPO_OWNER") or st.secrets.get("GITHUB_REPO_OWNER")
        repo_name    = os.environ.get("GITHUB_REPO_NAME")  or st.secrets.get("GITHUB_REPO_NAME")
        workflow_id  = os.environ.get("GITHUB_WORKFLOW_ID") or st.secrets.get("GITHUB_WORKFLOW_ID", "backtest.yml")

        if not all([github_token, repo_owner, repo_name]):
            st.error("❌ GitHub credentials not configured.")
            return False

        url  = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/{workflow_id}/dispatches"
        resp = requests.post(url, headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }, json={"ref": "main", "inputs": {"strategy_id": str(strategy_id)}})

        if resp.status_code == 204:
            st.success(f"✅ Backtest workflow triggered for strategy {strategy_id}")
            st.info("Refresh in a few minutes to see results.")
            return True
        st.error(f"Failed to trigger workflow: {resp.status_code} - {resp.text}")
        return False
    except Exception as e:
        st.error(f"Error triggering GitHub workflow: {e}")
        return False


# ── Main entry point ───────────────────────────────────────────────────────────
def render_backtesting_tab():
    st.subheader("Strategy Backtesting")
    st.markdown("Test your trading strategies against historical data")

    refresh_clicked, clear_confirmed = _render_cache_buttons(TAB_ID)
    if clear_confirmed:
        clear_all_cache()
        st.rerun()
    if refresh_clicked:
        refresh_cache()
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["📊 View Results", "✏️ Create Strategy", "⚙️ Manage Strategies"])

    # ──────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Strategy Results")
        strategies_df = _get_all_strategies()

        if strategies_df.empty:
            st.info("No strategies found. Create one in the 'Create Strategy' tab.")
        else:
            strategy_names = dict(zip(strategies_df['id'], strategies_df['name']))
            selected_id = st.selectbox(
                "Select Strategy:",
                options=list(strategy_names.keys()),
                format_func=lambda x: strategy_names[x],
                key="view_strategy_select",
            )

            if selected_id:
                strategy = strategies_df[strategies_df['id'] == selected_id].iloc[0]

                col1, col2, col3 = st.columns(3)
                col1.metric("Period",      f"{strategy['start_date']} → {strategy['end_date']}")
                col2.metric("Target Gain", f"{strategy['target_min_gain_pct']}%")
                col3.metric("Status",      strategy['run_status'])

                with st.expander("Indicator Criteria"):
                    criteria = strategy['indicator_criteria']
                    if isinstance(criteria, list):
                        for i, cond in enumerate(criteria, 1):
                            if cond.get('comparison_type') == 'indicator':
                                st.write(f"{i}. `{cond.get('indicator')} {cond.get('operator')} {cond.get('compare_to')}`")
                            else:
                                st.write(f"{i}. `{cond.get('indicator')} {cond.get('operator')} {cond.get('value')}`")

                daily_df, trades_df = _get_strategy_results(selected_id)

                if strategy['run_status'] == 'completed' and not daily_df.empty:
                    st.markdown("---")
                    st.markdown("### Results Summary")

                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Total Matches",   strategy.get('total_matches', 0))
                    col2.metric("True Positives",  strategy.get('true_positives', 0))
                    col3.metric("False Positives", strategy.get('false_positives', 0))
                    col4.metric("Missed Opps",     strategy.get('missed_opportunities', 0))
                    acc = strategy.get('accuracy_pct', 0)
                    col5.metric("Accuracy",        f"{acc}%" if acc else "N/A")

                    if not trades_df.empty and 'actual_gain_pct' in trades_df.columns:
                        matched_trades = trades_df[trades_df['matched_criteria'] == True]
                        if not matched_trades.empty:
                            winners = matched_trades[matched_trades['actual_gain_pct'] > 0]
                            losers  = matched_trades[matched_trades['actual_gain_pct'] < 0]

                            win_rate      = len(winners) / len(matched_trades) * 100 if len(matched_trades) > 0 else 0
                            avg_winner    = winners['actual_gain_pct'].mean() if len(winners) > 0 else 0
                            avg_loser     = losers['actual_gain_pct'].mean()  if len(losers)  > 0 else 0
                            total_gains   = winners['actual_gain_pct'].sum()  if len(winners) > 0 else 0
                            total_losses  = abs(losers['actual_gain_pct'].sum()) if len(losers) > 0 else 0
                            profit_factor = total_gains / total_losses if total_losses > 0 else None

                            intraday_hits = 0
                            for col in ['hit_target_intraday', 'target_hit_intraday', 'intraday_hit']:
                                if col in matched_trades.columns:
                                    intraday_hits = (matched_trades[col] == True).sum()
                                    break
                            intraday_rate = intraday_hits / len(matched_trades) * 100 if len(matched_trades) > 0 else 0

                            st.markdown("#### Advanced Metrics")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            col1.metric("Win Rate",          f"{win_rate:.1f}%")
                            col2.metric("Avg Winner",        f"{avg_winner:.2f}%")
                            col3.metric("Avg Loser",         f"{avg_loser:.2f}%")
                            col4.metric("Profit Factor",     f"{profit_factor:.2f}" if profit_factor else "N/A")
                            col5.metric("Intraday Hit Rate", f"{intraday_rate:.1f}%")

                    st.markdown("### Performance Charts")

                    # ── Cumulative P&L is the HERO chart — shown first ─────────────
                    fig_pnl = create_cumulative_pnl(trades_df)
                    if fig_pnl:
                        st.plotly_chart(fig_pnl, use_container_width=True)
                    else:
                        st.info("Cumulative P&L chart requires actual_gain_pct and matched_criteria columns.")

                    # ── Signal match chart below (clearly labelled count, not $) ───
                    fig1 = create_performance_chart(trades_df)
                    if fig1:
                        st.plotly_chart(fig1, use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        fig2 = create_confusion_matrix(trades_df)
                        if fig2:
                            st.plotly_chart(fig2, use_container_width=True)
                    with col2:
                        fig3 = create_gain_distribution(trades_df)
                        if fig3:
                            st.plotly_chart(fig3, use_container_width=True)

                    fig4 = create_exit_analysis_chart(trades_df)
                    if fig4:
                        st.plotly_chart(fig4, use_container_width=True)
                    else:
                        st.info("Exit analysis requires actual_gain_pct, max_possible_gain_pct, and max_drawdown_pct columns.")

                    st.markdown("### Trade Log")
                    if not trades_df.empty:
                        display_trades = trades_df.sort_values('signal_date', ascending=False).head(100)
                        display_cols   = [c for c in ['signal_date','symbol','matched_criteria','hit_target','actual_gain_pct','high_pct','low_pct'] if c in display_trades.columns]
                        st.dataframe(display_trades[display_cols], use_container_width=True, height=380)
                        st.download_button("📥 Download All Trades", trades_df.to_csv(index=False), f"trades_strategy_{selected_id}.csv", "text/csv")
                    else:
                        st.info("No trades to display.")

                elif strategy['run_status'] == 'pending':
                    st.info("This strategy hasn't been run yet. Go to ⚙️ Manage Strategies to trigger the backtest.")
                elif strategy['run_status'] == 'running':
                    st.info("⏳ This strategy is currently running. Refresh in a few minutes.")
                elif strategy['run_status'] == 'failed':
                    st.error("❌ This strategy failed to run. Check the GitHub Actions logs for details.")

    # ──────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Create New Strategy")

        min_date, max_date = _get_date_range()

        # ── Live validation state ──────────────────────────────────────────────
        name_key = f"{TAB_ID}_form_name"
        if name_key not in st.session_state:
            st.session_state[name_key] = ""

        name = st.text_input(
            "Strategy Name ✱",
            placeholder="e.g., High RSI Momentum",
            key=name_key,
        )
        name_ok = bool(name.strip())
        if name.strip():
            st.success("✓ Name looks good")
        elif name_key in st.session_state and st.session_state.get(f"{name_key}_touched"):
            st.error("✗ Strategy name is required")

        if name:
            st.session_state[f"{name_key}_touched"] = True

        description = st.text_area("Description (optional)", placeholder="Describe your strategy…")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date ✱",
                value=min_date or (datetime.now().date() - timedelta(days=365)),
                **({"min_value": min_date, "max_value": max_date} if min_date else {}),
            )
        with col2:
            end_date = st.date_input(
                "End Date ✱",
                value=max_date or datetime.now().date(),
                **({"min_value": min_date, "max_value": max_date} if min_date else {}),
            )

        date_ok = start_date < end_date
        if not date_ok:
            st.error("✗ Start date must be before end date")
        else:
            st.success(f"✓ Date range: {(end_date - start_date).days} days")

        target_gain = st.number_input("Target Min Gain %  ✱", min_value=0.1, max_value=100.0, value=5.0, step=0.5)

        st.markdown("#### Indicator Criteria")
        st.info("Each condition defines when a signal fires. All conditions must be true simultaneously.")

        num_conditions = st.number_input("Number of conditions", min_value=1, max_value=10, value=2)
        criteria       = []
        criteria_valid = True

        for i in range(num_conditions):
            st.markdown(f"**Condition {i+1}**")
            c1, c2, c3, c4 = st.columns([3, 2, 2, 3])
            with c1:
                indicator = st.text_input("Indicator", key=f"ind_{i}", placeholder="e.g., rsi, macd.macd, adx")
            with c2:
                operator  = st.selectbox("Operator", [">", "<", ">=", "<=", "=="], key=f"op_{i}")
            with c3:
                comp_type = st.selectbox("Compare to", ["value", "indicator"], key=f"comp_{i}")
            with c4:
                if comp_type == "value":
                    compare_value = st.number_input("Value", key=f"val_{i}", value=0.0, step=0.1)
                    criteria.append({'indicator': indicator, 'operator': operator, 'comparison_type': 'value', 'value': compare_value})
                else:
                    compare_indicator = st.text_input("Indicator", key=f"comp_ind_{i}", placeholder="e.g., ema20")
                    criteria.append({'indicator': indicator, 'operator': operator, 'comparison_type': 'indicator', 'compare_to': compare_indicator})

            if not indicator.strip():
                criteria_valid = False

        # Live preview of criteria
        if criteria:
            st.markdown("**Preview:**")
            preview_lines = []
            for i, cond in enumerate(criteria, 1):
                ind = cond.get('indicator', '?')
                op  = cond.get('operator', '?')
                if cond.get('comparison_type') == 'indicator':
                    rhs = cond.get('compare_to', '?')
                else:
                    rhs = str(cond.get('value', '?'))
                icon = "✓" if ind.strip() else "✗"
                preview_lines.append(f"  {icon} `{ind} {op} {rhs}`")
            st.code("\n".join(preview_lines), language=None)

        # Submit only enabled when all valid
        all_valid = name_ok and date_ok and criteria_valid

        if not all_valid:
            st.warning("⚠️ Fill in all required fields (✱) before saving.")

        st.markdown('<div class="btn-action">', unsafe_allow_html=True)
        if st.button(
            "✓ Create Strategy",
            key=f"{TAB_ID}_create",
            use_container_width=True,
            disabled=not all_valid,
        ):
            strategy_id = save_strategy(name, description, start_date, end_date, criteria, target_gain)
            if strategy_id:
                st.success(f"✅ Strategy created! ID: {strategy_id}")
                st.info("Go to ⚙️ Manage Strategies to run the backtest.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Manage Strategies")
        strategies_df = _get_all_strategies()

        if strategies_df.empty:
            st.info("No strategies found. Create one in the ✏️ Create Strategy tab.")
        else:
            # Status summary mini-metrics
            status_counts = strategies_df['run_status'].value_counts()
            s_cols = st.columns(len(status_counts))
            status_colors = {'completed': COLORS['secondary'], 'pending': COLORS['amber'], 'running': COLORS['primary'], 'failed': COLORS['red']}
            for i, (status, count) in enumerate(status_counts.items()):
                color = status_colors.get(status, '#6e8aaa')
                with s_cols[i]:
                    st.markdown(
                        f'<div style="font-family:JetBrains Mono;font-size:0.58rem;color:#3a5070;letter-spacing:.15em;text-transform:uppercase">{status}</div>'
                        f'<div style="font-size:1.6rem;font-weight:700;color:{color};font-family:JetBrains Mono">{count}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            for _, row in strategies_df.iterrows():
                status_icon = {'completed': '✅', 'pending': '⏳', 'running': '🔄', 'failed': '❌'}.get(row['run_status'], '•')
                with st.expander(f"{status_icon} **{row['name']}** — {row['run_status']}", expanded=False):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**Period:** {row['start_date']} → {row['end_date']}")
                        st.write(f"**Target:** {row['target_min_gain_pct']}%")
                    with col2:
                        st.write(f"**Status:** {row['run_status']}")
                        st.write(f"**Created:** {row['created_at'][:10]}")
                    with col3:
                        if row['run_status'] in ['pending', 'failed']:
                            st.markdown('<div class="btn-action">', unsafe_allow_html=True)
                            if st.button("▶ Run", key=f"run_{row['id']}", use_container_width=True):
                                with st.spinner("Triggering backtest…"):
                                    run_backtest_via_github(row['id'])
                            st.markdown('</div>', unsafe_allow_html=True)
                        elif row['run_status'] == 'running':
                            st.info("⏳ Running…")
                        else:
                            st.success("✅ Done")

                    if row.get('description'):
                        st.write(f"**Description:** {row['description']}")

                    st.write("**Criteria:**")
                    criteria = row['indicator_criteria']
                    if isinstance(criteria, list):
                        for i, cond in enumerate(criteria, 1):
                            if cond.get('comparison_type') == 'indicator':
                                st.write(f"{i}. `{cond.get('indicator')} {cond.get('operator')} {cond.get('compare_to')}`")
                            else:
                                st.write(f"{i}. `{cond.get('indicator')} {cond.get('operator')} {cond.get('value')}`")
