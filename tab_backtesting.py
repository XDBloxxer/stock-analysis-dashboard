"""
Backtesting Tab Module - st.cache_data PERSISTENT CACHE

CACHE STRATEGY:
  - Data is stored in @st.cache_data on the SERVER PROCESS, not in session_state.
  - Survives browser tab closes, page refreshes, and re-opening the page
    as long as the Streamlit server process is running.
  - On first load: fetches all strategies + results for every strategy_id.
  - Refresh button: fetches only strategies newer than the latest cached
    created_at, loads results for new strategies only.
    Existing strategy results are never re-fetched.
  - Clear Cache: calls .clear() on all cached functions → full re-fetch.
  - Switching strategies or sub-tabs: zero egress.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import os
import requests
from supabase import create_client, Client

TAB_ID = "backtesting"

CHART_THEME = {
    'plot_bgcolor': 'rgba(26, 29, 41, 0.6)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'font': dict(color='#e8eaf0'),
    'title_font': dict(size=18, color='#ffffff'),
    'xaxis': dict(gridcolor='rgba(255, 255, 255, 0.1)', color='#b8bcc8'),
    'yaxis': dict(gridcolor='rgba(255, 255, 255, 0.1)', color='#b8bcc8'),
}


# ── Supabase client ────────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase_client():
    supabase_url = os.environ.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url")
    supabase_key = os.environ.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("key")
    if not supabase_url or not supabase_key:
        st.error("❌ Supabase credentials not configured!")
        st.stop()
    return create_client(supabase_url, supabase_key)


# ── Cached DB fetchers ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _get_all_strategies() -> pd.DataFrame:
    """
    Fetch all strategies, ordered by created_at descending.
    Cached indefinitely until _get_all_strategies.clear() is called.
    """
    try:
        client   = get_supabase_client()
        response = (client.table("backtest_strategies")
                    .select("*")
                    .order("created_at", desc=True)
                    .limit(100)
                    .execute())
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
    """
    Fetch daily results and trades for a single strategy.
    Cached indefinitely per strategy_id.
    """
    try:
        client = get_supabase_client()
        daily_resp = (client.table("backtest_results")
                      .select("*")
                      .eq("strategy_id", strategy_id)
                      .order("test_date", desc=False)
                      .limit(1000)
                      .execute())
        trades_resp = (client.table("backtest_trades")
                       .select("*")
                       .eq("strategy_id", strategy_id)
                       .order("signal_date", desc=True)
                       .limit(1000)
                       .execute())
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
        min_r  = (client.table("historical_market_data")
                  .select("date").order("date", desc=False).limit(1).execute())
        max_r  = (client.table("historical_market_data")
                  .select("date").order("date", desc=True).limit(1).execute())
        if min_r.data and max_r.data:
            return (datetime.fromisoformat(min_r.data[0]['date']).date(),
                    datetime.fromisoformat(max_r.data[0]['date']).date())
        return None, None
    except Exception as e:
        st.warning(f"Could not get date range: {e}")
        return None, None


# ── Cache control ──────────────────────────────────────────────────────────────

def clear_all_cache():
    """Wipe all cached backtesting data → full re-fetch on next render."""
    _get_all_strategies.clear()
    _get_strategy_results.clear()
    _get_date_range.clear()


def refresh_cache():
    """
    Append-only refresh.
    Checks for strategies newer than the latest cached created_at.
    If found, clears _get_all_strategies so it re-fetches the full list,
    then pre-warms results for the new strategy IDs.
    Existing strategy result caches are untouched.
    """
    existing = _get_all_strategies()  # reads from cache

    if existing.empty:
        _get_all_strategies.clear()
        return

    latest_created = existing['created_at'].max()

    try:
        client   = get_supabase_client()
        response = (client.table("backtest_strategies")
                    .select("id,created_at")
                    .gt("created_at", latest_created)
                    .order("created_at", desc=True)
                    .limit(100)
                    .execute())
        new_ids = [row['id'] for row in response.data] if response.data else []
    except Exception as e:
        st.error(f"Error checking for new strategies: {e}")
        return

    if not new_ids:
        st.toast("✅ Cache is already up to date — no new strategies found.")
        return

    # Clear strategy list so it re-fetches with new entries included
    _get_all_strategies.clear()
    _get_all_strategies()  # re-populate immediately

    # Pre-warm results for new strategies only
    for sid in new_ids:
        _get_strategy_results(sid)

    st.toast(f"✅ Loaded {len(new_ids)} new strategy(ies).")


# ── Chart helpers ──────────────────────────────────────────────────────────────
def create_performance_chart(trades_df: pd.DataFrame):
    if trades_df.empty:
        return None

    date_col = next((c for c in ['signal_date', 'date', 'trade_date'] if c in trades_df.columns), None)
    if date_col is None:
        st.warning("No date column found in trades data")
        return None

    trades_df = trades_df.copy()
    trades_df[date_col] = pd.to_datetime(trades_df[date_col]).dt.date

    daily_stats = trades_df.groupby(date_col).agg({'matched_criteria': 'sum'}).reset_index()

    if 'hit_target' in trades_df.columns:
        tp = trades_df[trades_df['hit_target'] == True].groupby(date_col).size()
        daily_stats = daily_stats.merge(
            tp.to_frame('true_positives'), left_on=date_col, right_index=True, how='left'
        )
        daily_stats['true_positives'] = daily_stats['true_positives'].fillna(0).astype(int)
    else:
        daily_stats['true_positives'] = 0

    daily_stats.columns = [date_col, 'total_matches', 'true_positives']
    daily_stats['false_positives'] = daily_stats['total_matches'] - daily_stats['true_positives']

    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                        subplot_titles=("Cumulative Matches", "Daily Metrics"),
                        vertical_spacing=0.1)

    ds = daily_stats.sort_values(date_col)
    ds['cumulative_matches'] = ds['total_matches'].cumsum()

    fig.add_trace(go.Scatter(
        x=ds[date_col], y=ds['cumulative_matches'], mode='lines', name='Total Matches',
        line=dict(color='#667eea', width=2), fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)',
    ), row=1, col=1)

    fig.add_trace(go.Bar(x=daily_stats[date_col], y=daily_stats['true_positives'],
                         name='True Positives',  marker_color='#10b981'), row=2, col=1)
    fig.add_trace(go.Bar(x=daily_stats[date_col], y=daily_stats['false_positives'],
                         name='False Positives', marker_color='#ef4444'), row=2, col=1)

    fig.update_layout(height=600, showlegend=True, **CHART_THEME)
    fig.update_xaxes(**CHART_THEME['xaxis'])
    fig.update_yaxes(**CHART_THEME['yaxis'])
    return fig


def create_confusion_matrix(trades_df: pd.DataFrame):
    if trades_df.empty:
        return None

    tp = len(trades_df[(trades_df['matched_criteria'] == True)  & (trades_df['hit_target'] == True)])
    fp = len(trades_df[(trades_df['matched_criteria'] == True)  & (trades_df['hit_target'] == False)])
    fn = len(trades_df[(trades_df['matched_criteria'] == False) & (trades_df['hit_target'] == True)])
    tn = len(trades_df[(trades_df['matched_criteria'] == False) & (trades_df['hit_target'] == False)])

    fig = go.Figure(data=go.Heatmap(
        z=[[tp, fp], [fn, tn]],
        x=['Target Hit', 'Target Missed'],
        y=['Matched Criteria', 'Missed Criteria'],
        text=[['True Positive', 'False Positive'], ['False Negative', 'True Negative']],
        texttemplate='%{text}<br>%{z}', textfont={"size": 14, "color": "#ffffff"},
        colorscale=[[0, '#2d3142'], [1, '#667eea']], showscale=False,
    ))
    fig.update_layout(title="<b>Confusion Matrix</b>", height=400, **CHART_THEME)
    return fig


def create_gain_distribution(trades_df: pd.DataFrame):
    if trades_df.empty or 'actual_gain_pct' not in trades_df.columns:
        return None

    matched_gains   = trades_df[(trades_df['matched_criteria'] == True)  & trades_df['actual_gain_pct'].notna()]['actual_gain_pct']
    unmatched_gains = trades_df[(trades_df['matched_criteria'] == False) & trades_df['actual_gain_pct'].notna()]['actual_gain_pct']

    fig = go.Figure()
    if not matched_gains.empty:
        fig.add_trace(go.Histogram(x=matched_gains,   name='Matched Criteria',
                                   opacity=0.75, marker_color='#059669', nbinsx=30))
    if not unmatched_gains.empty:
        fig.add_trace(go.Histogram(x=unmatched_gains, name='Missed Criteria',
                                   opacity=0.75, marker_color='#d97706', nbinsx=30))

    fig.update_layout(title="<b>Gain Distribution</b>", xaxis_title="Gain %",
                      yaxis_title="Frequency", barmode='overlay', height=400, **CHART_THEME)
    fig.update_xaxes(**CHART_THEME['xaxis'])
    fig.update_yaxes(**CHART_THEME['yaxis'])
    return fig


def create_exit_analysis_chart(trades_df: pd.DataFrame):
    if trades_df.empty:
        return None

    required = ['actual_gain_pct', 'matched_criteria']
    if not all(c in trades_df.columns for c in required):
        return None

    high_col = next((c for c in ['max_possible_gain_pct', 'high_pct', 'exit_high'] if c in trades_df.columns), None)
    low_col  = next((c for c in ['max_drawdown_pct', 'low_pct', 'exit_low']         if c in trades_df.columns), None)
    if high_col is None or low_col is None:
        return None

    matched = trades_df[trades_df['matched_criteria'] == True].copy()
    if matched.empty:
        return None

    date_col = next((c for c in ['signal_date', 'date', 'trade_date'] if c in matched.columns), None)
    matched  = matched.sort_values(date_col) if date_col else matched.reset_index(drop=True)
    matched['trade_num'] = range(len(matched))

    fig = go.Figure()
    hover = matched['symbol'] if 'symbol' in matched.columns else None

    fig.add_trace(go.Scatter(x=matched['trade_num'], y=matched['actual_gain_pct'],
                             mode='markers', name='Actual Gain (Close)',
                             marker=dict(size=8, color='#1e40af'),
                             text=hover,
                             hovertemplate='%{text}<br>Gain: %{y:.2f}%<extra></extra>'))
    fig.add_trace(go.Scatter(x=matched['trade_num'], y=matched[high_col],
                             mode='markers', name='Max Possible Gain',
                             marker=dict(size=6, color='#10b981', symbol='triangle-up'),
                             hovertemplate='Max: %{y:.2f}%<extra></extra>'))
    fig.add_trace(go.Scatter(x=matched['trade_num'], y=matched[low_col],
                             mode='markers', name='Max Drawdown',
                             marker=dict(size=6, color='#ef4444', symbol='triangle-down'),
                             hovertemplate='Drawdown: %{y:.2f}%<extra></extra>'))

    fig.update_layout(title="<b>Exit Analysis: Actual vs Intraday Range</b>",
                      xaxis_title="Trade Number", yaxis_title="Gain %",
                      height=400, hovermode='closest', **CHART_THEME)
    fig.update_xaxes(**CHART_THEME['xaxis'])
    fig.update_yaxes(**CHART_THEME['yaxis'])
    return fig


def create_cumulative_pnl(trades_df: pd.DataFrame):
    if trades_df.empty:
        return None
    if 'actual_gain_pct' not in trades_df.columns or 'matched_criteria' not in trades_df.columns:
        return None

    matched = trades_df[trades_df['matched_criteria'] == True].copy()
    if matched.empty:
        return None

    matched = matched.sort_values('signal_date')
    matched['cumulative_pnl'] = matched['actual_gain_pct'].cumsum()

    fig = go.Figure(go.Scatter(
        x=matched['signal_date'], y=matched['cumulative_pnl'],
        mode='lines', name='Cumulative P&L',
        line=dict(color='#667eea', width=3),
        fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.2)',
    ))
    fig.update_layout(title="<b>Cumulative P&L</b>",
                      xaxis_title="Date", yaxis_title="Cumulative Gain %",
                      height=400, hovermode='x unified', **CHART_THEME)
    fig.update_xaxes(**CHART_THEME['xaxis'])
    fig.update_yaxes(**CHART_THEME['yaxis'])
    return fig


# ── Save / run helpers ─────────────────────────────────────────────────────────
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
            # Clear strategy list so the new strategy appears on next render
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

        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/{workflow_id}/dispatches"
        resp = requests.post(url, headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }, json={"ref": "main", "inputs": {"strategy_id": str(strategy_id)}})

        if resp.status_code == 204:
            st.success(f"✅ Backtest workflow triggered for strategy {strategy_id}")
            st.info("The backtest is now running. Refresh this page in a few minutes to see results.")
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

    col_r, col_c = st.columns(2)
    with col_r:
        refresh_clicked = st.button("🔄 Refresh", key=f"{TAB_ID}_refresh", use_container_width=True)
    with col_c:
        clear_clicked = st.button("🗑️ Clear Cache", key=f"{TAB_ID}_clear_cache", use_container_width=True)

    if clear_clicked:
        clear_all_cache()
        st.rerun()

    if refresh_clicked:
        refresh_cache()
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["View Results", "Create Strategy", "Manage Strategies"])

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
                col1.metric("Period",      f"{strategy['start_date']} to {strategy['end_date']}")
                col2.metric("Target Gain", f"{strategy['target_min_gain_pct']}%")
                col3.metric("Status",      strategy['run_status'])

                with st.expander("Indicator Criteria"):
                    criteria = strategy['indicator_criteria']
                    if isinstance(criteria, list):
                        for i, cond in enumerate(criteria, 1):
                            if cond.get('comparison_type') == 'indicator':
                                st.write(f"{i}. **{cond.get('indicator')}** {cond.get('operator')} **{cond.get('compare_to')}**")
                            else:
                                st.write(f"{i}. **{cond.get('indicator')}** {cond.get('operator')} {cond.get('value')}")

                # Results are cached per strategy_id — zero egress on repeat views
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

                            win_rate     = len(winners) / len(matched_trades) * 100 if len(matched_trades) > 0 else 0
                            avg_winner   = winners['actual_gain_pct'].mean() if len(winners) > 0 else 0
                            avg_loser    = losers['actual_gain_pct'].mean()  if len(losers)  > 0 else 0
                            total_gains  = winners['actual_gain_pct'].sum()  if len(winners) > 0 else 0
                            total_losses = abs(losers['actual_gain_pct'].sum()) if len(losers) > 0 else 0
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

                    fig1 = create_performance_chart(trades_df)
                    if fig1:
                        st.plotly_chart(fig1, use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        fig2 = create_confusion_matrix(trades_df)
                        if fig2:
                            st.plotly_chart(fig2, use_container_width=True)
                    with col2:
                        fig5 = create_cumulative_pnl(trades_df)
                        if fig5:
                            st.plotly_chart(fig5, use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        fig3 = create_gain_distribution(trades_df)
                        if fig3:
                            st.plotly_chart(fig3, use_container_width=True)
                    with col2:
                        fig4 = create_exit_analysis_chart(trades_df)
                        if fig4:
                            st.plotly_chart(fig4, use_container_width=True)
                        else:
                            st.info("Exit analysis requires actual_gain_pct, max_possible_gain_pct, and max_drawdown_pct columns.")

                    st.markdown("### Trade Log")
                    if not trades_df.empty:
                        display_trades = trades_df.sort_values('signal_date', ascending=False).head(100)
                        display_cols   = [c for c in ['signal_date', 'symbol', 'matched_criteria',
                                                       'hit_target', 'actual_gain_pct', 'high_pct', 'low_pct']
                                          if c in display_trades.columns]
                        st.dataframe(display_trades[display_cols], use_container_width=True, height=400)
                        st.download_button("📥 Download All Trades",
                                           trades_df.to_csv(index=False),
                                           f"trades_strategy_{selected_id}.csv", "text/csv")
                    else:
                        st.info("No trades to display.")

                elif strategy['run_status'] == 'pending':
                    st.info("This strategy hasn't been run yet. Click 'Run Backtest' in Manage Strategies.")
                elif strategy['run_status'] == 'running':
                    st.info("This strategy is currently running. Refresh in a few minutes.")
                elif strategy['run_status'] == 'failed':
                    st.error("This strategy failed to run. Check the logs for details.")

    # ──────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Create New Strategy")

        min_date, max_date = _get_date_range()

        with st.form("create_strategy_form"):
            name        = st.text_input("Strategy Name", placeholder="e.g., High RSI Momentum")
            description = st.text_area("Description",    placeholder="Describe your strategy…")

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date",
                    value=min_date or (datetime.now().date() - timedelta(days=365)),
                    **({"min_value": min_date, "max_value": max_date} if min_date else {}))
            with col2:
                end_date = st.date_input("End Date",
                    value=max_date or datetime.now().date(),
                    **({"min_value": min_date, "max_value": max_date} if min_date else {}))

            target_gain = st.number_input("Target Min Gain %", min_value=0.1, max_value=100.0,
                                          value=5.0, step=0.5)

            st.markdown("#### Indicator Criteria")
            st.info("Add conditions that define when to signal a trade.")

            num_conditions = st.number_input("Number of conditions", min_value=1, max_value=10, value=2)
            criteria = []

            for i in range(num_conditions):
                st.markdown(f"**Condition {i+1}**")
                c1, c2, c3, c4 = st.columns([3, 2, 2, 3])
                with c1:
                    indicator = st.text_input("Indicator", key=f"ind_{i}",
                                              placeholder="e.g., rsi, macd.macd, adx")
                with c2:
                    operator  = st.selectbox("Operator", [">", "<", ">=", "<=", "=="], key=f"op_{i}")
                with c3:
                    comp_type = st.selectbox("Compare to", ["value", "indicator"], key=f"comp_{i}")
                with c4:
                    if comp_type == "value":
                        compare_value = st.number_input("Value", key=f"val_{i}", value=0.0, step=0.1)
                        criteria.append({'indicator': indicator, 'operator': operator,
                                         'comparison_type': 'value', 'value': compare_value})
                    else:
                        compare_indicator = st.text_input("Indicator", key=f"comp_ind_{i}",
                                                          placeholder="e.g., ema20")
                        criteria.append({'indicator': indicator, 'operator': operator,
                                         'comparison_type': 'indicator', 'compare_to': compare_indicator})

            submitted = st.form_submit_button("Create Strategy", use_container_width=True)
            if submitted:
                if not name:
                    st.error("Please provide a strategy name.")
                elif start_date >= end_date:
                    st.error("Start date must be before end date.")
                else:
                    strategy_id = save_strategy(name, description, start_date, end_date,
                                                criteria, target_gain)
                    if strategy_id:
                        st.success(f"✅ Strategy created! ID: {strategy_id}")
                        st.info("Go to 'Manage Strategies' to run the backtest.")

    # ──────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Manage Strategies")
        strategies_df = _get_all_strategies()

        if strategies_df.empty:
            st.info("No strategies found.")
        else:
            for _, row in strategies_df.iterrows():
                with st.expander(f"**{row['name']}** — {row['run_status']}", expanded=False):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**Period:** {row['start_date']} to {row['end_date']}")
                        st.write(f"**Target:** {row['target_min_gain_pct']}%")
                    with col2:
                        st.write(f"**Status:** {row['run_status']}")
                        st.write(f"**Created:** {row['created_at'][:10]}")
                    with col3:
                        if row['run_status'] in ['pending', 'failed']:
                            if st.button("▶️ Run", key=f"run_{row['id']}", use_container_width=True):
                                with st.spinner("Triggering backtest…"):
                                    run_backtest_via_github(row['id'])
                        elif row['run_status'] == 'running':
                            st.info("Running…")
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
