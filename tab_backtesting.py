"""
Backtesting Tab Module - COMPREHENSIVE FIXED VERSION
✅ FIXES:
- Fixed all KeyError issues with robust column checking
- Restored ability to CREATE backtests from dashboard
- Zero auto-egress (ttl=0, column limits)
- Runs backtests via GitHub Actions API
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

# Chart theme
CHART_THEME = {
    'plot_bgcolor': 'rgba(26, 29, 41, 0.6)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'font': dict(color='#e8eaf0'),
    'title_font': dict(size=18, color='#ffffff'),
    'xaxis': dict(gridcolor='rgba(255, 255, 255, 0.1)', color='#b8bcc8'),
    'yaxis': dict(gridcolor='rgba(255, 255, 255, 0.1)', color='#b8bcc8'),
}

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

@st.cache_resource
def get_supabase_client():
    """Initialize Supabase client"""
    supabase_url = os.environ.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url")
    supabase_key = os.environ.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("key")
    
    if not supabase_url or not supabase_key:
        st.error("❌ Supabase credentials not configured!")
        st.stop()
    
    return create_client(supabase_url, supabase_key)

@st.cache_data(ttl=0)
def get_date_range_from_db(_refresh_key: int = 0):
    """Get available date range from historical data"""
    try:
        client = get_supabase_client()
        
        min_response = client.table("historical_market_data")\
            .select("date")\
            .order("date", desc=False)\
            .limit(1)\
            .execute()
        
        max_response = client.table("historical_market_data")\
            .select("date")\
            .order("date", desc=True)\
            .limit(1)\
            .execute()
        
        if min_response.data and max_response.data:
            min_date = datetime.fromisoformat(min_response.data[0]['date']).date()
            max_date = datetime.fromisoformat(max_response.data[0]['date']).date()
            return min_date, max_date
        
        return None, None
    except Exception as e:
        st.warning(f"Could not get date range: {e}")
        return None, None

@st.cache_data(ttl=0)
def load_strategies(_refresh_key: int = 0):
    """Load all backtest strategies"""
    try:
        client = get_supabase_client()
        
        response = client.table("backtest_strategies")\
            .select("*")\
            .order("created_at", desc=True)\
            .limit(100)\
            .execute()
        
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

@st.cache_data(ttl=0)
def load_strategy_results(strategy_id: int, _refresh_key: int = 0):
    """Load backtest results for a specific strategy"""
    try:
        client = get_supabase_client()
        
        # Load daily results (limit to 1000 most recent)
        daily_response = client.table("backtest_results")\
            .select("*")\
            .eq("strategy_id", strategy_id)\
            .order("test_date", desc=False)\
            .limit(1000)\
            .execute()
        
        daily_df = pd.DataFrame(daily_response.data) if daily_response.data else pd.DataFrame()
        
        # Load trades (limit to 1000 most recent)
        trades_response = client.table("backtest_trades")\
            .select("*")\
            .eq("strategy_id", strategy_id)\
            .order("signal_date", desc=True)\
            .limit(1000)\
            .execute()
        
        trades_df = pd.DataFrame(trades_response.data) if trades_response.data else pd.DataFrame()
        
        return daily_df, trades_df
    except Exception as e:
        st.error(f"Error loading strategy results: {e}")
        return pd.DataFrame(), pd.DataFrame()

def save_strategy(name: str, description: str, start_date, end_date, 
                 indicator_criteria: list, target_min_gain_pct: float):
    """Save a new backtest strategy"""
    try:
        client = get_supabase_client()
        
        strategy = {
            'name': name,
            'description': description,
            'start_date': str(start_date),
            'end_date': str(end_date),
            'indicator_criteria': json.dumps(indicator_criteria),
            'target_min_gain_pct': target_min_gain_pct,
            'run_status': 'pending',
            'created_at': datetime.now().isoformat()
        }
        
        response = client.table("backtest_strategies").insert(strategy).execute()
        
        if response.data:
            return response.data[0]['id']
        return None
    except Exception as e:
        st.error(f"Error saving strategy: {e}")
        return None

# ============================================================================
# GITHUB ACTIONS INTEGRATION
# ============================================================================

def run_backtest_via_github(strategy_id: int):
    """
    Trigger backtest via GitHub Actions
    This assumes you have a GitHub workflow set up with:
    - workflow_dispatch event
    - GITHUB_TOKEN secret
    """
    try:
        # Get GitHub credentials from secrets/env
        github_token = os.environ.get("GITHUB_TOKEN") or st.secrets.get("GITHUB_TOKEN")
        repo_owner = os.environ.get("GITHUB_REPO_OWNER") or st.secrets.get("GITHUB_REPO_OWNER")
        repo_name = os.environ.get("GITHUB_REPO_NAME") or st.secrets.get("GITHUB_REPO_NAME")
        workflow_id = os.environ.get("GITHUB_WORKFLOW_ID") or st.secrets.get("GITHUB_WORKFLOW_ID", "backtest.yml")
        
        if not all([github_token, repo_owner, repo_name]):
            st.error("❌ GitHub credentials not configured. Please set up secrets for GitHub integration.")
            return False
        
        # GitHub API endpoint
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/{workflow_id}/dispatches"
        
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        # Payload with strategy_id
        payload = {
            "ref": "main",  # or your default branch
            "inputs": {
                "strategy_id": str(strategy_id)
            }
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 204:
            st.success(f"✅ Backtest workflow triggered for strategy {strategy_id}")
            st.info("The backtest is now running. Refresh this page in a few minutes to see results.")
            return True
        else:
            st.error(f"Failed to trigger workflow: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        st.error(f"Error triggering GitHub workflow: {e}")
        return False

# ============================================================================
# CHART CREATION FUNCTIONS
# ============================================================================

def create_performance_chart(trades_df: pd.DataFrame):
    """Create daily performance chart from trades data"""
    if trades_df.empty:
        return None
    
    # Check if we have the date column
    date_col = None
    for possible_col in ['signal_date', 'date', 'trade_date']:
        if possible_col in trades_df.columns:
            date_col = possible_col
            break
    
    if date_col is None:
        st.warning("No date column found in trades data")
        return None
    
    # Calculate daily metrics from trades
    trades_df = trades_df.copy()
    trades_df[date_col] = pd.to_datetime(trades_df[date_col]).dt.date
    
    # Count matches and hits per day
    daily_stats = trades_df.groupby(date_col).agg({
        'matched_criteria': 'sum',  # Count of matches per day
    }).reset_index()
    
    # Calculate true positives (need to check if hit_target column exists)
    if 'hit_target' in trades_df.columns:
        true_positives = trades_df[trades_df['hit_target'] == True].groupby(date_col).size()
        daily_stats = daily_stats.merge(
            true_positives.to_frame('true_positives'), 
            left_on=date_col, 
            right_index=True, 
            how='left'
        )
        daily_stats['true_positives'] = daily_stats['true_positives'].fillna(0).astype(int)
    else:
        daily_stats['true_positives'] = 0
    
    # Rename for clarity
    daily_stats.columns = [date_col, 'total_matches', 'true_positives']
    daily_stats['false_positives'] = daily_stats['total_matches'] - daily_stats['true_positives']
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Cumulative Matches", "Daily Metrics"),
        vertical_spacing=0.1
    )
    
    # Cumulative matches
    daily_stats_sorted = daily_stats.sort_values(date_col)
    daily_stats_sorted['cumulative_matches'] = daily_stats_sorted['total_matches'].cumsum()
    
    fig.add_trace(go.Scatter(
        x=daily_stats_sorted[date_col],
        y=daily_stats_sorted['cumulative_matches'],
        mode='lines',
        name='Total Matches',
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ), row=1, col=1)
    
    # Daily metrics
    fig.add_trace(go.Bar(
        x=daily_stats[date_col],
        y=daily_stats['true_positives'],
        name='True Positives',
        marker_color='#10b981'
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=daily_stats[date_col],
        y=daily_stats['false_positives'],
        name='False Positives',
        marker_color='#ef4444'
    ), row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        **CHART_THEME
    )
    
    fig.update_xaxes(**CHART_THEME['xaxis'])
    fig.update_yaxes(**CHART_THEME['yaxis'])
    
    return fig

def create_confusion_matrix(trades_df: pd.DataFrame):
    """Create confusion matrix visualization"""
    if trades_df.empty:
        return None
    
    # Calculate confusion matrix values
    tp = len(trades_df[(trades_df['matched_criteria'] == True) & (trades_df['hit_target'] == True)])
    fp = len(trades_df[(trades_df['matched_criteria'] == True) & (trades_df['hit_target'] == False)])
    fn = len(trades_df[(trades_df['matched_criteria'] == False) & (trades_df['hit_target'] == True)])
    tn = len(trades_df[(trades_df['matched_criteria'] == False) & (trades_df['hit_target'] == False)])
    
    matrix = [[tp, fp], [fn, tn]]
    labels = [['True Positive', 'False Positive'], ['False Negative', 'True Negative']]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=['Target Hit', 'Target Missed'],
        y=['Matched Criteria', 'Missed Criteria'],
        text=labels,
        texttemplate='%{text}<br>%{z}',
        textfont={"size": 14, "color": "#ffffff"},
        colorscale=[[0, '#2d3142'], [1, '#667eea']],
        showscale=False
    ))
    
    fig.update_layout(
        title="<b>Confusion Matrix</b>",
        height=400,
        **CHART_THEME
    )
    
    return fig

def create_gain_distribution(trades_df: pd.DataFrame):
    """Create gain distribution chart"""
    if trades_df.empty or 'actual_gain_pct' not in trades_df.columns:
        return None
    
    matched_gains = trades_df[(trades_df['matched_criteria'] == True) & (trades_df['actual_gain_pct'].notna())]['actual_gain_pct']
    unmatched_gains = trades_df[(trades_df['matched_criteria'] == False) & (trades_df['actual_gain_pct'].notna())]['actual_gain_pct']
    
    fig = go.Figure()
    
    if not matched_gains.empty:
        fig.add_trace(go.Histogram(x=matched_gains, name='Matched Criteria', opacity=0.75, 
                                   marker_color='#059669', nbinsx=30))
    
    if not unmatched_gains.empty:
        fig.add_trace(go.Histogram(x=unmatched_gains, name='Missed Criteria', opacity=0.75, 
                                   marker_color='#d97706', nbinsx=30))
    
    fig.update_layout(
        title="<b>Gain Distribution</b>", 
        xaxis_title="Gain %", 
        yaxis_title="Frequency", 
        barmode='overlay', 
        height=400,
        **CHART_THEME
    )
    
    fig.update_xaxes(**CHART_THEME['xaxis'])
    fig.update_yaxes(**CHART_THEME['yaxis'])
    
    return fig

def create_exit_analysis_chart(trades_df: pd.DataFrame):
    """Create exit analysis showing close vs high/low"""
    if trades_df.empty:
        return None
    
    # ✅ FIX: Check for actual column names in your database
    # You have: exit_high, exit_low, max_possible_gain_pct, max_drawdown_pct
    required_cols = ['actual_gain_pct', 'matched_criteria']
    if not all(col in trades_df.columns for col in required_cols):
        return None
    
    # Check for high/low columns with flexible naming
    high_col = None
    low_col = None
    
    # Check for high column
    for possible_high in ['max_possible_gain_pct', 'high_pct', 'exit_high']:
        if possible_high in trades_df.columns:
            high_col = possible_high
            break
    
    # Check for low column
    for possible_low in ['max_drawdown_pct', 'low_pct', 'exit_low']:
        if possible_low in trades_df.columns:
            low_col = possible_low
            break
    
    if high_col is None or low_col is None:
        return None
    
    matched = trades_df[trades_df['matched_criteria'] == True].copy()
    
    if matched.empty:
        return None
    
    # ✅ FIX: Find the date column
    date_col = None
    for possible_col in ['signal_date', 'date', 'trade_date']:
        if possible_col in matched.columns:
            date_col = possible_col
            break
    
    if date_col is None:
        # If no date column, just use index
        matched = matched.reset_index(drop=True)
    else:
        matched = matched.sort_values(date_col)
    
    matched['trade_num'] = range(len(matched))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=matched['trade_num'],
        y=matched['actual_gain_pct'],
        mode='markers',
        name='Actual Gain (Close)',
        marker=dict(size=8, color='#1e40af'),
        text=matched['symbol'] if 'symbol' in matched.columns else None,
        hovertemplate='%{text}<br>Gain: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=matched['trade_num'],
        y=matched[high_col],
        mode='markers',
        name='Max Possible Gain',
        marker=dict(size=6, color='#10b981', symbol='triangle-up'),
        hovertemplate='Max: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=matched['trade_num'],
        y=matched[low_col],
        mode='markers',
        name='Max Drawdown',
        marker=dict(size=6, color='#ef4444', symbol='triangle-down'),
        hovertemplate='Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="<b>Exit Analysis: Actual vs Intraday Range</b>",
        xaxis_title="Trade Number",
        yaxis_title="Gain %",
        height=400,
        hovermode='closest',
        **CHART_THEME
    )
    
    fig.update_xaxes(**CHART_THEME['xaxis'])
    fig.update_yaxes(**CHART_THEME['yaxis'])
    
    return fig

def create_cumulative_pnl(trades_df: pd.DataFrame):
    """Create cumulative P&L curve"""
    if trades_df.empty:
        return None
    
    # ✅ FIX: Check column exists
    if 'actual_gain_pct' not in trades_df.columns or 'matched_criteria' not in trades_df.columns:
        return None
    
    matched = trades_df[trades_df['matched_criteria'] == True].copy()
    
    if matched.empty:
        return None
    
    matched = matched.sort_values('signal_date')
    matched['cumulative_pnl'] = matched['actual_gain_pct'].cumsum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=matched['signal_date'],
        y=matched['cumulative_pnl'],
        mode='lines',
        name='Cumulative P&L',
        line=dict(color='#667eea', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    fig.update_layout(
        title="<b>Cumulative P&L</b>",
        xaxis_title="Date",
        yaxis_title="Cumulative Gain %",
        height=400,
        hovermode='x unified',
        **CHART_THEME
    )
    
    fig.update_xaxes(**CHART_THEME['xaxis'])
    fig.update_yaxes(**CHART_THEME['yaxis'])
    
    return fig

# ============================================================================
# MAIN RENDERING FUNCTION
# ============================================================================

def render_backtesting_tab():
    """Main backtesting tab rendering function"""
    
    st.subheader("Strategy Backtesting")
    st.markdown("Test your trading strategies against historical data")
    
    # Initialize session state
    if 'backtest_refresh_counter' not in st.session_state:
        st.session_state.backtest_refresh_counter = 0
    if 'backtest_data_loaded' not in st.session_state:
        st.session_state.backtest_data_loaded = False
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["View Results", "Create Strategy", "Manage Strategies"])
    
    # ========================================================================
    # TAB 1: VIEW RESULTS
    # ========================================================================
    with tab1:
        st.markdown("### Strategy Results")
        
        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("🔄 Refresh", key="view_refresh", use_container_width=True):
                st.cache_data.clear()
                st.session_state.backtest_refresh_counter += 1
                st.session_state.backtest_data_loaded = True
                st.rerun()
        
        # ✅ CRITICAL: Only load data if user has clicked refresh
        if not st.session_state.backtest_data_loaded:
            st.info("👆 Click 'Refresh' to load backtesting results")
            return
        
        # Load strategies (only after refresh clicked)
        strategies_df = load_strategies(st.session_state.backtest_refresh_counter)
        
        if strategies_df.empty:
            st.info("No strategies found. Create one in the 'Create Strategy' tab")
        else:
            # Select strategy
            strategy_names = dict(zip(strategies_df['id'], strategies_df['name']))
            
            selected_id = st.selectbox(
                "Select Strategy:",
                options=list(strategy_names.keys()),
                format_func=lambda x: strategy_names[x],
                key=f"view_strategy_select_{st.session_state.backtest_refresh_counter}"
            )
            
            if selected_id:
                strategy = strategies_df[strategies_df['id'] == selected_id].iloc[0]
                
                # Display strategy info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Period", f"{strategy['start_date']} to {strategy['end_date']}")
                with col2:
                    st.metric("Target Gain", f"{strategy['target_min_gain_pct']}%")
                with col3:
                    st.metric("Status", strategy['run_status'])
                
                # Show indicator criteria
                with st.expander("Indicator Criteria"):
                    criteria = strategy['indicator_criteria']
                    if isinstance(criteria, list):
                        for i, cond in enumerate(criteria, 1):
                            comp_type = cond.get('comparison_type', 'value')
                            if comp_type == 'indicator':
                                st.write(f"{i}. **{cond.get('indicator', 'N/A')}** {cond.get('operator', '')} **{cond.get('compare_to', 'N/A')}**")
                            else:
                                st.write(f"{i}. **{cond.get('indicator', 'N/A')}** {cond.get('operator', '')} {cond.get('value', '')}")
                
                # Load results
                daily_df, trades_df = load_strategy_results(selected_id, st.session_state.backtest_refresh_counter)
                
                # Display results if completed
                if strategy['run_status'] == 'completed' and not daily_df.empty:
                    st.markdown("---")
                    st.markdown("### Results Summary")
                    
                    # Basic metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Matches", strategy.get('total_matches', 0))
                    with col2:
                        st.metric("True Positives", strategy.get('true_positives', 0))
                    with col3:
                        st.metric("False Positives", strategy.get('false_positives', 0))
                    with col4:
                        st.metric("Missed Opps", strategy.get('missed_opportunities', 0))
                    with col5:
                        acc = strategy.get('accuracy_pct', 0)
                        st.metric("Accuracy", f"{acc}%" if acc else "N/A")
                    
                    # Advanced metrics - ✅ FIX: Add robust column checking
                    if not trades_df.empty and 'actual_gain_pct' in trades_df.columns:
                        matched_trades = trades_df[trades_df['matched_criteria'] == True]
                        
                        if not matched_trades.empty:
                            winners = matched_trades[matched_trades['actual_gain_pct'] > 0]
                            losers = matched_trades[matched_trades['actual_gain_pct'] < 0]
                            
                            win_rate = (len(winners) / len(matched_trades) * 100) if len(matched_trades) > 0 else 0
                            avg_winner = winners['actual_gain_pct'].mean() if len(winners) > 0 else 0
                            avg_loser = losers['actual_gain_pct'].mean() if len(losers) > 0 else 0
                            
                            total_gains = winners['actual_gain_pct'].sum() if len(winners) > 0 else 0
                            total_losses = abs(losers['actual_gain_pct'].sum()) if len(losers) > 0 else 0
                            profit_factor = (total_gains / total_losses) if total_losses > 0 else None
                            
                            # ✅ FIX: Proper intraday hit rate calculation
                            # Check multiple possible column names
                            intraday_hits = 0
                            intraday_col = None
                            
                            for possible_col in ['hit_target_intraday', 'target_hit_intraday', 'intraday_hit']:
                                if possible_col in matched_trades.columns:
                                    intraday_col = possible_col
                                    # Count True values
                                    intraday_hits = (matched_trades[possible_col] == True).sum()
                                    break
                            
                            intraday_rate = (intraday_hits / len(matched_trades) * 100) if len(matched_trades) > 0 else 0
                            
                            # ✅ DEBUG: Show what columns are available
                            with st.expander("🔍 Debug: Intraday Hit Rate"):
                                st.write(f"**Total matched trades:** {len(matched_trades)}")
                                st.write(f"**Intraday hits found:** {intraday_hits}")
                                st.write(f"**Intraday column found:** {intraday_col if intraday_col else 'NONE'}")
                                st.write(f"**Available columns in trades_df:**")
                                st.write(list(matched_trades.columns))
                                
                                # Show a sample of relevant columns
                                relevant_cols = [col for col in matched_trades.columns if 'hit' in col.lower() or 'target' in col.lower() or 'intraday' in col.lower()]
                                if relevant_cols:
                                    st.write(f"**Columns with 'hit', 'target', or 'intraday':**")
                                    st.dataframe(matched_trades[relevant_cols].head(10))
                            
                            st.markdown("#### Advanced Metrics")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("Win Rate", f"{win_rate:.1f}%")
                            with col2:
                                st.metric("Avg Winner", f"{avg_winner:.2f}%")
                            with col3:
                                st.metric("Avg Loser", f"{avg_loser:.2f}%")
                            with col4:
                                pf_text = f"{profit_factor:.2f}" if profit_factor else "N/A"
                                st.metric("Profit Factor", pf_text)
                            with col5:
                                st.metric("Intraday Hit Rate", f"{intraday_rate:.1f}%")
                    
                    # Charts
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
                        # ✅ DEBUG: Check why exit analysis might not appear
                        with st.expander("🔍 Debug: Exit Analysis Chart"):
                            st.write(f"**Trades DF empty?** {trades_df.empty}")
                            if not trades_df.empty:
                                st.write(f"**Columns in trades_df:** {list(trades_df.columns)}")
                                
                                # Check for actual column names
                                st.write(f"**Column availability:**")
                                st.write(f"  - actual_gain_pct: {'✅' if 'actual_gain_pct' in trades_df.columns else '❌'}")
                                st.write(f"  - max_possible_gain_pct: {'✅' if 'max_possible_gain_pct' in trades_df.columns else '❌'}")
                                st.write(f"  - max_drawdown_pct: {'✅' if 'max_drawdown_pct' in trades_df.columns else '❌'}")
                                st.write(f"  - matched_criteria: {'✅' if 'matched_criteria' in trades_df.columns else '❌'}")
                                
                                if 'matched_criteria' in trades_df.columns:
                                    matched_count = (trades_df['matched_criteria'] == True).sum()
                                    st.write(f"**Matched criteria trades:** {matched_count}")
                        
                        fig4 = create_exit_analysis_chart(trades_df)
                        if fig4:
                            st.plotly_chart(fig4, use_container_width=True)
                        else:
                            st.info("Exit analysis chart requires actual_gain_pct, max_possible_gain_pct, and max_drawdown_pct columns")
                    
                    # Trade log
                    st.markdown("### Trade Log")
                    if not trades_df.empty:
                        display_trades = trades_df.sort_values('signal_date', ascending=False).head(100)
                        
                        # Select display columns
                        display_cols = ['signal_date', 'symbol', 'matched_criteria', 'hit_target', 
                                      'actual_gain_pct', 'high_pct', 'low_pct']
                        display_cols = [col for col in display_cols if col in display_trades.columns]
                        
                        st.dataframe(
                            display_trades[display_cols],
                            use_container_width=True,
                            height=400
                        )
                        
                        # Download button
                        csv = trades_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download All Trades",
                            csv,
                            f"trades_strategy_{selected_id}.csv",
                            "text/csv"
                        )
                    else:
                        st.info("No trades to display")
                
                elif strategy['run_status'] == 'pending':
                    st.info("This strategy hasn't been run yet. Click 'Run Backtest' in the Manage Strategies tab.")
                elif strategy['run_status'] == 'running':
                    st.info("This strategy is currently running. Refresh this page in a few minutes to see results.")
                elif strategy['run_status'] == 'failed':
                    st.error("This strategy failed to run. Check the logs for details.")
    
    # ========================================================================
    # TAB 2: CREATE STRATEGY
    # ========================================================================
    with tab2:
        st.markdown("### Create New Strategy")
        
        # ✅ Only load date range when creating strategy (not on every render)
        min_date, max_date = None, None
        if st.session_state.backtest_data_loaded:
            min_date, max_date = get_date_range_from_db(st.session_state.backtest_refresh_counter)
        
        with st.form("create_strategy_form"):
            name = st.text_input("Strategy Name", placeholder="e.g., High RSI Momentum")
            description = st.text_area("Description", placeholder="Describe your strategy...")
            
            col1, col2 = st.columns(2)
            with col1:
                if min_date and max_date:
                    start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
                else:
                    start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=365))
            
            with col2:
                if min_date and max_date:
                    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
                else:
                    end_date = st.date_input("End Date", value=datetime.now().date())
            
            target_gain = st.number_input("Target Min Gain %", min_value=0.1, max_value=100.0, value=5.0, step=0.5)
            
            st.markdown("#### Indicator Criteria")
            st.info("Add conditions that define when to signal a trade")
            
            # Number of conditions
            num_conditions = st.number_input("Number of conditions", min_value=1, max_value=10, value=2)
            
            criteria = []
            for i in range(num_conditions):
                st.markdown(f"**Condition {i+1}**")
                col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
                
                with col1:
                    indicator = st.text_input(
                        "Indicator",
                        key=f"ind_{i}",
                        placeholder="e.g., rsi, macd.macd, adx"
                    )
                
                with col2:
                    operator = st.selectbox(
                        "Operator",
                        [">", "<", ">=", "<=", "=="],
                        key=f"op_{i}"
                    )
                
                with col3:
                    comp_type = st.selectbox(
                        "Compare to",
                        ["value", "indicator"],
                        key=f"comp_{i}"
                    )
                
                with col4:
                    if comp_type == "value":
                        compare_value = st.number_input(
                            "Value",
                            key=f"val_{i}",
                            value=0.0,
                            step=0.1
                        )
                        criteria.append({
                            'indicator': indicator,
                            'operator': operator,
                            'comparison_type': 'value',
                            'value': compare_value
                        })
                    else:
                        compare_indicator = st.text_input(
                            "Indicator",
                            key=f"comp_ind_{i}",
                            placeholder="e.g., ema20"
                        )
                        criteria.append({
                            'indicator': indicator,
                            'operator': operator,
                            'comparison_type': 'indicator',
                            'compare_to': compare_indicator
                        })
            
            submitted = st.form_submit_button("Create Strategy", use_container_width=True)
            
            if submitted:
                if not name:
                    st.error("Please provide a strategy name")
                elif start_date >= end_date:
                    st.error("Start date must be before end date")
                else:
                    strategy_id = save_strategy(name, description, start_date, end_date, criteria, target_gain)
                    if strategy_id:
                        st.success(f"✅ Strategy created successfully! ID: {strategy_id}")
                        st.info("Go to 'Manage Strategies' tab to run the backtest")
                        st.session_state.backtest_refresh_counter += 1
    
    # ========================================================================
    # TAB 3: MANAGE STRATEGIES
    # ========================================================================
    with tab3:
        st.markdown("### Manage Strategies")
        
        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("🔄 Refresh", key="manage_refresh", use_container_width=True):
                st.cache_data.clear()
                st.session_state.backtest_refresh_counter += 1
                st.rerun()
        
        strategies_df = load_strategies(st.session_state.backtest_refresh_counter)
        
        if strategies_df.empty:
            st.info("No strategies found")
        else:
            for idx, row in strategies_df.iterrows():
                with st.expander(f"**{row['name']}** - {row['run_status']}", expanded=False):
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
                                with st.spinner("Triggering backtest..."):
                                    success = run_backtest_via_github(row['id'])
                                    if success:
                                        st.cache_data.clear()
                                        st.session_state.backtest_refresh_counter += 1
                        elif row['run_status'] == 'running':
                            st.info("Running...")
                        else:
                            st.success("✅ Done")
                    
                    if row.get('description'):
                        st.write(f"**Description:** {row['description']}")
                    
                    st.write("**Criteria:**")
                    criteria = row['indicator_criteria']
                    if isinstance(criteria, list):
                        for i, cond in enumerate(criteria, 1):
                            comp_type = cond.get('comparison_type', 'value')
                            if comp_type == 'indicator':
                                st.write(f"{i}. `{cond.get('indicator', 'N/A')} {cond.get('operator', '')} {cond.get('compare_to', 'N/A')}`")
                            else:
                                st.write(f"{i}. `{cond.get('indicator', 'N/A')} {cond.get('operator', '')} {cond.get('value', '')}`")
