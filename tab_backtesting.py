"""
Strategy Backtesting Tab Module - FIXED VERSION
No duplicate strategy creation - creates once, then runs
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
import sys
from pathlib import Path

# Import Supabase client
from supabase import create_client, Client


# ============================================================================
# SUPABASE CONNECTION
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


@st.cache_data(ttl=60)
def get_date_range_from_db():
    """Get available date range from historical_market_data table"""
    try:
        client = get_supabase_client()
        
        # Get min date
        response_min = client.table("historical_market_data") \
            .select("date") \
            .order("date", desc=False) \
            .limit(1) \
            .execute()
        
        # Get max date
        response_max = client.table("historical_market_data") \
            .select("date") \
            .order("date", desc=True) \
            .limit(1) \
            .execute()
        
        if response_min.data and response_max.data:
            min_date = datetime.fromisoformat(response_min.data[0]['date']).date()
            max_date = datetime.fromisoformat(response_max.data[0]['date']).date()
            return min_date, max_date
        
        # Fallback
        return datetime(2023, 1, 1).date(), datetime.now().date()
        
    except Exception as e:
        st.warning(f"Could not fetch date range: {e}")
        return datetime(2023, 1, 1).date(), datetime.now().date()


@st.cache_data(ttl=60)
def load_strategies(_refresh_key: int = 0):
    """Load all strategies"""
    try:
        client = get_supabase_client()
        response = client.table("backtest_strategies").select("*").order("created_at", desc=True).execute()
        
        if not response.data:
            return pd.DataFrame()
        
        df = pd.DataFrame(response.data)
        
        # Parse JSON fields
        if 'indicator_criteria' in df.columns:
            df['indicator_criteria'] = df['indicator_criteria'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        
        return df
    except Exception as e:
        st.error(f"Error loading strategies: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_strategy_results(strategy_id: int, _refresh_key: int = 0):
    """Load results for a strategy"""
    try:
        client = get_supabase_client()
        
        # Get daily results
        daily_response = client.table("backtest_results") \
            .select("*") \
            .eq("strategy_id", strategy_id) \
            .order("test_date") \
            .execute()
        
        # Get trades
        trades_response = client.table("backtest_trades") \
            .select("*") \
            .eq("strategy_id", strategy_id) \
            .execute()
        
        daily_df = pd.DataFrame(daily_response.data) if daily_response.data else pd.DataFrame()
        trades_df = pd.DataFrame(trades_response.data) if trades_response.data else pd.DataFrame()
        
        # Parse JSON in trades
        if not trades_df.empty and 'indicator_values' in trades_df.columns:
            trades_df['indicator_values'] = trades_df['indicator_values'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        
        return daily_df, trades_df
        
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return pd.DataFrame(), pd.DataFrame()


def run_backtest_via_github(strategy_id: int):
    """Trigger backtest via GitHub Actions workflow"""
    import requests
    
    github_token = None
    try:
        github_token = (
            os.environ.get("G_TOKEN") or st.secrets.get("secrets", {}).get("G_TOKEN")
        )
    except:
        pass
    
    github_owner = None
    try:
        github_owner = (
            os.environ.get("GITHUB_REPO_OWNER") or
            st.secrets.get("secrets", {}).get("GITHUB_REPO_OWNER")
        )
    except:
        pass
    
    if not github_token or not github_owner:
        st.warning("⚠️ GitHub not configured. Strategy created but not running.")
        st.info("Set G_TOKEN and GITHUB_REPO_OWNER in secrets to run via GitHub Actions")
        return
    
    try:
        repo_name = os.environ.get("GITHUB_REPO_NAME", "tradingview-analysis")
        api_base = f"https://api.github.com/repos/{github_owner}/{repo_name}"
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        workflow_file = "backtest.yml"
        url = f"{api_base}/actions/workflows/{workflow_file}/dispatches"
        
        payload = {
            "ref": "main",
            "inputs": {
                "strategy_id": str(strategy_id),
                "verbose": "false"
            }
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 204:
            st.success(f"✅ Backtest workflow triggered! Strategy ID: {strategy_id}")
        else:
            st.error(f"❌ Failed to trigger workflow: {response.status_code}")
            
    except Exception as e:
        st.error(f"❌ Error: {e}")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_performance_chart(daily_df: pd.DataFrame):
    """Create daily performance chart"""
    if daily_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Daily Matches & Results", "Gain Distribution"),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    fig.add_trace(
        go.Bar(x=daily_df['test_date'], y=daily_df['criteria_matches'], name='Total Matches', marker_color='lightblue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=daily_df['test_date'], y=daily_df['true_positives'], name='True Positives', marker_color='green'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=daily_df['test_date'], y=daily_df['false_positives'], name='False Positives', marker_color='red'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=daily_df['test_date'], y=daily_df['missed_opportunities'], name='Missed', mode='lines+markers', line=dict(color='orange', dash='dot')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=daily_df['test_date'], y=daily_df['avg_match_gain_pct'], name='Avg Match Gain %', mode='lines+markers', line=dict(color='green')),
        row=2, col=1
    )
    
    fig.update_layout(height=700, title_text="<b>Backtest Performance Over Time</b>", showlegend=True, hovermode='x unified')
    
    return fig


def create_confusion_matrix(trades_df: pd.DataFrame):
    """Create confusion matrix visualization"""
    if trades_df.empty:
        return None
    
    type_counts = trades_df['trade_type'].value_counts()
    
    true_pos = type_counts.get('true_positive', 0)
    false_pos = type_counts.get('false_positive', 0)
    false_neg = type_counts.get('false_negative', 0)
    true_neg = type_counts.get('true_negative', 0)
    
    matrix = [[true_pos, false_pos], [false_neg, true_neg]]
    labels = [[f"True Positive<br>{true_pos}", f"False Positive<br>{false_pos}"], [f"False Negative<br>{false_neg}", f"True Negative<br>{true_neg}"]]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        text=labels,
        texttemplate="%{text}",
        textfont={"size": 14},
        x=['Hit Target', 'Missed Target'],
        y=['Matched Criteria', 'Missed Criteria'],
        colorscale='RdYlGn',
        showscale=False
    ))
    
    fig.update_layout(title="<b>Confusion Matrix</b>", height=400)
    
    return fig


def create_gain_distribution(trades_df: pd.DataFrame):
    """Create gain distribution chart"""
    if trades_df.empty or 'actual_gain_pct' not in trades_df.columns:
        return None
    
    matched_gains = trades_df[(trades_df['matched_criteria'] == True) & (trades_df['actual_gain_pct'].notna())]['actual_gain_pct']
    unmatched_gains = trades_df[(trades_df['matched_criteria'] == False) & (trades_df['actual_gain_pct'].notna())]['actual_gain_pct']
    
    fig = go.Figure()
    
    if not matched_gains.empty:
        fig.add_trace(go.Histogram(x=matched_gains, name='Matched Criteria', opacity=0.7, marker_color='green', nbinsx=30))
    
    if not unmatched_gains.empty:
        fig.add_trace(go.Histogram(x=unmatched_gains, name='Missed Criteria', opacity=0.7, marker_color='orange', nbinsx=30))
    
    fig.update_layout(title="<b>Gain Distribution</b>", xaxis_title="Gain %", yaxis_title="Frequency", barmode='overlay', height=400)
    
    return fig


# ============================================================================
# AVAILABLE INDICATORS - EXPANDED LIST
# ============================================================================

AVAILABLE_INDICATORS = [
    # Momentum
    'rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'mom', 'w.r', 'uo', 'ao', 'cci20',
    
    # Trend
    'adx', 'adx+di', 'adx-di',
    'ema5', 'ema10', 'ema20', 'ema50', 'ema100', 'ema200',
    'sma5', 'sma10', 'sma20', 'sma50', 'sma100', 'sma200',
    
    # Volatility
    'atr', 'bb.upper', 'bb.lower', 'bb.middle', 'bb_width', 'bbpower', 'volatility_20d',
    
    # Volume
    'volume', 'volume_sma5', 'volume_sma20', 'volume_ratio',
    
    # Price
    'close', 'open', 'high', 'low',
    
    # Derived
    'price_change_1d', 'price_change_3d', 'price_change_5d', 'price_change_10d', 'price_change_20d',
    'gap_%', 'vwap',
    'high_52w', 'low_52w', 'price_vs_high_52w', 'price_vs_low_52w',
    
    # Trend Booleans
    'ema20_above_ema50', 'ema50_above_ema200', 'price_above_ema20', 'ema10_above_ema20',
    'gap_up', 'gap_down'
]


# ============================================================================
# MAIN TAB FUNCTION
# ============================================================================

def render_backtesting_tab():
    """Main backtesting tab rendering function"""
    
    st.subheader("Strategy Backtesting")
    st.markdown("Test your trading strategies against historical data")
    
    # Initialize refresh counter
    if 'backtest_refresh_counter' not in st.session_state:
        st.session_state.backtest_refresh_counter = 0
    
    # Get available date range from database
    min_date, max_date = get_date_range_from_db()
    
    st.info(f"📅 Available data range: **{min_date}** to **{max_date}**")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 View Results", "🆕 Create Strategy", "📋 Manage Strategies"])
    
    # ========================================================================
    # TAB 1: VIEW RESULTS
    # ========================================================================
    with tab1:
        st.markdown("### Strategy Results")
        
        # Refresh button at top
        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("🔄 Refresh", key="view_refresh", use_container_width=True):
                st.cache_data.clear()
                st.session_state.backtest_refresh_counter += 1
                st.rerun()
        
        strategies_df = load_strategies(st.session_state.backtest_refresh_counter)
        
        if strategies_df.empty:
            st.info("No strategies found. Create one in the 'Create Strategy' tab!")
        else:
            strategy_names = dict(zip(strategies_df['id'], strategies_df['name']))
            
            # Use a unique key that doesn't cause re-selection issues
            selected_id = st.selectbox(
                "Select Strategy:",
                options=list(strategy_names.keys()),
                format_func=lambda x: strategy_names[x],
                key=f"view_strategy_select_{st.session_state.backtest_refresh_counter}"
            )
            
            if selected_id:
                strategy = strategies_df[strategies_df['id'] == selected_id].iloc[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Period", f"{strategy['start_date']} to {strategy['end_date']}")
                with col2:
                    st.metric("Target Gain", f"{strategy['target_min_gain_pct']}%")
                with col3:
                    st.metric("Status", strategy['run_status'])
                
                with st.expander("📋 Indicator Criteria"):
                    criteria = strategy['indicator_criteria']
                    if isinstance(criteria, list):
                        for i, cond in enumerate(criteria, 1):
                            comp_type = cond.get('comparison_type', 'value')
                            if comp_type == 'indicator':
                                st.write(f"{i}. **{cond.get('indicator', 'N/A')}** {cond.get('operator', '')} **{cond.get('compare_to', 'N/A')}**")
                            else:
                                st.write(f"{i}. **{cond.get('indicator', 'N/A')}** {cond.get('operator', '')} {cond.get('value', '')}")
                
                daily_df, trades_df = load_strategy_results(selected_id, st.session_state.backtest_refresh_counter)
                
                if strategy['run_status'] == 'completed' and not daily_df.empty:
                    st.markdown("---")
                    st.markdown("### Results Summary")
                    
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
                    
                    st.markdown("### Performance Charts")
                    
                    fig1 = create_performance_chart(daily_df)
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
                    
                    st.markdown("### Trade Details")
                    
                    trade_type_filter = st.selectbox(
                        "Filter by type:",
                        ['All', 'True Positives', 'False Positives', 'Missed Opportunities'],
                        key=f"trade_filter_{st.session_state.backtest_refresh_counter}"
                    )
                    
                    if not trades_df.empty:
                        filtered_trades = trades_df.copy()
                        
                        if trade_type_filter == 'True Positives':
                            filtered_trades = filtered_trades[filtered_trades['trade_type'] == 'true_positive']
                        elif trade_type_filter == 'False Positives':
                            filtered_trades = filtered_trades[filtered_trades['trade_type'] == 'false_positive']
                        elif trade_type_filter == 'Missed Opportunities':
                            filtered_trades = filtered_trades[filtered_trades['trade_type'] == 'false_negative']
                        
                        display_cols = ['symbol', 'signal_date', 'entry_price', 'exit_price', 'actual_gain_pct', 'trade_type', 'matched_criteria', 'hit_target']
                        available_cols = [col for col in display_cols if col in filtered_trades.columns]
                        
                        st.dataframe(
                            filtered_trades[available_cols].sort_values('signal_date', ascending=False),
                            use_container_width=True,
                            height=400
                        )
                    
                elif strategy['run_status'] == 'pending':
                    st.info("⏳ Strategy created but not yet run")
                    if st.button("▶️ Run Now", key=f"run_pending_{selected_id}"):
                        run_backtest_via_github(selected_id)
                        st.cache_data.clear()
                        st.session_state.backtest_refresh_counter += 1
                        st.rerun()
                        
                elif strategy['run_status'] == 'running':
                    st.info("🔄 Backtest is currently running...")
                elif strategy['run_status'] == 'failed':
                    st.error("❌ Backtest failed. Check logs for details.")
    
    # ========================================================================
    # TAB 2: CREATE STRATEGY - DYNAMIC CRITERIA WITHOUT FORM
    # ========================================================================
    with tab2:
        st.markdown("### Create New Strategy")
        
        # Initialize criteria in session state
        if 'criteria_list' not in st.session_state:
            st.session_state.criteria_list = [
                {'indicator': 'rsi', 'operator': '>', 'comparison_type': 'value', 'value': 50.0, 'compare_to': None}
            ]
        
        # Basic info (no form needed)
        strategy_name = st.text_input("Strategy Name*", placeholder="e.g., High RSI Momentum", key="strategy_name_input")
        strategy_desc = st.text_area("Description", placeholder="Optional description", key="strategy_desc_input")
        
        # Date range with validation
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date*",
                value=max(min_date, datetime.now().date() - timedelta(days=365)),
                min_value=min_date,
                max_value=max_date,
                key="start_date_input"
            )
        with col2:
            end_date = st.date_input(
                "End Date*",
                value=min(max_date, datetime.now().date()),
                min_value=min_date,
                max_value=max_date,
                key="end_date_input"
            )
        
        # Target
        st.markdown("#### Target Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            target_gain = st.number_input("Target Gain %*", min_value=0.1, value=5.0, step=0.1, key="target_gain_input")
        with col2:
            target_days = st.number_input("Days to Hold*", min_value=1, max_value=30, value=1, key="target_days_input")
        
        st.info(f"Testing if stocks meeting criteria on Day 0 gain {target_gain}% or more by Day {target_days}")
        
        # Filters
        st.markdown("#### Stock Filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            min_price = st.number_input("Min Price ($)", min_value=0.01, value=0.25, step=0.10, key="min_price_input")
        with col2:
            max_price = st.number_input("Max Price ($)", min_value=0.0, value=0.0, step=1.0, key="max_price_input")
            max_price = max_price if max_price > 0 else None
        with col3:
            min_volume = st.number_input("Min Volume", min_value=1000, value=100000, step=10000, key="min_volume_input")
        
        # Indicator criteria with dynamic addition
        st.markdown("#### Indicator Criteria")
        st.markdown("Add conditions that stocks must meet")
        
        # Display all criteria
        criteria_to_remove = []
        for i, criterion in enumerate(st.session_state.criteria_list):
            st.markdown(f"**Condition {i+1}**")
            col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 0.5])
            
            with col1:
                indicator = st.selectbox(
                    "Indicator",
                    sorted(AVAILABLE_INDICATORS),
                    index=sorted(AVAILABLE_INDICATORS).index(criterion['indicator']) if criterion['indicator'] in AVAILABLE_INDICATORS else 0,
                    key=f"indicator_{i}"
                )
                st.session_state.criteria_list[i]['indicator'] = indicator
            
            with col2:
                operator = st.selectbox(
                    "Operator",
                    ['>', '<', '>=', '<=', '==', '!='],
                    index=['>', '<', '>=', '<=', '==', '!='].index(criterion['operator']) if criterion['operator'] in ['>', '<', '>=', '<=', '==', '!='] else 0,
                    key=f"operator_{i}"
                )
                st.session_state.criteria_list[i]['operator'] = operator
            
            with col3:
                comparison_type = st.selectbox(
                    "Compare to",
                    ['Value', 'Indicator'],
                    index=0 if criterion['comparison_type'] == 'value' else 1,
                    key=f"comptype_{i}"
                )
                st.session_state.criteria_list[i]['comparison_type'] = comparison_type.lower()
            
            with col4:
                if comparison_type == 'Value':
                    value = st.number_input(
                        "Value",
                        value=float(criterion.get('value', 0.0)),
                        step=0.1,
                        key=f"value_{i}"
                    )
                    st.session_state.criteria_list[i]['value'] = value
                    st.session_state.criteria_list[i]['compare_to'] = None
                else:
                    compare_indicator = st.selectbox(
                        "Indicator",
                        sorted(AVAILABLE_INDICATORS),
                        index=sorted(AVAILABLE_INDICATORS).index(criterion.get('compare_to', 'close')) if criterion.get('compare_to') in AVAILABLE_INDICATORS else 0,
                        key=f"compare_ind_{i}"
                    )
                    st.session_state.criteria_list[i]['compare_to'] = compare_indicator
                    st.session_state.criteria_list[i]['value'] = None
            
            with col5:
                if len(st.session_state.criteria_list) > 1:  # Only show delete if more than 1 criterion
                    if st.button("🗑️", key=f"remove_{i}", help="Remove this criterion"):
                        criteria_to_remove.append(i)
        
        # Remove criteria (outside the loop to avoid index issues)
        for idx in reversed(criteria_to_remove):
            st.session_state.criteria_list.pop(idx)
            st.rerun()
        
        # Add criterion button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("➕ Add Criterion", key="add_criterion_btn"):
                st.session_state.criteria_list.append({
                    'indicator': 'rsi',
                    'operator': '>',
                    'comparison_type': 'value',
                    'value': 0.0,
                    'compare_to': None
                })
                st.rerun()
        
        st.markdown("---")
        
        # Submit button (NOT in a form, so Enter won't trigger it)
        col1, col2, col3 = st.columns([1, 2, 2])
        with col1:
            submitted = st.button("🚀 Create & Run", type="primary", key="create_backtest_btn", use_container_width=True)
        
        # Handle submission
        if submitted:
            # Validate
            if not strategy_name:
                st.error("Please enter a strategy name")
            elif not st.session_state.criteria_list:
                st.error("Please add at least one criterion")
            elif start_date >= end_date:
                st.error("End date must be after start date")
            elif start_date < min_date or end_date > max_date:
                st.error(f"Dates must be between {min_date} and {max_date}")
            else:
                # Build criteria from session state
                criteria = []
                for crit in st.session_state.criteria_list:
                    if crit['comparison_type'] == 'value':
                        criteria.append({
                            'indicator': crit['indicator'],
                            'operator': crit['operator'],
                            'comparison_type': 'value',
                            'value': crit['value']
                        })
                    else:
                        criteria.append({
                            'indicator': crit['indicator'],
                            'operator': crit['operator'],
                            'comparison_type': 'indicator',
                            'compare_to': crit['compare_to']
                        })
                
                # Create strategy in database (ONLY ONCE)
                try:
                    client = get_supabase_client()
                    data = {
                        'name': strategy_name,
                        'description': strategy_desc,
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat(),
                        'target_min_gain_pct': target_gain,
                        'target_days': target_days,
                        'indicator_criteria': json.dumps(criteria),
                        'min_price': min_price,
                        'max_price': max_price,
                        'min_volume': min_volume,
                        'exchanges': ['NASDAQ', 'NYSE', 'AMEX'],
                        'run_status': 'pending'
                    }
                    
                    response = client.table("backtest_strategies").insert(data).execute()
                    strategy_id = response.data[0]['id']
                    
                    st.success(f"✅ Strategy created! ID: {strategy_id}")
                    
                    # Trigger workflow to RUN (not create) the strategy
                    run_backtest_via_github(strategy_id)
                    
                    # Reset criteria list
                    st.session_state.criteria_list = [
                        {'indicator': 'rsi', 'operator': '>', 'comparison_type': 'value', 'value': 50.0, 'compare_to': None}
                    ]
                    
                    # Refresh cache
                    st.cache_data.clear()
                    st.session_state.backtest_refresh_counter += 1
                    
                    st.info("Switch to 'View Results' tab to see progress")
                    
                except Exception as e:
                    st.error(f"Error creating strategy: {e}")
    
    # ========================================================================
    # TAB 3: MANAGE STRATEGIES
    # ========================================================================
    with tab3:
        st.markdown("### Manage Strategies")
        
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔄 Refresh", use_container_width=True, key="manage_refresh"):
                st.cache_data.clear()
                st.session_state.backtest_refresh_counter += 1
                st.rerun()
        
        strategies_df = load_strategies(st.session_state.backtest_refresh_counter)
        
        if strategies_df.empty:
            st.info("No strategies found")
        else:
            display_cols = ['id', 'name', 'start_date', 'end_date', 'target_min_gain_pct', 'run_status', 'accuracy_pct', 'created_at']
            available_cols = [col for col in display_cols if col in strategies_df.columns]
            
            st.dataframe(
                strategies_df[available_cols].sort_values('created_at', ascending=False),
                use_container_width=True,
                height=400
            )
            
            st.markdown("### Delete Strategy")
            strategy_to_delete = st.selectbox(
                "Select strategy to delete:",
                options=strategies_df['id'].tolist(),
                format_func=lambda x: strategies_df[strategies_df['id'] == x]['name'].iloc[0],
                key=f"delete_strategy_select_{st.session_state.backtest_refresh_counter}"
            )
            
            col1, col2, col3 = st.columns([1, 2, 2])
            with col1:
                if st.button("🗑️ Delete", type="secondary", key="delete_btn", use_container_width=True):
                    try:
                        client = get_supabase_client()
                        client.table("backtest_strategies").delete().eq("id", strategy_to_delete).execute()
                        st.success("Strategy deleted!")
                        st.cache_data.clear()
                        st.session_state.backtest_refresh_counter += 1
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting strategy: {e}")
