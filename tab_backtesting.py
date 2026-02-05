"""
Strategy Backtesting Tab Module - FIXED VERSION
Handles the backtesting UI and visualizations
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
# GITHUB WORKFLOW TRIGGER - FIXED
# ============================================================================

def run_backtest_via_github(strategy_id: int, strategy_config: dict):
    """
    Trigger backtest via GitHub Actions workflow - WORKING VERSION
    """
    import os
    import sys
    
    # Check if GitHub token is available (try multiple env var names)
    github_token = None
    try:
        github_token = (
            os.environ.get("G_TOKEN") or st.secrets.get("secrets", {}).get("G_TOKEN")
        )
    except Exception as e:
        st.warning(f"Error accessing secrets: {e}")
    
    github_owner = None
    try:
        github_owner = (
            os.environ.get("GITHUB_REPO_OWNER") or
            st.secrets.get("GITHUB_REPO_OWNER", "")
        )
    except Exception as e:
        st.warning(f"Error accessing owner: {e}")
    
    # DEBUG INFO
    st.info(f"🔍 Debug: Token found: {bool(github_token)}, Owner found: {bool(github_owner)}")
    
    if not github_token:
        st.warning("⚠️ GitHub token not configured. Running locally...")
        run_backtest_local(strategy_id, strategy_config)
        return
    
    if not github_owner:
        st.error("❌ GITHUB_REPO_OWNER not configured.")
        st.info("Add GITHUB_REPO_OWNER to Streamlit secrets (your GitHub username)")
        return
    
    try:
        # Set environment variables for the trigger
        os.environ["GITHUB_TOKEN"] = github_token
        os.environ["GITHUB_REPO_OWNER"] = github_owner
        
        # Import the GitHub trigger module
        import requests
        
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
                "verbose": "true"
            }
        }
        
        st.info(f"🔄 Triggering workflow at: {url}")
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 204:
            st.success(f"✅ Backtest workflow triggered via GitHub Actions!")
            st.info(f"🔄 Strategy ID: {strategy_id} is now running in GitHub Actions")
            
            # Show recent runs
            with st.expander("📊 Recent Workflow Runs", expanded=True):
                try:
                    runs_url = f"{api_base}/actions/workflows/{workflow_file}/runs"
                    runs_response = requests.get(runs_url, params={"per_page": 5}, headers=headers, timeout=10)
                    
                    if runs_response.status_code == 200:
                        runs = runs_response.json().get("workflow_runs", [])
                        if runs:
                            for run in runs[:5]:
                                status_emoji = {
                                    "completed": "✅",
                                    "in_progress": "🔄",
                                    "queued": "⏳",
                                    "failed": "❌"
                                }.get(run.get("status"), "❓")
                                
                                conclusion_emoji = {
                                    "success": "✅",
                                    "failure": "❌",
                                    "cancelled": "🚫"
                                }.get(run.get("conclusion", ""), "")
                                
                                st.markdown(
                                    f"{status_emoji} {conclusion_emoji} **Run #{run.get('run_number')}** - "
                                    f"{run.get('status')} - "
                                    f"{run.get('created_at', '')[:19]}"
                                )
                                st.markdown(f"[View Run]({run.get('html_url')})")
                                st.divider()
                        else:
                            st.info("No recent runs found")
                except Exception as e:
                    st.warning(f"Could not fetch workflow runs: {e}")
            
            st.info("⏱️ Refresh this page in 2-10 minutes to see results in 'View Results' tab.")
        else:
            st.error(f"❌ Failed to trigger workflow: {response.status_code}")
            st.error(f"Response: {response.text}")
            st.warning("Falling back to local execution...")
            run_backtest_local(strategy_id, strategy_config)
            
    except Exception as e:
        st.error(f"❌ Error triggering workflow: {e}")
        st.warning("Falling back to local execution...")
        run_backtest_local(strategy_id, strategy_config)


def run_backtest_local(strategy_id: int, strategy_config: dict):
    """
    Fallback: Run backtest locally using threading
    """
    import threading
    
    st.warning("🔄 Running backtest locally (GitHub Actions unavailable)")
    
    # Create and start background thread
    thread = threading.Thread(
        target=run_backtest_in_thread,
        args=(strategy_id, strategy_config),
        daemon=True
    )
    thread.start()
    
    st.success(f"✅ Backtest started locally! Strategy ID: {strategy_id}")
    st.info("🔄 The backtest is running in the background. Refresh this page in a few moments to see results.")
    st.info("⏱️ Depending on the date range, this may take 1-10 minutes.")
    
    # Show progress instructions
    with st.expander("📊 How to monitor progress"):
        st.write("""
        The backtest is now running. To see results:
        
        1. **Wait** 1-5 minutes (depends on date range)
        2. **Click 'Refresh'** button or reload page
        3. **Check 'View Results'** tab
        4. Look for strategy status changing from 'running' to 'completed'
        
        **Status indicators:**
        - `pending` - Strategy created, not started
        - `running` - Currently backtesting (wait for completion)
        - `completed` - Done! Results available
        - `failed` - Error occurred (check logs)
        """)

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


def run_backtest_in_thread(strategy_id: int, strategy_config: dict):
    """
    Run backtest in a background thread
    Updates Supabase as it progresses
    """
    import sys
    import threading
    from pathlib import Path
    
    # Import backtesting modules
    # Add tradingview-analysis path to system path
    # This assumes both repos are in the same parent directory
    parent_dir = Path(__file__).parent.parent
    analysis_repo = parent_dir / "tradingview-analysis"
    
    if analysis_repo.exists():
        sys.path.insert(0, str(analysis_repo))
    
    try:
        from backtesting.strategy_backtester import StrategyBacktester
        from backtesting.backtest_supabase_client import BacktestSupabaseClient
        
        # Load minimal config
        config = {'backtesting': {}}
        
        # Initialize components
        backtester = StrategyBacktester(config)
        supabase_client = BacktestSupabaseClient(config)
        
        # Update status to running
        supabase_client.update_strategy_status(strategy_id, 'running')
        
        # Define progress callback
        def progress_callback(current, total, date):
            # Could update database with progress here
            pass
        
        # Run backtest
        results = backtester.run_backtest(strategy_config, progress_callback)
        
        # Write results
        supabase_client.write_daily_results(strategy_id, results['daily_results'])
        supabase_client.write_trades(strategy_id, results['trades'])
        supabase_client.update_strategy_summary(strategy_id, results['overall_stats'])
        supabase_client.update_strategy_status(strategy_id, 'completed')
        
    except Exception as e:
        # Update status to failed
        try:
            supabase_client = BacktestSupabaseClient({'backtesting': {}})
            supabase_client.update_strategy_status(strategy_id, 'failed')
        except:
            pass
        raise e


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
    
    # Top chart: Matches and results
    fig.add_trace(
        go.Bar(
            x=daily_df['test_date'],
            y=daily_df['criteria_matches'],
            name='Total Matches',
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=daily_df['test_date'],
            y=daily_df['true_positives'],
            name='True Positives (Wins)',
            marker_color='green'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=daily_df['test_date'],
            y=daily_df['false_positives'],
            name='False Positives (Losses)',
            marker_color='red'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_df['test_date'],
            y=daily_df['missed_opportunities'],
            name='Missed Opportunities',
            mode='lines+markers',
            line=dict(color='orange', dash='dot')
        ),
        row=1, col=1
    )
    
    # Bottom chart: Gains
    fig.add_trace(
        go.Scatter(
            x=daily_df['test_date'],
            y=daily_df['avg_match_gain_pct'],
            name='Avg Match Gain %',
            mode='lines+markers',
            line=dict(color='green')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_df['test_date'],
            y=daily_df['max_gain_pct'],
            name='Max Gain %',
            mode='lines',
            line=dict(color='lightgreen', dash='dot'),
            opacity=0.5
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Gain %", row=2, col=1)
    
    fig.update_layout(
        height=700,
        title_text="<b>Backtest Performance Over Time</b>",
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def create_confusion_matrix(trades_df: pd.DataFrame):
    """Create confusion matrix visualization"""
    if trades_df.empty:
        return None
    
    # Count trade types
    type_counts = trades_df['trade_type'].value_counts()
    
    # Build matrix
    true_pos = type_counts.get('true_positive', 0)
    false_pos = type_counts.get('false_positive', 0)
    false_neg = type_counts.get('false_negative', 0)
    true_neg = type_counts.get('true_negative', 0)
    
    matrix = [
        [true_pos, false_pos],
        [false_neg, true_neg]
    ]
    
    labels = [
        [f"True Positive<br>{true_pos}", f"False Positive<br>{false_pos}"],
        [f"False Negative<br>{false_neg}", f"True Negative<br>{true_neg}"]
    ]
    
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
    
    fig.update_layout(
        title="<b>Confusion Matrix</b>",
        xaxis_title="Actual Outcome",
        yaxis_title="Predicted (Criteria)",
        height=400
    )
    
    return fig


def create_gain_distribution(trades_df: pd.DataFrame):
    """Create gain distribution chart"""
    if trades_df.empty or 'actual_gain_pct' not in trades_df.columns:
        return None
    
    # Filter out None values
    gains = trades_df[trades_df['actual_gain_pct'].notna()]['actual_gain_pct']
    
    if gains.empty:
        return None
    
    # Separate by match status
    matched_gains = trades_df[
        (trades_df['matched_criteria'] == True) & 
        (trades_df['actual_gain_pct'].notna())
    ]['actual_gain_pct']
    
    unmatched_gains = trades_df[
        (trades_df['matched_criteria'] == False) & 
        (trades_df['actual_gain_pct'].notna())
    ]['actual_gain_pct']
    
    fig = go.Figure()
    
    if not matched_gains.empty:
        fig.add_trace(go.Histogram(
            x=matched_gains,
            name='Matched Criteria',
            opacity=0.7,
            marker_color='green',
            nbinsx=30
        ))
    
    if not unmatched_gains.empty:
        fig.add_trace(go.Histogram(
            x=unmatched_gains,
            name='Missed Criteria',
            opacity=0.7,
            marker_color='orange',
            nbinsx=30
        ))
    
    fig.update_layout(
        title="<b>Gain Distribution</b>",
        xaxis_title="Gain %",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )
    
    return fig


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
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 View Results", "🆕 Create Strategy", "📋 Manage Strategies"])
    
    
    # ========================================================================
    # TAB 1: VIEW RESULTS
    # ========================================================================
    with tab1:
        st.markdown("### Strategy Results")
        
        # Load strategies
        strategies_df = load_strategies(st.session_state.backtest_refresh_counter)
        
        if strategies_df.empty:
            st.info("No strategies found. Create one in the 'Create Strategy' tab!")
        else:
            # Strategy selector
            strategy_names = dict(zip(strategies_df['id'], strategies_df['name']))
            selected_id = st.selectbox(
                "Select Strategy:",
                options=list(strategy_names.keys()),
                format_func=lambda x: strategy_names[x],
                key="view_strategy_select"
            )
            
            if selected_id:
                strategy = strategies_df[strategies_df['id'] == selected_id].iloc[0]
                
                # Show strategy info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Period", f"{strategy['start_date']} to {strategy['end_date']}")
                with col2:
                    st.metric("Target Gain", f"{strategy['target_min_gain_pct']}%")
                with col3:
                    st.metric("Status", strategy['run_status'])
                
                # Show criteria
                with st.expander("📋 Indicator Criteria"):
                    criteria = strategy['indicator_criteria']
                    if isinstance(criteria, list):
                        for i, cond in enumerate(criteria, 1):
                            st.write(f"{i}. **{cond.get('indicator', 'N/A')}** {cond.get('operator', '')} {cond.get('value', '')}")
                
                # Load results
                daily_df, trades_df = load_strategy_results(selected_id, st.session_state.backtest_refresh_counter)
                
                if strategy['run_status'] == 'completed' and not daily_df.empty:
                    st.markdown("---")
                    st.markdown("### Results Summary")
                    
                    # Metrics
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
                    
                    # Charts
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
                    
                    # Trade details
                    st.markdown("### Trade Details")
                    
                    trade_type_filter = st.selectbox(
                        "Filter by type:",
                        ['All', 'True Positives', 'False Positives', 'Missed Opportunities'],
                        key="trade_filter"
                    )
                    
                    if not trades_df.empty:
                        filtered_trades = trades_df.copy()
                        
                        if trade_type_filter == 'True Positives':
                            filtered_trades = filtered_trades[filtered_trades['trade_type'] == 'true_positive']
                        elif trade_type_filter == 'False Positives':
                            filtered_trades = filtered_trades[filtered_trades['trade_type'] == 'false_positive']
                        elif trade_type_filter == 'Missed Opportunities':
                            filtered_trades = filtered_trades[filtered_trades['trade_type'] == 'false_negative']
                        
                        # Display columns
                        display_cols = ['symbol', 'signal_date', 'entry_price', 'exit_price', 'actual_gain_pct', 'trade_type', 'matched_criteria', 'hit_target']
                        available_cols = [col for col in display_cols if col in filtered_trades.columns]
                        
                        st.dataframe(
                            filtered_trades[available_cols].sort_values('signal_date', ascending=False),
                            use_container_width=True,
                            height=400
                        )
                    
                elif strategy['run_status'] == 'pending':
                    st.info("⏳ Strategy created but not yet run")
                elif strategy['run_status'] == 'running':
                    st.info("🔄 Backtest is currently running...")
                elif strategy['run_status'] == 'failed':
                    st.error("❌ Backtest failed. Check logs for details.")
    
    # ========================================================================
    # TAB 2: CREATE STRATEGY
    # ========================================================================
    with tab2:
        st.markdown("### Create New Strategy")
        
        with st.form("create_strategy_form"):
            # Basic info
            strategy_name = st.text_input("Strategy Name*", placeholder="e.g., High RSI Momentum")
            strategy_desc = st.text_area("Description", placeholder="Optional description of your strategy")
            
            # Date range
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date*",
                    value=datetime.now().date() - timedelta(days=365),
                    max_value=datetime.now().date()
                )
            with col2:
                end_date = st.date_input(
                    "End Date*",
                    value=datetime.now().date(),
                    max_value=datetime.now().date()
                )
            
            # Target
            st.markdown("#### Target Performance")
            st.markdown("Define what you're looking for: how much gain and over what timeframe")
            
            col1, col2 = st.columns(2)
            with col1:
                target_gain = st.number_input(
                    "Target Gain %*",
                    min_value=0.1,
                    value=5.0,
                    step=0.1,
                    help="Minimum % gain to consider a successful trade"
                )
            with col2:
                target_days = st.number_input(
                    "Days to Hold*",
                    min_value=1,
                    max_value=30,
                    value=1,
                    help="How long to hold the position. 1 = same day (intraday), 2 = sell next day, etc."
                )
            
            # Explanation of holding period
            st.info(f"""
            **📅 Holding Period Explained:**
            
            You're testing if a stock that meets your criteria on Day 0 will gain {target_gain}% or more 
            by Day {target_days}.
            
            - **Day 0**: Stock meets your indicator criteria (e.g., RSI < 30)
            - **Day {target_days}**: Check if stock gained {target_gain}% or more
            
            Examples:
            - **1 day** = Same-day trading (buy at signal, sell at close)
            - **2 days** = Hold overnight (buy on Day 0, sell on Day 1)
            - **5 days** = Hold for a week (buy on Day 0, sell on Day 4)
            
            The backtest will check if stocks matching your criteria hit this target within this timeframe.
            """)
            
            # Filters
            st.markdown("#### Stock Filters")
            col1, col2, col3 = st.columns(3)
            with col1:
                min_price = st.number_input("Min Price ($)", min_value=0.01, value=0.50, step=0.10)
            with col2:
                max_price = st.number_input("Max Price ($)", min_value=0.0, value=0.0, step=1.0)
                max_price = max_price if max_price > 0 else None
            with col3:
                min_volume = st.number_input("Min Volume", min_value=1000, value=100000, step=10000)
            
            # Exchanges
            exchanges = st.multiselect(
                "Exchanges",
                ['NASDAQ', 'NYSE', 'AMEX'],
                default=['NASDAQ', 'NYSE']
            )
            
            # Indicator criteria
            st.markdown("#### Indicator Criteria")
            st.markdown("Add conditions that stocks must meet")
            
            # Available indicators
            available_indicators = [
                'rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'adx',
                'volume', 'volume_ratio', 'close', 'open',
                'ema_10', 'ema_20', 'ema_50', 'ema_200',
                'sma_10', 'sma_20', 'sma_50', 'sma_200',
                'bb_upper', 'bb_lower', 'bb_middle', 'atr'
            ]
            
            # Dynamic criteria builder
            num_criteria = st.number_input("Number of Criteria", min_value=1, max_value=10, value=2)
            
            criteria = []
            for i in range(int(num_criteria)):
                st.markdown(f"**Condition {i+1}**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    indicator = st.selectbox(
                        "Indicator",
                        available_indicators,
                        key=f"indicator_{i}"
                    )
                
                with col2:
                    operator = st.selectbox(
                        "Operator",
                        ['>', '<', '>=', '<=', '==', '!='],
                        key=f"operator_{i}"
                    )
                
                with col3:
                    value = st.number_input(
                        "Value",
                        value=0.0,
                        step=0.1,
                        key=f"value_{i}"
                    )
                
                criteria.append({
                    'indicator': indicator,
                    'operator': operator,
                    'value': value
                })
            
            # Submit
            submitted = st.form_submit_button("🚀 Create & Run Backtest", type="primary")
            
            if submitted:
                # Validate
                if not strategy_name:
                    st.error("Please enter a strategy name")
                elif not criteria:
                    st.error("Please add at least one criterion")
                elif start_date >= end_date:
                    st.error("End date must be after start date")
                else:
                    # Create strategy config
                    strategy_config = {
                        'name': strategy_name,
                        'description': strategy_desc,
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat(),
                        'target_min_gain_pct': target_gain,
                        'target_days': target_days,
                        'indicator_criteria': criteria,
                        'min_price': min_price,
                        'max_price': max_price,
                        'min_volume': min_volume,
                        'exchanges': exchanges
                    }
                    
                    # Create strategy in database
                    try:
                        client = get_supabase_client()
                        data = {
                            'name': strategy_config['name'],
                            'description': strategy_config.get('description', ''),
                            'start_date': strategy_config['start_date'],
                            'end_date': strategy_config['end_date'],
                            'target_min_gain_pct': strategy_config['target_min_gain_pct'],
                            'target_days': strategy_config.get('target_days', 1),
                            'indicator_criteria': json.dumps(strategy_config['indicator_criteria']),
                            'min_price': strategy_config.get('min_price', 0.50),
                            'max_price': strategy_config.get('max_price'),
                            'min_volume': strategy_config.get('min_volume', 100000),
                            'exchanges': strategy_config.get('exchanges', ['NASDAQ', 'NYSE', 'AMEX']),
                            'run_status': 'pending'
                        }
                        
                        response = client.table("backtest_strategies").insert(data).execute()
                        strategy_id = response.data[0]['id']
                                
                        st.success(f"✓ Strategy created! ID: {strategy_id}")
                                
                                # TRIGGER VIA GITHUB ACTIONS
                        run_backtest_via_github(strategy_id, strategy_config)
                                
                                # Refresh
                        st.session_state.backtest_refresh_counter += 1
                                
                    except Exception as e:
                        st.error(f"Error creating strategy: {e}")
    
    # ========================================================================
    # TAB 3: MANAGE STRATEGIES
    # ========================================================================
    with tab3:
        st.markdown("### Manage Strategies")
        
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.session_state.backtest_refresh_counter += 1
                st.rerun()
        
        strategies_df = load_strategies(st.session_state.backtest_refresh_counter)
        
        if strategies_df.empty:
            st.info("No strategies found")
        else:
            # Display table
            display_cols = ['id', 'name', 'start_date', 'end_date', 'target_min_gain_pct', 'run_status', 'accuracy_pct', 'created_at']
            available_cols = [col for col in display_cols if col in strategies_df.columns]
            
            st.dataframe(
                strategies_df[available_cols].sort_values('created_at', ascending=False),
                use_container_width=True,
                height=400
            )
            
            # Delete strategy
            st.markdown("### Delete Strategy")
            strategy_to_delete = st.selectbox(
                "Select strategy to delete:",
                options=strategies_df['id'].tolist(),
                format_func=lambda x: strategies_df[strategies_df['id'] == x]['name'].iloc[0],
                key="delete_strategy_select"
            )
            
            if st.button("🗑️ Delete Strategy", type="secondary"):
                try:
                    client = get_supabase_client()
                    client.table("backtest_strategies").delete().eq("id", strategy_to_delete).execute()
                    st.success("Strategy deleted!")
                    st.session_state.backtest_refresh_counter += 1
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting strategy: {e}")
            
            st.markdown("---")
            
            # Run pending strategies
            st.markdown("### Run Pending Strategies")
            pending_strategies = strategies_df[strategies_df['run_status'] == 'pending']
            
            if pending_strategies.empty:
                st.info("No pending strategies. All strategies have been run or are running.")
            else:
                strategy_to_run = st.selectbox(
                    "Select pending strategy to run:",
                    options=pending_strategies['id'].tolist(),
                    format_func=lambda x: pending_strategies[pending_strategies['id'] == x]['name'].iloc[0],
                    key="run_strategy_select"
                )
                
                if st.button("▶️ Run Backtest", type="primary"):
                    try:
                        # Get strategy details
                        strategy = strategies_df[strategies_df['id'] == strategy_to_run].iloc[0]
                        
                        # Build config
                        strategy_config = {
                            'name': strategy['name'],
                            'description': strategy.get('description', ''),
                            'start_date': strategy['start_date'],
                            'end_date': strategy['end_date'],
                            'target_min_gain_pct': strategy['target_min_gain_pct'],
                            'target_days': strategy.get('target_days', 1),
                            'indicator_criteria': strategy['indicator_criteria'],
                            'min_price': strategy.get('min_price', 0.50),
                            'max_price': strategy.get('max_price'),
                            'min_volume': strategy.get('min_volume', 100000),
                            'exchanges': strategy.get('exchanges', ['NASDAQ', 'NYSE', 'AMEX'])
                        }
                        
                        # Run backtest
                        run_backtest_via_github(strategy_to_run, strategy_config)
                        
                        st.session_state.backtest_refresh_counter += 1
                        
                    except Exception as e:
                        st.error(f"Error running backtest: {e}")
