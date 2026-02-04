"""
Strategy Backtest Tab Module
UI for configuring and running strategy backtests
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import subprocess
import sys
import os

from backtest_supabase_client import BacktestSupabaseClient
from src.utils import load_config


# ============================================================================
# CONFIGURATION
# ============================================================================

@st.cache_resource
def get_backtest_client():
    """Initialize Supabase client for backtests"""
    config = load_config("config.yaml")
    return BacktestSupabaseClient(config)


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================

def render_criteria_builder():
    """
    Render UI for building strategy criteria
    Returns criteria dictionary
    """
    st.markdown("### Strategy Criteria")
    st.markdown("Define the indicator conditions that stocks must meet")
    
    criteria = {}
    
    # Common indicators to choose from
    available_indicators = [
        'volume', 'price', 'rsi', 'macd', 'stoch_k', 'stoch_d',
        'adx', 'atr', 'bb_width', 'ema5', 'ema10', 'ema20', 'ema50', 'ema200',
        'sma20', 'sma50', 'volume_ratio'
    ]
    
    # Number of criteria
    num_criteria = st.number_input(
        "Number of criteria",
        min_value=1,
        max_value=10,
        value=3,
        key="num_criteria"
    )
    
    # Build criteria
    cols = st.columns(2)
    
    for i in range(num_criteria):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"**Criterion {i+1}**")
                
                indicator = st.selectbox(
                    "Indicator",
                    available_indicators,
                    key=f"indicator_{i}"
                )
                
                condition_type = st.selectbox(
                    "Condition",
                    ["Minimum", "Maximum", "Range", "Equals"],
                    key=f"condition_{i}"
                )
                
                if condition_type == "Minimum":
                    min_val = st.number_input(
                        "Min value",
                        value=0.0,
                        key=f"min_{i}"
                    )
                    criteria[indicator] = {'min': min_val}
                
                elif condition_type == "Maximum":
                    max_val = st.number_input(
                        "Max value",
                        value=100.0,
                        key=f"max_{i}"
                    )
                    criteria[indicator] = {'max': max_val}
                
                elif condition_type == "Range":
                    min_val = st.number_input(
                        "Min value",
                        value=0.0,
                        key=f"range_min_{i}"
                    )
                    max_val = st.number_input(
                        "Max value",
                        value=100.0,
                        key=f"range_max_{i}"
                    )
                    criteria[indicator] = {'min': min_val, 'max': max_val}
                
                else:  # Equals
                    eq_val = st.number_input(
                        "Value",
                        value=0.0,
                        key=f"equals_{i}"
                    )
                    criteria[indicator] = {'equals': eq_val}
                
                st.markdown("---")
    
    return criteria


def render_backtest_form():
    """
    Render form for backtest configuration
    Returns configuration dict or None if not submitted
    """
    st.markdown("## Configure Backtest")
    
    with st.form("backtest_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            strategy_name = st.text_input(
                "Strategy Name",
                value="My Strategy",
                help="Give your strategy a memorable name"
            )
            
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=180),
                max_value=datetime.now()
            )
            
            target_gain = st.number_input(
                "Target Gain %",
                min_value=0.1,
                max_value=100.0,
                value=5.0,
                step=0.5,
                help="Minimum gain percentage to consider a success"
            )
        
        with col2:
            st.markdown("###")  # Spacing
            
            end_date = st.date_input(
                "End Date",
                value=datetime.now() - timedelta(days=1),
                max_value=datetime.now()
            )
            
            holding_days = st.number_input(
                "Holding Period (days)",
                min_value=1,
                max_value=30,
                value=1,
                help="Number of days to hold the position"
            )
        
        st.markdown("---")
        
        # Criteria builder
        criteria = render_criteria_builder()
        
        st.markdown("---")
        
        # Symbol universe
        use_custom_symbols = st.checkbox("Use custom symbol list (otherwise uses default universe)")
        
        custom_symbols = None
        if use_custom_symbols:
            symbols_text = st.text_area(
                "Symbols (comma-separated)",
                value="AAPL,MSFT,GOOGL,AMZN,TSLA",
                help="Enter stock symbols separated by commas"
            )
            custom_symbols = [s.strip().upper() for s in symbols_text.split(',') if s.strip()]
        
        # Submit button
        submitted = st.form_submit_button("Run Backtest", use_container_width=True, type="primary")
        
        if submitted:
            # Validate
            if start_date >= end_date:
                st.error("Start date must be before end date")
                return None
            
            if not criteria:
                st.error("Please define at least one criterion")
                return None
            
            return {
                'strategy_name': strategy_name,
                'start_date': start_date,
                'end_date': end_date,
                'target_gain': target_gain,
                'holding_days': holding_days,
                'criteria': criteria,
                'symbols': custom_symbols
            }
    
    return None


def run_backtest_subprocess(config: dict) -> dict:
    """
    Run backtest by calling backtest_main.py as subprocess
    This keeps the UI responsive
    
    Args:
        config: Backtest configuration
        
    Returns:
        Result dictionary with success status and backtest_id
    """
    # Create criteria JSON file
    criteria_file = "/tmp/backtest_criteria.json"
    with open(criteria_file, 'w') as f:
        json.dump(config['criteria'], f)
    
    # Build command
    cmd = [
        sys.executable,
        "backtest_main.py",
        "--start-date", config['start_date'].strftime("%Y-%m-%d"),
        "--end-date", config['end_date'].strftime("%Y-%m-%d"),
        "--target-gain", str(config['target_gain']),
        "--holding-days", str(config['holding_days']),
        "--criteria-file", criteria_file,
        "--strategy-name", config['strategy_name'],
        "--save-to-supabase"
    ]
    
    if config.get('symbols'):
        cmd.extend(["--symbols"] + config['symbols'])
    
    # Run subprocess
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Clean up temp file
        if os.path.exists(criteria_file):
            os.remove(criteria_file)
        
        if result.returncode == 0:
            # Parse backtest_id from output (last line should contain it)
            output_lines = result.stdout.strip().split('\n')
            backtest_id = None
            
            for line in output_lines:
                if "Saved to Supabase with ID:" in line:
                    backtest_id = line.split(":")[-1].strip()
                    break
            
            return {
                'success': True,
                'backtest_id': backtest_id,
                'output': result.stdout
            }
        else:
            return {
                'success': False,
                'error': result.stderr,
                'output': result.stdout
            }
    
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': "Backtest timed out after 10 minutes"
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def render_backtest_results(backtest_id: str):
    """
    Render results for a completed backtest
    
    Args:
        backtest_id: ID of backtest to display
    """
    client = get_backtest_client()
    
    # Load backtest summary
    backtests_df = client.read_backtests()
    backtest = backtests_df[backtests_df['id'] == backtest_id].iloc[0]
    
    # Header
    st.markdown(f"## Results: {backtest['strategy_name']}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Date Range", f"{backtest['start_date']} to {backtest['end_date']}")
    with col2:
        st.metric("Target Gain", f"{backtest['target_gain_pct']}%")
    with col3:
        st.metric("Holding Period", f"{backtest['holding_days']} days")
    
    st.markdown("---")
    
    # Key Metrics
    st.markdown("### Performance Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Signals", backtest['total_signals'])
    with col2:
        st.metric(
            "Successful Hits",
            backtest['successful_hits'],
            delta=f"{backtest['success_rate']:.1f}%"
        )
    with col3:
        st.metric(
            "False Positives",
            backtest['false_positives'],
            delta=f"{backtest['false_positive_rate']:.1f}%",
            delta_color="inverse"
        )
    with col4:
        st.metric("Missed Opportunities", backtest['missed_opportunities'])
    with col5:
        st.metric("Total Return", f"{backtest['total_return']:.2f}%")
    
    st.markdown("---")
    
    # Detailed results tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Trade Details",
        "Success Analysis",
        "Missed Opportunities",
        "Strategy Criteria"
    ])
    
    with tab1:
        # Load trades
        trades_df = client.read_backtest_trades(backtest_id)
        
        if not trades_df.empty:
            st.markdown(f"### All Trades ({len(trades_df)} total)")
            
            # Add derived columns
            trades_df['result'] = trades_df['hit_target'].map({True: '✓ Success', False: '✗ Failed'})
            
            # Display table
            display_df = trades_df[[
                'trade_date', 'symbol', 'entry_price', 'exit_price',
                'actual_gain_pct', 'result'
            ]].copy()
            
            display_df.columns = ['Date', 'Symbol', 'Entry ($)', 'Exit ($)', 'Gain (%)', 'Result']
            
            st.dataframe(
                display_df.style.format({
                    'Entry ($)': '${:.2f}',
                    'Exit ($)': '${:.2f}',
                    'Gain (%)': '{:+.2f}%'
                }).background_gradient(subset=['Gain (%)'], cmap='RdYlGn'),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No trades recorded")
    
    with tab2:
        if not trades_df.empty:
            st.markdown("### Success vs Failure Analysis")
            
            # Success rate over time
            trades_df['trade_date'] = pd.to_datetime(trades_df['trade_date'])
            trades_df = trades_df.sort_values('trade_date')
            
            # Calculate cumulative success rate
            trades_df['cumulative_success_rate'] = (
                trades_df['hit_target'].cumsum() / (trades_df.index + 1) * 100
            )
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=trades_df['trade_date'],
                y=trades_df['cumulative_success_rate'],
                mode='lines',
                name='Success Rate',
                line=dict(width=3, color='#2ecc71')
            ))
            
            fig.add_hline(
                y=backtest['success_rate'],
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Overall: {backtest['success_rate']:.1f}%"
            )
            
            fig.update_layout(
                title="Cumulative Success Rate Over Time",
                xaxis_title="Date",
                yaxis_title="Success Rate (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Gain distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Successful Trades**")
                success_trades = trades_df[trades_df['hit_target']]
                if not success_trades.empty:
                    st.metric("Average Gain", f"{success_trades['actual_gain_pct'].mean():.2f}%")
                    st.metric("Median Gain", f"{success_trades['actual_gain_pct'].median():.2f}%")
                    st.metric("Best Trade", f"{success_trades['actual_gain_pct'].max():.2f}%")
            
            with col2:
                st.markdown("**Failed Trades**")
                failed_trades = trades_df[~trades_df['hit_target']]
                if not failed_trades.empty:
                    st.metric("Average Loss", f"{failed_trades['actual_gain_pct'].mean():.2f}%")
                    st.metric("Median Loss", f"{failed_trades['actual_gain_pct'].median():.2f}%")
                    st.metric("Worst Trade", f"{failed_trades['actual_gain_pct'].min():.2f}%")
    
    with tab3:
        # Load missed opportunities
        missed_df = client.read_missed_opportunities(backtest_id)
        
        if not missed_df.empty:
            st.markdown(f"### Missed Opportunities ({len(missed_df)} total)")
            st.markdown("Stocks that hit the target gain but didn't match your criteria")
            
            display_df = missed_df[['trade_date', 'symbol', 'actual_gain_pct']].copy()
            display_df.columns = ['Date', 'Symbol', 'Gain (%)']
            
            st.dataframe(
                display_df.style.format({
                    'Gain (%)': '{:+.2f}%'
                }),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No missed opportunities found (or sampling didn't detect any)")
    
    with tab4:
        st.markdown("### Strategy Criteria Used")
        
        criteria = backtest['strategy_criteria']
        
        if isinstance(criteria, str):
            criteria = json.loads(criteria)
        
        for indicator, conditions in criteria.items():
            st.markdown(f"**{indicator.upper()}**")
            for condition, value in conditions.items():
                st.write(f"  • {condition}: {value}")


# ============================================================================
# MAIN TAB FUNCTION
# ============================================================================

def render_strategy_backtest_tab():
    """Main Strategy Backtest tab rendering function"""
    
    st.markdown("# Strategy Backtester")
    st.markdown(
        "Test indicator-based trading strategies against historical data. "
        "Define your criteria, set a target gain, and see how well your strategy would have performed."
    )
    
    st.markdown("---")
    
    # Initialize session state
    if 'backtest_running' not in st.session_state:
        st.session_state.backtest_running = False
    
    if 'current_backtest_id' not in st.session_state:
        st.session_state.current_backtest_id = None
    
    # Main content
    tabs = st.tabs(["New Backtest", "View Results"])
    
    with tabs[0]:
        # Backtest form
        config = render_backtest_form()
        
        if config:
            # Run backtest
            st.session_state.backtest_running = True
            
            with st.spinner("Running backtest... This may take a few minutes."):
                result = run_backtest_subprocess(config)
            
            st.session_state.backtest_running = False
            
            if result['success']:
                st.success("✓ Backtest completed successfully!")
                
                if result['backtest_id']:
                    st.session_state.current_backtest_id = result['backtest_id']
                    st.info(f"Backtest ID: {result['backtest_id']}")
                    
                    # Show quick link to results
                    if st.button("View Results"):
                        st.rerun()
                
                # Show output log
                with st.expander("View detailed output"):
                    st.code(result['output'])
            else:
                st.error("✗ Backtest failed")
                st.code(result.get('error', 'Unknown error'))
                
                if result.get('output'):
                    with st.expander("View output"):
                        st.code(result['output'])
    
    with tabs[1]:
        st.markdown("## Previous Backtests")
        
        # Load all backtests
        client = get_backtest_client()
        backtests_df = client.read_backtests(limit=50)
        
        if backtests_df.empty:
            st.info("No backtests found. Create one in the 'New Backtest' tab.")
        else:
            # Summary table
            st.markdown("### Recent Backtests")
            
            display_df = backtests_df[[
                'created_at', 'strategy_name', 'start_date', 'end_date',
                'total_signals', 'success_rate', 'total_return'
            ]].copy()
            
            display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            display_df.columns = [
                'Created', 'Strategy', 'Start', 'End',
                'Signals', 'Success Rate (%)', 'Total Return (%)'
            ]
            
            st.dataframe(
                display_df.style.format({
                    'Success Rate (%)': '{:.1f}',
                    'Total Return (%)': '{:+.2f}'
                }),
                use_container_width=True
            )
            
            st.markdown("---")
            
            # Select backtest to view
            st.markdown("### View Detailed Results")
            
            backtest_options = {
                f"{row['strategy_name']} ({row['created_at']})": row['id']
                for _, row in backtests_df.iterrows()
            }
            
            # Pre-select current backtest if available
            default_index = 0
            if st.session_state.current_backtest_id:
                for i, (label, bid) in enumerate(backtest_options.items()):
                    if bid == st.session_state.current_backtest_id:
                        default_index = i
                        break
            
            selected_label = st.selectbox(
                "Select backtest to view:",
                list(backtest_options.keys()),
                index=default_index
            )
            
            if selected_label:
                backtest_id = backtest_options[selected_label]
                
                st.markdown("---")
                
                render_backtest_results(backtest_id)
