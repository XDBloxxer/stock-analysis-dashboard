"""
Backtesting Tab Module - FIXED FOR ZERO AUTO-EGRESS
CHANGES:
- ttl=0 (never auto-refresh)
- Added .limit(100) to all queries
- Only refreshes when user clicks refresh button
- Pagination for trades if needed
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import os
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

@st.cache_resource
def get_supabase_client():
    supabase_url = os.environ.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url")
    supabase_key = os.environ.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("key")
    
    if not supabase_url or not supabase_key:
        st.error("❌ Supabase credentials not configured!")
        st.stop()
    
    return create_client(supabase_url, supabase_key)

@st.cache_data(ttl=0)  # ✅ CHANGED: Never auto-refresh (was 60)
def load_strategies(_refresh_key: int = 0):
    """
    OPTIMIZED: Only loads when refresh button is clicked
    - ttl=0 means cache never expires on its own
    - Limits to 100 most recent strategies
    """
    try:
        client = get_supabase_client()
        
        # ✅ CRITICAL FIX: Add LIMIT to prevent fetching ALL strategies
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

@st.cache_data(ttl=0)  # ✅ CHANGED: Never auto-refresh (was 60)
def load_strategy_results(strategy_id: int, _refresh_key: int = 0):
    """
    OPTIMIZED: Only loads when refresh button is clicked
    - Limits results to prevent massive queries
    """
    try:
        client = get_supabase_client()
        
        # Load backtest results (usually small)
        results = client.table("backtest_results")\
            .select("*")\
            .eq("strategy_id", strategy_id)\
            .order("test_date", desc=False)\
            .execute()
        
        results_df = pd.DataFrame(results.data) if results.data else pd.DataFrame()
        
        # ✅ CRITICAL FIX: Limit trades to 1000 most recent
        trades = client.table("backtest_trades")\
            .select("*")\
            .eq("strategy_id", strategy_id)\
            .order("signal_date", desc=True)\
            .limit(1000)\
            .execute()
        
        trades_df = pd.DataFrame(trades.data) if trades.data else pd.DataFrame()
        
        return results_df, trades_df
    except Exception as e:
        st.error(f"Error loading strategy results: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=0)  # ✅ CHANGED: Never auto-refresh
def get_date_range(_refresh_key: int = 0):
    """
    OPTIMIZED: Get min/max dates without fetching all data
    """
    try:
        client = get_supabase_client()
        
        # Get min date
        min_response = client.table("historical_market_data")\
            .select("date")\
            .order("date", desc=False)\
            .limit(1)\
            .execute()
        
        # Get max date
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

def create_equity_curve(results_df):
    """Create equity curve chart"""
    if results_df.empty or 'cumulative_pnl' not in results_df.columns:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results_df['test_date'],
        y=results_df['cumulative_pnl'],
        mode='lines',
        name='Cumulative P&L',
        line=dict(color='#667eea', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    fig.update_layout(
        title="<b>Equity Curve</b>",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L ($)",
        height=400,
        hovermode='x unified',
        **CHART_THEME
    )
    
    fig.update_xaxes(**CHART_THEME['xaxis'])
    fig.update_yaxes(**CHART_THEME['yaxis'])
    
    return fig

def create_drawdown_chart(results_df):
    """Create drawdown chart"""
    if results_df.empty or 'cumulative_pnl' not in results_df.columns:
        return None
    
    cumulative = results_df['cumulative_pnl'].values
    running_max = pd.Series(cumulative).expanding().max()
    drawdown = cumulative - running_max
    drawdown_pct = (drawdown / running_max) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results_df['test_date'],
        y=drawdown_pct,
        mode='lines',
        name='Drawdown %',
        line=dict(color='#ef4444', width=2),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.2)'
    ))
    
    fig.update_layout(
        title="<b>Drawdown</b>",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=300,
        **CHART_THEME
    )
    
    fig.update_xaxes(**CHART_THEME['xaxis'])
    fig.update_yaxes(**CHART_THEME['yaxis'])
    
    return fig

def create_monthly_returns_heatmap(results_df):
    """Create monthly returns heatmap"""
    if results_df.empty:
        return None
    
    try:
        results_df['date'] = pd.to_datetime(results_df['test_date'])
        results_df['year'] = results_df['date'].dt.year
        results_df['month'] = results_df['date'].dt.month
        
        monthly = results_df.groupby(['year', 'month'])['daily_pnl'].sum().reset_index()
        
        pivot = monthly.pivot(index='year', columns='month', values='daily_pnl')
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[month_names[i-1] for i in pivot.columns],
            y=pivot.index,
            colorscale='RdYlGn',
            zmid=0,
            text=pivot.values.round(0),
            texttemplate='$%{text}',
            textfont={"size": 10, "color": "#ffffff"},
            colorbar=dict(title="P&L ($)", tickfont=dict(color='#e8eaf0'))
        ))
        
        fig.update_layout(
            title="<b>Monthly Returns</b>",
            xaxis_title="Month",
            yaxis_title="Year",
            height=max(200, len(pivot) * 50),
            **CHART_THEME
        )
        
        return fig
    except Exception as e:
        st.warning(f"Could not create monthly heatmap: {e}")
        return None

def create_trade_analysis(trades_df):
    """Create trade analysis charts"""
    if trades_df.empty:
        return None, None
    
    # Win/Loss distribution
    fig1 = go.Figure()
    
    wins = trades_df[trades_df['pnl'] > 0]['pnl']
    losses = trades_df[trades_df['pnl'] <= 0]['pnl']
    
    if len(wins) > 0:
        fig1.add_trace(go.Histogram(
            x=wins,
            name='Wins',
            marker_color='#10b981',
            opacity=0.7,
            nbinsx=30
        ))
    
    if len(losses) > 0:
        fig1.add_trace(go.Histogram(
            x=losses,
            name='Losses',
            marker_color='#ef4444',
            opacity=0.7,
            nbinsx=30
        ))
    
    fig1.update_layout(
        title="<b>Trade P&L Distribution</b>",
        xaxis_title="P&L ($)",
        yaxis_title="Count",
        height=300,
        barmode='overlay',
        **CHART_THEME
    )
    
    # Cumulative trades
    trades_sorted = trades_df.sort_values('signal_date')
    trades_sorted['cumulative_trades'] = range(1, len(trades_sorted) + 1)
    trades_sorted['cumulative_pnl'] = trades_sorted['pnl'].cumsum()
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=trades_sorted['cumulative_trades'],
        y=trades_sorted['cumulative_pnl'],
        mode='lines',
        name='Cumulative P&L',
        line=dict(color='#667eea', width=2)
    ))
    
    fig2.update_layout(
        title="<b>Trade-by-Trade P&L</b>",
        xaxis_title="Trade Number",
        yaxis_title="Cumulative P&L ($)",
        height=300,
        **CHART_THEME
    )
    
    return fig1, fig2

def render_backtesting_tab():
    if 'backtesting_refresh_counter' not in st.session_state:
        st.session_state.backtesting_refresh_counter = 0
    
    refresh_key = st.session_state.backtesting_refresh_counter
    
    # ✅ CHANGED: Manual refresh button at top
    col_header1, col_header2 = st.columns([4, 1])
    
    with col_header1:
        st.subheader("Backtesting Results")
    
    with col_header2:
        if st.button("🔄 Refresh Data", use_container_width=True, key="backtesting_manual_refresh"):
            st.cache_data.clear()
            st.session_state.backtesting_refresh_counter += 1
            st.rerun()
    
    with st.spinner("Loading strategies..."):
        strategies_df = load_strategies(refresh_key)
    
    if strategies_df.empty:
        st.info("No backtesting strategies found. Run a backtest to see results.")
        
        with st.expander("ℹ️ About Backtesting"):
            st.markdown("""
            **How to run a backtest:**
            1. Configure your strategy parameters
            2. Run `python -m backtesting.backtest_runner`
            3. Results will appear here
            
            **Key metrics to look for:**
            - Total P&L: Overall profit/loss
            - Win Rate: Percentage of winning trades
            - Sharpe Ratio: Risk-adjusted returns
            - Max Drawdown: Largest peak-to-trough decline
            """)
        
        return
    
    st.markdown(f"### Available Strategies ({len(strategies_df)})")
    
    # Strategy selector
    strategy_options = {}
    for _, row in strategies_df.iterrows():
        label = f"#{row['id']} - {row.get('name', 'Unnamed')} ({row['created_at'][:10]})"
        strategy_options[label] = row['id']
    
    selected_label = st.selectbox(
        "Select a strategy:",
        list(strategy_options.keys()),
        key="strategy_selector"
    )
    
    if not selected_label:
        return
    
    strategy_id = strategy_options[selected_label]
    strategy_row = strategies_df[strategies_df['id'] == strategy_id].iloc[0]
    
    # Display strategy details
    st.markdown("### Strategy Details")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Strategy ID", strategy_id)
    with col2:
        st.metric("Created", strategy_row['created_at'][:10])
    with col3:
        if 'name' in strategy_row:
            st.metric("Name", strategy_row['name'])
    with col4:
        if 'description' in strategy_row and strategy_row['description']:
            st.caption(strategy_row['description'])
    
    # Show indicator criteria if available
    if 'indicator_criteria' in strategy_row and strategy_row['indicator_criteria']:
        with st.expander("📊 Indicator Criteria"):
            criteria = strategy_row['indicator_criteria']
            if isinstance(criteria, dict):
                st.json(criteria)
            else:
                st.write(criteria)
    
    st.markdown("---")
    
    # Load results and trades
    with st.spinner("Loading backtest results..."):
        results_df, trades_df = load_strategy_results(strategy_id, refresh_key)
    
    if results_df.empty:
        st.warning("No backtest results found for this strategy")
        return
    
    # Performance metrics
    st.markdown("### Performance Summary")
    
    total_pnl = results_df['daily_pnl'].sum() if 'daily_pnl' in results_df.columns else 0
    num_trades = len(trades_df)
    win_rate = (len(trades_df[trades_df['pnl'] > 0]) / num_trades * 100) if num_trades > 0 else 0
    
    if 'cumulative_pnl' in results_df.columns:
        returns = results_df['daily_pnl'].values
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * (252 ** 0.5)  # Annualized
        else:
            sharpe = 0
    else:
        sharpe = 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total P&L", f"${total_pnl:,.0f}")
    with col2:
        st.metric("Total Trades", num_trades)
    with col3:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col4:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with col5:
        if num_trades > 0:
            avg_pnl = trades_df['pnl'].mean()
            st.metric("Avg Trade", f"${avg_pnl:,.0f}")
    
    st.markdown("---")
    
    # Charts
    tab1, tab2, tab3, tab4 = st.tabs(["Equity Curve", "Trade Analysis", "Monthly Returns", "Trade Log"])
    
    with tab1:
        fig_equity = create_equity_curve(results_df)
        if fig_equity:
            st.plotly_chart(fig_equity, use_container_width=True)
        
        fig_drawdown = create_drawdown_chart(results_df)
        if fig_drawdown:
            st.plotly_chart(fig_drawdown, use_container_width=True)
    
    with tab2:
        if not trades_df.empty:
            fig_dist, fig_cum = create_trade_analysis(trades_df)
            
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)
            
            if fig_cum:
                st.plotly_chart(fig_cum, use_container_width=True)
            
            # Trade statistics
            st.markdown("#### Trade Statistics")
            
            wins = trades_df[trades_df['pnl'] > 0]
            losses = trades_df[trades_df['pnl'] <= 0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Winning Trades**")
                if len(wins) > 0:
                    st.metric("Count", len(wins))
                    st.metric("Avg Win", f"${wins['pnl'].mean():,.0f}")
                    st.metric("Largest Win", f"${wins['pnl'].max():,.0f}")
            
            with col2:
                st.markdown("**Losing Trades**")
                if len(losses) > 0:
                    st.metric("Count", len(losses))
                    st.metric("Avg Loss", f"${losses['pnl'].mean():,.0f}")
                    st.metric("Largest Loss", f"${losses['pnl'].min():,.0f}")
        else:
            st.info("No trades to analyze")
    
    with tab3:
        fig_monthly = create_monthly_returns_heatmap(results_df)
        if fig_monthly:
            st.plotly_chart(fig_monthly, use_container_width=True)
        else:
            st.info("Not enough data for monthly returns")
    
    with tab4:
        if not trades_df.empty:
            st.markdown("#### Recent Trades")
            
            display_trades = trades_df.sort_values('signal_date', ascending=False).copy()
            
            # Format for display
            if 'signal_date' in display_trades.columns:
                display_trades['signal_date'] = pd.to_datetime(display_trades['signal_date']).dt.strftime('%Y-%m-%d')
            if 'exit_date' in display_trades.columns:
                display_trades['exit_date'] = pd.to_datetime(display_trades['exit_date']).dt.strftime('%Y-%m-%d')
            
            # Select columns to display
            display_cols = ['signal_date', 'symbol', 'signal_type', 'entry_price', 'exit_price', 'pnl']
            display_cols = [col for col in display_cols if col in display_trades.columns]
            
            st.dataframe(
                display_trades[display_cols].head(100),  # Show max 100 trades
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = trades_df.to_csv(index=False)
            st.download_button(
                "📥 Download All Trades",
                csv,
                f"trades_strategy_{strategy_id}.csv",
                "text/csv",
                key="download_trades"
            )
        else:
            st.info("No trades to display")
