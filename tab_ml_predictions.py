"""
ML Predictions Tab - AUTONOMOUS SYSTEM
PERSISTENT CACHE - survives browser refresh, minimizes egress
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
from supabase import create_client, Client
import requests


# ===== HELPER FUNCTIONS =====

@st.cache_resource
def get_supabase_client() -> Client:
    """Get Supabase client - cached"""
    url = st.secrets.get("supabase", {}).get("url")
    key = st.secrets.get("supabase", {}).get("key")
    
    if not url or not key:
        raise ValueError("Missing Supabase credentials")
    
    return create_client(url, key)


def trigger_workflow(workflow_name: str, inputs: dict = None) -> bool:
    """Trigger a workflow in tradingview-analysis repo"""
    github_token = st.session_state.get('github_token')
    github_repo = st.session_state.get('github_repo')
    
    if not github_token:
        st.warning("⚠️ GitHub token not configured. Cannot trigger workflows.")
        st.info("Set G_TOKEN environment variable to enable workflow triggers.")
        return False
    
    url = f"https://api.github.com/repos/{github_repo}/actions/workflows/{workflow_name}/dispatches"
    
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    payload = {
        "ref": "main",
        "inputs": inputs or {}
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Failed to trigger workflow: {e}")
        return False


@st.cache_resource
def _load_ml_data(_tab_id: str, table_name: str, filter_key: tuple = None, order_by: tuple = None, limit: int = 100, _refresh_key: int = 0):
    """
    PERSISTENT CACHE - only refreshes when _refresh_key changes
    Survives browser close/refresh
    Per-tab isolation via _tab_id
    """
    filters = dict(filter_key) if filter_key else None
    order_col, order_dir = order_by if order_by else (None, None)
    
    try:
        client = get_supabase_client()
        query = client.table(table_name).select("*")
        
        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)
        
        if order_col:
            query = query.order(order_col, desc=(order_dir == 'desc'))
        
        if limit:
            query = query.limit(limit)
        
        response = query.execute()
        
        if not response.data:
            return pd.DataFrame()
        
        return pd.DataFrame(response.data)
    
    except Exception as e:
        st.warning(f"Could not load from {table_name}: {str(e)}")
        return pd.DataFrame()


def load_ml_data(_tab_id: str, table_name: str, filters: dict = None, order_by: tuple = None, limit: int = 100, _refresh_key: int = 0):
    """Public wrapper that converts filters to hashable tuple"""
    filter_key = tuple(sorted(filters.items())) if filters else None
    return _load_ml_data(_tab_id, table_name, filter_key, order_by, limit, _refresh_key)


# ===== MAIN TAB FUNCTION =====

def render_ml_predictions_tab():
    """Main ML Predictions tab with sub-tabs"""
    
    TAB_ID = "ml_predictions"
    
    if f'{TAB_ID}_refresh_counter' not in st.session_state:
        st.session_state[f'{TAB_ID}_refresh_counter'] = 0
    
    refresh_key = st.session_state[f'{TAB_ID}_refresh_counter']
    
    st.subheader("🤖 ML Explosion Predictions (Autonomous)")
    
    # Control buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
    
    with col1:
        if st.button("🔄 Refresh", key=f"{TAB_ID}_refresh"):
            st.session_state[f'{TAB_ID}_refresh_counter'] += 1
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Cache", key=f"{TAB_ID}_clear_cache"):
            _load_ml_data.clear()
            st.session_state[f'{TAB_ID}_refresh_counter'] += 1
            st.success("Cache cleared!")
            st.rerun()
    
    with col3:
        if st.button("🔍 Screen Stocks", key=f"{TAB_ID}_screen", help="Trigger screening workflow"):
            with st.spinner("Triggering screening workflow..."):
                if trigger_workflow("ml_screen_and_predict.yml", {"universe": "auto", "top_n": "50"}):
                    st.success("✓ Screening workflow triggered! Check back in 15-30 minutes.")
                else:
                    st.info("💡 Manually run: `python ml_screen_and_predict.py`")
    
    # Info banner
    st.info("""
    **🤖 Autonomous Screening System**  
    • Screens 500-1500 stocks daily before market open  
    • Generates top 50 predictions with target gains  
    • Tracks comprehensive accuracy (precision + recall)  
    • Learns from failures to improve weekly
    """)
    
    # Create sub-tabs
    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
        "🎯 Latest Predictions",
        "✅ Predictions vs Actuals",
        "❌ Missed Opportunities",
        "📈 Performance Trends",
        "ℹ️ System Info"
    ])
    
    with subtab1:
        render_latest_predictions(TAB_ID, refresh_key)
    
    with subtab2:
        render_predictions_vs_actuals(TAB_ID, refresh_key)
    
    with subtab3:
        render_missed_opportunities(TAB_ID, refresh_key)
    
    with subtab4:
        render_performance_trends(TAB_ID, refresh_key)
    
    with subtab5:
        render_system_info(TAB_ID, refresh_key)


# ===== SUB-TAB 1: LATEST PREDICTIONS =====

def render_latest_predictions(tab_id: str, refresh_key: int):
    """Show latest predictions from autonomous screening"""
    
    # Get available dates (cached)
    dates_df = load_ml_data(tab_id, "ml_explosion_predictions", 
                           order_by=("prediction_date", "desc"), 
                           limit=30, _refresh_key=refresh_key)
    
    if dates_df.empty:
        st.warning("📭 No predictions available yet.")
        st.info("Run the screening workflow or wait for the scheduled run (daily at 3 PM Estonia).")
        return
    
    dates = sorted(dates_df['prediction_date'].unique().tolist(), reverse=True)
    
    # Date selector
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_date = st.selectbox(
            "Select Date:",
            dates,
            format_func=lambda x: datetime.fromisoformat(x).strftime("%A, %B %d, %Y"),
            key=f"{tab_id}_date_{refresh_key}"
        )
    
    with col2:
        st.markdown(f"**Prediction Date:** {selected_date}")
        prediction_dt = datetime.fromisoformat(selected_date).date()
        today = datetime.now().date()
        if prediction_dt >= today:
            st.success("🔮 These are forward-looking predictions")
        else:
            st.info("📊 Historical predictions - check Predictions vs Actuals for results")
    
    # Load predictions for selected date (cached)
    df = load_ml_data(tab_id, "ml_explosion_predictions",
                     filters={"prediction_date": selected_date},
                     order_by=("explosion_probability", "desc"),
                     limit=200,
                     _refresh_key=refresh_key)
    
    if df.empty:
        st.warning(f"No predictions for {selected_date}")
        return
    
    # Summary metrics
    st.markdown("### 📊 Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Screened", len(df))
    
    with col2:
        strong_buys = len(df[df['signal'] == 'STRONG BUY'])
        st.metric("🟢 STRONG BUY", strong_buys)
    
    with col3:
        buys = len(df[df['signal'] == 'BUY'])
        st.metric("🔵 BUY", buys)
    
    with col4:
        avg_prob = df['explosion_probability'].mean() * 100
        st.metric("Avg Probability", f"{avg_prob:.1f}%")
    
    with col5:
        avg_target = df['target_gain_pct'].mean()
        st.metric("Avg Target Gain", f"+{avg_target:.1f}%")
    
    # Charts
    st.markdown("### 📈 Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Probability distribution
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df['explosion_probability'] * 100,
            nbinsx=20,
            marker_color='#667eea',
            name='Probability'
        ))
        
        fig.update_layout(
            title="Explosion Probability Distribution",
            xaxis_title="Probability (%)",
            yaxis_title="Count",
            height=300,
            plot_bgcolor='rgba(26, 29, 41, 0.6)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8eaf0'),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Signal breakdown
        signal_counts = df['signal'].value_counts()
        
        colors = {
            'STRONG BUY': '#10b981',
            'BUY': '#667eea',
            'HOLD': '#f59e0b',
            'AVOID': '#ef4444'
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=signal_counts.index,
            values=signal_counts.values,
            marker=dict(colors=[colors.get(s, '#999') for s in signal_counts.index]),
            hole=0.4
        )])
        
        fig.update_layout(
            title="Signal Breakdown",
            height=300,
            plot_bgcolor='rgba(26, 29, 41, 0.6)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8eaf0')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Filters
    st.markdown("### 🔍 Filter Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        signal_filter = st.multiselect(
            "Signal:",
            options=['STRONG BUY', 'BUY', 'HOLD', 'AVOID'],
            default=['STRONG BUY', 'BUY'],
            key=f"{tab_id}_signal_filter_{refresh_key}"
        )
    
    with col2:
        min_prob = st.slider(
            "Min Probability:",
            0, 100, 50,
            key=f"{tab_id}_min_prob_{refresh_key}"
        )
    
    with col3:
        min_target = st.slider(
            "Min Target Gain (%):",
            0, 50, 0,
            key=f"{tab_id}_min_target_{refresh_key}"
        )
    
    # Apply filters
    filtered_df = df[
        (df['signal'].isin(signal_filter)) &
        (df['explosion_probability'] >= min_prob / 100) &
        (df['target_gain_pct'] >= min_target)
    ].copy()
    
    st.markdown(f"### 📋 Top Predictions ({len(filtered_df)} stocks)")
    
    if filtered_df.empty:
        st.warning("No stocks match the filters")
        return
    
    # Prepare display
    display_df = filtered_df[[
        'symbol', 'exchange', 'signal', 'explosion_probability',
        'current_price', 'target_price', 'target_gain_pct',
        'target_price_low', 'target_price_high'
    ]].copy()
    
    display_df['explosion_probability'] = display_df['explosion_probability'] * 100
    
    # Color code by signal
    def highlight_signal(row):
        if row['signal'] == 'STRONG BUY':
            return ['background-color: #10b98133'] * len(row)
        elif row['signal'] == 'BUY':
            return ['background-color: #667eea33'] * len(row)
        elif row['signal'] == 'HOLD':
            return ['background-color: #f59e0b33'] * len(row)
        else:
            return ['background-color: #ef444433'] * len(row)
    
    st.dataframe(
        display_df.style.format({
            'explosion_probability': '{:.2f}%',
            'current_price': '${:.2f}',
            'target_price': '${:.2f}',
            'target_price_low': '${:.2f}',
            'target_price_high': '${:.2f}',
            'target_gain_pct': '+{:.2f}%'
        }, na_rep='-').apply(highlight_signal, axis=1),
        use_container_width=True,
        height=600
    )
    
    # Download
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"ml_predictions_{selected_date}.csv",
        mime="text/csv",
        key=f"{tab_id}_download_{refresh_key}"
    )


# ===== SUB-TAB 2: PREDICTIONS VS ACTUALS =====

def render_predictions_vs_actuals(tab_id: str, refresh_key: int):
    """Show how predictions performed vs reality"""
    
    st.markdown("### 🎯 Prediction Accuracy Analysis")
    st.info("Compare predictions against actual market outcomes. Updated daily after market close.")
    
    # Get available dates (cached)
    dates_df = load_ml_data(tab_id, "ml_prediction_accuracy",
                           order_by=("prediction_date", "desc"),
                           limit=30,
                           _refresh_key=refresh_key)
    
    if dates_df.empty:
        st.warning("📭 No accuracy data available yet.")
        st.info("Accuracy tracking runs automatically after market close (5:30 AM Estonia).")
        return
    
    dates = sorted(dates_df['prediction_date'].unique().tolist(), reverse=True)
    
    selected_date = st.selectbox(
        "Select Date:",
        dates,
        format_func=lambda x: datetime.fromisoformat(x).strftime("%A, %B %d, %Y"),
        key=f"{tab_id}_accuracy_date_{refresh_key}"
    )
    
    # Load accuracy data (cached)
    accuracy_df = load_ml_data(tab_id, "ml_prediction_accuracy",
                               filters={"prediction_date": selected_date},
                               limit=500,
                               _refresh_key=refresh_key)
    
    if accuracy_df.empty:
        st.warning(f"No accuracy data for {selected_date}")
        return
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total = len(accuracy_df)
    correct = accuracy_df['prediction_correct'].sum()
    winners = accuracy_df['became_winner'].sum()
    predicted_winners = len(accuracy_df[accuracy_df['predicted_signal'].isin(['STRONG BUY', 'BUY'])])
    
    with col1:
        st.metric("Total Predictions", total)
    
    with col2:
        accuracy_pct = (correct / total * 100) if total > 0 else 0
        st.metric("Accuracy", f"{accuracy_pct:.1f}%")
    
    with col3:
        st.metric("Actual Winners", winners)
    
    with col4:
        st.metric("Predicted Winners", predicted_winners)
    
    with col5:
        tp = len(accuracy_df[(accuracy_df['predicted_signal'].isin(['STRONG BUY', 'BUY'])) & 
                            (accuracy_df['became_winner'] == True)])
        precision = (tp / predicted_winners * 100) if predicted_winners > 0 else 0
        st.metric("Precision", f"{precision:.1f}%")
    
    # Confusion Matrix
    st.markdown("#### Confusion Matrix")
    
    tp = len(accuracy_df[(accuracy_df['predicted_signal'].isin(['STRONG BUY', 'BUY'])) & 
                         (accuracy_df['became_winner'] == True)])
    fp = len(accuracy_df[(accuracy_df['predicted_signal'].isin(['STRONG BUY', 'BUY'])) & 
                         (accuracy_df['became_winner'] == False)])
    fn = len(accuracy_df[(~accuracy_df['predicted_signal'].isin(['STRONG BUY', 'BUY'])) & 
                         (accuracy_df['became_winner'] == True)])
    tn = len(accuracy_df[(~accuracy_df['predicted_signal'].isin(['STRONG BUY', 'BUY'])) & 
                         (accuracy_df['became_winner'] == False)])
    
    matrix = [[tp, fp], [fn, tn]]
    labels = [[f'True Positive<br>{tp}', f'False Positive<br>{fp}'],
              [f'False Negative<br>{fn}', f'True Negative<br>{tn}']]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=['Actually Exploded', 'Didn\'t Explode'],
        y=['Predicted Explosion', 'Predicted No Explosion'],
        text=labels,
        texttemplate='%{text}',
        textfont={"size": 14, "color": "#ffffff"},
        colorscale=[[0, '#2d3142'], [1, '#10b981']],
        showscale=False
    ))
    
    fig.update_layout(
        height=400,
        plot_bgcolor='rgba(26, 29, 41, 0.6)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8eaf0')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.markdown("#### Detailed Results")
    
    comparison_df = accuracy_df[[
        'symbol', 'predicted_signal', 'predicted_probability',
        'predicted_target_gain', 'became_winner',
        'actual_gain_pct', 'prediction_correct'
    ]].copy()
    
    comparison_df = comparison_df.sort_values('predicted_probability', ascending=False)
    
    def highlight_correct(row):
        if row['prediction_correct']:
            return ['background-color: #10b98133'] * len(row)
        else:
            return ['background-color: #ef444433'] * len(row)
    
    st.dataframe(
        comparison_df.style.format({
            'predicted_probability': '{:.2%}',
            'predicted_target_gain': '{:.2f}%',
            'actual_gain_pct': '{:.2f}%'
        }, na_rep='-').apply(highlight_correct, axis=1),
        use_container_width=True,
        height=600
    )


# ===== SUB-TAB 3: MISSED OPPORTUNITIES =====

def render_missed_opportunities(tab_id: str, refresh_key: int):
    """Show winners we missed (recall analysis)"""
    
    st.markdown("### ❌ Missed Opportunities (Recall Analysis)")
    st.info("Winners we didn't predict. Critical for improving the model's ability to catch opportunities.")
    
    # Get missed opportunities summary (cached)
    summary_df = load_ml_data(tab_id, "v_ml_missed_summary",
                             order_by=("detection_date", "desc"),
                             limit=30,
                             _refresh_key=refresh_key)
    
    if summary_df.empty:
        st.warning("📭 No missed opportunities data yet.")
        return
    
    # Select date
    dates = summary_df['detection_date'].tolist()
    selected_date = st.selectbox(
        "Select Date:",
        dates,
        format_func=lambda x: datetime.fromisoformat(x).strftime("%A, %B %d, %Y"),
        key=f"{tab_id}_missed_date_{refresh_key}"
    )
    
    # Show summary for date
    date_summary = summary_df[summary_df['detection_date'] == selected_date].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Missed", int(date_summary['total_missed']))
    
    with col2:
        st.metric("Not Screened", int(date_summary['not_screened']))
    
    with col3:
        st.metric("Screened But Not Predicted", int(date_summary['screened_but_not_predicted']))
    
    with col4:
        st.metric("Avg Missed Gain", f"+{date_summary['avg_missed_gain']:.1f}%")
    
    # Get detailed missed opportunities (cached)
    missed_df = load_ml_data(tab_id, "ml_missed_opportunities",
                            filters={"detection_date": selected_date},
                            order_by=("actual_gain_pct", "desc"),
                            limit=200,
                            _refresh_key=refresh_key)
    
    if missed_df.empty:
        st.info("No detailed data for this date")
        return
    
    st.markdown("#### Why We Missed Them")
    
    # Breakdown by reason
    reason_counts = missed_df['screening_failure_reason'].fillna('screened_but_low_prob').value_counts()
    
    fig = go.Figure(data=[go.Bar(
        x=reason_counts.index,
        y=reason_counts.values,
        marker_color='#ef4444'
    )])
    
    fig.update_layout(
        title="Missed Opportunities by Reason",
        xaxis_title="Failure Reason",
        yaxis_title="Count",
        height=300,
        plot_bgcolor='rgba(26, 29, 41, 0.6)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8eaf0')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Detailed Missed Winners")
    
    display_df = missed_df[[
        'symbol', 'actual_gain_pct', 'was_screened',
        'screening_failure_reason', 'predicted_probability'
    ]].copy()
    
    st.dataframe(
        display_df.style.format({
            'actual_gain_pct': '+{:.2f}%',
            'predicted_probability': '{:.2%}'
        }, na_rep='-'),
        use_container_width=True,
        height=500
    )


# ===== SUB-TAB 4: PERFORMANCE TRENDS =====

def render_performance_trends(tab_id: str, refresh_key: int):
    """Show accuracy trends over time"""
    
    st.markdown("### 📈 Model Performance Trends")
    
    # Get accuracy summary (cached)
    summary_df = load_ml_data(tab_id, "v_ml_daily_accuracy_summary",
                             order_by=("prediction_date", "desc"),
                             limit=30,
                             _refresh_key=refresh_key)
    
    if summary_df.empty:
        st.warning("No performance data yet")
        return
    
    summary_df = summary_df.sort_values('prediction_date')
    
    # Accuracy over time
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=summary_df['prediction_date'],
        y=summary_df['accuracy_pct'],
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='#667eea', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color='#f59e0b',
        annotation_text="50% (Random)",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Prediction Accuracy Over Time",
        xaxis_title="Date",
        yaxis_title="Accuracy (%)",
        height=400,
        plot_bgcolor='rgba(26, 29, 41, 0.6)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8eaf0')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Signal performance (cached)
    st.markdown("#### Performance by Signal")
    
    signal_df = load_ml_data(tab_id, "v_ml_signal_performance",
                            limit=10,
                            _refresh_key=refresh_key)
    
    if not signal_df.empty:
        fig = go.Figure(data=[go.Bar(
            x=signal_df['predicted_signal'],
            y=signal_df['success_rate_pct'],
            text=signal_df['success_rate_pct'].apply(lambda x: f"{x:.1f}%"),
            textposition='auto',
            marker_color=['#10b981', '#667eea', '#f59e0b', '#ef4444']
        )])
        
        fig.update_layout(
            title="Success Rate by Signal Type",
            xaxis_title="Signal",
            yaxis_title="Success Rate (%)",
            height=400,
            plot_bgcolor='rgba(26, 29, 41, 0.6)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8eaf0')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            signal_df.style.format({
                'success_rate_pct': '{:.2f}%',
                'avg_confidence': '{:.2f}%',
                'avg_gain_when_correct': '+{:.2f}%',
                'avg_loss_when_wrong': '{:.2f}%'
            }, na_rep='-'),
            use_container_width=True
        )


# ===== SUB-TAB 5: SYSTEM INFO =====

def render_system_info(tab_id: str, refresh_key: int):
    """Show system information and statistics"""
    
    st.markdown("### ℹ️ System Information")
    
    # Get latest screening log (cached)
    log_df = load_ml_data(tab_id, "ml_screening_logs",
                         order_by=("screening_date", "desc"),
                         limit=1,
                         _refresh_key=refresh_key)
    
    if not log_df.empty:
        log = log_df.iloc[0]
        
        st.markdown("#### Latest Screening Run")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Date", log['screening_date'])
        
        with col2:
            st.metric("Stocks Attempted", log['total_symbols_attempted'])
        
        with col3:
            success_rate = (log['symbols_fetched_successfully'] / log['total_symbols_attempted'] * 100) if log['total_symbols_attempted'] > 0 else 0
            st.metric("Fetch Success", f"{success_rate:.1f}%")
        
        with col4:
            st.metric("Predictions Made", log['total_predictions'])
        
        st.markdown("#### Screening Statistics")
        
        stats_df = pd.DataFrame([{
            'Metric': 'Total Attempted',
            'Value': log['total_symbols_attempted']
        }, {
            'Metric': 'Successfully Fetched',
            'Value': log['symbols_fetched_successfully']
        }, {
            'Metric': 'After Price Filter',
            'Value': log['symbols_after_price_filter']
        }, {
            'Metric': 'After Volume Filter',
            'Value': log['symbols_after_volume_filter']
        }, {
            'Metric': 'Final Predictions',
            'Value': log['total_predictions']
        }])
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # System schedule
    st.markdown("---")
    st.markdown("#### 📅 Automated Schedule")
    
    st.markdown("""
    **Daily Schedule (Estonia Time):**
    
    - **3:00 PM** - Autonomous stock screening & prediction
      - Screens 500-1500 stocks
      - Generates top 50 predictions
      - Stores in database
    
    - **11:30 PM** - Daily winners collection (existing system)
      - Collects actual 20%+ gainers
      - Stores indicators
    
    - **5:30 AM (next day)** - Comprehensive accuracy tracking
      - Compares predictions vs actual winners
      - Analyzes missed opportunities
      - Records failure patterns
    
    **Weekly Schedule:**
    
    - **Sunday 9:00 AM** - Model retraining
      - Uses last 90 days of data
      - Learns from successes and failures
      - Updates model automatically
    """)
    
    st.markdown("---")
    st.markdown("#### ⚠️ Disclaimer")
    
    st.warning("""
    **Important:**
    - This is an experimental ML system for research purposes
    - Past performance does not guarantee future results
    - Always do your own research
    - Never invest more than you can afford to lose
    - This is not financial advice
    """)
    
    st.markdown("---")
    st.markdown("#### 🛠️ Model Details")
    
    st.info("""
    **Model Type:** XGBoost Classifier  
    **Training Data:** 90-day rolling window  
    **Features:** 97 technical indicators  
    **Target:** 20%+ single-day gains  
    **Retraining:** Weekly (Sundays)  
    **Learning:** Continuous improvement from actual outcomes
    """)
