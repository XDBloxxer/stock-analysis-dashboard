"""
Stock Event Analysis Dashboard - Enhanced Version
Improved visualizations with proper scaling and meaningful insights
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from pathlib import Path
import numpy as np
from scipy import stats

# Page config
st.set_page_config(
    page_title="Stock Pattern Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem 2rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    h1 {
        color: #2d3748;
        font-weight: 700;
    }
    h2 {
        color: #4a5568;
        font-weight: 600;
        margin-top: 2rem;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_google_sheets_client():
    """Initialize Google Sheets client"""
    try:
        credentials_dict = st.secrets["google_sheets_credentials"]
        credentials = Credentials.from_service_account_info(
            credentials_dict,
            scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        )
    except:
        credentials_path = "credentials/google_sheets_credentials.json"
        if Path(credentials_path).exists():
            credentials = Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
            )
        else:
            st.error("❌ Google Sheets credentials not found!")
            st.stop()
    
    return gspread.authorize(credentials)


@st.cache_data(ttl=300)
def load_sheet_data(spreadsheet_id, sheet_name):
    """Load data from Google Sheets"""
    try:
        client = get_google_sheets_client()
        spreadsheet = client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        data = worksheet.get_all_records()
        
        if not data:
            return pd.DataFrame()
        
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading {sheet_name}: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_all_time_lags(spreadsheet_id):
    """Load all time lag sheets"""
    time_lags = {}
    for lag in [1, 3, 5, 10, 30]:
        sheet_name = f"Raw Data_T-{lag}"
        df = load_sheet_data(spreadsheet_id, sheet_name)
        if not df.empty:
            time_lags[f"T-{lag}"] = df
    return time_lags


def normalize_indicator_values(df, indicator_col):
    """Normalize indicator values to 0-100 scale for comparison"""
    if df[indicator_col].isna().all():
        return df[indicator_col]
    
    min_val = df[indicator_col].min()
    max_val = df[indicator_col].max()
    
    if max_val == min_val:
        return pd.Series(50, index=df.index)
    
    return ((df[indicator_col] - min_val) / (max_val - min_val)) * 100


def categorize_indicators(columns):
    """Categorize indicators by type"""
    categories = {
        'Momentum': [],
        'Trend': [],
        'Volatility': [],
        'Volume': [],
        'Price': [],
        'Other': []
    }
    
    momentum_keywords = ['rsi', 'stoch', 'mom', 'macd', 'cci', 'williams', 'w.r', 'uo', 'ao']
    trend_keywords = ['ema', 'sma', 'adx', 'trend', 'above']
    volatility_keywords = ['atr', 'bb', 'volatility', 'bollinger']
    volume_keywords = ['volume', 'vol']
    price_keywords = ['close', 'open', 'high', 'low', 'price', 'change', 'gap', 'vwap']
    
    for col in columns:
        col_lower = col.lower()
        
        if any(kw in col_lower for kw in momentum_keywords):
            categories['Momentum'].append(col)
        elif any(kw in col_lower for kw in trend_keywords):
            categories['Trend'].append(col)
        elif any(kw in col_lower for kw in volatility_keywords):
            categories['Volatility'].append(col)
        elif any(kw in col_lower for kw in volume_keywords):
            categories['Volume'].append(col)
        elif any(kw in col_lower for kw in price_keywords):
            categories['Price'].append(col)
        else:
            categories['Other'].append(col)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def create_dual_comparison_chart(analysis_df, time_lag='T-1', top_n=15):
    """Create side-by-side chart showing both absolute values and percentage difference"""
    if analysis_df.empty or 'Time_Lag' not in analysis_df.columns:
        return None
    
    df = analysis_df[analysis_df['Time_Lag'] == time_lag].copy()
    
    if df.empty:
        return None
    
    # Filter out price change indicators
    df = filter_predictive_indicators(df)
    
    if df.empty:
        return None
    
    # Calculate percentage difference
    df['Pct_Difference'] = np.where(
        df['AVG_Grinders'].abs() > 0,
        ((df['AVG_Spikers'] - df['AVG_Grinders']) / df['AVG_Grinders'].abs()) * 100,
        0
    )
    
    # Cap extreme outliers
    cap_value = np.percentile(df['Pct_Difference'].abs(), 99)
    df['Pct_Difference_Display'] = df['Pct_Difference'].clip(-cap_value, cap_value)
    df['Is_Capped'] = (df['Pct_Difference'].abs() > cap_value)
    
    # Sort by absolute percentage difference
    df['Abs_Pct_Diff'] = df['Pct_Difference'].abs()
    df = df.nlargest(top_n, 'Abs_Pct_Diff')
    df = df.sort_values('Pct_Difference_Display')
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            '<b>Actual Average Values</b>',
            '<b>Percentage Difference</b>'
        ),
        horizontal_spacing=0.15,
        column_widths=[0.5, 0.5]
    )
    
    # Left plot: Actual values comparison
    fig.add_trace(
        go.Bar(
            name='Spikers',
            y=df['Indicator'],
            x=df['AVG_Spikers'],
            orientation='h',
            marker_color='#e74c3c',
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            name='Grinders',
            y=df['Indicator'],
            x=df['AVG_Grinders'],
            orientation='h',
            marker_color='#3498db',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Right plot: Percentage difference
    colors = ['#27ae60' if x > 0 else '#e67e22' for x in df['Pct_Difference']]
    
    fig.add_trace(
        go.Bar(
            y=df['Indicator'],
            x=df['Pct_Difference_Display'],
            orientation='h',
            marker_color=colors,
            text=[f'{x:+.1f}%{"*" if c else ""}' for x, c in zip(df['Pct_Difference_Display'], df['Is_Capped'])],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Add zero line to right plot
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2, row=1, col=2)
    
    # Calculate symmetric x-axis range for right plot to center zero
    max_abs_display = df['Pct_Difference_Display'].abs().max()
    x_range = [-max_abs_display * 1.15, max_abs_display * 1.15]
    
    # Update axes
    fig.update_xaxes(title_text="Average Value", row=1, col=1)
    fig.update_xaxes(
        title_text="% Difference", 
        range=x_range, 
        zeroline=True, 
        zerolinewidth=2, 
        zerolinecolor='gray',
        row=1, col=2
    )
    fig.update_yaxes(showticklabels=True, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    
    fig.update_layout(
        title_text=f"<b>Dual View: Absolute vs Relative Differences ({time_lag})</b><br>" +
                   "<sub>Left: Compare actual values | Right: Relative difference (capped at 99th percentile)<br>" +
                   "Price change indicators excluded | * = Extreme outlier (capped for display)</sub>",
        height=max(500, top_n * 30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        barmode='group',
        title_x=0.5
    )
    
    return fig


def create_top_discriminators_chart(analysis_df, time_lag='T-1', top_n=15):
    """Create chart showing top discriminating indicators with better context"""
    if analysis_df.empty or 'Time_Lag' not in analysis_df.columns:
        return None
    
    df = analysis_df[analysis_df['Time_Lag'] == time_lag].copy()
    
    if df.empty:
        return None
    
    # Filter out price change indicators
    df = filter_predictive_indicators(df)
    
    if df.empty:
        return None
    
    # Calculate percentage difference from grinder baseline
    df['Pct_Difference'] = np.where(
        df['AVG_Grinders'].abs() > 0,
        ((df['AVG_Spikers'] - df['AVG_Grinders']) / df['AVG_Grinders'].abs()) * 100,
        0
    )
    
    # Cap extreme outliers at 99th percentile to prevent one outlier from dominating
    cap_value = np.percentile(df['Pct_Difference'].abs(), 99)
    df['Pct_Difference_Display'] = df['Pct_Difference'].clip(-cap_value, cap_value)
    df['Is_Capped'] = (df['Pct_Difference'].abs() > cap_value)
    
    # Calculate absolute difference too
    df['Abs_Difference'] = df['AVG_Spikers'] - df['AVG_Grinders']
    
    # Sort by absolute percentage difference (using original uncapped values)
    df['Abs_Pct_Diff'] = df['Pct_Difference'].abs()
    df = df.nlargest(top_n, 'Abs_Pct_Diff')
    df = df.sort_values('Pct_Difference_Display')
    
    # Create color based on direction
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in df['Pct_Difference']]
    
    # Format numbers for display
    def format_num(x):
        if abs(x) >= 1_000_000:
            return f'{x/1_000_000:.1f}M'
        elif abs(x) >= 1_000:
            return f'{x/1_000:.1f}K'
        else:
            return f'{x:.1f}'
    
    # Create hover text with capping indicator
    hover_texts = []
    for _, row in df.iterrows():
        capped_text = " (CAPPED - actual: {:.1f}%)".format(row['Pct_Difference']) if row['Is_Capped'] else ""
        hover_texts.append(
            f"<b>{row['Indicator']}</b><br>" +
            f"Percentage Difference: {row['Pct_Difference_Display']:.1f}%{capped_text}<br>" +
            f"<br>" +
            f"Spiker Avg: {row['AVG_Spikers']:.2f}<br>" +
            f"Grinder Avg: {row['AVG_Grinders']:.2f}<br>" +
            f"Absolute Diff: {row['Abs_Difference']:.2f}"
        )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['Indicator'],
        x=df['Pct_Difference_Display'],
        orientation='h',
        marker_color=colors,
        text=[f'{x:+.1f}%{"*" if c else ""}' for x, c in zip(df['Pct_Difference_Display'], df['Is_Capped'])],
        textposition='outside',
        hovertext=hover_texts,
        hoverinfo='text'
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2)
    
    # Calculate symmetric x-axis range to center zero
    max_abs_display = df['Pct_Difference_Display'].abs().max()
    x_range = [-max_abs_display * 1.15, max_abs_display * 1.15]  # 15% padding
    
    fig.update_layout(
        title=f"<b>Top {top_n} Discriminating Indicators ({time_lag})</b><br>" +
              "<sub>Percentage shows: (Spiker Avg - Grinder Avg) / Grinder Avg × 100<br>" +
              "Red = Spikers higher | Blue = Grinders higher | * = Extreme outlier (capped for display)<br>" +
            
        xaxis_title="Percentage Difference (%)",
        yaxis_title="",
        height=max(500, top_n * 30),
        showlegend=False,
        title_x=0.5,
        xaxis=dict(range=x_range, zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    )
    
    return fig


def create_indicator_category_comparison(analysis_df, time_lag='T-1'):
    """Compare indicator categories to see which types are most predictive"""
    if analysis_df.empty or 'Time_Lag' not in analysis_df.columns:
        return None
    
    df = analysis_df[analysis_df['Time_Lag'] == time_lag].copy()
    
    if df.empty:
        return None
    
    # Filter out price change indicators
    df = filter_predictive_indicators(df)
    
    if df.empty:
        return None
    
    # Calculate percentage difference
    df['Pct_Difference'] = np.where(
        df['AVG_Grinders'].abs() > 0,
        ((df['AVG_Spikers'] - df['AVG_Grinders']) / df['AVG_Grinders'].abs()) * 100,
        0
    )
    
    # Categorize indicators
    categories = categorize_indicators(df['Indicator'].tolist())
    
    # Calculate average absolute difference per category
    category_stats = []
    for category, indicators in categories.items():
        cat_df = df[df['Indicator'].isin(indicators)]
        if not cat_df.empty:
            avg_abs_diff = cat_df['Pct_Difference'].abs().mean()
            count = len(cat_df)
            category_stats.append({
                'Category': category,
                'Avg_Abs_Diff_%': avg_abs_diff,
                'Count': count
            })
    
    stats_df = pd.DataFrame(category_stats).sort_values('Avg_Abs_Diff_%', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=stats_df['Category'],
        x=stats_df['Avg_Abs_Diff_%'],
        orientation='h',
        marker_color='#667eea',
        text=[f'{x:.1f}%<br>({c} ind.)' for x, c in zip(stats_df['Avg_Abs_Diff_%'], stats_df['Count'])],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Avg Difference: %{x:.1f}%<br>Indicators: %{customdata}<extra></extra>',
        customdata=stats_df['Count']
    ))
    
    fig.update_layout(
        title=f"<b>Indicator Category Predictive Power ({time_lag})</b><br>" +
              "<sub>Average absolute % difference by indicator type (price change excluded)</sub>",
        xaxis_title="Average Absolute % Difference",
        yaxis_title="",
        height=400,
        showlegend=False,
        title_x=0.5
    )
    
    return fig


def calculate_predictive_power(analysis_df, time_lag='T-1'):
    """Calculate predictive power score for each indicator"""
    if analysis_df.empty or 'Time_Lag' not in analysis_df.columns:
        return pd.DataFrame()
    
    df = analysis_df[analysis_df['Time_Lag'] == time_lag].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    # Filter out price change indicators
    df = filter_predictive_indicators(df)
    
    if df.empty:
        return pd.DataFrame()
    
    # Calculate normalized difference score
    df['Pct_Difference'] = np.where(
        df['AVG_Grinders'].abs() > 0,
        ((df['AVG_Spikers'] - df['AVG_Grinders']) / df['AVG_Grinders'].abs()) * 100,
        0
    )
    
    df['Predictive_Score'] = df['Pct_Difference'].abs()
    
    # Categorize
    categories = categorize_indicators(df['Indicator'].tolist())
    df['Category'] = df['Indicator'].apply(
        lambda x: next((cat for cat, inds in categories.items() if x in inds), 'Other')
    )
    
    return df[['Indicator', 'Category', 'AVG_Spikers', 'AVG_Grinders', 'Pct_Difference', 'Predictive_Score']].sort_values('Predictive_Score', ascending=False)


def filter_predictive_indicators(df):
    """
    Filter out indicators that shouldn't be used for prediction
    (like price change, which is what we're trying to predict)
    """
    # Keywords that indicate the indicator is the target variable, not a predictor
    exclude_keywords = [
        'change',           # price change
        'cambio',          # change in Spanish
        'pct_change',      # percentage change
        'price_change',    # explicit price change
        '_change_',        # any change metric
        'return',          # returns
        'performance'      # performance metrics
    ]
    
    # Filter out indicators with these keywords (case insensitive)
    if 'Indicator' in df.columns:
        mask = df['Indicator'].str.lower().apply(
            lambda x: not any(keyword in x for keyword in exclude_keywords)
        )
        return df[mask].copy()
    
    return df


def create_consistency_heatmap(analysis_df, top_n=20):
    """Show which indicators are consistently different across time lags"""
    if analysis_df.empty or 'Time_Lag' not in analysis_df.columns:
        return None
    
    # Filter out price change indicators
    analysis_df = filter_predictive_indicators(analysis_df)
    
    if analysis_df.empty:
        return None
    
    # Calculate percentage difference for all indicators
    analysis_df['Pct_Difference'] = np.where(
        analysis_df['AVG_Grinders'].abs() > 0,
        ((analysis_df['AVG_Spikers'] - analysis_df['AVG_Grinders']) / analysis_df['AVG_Grinders'].abs()) * 100,
        0
    )
    
    # Find indicators with highest average absolute difference across all time lags
    indicator_scores = analysis_df.groupby('Indicator')['Pct_Difference'].apply(
        lambda x: x.abs().mean()
    ).sort_values(ascending=False).head(top_n)
    
    top_indicators = indicator_scores.index.tolist()
    
    # Build pivot table
    pivot_df = analysis_df[analysis_df['Indicator'].isin(top_indicators)].pivot(
        index='Indicator',
        columns='Time_Lag',
        values='Pct_Difference'
    )
    
    # Sort by T-1 values
    if 'T-1' in pivot_df.columns:
        pivot_df = pivot_df.sort_values('T-1', ascending=False)
    
    # Sort columns by time lag
    time_lag_order = sorted(pivot_df.columns, key=lambda x: int(x.split('-')[1]))
    pivot_df = pivot_df[time_lag_order]
    
    # Use percentile-based color scaling to handle outliers
    # This ensures that moderate values are visible even with extreme outliers
    values = pivot_df.values.flatten()
    values = values[~np.isnan(values)]
    
    if len(values) > 0:
        # Use 5th and 95th percentile to clip extreme outliers
        p5 = np.percentile(values, 5)
        p95 = np.percentile(values, 95)
        
        # Make it symmetric around zero
        max_abs = max(abs(p5), abs(p95))
        zmin = -max_abs
        zmax = max_abs
    else:
        zmin = -10
        zmax = 10
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale=[
            [0, '#2E86DE'],      # Strong blue
            [0.5, '#F8F9FA'],    # Very light gray
            [1, '#EE5A6F']       # Strong red
        ],
        zmid=0,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title="% Diff"),
        hovertemplate='<b>%{y}</b><br>%{x}<br>Difference: %{z:.1f}%<extra></extra>',
        text=[[f'{val:.1f}%' if not pd.isna(val) else '' for val in row] for row in pivot_df.values],
        texttemplate='%{text}',
        textfont={"size": 9}
    ))
    
    fig.update_layout(
        title=f"<b>Indicator Consistency Across Time</b><br>" +
              "<sub>Red = Spikers higher | Blue = Grinders higher | Intensity = Magnitude<br>" +
              "Color scale adjusted to 5-95 percentile to show moderate values</sub>",
        xaxis_title="Time Lag (Days Before Event)",
        yaxis_title="",
        height=max(600, top_n * 30),
        title_x=0.5
    )
    
    return fig


def create_distribution_comparison(raw_data_dict, indicator, category='All'):
    """Compare distributions of a specific indicator by category"""
    if not raw_data_dict:
        return None
    
    # Use T-1 data for most recent patterns
    if 'T-1' not in raw_data_dict:
        return None
    
    df = raw_data_dict['T-1']
    
    if indicator not in df.columns or 'Event_Type' not in df.columns:
        return None
    
    spikers = df[df['Event_Type'] == 'Spiker'][indicator].dropna()
    grinders = df[df['Event_Type'] == 'Grinder'][indicator].dropna()
    
    if len(spikers) == 0 or len(grinders) == 0:
        return None
    
    # Create distribution plot
    fig = go.Figure()
    
    # Add histograms
    fig.add_trace(go.Histogram(
        x=spikers,
        name='Spikers',
        opacity=0.7,
        marker_color='#e74c3c',
        nbinsx=30
    ))
    
    fig.add_trace(go.Histogram(
        x=grinders,
        name='Grinders',
        opacity=0.7,
        marker_color='#3498db',
        nbinsx=30
    ))
    
    # Add mean lines
    fig.add_vline(x=spikers.mean(), line_dash="dash", line_color='#e74c3c', 
                  annotation_text=f'Spiker Mean: {spikers.mean():.2f}',
                  annotation_position="top")
    fig.add_vline(x=grinders.mean(), line_dash="dash", line_color='#3498db',
                  annotation_text=f'Grinder Mean: {grinders.mean():.2f}',
                  annotation_position="bottom")
    
    fig.update_layout(
        title=f"<b>{indicator} Distribution (T-1)</b>",
        xaxis_title=indicator,
        yaxis_title="Count",
        barmode='overlay',
        height=400,
        showlegend=True,
        title_x=0.5
    )
    
    return fig


def create_correlation_matrix(raw_data_dict, category_indicators):
    """Show correlation between indicators within a category"""
    if not raw_data_dict or not category_indicators:
        return None
    
    if 'T-1' not in raw_data_dict:
        return None
    
    df = raw_data_dict['T-1']
    
    # Filter to available indicators
    available = [ind for ind in category_indicators if ind in df.columns]
    
    if len(available) < 2:
        return None
    
    # Calculate correlation for spikers
    spikers = df[df['Event_Type'] == 'Spiker'][available].corr()
    grinders = df[df['Event_Type'] == 'Grinder'][available].corr()
    
    # Calculate difference in correlations
    corr_diff = spikers - grinders
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_diff.values,
        x=corr_diff.columns,
        y=corr_diff.index,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title="Correlation<br>Difference"),
        hovertemplate='%{x}<br>%{y}<br>Diff: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="<b>Correlation Difference: Spikers vs Grinders</b><br>" +
              "<sub>Red = Stronger correlation in Spikers</sub>",
        height=600,
        title_x=0.5
    )
    
    return fig


def calculate_predictive_power(analysis_df, time_lag='T-1'):
    """Calculate predictive power score for each indicator"""
    if analysis_df.empty or 'Time_Lag' not in analysis_df.columns:
        return pd.DataFrame()
    
    df = analysis_df[analysis_df['Time_Lag'] == time_lag].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    # Calculate normalized difference score
    df['Pct_Difference'] = np.where(
        df['AVG_Grinders'].abs() > 0,
        ((df['AVG_Spikers'] - df['AVG_Grinders']) / df['AVG_Grinders'].abs()) * 100,
        0
    )
    
    df['Predictive_Score'] = df['Pct_Difference'].abs()
    
    # Categorize
    categories = categorize_indicators(df['Indicator'].tolist())
    df['Category'] = df['Indicator'].apply(
        lambda x: next((cat for cat, inds in categories.items() if x in inds), 'Other')
    )
    
    return df[['Indicator', 'Category', 'AVG_Spikers', 'AVG_Grinders', 'Pct_Difference', 'Predictive_Score']].sort_values('Predictive_Score', ascending=False)


def main():
    """Main dashboard"""
    
    st.title("🎯 Stock Pattern Analysis Dashboard")
    st.markdown("**Discover what distinguishes explosive movers from steady grinders**")
    
    # Configuration
    with st.expander("⚙️ Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            spreadsheet_id = st.text_input(
                "Google Sheets ID",
                value="1rgJQiGk6eKc1Eks9cyn0X9ukkA3C6nk44QDMamPGwC0"
            )
        with col2:
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
    
    # Load data
    with st.spinner("Loading data..."):
        candidates_df = load_sheet_data(spreadsheet_id, "Candidates")
        analysis_df = load_sheet_data(spreadsheet_id, "Analysis")
        summary_df = load_sheet_data(spreadsheet_id, "Summary Stats")
        raw_data_dict = load_all_time_lags(spreadsheet_id)
    
    if analysis_df.empty and candidates_df.empty:
        st.warning("⚠️ No data found. Run the analysis pipeline first: `python main.py --all`")
        return
    
    # Summary metrics
    if not summary_df.empty:
        summary_dict = dict(zip(summary_df['Metric'], summary_df['Value']))
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Events", f"{summary_dict.get('total_events', 0):.0f}")
        with col2:
            st.metric("Spikers", f"{summary_dict.get('total_spikers', 0):.0f}")
        with col3:
            st.metric("Grinders", f"{summary_dict.get('total_grinders', 0):.0f}")
        with col4:
            st.metric("Avg Spiker Move", f"{summary_dict.get('avg_spiker_change_pct', 0):.1f}%")
        with col5:
            st.metric("Avg Grinder Move", f"{summary_dict.get('avg_grinder_change_pct', 0):.1f}%")
    
    st.markdown("---")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Key Discriminators",
        "📊 Category Analysis", 
        "🔍 Deep Dive",
        "📋 Raw Data"
    ])
    
    # TAB 1: KEY DISCRIMINATORS
    with tab1:
        if not analysis_df.empty and 'Time_Lag' in analysis_df.columns:
            st.header("🎯 Most Discriminating Indicators")
            
            available_lags = sorted(analysis_df['Time_Lag'].unique(), 
                                   key=lambda x: int(x.split('-')[1]))
            
            col1, col2 = st.columns([1, 2])
            with col1:
                selected_lag = st.selectbox("Time Period:", available_lags, index=0)
            with col2:
                top_n = st.slider("Number of indicators:", 10, 50, 20)
            
            st.info("💡 **How to read:** If RSI shows +50%, it means Spikers have RSI that's 50% higher than Grinders on average. " +
                   "Hover over bars to see actual values. Larger percentages = stronger discriminators.")
            
            # View mode selector
            view_mode = st.radio(
                "Chart View:",
                ["Percentage Only", "Dual View (Absolute + Percentage)"],
                horizontal=True,
                help="Dual View shows actual values alongside percentage differences for better context"
            )
            
            # Main discriminators chart
            if view_mode == "Percentage Only":
                fig = create_top_discriminators_chart(analysis_df, selected_lag, top_n)
            else:
                fig = create_dual_comparison_chart(analysis_df, selected_lag, top_n)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Consistency across time
            st.subheader("⏱️ Pattern Consistency Across Time")
            st.markdown("**Which indicators maintain their differences across all time periods?** " +
                       "Consistent patterns (same color across columns) are more reliable.")
            
            consistency_n = st.slider("Indicators to show:", 15, 50, 25, key='consistency_n')
            fig_heatmap = create_consistency_heatmap(analysis_df, consistency_n)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown("---")
            
            # Predictive power table
            st.subheader("📊 Predictive Power Rankings")
            
            # Add explanation expander
            with st.expander("ℹ️ How to Interpret the Percentage Difference"):
                st.markdown("""
                The **Percentage Difference** shows how much Spikers differ from Grinders, relative to the Grinder baseline.
                
                **Formula:** `(Spiker Average - Grinder Average) / |Grinder Average| × 100`
                
                **Examples:**
                - **RSI: +50%** → Spikers have RSI 50% higher than Grinders  
                  If Grinders avg RSI = 40, then Spikers avg RSI ≈ 60
                  
                - **Volume: +200%** → Spikers have 3× the volume of Grinders  
                  If Grinders avg 500K volume, then Spikers avg 1.5M volume
                  
                - **ADX: -30%** → Spikers have ADX 30% lower than Grinders  
                  If Grinders avg ADX = 30, then Spikers avg ADX ≈ 21
                
                **What to look for:**
                - **Large percentages** (>30%) = Strong discriminators
                - **Consistent sign across time lags** = Reliable pattern
                - **Category trends** = Which indicator types matter most
                """)
            
            power_df = calculate_predictive_power(analysis_df, selected_lag)
            
            if not power_df.empty:
                # Format display
                display_df = power_df.head(20).copy()
                display_df['Spiker Avg'] = display_df['AVG_Spikers'].apply(lambda x: f'{x:.2f}')
                display_df['Grinder Avg'] = display_df['AVG_Grinders'].apply(lambda x: f'{x:.2f}')
                display_df['% Difference'] = display_df['Pct_Difference'].apply(lambda x: f'{x:+.1f}%')
                display_df['Score'] = display_df['Predictive_Score'].apply(lambda x: f'{x:.1f}')
                
                st.dataframe(
                    display_df[['Indicator', 'Category', 'Spiker Avg', 'Grinder Avg', '% Difference', 'Score']],
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("Analysis data not available. Run the pipeline to generate insights.")
    
    # TAB 2: CATEGORY ANALYSIS
    with tab2:
        if not analysis_df.empty and 'Time_Lag' in analysis_df.columns:
            st.header("📊 Indicator Category Analysis")
            
            available_lags = sorted(analysis_df['Time_Lag'].unique(), 
                                   key=lambda x: int(x.split('-')[1]))
            
            selected_lag_cat = st.selectbox("Time Period:", available_lags, index=0, key='cat_lag')
            
            # Category comparison
            st.subheader("Category Predictive Power")
            fig_cat = create_indicator_category_comparison(analysis_df, selected_lag_cat)
            if fig_cat:
                st.plotly_chart(fig_cat, use_container_width=True)
            
            st.markdown("---")
            
            # Detailed category analysis
            st.subheader("Deep Dive by Category")
            
            # Get categories (filter out price change first)
            lag_df = analysis_df[analysis_df['Time_Lag'] == selected_lag_cat].copy()
            lag_df = filter_predictive_indicators(lag_df)
            
            if lag_df.empty:
                st.warning("No predictive indicators available after filtering.")
            else:
                categories = categorize_indicators(lag_df['Indicator'].tolist())
                
                selected_category = st.selectbox("Select Category:", list(categories.keys()))
                
                if selected_category and categories[selected_category]:
                    category_indicators = categories[selected_category]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Top indicators in category
                        cat_df = lag_df[lag_df['Indicator'].isin(category_indicators)].copy()
                        cat_df['Pct_Diff'] = np.where(
                            cat_df['AVG_Grinders'].abs() > 0,
                            ((cat_df['AVG_Spikers'] - cat_df['AVG_Grinders']) / cat_df['AVG_Grinders'].abs()) * 100,
                            0
                        )
                        
                        # Cap extreme outliers
                        cap_value = np.percentile(cat_df['Pct_Diff'].abs(), 99)
                        cat_df['Pct_Diff_Display'] = cat_df['Pct_Diff'].clip(-cap_value, cap_value)
                        
                        cat_df = cat_df.sort_values('Pct_Diff_Display', key=abs, ascending=False).head(10)
                        
                        fig_cat_detail = go.Figure()
                        colors = ['#e74c3c' if x > 0 else '#3498db' for x in cat_df['Pct_Diff']]
                        
                        fig_cat_detail.add_trace(go.Bar(
                            y=cat_df['Indicator'],
                            x=cat_df['Pct_Diff_Display'],
                            orientation='h',
                            marker_color=colors,
                            text=[f'{x:+.1f}%' for x in cat_df['Pct_Diff_Display']],
                            textposition='outside'
                        ))
                        
                        # Center zero line
                        max_abs = cat_df['Pct_Diff_Display'].abs().max()
                        x_range = [-max_abs * 1.15, max_abs * 1.15]
                        
                        fig_cat_detail.update_layout(
                            title=f"Top {selected_category} Indicators",
                            xaxis_title="% Difference",
                            height=400,
                            showlegend=False,
                            xaxis=dict(range=x_range, zeroline=True, zerolinewidth=2, zerolinecolor='gray')
                        )
                        
                        st.plotly_chart(fig_cat_detail, use_container_width=True)
                    
                    with col2:
                        # Distribution for selected indicator
                        if not cat_df.empty and raw_data_dict:
                            selected_ind = st.selectbox(
                                "View distribution:",
                                cat_df['Indicator'].tolist(),
                                key='dist_ind'
                            )
                            
                            if selected_ind:
                                fig_dist = create_distribution_comparison(raw_data_dict, selected_ind)
                                if fig_dist:
                                    st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Analysis data not available.")
    
    # TAB 3: DEEP DIVE
    with tab3:
        st.header("🔍 Indicator Deep Dive")
        
        if raw_data_dict and not analysis_df.empty:
            first_lag_df = list(raw_data_dict.values())[0]
            metadata_cols = {'Symbol', 'Event_Date', 'Event_Type', 'Exchange', 'Time_Lag'}
            indicator_cols = [col for col in first_lag_df.columns if col not in metadata_cols]
            
            if indicator_cols:
                # Categorize for easier selection
                categories = categorize_indicators(indicator_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_category = st.selectbox("Category:", ['All'] + list(categories.keys()))
                
                with col2:
                    if selected_category == 'All':
                        available_indicators = sorted(indicator_cols)
                    else:
                        available_indicators = sorted(categories[selected_category])
                    
                    selected_indicator = st.selectbox("Indicator:", available_indicators)
                
                if selected_indicator:
                    # Distribution comparison
                    fig_dist = create_distribution_comparison(raw_data_dict, selected_indicator)
                    if fig_dist:
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Statistical summary
                    st.subheader("Statistical Analysis")
                    
                    df_t1 = raw_data_dict.get('T-1', pd.DataFrame())
                    if not df_t1.empty and selected_indicator in df_t1.columns:
                        spikers = df_t1[df_t1['Event_Type'] == 'Spiker'][selected_indicator].dropna()
                        grinders = df_t1[df_t1['Event_Type'] == 'Grinder'][selected_indicator].dropna()
                        
                        if len(spikers) > 1 and len(grinders) > 1:
                            t_stat, p_value = stats.ttest_ind(spikers, grinders)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Spiker Mean", f"{spikers.mean():.2f}")
                            with col2:
                                st.metric("Grinder Mean", f"{grinders.mean():.2f}")
                            with col3:
                                diff_pct = ((spikers.mean() - grinders.mean()) / abs(grinders.mean()) * 100) if grinders.mean() != 0 else 0
                                st.metric("% Difference", f"{diff_pct:+.1f}%")
                            with col4:
                                significance = "✅ Significant" if p_value < 0.05 else "❌ Not Significant"
                                st.metric("P-Value", f"{p_value:.4f}", delta=significance)
                            
                            # Interpretation
                            if p_value < 0.05:
                                st.success(f"✅ **Statistically significant difference** (p < 0.05). " +
                                         f"This indicator reliably distinguishes between Spikers and Grinders.")
                            else:
                                st.warning(f"⚠️ **Not statistically significant** (p = {p_value:.4f}). " +
                                         f"Differences may be due to chance.")
        else:
            st.info("Load raw data to perform deep dive analysis")
    
    # TAB 4: RAW DATA
    with tab4:
        st.header("📋 Data Explorer")
        
        data_view = st.radio(
            "Select data view:",
            ["Candidates", "Analysis Results", "Raw Data"],
            horizontal=True
        )
        
        if data_view == "Candidates" and not candidates_df.empty:
            st.subheader("Candidate Events")
            st.dataframe(candidates_df, use_container_width=True, height=400)
            
            csv = candidates_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "candidates.csv", "text/csv")
        
        elif data_view == "Analysis Results" and not analysis_df.empty:
            st.subheader("Analysis Results")
            
            if 'Time_Lag' in analysis_df.columns:
                selected_view_lag = st.selectbox(
                    "Filter by time lag:",
                    ['All'] + sorted(analysis_df['Time_Lag'].unique())
                )
                
                if selected_view_lag != 'All':
                    display_df = analysis_df[analysis_df['Time_Lag'] == selected_view_lag]
                else:
                    display_df = analysis_df
            else:
                display_df = analysis_df
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            csv = display_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "analysis.csv", "text/csv")
        
        elif data_view == "Raw Data" and raw_data_dict:
            st.subheader("Raw Indicator Data")
            
            selected_raw_lag = st.selectbox("Select time lag:", list(raw_data_dict.keys()))
            
            if selected_raw_lag in raw_data_dict:
                raw_df = raw_data_dict[selected_raw_lag]
                st.info(f"Shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
                st.dataframe(raw_df, use_container_width=True, height=400)
                
                csv = raw_df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, f"raw_data_{selected_raw_lag}.csv", "text/csv")
        else:
            st.info("No data available for selected view")


if __name__ == "__main__":
    main()
