"""
ML Predictions Tab - AUTONOMOUS SYSTEM
PERSISTENT CACHE - survives browser refresh, minimizes egress

REAL TABLE MAPPING:
  ml_explosion_predictions  → Latest Predictions (subtab 1)
  ml_prediction_accuracy    → Predictions vs Actuals (subtab 2)
  ml_missed_opportunities   → Missed Opportunities (subtab 3) — per-symbol rows only
  ml_prediction_accuracy    → Performance Trends (subtab 4) — aggregated client-side
  ml_screening_logs         → System Info (subtab 5) — actual columns only

EGRESS STRATEGY:
  - All DB calls go through _load_ml_data() which is @st.cache_resource
  - Cache key = (tab_id, table, filter_key, order_by, limit, refresh_key)
  - refresh_key only changes on manual Refresh/Clear Cache button press
  - Expensive aggregations (trends, signal perf) are computed client-side
    from already-cached data — zero extra DB calls
  - Date-list queries re-use the same cached full-table slice (limit=500)
    rather than firing a separate distinct query
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
from supabase import create_client, Client
import requests


# ──────────────────────────────────────────────────────────────────────────────
# SUPABASE CLIENT — shared, cached forever
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets.get("supabase", {}).get("url")
    key = st.secrets.get("supabase", {}).get("key")
    if not url or not key:
        raise ValueError("Missing Supabase credentials")
    return create_client(url, key)


# ──────────────────────────────────────────────────────────────────────────────
# CORE CACHED LOADER — all DB reads funnel through here
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _load_ml_data(
    _tab_id: str,
    table_name: str,
    filter_key: tuple = None,       # hashable version of filters dict
    select_cols: str = "*",
    order_by: tuple = None,         # (col, "asc"|"desc")
    limit: int = 500,
    _refresh_key: int = 0,
):
    """
    PERSISTENT CACHE — only re-fetches when _refresh_key changes.
    Survives browser close/refresh.
    _tab_id isolates each tab's cache namespace.
    select_cols lets callers request only the columns they need (saves egress).
    """
    filters = dict(filter_key) if filter_key else None
    order_col, order_dir = order_by if order_by else (None, None)

    try:
        client = get_supabase_client()
        query = client.table(table_name).select(select_cols)

        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)

        if order_col:
            query = query.order(order_col, desc=(order_dir == "desc"))

        if limit:
            query = query.limit(limit)

        response = query.execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()

    except Exception as e:
        st.warning(f"Could not load `{table_name}`: {e}")
        return pd.DataFrame()


def load_ml_data(
    tab_id: str,
    table_name: str,
    filters: dict = None,
    select_cols: str = "*",
    order_by: tuple = None,
    limit: int = 500,
    refresh_key: int = 0,
) -> pd.DataFrame:
    """Public wrapper — converts mutable dict to hashable tuple for cache key."""
    filter_key = tuple(sorted(filters.items())) if filters else None
    return _load_ml_data(tab_id, table_name, filter_key, select_cols, order_by, limit, refresh_key)


# ──────────────────────────────────────────────────────────────────────────────
# GITHUB WORKFLOW TRIGGERS
# ──────────────────────────────────────────────────────────────────────────────

def _trigger_workflow(workflow_id: str, label: str) -> bool:
    try:
        token      = st.secrets.get("secrets", {}).get("G_TOKEN")
        repo_owner = st.secrets.get("secrets", {}).get("GITHUB_REPO_OWNER")
        repo_name  = st.secrets.get("secrets", {}).get("GITHUB_REPO_NAME", "")

        if "/" in repo_name and not repo_owner:
            repo_owner, repo_name = repo_name.split("/", 1)

        if not all([token, repo_owner, repo_name]):
            st.error("❌ GitHub credentials not fully configured.")
            return False

        url = (
            f"https://api.github.com/repos/{repo_owner}/{repo_name}"
            f"/actions/workflows/{workflow_id}/dispatches"
        )
        resp = requests.post(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={"ref": "main", "inputs": {}},
        )
        if resp.status_code == 204:
            st.success(f"✅ {label} triggered!")
            return True
        st.error(f"Workflow trigger failed {resp.status_code}: {resp.text}")
        return False
    except Exception as e:
        st.error(f"Error triggering {label}: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# SHARED CHART THEME
# ──────────────────────────────────────────────────────────────────────────────

_LAYOUT = dict(
    plot_bgcolor="rgba(26, 29, 41, 0.6)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e8eaf0"),
    margin=dict(t=40, b=20, l=20, r=20),
)
_AXIS = dict(gridcolor="rgba(255,255,255,0.1)", color="#b8bcc8")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TAB ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def render_ml_predictions_tab():
    TAB_ID = "ml_predictions"

    if f"{TAB_ID}_refresh_counter" not in st.session_state:
        st.session_state[f"{TAB_ID}_refresh_counter"] = 0

    refresh_key = st.session_state[f"{TAB_ID}_refresh_counter"]

    st.subheader("🤖 ML Explosion Predictions (Autonomous)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔄 Refresh", key=f"{TAB_ID}_refresh"):
            st.session_state[f"{TAB_ID}_refresh_counter"] += 1
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Cache", key=f"{TAB_ID}_clear_cache"):
            _load_ml_data.clear()
            st.session_state[f"{TAB_ID}_refresh_counter"] += 1
            st.success("Cache cleared!")
            st.rerun()
    with col3:
        if st.button("🔍 Screen Stocks", key=f"{TAB_ID}_screen"):
            with st.spinner("Triggering screening workflow..."):
                _trigger_workflow("ml_screen_and_predict.yml", "Screening workflow")
    with col4:
        if st.button("📊 Track Accuracy", key=f"{TAB_ID}_track"):
            with st.spinner("Triggering accuracy tracking..."):
                _trigger_workflow("ml_track_accuracy.yml", "Accuracy tracking workflow")

    st.info(
        "**🤖 Autonomous Screening System** — screens 500-1500 stocks daily, "
        "generates top predictions with target gains, and tracks comprehensive accuracy."
    )

    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
        "🎯 Latest Predictions",
        "✅ Predictions vs Actuals",
        "❌ Missed Opportunities",
        "📈 Performance Trends",
        "ℹ️ System Info",
    ])

    with subtab1:
        _render_latest_predictions(TAB_ID, refresh_key)
    with subtab2:
        _render_predictions_vs_actuals(TAB_ID, refresh_key)
    with subtab3:
        _render_missed_opportunities(TAB_ID, refresh_key)
    with subtab4:
        _render_performance_trends(TAB_ID, refresh_key)
    with subtab5:
        _render_system_info(TAB_ID, refresh_key)


# ──────────────────────────────────────────────────────────────────────────────
# SUB-TAB 1 — LATEST PREDICTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _render_latest_predictions(tab_id: str, refresh_key: int):
    # One fetch — reused for date list AND current-date data via client-side filter
    all_preds = load_ml_data(
        tab_id, "ml_explosion_predictions",
        select_cols=(
            "prediction_date,symbol,exchange,signal,explosion_probability,"
            "current_price,target_price,target_gain_pct,target_price_low,target_price_high"
        ),
        order_by=("prediction_date", "desc"),
        limit=500,
        refresh_key=refresh_key,
    )

    if all_preds.empty:
        st.warning("📭 No predictions available yet.")
        st.info("Run the screening workflow or wait for the scheduled run.")
        return

    dates = sorted(all_preds["prediction_date"].unique().tolist(), reverse=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_date = st.selectbox(
            "Select Date:",
            dates,
            format_func=lambda x: datetime.fromisoformat(x).strftime("%A, %B %d, %Y"),
            key=f"{tab_id}_pred_date_{refresh_key}",
        )
    with col2:
        pred_dt = datetime.fromisoformat(selected_date).date()
        today   = datetime.now().date()
        if pred_dt >= today:
            st.success("🔮 Forward-looking predictions")
        else:
            st.info("📊 Historical — see Predictions vs Actuals for results")

    # Filter client-side — no extra DB call
    df = all_preds[all_preds["prediction_date"] == selected_date].copy()

    if df.empty:
        st.warning(f"No predictions for {selected_date}")
        return

    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Screened", len(df))
    col2.metric("🟢 STRONG BUY", int((df["signal"] == "STRONG BUY").sum()))
    col3.metric("🔵 BUY", int((df["signal"] == "BUY").sum()))
    col4.metric("Avg Probability", f"{df['explosion_probability'].mean() * 100:.1f}%")
    col5.metric("Avg Target Gain", f"+{df['target_gain_pct'].mean():.1f}%")

    # Charts (side by side)
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Histogram(
            x=df["explosion_probability"] * 100, nbinsx=20,
            marker_color="#667eea",
        ))
        fig.update_layout(title="Probability Distribution", xaxis_title="Probability (%)",
                          yaxis_title="Count", height=300, showlegend=False, **_LAYOUT)
        fig.update_xaxes(**_AXIS); fig.update_yaxes(**_AXIS)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sc = df["signal"].value_counts()
        _colors = {"STRONG BUY": "#10b981", "BUY": "#667eea", "HOLD": "#f59e0b", "AVOID": "#ef4444"}
        fig = go.Figure(go.Pie(
            labels=sc.index, values=sc.values,
            marker=dict(colors=[_colors.get(s, "#999") for s in sc.index]),
            hole=0.4,
        ))
        fig.update_layout(title="Signal Breakdown", height=300, **_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # Filters
    st.markdown("### 🔍 Filter")
    col1, col2, col3 = st.columns(3)
    with col1:
        sig_filter = st.multiselect(
            "Signal:", ["STRONG BUY", "BUY", "HOLD", "AVOID"],
            default=["STRONG BUY", "BUY", "HOLD", "AVOID"],
            key=f"{tab_id}_sig_f_{refresh_key}",
        )
    with col2:
        min_prob = st.slider("Min Probability:", 0, 100, 0, key=f"{tab_id}_prob_f_{refresh_key}")
    with col3:
        min_tgt  = st.slider("Min Target Gain (%):", 0, 50, 0, key=f"{tab_id}_tgt_f_{refresh_key}")

    fdf = df[
        df["signal"].isin(sig_filter) &
        (df["explosion_probability"] >= min_prob / 100) &
        (df["target_gain_pct"] >= min_tgt)
    ].copy()

    st.markdown(f"### 📋 Predictions ({len(fdf)} stocks)")
    if fdf.empty:
        st.warning("No stocks match the filters")
        return

    fdf["explosion_probability"] = fdf["explosion_probability"] * 100

    _SIG_BG = {"STRONG BUY": "#10b98133", "BUY": "#667eea33", "HOLD": "#f59e0b33", "AVOID": "#ef444433"}

    def _highlight_sig(row):
        bg = _SIG_BG.get(row["signal"], "")
        return [f"background-color: {bg}"] * len(row)

    display_cols = ["symbol", "exchange", "signal", "explosion_probability",
                    "current_price", "target_price", "target_gain_pct",
                    "target_price_low", "target_price_high"]
    display_cols = [c for c in display_cols if c in fdf.columns]

    st.dataframe(
        fdf[display_cols].style.format({
            "explosion_probability": "{:.2f}%",
            "current_price":         "${:.2f}",
            "target_price":          "${:.2f}",
            "target_price_low":      "${:.2f}",
            "target_price_high":     "${:.2f}",
            "target_gain_pct":       "+{:.2f}%",
        }, na_rep="-").apply(_highlight_sig, axis=1),
        use_container_width=True, height=600,
    )

    st.download_button(
        "📥 Download CSV", fdf[display_cols].to_csv(index=False),
        f"ml_predictions_{selected_date}.csv", "text/csv",
        key=f"{tab_id}_dl_{refresh_key}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# SUB-TAB 2 — PREDICTIONS VS ACTUALS
# ──────────────────────────────────────────────────────────────────────────────

def _render_predictions_vs_actuals(tab_id: str, refresh_key: int):
    st.markdown("### 🎯 Prediction Accuracy Analysis")
    st.info("Compare predictions against actual market outcomes.")

    # Columns that actually exist in ml_prediction_accuracy (from tracker code):
    #   symbol, prediction_date, predicted_probability, predicted_signal,
    #   predicted_target_gain, predicted_target_price, became_winner,
    #   actual_gain_pct, actual_high_pct, actual_price,
    #   prediction_correct, gain_error_pct, gain_error_ratio, actual_recorded_at
    SELECT = (
        "symbol,prediction_date,predicted_probability,predicted_signal,"
        "predicted_target_gain,became_winner,actual_gain_pct,actual_high_pct,"
        "actual_price,prediction_correct,gain_error_pct"
    )

    all_acc = load_ml_data(
        tab_id, "ml_prediction_accuracy",
        select_cols=SELECT,
        order_by=("prediction_date", "desc"),
        limit=500,
        refresh_key=refresh_key,
    )

    if all_acc.empty:
        st.warning("📭 No accuracy data available yet.")
        st.info("Accuracy tracking runs automatically after market close.")
        return

    dates = sorted(all_acc["prediction_date"].unique().tolist(), reverse=True)
    selected_date = st.selectbox(
        "Select Date:", dates,
        format_func=lambda x: datetime.fromisoformat(x).strftime("%A, %B %d, %Y"),
        key=f"{tab_id}_acc_date_{refresh_key}",
    )

    df = all_acc[all_acc["prediction_date"] == selected_date].copy()

    if df.empty:
        st.warning(f"No accuracy data for {selected_date}")
        return

    # Coerce types
    df["became_winner"]       = df["became_winner"].astype(bool)
    df["prediction_correct"]  = df["prediction_correct"].astype(bool)
    df["predicted_probability"] = pd.to_numeric(df["predicted_probability"], errors="coerce")
    df["actual_gain_pct"]     = pd.to_numeric(df["actual_gain_pct"],     errors="coerce")
    df["actual_high_pct"]     = pd.to_numeric(df["actual_high_pct"],     errors="coerce")
    df["gain_error_pct"]      = pd.to_numeric(df["gain_error_pct"],      errors="coerce")

    pos_mask = df["predicted_signal"].isin(["STRONG BUY", "BUY"])
    total             = len(df)
    correct           = int(df["prediction_correct"].sum())
    actual_winners    = int(df["became_winner"].sum())
    predicted_winners = int(pos_mask.sum())
    tp = int((pos_mask & df["became_winner"]).sum())
    fp = int((pos_mask & ~df["became_winner"]).sum())
    fn = int((~pos_mask & df["became_winner"]).sum())
    tn = int((~pos_mask & ~df["became_winner"]).sum())

    accuracy_pct = correct / total * 100 if total else 0
    precision    = tp / predicted_winners * 100 if predicted_winners else 0
    recall       = tp / actual_winners * 100 if actual_winners else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total",            total)
    col2.metric("Accuracy",         f"{accuracy_pct:.1f}%")
    col3.metric("Actual Winners",   actual_winners)
    col4.metric("Pred. Winners",    predicted_winners)
    col5.metric("Precision",        f"{precision:.1f}%")
    col6.metric("Recall",           f"{recall:.1f}%")

    # Additional gain metrics (only possible now that yfinance fills actual_gain_pct)
    gain_populated = df["actual_gain_pct"].notna().sum()
    if gain_populated > 0:
        st.caption(f"actual_gain_pct populated for {gain_populated}/{total} symbols")
        col1, col2, col3, col4 = st.columns(4)
        winner_gains = df.loc[df["became_winner"] & df["actual_gain_pct"].notna(), "actual_gain_pct"]
        non_gains    = df.loc[~df["became_winner"] & df["actual_gain_pct"].notna(), "actual_gain_pct"]
        col1.metric("Avg Winner Gain",     f"{winner_gains.mean():.2f}%" if not winner_gains.empty else "—")
        col2.metric("Avg Non-Winner Gain", f"{non_gains.mean():.2f}%"   if not non_gains.empty   else "—")
        err_df = df.loc[df["gain_error_pct"].notna(), "gain_error_pct"]
        col3.metric("Avg Gain Error",      f"{err_df.mean():.2f}%"      if not err_df.empty      else "—")
        high_populated = df["actual_high_pct"].notna().sum()
        col4.metric("Intraday High Pct populated", high_populated)

    # Confusion matrix heatmap
    st.markdown("#### Confusion Matrix")
    labels = [
        [f"True Positive<br>{tp}", f"False Positive<br>{fp}"],
        [f"False Negative<br>{fn}", f"True Negative<br>{tn}"],
    ]
    fig = go.Figure(go.Heatmap(
        z=[[tp, fp], [fn, tn]],
        x=["Actually Exploded", "Didn't Explode"],
        y=["Predicted Explosion", "Predicted No Explosion"],
        text=labels,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "#ffffff"},
        colorscale=[[0, "#2d3142"], [1, "#10b981"]],
        showscale=False,
    ))
    fig.update_layout(height=350, **_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Gain distribution for predicted-positive stocks
    if gain_populated > 0:
        st.markdown("#### Actual Gain Distribution")
        fig = go.Figure()
        g_tp = df.loc[pos_mask & df["became_winner"] & df["actual_gain_pct"].notna(), "actual_gain_pct"]
        g_fp = df.loc[pos_mask & ~df["became_winner"] & df["actual_gain_pct"].notna(), "actual_gain_pct"]
        if not g_tp.empty:
            fig.add_trace(go.Histogram(x=g_tp, nbinsx=20, name="True Positive",
                                       marker_color="#10b981", opacity=0.75))
        if not g_fp.empty:
            fig.add_trace(go.Histogram(x=g_fp, nbinsx=20, name="False Positive",
                                       marker_color="#ef4444", opacity=0.75))
        fig.update_layout(barmode="overlay", xaxis_title="Actual Gain %", yaxis_title="Count",
                          height=350, **_LAYOUT)
        fig.update_xaxes(**_AXIS); fig.update_yaxes(**_AXIS)
        st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown("#### Detailed Results")
    display_cols = [c for c in [
        "symbol", "predicted_signal", "predicted_probability",
        "predicted_target_gain", "became_winner", "actual_gain_pct",
        "actual_high_pct", "gain_error_pct", "prediction_correct",
    ] if c in df.columns]

    def _highlight_correct(row):
        if row.get("prediction_correct"):
            return ["background-color: #10b98133"] * len(row)
        return ["background-color: #ef444433"] * len(row)

    fmt = {
        "predicted_probability": "{:.2%}",
        "predicted_target_gain": "{:.2f}%",
        "actual_gain_pct":       "{:.2f}%",
        "actual_high_pct":       "{:.2f}%",
        "gain_error_pct":        "{:.2f}%",
    }
    st.dataframe(
        df[display_cols].sort_values("predicted_probability", ascending=False)
        .style.format({k: v for k, v in fmt.items() if k in display_cols}, na_rep="-")
        .apply(_highlight_correct, axis=1),
        use_container_width=True, height=600,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SUB-TAB 3 — MISSED OPPORTUNITIES
# ──────────────────────────────────────────────────────────────────────────────

def _render_missed_opportunities(tab_id: str, refresh_key: int):
    """
    Real schema (from ml_track_comprehensive_accuracy.py):
        symbol, detection_date, exchange, actual_gain_pct, actual_high_pct,
        actual_price, actual_volume, was_screened, screening_failure_reason,
        predicted_probability, predicted_signal
    Aggregation is done client-side — no separate summary table.
    """
    st.markdown("### ❌ Missed Opportunities (Recall Analysis)")
    st.info("Winners we didn't predict — grouped and aggregated client-side from raw rows.")

    SELECT = (
        "symbol,detection_date,exchange,actual_gain_pct,actual_high_pct,"
        "actual_price,actual_volume,was_screened,screening_failure_reason,"
        "predicted_probability,predicted_signal"
    )

    all_missed = load_ml_data(
        tab_id, "ml_missed_opportunities",
        select_cols=SELECT,
        order_by=("detection_date", "desc"),
        limit=500,
        refresh_key=refresh_key,
    )

    if all_missed.empty:
        st.warning("📭 No missed opportunities data yet.")
        return

    all_missed["actual_gain_pct"] = pd.to_numeric(all_missed["actual_gain_pct"], errors="coerce")
    all_missed["was_screened"]     = all_missed["was_screened"].astype(bool)

    dates = sorted(all_missed["detection_date"].unique().tolist(), reverse=True)
    selected_date = st.selectbox(
        "Select Date:", dates,
        format_func=lambda x: datetime.fromisoformat(x).strftime("%A, %B %d, %Y"),
        key=f"{tab_id}_missed_date_{refresh_key}",
    )

    df = all_missed[all_missed["detection_date"] == selected_date].copy()

    if df.empty:
        st.info("No missed opportunities for this date")
        return

    # Aggregate client-side
    total_missed    = len(df)
    not_screened    = int((~df["was_screened"]).sum())
    screened_no_pred = int(df["was_screened"].sum())
    avg_gain        = df["actual_gain_pct"].mean()
    best_gain       = df["actual_gain_pct"].max()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Missed",            total_missed)
    col2.metric("Not Screened",            not_screened)
    col3.metric("Screened, Not Predicted", screened_no_pred)
    col4.metric("Avg Missed Gain",         f"+{avg_gain:.1f}%" if pd.notna(avg_gain) else "—")
    col5.metric("Best Missed Gain",        f"+{best_gain:.1f}%" if pd.notna(best_gain) else "—")

    # Failure reasons bar chart
    st.markdown("#### Why We Missed Them")
    reason_col = df["screening_failure_reason"].fillna(
        df["was_screened"].map({True: "screened_but_low_probability", False: "not_in_screener"})
    )
    reason_counts = reason_col.value_counts()

    fig = go.Figure(go.Bar(
        x=reason_counts.index.tolist(),
        y=reason_counts.values.tolist(),
        marker_color="#ef4444",
        text=reason_counts.values.tolist(),
        textposition="outside",
    ))
    fig.update_layout(
        title="Missed Opportunities by Failure Reason",
        xaxis_title="Reason", yaxis_title="Count",
        height=350, **_LAYOUT,
    )
    fig.update_xaxes(**_AXIS); fig.update_yaxes(**_AXIS)
    st.plotly_chart(fig, use_container_width=True)

    # Gain distribution of missed opportunities
    gain_data = df["actual_gain_pct"].dropna()
    if not gain_data.empty:
        fig = go.Figure(go.Histogram(
            x=gain_data, nbinsx=20, marker_color="#f59e0b",
        ))
        fig.update_layout(
            title="Distribution of Missed Gains",
            xaxis_title="Actual Gain %", yaxis_title="Count",
            height=300, **_LAYOUT,
        )
        fig.update_xaxes(**_AXIS); fig.update_yaxes(**_AXIS)
        st.plotly_chart(fig, use_container_width=True)

    # Detail table
    st.markdown("#### Missed Winners Detail")
    display_cols = [c for c in [
        "symbol", "exchange", "actual_gain_pct", "actual_high_pct",
        "actual_price", "actual_volume", "was_screened",
        "screening_failure_reason", "predicted_probability",
    ] if c in df.columns]

    st.dataframe(
        df[display_cols].sort_values("actual_gain_pct", ascending=False)
        .style.format({
            "actual_gain_pct":      "+{:.2f}%",
            "actual_high_pct":      "+{:.2f}%",
            "actual_price":         "${:.2f}",
            "actual_volume":        "{:,.0f}",
            "predicted_probability": "{:.2%}",
        }, na_rep="-"),
        use_container_width=True, height=500,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SUB-TAB 4 — PERFORMANCE TRENDS
# ──────────────────────────────────────────────────────────────────────────────

def _render_performance_trends(tab_id: str, refresh_key: int):
    """
    No DB views used — all aggregation is client-side from ml_prediction_accuracy.
    The data is already cached from sub-tab 2's fetch, but since cache keys differ
    (select_cols, order_by) we use the same arguments to hit the same cached entry.
    """
    st.markdown("### 📈 Model Performance Trends")

    SELECT = (
        "symbol,prediction_date,predicted_signal,predicted_probability,"
        "became_winner,actual_gain_pct,prediction_correct,gain_error_pct"
    )

    all_acc = load_ml_data(
        tab_id, "ml_prediction_accuracy",
        select_cols=SELECT,
        order_by=("prediction_date", "desc"),
        limit=500,
        refresh_key=refresh_key,
    )

    if all_acc.empty:
        st.warning("No accuracy data available yet.")
        return

    all_acc["became_winner"]      = all_acc["became_winner"].astype(bool)
    all_acc["prediction_correct"] = all_acc["prediction_correct"].astype(bool)
    all_acc["actual_gain_pct"]    = pd.to_numeric(all_acc["actual_gain_pct"], errors="coerce")
    all_acc["predicted_probability"] = pd.to_numeric(all_acc["predicted_probability"], errors="coerce")

    # ── Daily aggregation client-side ──────────────────────────────────────
    def _daily_agg(gdf):
        pos_mask    = gdf["predicted_signal"].isin(["STRONG BUY", "BUY"])
        total       = len(gdf)
        correct     = int(gdf["prediction_correct"].sum())
        tp          = int((pos_mask & gdf["became_winner"]).sum())
        pred_pos    = int(pos_mask.sum())
        actual_wins = int(gdf["became_winner"].sum())
        precision   = tp / pred_pos * 100  if pred_pos    else 0.0
        recall      = tp / actual_wins * 100 if actual_wins else 0.0
        accuracy    = correct / total * 100  if total       else 0.0
        avg_gain    = gdf.loc[gdf["became_winner"] & gdf["actual_gain_pct"].notna(),
                               "actual_gain_pct"].mean()
        return pd.Series({
            "total":          total,
            "correct":        correct,
            "accuracy_pct":   accuracy,
            "precision_pct":  precision,
            "recall_pct":     recall,
            "true_positives": tp,
            "predicted_pos":  pred_pos,
            "actual_winners": actual_wins,
            "avg_winner_gain": avg_gain,
        })

    daily = (
        all_acc.groupby("prediction_date", group_keys=False)
        .apply(_daily_agg)
        .reset_index()
        .sort_values("prediction_date")
    )

    # Accuracy / precision / recall over time
    fig = go.Figure()
    for metric, color, name in [
        ("accuracy_pct",  "#667eea", "Accuracy"),
        ("precision_pct", "#10b981", "Precision"),
        ("recall_pct",    "#f59e0b", "Recall"),
    ]:
        fig.add_trace(go.Scatter(
            x=daily["prediction_date"], y=daily[metric],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2), marker=dict(size=6),
        ))
    fig.add_hline(y=50, line_dash="dash", line_color="#6b7280",
                  annotation_text="50% baseline")
    fig.update_layout(
        title="Accuracy / Precision / Recall Over Time",
        xaxis_title="Date", yaxis_title="%",
        height=400, hovermode="x unified", **_LAYOUT,
    )
    fig.update_xaxes(**_AXIS); fig.update_yaxes(**_AXIS)
    st.plotly_chart(fig, use_container_width=True)

    # TP / FP count bars
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily["prediction_date"], y=daily["true_positives"],
                             name="True Positives", marker_color="#10b981"))
        fig.add_trace(go.Bar(x=daily["prediction_date"],
                             y=daily["predicted_pos"] - daily["true_positives"],
                             name="False Positives", marker_color="#ef4444"))
        fig.update_layout(barmode="stack", title="Predicted Positives Breakdown",
                          height=350, **_LAYOUT)
        fig.update_xaxes(**_AXIS); fig.update_yaxes(**_AXIS)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Scatter(
            x=daily["prediction_date"], y=daily["avg_winner_gain"],
            mode="lines+markers", name="Avg Winner Gain",
            line=dict(color="#f59e0b", width=2), marker=dict(size=6),
        ))
        fig.update_layout(title="Average Actual Gain (Winners Only)",
                          xaxis_title="Date", yaxis_title="Gain %",
                          height=350, **_LAYOUT)
        fig.update_xaxes(**_AXIS); fig.update_yaxes(**_AXIS)
        st.plotly_chart(fig, use_container_width=True)

    # ── Per-signal performance (client-side) ──────────────────────────────
    st.markdown("#### Performance by Signal Type")
    pos_signals = all_acc[all_acc["predicted_signal"].isin(["STRONG BUY", "BUY", "HOLD", "AVOID"])]

    def _sig_agg(gdf):
        total    = len(gdf)
        hits     = int(gdf["became_winner"].sum())
        avg_prob = gdf["predicted_probability"].mean() * 100
        avg_gain = gdf.loc[gdf["became_winner"] & gdf["actual_gain_pct"].notna(),
                            "actual_gain_pct"].mean()
        avg_loss = gdf.loc[~gdf["became_winner"] & gdf["actual_gain_pct"].notna(),
                            "actual_gain_pct"].mean()
        return pd.Series({
            "total_predictions": total,
            "became_winner":     hits,
            "success_rate_pct":  hits / total * 100 if total else 0,
            "avg_probability":   avg_prob,
            "avg_gain_correct":  avg_gain,
            "avg_gain_wrong":    avg_loss,
        })

    sig_perf = (
        pos_signals.groupby("predicted_signal", group_keys=False)
        .apply(_sig_agg)
        .reset_index()
    )

    _sig_colors = {"STRONG BUY": "#10b981", "BUY": "#667eea", "HOLD": "#f59e0b", "AVOID": "#ef4444"}
    fig = go.Figure(go.Bar(
        x=sig_perf["predicted_signal"],
        y=sig_perf["success_rate_pct"],
        text=sig_perf["success_rate_pct"].map(lambda x: f"{x:.1f}%"),
        textposition="outside",
        marker_color=[_sig_colors.get(s, "#999") for s in sig_perf["predicted_signal"]],
    ))
    fig.update_layout(title="Success Rate by Signal Type", xaxis_title="Signal",
                      yaxis_title="Success Rate (%)", height=350, **_LAYOUT)
    fig.update_xaxes(**_AXIS); fig.update_yaxes(**_AXIS)
    st.plotly_chart(fig, use_container_width=True)

    sig_display = sig_perf.copy()
    st.dataframe(
        sig_display.style.format({
            "success_rate_pct": "{:.2f}%",
            "avg_probability":  "{:.2f}%",
            "avg_gain_correct": "+{:.2f}%",
            "avg_gain_wrong":   "{:.2f}%",
        }, na_rep="-"),
        use_container_width=True,
        hide_index=True,
    )

    # ── Summary table ──────────────────────────────────────────────────────
    st.markdown("#### Daily Summary Table")
    st.dataframe(
        daily.sort_values("prediction_date", ascending=False).style.format({
            "accuracy_pct":   "{:.1f}%",
            "precision_pct":  "{:.1f}%",
            "recall_pct":     "{:.1f}%",
            "avg_winner_gain": "{:.2f}%",
        }, na_rep="-"),
        use_container_width=True,
        hide_index=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SUB-TAB 5 — SYSTEM INFO
# ──────────────────────────────────────────────────────────────────────────────

def _render_system_info(tab_id: str, refresh_key: int):
    """
    Real schema for ml_screening_logs (from ml_track_comprehensive_accuracy.py context):
        screening_date, total_symbols_attempted, symbols_fetched_successfully,
        symbols_after_price_filter, symbols_after_volume_filter, total_predictions
    Falls back gracefully if columns differ.
    """
    st.markdown("### ℹ️ System Information")

    # Fetch only the most recent log row — minimal egress
    log_df = load_ml_data(
        tab_id, "ml_screening_logs",
        order_by=("screening_date", "desc"),
        limit=10,          # fetch 10 so we can show history too
        refresh_key=refresh_key,
    )

    if not log_df.empty:
        log = log_df.iloc[0]

        st.markdown("#### Latest Screening Run")

        # Build metrics only from columns that actually exist
        def _safe(col, default="—"):
            return log[col] if col in log.index and pd.notna(log[col]) else default

        screening_date = _safe("screening_date")
        attempted      = _safe("total_symbols_attempted")
        fetched        = _safe("symbols_fetched_successfully")
        predictions    = _safe("total_predictions")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Date",              str(screening_date))
        col2.metric("Stocks Attempted",  str(attempted))
        col3.metric("Predictions Made",  str(predictions))

        if attempted not in ("—", 0) and fetched not in ("—",):
            success_rate = int(fetched) / int(attempted) * 100
            col4.metric("Fetch Success Rate", f"{success_rate:.1f}%")
        else:
            col4.metric("Successfully Fetched", str(fetched))

        # Dynamic screening funnel — only show rows where column exists
        funnel_steps = [
            ("total_symbols_attempted",   "Total Attempted"),
            ("symbols_fetched_successfully", "Successfully Fetched"),
            ("symbols_after_price_filter",   "After Price Filter"),
            ("symbols_after_volume_filter",  "After Volume Filter"),
            ("total_predictions",            "Final Predictions"),
        ]
        available_steps = [(col, label) for col, label in funnel_steps if col in log.index and pd.notna(log[col])]

        if available_steps:
            st.markdown("#### Screening Funnel")
            funnel_df = pd.DataFrame(available_steps, columns=["Field", "Stage"])
            funnel_df["Value"] = funnel_df["Field"].map(lambda c: int(log[c]))
            funnel_df = funnel_df[["Stage", "Value"]]

            fig = go.Figure(go.Funnel(
                y=funnel_df["Stage"].tolist(),
                x=funnel_df["Value"].tolist(),
                textinfo="value+percent initial",
                marker=dict(color=["#667eea", "#10b981", "#f59e0b", "#3b82f6", "#8b5cf6"][:len(funnel_df)]),
            ))
            fig.update_layout(title="Screening Funnel", height=350, **_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        # Recent runs table
        if len(log_df) > 1:
            st.markdown("#### Recent Screening Runs")
            show_cols = [c for c in log_df.columns if c != "id"]
            st.dataframe(log_df[show_cols], use_container_width=True, hide_index=True)

        # All other columns not in funnel
        extra_cols = [c for c in log.index if c not in [col for col, _ in funnel_steps] + ["id"]]
        if extra_cols:
            with st.expander("All log fields"):
                st.json({c: str(log[c]) for c in extra_cols if pd.notna(log[c])})

    else:
        st.info("No screening logs found yet.")

    # ── Cross-table summary ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Database Summary")

    # Reuse already-cached data (zero new DB calls if sub-tabs were visited)
    preds_df   = load_ml_data(tab_id, "ml_explosion_predictions",
                               select_cols="prediction_date,symbol",
                               order_by=("prediction_date", "desc"),
                               limit=500, refresh_key=refresh_key)
    acc_df     = load_ml_data(tab_id, "ml_prediction_accuracy",
                               select_cols="prediction_date,became_winner,prediction_correct",
                               order_by=("prediction_date", "desc"),
                               limit=500, refresh_key=refresh_key)
    missed_df  = load_ml_data(tab_id, "ml_missed_opportunities",
                               select_cols="detection_date,symbol",
                               order_by=("detection_date", "desc"),
                               limit=500, refresh_key=refresh_key)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Prediction Records",      len(preds_df))
    col2.metric("Total Accuracy Records",        len(acc_df))
    col3.metric("Total Missed Opp. Records",     len(missed_df))

    if not acc_df.empty:
        acc_df["became_winner"]      = acc_df["became_winner"].astype(bool)
        acc_df["prediction_correct"] = acc_df["prediction_correct"].astype(bool)
        col1, col2 = st.columns(2)
        overall_acc = acc_df["prediction_correct"].mean() * 100
        overall_wr  = acc_df["became_winner"].mean() * 100
        col1.metric("Overall Accuracy (all dates)", f"{overall_acc:.1f}%")
        col2.metric("Overall Winner Rate",          f"{overall_wr:.1f}%")

    # ── Automated schedule ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📅 Automated Schedule (Estonia Time)")
    st.markdown("""
| Time | Job |
|------|-----|
| **3:00 PM** | Stock screening & prediction (500-1500 stocks) |
| **11:30 PM** | Daily winners collection |
| **5:30 AM +1** | Comprehensive accuracy tracking + yfinance gain fetch |
| **Sunday 9:00 AM** | Weekly model retraining (90-day rolling window) |
    """)

    st.markdown("---")
    st.markdown("#### 🛠️ Model Details")
    st.info(
        "**Model:** XGBoost Classifier  |  **Features:** 97 technical indicators  "
        "|  **Target:** 20%+ single-day gain  |  **Training window:** 90 days rolling  "
        "|  **actual_gain_pct source:** yfinance (all predicted symbols, not just winners)"
    )

    st.warning(
        "⚠️ **Disclaimer:** Experimental ML system for research only. "
        "Not financial advice. Past performance ≠ future results."
    )
