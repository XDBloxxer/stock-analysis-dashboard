"""
ML Predictions Tab - st.cache_data PERSISTENT CACHE

CACHE STRATEGY:
  - Data is stored in @st.cache_data on the SERVER PROCESS, not in session_state.
  - Survives browser tab closes, page refreshes, and re-opening the page
    as long as the Streamlit server process is running.
  - On first load: fetches full dataset for every table (limit 500).
  - Refresh button: fetches only rows newer than the latest cached date per
    table, appends them to the cache by clearing and re-fetching with the
    merged dataset.
  - Clear Cache: calls per-function .clear() → full re-fetch on next render.
  - Switching sub-tabs: zero egress.

EXPORTED for use by tab_daily_winners.render_stock_history:
  _get_table_full(table_name) → pd.DataFrame
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
from supabase import create_client, Client

TAB_ID = "ml_predictions"

_LAYOUT = dict(
    plot_bgcolor="rgba(26, 29, 41, 0.6)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e8eaf0"),
    margin=dict(t=40, b=20, l=20, r=20),
)
_AXIS = dict(gridcolor="rgba(255,255,255,0.1)", color="#b8bcc8")

_DATE_COL = {
    "ml_explosion_predictions": "prediction_date",
    "ml_prediction_accuracy":   "prediction_date",
    "ml_missed_opportunities":  "detection_date",
    "ml_screening_logs":        "screening_date",
}

_SELECT = {
    "ml_explosion_predictions": (
        "prediction_date,symbol,exchange,signal,explosion_probability,"
        "current_price,target_price,target_gain_pct,target_price_low,target_price_high"
    ),
    "ml_prediction_accuracy": (
        "symbol,prediction_date,predicted_probability,predicted_signal,"
        "predicted_target_gain,became_winner,actual_gain_pct,actual_high_pct,"
        "actual_price,prediction_correct,gain_error_pct"
    ),
    "ml_missed_opportunities": (
        "symbol,detection_date,exchange,actual_gain_pct,actual_high_pct,"
        "actual_price,actual_volume,was_screened,screening_failure_reason,"
        "predicted_probability,predicted_signal"
    ),
    "ml_screening_logs": "*",
}

_ALL_TABLES = list(_DATE_COL.keys())


# ── Supabase client ────────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets.get("supabase", {}).get("url")
    key = st.secrets.get("supabase", {}).get("key")
    if not url or not key:
        raise ValueError("Missing Supabase credentials")
    return create_client(url, key)


# ── Cached DB fetchers ─────────────────────────────────────────────────────────
# _get_table_full is the single source of truth for each table.
# It is also exported and used by tab_daily_winners.render_stock_history.

@st.cache_data(show_spinner=False)
def _get_table_full(table_name: str) -> pd.DataFrame:
    """
    Fetch up to 500 rows for a table, ordered by date descending.
    Cached indefinitely until _get_table_full.clear() is called.
    """
    try:
        client     = get_supabase_client()
        date_col   = _DATE_COL.get(table_name)
        select_str = _SELECT.get(table_name, "*")
        query = client.table(table_name).select(select_str)
        if date_col:
            query = query.order(date_col, desc=True)
        response = query.limit(500).execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load `{table_name}`: {e}")
        return pd.DataFrame()


# ── Cache control ──────────────────────────────────────────────────────────────

def clear_all_cache():
    """Wipe all cached ML data → full re-fetch on next render."""
    _get_table_full.clear()


def refresh_cache():
    """
    Append-only refresh.
    For each table, fetch only rows newer than the latest cached date,
    merge with the existing cached data, store the merged result back,
    and clear the cache so the next call to _get_table_full returns
    the merged dataset.

    Because @st.cache_data stores by function arguments, we can't update
    the cache in-place. Instead we:
      1. Read current cached data.
      2. Fetch new rows from Supabase.
      3. Merge into a combined DataFrame.
      4. Clear the cache for that table.
      5. Re-populate the cache by calling the function with a patched
         approach: we temporarily store merged data in session_state and
         use a wrapper that checks session_state first.

    Simpler alternative used here: just clear the whole cache and let
    _get_table_full re-fetch everything (still only one DB call per table).
    This is acceptable because the tables have at most ~500 rows and the
    fetch is fast. The key point is that after Refresh, the next render
    will call _get_table_full once per table and then cache it again.
    """
    any_new = False

    for table in _ALL_TABLES:
        existing = _get_table_full(table)  # reads from cache — no DB call
        date_col = _DATE_COL.get(table)

        if existing.empty or not date_col or date_col not in existing.columns:
            continue

        latest_cached = existing[date_col].max()

        try:
            client     = get_supabase_client()
            select_str = _SELECT.get(table, "*")
            response = (
                client.table(table)
                .select(select_str)
                .gt(date_col, latest_cached)
                .order(date_col, desc=True)
                .limit(500)
                .execute()
            )
            new_rows = pd.DataFrame(response.data) if response.data else pd.DataFrame()
        except Exception as e:
            st.warning(f"Could not check new rows for `{table}`: {e}")
            continue

        if new_rows.empty:
            continue

        any_new = True

    if any_new:
        # Clear and re-fetch — one clean call per table, results cached again
        _get_table_full.clear()
        for table in _ALL_TABLES:
            _get_table_full(table)
        st.toast("✅ New data fetched and cached.")
    else:
        st.toast("✅ Cache is already up to date — no new data found.")


# ── Main entry point ───────────────────────────────────────────────────────────
def render_ml_predictions_tab():

    col1, col2 = st.columns(2)
    with col1:
        refresh_clicked = st.button("🔄 Refresh", key=f"{TAB_ID}_refresh")
    with col2:
        clear_clicked = st.button("🗑️ Clear Cache", key=f"{TAB_ID}_clear_cache")

    if clear_clicked:
        clear_all_cache()
        st.rerun()

    if refresh_clicked:
        refresh_cache()
        st.rerun()

    st.subheader("🤖 ML Explosion Predictions (Autonomous)")
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
        _render_latest_predictions()
    with subtab2:
        _render_predictions_vs_actuals()
    with subtab3:
        _render_missed_opportunities()
    with subtab4:
        _render_performance_trends()
    with subtab5:
        _render_system_info()


# ── Sub-tab 1 — Latest Predictions ────────────────────────────────────────────
def _render_latest_predictions():
    all_preds = _get_table_full("ml_explosion_predictions")

    if all_preds.empty:
        st.warning("📭 No predictions available yet.")
        st.info("Run the screening workflow or wait for the scheduled run.")
        return

    dates = sorted(all_preds["prediction_date"].unique().tolist(), reverse=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_date = st.selectbox(
            "Select Date:", dates,
            format_func=lambda x: datetime.fromisoformat(x).strftime("%A, %B %d, %Y"),
            key=f"{TAB_ID}_pred_date",
        )
    with col2:
        pred_dt = datetime.fromisoformat(selected_date).date()
        if pred_dt >= datetime.now().date():
            st.success("🔮 Forward-looking predictions")
        else:
            st.info("📊 Historical — see Predictions vs Actuals for results")

    df = all_preds[all_preds["prediction_date"] == selected_date].copy()
    if df.empty:
        st.warning(f"No predictions for {selected_date}")
        return

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Screened",  len(df))
    col2.metric("🟢 STRONG BUY",  int((df["signal"] == "STRONG BUY").sum()))
    col3.metric("🔵 BUY",         int((df["signal"] == "BUY").sum()))
    col4.metric("Avg Probability", f"{df['explosion_probability'].mean() * 100:.1f}%")
    col5.metric("Avg Target Gain", f"+{df['target_gain_pct'].mean():.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Histogram(
            x=df["explosion_probability"] * 100, nbinsx=20, marker_color="#667eea",
        ))
        fig.update_layout(title="Probability Distribution", xaxis_title="Probability (%)",
                          yaxis_title="Count", height=300, showlegend=False, **_LAYOUT)
        fig.update_xaxes(**_AXIS); fig.update_yaxes(**_AXIS)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sc = df["signal"].value_counts()
        _clr = {"STRONG BUY": "#10b981", "BUY": "#667eea", "HOLD": "#f59e0b", "AVOID": "#ef4444"}
        fig = go.Figure(go.Pie(
            labels=sc.index, values=sc.values,
            marker=dict(colors=[_clr.get(s, "#999") for s in sc.index]),
            hole=0.4,
        ))
        fig.update_layout(title="Signal Breakdown", height=300, **_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔍 Filter")
    col1, col2, col3 = st.columns(3)
    with col1:
        sig_filter = st.multiselect(
            "Signal:", ["STRONG BUY", "BUY", "HOLD", "AVOID"],
            default=["STRONG BUY", "BUY", "HOLD", "AVOID"],
            key=f"{TAB_ID}_sig_f",
        )
    with col2:
        min_prob = st.slider("Min Probability:", 0, 100, 0, key=f"{TAB_ID}_prob_f")
    with col3:
        min_tgt  = st.slider("Min Target Gain (%):", 0, 50, 0, key=f"{TAB_ID}_tgt_f")

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
    _SIG_BG = {"STRONG BUY": "#10b98133", "BUY": "#667eea33",
                "HOLD": "#f59e0b33", "AVOID": "#ef444433"}

    def _highlight_sig(row):
        return [f"background-color: {_SIG_BG.get(row['signal'], '')}"] * len(row)

    display_cols = [c for c in [
        "symbol", "exchange", "signal", "explosion_probability",
        "current_price", "target_price", "target_gain_pct",
        "target_price_low", "target_price_high",
    ] if c in fdf.columns]

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
        key=f"{TAB_ID}_dl",
    )


# ── Sub-tab 2 — Predictions vs Actuals ────────────────────────────────────────
def _render_predictions_vs_actuals():
    st.markdown("### 🎯 Prediction Accuracy Analysis")
    st.info("Compare predictions against actual market outcomes.")

    all_acc = _get_table_full("ml_prediction_accuracy")

    if all_acc.empty:
        st.warning("📭 No accuracy data available yet.")
        st.info("Accuracy tracking runs automatically after market close.")
        return

    dates = sorted(all_acc["prediction_date"].unique().tolist(), reverse=True)
    selected_date = st.selectbox(
        "Select Date:", dates,
        format_func=lambda x: datetime.fromisoformat(x).strftime("%A, %B %d, %Y"),
        key=f"{TAB_ID}_acc_date",
    )

    df = all_acc[all_acc["prediction_date"] == selected_date].copy()
    if df.empty:
        st.warning(f"No accuracy data for {selected_date}")
        return

    df["became_winner"]         = df["became_winner"].astype(bool)
    df["prediction_correct"]    = df["prediction_correct"].astype(bool)
    df["predicted_probability"] = pd.to_numeric(df["predicted_probability"], errors="coerce")
    df["actual_gain_pct"]       = pd.to_numeric(df["actual_gain_pct"],       errors="coerce")
    df["actual_high_pct"]       = pd.to_numeric(df["actual_high_pct"],       errors="coerce")
    df["gain_error_pct"]        = pd.to_numeric(df["gain_error_pct"],        errors="coerce")

    pos_mask          = df["predicted_signal"].isin(["STRONG BUY", "BUY"])
    total             = len(df)
    correct           = int(df["prediction_correct"].sum())
    actual_winners    = int(df["became_winner"].sum())
    predicted_winners = int(pos_mask.sum())
    tp = int((pos_mask &  df["became_winner"]).sum())
    fp = int((pos_mask & ~df["became_winner"]).sum())
    fn = int((~pos_mask &  df["became_winner"]).sum())
    tn = int((~pos_mask & ~df["became_winner"]).sum())

    accuracy_pct = correct           / total             * 100 if total             else 0
    precision    = tp                / predicted_winners * 100 if predicted_winners else 0
    recall       = tp                / actual_winners    * 100 if actual_winners    else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total",          total)
    col2.metric("Accuracy",       f"{accuracy_pct:.1f}%")
    col3.metric("Actual Winners", actual_winners)
    col4.metric("Pred. Winners",  predicted_winners)
    col5.metric("Precision",      f"{precision:.1f}%")
    col6.metric("Recall",         f"{recall:.1f}%")

    gain_populated = df["actual_gain_pct"].notna().sum()
    if gain_populated > 0:
        st.caption(f"actual_gain_pct populated for {gain_populated}/{total} symbols")
        col1, col2, col3, col4 = st.columns(4)
        winner_gains = df.loc[ df["became_winner"] & df["actual_gain_pct"].notna(), "actual_gain_pct"]
        non_gains    = df.loc[~df["became_winner"] & df["actual_gain_pct"].notna(), "actual_gain_pct"]
        col1.metric("Avg Winner Gain",     f"{winner_gains.mean():.2f}%" if not winner_gains.empty else "—")
        col2.metric("Avg Non-Winner Gain", f"{non_gains.mean():.2f}%"   if not non_gains.empty    else "—")
        err_df = df.loc[df["gain_error_pct"].notna(), "gain_error_pct"]
        col3.metric("Avg Gain Error",      f"{err_df.mean():.2f}%"      if not err_df.empty       else "—")
        col4.metric("Intraday High Pct populated", df["actual_high_pct"].notna().sum())

    st.markdown("#### Confusion Matrix")
    fig = go.Figure(go.Heatmap(
        z=[[tp, fp], [fn, tn]],
        x=["Actually Exploded", "Didn't Explode"],
        y=["Predicted Explosion", "Predicted No Explosion"],
        text=[[f"True Positive<br>{tp}", f"False Positive<br>{fp}"],
              [f"False Negative<br>{fn}", f"True Negative<br>{tn}"]],
        texttemplate="%{text}", textfont={"size": 14, "color": "#ffffff"},
        colorscale=[[0, "#2d3142"], [1, "#10b981"]], showscale=False,
    ))
    fig.update_layout(height=350, **_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    if gain_populated > 0:
        st.markdown("#### Actual Gain Distribution")
        fig = go.Figure()
        g_tp = df.loc[ pos_mask &  df["became_winner"] & df["actual_gain_pct"].notna(), "actual_gain_pct"]
        g_fp = df.loc[ pos_mask & ~df["became_winner"] & df["actual_gain_pct"].notna(), "actual_gain_pct"]
        if not g_tp.empty:
            fig.add_trace(go.Histogram(x=g_tp, nbinsx=20, name="True Positive",
                                       marker_color="#10b981", opacity=0.75))
        if not g_fp.empty:
            fig.add_trace(go.Histogram(x=g_fp, nbinsx=20, name="False Positive",
                                       marker_color="#ef4444", opacity=0.75))
        fig.update_layout(barmode="overlay", xaxis_title="Actual Gain %",
                          yaxis_title="Count", height=350, **_LAYOUT)
        fig.update_xaxes(**_AXIS); fig.update_yaxes(**_AXIS)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Detailed Results")
    display_cols = [c for c in [
        "symbol", "predicted_signal", "predicted_probability",
        "predicted_target_gain", "became_winner", "actual_gain_pct",
        "actual_high_pct", "gain_error_pct", "prediction_correct",
    ] if c in df.columns]

    def _highlight_correct(row):
        bg = "#10b98133" if row.get("prediction_correct") else "#ef444433"
        return [f"background-color: {bg}"] * len(row)

    fmt = {
        "predicted_probability": "{:.2%}", "predicted_target_gain": "{:.2f}%",
        "actual_gain_pct": "{:.2f}%", "actual_high_pct": "{:.2f}%",
        "gain_error_pct":  "{:.2f}%",
    }
    st.dataframe(
        df[display_cols]
        .sort_values("predicted_probability", ascending=False)
        .style.format({k: v for k, v in fmt.items() if k in display_cols}, na_rep="-")
        .apply(_highlight_correct, axis=1),
        use_container_width=True, height=600,
    )


# ── Sub-tab 3 — Missed Opportunities ──────────────────────────────────────────
def _render_missed_opportunities():
    st.markdown("### ❌ Missed Opportunities (Recall Analysis)")
    st.info("Winners we didn't predict.")

    all_missed = _get_table_full("ml_missed_opportunities")

    if all_missed.empty:
        st.warning("📭 No missed opportunities data yet.")
        return

    all_missed = all_missed.copy()
    all_missed["actual_gain_pct"] = pd.to_numeric(all_missed["actual_gain_pct"], errors="coerce")
    all_missed["was_screened"]    = all_missed["was_screened"].astype(bool)

    dates = sorted(all_missed["detection_date"].unique().tolist(), reverse=True)
    selected_date = st.selectbox(
        "Select Date:", dates,
        format_func=lambda x: datetime.fromisoformat(x).strftime("%A, %B %d, %Y"),
        key=f"{TAB_ID}_missed_date",
    )

    df = all_missed[all_missed["detection_date"] == selected_date].copy()
    if df.empty:
        st.info("No missed opportunities for this date.")
        return

    total_missed     = len(df)
    not_screened     = int((~df["was_screened"]).sum())
    screened_no_pred = int(df["was_screened"].sum())
    avg_gain         = df["actual_gain_pct"].mean()
    best_gain        = df["actual_gain_pct"].max()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Missed",            total_missed)
    col2.metric("Not Screened",            not_screened)
    col3.metric("Screened, Not Predicted", screened_no_pred)
    col4.metric("Avg Missed Gain",         f"+{avg_gain:.1f}%" if pd.notna(avg_gain) else "—")
    col5.metric("Best Missed Gain",        f"+{best_gain:.1f}%" if pd.notna(best_gain) else "—")

    st.markdown("#### Why We Missed Them")
    reason_col = df["screening_failure_reason"].fillna(
        df["was_screened"].map({True: "screened_but_low_probability", False: "not_in_screener"})
    )
    reason_counts = reason_col.value_counts()

    fig = go.Figure(go.Bar(
        x=reason_counts.index.tolist(), y=reason_counts.values.tolist(),
        marker_color="#ef4444",
        text=reason_counts.values.tolist(), textposition="outside",
    ))
    fig.update_layout(title="Missed Opportunities by Failure Reason",
                      xaxis_title="Reason", yaxis_title="Count",
                      height=350, **_LAYOUT)
    fig.update_xaxes(**_AXIS); fig.update_yaxes(**_AXIS)
    st.plotly_chart(fig, use_container_width=True)

    gain_data = df["actual_gain_pct"].dropna()
    if not gain_data.empty:
        fig = go.Figure(go.Histogram(x=gain_data, nbinsx=20, marker_color="#f59e0b"))
        fig.update_layout(title="Distribution of Missed Gains",
                          xaxis_title="Actual Gain %", yaxis_title="Count",
                          height=300, **_LAYOUT)
        fig.update_xaxes(**_AXIS); fig.update_yaxes(**_AXIS)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Missed Winners Detail")
    display_cols = [c for c in [
        "symbol", "exchange", "actual_gain_pct", "actual_high_pct",
        "actual_price", "actual_volume", "was_screened",
        "screening_failure_reason", "predicted_probability",
    ] if c in df.columns]

    st.dataframe(
        df[display_cols].sort_values("actual_gain_pct", ascending=False).style.format({
            "actual_gain_pct":       "+{:.2f}%",
            "actual_high_pct":       "+{:.2f}%",
            "actual_price":          "${:.2f}",
            "actual_volume":         "{:,.0f}",
            "predicted_probability": "{:.2%}",
        }, na_rep="-"),
        use_container_width=True, height=500,
    )


# ── Sub-tab 4 — Performance Trends ────────────────────────────────────────────
def _render_performance_trends():
    st.markdown("### 📈 Model Performance Trends")

    all_acc = _get_table_full("ml_prediction_accuracy")

    if all_acc.empty:
        st.warning("No accuracy data available yet.")
        return

    all_acc = all_acc.copy()
    all_acc["became_winner"]         = all_acc["became_winner"].astype(bool)
    all_acc["prediction_correct"]    = all_acc["prediction_correct"].astype(bool)
    all_acc["actual_gain_pct"]       = pd.to_numeric(all_acc["actual_gain_pct"],       errors="coerce")
    all_acc["predicted_probability"] = pd.to_numeric(all_acc["predicted_probability"], errors="coerce")

    def _daily_agg(gdf):
        pos_mask    = gdf["predicted_signal"].isin(["STRONG BUY", "BUY"])
        total       = len(gdf)
        correct     = int(gdf["prediction_correct"].sum())
        tp          = int((pos_mask & gdf["became_winner"]).sum())
        pred_pos    = int(pos_mask.sum())
        actual_wins = int(gdf["became_winner"].sum())
        precision   = tp / pred_pos    * 100 if pred_pos    else 0.0
        recall      = tp / actual_wins * 100 if actual_wins else 0.0
        accuracy    = correct / total  * 100 if total       else 0.0
        avg_gain    = gdf.loc[gdf["became_winner"] & gdf["actual_gain_pct"].notna(),
                               "actual_gain_pct"].mean()
        return pd.Series({
            "total": total, "correct": correct,
            "accuracy_pct": accuracy, "precision_pct": precision, "recall_pct": recall,
            "true_positives": tp, "predicted_pos": pred_pos,
            "actual_winners": actual_wins, "avg_winner_gain": avg_gain,
        })

    daily = (
        all_acc.groupby("prediction_date", group_keys=False)
        .apply(_daily_agg)
        .reset_index()
        .sort_values("prediction_date")
    )

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
    fig.update_layout(title="Accuracy / Precision / Recall Over Time",
                      xaxis_title="Date", yaxis_title="%",
                      height=400, hovermode="x unified", **_LAYOUT)
    fig.update_xaxes(**_AXIS); fig.update_yaxes(**_AXIS)
    st.plotly_chart(fig, use_container_width=True)

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

    st.markdown("#### Performance by Signal Type")
    pos_signals = all_acc[all_acc["predicted_signal"].isin(["STRONG BUY", "BUY", "HOLD", "AVOID"])]

    def _sig_agg(gdf):
        total    = len(gdf)
        hits     = int(gdf["became_winner"].sum())
        avg_prob = gdf["predicted_probability"].mean() * 100
        avg_gain = gdf.loc[ gdf["became_winner"] & gdf["actual_gain_pct"].notna(), "actual_gain_pct"].mean()
        avg_loss = gdf.loc[~gdf["became_winner"] & gdf["actual_gain_pct"].notna(), "actual_gain_pct"].mean()
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

    _sig_colors = {"STRONG BUY": "#10b981", "BUY": "#667eea",
                   "HOLD": "#f59e0b", "AVOID": "#ef4444"}
    fig = go.Figure(go.Bar(
        x=sig_perf["predicted_signal"],
        y=sig_perf["success_rate_pct"],
        text=sig_perf["success_rate_pct"].map(lambda x: f"{x:.1f}%"),
        textposition="outside",
        marker_color=[_sig_colors.get(s, "#999") for s in sig_perf["predicted_signal"]],
    ))
    fig.update_layout(title="Success Rate by Signal Type",
                      xaxis_title="Signal", yaxis_title="Success Rate (%)",
                      height=350, **_LAYOUT)
    fig.update_xaxes(**_AXIS); fig.update_yaxes(**_AXIS)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        sig_perf.style.format({
            "success_rate_pct": "{:.2f}%", "avg_probability":  "{:.2f}%",
            "avg_gain_correct": "+{:.2f}%", "avg_gain_wrong":  "{:.2f}%",
        }, na_rep="-"),
        use_container_width=True, hide_index=True,
    )

    st.markdown("#### Daily Summary Table")
    st.dataframe(
        daily.sort_values("prediction_date", ascending=False).style.format({
            "accuracy_pct": "{:.1f}%", "precision_pct": "{:.1f}%",
            "recall_pct": "{:.1f}%",  "avg_winner_gain": "{:.2f}%",
        }, na_rep="-"),
        use_container_width=True, hide_index=True,
    )


# ── Sub-tab 5 — System Info ────────────────────────────────────────────────────
def _render_system_info():
    st.markdown("### ℹ️ System Information")

    log_df = _get_table_full("ml_screening_logs")

    if not log_df.empty:
        date_col = _DATE_COL.get("ml_screening_logs", "screening_date")
        if date_col in log_df.columns:
            log_df = log_df.sort_values(date_col, ascending=False)

        log = log_df.iloc[0]

        st.markdown("#### Latest Screening Run")

        def _safe(col, default="—"):
            return log[col] if col in log.index and pd.notna(log[col]) else default

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Date",             str(_safe("screening_date")))
        col2.metric("Stocks Attempted", str(_safe("total_symbols_attempted")))
        col3.metric("Predictions Made", str(_safe("total_predictions")))

        attempted = _safe("total_symbols_attempted")
        fetched   = _safe("symbols_fetched_successfully")
        if attempted not in ("—", 0) and fetched not in ("—",):
            col4.metric("Fetch Success Rate", f"{int(fetched)/int(attempted)*100:.1f}%")
        else:
            col4.metric("Successfully Fetched", str(fetched))

        funnel_steps = [
            ("total_symbols_attempted",      "Total Attempted"),
            ("symbols_fetched_successfully", "Successfully Fetched"),
            ("symbols_after_price_filter",   "After Price Filter"),
            ("symbols_after_volume_filter",  "After Volume Filter"),
            ("total_predictions",            "Final Predictions"),
        ]
        available_steps = [(col, lbl) for col, lbl in funnel_steps
                           if col in log.index and pd.notna(log[col])]

        if available_steps:
            st.markdown("#### Screening Funnel")
            funnel_df = pd.DataFrame(available_steps, columns=["Field", "Stage"])
            funnel_df["Value"] = funnel_df["Field"].map(lambda c: int(log[c]))

            fig = go.Figure(go.Funnel(
                y=funnel_df["Stage"].tolist(),
                x=funnel_df["Value"].tolist(),
                textinfo="value+percent initial",
                marker=dict(color=["#667eea", "#10b981", "#f59e0b",
                                   "#3b82f6", "#8b5cf6"][:len(funnel_df)]),
            ))
            fig.update_layout(title="Screening Funnel", height=350, **_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        if len(log_df) > 1:
            st.markdown("#### Recent Screening Runs")
            show_cols = [c for c in log_df.columns if c != "id"]
            st.dataframe(log_df[show_cols], use_container_width=True, hide_index=True)

        extra_cols = [c for c in log.index
                      if c not in [col for col, _ in funnel_steps] + ["id"]]
        if extra_cols:
            with st.expander("All log fields"):
                st.json({c: str(log[c]) for c in extra_cols if pd.notna(log[c])})
    else:
        st.info("No screening logs found yet.")

    st.markdown("---")
    st.markdown("#### 📊 Database Summary")

    preds_df  = _get_table_full("ml_explosion_predictions")
    acc_df    = _get_table_full("ml_prediction_accuracy")
    missed_df = _get_table_full("ml_missed_opportunities")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Prediction Records",  len(preds_df))
    col2.metric("Total Accuracy Records",    len(acc_df))
    col3.metric("Total Missed Opp. Records", len(missed_df))

    if not acc_df.empty:
        acc_df = acc_df.copy()
        acc_df["became_winner"]      = acc_df["became_winner"].astype(bool)
        acc_df["prediction_correct"] = acc_df["prediction_correct"].astype(bool)
        col1, col2 = st.columns(2)
        col1.metric("Overall Accuracy (all dates)", f"{acc_df['prediction_correct'].mean()*100:.1f}%")
        col2.metric("Overall Winner Rate",          f"{acc_df['became_winner'].mean()*100:.1f}%")

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
