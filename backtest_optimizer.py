"""
backtest_optimizer.py — Parameter optimizer for the Cumulative Gain Simulator.

WHY THIS EXISTS
    The expensive part of a backtest here is fetching 5-minute bars from
    yfinance (one call per symbol). The simulation itself is cheap. So this
    module fetches bars ONCE, keeps every trade's full intraday price path
    in memory, and then sweeps hundreds/thousands of parameter combinations
    (stop type, stop %, target, max positions, sizing, deployment) without
    touching the network again.

WHAT IT SEARCHES
    - Stop type:   none / fixed % / trailing %
    - Stop %:      a range you choose
    - Target:      none / the model's predicted_target_gain / fixed %
    - Max positions per day
    - Sizing:      equal split vs. confidence-weighted
    - Capital deployed per day

HOW IT GUARDS AGAINST OVERFITTING (the real danger with ≤60 days of data)
    1. Chronological train/test split: combos are ranked on the first part of
       the period only, then re-scored on the untouched later days.
    2. Plateau scoring: a combo's score is averaged with its immediate
       neighbours (stop% ±1 step, max-positions ±1 step). A lone spike
       surrounded by bad settings is almost certainly noise; a broad
       plateau is much more likely to be real.
    3. Drawdown cap: combos that breach your max-drawdown limit are excluded.
    4. The number of combos tested is shown — the more you test, the more
       likely the "best" one is just the luckiest.

LAYOUT
    Pure engine (no Streamlit): TradePaths, build_trade_paths, resolve_gains,
    run_portfolio, run_grid.  UI: render_optimizer_panel (imports helpers
    from tab_backtesting lazily to avoid a circular import).
"""

from __future__ import annotations

import datetime as _dt
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

OUTCOME_CLOSE, OUTCOME_STOP, OUTCOME_TARGET = 0, 1, 2

OBJECTIVES = {
    "Total return":                  "total_return",
    "Sharpe-like ratio":             "sharpe_like",
    "Return ÷ max drawdown":         "return_over_dd",
    "Profit factor":                 "profit_factor",
}


@dataclass
class TradePaths:
    """Every resolvable trade's intraday path, as %-moves vs. its entry price,
    padded into rectangular arrays so a whole parameter setting can be
    resolved across ALL trades in a handful of numpy operations.

    Padding is chosen so padded cells can never trigger anything:
    lo → +inf (never <= a stop), hi → -inf (never >= a target)."""
    meta: pd.DataFrame        # symbol, prediction_date, predicted_probability, predicted_signal, target_pct
    lo: np.ndarray            # (n, B) bar Low  as % vs entry
    hi: np.ndarray            # (n, B) bar High as % vs entry
    op: np.ndarray            # (n, B) bar Open as % vs entry (NaN in padding)
    final_close: np.ndarray   # (n,)  last close of the session as % vs entry
    drop_reasons: dict
    n_eligible: int

    @property
    def n(self) -> int:
        return len(self.meta)


def build_trade_paths(eligible: pd.DataFrame, bars_by_symbol: dict) -> TradePaths:
    """Mirror of tab_backtesting._resolve_trade_from_bars's data handling:
    entry = last close strictly before 4:00 AM ET on the trade date; the walk
    covers 4:00 AM → 4:00 PM ET inclusive. Trades that can't be priced are
    dropped (never guessed), with reasons tallied."""
    rows, los, his, ops, finals = [], [], [], [], []
    drops: dict[str, int] = {}

    def _drop(reason):
        drops[reason] = drops.get(reason, 0) + 1

    for r in eligible.itertuples():
        bars = bars_by_symbol.get(r.symbol)
        if bars is None or bars.empty or "Close" not in bars.columns:
            _drop("no_bars_fetched")
            continue
        tz = bars.index.tz
        d = r.prediction_date.date()
        t_open = pd.Timestamp.combine(d, _dt.time(4, 0)).tz_localize(tz)
        t_close = pd.Timestamp.combine(d, _dt.time(16, 0)).tz_localize(tz)

        i_open = bars.index.searchsorted(t_open, side="left")
        if i_open < 1:
            _drop("no_prior_bar_for_entry_price")
            continue
        entry = float(bars["Close"].iloc[i_open - 1])
        if not entry or entry <= 0 or np.isnan(entry):
            _drop("invalid_entry_price")
            continue
        i_end = bars.index.searchsorted(t_close, side="right")
        sess = bars.iloc[i_open:i_end]
        if sess.empty or not {"High", "Low", "Open"}.issubset(sess.columns):
            _drop("no_session_bars_for_trade_date")
            continue
        closes = sess["Close"].dropna()
        if closes.empty:
            _drop("no_session_bars_for_trade_date")
            continue

        lo = (sess["Low"].to_numpy(float) / entry - 1.0) * 100.0
        hi = (sess["High"].to_numpy(float) / entry - 1.0) * 100.0
        op = (sess["Open"].to_numpy(float) / entry - 1.0) * 100.0
        final = (float(closes.iloc[-1]) / entry - 1.0) * 100.0

        tgt = getattr(r, "predicted_target_gain", np.nan)
        tgt = float(tgt) if tgt is not None and not pd.isna(tgt) else np.nan
        rows.append({
            "symbol": r.symbol,
            "prediction_date": r.prediction_date,
            "predicted_probability": getattr(r, "predicted_probability", np.nan),
            "predicted_signal": getattr(r, "predicted_signal", None),
            "target_pct": tgt,
        })
        los.append(lo); his.append(hi); ops.append(op); finals.append(final)

    if not rows:
        empty = np.empty((0, 1))
        return TradePaths(pd.DataFrame(columns=["symbol", "prediction_date", "predicted_probability",
                                                "predicted_signal", "target_pct"]),
                          empty, empty, empty, np.empty(0), drops, len(eligible))

    n, B = len(rows), max(len(a) for a in los)
    LO = np.full((n, B), np.inf)
    HI = np.full((n, B), -np.inf)
    OP = np.full((n, B), np.nan)
    for i in range(n):
        k = len(los[i])
        # NaN lows/highs (data glitches) must never trigger: NaN compares False.
        LO[i, :k] = los[i]; HI[i, :k] = his[i]; OP[i, :k] = ops[i]
    meta = pd.DataFrame(rows).reset_index(drop=True)
    return TradePaths(meta, LO, HI, OP, np.array(finals), drops, len(eligible))


def _first_true(mask: np.ndarray) -> np.ndarray:
    """Index of the first True per row, or B (= 'never') if none."""
    B = mask.shape[1]
    return np.where(mask.any(axis=1), mask.argmax(axis=1), B)


def resolve_gains(paths: TradePaths, stop_type: str, stop_pct: float | None,
                  target_mode: str, target_value: float | None,
                  gap_fills: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Resolve every trade under one (stop, target) setting. Returns
    (gain_pct[n], outcome[n]) where outcome is OUTCOME_CLOSE/STOP/TARGET.

    Rules (same as the live 5-min resolver, plus trailing + gap realism):
      - Bars are walked in order. Within one bar the STOP is checked before
        the TARGET (conservative).
      - fixed stop:    triggers when a bar's Low <= -stop%.
      - trailing stop: stop level = (highest High of PRIOR bars) × (1 - stop%),
        starting from the entry price. Using prior bars' highs only avoids
        the look-ahead of assuming we knew a bar's high before its low.
      - gap_fills=True: if a bar OPENS beyond the stop, you're filled at the
        open (worse than the stop); if it opens beyond a target you're filled
        at the open (better). gap_fills=False fills at the exact level, which
        matches the original single-run simulator.
      - Neither touched → exit at the session's final close."""
    n, B = paths.lo.shape
    gain = paths.final_close.copy()
    outcome = np.zeros(n, dtype=np.int8)
    if n == 0:
        return gain, outcome

    # ── stop ──
    stop_idx = np.full(n, B)
    level = None
    if stop_type != "none" and stop_pct is not None:
        s = abs(float(stop_pct))
        if stop_type == "fixed":
            level = np.full((n, B), -s)
        elif stop_type == "trailing":
            peak = np.maximum.accumulate(np.maximum(paths.hi, 0.0), axis=1)
            peak_prev = np.concatenate([np.zeros((n, 1)), peak[:, :-1]], axis=1)
            level = (1.0 + peak_prev / 100.0) * (1.0 - s / 100.0) * 100.0 - 100.0
        else:
            raise ValueError(f"unknown stop_type {stop_type!r}")
        stop_idx = _first_true(paths.lo <= level)

    # ── target ──
    if target_mode == "model":
        tgt = paths.meta["target_pct"].to_numpy(float)
    elif target_mode == "fixed" and target_value is not None:
        tgt = np.full(n, float(target_value))
    else:
        tgt = np.full(n, np.nan)
    tgt_idx = _first_true(paths.hi >= tgt[:, None])   # NaN target → never

    rows = np.arange(n)
    stop_first = (stop_idx < B) & (stop_idx <= tgt_idx)
    tgt_first = (tgt_idx < B) & (tgt_idx < stop_idx)

    if stop_first.any():
        r, c = rows[stop_first], stop_idx[stop_first]
        lv = level[r, c]
        gain[r] = np.fmin(lv, paths.op[r, c]) if gap_fills else lv
        outcome[r] = OUTCOME_STOP
    if tgt_first.any():
        r, c = rows[tgt_first], tgt_idx[tgt_first]
        tv = tgt[r]
        gain[r] = np.fmax(tv, paths.op[r, c]) if gap_fills else tv
        outcome[r] = OUTCOME_TARGET
    return gain, outcome


@dataclass
class DayGroups:
    dates: np.ndarray        # sorted unique trade dates (datetime64[D])
    order: list              # per-day trade indices, best model confidence first
    probs: np.ndarray        # (n,) predicted_probability, NaN → 0


def build_day_groups(paths: TradePaths) -> DayGroups:
    meta = paths.meta
    probs = pd.to_numeric(meta["predicted_probability"], errors="coerce").to_numpy(float)
    dates = meta["prediction_date"].dt.normalize().to_numpy("datetime64[D]")
    uniq = np.unique(dates)
    order = []
    for d in uniq:
        idx = np.where(dates == d)[0]
        key = np.where(np.isnan(probs[idx]), -np.inf, probs[idx])
        order.append(idx[np.argsort(-key, kind="stable")])
    return DayGroups(uniq, order, np.nan_to_num(probs, nan=0.0))


def run_portfolio(gains: np.ndarray, groups: DayGroups, day_lo: int, day_hi: int,
                  start_capital: float, commission: float, slippage_bps: float,
                  max_positions: int, weight_conf: bool, deploy_frac: float,
                  want_equity: bool = False) -> dict:
    """Same accounting as tab_backtesting._simulate, over days [day_lo, day_hi).
    Per day: take the top-`max_positions` signals by confidence, split the
    deployed share of capital (equally or by confidence), charge commission
    and slippage per trade, pool capital back at end of day. Idle cash is
    carried at 0% return."""
    deploy = max(0.0, min(1.0, deploy_frac))
    cap = float(start_capital)
    equity, eq_dates = [], []
    trade_gain, trade_pnl = [], []
    for di in range(day_lo, day_hi):
        idx = groups.order[di][:max_positions]
        n = len(idx)
        if n == 0:
            continue
        if weight_conf:
            p = groups.probs[idx]
            tot = p.sum()
            w = p / tot if tot > 0 else np.full(n, 1.0 / n)
        else:
            w = np.full(n, 1.0 / n)
        pos = cap * w * deploy
        g = gains[idx]
        pnl = pos * (g / 100.0) - commission - pos * (slippage_bps / 1e4)
        resolved = np.maximum(pos + pnl, 0.0)
        cap = cap * (1.0 - deploy) + resolved.sum()
        equity.append(cap)
        eq_dates.append(groups.dates[di])
        trade_gain.append(g); trade_pnl.append(pnl)

    if not equity:
        return {"n_days": 0, "n_trades": 0, "total_return": 0.0, "sharpe_like": np.nan,
                "max_drawdown": 0.0, "win_rate": 0.0, "profit_factor": np.nan,
                "return_over_dd": 0.0, "final_value": start_capital,
                **({"equity": None, "eq_dates": None} if want_equity else {})}

    eq = np.array(equity)
    tg, tp = np.concatenate(trade_gain), np.concatenate(trade_pnl)
    total_return = (eq[-1] - start_capital) / start_capital * 100.0
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / peak * 100.0).min())

    rets = np.diff(eq) / eq[:-1]
    sharpe = np.nan
    if len(rets) >= 2 and rets.std(ddof=1) > 0:
        span = (eq_dates[-1] - eq_dates[0]).astype("timedelta64[D]").astype(int)
        ppy = len(rets) / span * 365.25 if span > 0 else 252.0
        sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(ppy)

    wins, losses = tp[tp > 0].sum(), -tp[tp < 0].sum()
    pf = wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)
    out = {
        "n_days": len(eq), "n_trades": int(len(tg)),
        "total_return": total_return, "sharpe_like": sharpe,
        "max_drawdown": max_dd, "win_rate": float((tg > 0).mean() * 100.0),
        "profit_factor": pf,
        # floored at 1% so a near-zero drawdown can't produce an absurd ratio
        "return_over_dd": total_return / max(abs(max_dd), 1.0),
        "final_value": float(eq[-1]),
    }
    if want_equity:
        out["equity"], out["eq_dates"] = eq, np.array(eq_dates)
    return out


def _score(stats: dict, objective: str) -> float:
    v = stats.get(OBJECTIVES[objective])
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    return float(min(v, 1e6))        # clip +inf profit factor


def _smooth_over_neighbours(df: pd.DataFrame, col: str, stop_vals: list, pos_vals: list) -> pd.Series:
    """Average `col` over each row's 3×3 neighbourhood on the (stop %, max
    positions) grid, within the same categorical cell (stop type, target,
    sizing, deployment). This is the plateau score."""
    s_rank = {v: i for i, v in enumerate(stop_vals)}
    p_rank = {v: i for i, v in enumerate(pos_vals)}
    cell_cols = ["stop_type", "target", "weight_conf", "deploy_pct"]
    lookup = {}
    for row in df[cell_cols + ["stop_pct", "max_positions", col]].itertuples(index=False):
        k = (row[0], row[1], row[2], row[3], s_rank.get(row[4], 0), p_rank[row[5]])
        lookup[k] = row[6]
    out = []
    for row in df[cell_cols + ["stop_pct", "max_positions"]].itertuples(index=False):
        si, pi = s_rank.get(row[4], 0), p_rank[row[5]]
        vals = []
        for ds, dp in itertools.product((-1, 0, 1), repeat=2):
            v = lookup.get((row[0], row[1], row[2], row[3], si + ds, pi + dp))
            if v is not None and not pd.isna(v):
                vals.append(v)
        out.append(np.mean(vals) if vals else np.nan)
    return pd.Series(out, index=df.index)


def count_grid(stop_types, stop_vals, targets, pos_vals, weights, deploys) -> int:
    n_stop = sum(1 if t == "none" else len(stop_vals) for t in stop_types)
    return n_stop * len(targets) * len(pos_vals) * len(weights) * len(deploys)


def run_grid(paths: TradePaths, *, stop_types, stop_vals, targets, pos_vals, weights, deploys,
             start_capital, commission, slippage_bps, objective, dd_cap, train_frac,
             gap_fills=True, progress_cb=None) -> tuple[pd.DataFrame, dict]:
    """Evaluate every combination. `targets` is a list of (mode, value) with
    mode in {none, model, fixed}; `weights` a list of bools (weight_conf);
    `deploys` a list of percents. Returns (results_df, meta)."""
    groups = build_day_groups(paths)
    n_days = len(groups.dates)
    split = int(round(n_days * train_frac)) if train_frac < 1.0 else n_days
    split = max(1, min(split, n_days))
    has_holdout = split < n_days and (n_days - split) >= 3 and split >= 3

    res_combos = []
    for st_type in stop_types:
        for sv in ([None] if st_type == "none" else stop_vals):
            for tm, tv in targets:
                res_combos.append((st_type, sv, tm, tv))

    rows = []
    for ci, (st_type, sv, tm, tv) in enumerate(res_combos):
        gains, outcome = resolve_gains(paths, st_type, sv, tm, tv, gap_fills)
        stop_rate = float((outcome == OUTCOME_STOP).mean() * 100.0)
        tgt_rate = float((outcome == OUTCOME_TARGET).mean() * 100.0)
        t_label = "none" if tm == "none" else ("model target" if tm == "model" else f"fixed {tv:g}%")
        for mp, wc, dp in itertools.product(pos_vals, weights, deploys):
            args = (start_capital, commission, slippage_bps, mp, wc, dp / 100.0)
            full = run_portfolio(gains, groups, 0, n_days, *args)
            row = {
                "stop_type": st_type, "stop_pct": sv if sv is not None else np.nan,
                "target": t_label, "max_positions": mp, "weight_conf": wc, "deploy_pct": dp,
                "stop_hit_pct": stop_rate, "target_hit_pct": tgt_rate,
                **{f"full_{k}": full[k] for k in
                   ("total_return", "sharpe_like", "max_drawdown", "win_rate",
                    "profit_factor", "return_over_dd", "n_trades", "final_value")},
            }
            row["full_obj"] = _score(full, objective)
            if has_holdout:
                tr = run_portfolio(gains, groups, 0, split, *args)
                te = run_portfolio(gains, groups, split, n_days, *args)
                row.update(train_obj=_score(tr, objective), train_return=tr["total_return"],
                           train_dd=tr["max_drawdown"], test_obj=_score(te, objective),
                           test_return=te["total_return"], test_dd=te["max_drawdown"],
                           test_win_rate=te["win_rate"])
            rows.append(row)
        if progress_cb:
            progress_cb(ci + 1, len(res_combos))

    df = pd.DataFrame(rows)
    # Treat "stop type none" as a single column in the stop axis
    stop_axis = sorted(set(stop_vals))
    pos_axis = sorted(set(pos_vals))
    # none-stop rows have NaN stop_pct → give them rank 0 inside the smoother
    df["_stop_key"] = df["stop_pct"].where(df["stop_pct"].notna(), stop_axis[0] if stop_axis else 0.0)
    sm_df = df.assign(stop_pct=df["_stop_key"])
    df["full_robust"] = _smooth_over_neighbours(sm_df, "full_obj", stop_axis or [0.0], pos_axis)
    if has_holdout:
        df["train_robust"] = _smooth_over_neighbours(sm_df, "train_obj", stop_axis or [0.0], pos_axis)
    df = df.drop(columns="_stop_key")

    # Drawdown cap applies to the full-sample drawdown of the combo itself.
    df["within_dd_cap"] = df["full_max_drawdown"] >= -abs(dd_cap)
    meta = {"n_days": n_days, "split_day": split, "has_holdout": has_holdout,
            "split_date": pd.Timestamp(groups.dates[split]) if has_holdout else None,
            "first_date": pd.Timestamp(groups.dates[0]), "last_date": pd.Timestamp(groups.dates[-1]),
            "n_combos": len(df), "n_trades_pool": paths.n}
    return df, meta


def replay_config(paths: TradePaths, row: pd.Series, *, start_capital, commission,
                  slippage_bps, gap_fills=True) -> tuple[dict, pd.DataFrame]:
    """Re-run one result row with full detail → (stats, trade-level frame)."""
    groups = build_day_groups(paths)
    tm = "none" if row["target"] == "none" else ("model" if row["target"] == "model target" else "fixed")
    tv = float(row["target"].split()[1].rstrip("%")) if tm == "fixed" else None
    sv = None if row["stop_type"] == "none" else float(row["stop_pct"])
    gains, outcome = resolve_gains(paths, row["stop_type"], sv, tm, tv, gap_fills)
    stats = run_portfolio(gains, groups, 0, len(groups.dates), start_capital, commission, slippage_bps,
                          int(row["max_positions"]), bool(row["weight_conf"]),
                          row["deploy_pct"] / 100.0, want_equity=True)
    tl = paths.meta.assign(resolved_gain_pct=gains,
                           exit=np.array(["close", "stop", "target"])[outcome])
    return stats, tl


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

_STOP_LABEL = {"none": "No stop", "fixed": "Fixed stop", "trailing": "Trailing stop"}


def _describe(row) -> str:
    stop = "no stop" if row["stop_type"] == "none" else f"{row['stop_type']} stop {row['stop_pct']:g}%"
    tgt = "no target" if row["target"] == "none" else row["target"]
    sizing = "confidence-weighted" if row["weight_conf"] else "equal-weight"
    return (f"{stop} · {tgt} · max {int(row['max_positions'])} positions · "
            f"{sizing} · {int(row['deploy_pct'])}% deployed")


def render_optimizer_panel(pos_signals: pd.DataFrame, min_date, max_date) -> None:
    """Entry point called from tab_backtesting, which wraps it in an @st.fragment
    so submit clicks don't rerun the whole app."""
    import streamlit as st
    import plotly.graph_objects as go
    # Lazy imports: tab_backtesting imports this module lazily too, so there
    # is no import-time cycle.
    from tab_backtesting import (_fetch_5min_bars_batch, _filter_sim_trades,
                                 _five_min_cutoff_date, _info_card, _ALL_SIGNALS)
    from chart_utils import LAYOUT, AXIS_STYLE
    from dashboard_styles import render_labeled_divider, render_hero_metric
    import user_state

    st.markdown('<div class="form-section-label">Search space</div>', unsafe_allow_html=True)
    with st.form("opt_form", border=True):
        c1, c2, c3 = st.columns(3)
        start_capital = c1.number_input("Starting capital ($)", min_value=1.0, value=10000.0,
                                        step=100.0, key="opt_capital")
        commission = c2.number_input("Commission per trade ($)", min_value=0.0, step=0.5,
                                     value=float(st.session_state.get("commission_fee", 0.0)),
                                     key="opt_commission")
        slippage_bps = c3.number_input("Est. slippage (bps, round-trip)", min_value=0.0, step=1.0,
                                       value=float(st.session_state.get("slippage_bps", 10.0)),
                                       key="opt_slippage")
        d1, d2 = st.columns(2)
        date_range = d1.date_input("Date range", value=(min_date, max_date), min_value=min_date,
                                   max_value=max_date, key="opt_dates",
                                   help="Limited to ~60 days by yfinance's 5-minute history.")
        signals = d2.multiselect("Signal(s) to act on", _ALL_SIGNALS,
                                 default=["STRONG BUY", "BUY"], key="opt_signals")

        st.markdown("**Stop-loss**")
        s1, s2, s3 = st.columns([2, 2, 1])
        stop_types = s1.multiselect("Stop types to compare", list(_STOP_LABEL),
                                    default=["none", "fixed", "trailing"],
                                    format_func=_STOP_LABEL.get, key="opt_stop_types")
        stop_lo_hi = s2.slider("Stop % range", 1.0, 25.0, (2.0, 10.0), step=0.5, key="opt_stop_range")
        stop_step = s3.selectbox("Step", [0.5, 1.0, 1.5, 2.0, 2.5], index=1, key="opt_stop_step")

        st.markdown("**Exit target**")
        t1, t2 = st.columns(2)
        target_modes = t1.multiselect("Target modes", ["No target (hold to close)", "Model's predicted target"],
                                      default=["No target (hold to close)", "Model's predicted target"],
                                      key="opt_target_modes")
        fixed_targets = t2.multiselect("Also test fixed targets (%)", [2, 3, 4, 5, 6, 8, 10, 15],
                                       default=[], key="opt_fixed_targets")

        st.markdown("**Portfolio construction**")
        p1, p2, p3 = st.columns(3)
        pos_vals = p1.multiselect("Max positions per day", [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25],
                                  default=[2, 3, 4, 5, 6, 8, 10, 15], key="opt_positions")
        weights = p2.multiselect("Position sizing", ["Equal split", "Confidence-weighted"],
                                 default=["Equal split"], key="opt_weights")
        deploys = p3.multiselect("Capital deployed per day (%)", [25, 50, 75, 100],
                                 default=[100], key="opt_deploys")

        st.markdown("**How to judge & guard against overfitting**")
        o1, o2, o3, o4 = st.columns(4)
        objective = o1.selectbox("Optimize for", list(OBJECTIVES), key="opt_objective")
        dd_cap = o2.number_input("Max drawdown limit (%)", min_value=1.0, max_value=100.0, value=25.0,
                                 step=1.0, key="opt_dd_cap",
                                 help="Combos whose full-period max drawdown is worse than this are excluded from rankings.")
        train_pct = o3.slider("Train on first (% of days)", 50, 100, 65, step=5, key="opt_train_pct",
                              help="Combos are ranked on this early slice, then re-scored on the later, unseen days. 100 = no holdout.")
        gap_fills = o4.checkbox("Realistic gap fills", value=True, key="opt_gap_fills",
                                help="Stops that gap through are filled at the open (worse than the stop); targets that gap up fill at the open (better). Off = fill at the exact level, like the single-run simulator.")

        with st.container(key="cta_run_optimizer"):
            submitted = st.form_submit_button("🎯  Find optimal parameters", use_container_width=True)

    # ── validate inputs ──
    if isinstance(date_range, tuple) and len(date_range) == 2:
        sim_start, sim_end = date_range
    else:
        sim_start, sim_end = min_date, max_date
    stop_vals = list(np.arange(stop_lo_hi[0], stop_lo_hi[1] + 1e-9, stop_step).round(2))
    targets = []
    if "No target (hold to close)" in target_modes:
        targets.append(("none", None))
    if "Model's predicted target" in target_modes:
        targets.append(("model", None))
    targets += [("fixed", float(v)) for v in fixed_targets]
    weight_flags = [w == "Confidence-weighted" for w in weights]

    problems = []
    if not signals: problems.append("pick at least one signal")
    if not stop_types: problems.append("pick at least one stop type")
    if not targets: problems.append("pick at least one target mode")
    if not pos_vals: problems.append("pick at least one max-positions value")
    if not weights: problems.append("pick at least one sizing mode")
    if not deploys: problems.append("pick at least one deployment level")
    n_combos = 0 if problems else count_grid(stop_types, stop_vals, targets, pos_vals, weights, deploys)
    MAX_COMBOS = 8000

    if submitted:
        if problems:
            st.warning("To run the optimizer, " + ", ".join(problems) + ".")
        elif n_combos > MAX_COMBOS:
            st.error(f"That's {n_combos:,} combinations — over the {MAX_COMBOS:,} limit. "
                     "Widen the stop step or trim a few lists.")
        else:
            eligible = _filter_sim_trades(pos_signals, signals, sim_start, sim_end)
            if eligible.empty:
                st.warning("No trades match this signal/date selection.")
            else:
                # Bars are the slow part and don't depend on the parameter grid,
                # so cache the per-trade paths and reuse them on re-runs.
                fetch_key = (tuple(sorted(eligible["symbol"].dropna().unique())), sim_start, sim_end,
                             tuple(sorted(signals)))
                cache = st.session_state.get("opt_paths_cache")
                if cache is not None and cache["key"] == fetch_key:
                    paths = cache["paths"]
                    st.toast("♻️ Reusing cached 5-min bars — no re-fetch needed.")
                else:
                    symbols = fetch_key[0]
                    bar = st.progress(0, text=f"Fetching 5-minute bars for {len(symbols)} symbol(s)…")

                    def _cb(done, total, sym):
                        bar.progress(0.7 * done / total,
                                     text=f"Fetching 5-minute bars… {done}/{total} (last: {sym})")

                    dates = eligible["prediction_date"].dt.date
                    fetch_start = max(dates.min() - _dt.timedelta(days=5), _five_min_cutoff_date())
                    bars = _fetch_5min_bars_batch(symbols, fetch_start, dates.max(), progress_cb=_cb)
                    bar.progress(0.72, text="Building intraday paths…")
                    paths = build_trade_paths(eligible, bars)
                    bar.empty()
                    st.session_state["opt_paths_cache"] = {"key": fetch_key, "paths": paths}

                if paths.n == 0:
                    st.warning("None of the eligible trades could be priced from 5-minute bars "
                               f"(drop reasons: {paths.drop_reasons}). This is usually Yahoo rate-limiting — "
                               "check the Debug Log and retry in a few minutes.")
                else:
                    bar = st.progress(0.75, text=f"Evaluating {n_combos:,} combinations…")
                    df, meta = run_grid(
                        paths, stop_types=stop_types, stop_vals=stop_vals, targets=targets,
                        pos_vals=sorted(pos_vals), weights=weight_flags, deploys=sorted(deploys),
                        start_capital=start_capital, commission=commission, slippage_bps=slippage_bps,
                        objective=objective, dd_cap=dd_cap, train_frac=train_pct / 100.0,
                        gap_fills=gap_fills,
                        progress_cb=lambda i, n: bar.progress(0.75 + 0.25 * i / n),
                    )
                    bar.empty()
                    st.session_state["opt_results"] = {
                        "df": df, "meta": meta, "objective": objective, "dd_cap": dd_cap,
                        "costs": dict(start_capital=start_capital, commission=commission,
                                      slippage_bps=slippage_bps, gap_fills=gap_fills),
                        "signals": signals, "stop_vals": stop_vals, "pos_vals": sorted(pos_vals),
                        "dropped": paths.n_eligible - paths.n,
                    }
                    user_state.persist(commission_fee=commission, slippage_bps=slippage_bps)

    res = st.session_state.get("opt_results")
    if res is None:
        if not problems:
            st.caption(f"This setup will test **{n_combos:,}** combinations. Bars are fetched once, "
                       "then every combination is evaluated in memory.")
        st.info("Choose your search space above and click \"Find optimal parameters\".")
        return

    df, meta, objective = res["df"], res["meta"], res["objective"]
    paths = (st.session_state.get("opt_paths_cache") or {}).get("paths")
    costs = res["costs"]

    # ── interpret results ──
    render_labeled_divider("Result")
    ok = df[df["within_dd_cap"]].copy()
    if ok.empty:
        st.warning(f"No combination stayed within the {res['dd_cap']:.0f}% drawdown limit. "
                   "Raise the limit or widen the search.")
        return

    has_holdout = meta["has_holdout"]
    rank_col = "train_robust" if has_holdout else "full_robust"
    ranked = ok.dropna(subset=[rank_col]).sort_values(rank_col, ascending=False)
    if ranked.empty:
        st.warning("Not enough valid results to rank.")
        return
    best = ranked.iloc[0]

    span = f"{meta['first_date']:%b %d} → {meta['last_date']:%b %d}"
    _info_card(
        f"<b>{meta['n_combos']:,} combinations</b> tested over <b>{meta['n_days']} trading days</b> "
        f"({span}), {meta['n_trades_pool']} priceable trades"
        + (f" ({res['dropped']} dropped for missing bars)" if res["dropped"] else "")
        + f". Ranking by <b>{objective.lower()}</b>, smoothed across neighbouring settings"
        + (f", selected on days before {meta['split_date']:%b %d} and re-scored on the unseen days after."
           if has_holdout else " (no holdout — you set training to 100%).")
        , muted=True)
    if meta["n_days"] < 30:
        st.warning(f"Only {meta['n_days']} trading days of data. Treat every result below as a hypothesis "
                   "to monitor, not a settled answer — a few lucky days can dominate.")

    st.markdown(f"#### Recommended: {_describe(best)}")
    cols = st.columns(4)
    if has_holdout:
        cols[0].metric("Train return", f"{best['train_return']:+.1f}%", help="Period the combo was selected on.")
        cols[1].metric("Unseen (test) return", f"{best['test_return']:+.1f}%",
                       delta=f"{best['test_return'] - best['train_return']:+.1f} vs train",
                       help="Days the selection never saw. This is the honest number.")
        cols[2].metric("Test max drawdown", f"{best['test_dd']:.1f}%")
        cols[3].metric("Test win rate", f"{best['test_win_rate']:.0f}%")
        if best["test_return"] < 0:
            st.error("⚠️ This combo LOST money on the unseen days. The in-sample edge likely isn't real — "
                     "don't rely on it.")
        elif best["test_return"] < 0.4 * best["train_return"]:
            st.warning("The unseen-days result is much weaker than training — a classic sign of overfitting. "
                       "Prefer settings from a broad plateau in the heatmap below over this single peak.")
    else:
        cols[0].metric("Total return", f"{best['full_total_return']:+.1f}%")
        cols[1].metric("Max drawdown", f"{best['full_max_drawdown']:.1f}%")
        cols[2].metric("Win rate", f"{best['full_win_rate']:.0f}%")
        cols[3].metric("Trades", f"{int(best['full_n_trades'])}")
    st.caption(f"Stop triggers on {best['stop_hit_pct']:.0f}% of trades"
               + (f", target on {best['target_hit_pct']:.0f}%" if best["target"] != "none" else "") + ".")

    # ── Stop-strategy comparison ──
    render_labeled_divider("Which stop strategy wins?")
    ref = best
    same_cell = ok[(ok["target"] == ref["target"]) & (ok["max_positions"] == ref["max_positions"])
                   & (ok["weight_conf"] == ref["weight_conf"]) & (ok["deploy_pct"] == ref["deploy_pct"])]
    fig = go.Figure()
    palette = {"fixed": "#e0a83c", "trailing": "#38bdf8"}
    for stype in ("fixed", "trailing"):
        sub = same_cell[same_cell["stop_type"] == stype].sort_values("stop_pct")
        if not sub.empty:
            fig.add_trace(go.Scatter(x=sub["stop_pct"], y=sub["full_obj"], mode="lines+markers",
                                     name=_STOP_LABEL[stype], line=dict(color=palette[stype])))
    nostop = same_cell[same_cell["stop_type"] == "none"]
    if not nostop.empty:
        fig.add_hline(y=float(nostop["full_obj"].iloc[0]), line_dash="dash",
                      line_color="rgba(255,255,255,0.45)", annotation_text="No stop", annotation_font_size=10)
    fig.update_layout(title=f"{objective} vs stop % (other settings held at the recommendation)",
                      xaxis_title="Stop %", yaxis_title=objective, height=340, hovermode="x unified", **LAYOUT)
    fig.update_xaxes(**AXIS_STYLE); fig.update_yaxes(**AXIS_STYLE)
    if fig.data or not nostop.empty:
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Read this for SHAPE, not the peak: a wide, flat hump means the setting is forgiving; "
               "a single spike usually means noise. If a stop line never rises above the dashed 'No stop' "
               "line, stops aren't helping this model's signals in this period.")

    # ── Heatmap ──
    render_labeled_divider("Stop % × max positions")
    h1, h2, h3 = st.columns(3)
    avail_types = [t for t in ("fixed", "trailing") if t in set(ok["stop_type"])]
    if not avail_types:
        st.caption("Heatmap needs a fixed or trailing stop in the search.")
    else:
        h_type = h1.selectbox("Stop type", avail_types, key="opt_hm_type",
                              index=avail_types.index(best["stop_type"]) if best["stop_type"] in avail_types else 0,
                              format_func=_STOP_LABEL.get)
        targets_avail = sorted(ok["target"].unique())
        h_tgt = h2.selectbox("Target", targets_avail, key="opt_hm_target",
                             index=targets_avail.index(best["target"]) if best["target"] in targets_avail else 0)
        metric_opts = {f"{objective} (full period)": "full_obj", "Total return %": "full_total_return",
                       "Max drawdown %": "full_max_drawdown"}
        if has_holdout:
            metric_opts[f"{objective} (unseen days)"] = "test_obj"
            metric_opts["Return % (unseen days)"] = "test_return"
        h_metric = h3.selectbox("Colour by", list(metric_opts), key="opt_hm_metric")
        sub = ok[(ok["stop_type"] == h_type) & (ok["target"] == h_tgt)
                 & (ok["weight_conf"] == best["weight_conf"]) & (ok["deploy_pct"] == best["deploy_pct"])]
        if sub.empty:
            st.caption("No rows for that combination.")
        else:
            piv = sub.pivot_table(index="max_positions", columns="stop_pct", values=metric_opts[h_metric])
            hm = go.Figure(go.Heatmap(
                z=piv.values, x=[f"{c:g}%" for c in piv.columns], y=[str(i) for i in piv.index],
                colorscale="RdYlGn", zmid=0 if "return" in metric_opts[h_metric] or "obj" in metric_opts[h_metric] else None,
                text=np.round(piv.values, 1), texttemplate="%{text}", colorbar=dict(title=""),
                hovertemplate="stop %{x}, max pos %{y}: %{z:.2f}<extra></extra>"))
            hm.update_layout(title=h_metric, xaxis_title="Stop %", yaxis_title="Max positions / day",
                             height=380, **LAYOUT)
            st.plotly_chart(hm, use_container_width=True)
            st.caption("Look for a green BLOCK, not a single green cell.")

    # ── Leaderboard ──
    render_labeled_divider("Leaderboard")
    show_n = st.slider("Rows to show", 5, 50, 15, key="opt_show_n")
    board = ranked.head(show_n).copy()
    board["Config"] = board.apply(_describe, axis=1)
    cols_out = {"Config": "Config", "full_total_return": "Return % (all)", "full_max_drawdown": "Max DD %",
                "full_sharpe_like": "Sharpe-like", "full_win_rate": "Win %", "full_profit_factor": "Profit factor",
                "stop_hit_pct": "Stop hit %", "full_robust": "Plateau score"}
    if has_holdout:
        cols_out.update({"train_return": "Train ret %", "test_return": "Test ret %", "test_dd": "Test DD %"})
    show = board[list(cols_out)].rename(columns=cols_out)
    st.dataframe(show, use_container_width=True, hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in show.columns if c != "Config"})
    st.download_button("⬇ Download all results (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="optimizer_results.csv", mime="text/csv", key="opt_dl")

    # ── Replay ──
    render_labeled_divider("Replay a configuration")
    options = {f"#{i + 1}  {_describe(r)}": r for i, (_, r) in enumerate(ranked.head(10).iterrows())}
    pick = st.selectbox("Pick one of the top 10", list(options), key="opt_replay_pick")
    if paths is not None and pick:
        stats, tl = replay_config(paths, options[pick], **costs)
        if stats.get("equity") is not None:
            ef = go.Figure(go.Scatter(x=stats["eq_dates"], y=stats["equity"], mode="lines",
                                      line=dict(color="#e0a83c"), name="Portfolio"))
            ef.add_hline(y=costs["start_capital"], line_dash="dash", line_color="rgba(255,255,255,0.15)")
            if has_holdout:
                ef.add_vline(x=meta["split_date"], line_dash="dot", line_color="rgba(56,189,248,0.7)",
                             annotation_text="train | unseen", annotation_font_size=10)
            ef.update_layout(title="Equity curve", height=340, xaxis_title="Date",
                             yaxis_title="Portfolio value ($)", **LAYOUT)
            ef.update_xaxes(**AXIS_STYLE); ef.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(ef, use_container_width=True)
            m = st.columns(4)
            m[0].metric("Return", f"{stats['total_return']:+.1f}%")
            m[1].metric("Max drawdown", f"{stats['max_drawdown']:.1f}%")
            m[2].metric("Win rate", f"{stats['win_rate']:.0f}%")
            m[3].metric("Exits: stop / target / close",
                        f"{(tl['exit'] == 'stop').mean() * 100:.0f}% / {(tl['exit'] == 'target').mean() * 100:.0f}% / "
                        f"{(tl['exit'] == 'close').mean() * 100:.0f}%")

    _info_card(
        "<b>Read this before acting on it.</b> The optimizer searches a lot of settings against a short "
        "history, so the top result is partly luck by construction. The unseen-days score and the plateau "
        "shape are the things to trust. Same-day-only exits are assumed (as in the rest of the simulator), "
        "fills ignore liquidity, and slippage is a flat estimate. Re-run as new days accumulate — settings "
        "that keep landing on the plateau are the ones worth using.",
        muted=True)
