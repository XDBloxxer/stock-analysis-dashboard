"""
chart_utils.py — Shared chart theme, colors, and helper utilities.
Import CHART_THEME, LAYOUT, AXIS_STYLE from here in all tabs.

v2 clean pass: charts now use the same palette/fonts as dashboard_styles.py
(previously they were on a different cyan + a font stack that isn't loaded
anywhere, so they silently fell back to the browser default). Grid and axis
colors are also bumped from near-invisible to legible without looking busy.
"""

# ── Plotly chart theme ─────────────────────────────────────────────────────────
CHART_THEME = {
    'plot_bgcolor':  'rgba(16, 22, 31, 0.6)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'font':          dict(color='#cbd5e1', family='DM Mono, monospace'),
    'title_font':    dict(size=15, color='#f8fafc', family='Syne, sans-serif'),
    'xaxis':         dict(
        gridcolor='rgba(148,163,184,0.12)',
        color='#8695ab',
        linecolor='rgba(148,163,184,0.18)',
        tickfont=dict(size=11),
    ),
    'yaxis':         dict(
        gridcolor='rgba(148,163,184,0.12)',
        color='#8695ab',
        linecolor='rgba(148,163,184,0.18)',
        tickfont=dict(size=11),
    ),
}

# Compact layout for plotly (used in _LAYOUT style dicts)
LAYOUT = dict(
    plot_bgcolor='rgba(16, 22, 31, 0.6)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#cbd5e1', family='DM Mono, monospace'),
    margin=dict(t=44, b=24, l=24, r=16),
)

AXIS_STYLE = dict(
    gridcolor='rgba(148,163,184,0.12)',
    color='#8695ab',
    linecolor='rgba(148,163,184,0.18)',
    tickfont=dict(size=11),
)

# Smaller tick font variant for compact mini-charts (no tickfont conflict when passed alone)
AXIS_STYLE_SM = dict(
    gridcolor='rgba(148,163,184,0.12)',
    color='#8695ab',
    linecolor='rgba(148,163,184,0.18)',
    tickfont=dict(size=10),
)

# ── Brand color palette ────────────────────────────────────────────────────────
# 'primary' repointed from cyan (#22d3ee) to brass/gold (#e0a83c) to match the
# v8 "Brass & Phosphor" theme in dashboard_styles.py. Gain/loss colors
# (secondary=green, red) are untouched — those are semantic, not brand accent.
COLORS = {
    'primary':   '#e0a83c',
    'secondary': '#34d399',
    'amber':     '#fbbf24',
    'red':       '#f87171',
    'purple':    '#8b5cf6',
    'blue':      '#38bdf8',
    'teal':      '#2dd4bf',

    # Signal colors
    'strong_buy': '#10b981',
    'buy':        '#e0a83c',
    'hold':       '#f59e0b',
    'avoid':      '#ef4444',

    # Chart series palette (10 colors, never repeat adjacent)
    'series': [
        '#e0a83c', '#34d399', '#38bdf8', '#f87171', '#a78bfa',
        '#2dd4bf', '#fbbf24', '#fb923c', '#a3e635', '#f472b6',
    ],
}

SIGNAL_COLORS = {
    'STRONG BUY': COLORS['strong_buy'],
    'BUY':        COLORS['buy'],
    'HOLD':       COLORS['hold'],
    'AVOID':      COLORS['avoid'],
}

SIGNAL_BG = {
    'STRONG BUY': '#10b98122',
    'BUY':        '#e0a83c22',
    'HOLD':       '#f59e0b22',
    'AVOID':      '#ef444422',
}

# ── Confusion matrix 4-color scheme ───────────────────────────────────────────
CONFUSION_COLORS = {
    'tp': '#10b981',  # green  — True Positive
    'fp': '#ef4444',  # red    — False Positive
    'fn': '#f59e0b',  # amber  — False Negative
    'tn': '#1e3a5f',  # dark   — True Negative
}
