"""
chart_utils.py — Shared chart theme, colors, and helper utilities.
Import CHART_THEME, LAYOUT, AXIS_STYLE from here in all tabs.
"""

# ── Plotly chart theme ─────────────────────────────────────────────────────────
CHART_THEME = {
    'plot_bgcolor':  'rgba(11, 17, 24, 0.8)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'font':          dict(color='#c8d8e8', family='JetBrains Mono, monospace'),
    'title_font':    dict(size=15, color='#e2ecf8', family='Space Grotesk, sans-serif'),
    'xaxis':         dict(
        gridcolor='rgba(0,212,255,0.06)',
        color='#4a6a8a',
        linecolor='rgba(0,212,255,0.1)',
        tickfont=dict(size=10),
    ),
    'yaxis':         dict(
        gridcolor='rgba(0,212,255,0.06)',
        color='#4a6a8a',
        linecolor='rgba(0,212,255,0.1)',
        tickfont=dict(size=10),
    ),
}

# Compact layout for plotly (used in _LAYOUT style dicts)
LAYOUT = dict(
    plot_bgcolor='rgba(11, 17, 24, 0.8)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#c8d8e8', family='JetBrains Mono, monospace'),
    margin=dict(t=44, b=24, l=24, r=16),
)

AXIS_STYLE = dict(
    gridcolor='rgba(0,212,255,0.06)',
    color='#4a6a8a',
    linecolor='rgba(0,212,255,0.1)',
    tickfont=dict(size=10),
)

# ── Brand color palette ────────────────────────────────────────────────────────
COLORS = {
    'primary':   '#00d4ff',
    'secondary': '#00ff88',
    'amber':     '#ffb800',
    'red':       '#ff3860',
    'purple':    '#b060ff',
    'blue':      '#1a8cff',
    'teal':      '#00c896',

    # Signal colors
    'strong_buy': '#10b981',
    'buy':        '#00d4ff',
    'hold':       '#f59e0b',
    'avoid':      '#ef4444',

    # Chart series palette (10 colors, never repeat adjacent)
    'series': [
        '#00d4ff', '#00ff88', '#ffb800', '#ff3860', '#b060ff',
        '#1a8cff', '#00c896', '#f97316', '#84cc16', '#ec4899',
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
    'BUY':        '#00d4ff22',
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
