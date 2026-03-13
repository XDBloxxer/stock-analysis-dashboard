"""
dashboard_styles.py — Refined Terminal Noir v4
Design direction: Pull back the visual noise. Cleaner type hierarchy,
restrained use of glows, generous breathing room. Still dark, still
distinctive — but professional rather than overwhelming.
"""

DASHBOARD_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;500;600;700;800&family=Instrument+Serif:ital@0;1&display=swap');

/* ── CSS Variables ────────────────────────────────────────────────────────── */
:root {
    --bg-0:          #080b0f;
    --bg-1:          #0d1117;
    --bg-2:          #131920;
    --bg-3:          #18222c;
    --bg-4:          #1e2d3a;
    --bg-glass:      rgba(13, 17, 23, 0.75);

    --cyan:          #38bdf8;
    --cyan-dim:      rgba(56, 189, 248, 0.12);
    --cyan-border:   rgba(56, 189, 248, 0.18);
    --cyan-glow:     rgba(56, 189, 248, 0.08);
    --green:         #34d399;
    --green-dim:     rgba(52, 211, 153, 0.1);
    --amber:         #fbbf24;
    --amber-dim:     rgba(251, 191, 36, 0.1);
    --red:           #f87171;
    --red-dim:       rgba(248, 113, 113, 0.1);
    --purple:        #a78bfa;

    --text-0:        #f0f4f8;
    --text-1:        #94a3b8;
    --text-2:        #4a5a6a;
    --text-3:        #2a3a4a;

    --border:        rgba(255, 255, 255, 0.06);
    --border-focus:  rgba(56, 189, 248, 0.3);

    --font-display:  'Syne', sans-serif;
    --font-body:     'DM Mono', monospace;
    --font-serif:    'Instrument Serif', serif;

    --radius:        8px;
    --radius-sm:     5px;
    --shadow:        0 2px 12px rgba(0, 0, 0, 0.35);
    --shadow-lg:     0 8px 32px rgba(0, 0, 0, 0.5);
}

/* ── Base ────────────────────────────────────────────────────────────────── */
.stApp {
    background: var(--bg-0) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 15% 0%, rgba(56, 189, 248, 0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 85% 100%, rgba(52, 211, 153, 0.03) 0%, transparent 60%);
    font-family: var(--font-body);
}

/* Subtle dot grid — much quieter than before */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: radial-gradient(rgba(56, 189, 248, 0.06) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none;
    z-index: 0;
}

/* ── Main Container ───────────────────────────────────────────────────────── */
.main .block-container {
    padding-top: 1.75rem !important;
    padding-bottom: 4rem !important;
    max-width: 1400px !important;
    animation: fadeUp 0.4s ease forwards;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Typography ───────────────────────────────────────────────────────────── */
h1 {
    font-family: var(--font-display) !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--text-0) !important;
    background: none !important;
    -webkit-text-fill-color: var(--text-0) !important;
    filter: none !important;
    text-shadow: none !important;
    margin: 0 !important;
}

h2 {
    font-family: var(--font-display) !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--cyan) !important;
}

h3 {
    font-family: var(--font-display) !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: var(--text-0) !important;
    border: none !important;
    padding: 0 !important;
    margin-top: 1.25rem !important;
}

h4 {
    font-family: var(--font-display) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-2) !important;
}

p, li {
    font-family: var(--font-body) !important;
    font-size: 0.875rem !important;
    color: var(--text-1) !important;
}

/* ── Metric Cards ─────────────────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 16px 18px 14px !important;
    position: relative !important;
    overflow: hidden !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    min-height: 88px !important;
}

div[data-testid="metric-container"]:hover {
    border-color: var(--cyan-border) !important;
    box-shadow: 0 0 0 1px var(--cyan-dim), var(--shadow) !important;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] > div,
div[data-testid="stMetricLabel"] div {
    font-family: var(--font-body) !important;
    font-size: 0.65rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--text-2) !important;
    margin-bottom: 6px !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] > div,
div[data-testid="stMetricValue"] div {
    font-family: var(--font-body) !important;
    font-size: 1.9rem !important;
    font-weight: 500 !important;
    color: var(--text-0) !important;
    letter-spacing: -0.03em !important;
    line-height: 1.1 !important;
}

div[data-testid="stMetricDelta"],
div[data-testid="stMetricDelta"] > div,
div[data-testid="stMetricDelta"] div {
    font-family: var(--font-body) !important;
    font-size: 0.72rem !important;
    font-weight: 400 !important;
    margin-top: 4px !important;
    opacity: 0.75 !important;
}

/* ── SMALL metric variant ─────────────────────────────────────────────────── */
.metrics-sm div[data-testid="metric-container"] {
    min-height: 70px !important;
    padding: 11px 14px 9px !important;
}
.metrics-sm div[data-testid="stMetricValue"] > div {
    font-size: 1.3rem !important;
}
.metrics-sm div[data-testid="stMetricLabel"] > div {
    font-size: 0.58rem !important;
}

/* ── Tab Bar ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-1) !important;
    border-radius: var(--radius) !important;
    padding: 3px !important;
    gap: 1px !important;
    border: 1px solid var(--border) !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-2) !important;
    font-family: var(--font-body) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 7px 18px !important;
    border: none !important;
    transition: color 0.15s, background 0.15s !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-1) !important;
    background: rgba(255, 255, 255, 0.03) !important;
}

.stTabs [aria-selected="true"] {
    background: var(--bg-3) !important;
    color: var(--cyan) !important;
    border: 1px solid var(--cyan-border) !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 18px !important;
}

/* Suppress .arrow_right leak */
.stTabs [data-baseweb="tab"] [data-testid="stIconMaterial"],
.stTabs [data-baseweb="tab"] svg ~ span,
.stTabs [data-baseweb="tab"] > span:last-child:not(:first-child) {
    font-size: 0 !important;
    color: transparent !important;
    width: 0 !important;
    overflow: hidden !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background: var(--bg-3) !important;
    color: var(--text-1) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-body) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 7px 16px !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    color: var(--text-0) !important;
    border-color: rgba(255, 255, 255, 0.14) !important;
    background: var(--bg-4) !important;
}

/* Refresh — cyan accent */
.btn-refresh .stButton > button {
    color: var(--cyan) !important;
    border-color: var(--cyan-border) !important;
    background: var(--cyan-dim) !important;
}
.btn-refresh .stButton > button:hover {
    background: rgba(56, 189, 248, 0.16) !important;
}

/* Danger — red */
.btn-danger .stButton > button {
    color: var(--red) !important;
    border-color: rgba(248, 113, 113, 0.2) !important;
    background: var(--red-dim) !important;
}
.btn-danger .stButton > button:hover {
    background: rgba(248, 113, 113, 0.15) !important;
}

/* Action — green */
.btn-action .stButton > button {
    color: var(--green) !important;
    border-color: rgba(52, 211, 153, 0.2) !important;
    background: var(--green-dim) !important;
}
.btn-action .stButton > button:hover {
    background: rgba(52, 211, 153, 0.15) !important;
}

/* ── Selectbox ────────────────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stSelectbox [data-baseweb="select"] > div {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-0) !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    transition: border-color 0.15s !important;
}
.stSelectbox > div > div:focus-within,
.stSelectbox [data-baseweb="select"] > div:focus-within {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px var(--cyan-dim) !important;
}

/* ── Dataframes ───────────────────────────────────────────────────────────── */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow) !important;
}

/* ── Dividers ─────────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: var(--border) !important;
    margin: 20px 0 !important;
}
/* Override the diamond pseudo-element — too ornate */
hr::after { display: none !important; }

/* ── Expanders ────────────────────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    transition: border-color 0.15s !important;
    margin-bottom: 8px !important;
}
div[data-testid="stExpander"]:hover {
    border-color: rgba(255, 255, 255, 0.1) !important;
}
div[data-testid="stExpander"] > details > summary,
div[data-testid="stExpander"] details summary {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 11px 15px !important;
    cursor: pointer !important;
    list-style: none !important;
    -webkit-appearance: none !important;
    background: transparent !important;
}
div[data-testid="stExpander"] details summary::-webkit-details-marker { display: none !important; }
div[data-testid="stExpander"] details summary::marker { display: none !important; content: '' !important; }
div[data-testid="stExpanderDetails"],
div[data-testid="stExpander"] details summary > div:not([data-testid="stExpanderToggleIcon"]),
div[data-testid="stExpander"] details summary p,
div[data-testid="stExpander"] details summary span {
    font-family: var(--font-body) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: var(--text-1) !important;
    letter-spacing: 0.04em !important;
    margin: 0 !important;
    padding: 0 !important;
}
div[data-testid="stExpanderToggleIcon"] {
    font-size: 0 !important;
    color: transparent !important;
    display: flex !important;
    align-items: center !important;
    width: 16px !important;
    height: 16px !important;
    overflow: hidden !important;
}
div[data-testid="stExpanderToggleIcon"] * { font-size: 0 !important; color: transparent !important; }
div[data-testid="stExpanderToggleIcon"] svg {
    color: var(--text-2) !important;
    width: 13px !important;
    height: 13px !important;
    display: block !important;
    visibility: visible !important;
    transition: transform 0.2s ease, color 0.15s !important;
}
div[data-testid="stExpander"] details[open] div[data-testid="stExpanderToggleIcon"] svg {
    transform: rotate(90deg) !important;
    color: var(--cyan) !important;
}
div[data-testid="stExpander"] details summary:hover div[data-testid="stExpanderToggleIcon"] svg {
    color: var(--text-1) !important;
}
div[data-testid="stExpander"] details > div { padding: 4px 15px 14px !important; }

/* ── Alerts ───────────────────────────────────────────────────────────────── */
.stAlert, div[data-testid="stNotification"] {
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-body) !important;
    font-size: 0.78rem !important;
}
div[data-baseweb="notification"][kind="info"],
div[data-testid="stAlert"] {
    background: rgba(56, 189, 248, 0.05) !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
}

/* ── Text inputs ──────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-0) !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    transition: border-color 0.15s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px var(--cyan-dim) !important;
    outline: none !important;
}

/* Search bar variant */
.search-bar .stTextInput > div > div > input {
    background: var(--bg-2) !important;
    border-color: var(--cyan-border) !important;
    font-size: 0.875rem !important;
    padding: 10px 14px !important;
}

/* ── Sliders ──────────────────────────────────────────────────────────────── */
.stSlider > div > div > div { background: rgba(56, 189, 248, 0.1) !important; }
.stSlider > div > div > div > div { background: var(--cyan) !important; }
.stSlider [role="slider"] {
    background: var(--cyan) !important;
    border: 2px solid var(--bg-0) !important;
    box-shadow: 0 0 0 3px var(--cyan-dim) !important;
}

/* ── Multiselect ──────────────────────────────────────────────────────────── */
.stMultiSelect span[data-baseweb="tag"] {
    background: var(--cyan-dim) !important;
    border: 1px solid var(--cyan-border) !important;
    border-radius: 3px !important;
    color: var(--cyan) !important;
    font-family: var(--font-body) !important;
    font-size: 0.65rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Download button ──────────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid var(--cyan-border) !important;
    color: var(--cyan) !important;
    font-family: var(--font-body) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.15s !important;
}
.stDownloadButton > button:hover {
    background: var(--cyan-dim) !important;
    box-shadow: 0 0 0 1px var(--cyan-border) !important;
}

/* ── Spinner ──────────────────────────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--cyan) !important;
    border-right-color: var(--cyan-dim) !important;
    border-bottom-color: transparent !important;
    border-left-color: transparent !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-1) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-0); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.14); }

/* ── Skeleton loader ──────────────────────────────────────────────────────── */
.skeleton {
    background: linear-gradient(90deg, var(--bg-2) 0%, var(--bg-3) 50%, var(--bg-2) 100%);
    background-size: 200% 100%;
    animation: shimmer 1.8s ease-in-out infinite;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    width: 100%;
}
@keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ── Badge helper ─────────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-family: var(--font-body);
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-green  { background: var(--green-dim);  color: var(--green);  border: 1px solid rgba(52,211,153,0.2); }
.badge-amber  { background: var(--amber-dim);  color: var(--amber);  border: 1px solid rgba(251,191,36,0.2); }
.badge-red    { background: var(--red-dim);    color: var(--red);    border: 1px solid rgba(248,113,113,0.2); }
.badge-blue   { background: var(--cyan-dim);   color: var(--cyan);   border: 1px solid var(--cyan-border); }

/* ── Ticker pill ──────────────────────────────────────────────────────────── */
.ticker {
    display: inline-block;
    padding: 2px 8px;
    background: var(--cyan-dim);
    border: 1px solid var(--cyan-border);
    border-radius: 3px;
    font-family: var(--font-body);
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--cyan);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Status dot ───────────────────────────────────────────────────────────── */
.status-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    vertical-align: middle;
}
.status-dot.live    { background: var(--green); animation: pulse 3s ease-in-out infinite; }
.status-dot.warning { background: var(--amber); }
.status-dot.error   { background: var(--red);   animation: pulse 2s ease-in-out infinite; }
.status-dot.idle    { background: var(--text-3); }

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.6; transform: scale(0.8); }
}

/* ── Number highlight ─────────────────────────────────────────────────────── */
.num          { font-family: var(--font-body); font-weight: 500; font-size: 0.875rem; }
.num.positive { color: var(--green); }
.num.negative { color: var(--red); }
.num.neutral  { color: var(--cyan); }

/* ── Section header ───────────────────────────────────────────────────────── */
.section-header {
    font-family: var(--font-body);
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--text-2);
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 14px;
}

/* ── Data card ────────────────────────────────────────────────────────────── */
.data-card {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    transition: border-color 0.15s;
}
.data-card:hover { border-color: rgba(255,255,255,0.1); }

/* ── Search result card ───────────────────────────────────────────────────── */
.search-result-card {
    background: var(--cyan-dim);
    border: 1px solid var(--cyan-border);
    border-left: 2px solid var(--cyan);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-family: var(--font-body);
}

/* ── Cache warning ────────────────────────────────────────────────────────── */
.cache-warning {
    background: var(--red-dim);
    border: 1px solid rgba(248,113,113,0.2);
    border-left: 2px solid var(--red);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 10px 14px;
    font-family: var(--font-body);
    font-size: 0.76rem;
    color: var(--red);
    margin-bottom: 10px;
}

/* ── Preset chips ─────────────────────────────────────────────────────────── */
.preset-chips { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
.chip {
    padding: 3px 10px;
    border-radius: 3px;
    font-family: var(--font-body);
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.12s;
    background: var(--bg-3);
    border: 1px solid var(--border);
    color: var(--text-2);
}
.chip:hover, .chip.active {
    background: var(--cyan-dim);
    border-color: var(--cyan-border);
    color: var(--cyan);
}

/* ── Form submit ──────────────────────────────────────────────────────────── */
.stFormSubmitButton > button {
    background: var(--cyan-dim) !important;
    border: 1px solid var(--cyan-border) !important;
    color: var(--cyan) !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.15s !important;
}
.stFormSubmitButton > button:hover {
    background: rgba(56,189,248,0.18) !important;
}

/* ── Plotly chart containers ──────────────────────────────────────────────── */
div[data-testid="stPlotlyChart"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ── Number input ─────────────────────────────────────────────────────────── */
.stNumberInput button {
    background: var(--bg-3) !important;
    border-color: var(--border) !important;
    color: var(--text-2) !important;
}
.stNumberInput button:hover {
    background: var(--bg-4) !important;
    color: var(--text-1) !important;
}

/* ── Checkbox & Radio ─────────────────────────────────────────────────────── */
.stCheckbox label, .stRadio label {
    font-family: var(--font-body) !important;
    font-size: 0.8rem !important;
    color: var(--text-1) !important;
}

/* ── Progress bar ─────────────────────────────────────────────────────────── */
div[data-testid="stProgressBar"] > div > div {
    background: var(--cyan) !important;
}
div[data-testid="stProgressBar"] > div {
    background: var(--bg-3) !important;
    border-radius: 2px !important;
}

/* ── Caption ──────────────────────────────────────────────────────────────── */
.stCaption, small,
div[data-testid="stCaptionContainer"],
div[data-testid="stCaptionContainer"] p {
    font-family: var(--font-body) !important;
    font-size: 0.78rem !important;
    color: var(--text-2) !important;
    letter-spacing: 0.03em !important;
}

/* ── Toast ────────────────────────────────────────────────────────────────── */
div[data-testid="toastContainer"] {
    font-family: var(--font-body) !important;
    font-size: 0.76rem !important;
}

/* ── Staggered entry — subtle ─────────────────────────────────────────────── */
.main .block-container > div > div:nth-child(1) { animation: fadeUp 0.35s ease 0.03s both; }
.main .block-container > div > div:nth-child(2) { animation: fadeUp 0.35s ease 0.06s both; }
.main .block-container > div > div:nth-child(3) { animation: fadeUp 0.35s ease 0.09s both; }
.main .block-container > div > div:nth-child(4) { animation: fadeUp 0.35s ease 0.12s both; }
.main .block-container > div > div:nth-child(5) { animation: fadeUp 0.35s ease 0.15s both; }

</style>
"""
