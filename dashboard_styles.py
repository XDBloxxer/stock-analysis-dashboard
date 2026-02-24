"""
Dashboard UI Styles - Inject via st.markdown in dashboard.py
"""

DASHBOARD_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');

/* ── CSS Variables ────────────────────────────────────────────────────────── */
:root {
    --bg-primary:      #0a0c14;
    --bg-secondary:    #111422;
    --bg-card:         #141826;
    --bg-card-hover:   #1a1f30;
    --accent-primary:  #6c8aff;
    --accent-teal:     #0ef5c8;
    --accent-amber:    #f5a623;
    --accent-red:      #ff4d6a;
    --accent-green:    #0ef5a0;
    --text-primary:    #e8edf8;
    --text-secondary:  #7a85a0;
    --text-muted:      #4a5270;
    --border-subtle:   rgba(108, 138, 255, 0.12);
    --border-glow:     rgba(108, 138, 255, 0.35);
    --font-display:    'Syne', sans-serif;
    --font-mono:       'IBM Plex Mono', monospace;
}

/* ── Base App ─────────────────────────────────────────────────────────────── */
.stApp {
    background-color: var(--bg-primary) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 15% 10%, rgba(108, 138, 255, 0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 85% 80%, rgba(14, 245, 200, 0.04) 0%, transparent 55%),
        radial-gradient(ellipse 40% 60% at 70% 20%, rgba(108, 80, 200, 0.05) 0%, transparent 50%);
    font-family: var(--font-mono);
}

/* Subtle noise texture overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* ── Typography ───────────────────────────────────────────────────────────── */
h1, h2, h3 {
    font-family: var(--font-display) !important;
    letter-spacing: -0.02em !important;
}

h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, var(--text-primary) 0%, var(--accent-primary) 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

h2 {
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}

h3 {
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    font-size: 1.1rem !important;
}

p, li, span, div {
    font-family: var(--font-mono) !important;
}

/* ── Metric Cards ─────────────────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-left: 3px solid var(--accent-primary) !important;
    border-radius: 8px !important;
    padding: 16px 18px !important;
    position: relative !important;
    overflow: hidden !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

div[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 60px; height: 60px;
    background: radial-gradient(circle, rgba(108, 138, 255, 0.08) 0%, transparent 70%);
    border-radius: 0 8px 0 0;
}

div[data-testid="metric-container"]:hover {
    border-color: var(--border-glow) !important;
    box-shadow: 0 0 20px rgba(108, 138, 255, 0.08), 0 4px 16px rgba(0,0,0,0.3) !important;
}

div[data-testid="stMetricLabel"] > div {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}

div[data-testid="stMetricValue"] > div {
    font-family: var(--font-mono) !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
}

div[data-testid="stMetricDelta"] > div {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}

/* Positive delta = green left border */
div[data-testid="metric-container"]:has([data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Up"]) {
    border-left-color: var(--accent-green) !important;
}
div[data-testid="metric-container"]:has([data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Up"])::before {
    background: radial-gradient(circle, rgba(14, 245, 160, 0.08) 0%, transparent 70%);
}

/* Negative delta = red left border */
div[data-testid="metric-container"]:has([data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Down"]) {
    border-left-color: var(--accent-red) !important;
}
div[data-testid="metric-container"]:has([data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Down"])::before {
    background: radial-gradient(circle, rgba(255, 77, 106, 0.08) 0%, transparent 70%);
}

/* ── Tab Bar ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid var(--border-subtle) !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 7px !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    padding: 8px 16px !important;
    border: none !important;
    transition: color 0.15s ease, background 0.15s ease !important;
    position: relative !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: rgba(108, 138, 255, 0.08) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(108, 138, 255, 0.2), rgba(14, 245, 200, 0.1)) !important;
    color: var(--text-primary) !important;
    box-shadow: 0 0 0 1px var(--border-glow), inset 0 1px 0 rgba(255,255,255,0.05) !important;
}

/* Animated underline on selected tab */
.stTabs [aria-selected="true"]::after {
    content: '';
    position: absolute;
    bottom: 0; left: 20%; right: 20%;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-primary), var(--accent-teal));
    border-radius: 2px;
    animation: tabSlideIn 0.2s ease forwards;
}

@keyframes tabSlideIn {
    from { opacity: 0; transform: scaleX(0); }
    to   { opacity: 1; transform: scaleX(1); }
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 20px !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 7px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    padding: 8px 16px !important;
    transition: all 0.15s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button:hover {
    background: var(--bg-card-hover) !important;
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 16px rgba(108, 138, 255, 0.15) !important;
    color: var(--text-primary) !important;
}

.stButton > button:active {
    transform: scale(0.98) !important;
}

/* Primary button (first button in a row) */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(108, 138, 255, 0.25), rgba(14, 245, 200, 0.15)) !important;
    border-color: var(--accent-primary) !important;
}

/* ── Selectbox / Dropdowns ────────────────────────────────────────────────── */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 7px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}

.stSelectbox > div > div:focus-within {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 2px rgba(108, 138, 255, 0.15) !important;
}

/* ── Dataframes / Tables ──────────────────────────────────────────────────── */
.stDataFrame {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

.stDataFrame [data-testid="stDataFrameResizable"] {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}

/* Table header */
.stDataFrame thead tr th {
    background: var(--bg-secondary) !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    padding: 10px 12px !important;
}

/* Table rows — zebra striping */
.stDataFrame tbody tr:nth-child(even) td {
    background: rgba(255,255,255,0.015) !important;
}

.stDataFrame tbody tr:hover td {
    background: rgba(108, 138, 255, 0.06) !important;
}

.stDataFrame tbody tr td {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    color: var(--text-primary) !important;
    border-bottom: 1px solid rgba(255,255,255,0.03) !important;
    padding: 9px 12px !important;
}

/* ── Dividers → gradient fades ────────────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(
        90deg,
        transparent 0%,
        rgba(108, 138, 255, 0.3) 30%,
        rgba(14, 245, 200, 0.2) 70%,
        transparent 100%
    ) !important;
    margin: 24px 0 !important;
    position: relative !important;
}

/* ── Expanders ────────────────────────────────────────────────────────────── */
.stExpander {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

.stExpander:hover {
    border-color: rgba(108, 138, 255, 0.25) !important;
}

.stExpander summary {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    padding: 12px 16px !important;
}

.stExpander summary:hover {
    color: var(--text-primary) !important;
    background: rgba(108, 138, 255, 0.04) !important;
}

/* ── Info / Warning / Error boxes ────────────────────────────────────────── */
.stAlert {
    border-radius: 8px !important;
    border: 1px solid !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}

div[data-testid="stNotification"] {
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}

/* ── Text inputs ──────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 7px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 2px rgba(108, 138, 255, 0.15) !important;
    outline: none !important;
}

/* ── Sliders ──────────────────────────────────────────────────────────────── */
.stSlider > div > div > div {
    background: var(--border-subtle) !important;
}

.stSlider > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-primary), var(--accent-teal)) !important;
}

.stSlider [role="slider"] {
    background: var(--accent-primary) !important;
    border: 2px solid var(--bg-card) !important;
    box-shadow: 0 0 8px rgba(108, 138, 255, 0.4) !important;
}

/* ── Multiselect tags ─────────────────────────────────────────────────────── */
.stMultiSelect span[data-baseweb="tag"] {
    background: rgba(108, 138, 255, 0.2) !important;
    border: 1px solid rgba(108, 138, 255, 0.35) !important;
    border-radius: 4px !important;
    color: var(--accent-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
}

/* ── Download button ──────────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid var(--border-glow) !important;
    color: var(--accent-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
    border-radius: 6px !important;
    transition: all 0.15s ease !important;
}

.stDownloadButton > button:hover {
    background: rgba(108, 138, 255, 0.1) !important;
    box-shadow: 0 0 12px rgba(108, 138, 255, 0.2) !important;
}

/* ── Spinner ──────────────────────────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--accent-primary) !important;
    border-right-color: var(--accent-teal) !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 5px;
    height: 5px;
}
::-webkit-scrollbar-track {
    background: var(--bg-secondary);
}
::-webkit-scrollbar-thumb {
    background: var(--border-glow);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--accent-primary);
}

/* ── Status badge helper class ────────────────────────────────────────────── */
/* Use via st.markdown('<span class="badge badge-green">completed</span>', unsafe_allow_html=True) */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-green  { background: rgba(14,245,160,0.15); color: #0ef5a0; border: 1px solid rgba(14,245,160,0.3); }
.badge-amber  { background: rgba(245,166,35,0.15);  color: #f5a623; border: 1px solid rgba(245,166,35,0.3);  }
.badge-red    { background: rgba(255,77,106,0.15);  color: #ff4d6a; border: 1px solid rgba(255,77,106,0.3);  }
.badge-blue   { background: rgba(108,138,255,0.15); color: #6c8aff; border: 1px solid rgba(108,138,255,0.3); }

/* ── Symbol / ticker pill ─────────────────────────────────────────────────── */
/* Use via st.markdown('<span class="ticker">AAPL</span>', unsafe_allow_html=True) */
.ticker {
    display: inline-block;
    padding: 2px 8px;
    background: linear-gradient(135deg, rgba(108,138,255,0.15), rgba(14,245,200,0.08));
    border: 1px solid rgba(108,138,255,0.25);
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--accent-primary);
    letter-spacing: 0.05em;
}

/* ── Caption / small text ─────────────────────────────────────────────────── */
.stCaption, small {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
}

/* ── Form submit button ───────────────────────────────────────────────────── */
.stFormSubmitButton > button {
    background: linear-gradient(135deg, rgba(108,138,255,0.3), rgba(14,245,200,0.15)) !important;
    border: 1px solid var(--border-glow) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border-radius: 7px !important;
    transition: all 0.2s ease !important;
}

.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, rgba(108,138,255,0.45), rgba(14,245,200,0.25)) !important;
    box-shadow: 0 0 20px rgba(108,138,255,0.2) !important;
}

/* ── Page load fade-in ────────────────────────────────────────────────────── */
.main .block-container {
    animation: fadeIn 0.4s ease forwards;
    padding-top: 2rem !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Staggered section reveals ────────────────────────────────────────────── */
.main .block-container > div > div:nth-child(1) { animation-delay: 0.05s; }
.main .block-container > div > div:nth-child(2) { animation-delay: 0.10s; }
.main .block-container > div > div:nth-child(3) { animation-delay: 0.15s; }
.main .block-container > div > div:nth-child(4) { animation-delay: 0.20s; }

</style>
"""
