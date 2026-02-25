"""
Dashboard UI Styles - Enhanced Terminal Aesthetic
Drop-in replacement for dashboard_styles.py
Usage: from dashboard_styles import DASHBOARD_CSS
       st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
"""

DASHBOARD_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600;700&family=Orbitron:wght@400;500;600;700;800;900&display=swap');

/* ── CSS Variables ────────────────────────────────────────────────────────── */
:root {
    --bg-void:         #030508;
    --bg-deep:         #070c12;
    --bg-base:         #0b1118;
    --bg-surface:      #0f1820;
    --bg-elevated:     #141f2a;
    --bg-glass:        rgba(15, 24, 36, 0.7);
    --bg-glass-light:  rgba(20, 31, 42, 0.5);

    --neon-primary:    #00d4ff;
    --neon-secondary:  #00ff88;
    --neon-amber:      #ffb800;
    --neon-red:        #ff3860;
    --neon-purple:     #b060ff;

    --accent-blue:     #1a8cff;
    --accent-teal:     #00c896;
    --accent-gold:     #e8a020;

    --text-primary:    #e2ecf8;
    --text-secondary:  #6e8aaa;
    --text-muted:      #3a5070;
    --text-dim:        #243040;

    --border-glass:    rgba(0, 212, 255, 0.08);
    --border-glow:     rgba(0, 212, 255, 0.25);
    --border-bright:   rgba(0, 212, 255, 0.5);

    --font-display:    'Orbitron', sans-serif;
    --font-body:       'Space Grotesk', sans-serif;
    --font-mono:       'JetBrains Mono', monospace;

    --shadow-sm:       0 2px 8px rgba(0,0,0,0.4);
    --shadow-md:       0 4px 20px rgba(0,0,0,0.5);
    --shadow-lg:       0 8px 40px rgba(0,0,0,0.6);
    --glow-blue:       0 0 20px rgba(0, 212, 255, 0.15), 0 0 40px rgba(0, 212, 255, 0.05);
    --glow-green:      0 0 20px rgba(0, 255, 136, 0.15), 0 0 40px rgba(0, 255, 136, 0.05);
    --glow-red:        0 0 20px rgba(255, 56, 96, 0.15), 0 0 40px rgba(255, 56, 96, 0.05);
}

/* ── Animated Background ──────────────────────────────────────────────────── */
.stApp {
    background-color: var(--bg-void) !important;
    background-image:
        radial-gradient(ellipse 120% 80% at 10% -10%, rgba(0, 100, 200, 0.07) 0%, transparent 50%),
        radial-gradient(ellipse 80% 60% at 90% 110%, rgba(0, 196, 150, 0.05) 0%, transparent 50%),
        radial-gradient(ellipse 60% 80% at 50% 50%, rgba(26, 140, 255, 0.02) 0%, transparent 60%);
    animation: bgPulse 12s ease-in-out infinite alternate;
    font-family: var(--font-mono);
}

@keyframes bgPulse {
    0%   { background-position: 0% 0%, 100% 100%, 50% 50%; }
    100% { background-position: 5% 5%, 95% 95%, 52% 48%; }
}

/* Grid overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0, 212, 255, 0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 212, 255, 0.015) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* Scanline overlay */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 0, 0, 0.03) 2px,
        rgba(0, 0, 0, 0.03) 4px
    );
    pointer-events: none;
    z-index: 0;
}

/* ── Main Container ───────────────────────────────────────────────────────── */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px !important;
    animation: containerReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

@keyframes containerReveal {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Typography ───────────────────────────────────────────────────────────── */
h1 {
    font-family: var(--font-display) !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg, var(--neon-primary) 0%, #4488ff 40%, var(--neon-secondary) 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    text-shadow: none !important;
    filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.3));
    position: relative;
}

h2 {
    font-family: var(--font-display) !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--neon-primary) !important;
}

h3 {
    font-family: var(--font-body) !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    color: var(--text-primary) !important;
    padding-left: 10px !important;
    border-left: 2px solid var(--neon-primary) !important;
}

p, li, span {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}

/* ── Metric Cards ─────────────────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: var(--bg-glass) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--border-glass) !important;
    border-top: 1px solid rgba(0, 212, 255, 0.12) !important;
    border-radius: 6px !important;
    padding: 18px 20px 16px !important;  /* more vertical room */
    position: relative !important;
    overflow: hidden !important;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
    min-height: 100px !important;        /* ensures consistent card height */
}

/* Top accent line */
div[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--neon-primary), transparent);
    opacity: 0.5;
    transition: opacity 0.2s;
}

/* Corner bracket decoration */
div[data-testid="metric-container"]::after {
    content: '';
    position: absolute;
    bottom: 6px; right: 8px;
    width: 16px; height: 16px;
    border-right: 1px solid rgba(0, 212, 255, 0.2);
    border-bottom: 1px solid rgba(0, 212, 255, 0.2);
    transition: opacity 0.2s;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    border-color: var(--border-glow) !important;
    box-shadow: var(--glow-blue), var(--shadow-md) !important;
}

div[data-testid="metric-container"]:hover::before {
    opacity: 1;
}

div[data-testid="stMetricLabel"] > div {
    font-family: var(--font-mono) !important;
    font-size: 0.58rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    margin-bottom: 4px !important;
}

div[data-testid="stMetricValue"] > div {
    font-family: var(--font-mono) !important;
    font-size: 2.1rem !important;       /* ↑ was 1.55rem — big, scannable */
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.03em !important;
    line-height: 1.1 !important;
    text-shadow: 0 0 30px rgba(226, 236, 248, 0.08) !important;
}

div[data-testid="stMetricDelta"] > div {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;      /* ↑ slightly bigger delta too */
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    margin-top: 2px !important;
}

/* Green metrics */
div[data-testid="metric-container"]:has([data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Up"]) {
    border-top-color: rgba(0, 255, 136, 0.3) !important;
    border-color: rgba(0, 255, 136, 0.1) !important;
}
div[data-testid="metric-container"]:has([data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Up"])::before {
    background: linear-gradient(90deg, transparent, var(--neon-secondary), transparent);
}
div[data-testid="metric-container"]:has([data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Up"]):hover {
    box-shadow: var(--glow-green), var(--shadow-md) !important;
}

/* Red metrics */
div[data-testid="metric-container"]:has([data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Down"]) {
    border-top-color: rgba(255, 56, 96, 0.3) !important;
    border-color: rgba(255, 56, 96, 0.1) !important;
}
div[data-testid="metric-container"]:has([data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Down"])::before {
    background: linear-gradient(90deg, transparent, var(--neon-red), transparent);
}
div[data-testid="metric-container"]:has([data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Down"]):hover {
    box-shadow: var(--glow-red), var(--shadow-md) !important;
}

/* ── Tab Bar ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(7, 12, 18, 0.8) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 6px !important;
    padding: 5px !important;
    gap: 2px !important;
    border: 1px solid var(--border-glass) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.02) !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 4px !important;
    color: var(--text-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 8px 18px !important;
    border: none !important;
    transition: color 0.15s ease, background 0.15s ease !important;
    position: relative !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-secondary) !important;
    background: rgba(0, 212, 255, 0.04) !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(0, 212, 255, 0.06) !important;
    color: var(--neon-primary) !important;
    box-shadow:
        inset 0 0 0 1px rgba(0, 212, 255, 0.2),
        0 0 12px rgba(0, 212, 255, 0.08) !important;
}

.stTabs [aria-selected="true"]::after {
    content: '';
    position: absolute;
    bottom: 0; left: 15%; right: 15%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--neon-primary), transparent);
    animation: tabGlow 0.3s ease forwards;
}

@keyframes tabGlow {
    from { opacity: 0; transform: scaleX(0.3); }
    to   { opacity: 1; transform: scaleX(1); }
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 24px !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button {
    background: rgba(10, 18, 28, 0.8) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 4px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
    position: relative !important;
    overflow: hidden !important;
    backdrop-filter: blur(8px) !important;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.06), transparent);
    transition: left 0.4s ease;
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:hover {
    color: var(--neon-primary) !important;
    border-color: var(--border-glow) !important;
    box-shadow: 0 0 16px rgba(0, 212, 255, 0.12), var(--shadow-sm) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0) scale(0.99) !important;
}

/* ── Selectbox ────────────────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stSelectbox [data-baseweb="select"] > div {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 4px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.stSelectbox > div > div:focus-within,
.stSelectbox [data-baseweb="select"] > div:focus-within {
    border-color: rgba(0, 212, 255, 0.4) !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.08), 0 0 12px rgba(0, 212, 255, 0.08) !important;
}

/* ── Dataframes / Tables ──────────────────────────────────────────────────── */
.stDataFrame {
    border: 1px solid var(--border-glass) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-md) !important;
}

.stDataFrame iframe {
    background: var(--bg-surface) !important;
}

/* ── Dividers ─────────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(
        90deg,
        transparent 0%,
        rgba(0, 212, 255, 0.2) 20%,
        rgba(0, 212, 255, 0.4) 50%,
        rgba(0, 212, 255, 0.2) 80%,
        transparent 100%
    ) !important;
    margin: 28px 0 !important;
    position: relative !important;
}

hr::after {
    content: '◆';
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    color: rgba(0, 212, 255, 0.3);
    font-size: 6px;
    background: var(--bg-void);
    padding: 0 4px;
}

/* ── Expanders ────────────────────────────────────────────────────────────── */
.stExpander {
    background: var(--bg-glass) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    transition: border-color 0.2s ease !important;
}

.stExpander:hover {
    border-color: rgba(0, 212, 255, 0.15) !important;
}

/* Fix arrow + label overlap — force flex layout with a proper gap */
.stExpander summary {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.04em !important;
    padding: 13px 16px !important;
    cursor: pointer !important;
    list-style: none !important; /* removes default marker in some browsers */
    user-select: none !important;
}

/* The chevron/arrow SVG that Streamlit injects */
.stExpander summary svg {
    flex-shrink: 0 !important;
    width: 14px !important;
    height: 14px !important;
    color: var(--text-muted) !important;
    transition: transform 0.2s ease, color 0.2s ease !important;
}

/* Rotate arrow when open */
details[open] .stExpander summary svg,
.stExpander details[open] > summary svg {
    transform: rotate(90deg) !important;
}

/* The bold label text Streamlit wraps in a <p> or <span> inside summary */
.stExpander summary p,
.stExpander summary span,
.stExpander summary > div {
    margin: 0 !important;
    padding: 0 !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: inherit !important;
    line-height: 1.4 !important;
}

.stExpander summary:hover {
    color: var(--text-primary) !important;
}

.stExpander summary:hover svg {
    color: var(--neon-primary) !important;
}

/* ── Alerts / Info boxes ──────────────────────────────────────────────────── */
.stAlert {
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    backdrop-filter: blur(8px) !important;
}

div[data-testid="stNotification"] {
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}

/* Info alert → cyan tint */
div[data-baseweb="notification"][kind="info"],
div[data-testid="stAlert"] {
    background: rgba(0, 90, 140, 0.15) !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
}

/* ── Text inputs ──────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 4px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stNumberInput > div > div > input:focus {
    border-color: rgba(0, 212, 255, 0.4) !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.08), 0 0 16px rgba(0, 212, 255, 0.06) !important;
    outline: none !important;
}

/* ── Sliders ──────────────────────────────────────────────────────────────── */
.stSlider > div > div > div {
    background: rgba(0, 212, 255, 0.1) !important;
}

.stSlider > div > div > div > div {
    background: linear-gradient(90deg, var(--neon-primary), var(--neon-secondary)) !important;
}

.stSlider [role="slider"] {
    background: var(--neon-primary) !important;
    border: 2px solid var(--bg-base) !important;
    box-shadow: 0 0 12px rgba(0, 212, 255, 0.5) !important;
    transition: box-shadow 0.2s ease !important;
}

.stSlider [role="slider"]:hover {
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.8) !important;
}

/* ── Multiselect tags ─────────────────────────────────────────────────────── */
.stMultiSelect span[data-baseweb="tag"] {
    background: rgba(0, 212, 255, 0.08) !important;
    border: 1px solid rgba(0, 212, 255, 0.25) !important;
    border-radius: 3px !important;
    color: var(--neon-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Download button ──────────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid var(--border-glow) !important;
    color: var(--neon-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
}

.stDownloadButton > button:hover {
    background: rgba(0, 212, 255, 0.06) !important;
    box-shadow: 0 0 16px rgba(0, 212, 255, 0.2) !important;
    transform: translateY(-1px) !important;
}

/* ── Spinner ──────────────────────────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--neon-primary) !important;
    border-right-color: rgba(0, 212, 255, 0.3) !important;
    border-bottom-color: transparent !important;
    border-left-color: transparent !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(5, 10, 18, 0.95) !important;
    border-right: 1px solid var(--border-glass) !important;
    backdrop-filter: blur(20px) !important;
}

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb {
    background: rgba(0, 212, 255, 0.2);
    border-radius: 2px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 212, 255, 0.4);
    box-shadow: 0 0 6px rgba(0, 212, 255, 0.4);
}

/* ── Badge helper ─────────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 2px;
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.badge-green  {
    background: rgba(0,255,136,0.06);
    color: var(--neon-secondary);
    border: 1px solid rgba(0,255,136,0.25);
    box-shadow: 0 0 8px rgba(0,255,136,0.1);
}
.badge-amber  {
    background: rgba(255,184,0,0.06);
    color: var(--neon-amber);
    border: 1px solid rgba(255,184,0,0.25);
    box-shadow: 0 0 8px rgba(255,184,0,0.1);
}
.badge-red    {
    background: rgba(255,56,96,0.06);
    color: var(--neon-red);
    border: 1px solid rgba(255,56,96,0.25);
    box-shadow: 0 0 8px rgba(255,56,96,0.1);
}
.badge-blue   {
    background: rgba(0,212,255,0.06);
    color: var(--neon-primary);
    border: 1px solid rgba(0,212,255,0.25);
    box-shadow: 0 0 8px rgba(0,212,255,0.1);
}

/* ── Ticker pill ──────────────────────────────────────────────────────────── */
.ticker {
    display: inline-block;
    padding: 2px 8px;
    background: rgba(0, 212, 255, 0.06);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--neon-primary);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Status / Caption text ────────────────────────────────────────────────── */
.stCaption, small {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.05em !important;
}

/* ── Section header with glow line ───────────────────────────────────────── */
/* Usage: st.markdown('<div class="section-header">Title</div>', unsafe_allow_html=True) */
.section-header {
    font-family: var(--font-display);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 4px 0 12px;
    border-bottom: 1px solid var(--border-glass);
    margin-bottom: 16px;
    position: relative;
}
.section-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    width: 40px; height: 1px;
    background: var(--neon-primary);
    box-shadow: 0 0 8px var(--neon-primary);
}

/* ── Data card ────────────────────────────────────────────────────────────── */
/* Usage: st.markdown('<div class="data-card">...</div>', unsafe_allow_html=True) */
.data-card {
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: 6px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.data-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--neon-primary), transparent);
    opacity: 0.4;
}
.data-card:hover {
    border-color: var(--border-glow);
    transform: translateY(-2px);
    box-shadow: var(--glow-blue);
}

/* ── Status indicator dot ─────────────────────────────────────────────────── */
/* Usage: <span class="status-dot live"></span> */
.status-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.status-dot.live    { background: var(--neon-secondary); box-shadow: 0 0 6px var(--neon-secondary); animation: statusPulse 2s ease-in-out infinite; }
.status-dot.warning { background: var(--neon-amber);     box-shadow: 0 0 6px var(--neon-amber); }
.status-dot.error   { background: var(--neon-red);       box-shadow: 0 0 6px var(--neon-red); animation: statusPulse 1s ease-in-out infinite; }
.status-dot.idle    { background: var(--text-muted); }

@keyframes statusPulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
}

/* ── Number highlight ─────────────────────────────────────────────────────── */
/* Usage: <span class="num positive">+12.4%</span> */
.num          { font-family: var(--font-mono); font-weight: 600; font-size: 0.9rem; }
.num.positive { color: var(--neon-secondary); text-shadow: 0 0 8px rgba(0,255,136,0.3); }
.num.negative { color: var(--neon-red);       text-shadow: 0 0 8px rgba(255,56,96,0.3); }
.num.neutral  { color: var(--neon-primary); }

/* ── Form submit ──────────────────────────────────────────────────────────── */
.stFormSubmitButton > button {
    background: rgba(0, 212, 255, 0.06) !important;
    border: 1px solid var(--border-glow) !important;
    color: var(--neon-primary) !important;
    font-family: var(--font-mono) !important;
    font-weight: 600 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
}

.stFormSubmitButton > button:hover {
    background: rgba(0, 212, 255, 0.12) !important;
    box-shadow: 0 0 24px rgba(0, 212, 255, 0.2) !important;
    transform: translateY(-1px) !important;
}

/* ── Staggered entry animations ───────────────────────────────────────────── */
.main .block-container > div > div:nth-child(1) { animation: slideIn 0.5s cubic-bezier(0.16,1,0.3,1) 0.05s both; }
.main .block-container > div > div:nth-child(2) { animation: slideIn 0.5s cubic-bezier(0.16,1,0.3,1) 0.10s both; }
.main .block-container > div > div:nth-child(3) { animation: slideIn 0.5s cubic-bezier(0.16,1,0.3,1) 0.15s both; }
.main .block-container > div > div:nth-child(4) { animation: slideIn 0.5s cubic-bezier(0.16,1,0.3,1) 0.20s both; }
.main .block-container > div > div:nth-child(5) { animation: slideIn 0.5s cubic-bezier(0.16,1,0.3,1) 0.25s both; }

@keyframes slideIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Plotly chart containers ──────────────────────────────────────────────── */
div[data-testid="stPlotlyChart"] {
    border-radius: 6px !important;
    overflow: hidden !important;
    border: 1px solid var(--border-glass) !important;
    transition: border-color 0.2s ease !important;
}
div[data-testid="stPlotlyChart"]:hover {
    border-color: rgba(0, 212, 255, 0.15) !important;
}

/* ── Number input ─────────────────────────────────────────────────────────── */
.stNumberInput button {
    background: var(--bg-elevated) !important;
    border-color: var(--border-glass) !important;
    color: var(--text-secondary) !important;
}
.stNumberInput button:hover {
    background: rgba(0, 212, 255, 0.08) !important;
    color: var(--neon-primary) !important;
}

/* ── Warning / Error / Success ────────────────────────────────────────────── */
div[data-testid="stAlert"][data-baseweb="notification"] {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    border-radius: 6px !important;
}

/* ── Checkbox & Radio ─────────────────────────────────────────────────────── */
.stCheckbox label, .stRadio label {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    color: var(--text-secondary) !important;
}

/* ── Toast notifications ──────────────────────────────────────────────────── */
div[data-testid="toastContainer"] {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}

/* ── Progress bar ─────────────────────────────────────────────────────────── */
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--neon-primary), var(--neon-secondary)) !important;
    box-shadow: 0 0 8px rgba(0, 212, 255, 0.4) !important;
}
div[data-testid="stProgressBar"] > div {
    background: rgba(0, 212, 255, 0.08) !important;
    border-radius: 2px !important;
}

/* ── Column containers — subtle separator on hover ────────────────────────── */
[data-testid="column"]:has(div[data-testid="metric-container"]) {
    transition: filter 0.2s ease;
}

</style>
"""
