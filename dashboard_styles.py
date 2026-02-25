"""
dashboard_styles.py — Enhanced Terminal Aesthetic v3
Changes v3:
  - Added bulletproof CSS to suppress .arrow_right / raw SVG text that leaks
    from stExpanderToggleIcon in certain Streamlit builds
  - All other styles unchanged from v2
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
    --glow-blue:       0 0 20px rgba(0, 212, 255, 0.12), 0 0 40px rgba(0, 212, 255, 0.04);
    --glow-green:      0 0 20px rgba(0, 255, 136, 0.12), 0 0 40px rgba(0, 255, 136, 0.04);
    --glow-red:        0 0 20px rgba(255, 56, 96, 0.12), 0 0 40px rgba(255, 56, 96, 0.04);
}

/* ── Animated Background ──────────────────────────────────────────────────── */
.stApp {
    background-color: var(--bg-void) !important;
    background-image:
        radial-gradient(ellipse 120% 80% at 10% -10%, rgba(0, 100, 200, 0.06) 0%, transparent 50%),
        radial-gradient(ellipse 80% 60% at 90% 110%, rgba(0, 196, 150, 0.04) 0%, transparent 50%),
        radial-gradient(ellipse 60% 80% at 50% 50%, rgba(26, 140, 255, 0.02) 0%, transparent 60%);
    animation: bgPulse 16s ease-in-out infinite alternate;
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
        linear-gradient(rgba(0, 212, 255, 0.008) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 212, 255, 0.008) 1px, transparent 1px);
    background-size: 56px 56px;
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
        transparent 3px,
        rgba(0, 0, 0, 0.015) 3px,
        rgba(0, 0, 0, 0.015) 4px
    );
    pointer-events: none;
    z-index: 0;
}

/* ── Main Container ───────────────────────────────────────────────────────── */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1440px !important;
    animation: containerReveal 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

@keyframes containerReveal {
    from { opacity: 0; transform: translateY(10px); }
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
    filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.25));
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
    font-size: 0.9rem !important;
}

/* ── Metric Cards ─────────────────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: var(--bg-glass) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--border-glass) !important;
    border-top: 1px solid rgba(0, 212, 255, 0.1) !important;
    border-radius: 6px !important;
    padding: 16px 18px 14px !important;
    position: relative !important;
    overflow: hidden !important;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
    min-height: 90px !important;
}

div[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--neon-primary), transparent);
    opacity: 0.4;
    transition: opacity 0.2s;
}

div[data-testid="metric-container"]::after {
    content: '';
    position: absolute;
    bottom: 5px; right: 7px;
    width: 14px; height: 14px;
    border-right: 1px solid rgba(0, 212, 255, 0.18);
    border-bottom: 1px solid rgba(0, 212, 255, 0.18);
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    border-color: var(--border-glow) !important;
    box-shadow: var(--glow-blue), var(--shadow-md) !important;
}

div[data-testid="metric-container"]:hover::before { opacity: 1; }

div[data-testid="stMetricLabel"] > div {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    margin-bottom: 4px !important;
}

div[data-testid="stMetricValue"] > div {
    font-family: var(--font-mono) !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.03em !important;
    line-height: 1.1 !important;
}

div[data-testid="stMetricDelta"] > div {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    margin-top: 2px !important;
}

/* Green metrics */
div[data-testid="metric-container"]:has([data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Up"]) {
    border-top-color: rgba(0, 255, 136, 0.25) !important;
    border-color: rgba(0, 255, 136, 0.08) !important;
}
div[data-testid="metric-container"]:has([data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Up"])::before {
    background: linear-gradient(90deg, transparent, var(--neon-secondary), transparent);
}
div[data-testid="metric-container"]:has([data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Up"]):hover {
    box-shadow: var(--glow-green), var(--shadow-md) !important;
}

/* Red metrics */
div[data-testid="metric-container"]:has([data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Down"]) {
    border-top-color: rgba(255, 56, 96, 0.25) !important;
    border-color: rgba(255, 56, 96, 0.08) !important;
}
div[data-testid="metric-container"]:has([data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Down"])::before {
    background: linear-gradient(90deg, transparent, var(--neon-red), transparent);
}
div[data-testid="metric-container"]:has([data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Down"]):hover {
    box-shadow: var(--glow-red), var(--shadow-md) !important;
}

/* ── SMALL metric variant ─────────────────────────────────────────────────── */
.metrics-sm div[data-testid="metric-container"] {
    min-height: 68px !important;
    padding: 10px 12px 8px !important;
}
.metrics-sm div[data-testid="stMetricValue"] > div {
    font-size: 1.35rem !important;
}
.metrics-sm div[data-testid="stMetricLabel"] > div {
    font-size: 0.6rem !important;
}

/* ── Tab Bar ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(7, 12, 18, 0.85) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 6px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid var(--border-glass) !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 4px !important;
    color: var(--text-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 7px 16px !important;
    border: none !important;
    transition: color 0.15s ease, background 0.15s ease !important;
    position: relative !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-secondary) !important;
    background: rgba(0, 212, 255, 0.04) !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(0, 212, 255, 0.07) !important;
    color: var(--neon-primary) !important;
    box-shadow: inset 0 0 0 1px rgba(0, 212, 255, 0.2), 0 0 10px rgba(0, 212, 255, 0.06) !important;
}

.stTabs [aria-selected="true"]::after {
    content: '';
    position: absolute;
    bottom: 0; left: 15%; right: 15%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--neon-primary), transparent);
    animation: tabGlow 0.25s ease forwards;
}

@keyframes tabGlow {
    from { opacity: 0; transform: scaleX(0.3); }
    to   { opacity: 1; transform: scaleX(1); }
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 20px !important;
}

/* ── Buttons — base ───────────────────────────────────────────────────────── */
.stButton > button {
    background: rgba(10, 18, 28, 0.8) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 4px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 7px 18px !important;
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
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.05), transparent);
    transition: left 0.35s ease;
}

.stButton > button:hover::before { left: 100%; }

.stButton > button:hover {
    color: var(--neon-primary) !important;
    border-color: var(--border-glow) !important;
    box-shadow: 0 0 14px rgba(0, 212, 255, 0.1), var(--shadow-sm) !important;
    transform: translateY(-1px) !important;
}

/* ── Refresh button — cyan accent ─────────────────────────────────────────── */
.btn-refresh .stButton > button {
    border-color: rgba(0, 212, 255, 0.3) !important;
    color: var(--neon-primary) !important;
}

/* ── Clear Cache button — red danger ──────────────────────────────────────── */
.btn-danger .stButton > button {
    border-color: rgba(255, 56, 96, 0.25) !important;
    color: #ff6080 !important;
    background: rgba(255, 56, 96, 0.04) !important;
}
.btn-danger .stButton > button:hover {
    color: var(--neon-red) !important;
    border-color: rgba(255, 56, 96, 0.5) !important;
    box-shadow: 0 0 14px rgba(255, 56, 96, 0.15) !important;
}

/* ── Run / Action button — green accent ───────────────────────────────────── */
.btn-action .stButton > button {
    border-color: rgba(0, 255, 136, 0.25) !important;
    color: var(--neon-secondary) !important;
    background: rgba(0, 255, 136, 0.04) !important;
}
.btn-action .stButton > button:hover {
    box-shadow: 0 0 14px rgba(0, 255, 136, 0.15) !important;
}

/* ── Selectbox ────────────────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stSelectbox [data-baseweb="select"] > div {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 4px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.stSelectbox > div > div:focus-within,
.stSelectbox [data-baseweb="select"] > div:focus-within {
    border-color: rgba(0, 212, 255, 0.35) !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.06), 0 0 10px rgba(0, 212, 255, 0.06) !important;
}

/* ── Dataframes / Tables ──────────────────────────────────────────────────── */
.stDataFrame {
    border: 1px solid var(--border-glass) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-md) !important;
}

.stDataFrame iframe { background: var(--bg-surface) !important; }

/* ── Dividers ─────────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(
        90deg,
        transparent 0%,
        rgba(0, 212, 255, 0.15) 20%,
        rgba(0, 212, 255, 0.35) 50%,
        rgba(0, 212, 255, 0.15) 80%,
        transparent 100%
    ) !important;
    margin: 24px 0 !important;
    position: relative !important;
}
hr::after {
    content: '◆';
    position: absolute;
    left: 50%; top: 50%;
    transform: translate(-50%, -50%);
    color: rgba(0, 212, 255, 0.25);
    font-size: 6px;
    background: var(--bg-void);
    padding: 0 4px;
}

/* ── Expanders ────────────────────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    background: var(--bg-glass) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    transition: border-color 0.2s ease !important;
    margin-bottom: 8px !important;
}
div[data-testid="stExpander"]:hover {
    border-color: rgba(0, 212, 255, 0.16) !important;
}
div[data-testid="stExpander"] > details > summary,
div[data-testid="stExpander"] details summary {
    display: flex !important;
    flex-direction: row !important;
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
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.04em !important;
    line-height: 1.4 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* ── Expander toggle: nuke any leaked .arrow_right text, keep SVG ─────────── */
/* Streamlit injects the SVG icon name as a text node in some builds.          */
/* Strategy: hide everything inside the toggle div, un-hide only the SVG.      */
div[data-testid="stExpanderToggleIcon"] {
    font-size: 0 !important;
    line-height: 0 !important;
    color: transparent !important;
    display: flex !important;
    align-items: center !important;
    width: 18px !important;
    height: 18px !important;
    overflow: hidden !important;
}
div[data-testid="stExpanderToggleIcon"] * {
    font-size: 0 !important;
    color: transparent !important;
}
div[data-testid="stExpanderToggleIcon"] svg {
    font-size: initial !important;
    color: var(--text-muted) !important;
    width: 14px !important;
    height: 14px !important;
    display: block !important;
    visibility: visible !important;
    flex-shrink: 0 !important;
    transition: transform 0.22s ease, color 0.18s ease !important;
}
div[data-testid="stExpander"] details[open] div[data-testid="stExpanderToggleIcon"] svg {
    transform: rotate(90deg) !important;
    color: var(--neon-primary) !important;
}
div[data-testid="stExpander"] details summary:hover div[data-testid="stExpanderToggleIcon"] svg {
    color: var(--neon-primary) !important;
}
div[data-testid="stExpander"] details > div { padding: 4px 15px 14px !important; }

/* ── Alerts ───────────────────────────────────────────────────────────────── */
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
div[data-baseweb="notification"][kind="info"],
div[data-testid="stAlert"] {
    background: rgba(0, 90, 140, 0.12) !important;
    border: 1px solid rgba(0, 212, 255, 0.18) !important;
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
    font-size: 0.88rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stNumberInput > div > div > input:focus {
    border-color: rgba(0, 212, 255, 0.35) !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.06), 0 0 14px rgba(0, 212, 255, 0.05) !important;
    outline: none !important;
}

/* ── Search bar — prominent variant ──────────────────────────────────────── */
.search-bar .stTextInput > div > div > input {
    background: rgba(0, 212, 255, 0.04) !important;
    border-color: rgba(0, 212, 255, 0.2) !important;
    font-size: 0.9rem !important;
    padding: 10px 14px !important;
    letter-spacing: 0.05em !important;
}
.search-bar .stTextInput > div > div > input:focus {
    border-color: rgba(0, 212, 255, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.06), 0 0 20px rgba(0, 212, 255, 0.08) !important;
}

/* ── Sliders ──────────────────────────────────────────────────────────────── */
.stSlider > div > div > div { background: rgba(0, 212, 255, 0.08) !important; }
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, var(--neon-primary), var(--neon-secondary)) !important;
}
.stSlider [role="slider"] {
    background: var(--neon-primary) !important;
    border: 2px solid var(--bg-base) !important;
    box-shadow: 0 0 10px rgba(0, 212, 255, 0.4) !important;
}

/* ── Multiselect tags ─────────────────────────────────────────────────────── */
.stMultiSelect span[data-baseweb="tag"] {
    background: rgba(0, 212, 255, 0.07) !important;
    border: 1px solid rgba(0, 212, 255, 0.22) !important;
    border-radius: 3px !important;
    color: var(--neon-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.66rem !important;
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
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: rgba(0, 212, 255, 0.05) !important;
    box-shadow: 0 0 14px rgba(0, 212, 255, 0.18) !important;
    transform: translateY(-1px) !important;
}

/* ── Spinner ──────────────────────────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--neon-primary) !important;
    border-right-color: rgba(0, 212, 255, 0.25) !important;
    border-bottom-color: transparent !important;
    border-left-color: transparent !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(5, 10, 18, 0.97) !important;
    border-right: 1px solid var(--border-glass) !important;
    backdrop-filter: blur(20px) !important;
}

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: rgba(0, 212, 255, 0.18); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0, 212, 255, 0.35); }

/* ── Skeleton loader ──────────────────────────────────────────────────────── */
.skeleton {
    background: linear-gradient(
        90deg,
        rgba(0, 212, 255, 0.03) 0%,
        rgba(0, 212, 255, 0.07) 40%,
        rgba(0, 212, 255, 0.03) 80%
    );
    background-size: 200% 100%;
    animation: skeletonShimmer 1.6s ease-in-out infinite;
    border-radius: 6px;
    border: 1px solid rgba(0, 212, 255, 0.06);
    width: 100%;
}
@keyframes skeletonShimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

.skeleton-row {
    display: flex;
    gap: 12px;
    margin-bottom: 12px;
}

/* ── Badge helper ─────────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 2px;
    font-family: var(--font-mono);
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.badge-green  { background: rgba(0,255,136,0.06); color: var(--neon-secondary); border: 1px solid rgba(0,255,136,0.22); }
.badge-amber  { background: rgba(255,184,0,0.06);  color: var(--neon-amber);     border: 1px solid rgba(255,184,0,0.22); }
.badge-red    { background: rgba(255,56,96,0.06);  color: var(--neon-red);       border: 1px solid rgba(255,56,96,0.22); }
.badge-blue   { background: rgba(0,212,255,0.06);  color: var(--neon-primary);   border: 1px solid rgba(0,212,255,0.22); }

/* ── Ticker pill ──────────────────────────────────────────────────────────── */
.ticker {
    display: inline-block;
    padding: 2px 8px;
    background: rgba(0, 212, 255, 0.05);
    border: 1px solid rgba(0, 212, 255, 0.18);
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--neon-primary);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Status indicator dot ─────────────────────────────────────────────────── */
.status-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    vertical-align: middle;
}
.status-dot.live    { background: var(--neon-secondary); box-shadow: 0 0 5px var(--neon-secondary); animation: statusPulse 4s ease-in-out infinite; }
.status-dot.warning { background: var(--neon-amber);     box-shadow: 0 0 5px var(--neon-amber); }
.status-dot.error   { background: var(--neon-red);       box-shadow: 0 0 5px var(--neon-red);   animation: statusPulse 2s ease-in-out infinite; }
.status-dot.idle    { background: var(--text-muted); }

@keyframes statusPulse {
    0%, 100% { opacity: 1;   transform: scale(1); }
    50%       { opacity: 0.7; transform: scale(0.85); }
}

/* ── Number highlight ─────────────────────────────────────────────────────── */
.num          { font-family: var(--font-mono); font-weight: 600; font-size: 0.9rem; }
.num.positive { color: var(--neon-secondary); text-shadow: 0 0 7px rgba(0,255,136,0.25); }
.num.negative { color: var(--neon-red);       text-shadow: 0 0 7px rgba(255,56,96,0.25); }
.num.neutral  { color: var(--neon-primary); }

/* ── Section header with glow line ───────────────────────────────────────── */
.section-header {
    font-family: var(--font-display);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 3px 0 11px;
    border-bottom: 1px solid var(--border-glass);
    margin-bottom: 14px;
    position: relative;
}
.section-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    width: 36px; height: 1px;
    background: var(--neon-primary);
    box-shadow: 0 0 7px var(--neon-primary);
}

/* ── Data card ────────────────────────────────────────────────────────────── */
.data-card {
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: 6px;
    padding: 15px 18px;
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
    opacity: 0.35;
}
.data-card:hover { border-color: var(--border-glow); transform: translateY(-2px); box-shadow: var(--glow-blue); }

/* ── Symbol search result card ────────────────────────────────────────────── */
.search-result-card {
    background: rgba(0, 212, 255, 0.03);
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-left: 3px solid var(--neon-primary);
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-family: var(--font-mono);
}

/* ── Preset chips ─────────────────────────────────────────────────────────── */
.preset-chips {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 12px;
}
.chip {
    padding: 3px 12px;
    border-radius: 12px;
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.15s ease;
    background: rgba(0, 212, 255, 0.05);
    border: 1px solid rgba(0, 212, 255, 0.18);
    color: var(--text-secondary);
}
.chip:hover { background: rgba(0, 212, 255, 0.1); border-color: rgba(0, 212, 255, 0.35); color: var(--neon-primary); }
.chip.active { background: rgba(0, 212, 255, 0.1); border-color: rgba(0, 212, 255, 0.4); color: var(--neon-primary); }

/* ── Form submit ──────────────────────────────────────────────────────────── */
.stFormSubmitButton > button {
    background: rgba(0, 212, 255, 0.05) !important;
    border: 1px solid var(--border-glow) !important;
    color: var(--neon-primary) !important;
    font-family: var(--font-mono) !important;
    font-weight: 600 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
}
.stFormSubmitButton > button:hover {
    background: rgba(0, 212, 255, 0.1) !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.18) !important;
    transform: translateY(-1px) !important;
}

/* ── Staggered entry animations ───────────────────────────────────────────── */
.main .block-container > div > div:nth-child(1) { animation: slideIn 0.45s cubic-bezier(0.16,1,0.3,1) 0.04s both; }
.main .block-container > div > div:nth-child(2) { animation: slideIn 0.45s cubic-bezier(0.16,1,0.3,1) 0.08s both; }
.main .block-container > div > div:nth-child(3) { animation: slideIn 0.45s cubic-bezier(0.16,1,0.3,1) 0.12s both; }
.main .block-container > div > div:nth-child(4) { animation: slideIn 0.45s cubic-bezier(0.16,1,0.3,1) 0.16s both; }
.main .block-container > div > div:nth-child(5) { animation: slideIn 0.45s cubic-bezier(0.16,1,0.3,1) 0.20s both; }

@keyframes slideIn {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Plotly chart containers ──────────────────────────────────────────────── */
div[data-testid="stPlotlyChart"] {
    border-radius: 6px !important;
    overflow: hidden !important;
    border: 1px solid var(--border-glass) !important;
    transition: border-color 0.2s ease !important;
}
div[data-testid="stPlotlyChart"]:hover { border-color: rgba(0, 212, 255, 0.14) !important; }

/* ── Number input ─────────────────────────────────────────────────────────── */
.stNumberInput button {
    background: var(--bg-elevated) !important;
    border-color: var(--border-glass) !important;
    color: var(--text-secondary) !important;
}
.stNumberInput button:hover {
    background: rgba(0, 212, 255, 0.07) !important;
    color: var(--neon-primary) !important;
}

/* ── Checkbox & Radio ─────────────────────────────────────────────────────── */
.stCheckbox label, .stRadio label {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    color: var(--text-secondary) !important;
}

/* ── Progress bar ─────────────────────────────────────────────────────────── */
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--neon-primary), var(--neon-secondary)) !important;
    box-shadow: 0 0 7px rgba(0, 212, 255, 0.35) !important;
}
div[data-testid="stProgressBar"] > div {
    background: rgba(0, 212, 255, 0.07) !important;
    border-radius: 2px !important;
}

/* ── Warning banner for cache clear ──────────────────────────────────────── */
.cache-warning {
    background: rgba(255, 56, 96, 0.07);
    border: 1px solid rgba(255, 56, 96, 0.25);
    border-left: 3px solid var(--neon-red);
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-family: var(--font-mono);
    font-size: 0.76rem;
    color: #ff8090;
    margin-bottom: 10px;
}

/* ── Caption / small ──────────────────────────────────────────────────────── */
.stCaption, small {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.05em !important;
}

/* ── Toast ────────────────────────────────────────────────────────────────── */
div[data-testid="toastContainer"] {
    font-family: var(--font-mono) !important;
    font-size: 0.76rem !important;
}

/* ── Column containers ────────────────────────────────────────────────────── */
[data-testid="column"]:has(div[data-testid="metric-container"]) {
    transition: filter 0.18s ease;
}

</style>
"""
