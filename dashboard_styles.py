"""
dashboard_styles.py — Terminal Noir v8 "Brass & Phosphor"

Full visual pass, prompted by the tab-bar being both low-contrast (a global
`p { color }` rule was silently beating the tab-selected color — fixed by
targeting text nodes explicitly instead of relying on inheritance) and
visually flat before interaction (unselected tabs had zero affordance).

Also repoints the brand accent from a generic cyan-on-near-black (the
default "AI dashboard" look) to a warm brass/gold — closer to the phosphor
glow of a real trading terminal / ticker display, which fits the brief
("Market Intelligence Terminal") rather than being a stock choice. Green/red
stay reserved purely for gain/loss, matching real finance convention, so the
accent never gets confused with a P&L signal.

Ambient signature: a slow vertical "scan" sweep + faint CRT scanline
texture across the whole app background, and HUD-style corner brackets on
card containers — quiet, not distracting, but unmistakably a terminal
rather than a generic dark dashboard.

--cyan / --cyan-dim / --cyan-border / --cyan-glow variable NAMES are kept
as-is (dozens of call sites across dashboard.py / tab_ml_predictions.py
reference them directly) — only their color VALUES changed, from cyan to
brass/gold.
"""

DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500;1,400&family=Syne:wght@600;700;800&display=swap');

:root {
    /* Backgrounds — clear, ordered steps */
    --bg-0:        #05070a;
    --bg-1:        #090c11;
    --bg-2:        #0f131a;
    --bg-3:        #161b24;
    --bg-4:        #1e2530;

    /* Accents — brass/gold "phosphor" accent (was cyan); names kept for
       backward compatibility with existing call sites, values repointed. */
    --cyan:        #e0a83c;
    --cyan-dim:    rgba(224, 168, 60, 0.12);
    --cyan-border: rgba(224, 168, 60, 0.34);
    --cyan-glow:   rgba(224, 168, 60, 0.16);
    --green:       #10b981;
    --green-bright:#34d399;
    --green-dim:   rgba(16, 185, 129, 0.12);
    --amber:       #f59e0b;
    --amber-bright:#fbbf24;
    --amber-dim:   rgba(245, 158, 11, 0.12);
    --red:         #ef4444;
    --red-bright:  #f87171;
    --red-dim:     rgba(239, 68, 68, 0.12);
    --purple:      #8b5cf6;
    --purple-dim:  rgba(139, 92, 246, 0.12);

    /* Text — real contrast at every step, checked against bg-2/bg-3 */
    --text-0:      #f8fafc;   /* headlines, values that matter */
    --text-1:      #cbd5e1;   /* body copy */
    --text-2:      #8695ab;   /* labels, captions — legible, not decorative */
    --text-3:      #4b5a70;   /* deliberately muted, used sparingly */

    --border:      rgba(255,255,255,0.08);
    --border-mid:  rgba(255,255,255,0.14);
    --border-focus:rgba(224, 168, 60, 0.55);

    --font-display:'Syne', sans-serif;
    --font-body:   'DM Mono', monospace;

    --radius:      10px;
    --radius-sm:   6px;
}

/* ── Base — quiet gradients + faint scanline texture + slow scan sweep ──── */
.stApp {
    background: var(--bg-0) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 0% 0%,   rgba(224,168,60,0.055) 0%, transparent 55%),
        radial-gradient(ellipse 60% 45% at 100% 100%, rgba(16,185,129,0.045) 0%, transparent 55%);
    font-family: var(--font-body);
    position: relative;
    z-index: 0; /* REQUIRED: position:relative alone does NOT create a stacking
       context — only relative/absolute + a non-auto z-index does. Without this,
       the .stApp::before/::after z-index:-1 below aren't actually scoped behind
       .stApp's own children; they escape to the nearest real stacking context
       (page root or a Streamlit wrapper), which can land them ABOVE the real
       content depending on Streamlit's internal DOM — i.e. exactly the
       "only the static is visible" bug. */
}

/* ── DIAGNOSTIC BUILD ──────────────────────────────────────────────────────
   .stApp::before / ::after (scanline texture + ambient sweep) TEMPORARILY
   REMOVED to test whether they are the actual cause of the invisible-
   dashboard bug. If the dashboard renders fine with this block gone, the
   effect will be rebuilt as a real fixed <div> (via st.markdown) instead of
   pseudo-elements, so it can never interfere with .stApp's own stacking
   context again. If the dashboard is STILL blank/static-only with this
   removed, the cause is something else entirely and these rules were never
   the problem. Original code kept below, commented out, for easy restore.

.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background-image: repeating-linear-gradient(
        0deg, rgba(255,255,255,0.014) 0px, rgba(255,255,255,0.014) 1px,
        transparent 1px, transparent 3px
    );
    pointer-events: none;
    z-index: -1;
}
.stApp::after {
    content: '';
    position: fixed; left: 0; right: 0;
    height: 34vh; top: -34vh;
    background: linear-gradient(180deg, transparent 0%, rgba(224,168,60,0.05) 50%, transparent 100%);
    pointer-events: none;
    z-index: -1;
    animation: scanSweep 11s linear infinite;
}
@keyframes scanSweep {
    0%   { transform: translateY(0); }
    100% { transform: translateY(394vh); }
}
@media (prefers-reduced-motion: reduce) {
    .stApp::after { animation: none; display: none; }
}
──────────────────────────────────────────────────────────────────────── */


/* ── Main Container ─────────────────────────────────────────────────────── */
/* NOTE: previously had `animation: fadeUp 0.35s ease forwards;` here, whose
   keyframe starts at opacity:0. Streamlit reruns the whole script (and
   rebuilds this container) far more often than a typical page — on every
   widget interaction, every cached-data refresh, etc. If a rerun retriggers
   the animation and the browser doesn't finish/settle it for any reason
   (a "prefers-reduced-motion" override that zeroes animation-duration but
   freezes on the FROM keyframe rather than the TO one is a known browser
   quirk on some platforms), the container is left stuck at opacity:0 —
   i.e. permanently invisible content with only the background showing,
   which matches the exact symptom reported. Removed entirely rather than
   patched further: an entrance fade adds nothing worth re-risking this. */
.main .block-container {
    padding-top: 1.75rem !important;
    padding-bottom: 4rem !important;
    max-width: 1400px !important;
    position: relative !important;
    z-index: 1 !important;
}

/* ── Typography ─────────────────────────────────────────────────────────── */
h1 {
    font-family: var(--font-display) !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.02em !important;
    text-transform: uppercase !important;
    color: var(--text-0) !important;
    background: none !important;
    -webkit-text-fill-color: var(--text-0) !important;
    line-height: 1.1 !important;
    margin: 0 !important;
}

h2 {
    font-family: var(--font-display) !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    color: var(--cyan) !important;
}

h3 {
    font-family: var(--font-display) !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-2) !important;
    border: none !important;
    padding: 0 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.6rem !important;
}

h4 {
    font-family: var(--font-display) !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-2) !important;
}

p, li {
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
    line-height: 1.55 !important;
    color: var(--text-1) !important;
}

/* Subheader — with cyan underline bar */
[data-testid="stHeadingWithActionElements"] h2,
div.stSubheader h2 {
    font-size: 1.05rem !important;
    letter-spacing: 0.03em !important;
    color: var(--text-0) !important;
    position: relative !important;
    padding-bottom: 12px !important;
    margin-bottom: 8px !important;
}
[data-testid="stHeadingWithActionElements"] h2::after,
div.stSubheader h2::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    width: 32px; height: 2px;
    background: var(--cyan);
    border-radius: 1px;
}

/* ── Metric Cards — the numbers should sing ─────────────────────────────── */
div[data-testid="metric-container"] {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 18px 20px 16px !important;
    position: relative !important;
    overflow: visible !important;
    transition: border-color 0.15s, background 0.15s !important;
    min-height: 92px !important;
}

/* HUD corner brackets — quiet terminal signature, brightens on hover */
div[data-testid="metric-container"]::after {
    content: '';
    position: absolute; top: -1px; right: -1px;
    width: 9px; height: 9px;
    border-top: 1.5px solid var(--border-mid);
    border-right: 1.5px solid var(--border-mid);
    border-radius: 0 var(--radius) 0 0;
    transition: border-color 0.15s;
}
div[data-testid="metric-container"]:hover::after { border-color: var(--cyan); }

/* Left accent bar */
div[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    left: 0; top: 18%; bottom: 18%;
    width: 3px;
    background: var(--border-mid);
    border-radius: 0 2px 2px 0;
    transition: background 0.15s;
}

div[data-testid="metric-container"]:hover {
    border-color: var(--cyan-border) !important;
    background: var(--bg-3) !important;
}
div[data-testid="metric-container"]:hover::before {
    background: var(--cyan);
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] > div,
div[data-testid="stMetricLabel"] div {
    font-family: var(--font-body) !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-2) !important;
    margin-bottom: 8px !important;
}

/* Big, legible numbers — heavier weight for at-a-glance reading */
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] > div,
div[data-testid="stMetricValue"] div {
    font-family: var(--font-body) !important;
    font-size: 2rem !important;
    font-weight: 500 !important;
    color: var(--text-0) !important;
    letter-spacing: -0.02em !important;
    line-height: 1 !important;
}

div[data-testid="stMetricDelta"],
div[data-testid="stMetricDelta"] > div,
div[data-testid="stMetricDelta"] div {
    font-family: var(--font-body) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    margin-top: 7px !important;
}

/* Color the delta icons properly */
div[data-testid="stMetricDelta"] [data-testid="stMetricDeltaIcon-Up"],
div[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"]) {
    color: var(--green-bright) !important;
}
div[data-testid="stMetricDelta"] [data-testid="stMetricDeltaIcon-Down"],
div[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]) {
    color: var(--red-bright) !important;
}

/* Green-accented cards */
div[data-testid="metric-container"]:has([data-testid="stMetricDeltaIcon-Up"])::before {
    background: var(--green);
}
div[data-testid="metric-container"]:has([data-testid="stMetricDeltaIcon-Up"]):hover::before {
    background: var(--green-bright);
}

/* Red-accented cards */
div[data-testid="metric-container"]:has([data-testid="stMetricDeltaIcon-Down"])::before {
    background: var(--red);
}
div[data-testid="metric-container"]:has([data-testid="stMetricDeltaIcon-Down"]):hover::before {
    background: var(--red-bright);
}

.metrics-sm div[data-testid="metric-container"] {
    min-height: 72px !important;
    padding: 13px 16px 11px !important;
}
.metrics-sm div[data-testid="stMetricValue"] > div {
    font-size: 1.45rem !important;
}
.metrics-sm div[data-testid="stMetricLabel"] > div {
    font-size: 0.65rem !important;
}

/* ── Tab Bar — segmented pill control, visible before AND after selection ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-1) !important;
    border-radius: var(--radius) !important;
    padding: 5px !important;
    gap: 6px !important;
    border: 1px solid var(--border-mid) !important;
    display: inline-flex !important;
    width: auto !important;
}

/* Base chip state — a real bordered/filled button look, not bare text,
   so it reads as clickable even before any interaction. */
.stTabs [data-baseweb="tab"] {
    background: var(--bg-3) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-mid) !important;
    font-family: var(--font-display) !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 10px 20px !important;
    margin-bottom: 0 !important;
    transition: color 0.15s, background 0.15s, border-color 0.15s !important;
}

/* Force text color on every descendant text node explicitly — Streamlit
   wraps tab labels in nested <p>/<div>/<span>, and an earlier global rule
   (`p, li { color: ... !important }`) matches those nodes directly and
   otherwise wins over a color set only on the ancestor tab element. */
.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] div,
.stTabs [data-baseweb="tab"] span {
    color: var(--text-1) !important;
}
.stTabs [data-baseweb="tab"]:hover,
.stTabs [data-baseweb="tab"]:hover p,
.stTabs [data-baseweb="tab"]:hover div,
.stTabs [data-baseweb="tab"]:hover span {
    color: var(--text-0) !important;
}
.stTabs [data-baseweb="tab"]:hover { background: var(--bg-4) !important; border-color: rgba(255,255,255,0.22) !important; }

.stTabs [aria-selected="true"] {
    background: var(--cyan) !important;
    border-color: var(--cyan) !important;
    box-shadow: 0 2px 12px rgba(224,168,60,0.28) !important;
}
.stTabs [aria-selected="true"],
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] div,
.stTabs [aria-selected="true"] span {
    color: #14100a !important;   /* near-black — guaranteed contrast on the gold fill */
}
.stTabs [aria-selected="true"]:hover { background: var(--cyan) !important; }

/* Remove baseweb's own underline/highlight bar — the chip fill is the signal now */
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; background: transparent !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 24px !important; }

/* Suppress .arrow_right leak */
.stTabs [data-baseweb="tab"] [data-testid="stIconMaterial"],
.stTabs [data-baseweb="tab"] svg ~ span,
.stTabs [data-baseweb="tab"] > span:last-child:not(:first-child) {
    font-size: 0 !important; color: transparent !important;
    width: 0 !important; overflow: hidden !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
    background: transparent !important;
    color: var(--text-1) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-body) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 8px 16px !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    color: var(--text-0) !important;
    border-color: rgba(255,255,255,0.24) !important;
    background: var(--bg-3) !important;
}

.btn-refresh .stButton > button {
    color: var(--cyan) !important;
    border-color: var(--cyan-border) !important;
}
.btn-refresh .stButton > button:hover { background: var(--cyan-dim) !important; }

.btn-danger .stButton > button {
    color: var(--red-bright) !important;
    border-color: rgba(239,68,68,0.35) !important;
}
.btn-danger .stButton > button:hover { background: var(--red-dim) !important; }

.btn-action .stButton > button {
    color: var(--green-bright) !important;
    border-color: rgba(16,185,129,0.35) !important;
}
.btn-action .stButton > button:hover { background: var(--green-dim) !important; }

/* ── Selectbox ──────────────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stSelectbox [data-baseweb="select"] > div {
    background: var(--bg-2) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-0) !important;
    font-family: var(--font-body) !important;
    font-size: 0.88rem !important;
}
.stSelectbox > div > div:focus-within,
.stSelectbox [data-baseweb="select"] > div:focus-within {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px var(--cyan-glow) !important;
}

/* ── Dataframes ─────────────────────────────────────────────────────────── */
.stDataFrame {
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* ── Dividers ───────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: var(--border-mid) !important;
    margin: 26px 0 !important;
}
hr::after { display: none !important; }

/* ── Expanders ──────────────────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    background: var(--bg-2) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    transition: border-color 0.15s !important;
    margin-bottom: 10px !important;
}
div[data-testid="stExpander"]:hover { border-color: rgba(255,255,255,0.2) !important; }
div[data-testid="stExpander"] > details > summary,
div[data-testid="stExpander"] details summary {
    display: flex !important; align-items: center !important; gap: 10px !important;
    padding: 12px 16px !important; cursor: pointer !important;
    list-style: none !important; -webkit-appearance: none !important;
    background: transparent !important;
}
div[data-testid="stExpander"] details summary::-webkit-details-marker { display:none !important; }
div[data-testid="stExpander"] details summary::marker { display:none !important; content:'' !important; }
div[data-testid="stExpanderDetails"],
div[data-testid="stExpander"] details summary > div:not([data-testid="stExpanderToggleIcon"]),
div[data-testid="stExpander"] details summary p,
div[data-testid="stExpander"] details summary span {
    font-family: var(--font-body) !important; font-size: 0.8rem !important;
    font-weight: 500 !important; color: var(--text-1) !important;
    letter-spacing: 0.02em !important; margin: 0 !important; padding: 0 !important;
}
div[data-testid="stExpanderToggleIcon"] {
    font-size:0 !important; color:transparent !important;
    display:flex !important; align-items:center !important;
    width:16px !important; height:16px !important; overflow:hidden !important;
}
div[data-testid="stExpanderToggleIcon"] * { font-size:0 !important; color:transparent !important; }
div[data-testid="stExpanderToggleIcon"] svg {
    color: var(--text-2) !important;
    width:13px !important; height:13px !important; display:block !important;
    visibility:visible !important; transition: transform 0.2s, color 0.15s !important;
}
div[data-testid="stExpander"] details[open] div[data-testid="stExpanderToggleIcon"] svg {
    transform: rotate(90deg) !important; color: var(--cyan) !important;
}
div[data-testid="stExpander"] details summary:hover div[data-testid="stExpanderToggleIcon"] svg {
    color: var(--text-1) !important;
}
div[data-testid="stExpander"] details > div { padding: 6px 16px 16px !important; }

/* ── Alerts ─────────────────────────────────────────────────────────────── */
.stAlert, div[data-testid="stNotification"] {
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-body) !important; font-size: 0.85rem !important;
}
div[data-baseweb="notification"][kind="info"],
div[data-testid="stAlert"] {
    background: rgba(34,211,238,0.06) !important;
    border: 1px solid rgba(34,211,238,0.22) !important;
}

/* ── Text inputs ────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background: var(--bg-2) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-0) !important;
    font-family: var(--font-body) !important; font-size: 0.88rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px var(--cyan-glow) !important;
    outline: none !important;
}
.search-bar .stTextInput > div > div > input {
    border-color: var(--cyan-border) !important;
}

/* ── Sliders ────────────────────────────────────────────────────────────── */
.stSlider > div > div > div { background: rgba(34,211,238,0.16) !important; }
.stSlider > div > div > div > div { background: var(--cyan) !important; }
.stSlider [role="slider"] {
    background: var(--bg-0) !important;
    border: 2px solid var(--cyan) !important;
    box-shadow: 0 0 0 4px var(--cyan-glow) !important;
}

/* ── Multiselect ────────────────────────────────────────────────────────── */
.stMultiSelect span[data-baseweb="tag"] {
    background: var(--cyan-dim) !important; border: 1px solid var(--cyan-border) !important;
    border-radius: 4px !important; color: var(--cyan) !important;
    font-family: var(--font-body) !important; font-size: 0.7rem !important;
    font-weight: 500 !important; letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* ── Download button ────────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: transparent !important; border: 1px solid var(--cyan-border) !important;
    color: var(--cyan) !important; font-family: var(--font-body) !important;
    font-size: 0.76rem !important; letter-spacing: 0.05em !important;
    text-transform: uppercase !important; border-radius: var(--radius-sm) !important;
    transition: all 0.15s !important;
}
.stDownloadButton > button:hover { background: var(--cyan-dim) !important; }

/* ── Spinner ────────────────────────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--cyan) !important;
    border-right-color: var(--cyan-glow) !important;
    border-bottom-color: transparent !important; border-left-color: transparent !important;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-1) !important; border-right: 1px solid var(--border-mid) !important;
}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-0); }
::-webkit-scrollbar-thumb { background: rgba(34,211,238,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(34,211,238,0.35); }

/* ── Skeleton ───────────────────────────────────────────────────────────── */
.skeleton {
    background: linear-gradient(90deg, var(--bg-2) 0%, var(--bg-3) 50%, var(--bg-2) 100%);
    background-size: 200% 100%;
    animation: shimmer 1.8s ease-in-out infinite;
    border-radius: var(--radius); border: 1px solid var(--border); width: 100%;
}
@keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ── Badges ─────────────────────────────────────────────────────────────── */
.badge {
    display: inline-block; padding: 3px 9px; border-radius: 4px;
    font-family: var(--font-body); font-size: 0.68rem; font-weight: 500;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.badge-green { background: var(--green-dim);  color: var(--green-bright); border: 1px solid rgba(16,185,129,0.3); }
.badge-amber { background: var(--amber-dim);  color: var(--amber-bright); border: 1px solid rgba(245,158,11,0.3); }
.badge-red   { background: var(--red-dim);    color: var(--red-bright);   border: 1px solid rgba(239,68,68,0.3); }
.badge-blue  { background: var(--cyan-dim);   color: var(--cyan);         border: 1px solid var(--cyan-border); }

/* ── Ticker ─────────────────────────────────────────────────────────────── */
.ticker {
    display: inline-block; padding: 3px 9px;
    background: var(--cyan-dim); border: 1px solid var(--cyan-border); border-radius: 4px;
    font-family: var(--font-body); font-size: 0.78rem; font-weight: 500;
    color: var(--cyan); letter-spacing: 0.04em; text-transform: uppercase;
}

/* ── Status dot ─────────────────────────────────────────────────────────── */
.status-dot {
    display: inline-block; width: 7px; height: 7px;
    border-radius: 50%; vertical-align: middle;
}
.status-dot.live    { background: var(--green-bright); animation: pulse 3s ease-in-out infinite; }
.status-dot.warning { background: var(--amber-bright); }
.status-dot.error   { background: var(--red-bright);   animation: pulse 2s ease-in-out infinite; }
.status-dot.idle    { background: var(--text-3); }

@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.6; transform:scale(0.85); }
}

/* ── Num helpers ────────────────────────────────────────────────────────── */
.num          { font-family: var(--font-body); font-weight: 500; font-size: 0.9rem; }
.num.positive { color: var(--green-bright); }
.num.negative { color: var(--red-bright); }
.num.neutral  { color: var(--cyan); }

/* ── Section header ─────────────────────────────────────────────────────── */
.section-header {
    font-family: var(--font-body); font-size: 0.68rem; font-weight: 500;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--text-2);
    padding-bottom: 10px; border-bottom: 1px solid var(--border-mid); margin-bottom: 16px;
}

/* ── Data / search / warning cards ─────────────────────────────────────── */
.data-card {
    background: var(--bg-2); border: 1px solid var(--border-mid);
    border-radius: var(--radius); padding: 16px 18px; transition: border-color 0.15s;
    position: relative;
}
.data-card:hover { border-color: var(--cyan-border); }
.data-card::after {
    content: ''; position: absolute; top: -1px; right: -1px; width: 9px; height: 9px;
    border-top: 1.5px solid var(--border-mid); border-right: 1.5px solid var(--border-mid);
    border-radius: 0 var(--radius) 0 0; transition: border-color 0.15s;
}
.data-card:hover::after { border-color: var(--cyan); }

/* Native st.container(border=True) — used for grouping controls/notes that
   would otherwise float without a visual boundary (e.g. simulator config
   panels, disclaimer notes). Styled to match .data-card so both approaches
   (raw HTML vs. real Streamlit layout containers) look identical. */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius) !important;
    background: var(--bg-2) !important;
    position: relative !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: var(--radius) !important; }
div[data-testid="stVerticalBlockBorderWrapper"]::after {
    content: ''; position: absolute; top: -1px; right: -1px; width: 9px; height: 9px;
    border-top: 1.5px solid var(--border-mid); border-right: 1.5px solid var(--border-mid);
    border-radius: 0 var(--radius) 0 0; pointer-events: none;
}

/* Text-only info/note block — left accent bar, quieter than a full data-card */
.info-card {
    background: var(--bg-2); border: 1px solid var(--border-mid);
    border-left: 3px solid var(--cyan); border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 12px 16px; margin: 8px 0 16px;
    font-family: var(--font-body); font-size: 0.85rem; line-height: 1.55;
    color: var(--text-1);
}
.info-card.muted { border-left-color: var(--text-3); }

.search-result-card {
    background: var(--cyan-dim); border: 1px solid var(--cyan-border);
    border-left: 3px solid var(--cyan); border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 12px 16px; margin-bottom: 8px; font-family: var(--font-body);
}

.cache-warning {
    background: var(--red-dim); border: 1px solid rgba(239,68,68,0.3);
    border-left: 3px solid var(--red); border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 10px 14px; font-family: var(--font-body); font-size: 0.82rem;
    color: var(--red-bright); margin-bottom: 10px;
}

/* ── Chips ──────────────────────────────────────────────────────────────── */
.preset-chips { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px; }
.chip {
    padding: 4px 11px; border-radius: 4px; font-family: var(--font-body);
    font-size: 0.68rem; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase;
    cursor: pointer; transition: all 0.12s;
    background: transparent; border: 1px solid var(--border-mid); color: var(--text-2);
}
.chip:hover, .chip.active {
    background: var(--cyan-dim); border-color: var(--cyan-border); color: var(--cyan);
}

/* ── Form submit ────────────────────────────────────────────────────────── */
.stFormSubmitButton > button {
    background: transparent !important; border: 1px solid var(--cyan-border) !important;
    color: var(--cyan) !important; font-family: var(--font-body) !important;
    font-weight: 500 !important; font-size: 0.78rem !important;
    letter-spacing: 0.05em !important; text-transform: uppercase !important;
    border-radius: var(--radius-sm) !important; transition: all 0.15s !important;
}
.stFormSubmitButton > button:hover { background: var(--cyan-dim) !important; }

/* ── Plotly charts ──────────────────────────────────────────────────────── */
div[data-testid="stPlotlyChart"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important; border: 1px solid var(--border-mid) !important;
}

/* ── Number input ───────────────────────────────────────────────────────── */
.stNumberInput button {
    background: var(--bg-3) !important; border-color: var(--border-mid) !important;
    color: var(--text-2) !important;
}
.stNumberInput button:hover { color: var(--text-1) !important; }

/* ── Checkbox & Radio ───────────────────────────────────────────────────── */
.stCheckbox label, .stRadio label {
    font-family: var(--font-body) !important; font-size: 0.85rem !important;
    color: var(--text-1) !important;
}

/* ── Progress bar ───────────────────────────────────────────────────────── */
div[data-testid="stProgressBar"] > div > div {
    background: var(--cyan) !important;
}
div[data-testid="stProgressBar"] > div {
    background: var(--bg-4) !important; border-radius: 2px !important; height: 4px !important;
}

/* ── Caption ────────────────────────────────────────────────────────────── */
.stCaption, small,
div[data-testid="stCaptionContainer"],
div[data-testid="stCaptionContainer"] p {
    font-family: var(--font-body) !important; font-size: 0.78rem !important;
    color: var(--text-2) !important; letter-spacing: 0.01em !important;
}

/* ── Toast ──────────────────────────────────────────────────────────────── */
div[data-testid="toastContainer"] {
    font-family: var(--font-body) !important; font-size: 0.82rem !important;
}

/* Per-block entrance animations removed — same opacity:0-stuck risk as the
   block-container animation above, for the same reason (frequent Streamlit
   reruns rebuilding these nodes). Not worth the risk for a fade-in. */

</style>
"""

# ── Optional compact-density override ─────────────────────────────────────────
# Injected in addition to DASHBOARD_CSS (not instead of) when the user turns on
# "Compact mode" from the header — tightens vertical rhythm for people who want
# more rows/charts visible without scrolling, without touching the base theme.
COMPACT_CSS = """
<style>
.main .block-container { padding-top: 1.1rem !important; padding-bottom: 1.1rem !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0.35rem !important; }
hr { margin: 8px 0 !important; }
div[data-testid="stMetric"] { padding: 6px 10px !important; }
div[data-testid="stMetricValue"] { font-size: 1.15rem !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px !important; }
.stTabs [data-baseweb="tab"] { padding: 6px 12px !important; }
div[data-testid="stExpander"] { margin-bottom: 6px !important; }
.stMarkdown h4 { margin-top: 6px !important; margin-bottom: 6px !important; }
</style>
"""
