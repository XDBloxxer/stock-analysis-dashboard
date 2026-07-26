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
    background-color: var(--bg-0) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 0% 0%,   rgba(224,168,60,0.055) 0%, transparent 55%),
        radial-gradient(ellipse 60% 45% at 100% 100%, rgba(16,185,129,0.045) 0%, transparent 55%) !important;
    font-family: var(--font-body);
    /* NOTE: deliberately NO position/z-index here. An earlier attempt added
       `position:relative; z-index:0;` to force .stApp into its own stacking
       context, on the theory that it was needed for the ::before/::after
       z-index:-1 layers below to stay behind content. That was wrong and is
       what actually caused the blank-dashboard regression — confirmed by
       A/B diff against a known-working build where this rule is plain. Do
       not re-add position/z-index to .stApp without testing live first. */
}

/* Ambient background wash — a very slow, soft drift of brass/emerald light
   behind the content. Replaces the old CRT scanline texture + hard sweep
   band, which read as "TV static" at normal viewing distance rather than
   a quiet terminal signature. This version has no repeating line texture
   at all, moves gently over ~30s, and stays well under the content so it
   never competes with data. */
[data-testid="stAppViewContainer"] {
    z-index: 0; /* stAppViewContainer is already positioned by Streamlit's
       own default CSS, so adding only z-index (no position override) is
       safe — creates a stacking context without touching layout. This is
       where the ambient-wash pseudo-element now lives instead of on the
       root .stApp: .stApp itself is a plain, huge, non-positioned div, so
       its own opaque background paints AFTER its negative-z-index ::before
       at the root stacking context level — silently covering it. Scoping
       the effect one level down avoids that trap entirely. */
}
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: -10%;
    background-image:
        radial-gradient(ellipse 55% 40% at 15% 20%, rgba(224,168,60,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 85% 75%, rgba(16,185,129,0.04) 0%, transparent 60%);
    background-repeat: no-repeat;
    pointer-events: none;
    z-index: -1;
    /* Auto-drift (keyframes) is layered with a cursor parallax nudge via
       the --px/--py custom properties (set alongside --mx/--my by
       inject_mouse_glow_script()'s JS below). Mixing a % keyframe value
       with a px var() inside calc() is valid CSS — the blobs keep their
       slow autonomous drift and *also* lean a few px toward the cursor,
       rather than the cursor fully driving position (which would fight
       the animation and feel jittery next to dense data). */
    transform: translate(calc(var(--px, 0px)), calc(var(--py, 0px))) scale(1);
    animation: ambientDrift 32s ease-in-out infinite;
    will-change: transform;
}
@keyframes ambientDrift {
    0%   { transform: translate(calc(var(--px, 0px)), calc(var(--py, 0px))) scale(1); }
    50%  { transform: translate(calc(-2.5% + var(--px, 0px)), calc(2% + var(--py, 0px))) scale(1.04); }
    100% { transform: translate(calc(var(--px, 0px)), calc(var(--py, 0px))) scale(1); }
}
@media (prefers-reduced-motion: reduce) {
    [data-testid="stAppViewContainer"]::before { animation: none; transform: none; }
}

/* ── Cursor glow — a soft brass light that follows the mouse ────────────────
   Position comes from --mx/--my custom properties on <html>, updated by
   inject_mouse_glow_script() below via requestAnimationFrame — the DOM
   write only ever touches a handful of CSS variables, never element
   styles/layout, so this stays compositor-only and cheap even on a page
   this data-dense.
   Lives on ::after (::before is already the ambient drift layer) and sits
   at the same z-index: -1, i.e. still well under all content.
   Defaults to page center via the fallback in var(--mx, 50vw) so there's
   no jump/pop before the first mousemove event fires. Skipped entirely for
   reduced-motion users and on touch devices (no hover = no cursor to
   follow), same posture as the ambient drift animation above.
   Kept deliberately faint — low opacity, wide soft falloff, no hard edge —
   so it reads as light catching a screen rather than a spotlight chasing
   the pointer around a data-dense finance UI. */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed; inset: 0;
    background: radial-gradient(
        circle 380px at var(--mx, 50vw) var(--my, 40vh),
        rgba(224, 168, 60, 0.028) 0%,
        rgba(224, 168, 60, 0.012) 45%,
        transparent 72%
    );
    pointer-events: none;
    z-index: -1;
    transition: opacity 0.4s ease;
}
@media (prefers-reduced-motion: reduce), (hover: none) {
    [data-testid="stAppViewContainer"]::after { opacity: 0 !important; }
}


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
.main .block-container,
[data-testid="stMainBlockContainer"] {
    padding-top: 5.5rem !important; /* was 1.75rem — too little to clear
       Streamlit's fixed-position native header (~3.75rem), which was
       overlapping/hiding the first row of content (market status, clock). */
    padding-bottom: 4rem !important;
    max-width: 1400px !important;
    position: relative !important;
    z-index: 1 !important;
}

/* ── Native Streamlit header bar ───────────────────────────────────────────
   Previously completely unstyled — the default light/translucent Streamlit
   chrome (Stop/Share/Star/GitHub/⋮) sat untouched on top of the custom dark
   theme, which read as "two different themes stitched together." Restyled
   to match the app background so it reads as one cohesive surface. */
[data-testid="stHeader"] {
    background: var(--bg-0) !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stHeader"] * {
    color: var(--text-1) !important;
}
[data-testid="stToolbar"] button:hover {
    background: var(--bg-3) !important;
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
div[data-testid="stMetric"] {
    /* background-color kept separate from background-image (below) so both
       layers survive — a plain `background:` shorthand here would wipe out
       the accent-bar gradient every time. */
    background-color: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 18px 20px 16px !important;
    position: relative !important;
    overflow: visible !important;
    transition: border-color 0.15s, background-color 0.15s, background-size 0.35s cubic-bezier(0.16,1,0.3,1), box-shadow 0.15s !important;
    min-height: 92px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.28) !important;

    /* Top-accent-bar reveal on hover — a thin gradient line that grows in
       from the left, borrowed from a portfolio site's `.project-card::before`
       hover flourish. Implemented as a second background layer (rather than
       a new pseudo-element) since ::before is the left accent bar and
       ::after is the corner bracket on this same card. */
    background-image: linear-gradient(90deg, var(--cyan), var(--purple)) !important;
    background-repeat: no-repeat !important;
    background-position: top left !important;
    background-size: 0% 2px !important;
}
div[data-testid="stMetric"]:hover {
    background-size: 100% 2px !important;
}

/* HUD corner brackets — quiet terminal signature, brightens on hover */
div[data-testid="stMetric"]::after {
    content: '';
    position: absolute; top: -1px; right: -1px;
    width: 9px; height: 9px;
    border-top: 1.5px solid var(--border-mid);
    border-right: 1.5px solid var(--border-mid);
    border-radius: 0 var(--radius) 0 0;
    transition: border-color 0.15s;
}
div[data-testid="stMetric"]:hover::after { border-color: var(--cyan); }

/* Left accent bar */
div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    left: 0; top: 18%; bottom: 18%;
    width: 3px;
    background: var(--border-mid);
    border-radius: 0 2px 2px 0;
    transition: background 0.15s;
}

div[data-testid="stMetric"]:hover {
    border-color: var(--cyan-border) !important;
    background-color: var(--bg-3) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.32) !important;
}
div[data-testid="stMetric"]:hover::before {
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
div[data-testid="stMetric"]:has([data-testid="stMetricDeltaIcon-Up"])::before {
    background: var(--green);
}
div[data-testid="stMetric"]:has([data-testid="stMetricDeltaIcon-Up"]):hover::before {
    background: var(--green-bright);
}

/* Red-accented cards */
div[data-testid="stMetric"]:has([data-testid="stMetricDeltaIcon-Down"])::before {
    background: var(--red);
}
div[data-testid="stMetric"]:has([data-testid="stMetricDeltaIcon-Down"]):hover::before {
    background: var(--red-bright);
}

.metrics-sm div[data-testid="stMetric"] {
    min-height: 72px !important;
    padding: 13px 16px 11px !important;
}
.metrics-sm div[data-testid="stMetricValue"] > div {
    font-size: 1.45rem !important;
}
.metrics-sm div[data-testid="stMetricLabel"] > div {
    font-size: 0.65rem !important;
}

/* ── Hero metric — one visually-dominant number per section ─────────────────
   Pairs with render_hero_metric() below. A row of equal-weight st.metric
   cards has no visual entry point — the eye doesn't know which of 5-7
   numbers matters most. This is deliberately ~2x the size/brightness of a
   normal stMetricValue, with a soft color-matched glow, so the one stat a
   section is really about (best performer, top pick's confidence, total
   return) reads as the headline at a glance, before anything else. */
.hero-metric {
    background: var(--bg-2);
    border: 1px solid var(--border-mid);
    border-left: 3px solid var(--hero-color, var(--cyan));
    border-radius: var(--radius);
    padding: 18px 22px 16px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
    animation: tabFadeIn 0.3s ease-out;
    transition: transform 0.22s ease, border-color 0.22s ease,
                box-shadow 0.22s ease, background 0.22s ease;
    cursor: default;
}
.hero-metric:hover {
    transform: translateY(-2px);
    border-color: var(--hero-color, var(--cyan));
    background: var(--bg-3);
    box-shadow: 0 0 0 1px var(--hero-glow, var(--cyan-glow)),
                0 8px 28px -6px var(--hero-glow, var(--cyan-glow)),
                0 0 46px var(--hero-glow, var(--cyan-glow));
}
.hero-metric:hover::before {
    content: '';
    position: absolute; inset: -40%;
    background: radial-gradient(circle, var(--hero-glow, var(--cyan-glow)) 0%, transparent 65%);
    animation: heroAuraPulse 2.2s ease-in-out infinite;
    pointer-events: none;
}
.hero-metric:hover .hero-metric-value {
    text-shadow: 0 0 40px var(--hero-glow, var(--cyan-glow)),
                 0 0 14px var(--hero-glow, var(--cyan-glow));
}
@keyframes heroAuraPulse {
    0%, 100% { opacity: 0.55; transform: scale(1); }
    50%      { opacity: 1;    transform: scale(1.06); }
}
.hero-metric::after {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 70% 90% at 0% 0%, var(--hero-glow, var(--cyan-glow)) 0%, transparent 70%);
    pointer-events: none;
}
.hero-metric-label {
    font-family: var(--font-body); font-size: 0.68rem; font-weight: 500;
    letter-spacing: 0.16em; text-transform: uppercase; color: var(--text-2);
    position: relative;
}
.hero-metric-value {
    font-family: var(--font-display); font-size: 2.6rem; font-weight: 800;
    line-height: 1.05; color: var(--hero-color, var(--cyan)); letter-spacing: -0.01em;
    margin-top: 4px; position: relative;
    text-shadow: 0 0 28px var(--hero-glow, var(--cyan-glow));
    transition: text-shadow 0.22s ease;
}
.hero-metric-glyph { font-size: 0.55em; opacity: 0.7; margin-right: 8px; vertical-align: middle; }
.hero-metric-sub {
    font-family: var(--font-body); font-size: 0.78rem; color: var(--text-1);
    margin-top: 6px; position: relative;
}
@media (max-width: 640px) {
    .hero-metric-value { font-size: 2rem; }
}

/* ── Indicator snapshot grid — compact label/value pairs ────────────────────
   For dense technical-indicator readouts (10-24 fields per group), a grid
   of bordered stMetric cards is too heavy — each field is a single number,
   not a headline stat. This renders the same data as a tight CSS-grid of
   label/value cells inside one card-like surface instead, so a full
   indicator group reads as one scannable table rather than a wall of boxes.
   Pairs with render_indicator_grid() below. */
.indicator-snapshot {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    overflow: hidden;
    margin-bottom: 8px;
}
.indicator-snapshot .is-cell {
    background: var(--bg-2);
    padding: 9px 12px 8px;
    transition: background 0.15s ease;
}
.indicator-snapshot .is-cell:hover {
    background: var(--bg-3);
}
.indicator-snapshot .is-label {
    font-family: var(--font-body);
    font-size: 0.62rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-2);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 3px;
}
.indicator-snapshot .is-value {
    font-family: var(--font-body);
    font-size: 0.92rem;
    font-weight: 500;
    color: var(--text-0);
}
.indicator-snapshot .is-value.is-true  { color: var(--green-bright); }
.indicator-snapshot .is-value.is-false { color: var(--text-3); }

/* ── Tab Bar — segmented pill control, visible before AND after selection ──
   Selectors target [role="tablist"]/[role="tab"] (standard ARIA, stable
   across versions) rather than relying only on data-baseweb hooks — a
   Streamlit/BaseWeb bump dropped those data-baseweb attributes from the
   tab-list and tab elements while leaving aria-selected untouched, which is
   why only the selected-tab fill kept working and everything else quietly
   fell back to unstyled text. data-baseweb selectors are kept alongside as
   a harmless no-op fallback for older builds. */
.stTabs [role="tablist"],
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-1) !important;
    border-radius: var(--radius) !important;
    padding: 5px !important;
    gap: 6px !important;
    border: 1px solid var(--border-mid) !important;
    display: inline-flex !important;
    width: auto !important;
    max-width: 100% !important;
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
    scrollbar-width: thin !important;

    /* Sticky top-level nav — the tab bar stays visible while scrolling a
       long tab's content instead of scrolling out of view. Pinned just
       under the app header; a faint bottom shadow appears once it's
       actually stuck (see `top` value matching the app's fixed spacing). */
    position: sticky !important;
    top: 0 !important;
    z-index: 20 !important;
    box-shadow: 0 6px 12px -6px rgba(0,0,0,0.4) !important;
}
/* Un-stick nested tab bars (tabs-within-a-tab, e.g. the sub-tabs inside
   Today's Picks) — letting every level stick would pile multiple bars on
   top of each other at the same `top: 0`, rather than stacking sensibly. */
.stTabs [data-baseweb="tab-panel"] .stTabs [role="tablist"],
.stTabs [data-baseweb="tab-panel"] .stTabs [data-baseweb="tab-list"] {
    position: static !important;
    top: auto !important;
    z-index: auto !important;
    box-shadow: none !important;
}
/* Narrow screens (phones): let the pill bar wrap onto multiple lines
   instead of forcing a tiny horizontal scrollbar users may not notice —
   5+ tabs on a ~375px viewport otherwise get clipped or unreadable. */
@media (max-width: 640px) {
    .stTabs [role="tablist"],
    .stTabs [data-baseweb="tab-list"] {
        flex-wrap: wrap !important;
        overflow-x: visible !important;
    }
    .stTabs [role="tab"],
    .stTabs [data-baseweb="tab"] {
        padding: 8px 14px !important;
        font-size: 0.72rem !important;
    }
}

/* Base chip state — a real bordered/filled button look, not bare text,
   so it reads as clickable even before any interaction. Uses a slight
   gradient (lighter top edge) plus a crisp drop shadow and inset top
   highlight to read as a physically raised key, not a text label. */
.stTabs [role="tab"],
.stTabs [data-baseweb="tab"] {
    background: linear-gradient(180deg, var(--bg-4) 0%, #171d27 100%) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-mid) !important;
    font-family: var(--font-display) !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 10px 20px !important;
    margin-bottom: 0 !important;
    cursor: pointer !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.05) inset,
        0 2px 5px rgba(0,0,0,0.45) !important; /* inset top highlight + real
        drop shadow so it reads as a pressable, physically raised chip
        rather than plain text sitting on the track */
    transform: translateY(0) !important;
    transition: color 0.15s, background 0.15s, border-color 0.15s,
        box-shadow 0.15s, transform 0.15s !important;
}
/* Browser's default blue focus ring (tabs are real <button> elements)
   showed through uncontrolled on click — replaced with a themed outline. */
.stTabs [role="tab"]:focus,
.stTabs [role="tab"]:focus-visible,
.stTabs [data-baseweb="tab"]:focus,
.stTabs [data-baseweb="tab"]:focus-visible {
    outline: 2px solid var(--cyan-border) !important;
    outline-offset: 1px !important;
    box-shadow: none !important;
}

/* Force text color on every descendant text node explicitly — Streamlit
   wraps tab labels in nested <p>/<div>/<span>, and an earlier global rule
   (`p, li { color: ... !important }`) matches those nodes directly and
   otherwise wins over a color set only on the ancestor tab element. */
.stTabs [role="tab"],
.stTabs [role="tab"] p,
.stTabs [role="tab"] div,
.stTabs [role="tab"] span,
.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] div,
.stTabs [data-baseweb="tab"] span {
    color: var(--text-1) !important;
}
.stTabs [role="tab"]:hover,
.stTabs [role="tab"]:hover p,
.stTabs [role="tab"]:hover div,
.stTabs [role="tab"]:hover span,
.stTabs [data-baseweb="tab"]:hover,
.stTabs [data-baseweb="tab"]:hover p,
.stTabs [data-baseweb="tab"]:hover div,
.stTabs [data-baseweb="tab"]:hover span {
    color: var(--text-0) !important;
}
.stTabs [role="tab"]:hover,
.stTabs [data-baseweb="tab"]:hover {
    background: linear-gradient(180deg, #232a36 0%, var(--bg-4) 100%) !important;
    border-color: rgba(255,255,255,0.24) !important;
    transform: translateY(-1px) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.06) inset,
        0 4px 10px rgba(0,0,0,0.5) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, var(--cyan-bright, #f0b94e) 0%, var(--cyan) 100%) !important;
    border-color: var(--cyan) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.35) inset,
        0 3px 14px rgba(224,168,60,0.38) !important;
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
.stTabs [role="tab"] [data-testid="stIconMaterial"],
.stTabs [role="tab"] svg ~ span,
.stTabs [role="tab"] > span:last-child:not(:first-child),
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
/* Toolbar (search / download / fullscreen icons) — default is a flat grey
   that reads as unstyled against the rest of the theme; these are real DOM
   elements (unlike the canvas-rendered grid body), so a light touch here
   goes a long way. */
[data-testid="stElementToolbar"] {
    background: var(--bg-2) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stElementToolbar"] button svg { color: var(--text-2) !important; }
[data-testid="stElementToolbar"] button:hover svg { color: var(--cyan) !important; }

/* ── Empty states ───────────────────────────────────────────────────────────
   Replaces plain st.info/st.warning "no data" boxes with something that
   matches the terminal aesthetic instead of falling back to default
   Streamlit chrome mid-experience. Used via render_empty_state(). */
.empty-state {
    display: flex; flex-direction: column; align-items: center; text-align: center;
    gap: 10px; padding: 40px 24px; background: var(--bg-2);
    border: 1px dashed var(--border-mid); border-radius: var(--radius);
    color: var(--text-2);
}
.empty-state .es-glyph {
    font-family: var(--font-body); font-size: 1.4rem; color: var(--text-3); opacity: 0.7;
}
.empty-state .es-msg {
    font-family: var(--font-body); font-size: 0.85rem; color: var(--text-2); line-height: 1.6;
    max-width: 420px;
}

/* ── Tab-switch fade-in ─────────────────────────────────────────────────────
   Streamlit re-renders the panel body on every tab click, so a short
   entrance animation on the panel itself gives a soft fade instead of a
   hard pop each time the content swaps in. */
.stTabs [data-baseweb="tab-panel"] {
    animation: tabFadeIn 0.22s ease-out;
}
@keyframes tabFadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Dividers ───────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: var(--border-mid) !important;
    margin: 26px 0 !important;
}
hr::after { display: none !important; }

/* Labeled divider — "— Filters —" style, for splitting a page into named
   blocks without a full section header. Used via render_labeled_divider(). */
.labeled-divider {
    display: flex; align-items: center; gap: 14px; margin: 26px 0;
}
.labeled-divider .ld-line { flex: 1; height: 1px; background: var(--border-mid); }
.labeled-divider .ld-text {
    font-family: var(--font-body); font-size: 0.66rem; font-weight: 500;
    letter-spacing: 0.16em; text-transform: uppercase; color: var(--text-2);
    white-space: nowrap;
}

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
    background: rgba(224,168,60,0.06) !important;
    border: 1px solid rgba(224,168,60,0.22) !important;
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
.stSlider > div > div > div { background: rgba(224,168,60,0.16) !important; }
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
::-webkit-scrollbar-thumb { background: rgba(224,168,60,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(224,168,60,0.35); }

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

/* Subtle "this one matters" cue for the top-tier signal only — reuses the
   same slow pulse already driving the market-status dot, just on the
   badge's border/glow rather than opacity, so it reads as emphasis rather
   than a blinking alert. */
.badge-pulse {
    animation: badgePulse 2.4s ease-in-out infinite;
}
@keyframes badgePulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.35); }
    50%      { box-shadow: 0 0 0 4px rgba(16,185,129,0); }
}

/* ── Ticker ─────────────────────────────────────────────────────────────── */
.ticker {
    display: inline-block; padding: 3px 9px;
    background: var(--cyan-dim); border: 1px solid var(--cyan-border); border-radius: 4px;
    font-family: var(--font-body); font-size: 0.78rem; font-weight: 500;
    color: var(--cyan); letter-spacing: 0.04em; text-transform: uppercase;
}

/* Click-to-copy ticker symbols — a dashed underline hints it's
   interactive; click briefly flashes green + "✓ copied" via the inline
   onclick handler built by ticker_copy_html() (plain JS, not a component,
   since this needs to run in the main document, not a sandboxed iframe). */
.ticker-copy {
    cursor: pointer; border-bottom: 1px dashed transparent; transition: border-color 0.15s;
}
.ticker-copy:hover { border-bottom-color: var(--cyan-border); }

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

/* ── Animated fill bar ──────────────────────────────────────────────────────
   For any place a percentage was previously shown as plain text only
   (confidence, precision/recall, etc.) — the bar makes relative magnitude
   scannable across a list of rows without reading every number. Grows in
   on load via a CSS animation (not a hover effect), same "grow from 0"
   technique as a portfolio site's animated skill bars. Color is left to
   the caller via an inline `background` on `.bar-fill` (green/cyan/amber
   already carry meaning elsewhere in this app, e.g. signal strength). */
.bar-track {
    height: 4px; background: var(--bg-4); border-radius: 2px; overflow: hidden;
    margin-top: 5px;
}
.bar-fill {
    height: 100%; border-radius: 2px;
    transform-origin: left; transform: scaleX(0);
    animation: barGrow 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
@keyframes barGrow { to { transform: scaleX(1); } }

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

/* ── Numbered section header — "01 — Section" + trailing line ─────────────
   Borrowed from a personal portfolio site's `.section-header` pattern
   (numbered label, thin rule trailing off to the right). Used via
   render_section_header() for the major heading in each sub-tab, in place
   of a plain st.subheader(), so the hierarchy reads as designed rather than
   default-Streamlit. */
.section-header-num {
    display: flex; align-items: center; gap: 14px;
    margin: 4px 0 18px;
}
.section-header-num .sh-num {
    font-family: var(--font-body); font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.18em; color: var(--cyan); opacity: 0.8; flex-shrink: 0;
    white-space: nowrap;
}
.section-header-num .sh-title {
    font-family: var(--font-display); font-size: 1.1rem; font-weight: 700;
    color: var(--text-0); letter-spacing: 0.01em; white-space: nowrap;
}
.section-header-num .sh-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border-mid), transparent 90%);
}

/* ── Data / search / warning cards ─────────────────────────────────────── */
.data-card {
    background: var(--bg-2); border: 1px solid var(--border-mid);
    border-radius: var(--radius); padding: 16px 18px;
    transition: border-color 0.15s, background 0.15s, box-shadow 0.15s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.28);
    position: relative;
}
.data-card:hover {
    border-color: var(--cyan-border);
    background: var(--bg-3);
    box-shadow: 0 2px 8px rgba(0,0,0,0.32);
}
.data-card::after {
    content: ''; position: absolute; top: -1px; right: -1px; width: 9px; height: 9px;
    border-top: 1.5px solid var(--border-mid); border-right: 1.5px solid var(--border-mid);
    border-radius: 0 var(--radius) 0 0; transition: border-color 0.15s;
    z-index: 2;
}
.data-card:hover::after { border-color: var(--cyan); }

/* Cursor-proximity glow — a soft brass light that tracks the pointer
   *inside* the card while hovered, via the --cx/--cy custom properties
   set (per-card, in local %) by the same mousemove handler that drives
   the page-level cursor glow above. This is a border/wash brighten, not
   a 3D tilt: a tilt effect reads as playful/gamey and would clash with a
   data-dense finance UI where numbers need to stay flat and legible. */
.data-card::before {
    content: ''; position: absolute; inset: 0; border-radius: inherit;
    background: radial-gradient(
        260px circle at var(--cx, 50%) var(--cy, 50%),
        rgba(224, 168, 60, 0.10),
        transparent 72%
    );
    opacity: 0; transition: opacity 0.25s ease; pointer-events: none; z-index: 0;
}
.data-card:hover::before { opacity: 1; }
@media (prefers-reduced-motion: reduce), (hover: none) {
    .data-card::before { display: none; }
}

/* Signal-strength color bleed — STRONG BUY cards get a faint green wash
   across the whole card, not just the badge, so the strongest signals are
   readable from across the room rather than requiring a close look at each
   badge individually. Add alongside `data-card` (e.g. class="data-card
   card-strong-buy"). */
.data-card.card-strong-buy {
    background: linear-gradient(180deg, rgba(16,185,129,0.06), var(--bg-2) 60%);
    border-color: rgba(16,185,129,0.22);
}
.data-card.card-strong-buy:hover {
    background: linear-gradient(180deg, rgba(16,185,129,0.1), var(--bg-3) 60%);
    border-color: rgba(16,185,129,0.35);
}

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

/* ── App footer ─────────────────────────────────────────────────────────── */
.app-footer {
    margin-top: 40px; padding-top: 16px;
    border-top: 1px solid var(--border);
    font-family: var(--font-body); font-size: 0.7rem;
    letter-spacing: 0.02em; line-height: 1.6;
    color: var(--text-3); text-align: center;
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
.stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"] { gap: 4px !important; }
.stTabs [role="tab"], .stTabs [data-baseweb="tab"] { padding: 6px 12px !important; }
div[data-testid="stExpander"] { margin-bottom: 6px !important; }
.stMarkdown h4 { margin-top: 6px !important; margin-bottom: 6px !important; }
</style>
"""

# ── Helpers ────────────────────────────────────────────────────────────────────
def ticker_copy_html(symbol, style=""):
    """Inline clickable ticker span — copies `symbol` to the clipboard on
    click, flashes green + "✓ copied" for ~900ms, then reverts.

    Built as a plain onclick handler (not a components.html script) because
    st.markdown content lives directly in the main document rather than a
    sandboxed iframe, so `this` + the Clipboard API both work without the
    parent-document workaround the count-up script needs.

    Tries navigator.clipboard.writeText() first, but that call silently
    rejects in some hosting/browser setups (permissions-policy quirks,
    non-secure contexts, older browsers) — with no visible error, it just
    looks like a dead button. Falls back to the older
    document.execCommand('copy') via a temporary offscreen textarea, which
    works in effectively every environment a click handler runs in.

        st.markdown(ticker_copy_html("AAPL", style="font-size:1.05rem;font-weight:700;"),
                    unsafe_allow_html=True)
    """
    js = (
        "var el=this;"
        "function _flash(){"
        "var o=el.dataset.orig||el.textContent;el.dataset.orig=o;"
        "el.textContent='\\u2713 copied';el.style.color='var(--green-bright)';"
        "setTimeout(function(){el.textContent=o;el.style.color='';},900);"
        "}"
        "function _fallback(t){"
        "var ta=document.createElement('textarea');ta.value=t;"
        "ta.style.position='fixed';ta.style.opacity='0';ta.style.top='0';ta.style.left='0';"
        "document.body.appendChild(ta);ta.focus();ta.select();"
        "try{document.execCommand('copy');}catch(e){}"
        "document.body.removeChild(ta);"
        "}"
        f"if(navigator.clipboard&&navigator.clipboard.writeText){{"
        f"navigator.clipboard.writeText('{symbol}').then(_flash).catch(function(){{_fallback('{symbol}');_flash();}});"
        f"}}else{{_fallback('{symbol}');_flash();}}"
    )
    return (
        f'<span class="ticker-copy" style="{style}" '
        f'title="Click to copy {symbol}" onclick="{js}">{symbol}</span>'
    )


def render_section_header(num, title):
    """Numbered section heading: "01 — Title" with a thin trailing line.

    Drop-in replacement for st.subheader() at the top of a sub-tab's major
    section — pairs with the `.section-header-num` CSS above. `num` can be
    an int (auto zero-padded) or a string if you want custom numbering.

        render_section_header(1, "Today's Picks")
        render_section_header("02", "Predictions vs Actuals")
    """
    import streamlit as st
    label = f"{int(num):02d}" if isinstance(num, int) else str(num)
    st.markdown(
        f'<div class="section-header-num">'
        f'<span class="sh-num">{label} —</span>'
        f'<span class="sh-title">{title}</span>'
        f'<span class="sh-line"></span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_labeled_divider(text):
    """"— Filters —" style divider: thin line, small-caps label, thin line.

    Use in place of a bare st.markdown("---") between named blocks on a
    page, so the break reads as a designed transition rather than a
    leftover Markdown rule.

        render_labeled_divider("Filters")
    """
    import streamlit as st
    st.markdown(
        f'<div class="labeled-divider">'
        f'<div class="ld-line"></div>'
        f'<div class="ld-text">{text}</div>'
        f'<div class="ld-line"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# Count-up animation for st.metric values. st.markdown-injected <script>
# tags run inside Streamlit's sandboxed iframe and can't reliably reach the
# real metric DOM nodes, so this uses components.v1.html (its own iframe)
# and reaches out via `window.parent.document` — a well-worn pattern for
# touching the main app DOM from a components iframe. Best-effort: it
# parses whatever text is in each stMetricValue node (stripping a
# currency/percent affix), so anything that doesn't look like "$1,234.5" or
# "87%" is left alone rather than guessed at.
_COUNT_UP_JS = """
<script>
(function() {
  function processNode(node) {
    if (node._cuAnimating) return;
    var text = (node.textContent || '').trim();
    var m = text.match(/^([^0-9\\-]*)(-?[0-9]*\\.?[0-9]+)(.*)$/);
    if (!m) return;
    var prefix = m[1] || '', suffix = m[3] || '';
    var end = parseFloat(m[2]);
    if (isNaN(end)) return;
    var decimals = (m[2].split('.')[1] || '').length;
    if (node._cuLastVal === end) return;
    var start = (node._cuLastVal === undefined) ? 0 : node._cuLastVal;
    node._cuLastVal = end;
    node._cuAnimating = true;
    var duration = 700, t0 = null;
    function step(ts) {
      if (!t0) t0 = ts;
      var p = Math.min((ts - t0) / duration, 1);
      var eased = 1 - Math.pow(1 - p, 3);
      var cur = start + (end - start) * eased;
      node.textContent = prefix + cur.toFixed(decimals) + suffix;
      if (p < 1) { requestAnimationFrame(step); }
      else { node.textContent = prefix + end.toFixed(decimals) + suffix; node._cuAnimating = false; }
    }
    requestAnimationFrame(step);
  }
  function scan() {
    try {
      var doc = window.parent.document;
      doc.querySelectorAll('div[data-testid="stMetricValue"]').forEach(function(container) {
        var valueEl = container.querySelector('div') || container;
        processNode(valueEl);
      });
    } catch (e) { /* cross-origin or DOM not ready yet — skip this pass */ }
  }
  try {
    var target = window.parent.document.body;
    new MutationObserver(scan).observe(target, {childList: true, subtree: true, characterData: true});
  } catch (e) {}
  scan();
  setTimeout(scan, 300);
})();
</script>
"""


def inject_count_up_script():
    """Call once per page render (e.g. near the top of main()) to make every
    st.metric value animate up from 0 instead of appearing instantly."""
    import streamlit.components.v1 as components
    components.html(_COUNT_UP_JS, height=0, width=0)


# Cursor-follow glow — updates --mx/--my (page-level glow position),
# --px/--py (ambient blob parallax offset), and --cx/--cy (per-card local
# glow position, set on the hovered .data-card itself) all from a single
# rAF-throttled mousemove handler.
# Same iframe-to-parent pattern as _COUNT_UP_JS: components.html runs this
# in its own sandboxed iframe, so it reaches the real page through
# window.parent.document rather than touching its own (invisible) document.
# Skipped entirely for reduced-motion/touch, matching the CSS media queries
# that also hide/disable the glow layers for those users.
_MOUSE_GLOW_JS = """
<script>
(function() {
  try {
    var doc = window.parent.document;
    var root = doc.documentElement;
    var reduced = window.parent.matchMedia('(prefers-reduced-motion: reduce)').matches
               || window.parent.matchMedia('(hover: none)').matches;
    if (reduced) return;
    var pending = false, lastX = null, lastY = null, hoveredCard = null;
    // Parallax divisor: larger = more subtle. 40px of mouse travel from
    // center yields ~1px of blob shift at PARALLAX_DIV=40.
    var PARALLAX_DIV = 40, PARALLAX_MAX = 14;
    function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
    function apply() {
      pending = false;
      if (lastX === null) return;
      var w = window.parent.innerWidth, h = window.parent.innerHeight;

      // Page-level cursor glow position.
      root.style.setProperty('--mx', lastX + 'px');
      root.style.setProperty('--my', lastY + 'px');

      // Ambient blob parallax — offset from viewport center, scaled down
      // and clamped so blobs only ever lean a few px, never chase fully.
      var px = clamp((lastX - w / 2) / PARALLAX_DIV, -PARALLAX_MAX, PARALLAX_MAX);
      var py = clamp((lastY - h / 2) / PARALLAX_DIV, -PARALLAX_MAX, PARALLAX_MAX);
      root.style.setProperty('--px', px.toFixed(1) + 'px');
      root.style.setProperty('--py', py.toFixed(1) + 'px');

      // Per-card local glow position, only while a card is hovered.
      if (hoveredCard) {
        var rect = hoveredCard.getBoundingClientRect();
        var cx = ((lastX - rect.left) / rect.width) * 100;
        var cy = ((lastY - rect.top) / rect.height) * 100;
        hoveredCard.style.setProperty('--cx', cx.toFixed(1) + '%');
        hoveredCard.style.setProperty('--cy', cy.toFixed(1) + '%');
      }
    }
    doc.addEventListener('mousemove', function(e) {
      lastX = e.clientX; lastY = e.clientY;
      hoveredCard = e.target.closest ? e.target.closest('.data-card') : null;
      if (!pending) { pending = true; window.parent.requestAnimationFrame(apply); }
    }, { passive: true });
  } catch (e) { /* cross-origin or DOM not ready — effects just stay static */ }
})();
</script>
"""


def inject_mouse_glow_script():
    """Call once per page render to wire up all cursor-driven effects:
    the page-level background glow, the ambient-blob parallax lean, and
    the per-card proximity glow (see DASHBOARD_CSS: the
    `[data-testid="stAppViewContainer"]::after/::before` rules and
    `.data-card::before`). No-op for reduced-motion or touch/no-hover
    users."""
    import streamlit.components.v1 as components
    components.html(_MOUSE_GLOW_JS, height=0, width=0)


def render_empty_state(message, glyph="◇"):
    """Styled "no data" placeholder — pairs with `.empty-state` CSS above.

    Use in place of a bare st.info()/st.warning() where the user has hit a
    genuine dead end (no rows for this filter/date), not for transient
    warnings, so the terminal look doesn't break for the most common
    non-error state a data dashboard hits.

        render_empty_state("No BUY / STRONG BUY signals for this date.")
    """
    import streamlit as st
    st.markdown(
        f'<div class="empty-state">'
        f'<div class="es-glyph">{glyph}</div>'
        f'<div class="es-msg">{message}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_indicator_grid(items):
    """Compact label/value grid for dense field lists (e.g. a technical
    indicator group with 10-24 fields) — pairs with the `.indicator-snapshot`
    CSS above. Drop-in replacement for a `st.columns(4)` grid of `st.metric`
    calls, which turns into a wall of bordered cards once a group has more
    than a handful of fields.

    items: list of (label, display_value, kind) tuples, where kind is
        "true" / "false" for boolean fields (colored accordingly) or None
        for plain numeric/text values.

        render_indicator_grid([
            ("RSI", "61.2", None),
            ("EMA20 > EMA50", "Yes", "true"),
        ])
    """
    import streamlit as st
    cells = []
    for label, value, kind in items:
        value_cls = " is-true" if kind == "true" else (" is-false" if kind == "false" else "")
        cells.append(
            f'<div class="is-cell">'
            f'<div class="is-label">{label}</div>'
            f'<div class="is-value{value_cls}">{value}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="indicator-snapshot">{"".join(cells)}</div>', unsafe_allow_html=True)


def render_hero_metric(label, value, sub=None, accent="cyan", glyph=None):
    """One visually-dominant number for the single stat a section is really
    about — pairs with the `.hero-metric` CSS above.

    Every st.metric on a page currently renders at the same weight, so on a
    dense tab (5-7 metrics in a row) nothing tells the eye which one to look
    at first. This renders one large, bright, glow-accented value — sized
    and colored distinctly from the surrounding st.metric grid — so the
    single number a partner should see in the first second (best performer,
    top pick's confidence, total return, ...) actually reads as the
    headline instead of competing at equal size with five supporting ones.

    accent: "cyan" (brand/neutral), "green" (gain), "red" (loss), or "amber".

        render_hero_metric("Best Performer", "+18.4%", sub="AAPL · today", accent="green")
    """
    import streamlit as st
    color_var = {
        "cyan":  "var(--cyan)",
        "green": "var(--green-bright)",
        "red":   "var(--red-bright)",
        "amber": "var(--amber-bright)",
    }.get(accent, "var(--cyan)")
    glow_var = {
        "cyan":  "var(--cyan-glow)",
        "green": "rgba(52, 211, 153, 0.16)",
        "red":   "rgba(248, 113, 113, 0.16)",
        "amber": "rgba(251, 191, 36, 0.16)",
    }.get(accent, "var(--cyan-glow)")
    glyph_html = f'<span class="hero-metric-glyph">{glyph}</span>' if glyph else ""
    sub_html = f'<div class="hero-metric-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="hero-metric" style="--hero-color:{color_var};--hero-glow:{glow_var};">'
        f'<div class="hero-metric-label">{label}</div>'
        f'<div class="hero-metric-value">{glyph_html}{value}</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_skeleton_rows(n=3, height=70, gap=8):
    """n stacked shimmering placeholder rows, shown while a slow load runs.

    Pairs with the `.skeleton` CSS (shimmer keyframe already defined above).
    Use with st.empty() so the placeholder can be cleared once real data is
    ready, e.g.:

        placeholder = st.empty()
        with placeholder.container():
            render_skeleton_rows(3)
        data = slow_fetch()
        placeholder.empty()
    """
    import streamlit as st
    rows_html = "".join(
        f'<div class="skeleton" style="height:{height}px;margin-bottom:{gap}px;"></div>'
        for _ in range(n)
    )
    st.markdown(rows_html, unsafe_allow_html=True)
