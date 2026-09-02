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
    /* Genuine blue — used for the BUY signal specifically, since --cyan
       above is brass/gold and was visually indistinguishable from HOLD's
       amber wherever both appeared side by side (badges, live table). */
    --blue-bright: #38bdf8;
    --blue-dim:    rgba(56, 189, 248, 0.12);
    --blue-border: rgba(56, 189, 248, 0.34);

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

/* ── Session-aware ambient tint ────────────────────────────────────────────
   Reads the `data-market-session` attribute the live-clock script now sets
   on <body> (values mirror the status-dot classes: 'live' / 'warning' /
   'idle' — see _LIVE_CLOCK_JS's `update()`). Nudges the ambient wash's
   warmth to match what's actually happening: a touch brighter/greener
   while the market is open, a warm amber lean pre/post-market, and a
   slightly cooler, quieter wash once it's closed for the day. Deliberately
   subtle — a mood shift you'd only consciously notice switching tabs at
   different times of day, not a theme change. */
body[data-market-session="live"] .stApp {
    background-image:
        radial-gradient(ellipse 80% 50% at 0% 0%,   rgba(16,185,129,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 60% 45% at 100% 100%, rgba(224,168,60,0.05) 0%, transparent 55%) !important;
}
body[data-market-session="warning"] .stApp {
    background-image:
        radial-gradient(ellipse 80% 50% at 0% 0%,   rgba(224,168,60,0.07) 0%, transparent 55%),
        radial-gradient(ellipse 60% 45% at 100% 100%, rgba(224,168,60,0.04) 0%, transparent 55%) !important;
}
body[data-market-session="idle"] .stApp {
    background-image:
        radial-gradient(ellipse 80% 50% at 0% 0%,   rgba(134,149,171,0.04) 0%, transparent 55%),
        radial-gradient(ellipse 60% 45% at 100% 100%, rgba(134,149,171,0.03) 0%, transparent 55%) !important;
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

/* ── Idle drift ─────────────────────────────────────────────────────────────
   Set by the `mit-idle` class the mouse-glow script (below) toggles onto
   <body> after ~20s of no mouse/key activity, cleared on the next
   interaction. Purely a "still awake, just waiting" tell: the ambient wash
   drifts a little faster and the ticker tape picks up pace, then both
   settle back to normal the instant you touch the page again. */
body.mit-idle [data-testid="stAppViewContainer"]::before {
    animation-duration: 14s !important;
}
body.mit-idle .ticker-tape-track {
    animation-duration: calc(var(--ticker-duration, 30s) * 0.55) !important;
}

/* ── Matrix mode (Konami code easter egg) ────────────────────────────────
   Toggled for 10s by the Konami-code listener in _MOUSE_GLOW_JS. Full
   green-on-black swap — purely novelty, reverts itself, doesn't touch any
   real data or layout, just palette + a flicker. */
body.mit-matrix-mode {
    animation: matrixFlickerIn 0.15s steps(2, end);
}
body.mit-matrix-mode .stApp {
    background-color: #000502 !important;
}
body.mit-matrix-mode [data-testid="stAppViewContainer"]::before,
body.mit-matrix-mode [data-testid="stAppViewContainer"]::after {
    opacity: 0 !important;
}
body.mit-matrix-mode * {
    border-color: rgba(34,255,120,0.35) !important;
}
body.mit-matrix-mode .stTabs [aria-selected="true"]::after,
body.mit-matrix-mode .gauge-fill,
body.mit-matrix-mode .bar-fill {
    background: #22ff78 !important;
    box-shadow: 0 0 8px rgba(34,255,120,0.6) !important;
}
body.mit-matrix-mode h1, body.mit-matrix-mode h2, body.mit-matrix-mode h3,
body.mit-matrix-mode p, body.mit-matrix-mode span, body.mit-matrix-mode div,
body.mit-matrix-mode label {
    color: #22ff78 !important;
    text-shadow: 0 0 3px rgba(34,255,120,0.35) !important;
}
@keyframes matrixFlickerIn {
    0%   { filter: brightness(2) saturate(0); }
    50%  { filter: brightness(0.6) saturate(0.5); }
    100% { filter: brightness(1) saturate(1); }
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

    /* Container-query sizing base — lets stMetricValue below scale its
       font-size off THIS element's actual rendered width (via `cqw` units)
       instead of a fixed rem value, so values that would otherwise get
       ellipsis-truncated (e.g. "+24.35%" in a narrow 7-up column row)
       shrink to fit instead of getting cut off. */
    container-type: inline-size !important;
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
    white-space: normal !important;
    overflow-wrap: anywhere !important;
}

/* Big, legible numbers — heavier weight for at-a-glance reading.
   font-size is a clamp() driven by `cqw` (container-query width) units so
   it scales with the ACTUAL rendered width of the stMetric card (set via
   `container-type: inline-size` above) rather than a fixed rem value —
   a value in a cramped 7-column row shrinks automatically instead of
   getting clipped with "…". `overflow-wrap: anywhere` is a last-resort
   fallback if a value is still too wide even at the clamp floor. */
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] > div,
div[data-testid="stMetricValue"] div {
    font-family: var(--font-body) !important;
    font-size: clamp(1.05rem, 15cqw, 2rem) !important;
    font-weight: 500 !important;
    color: var(--text-0) !important;
    letter-spacing: -0.02em !important;
    line-height: 1.15 !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    overflow-wrap: anywhere !important;
}

div[data-testid="stMetricDelta"],
div[data-testid="stMetricDelta"] > div,
div[data-testid="stMetricDelta"] div {
    font-family: var(--font-body) !important;
    font-size: clamp(0.62rem, 8cqw, 0.78rem) !important;
    font-weight: 500 !important;
    margin-top: 7px !important;
    white-space: normal !important;
    overflow-wrap: anywhere !important;
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
    font-size: clamp(0.85rem, 15cqw, 1.45rem) !important;
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
    container-type: inline-size;
}
.hero-metric:hover {
    transform: translateY(-2px);
    border-color: var(--hero-color, var(--cyan));
    background: var(--bg-3);
    box-shadow: 0 0 0 1px var(--hero-glow, var(--cyan-glow)),
                0 8px 28px -6px var(--hero-glow, var(--cyan-glow)),
                0 0 46px var(--hero-glow, var(--cyan-glow));
}
.hero-metric::before {
    content: '';
    position: absolute; inset: -40%;
    background: radial-gradient(circle, var(--hero-glow, var(--cyan-glow)) 0%, transparent 65%);
    opacity: 0.35;
    animation: heroAuraIdle 3.6s ease-in-out infinite;
    pointer-events: none;
    transition: opacity 0.22s ease;
}
@keyframes heroAuraIdle {
    0%, 100% { opacity: 0.22; transform: scale(1); }
    50%      { opacity: 0.42; transform: scale(1.03); }
}
.hero-metric:hover::before {
    animation: heroAuraPulse 2.2s ease-in-out infinite;
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
    font-family: var(--font-display); font-size: clamp(1.6rem, 8cqw, 2.6rem); font-weight: 800;
    line-height: 1.05; color: var(--hero-color, var(--cyan)); letter-spacing: -0.01em;
    margin-top: 4px; position: relative;
    text-shadow: 0 0 28px var(--hero-glow, var(--cyan-glow));
    transition: text-shadow 0.22s ease;
    overflow-wrap: anywhere;
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
    background: transparent !important;
    border-radius: 0 !important;
    padding: 0 0 2px 0 !important;
    gap: 2px !important;
    border: none !important;
    border-bottom: 1px solid var(--border-mid) !important;
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
        padding: 8px 12px 8px 24px !important;
        font-size: 0.68rem !important;
    }
    .stTabs [role="tab"]::before,
    .stTabs [data-baseweb="tab"]::before {
        left: 8px !important;
        font-size: 0.56rem !important;
    }
}

/* Base tab state — deliberately NOT the glossy gold-gradient pill-with-glow
   look (that "raised chip + drop shadow + saturated glow" combo is the
   single most recognizable AI-dashboard tell). Instead: flat ghost buttons
   that sit on the track almost invisibly until touched, each carrying its
   own tiny index tick (a numbered dash, not a generic dot) that only
   colors in once the tab is hovered/active — a small hand-placed detail
   a template wouldn't bother with. Counter-driven, so it needs zero markup
   changes and stays in sync if tabs are added/removed. */
.stTabs [role="tablist"],
.stTabs [data-baseweb="tab-list"] {
    counter-reset: dashtab !important;
}
.stTabs [role="tab"],
.stTabs [data-baseweb="tab"] {
    counter-increment: dashtab;
    position: relative !important;
    background: transparent !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid transparent !important;
    font-family: var(--font-display) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    padding: 9px 18px 9px 30px !important;
    margin-bottom: 0 !important;
    cursor: pointer !important;
    box-shadow: none !important;
    transform: translateY(0) scale(1) !important;
    transition: color 0.18s ease, background 0.18s ease,
        border-color 0.18s ease, transform 0.15s ease !important;
}
/* The index tick — "01", "02"... — sits left of the label, dim by default,
   ignites amber on hover/active. Reads as an editorial nav rail, not a
   button factory. */
.stTabs [role="tab"]::before,
.stTabs [data-baseweb="tab"]::before {
    content: counter(dashtab, decimal-leading-zero) !important;
    position: absolute !important;
    left: 10px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.6rem !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    color: var(--text-3, #64748b) !important;
    opacity: 0.55 !important;
    transition: color 0.18s ease, opacity 0.18s ease !important;
}
/* Underline indicator — grows from the center outward instead of a flat
   static bar, so selection reads as a deliberate motion cue rather than
   a CSS default. */
.stTabs [role="tab"]::after,
.stTabs [data-baseweb="tab"]::after {
    content: "" !important;
    position: absolute !important;
    left: 14px !important;
    right: 14px !important;
    bottom: 4px !important;
    height: 2px !important;
    border-radius: 2px !important;
    background: var(--cyan) !important;
    transform: scaleX(0) !important;
    transform-origin: center !important;
    transition: transform 0.22s cubic-bezier(.4,0,.2,1) !important;
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
    color: var(--text-2, #94a3b8) !important;
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
    background: rgba(255,255,255,0.035) !important;
    border-color: rgba(255,255,255,0.08) !important;
    transform: translateY(-1px) !important;
}
.stTabs [role="tab"]:hover::before,
.stTabs [data-baseweb="tab"]:hover::before {
    color: var(--cyan) !important;
    opacity: 0.9 !important;
}
.stTabs [role="tab"]:hover::after,
.stTabs [data-baseweb="tab"]:hover::after {
    transform: scaleX(0.4) !important;
    background: rgba(224,168,60,0.5) !important;
}

.stTabs [aria-selected="true"] {
    background: var(--cyan-dim) !important;
    border-color: var(--cyan-border) !important;
}
.stTabs [aria-selected="true"],
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] div,
.stTabs [aria-selected="true"] span {
    color: var(--cyan) !important;
    font-weight: 700 !important;
}
.stTabs [aria-selected="true"]::before {
    color: var(--cyan) !important;
    opacity: 1 !important;
}
.stTabs [aria-selected="true"]::after {
    transform: scaleX(1) !important;
    background: var(--cyan) !important;
    box-shadow: 0 0 8px rgba(224,168,60,0.55) !important;
}
.stTabs [aria-selected="true"]:hover { background: var(--cyan-dim) !important; }

/* Remove baseweb's own underline/highlight bar — the ::after tick is the
   signal now, and letting both run at once double-draws the indicator. */
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; background: transparent !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 24px !important; }

/* Suppress .arrow_right leak — same failure mode as the expander toggle
   icon above (a ligature-text `[data-testid="stIconMaterial"]` span
   winning a specificity fight against text-styling rules and rendering
   its raw name on top of the tab label). Using the same `display: none`
   fix here too, since the font-size/width trick previously used is the
   one that turned out not to hold up. */
.stTabs [role="tab"] svg,
.stTabs [role="tab"] [data-testid="stIconMaterial"],
.stTabs [data-baseweb="tab"] svg,
.stTabs [data-baseweb="tab"] [data-testid="stIconMaterial"] {
    display: none !important;
}

/* ── Utility buttons (Refresh / Clear Cache) ───────────────────────────────
   Deliberately the quietest buttons in the app — plain text, no fill, no
   accent border by default. These are maintenance actions a user reaches
   for occasionally, not something that should visually compete with the
   data above it. Color only shows up on hover, as a hint rather than a
   standing claim on the user's attention. */
.btn-utility .stButton > button {
    background: transparent !important;
    color: var(--text-3) !important;
    border: 1px solid transparent !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: none !important;
    padding: 4px 10px !important;
    box-shadow: none !important;
}
.btn-utility.util-refresh .stButton > button:hover {
    color: var(--cyan) !important;
    border-color: var(--cyan-border) !important;
    background: var(--cyan-dim) !important;
}
.btn-utility.util-danger .stButton > button:hover {
    color: var(--red-bright) !important;
    border-color: rgba(239,68,68,0.35) !important;
    background: var(--red-dim) !important;
}
.btn-utility.util-confirm .stButton > button {
    color: var(--red-bright) !important;
    border-color: rgba(239,68,68,0.3) !important;
}
.btn-utility.util-confirm .stButton > button:hover { background: var(--red-dim) !important; }
.btn-utility.util-cancel .stButton > button:hover {
    color: var(--text-0) !important;
    border-color: rgba(255,255,255,0.2) !important;
}
/* The expander that tucks the whole control strip away — a plain small
   caption-style trigger instead of the app's usual bordered-card expander,
   so it reads as "there's a small utility drawer here" rather than another
   content section. */
.cache-controls-expander div[data-testid="stExpander"] {
    border: none !important;
    background: transparent !important;
}
.cache-controls-expander div[data-testid="stExpander"]:hover { border: none !important; }
.cache-controls-expander details summary {
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-3) !important;
    padding: 2px 0 !important;
}
.cache-controls-expander details summary:hover { color: var(--text-2) !important; }
.cache-controls-expander div[data-testid="stExpanderDetails"] { padding: 8px 0 0 !important; }

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
div[data-testid="stExpander"] details summary > div,
div[data-testid="stExpander"] details summary p,
div[data-testid="stExpander"] details summary span {
    font-family: var(--font-body) !important; font-size: 0.8rem !important;
    font-weight: 500 !important; color: var(--text-1) !important;
    letter-spacing: 0.02em !important; margin: 0 !important; padding: 0 !important;
}

/* Streamlit's own toggle icon has been fought here twice now — first as an
   <svg>, then as a `[data-testid="stIconMaterial"]` ligature-text span that
   kept winning a CSS-specificity fight against the label-text rule above
   and rendering its raw text ("keyboard_arrow_right") on top of the title.
   Chasing its exact current markup/testid is a losing game since it keeps
   changing between Streamlit versions. Instead: remove whatever Streamlit
   renders for it from layout entirely with `display: none` (which — unlike
   font-size/color tricks — can't lose a specificity fight, since there's
   nothing left to compete over), and draw our own chevron from scratch as
   a `summary::before` pseudo-element that we fully control. This is
   version-proof: it no longer matters what Streamlit's icon looks like
   internally, because it's never in the DOM's rendered output at all. */
div[data-testid="stExpander"] details summary svg,
div[data-testid="stExpander"] details summary [data-testid="stExpanderToggleIcon"],
div[data-testid="stExpander"] details summary [data-testid="stIconMaterial"] {
    display: none !important;
}
div[data-testid="stExpander"] details summary::before {
    content: '❯';
    order: -1;
    flex-shrink: 0;
    display: inline-block;
    font-family: var(--font-body) !important;
    font-size: 0.68rem !important;
    color: var(--text-2);
    transition: transform 0.2s ease, color 0.15s ease;
}
div[data-testid="stExpander"] details[open] summary::before {
    transform: rotate(90deg);
    color: var(--cyan);
}
div[data-testid="stExpander"] details summary:hover::before {
    color: var(--text-1);
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
.badge-blue  { background: var(--blue-dim);   color: var(--blue-bright);  border: 1px solid var(--blue-border); }

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

/* ── Live price flash ───────────────────────────────────────────────────────
   Fires once (not looping) when a live quote's price actually changes
   between reruns, so a tick reads as motion rather than just a number
   silently being different than it was a second ago. Direction picked in
   Python (price up vs down since last render) and passed in as a class. */
.price-flash-up   { animation: priceFlashUp 0.7s ease-out; border-radius: 4px; }
.price-flash-down { animation: priceFlashDown 0.7s ease-out; border-radius: 4px; }
@keyframes priceFlashUp {
    0%   { background: rgba(16,185,129,0.28); }
    100% { background: transparent; }
}
@keyframes priceFlashDown {
    0%   { background: rgba(239,68,68,0.28); }
    100% { background: transparent; }
}

/* ── Live indicator dot ─────────────────────────────────────────────────────
   Small breathing dot next to "Live Market View" so the auto-refresh is
   visible at a glance, not just claimed in a caption. */
.live-pulse-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green-bright);
    box-shadow: 0 0 0 0 rgba(16,185,129,0.5);
    animation: livePulseDot 2s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes livePulseDot {
    0%   { box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }
    70%  { box-shadow: 0 0 0 6px rgba(16,185,129,0); }
    100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
}

/* ── Live table row tint ────────────────────────────────────────────────────
   A faint full-row green/red wash keyed to day change %, so scanning the
   table for "what's moving" doesn't require reading every number — the
   color alone sorts winners from losers at a glance. Intentionally subtle
   (low alpha) so it reads as a background cue, not a colored block that
   competes with the badges/text sitting on top of it. */
.mkt-row-up   { background: rgba(16,185,129,0.05); border-radius: 6px; }
.mkt-row-down { background: rgba(239,68,68,0.05);  border-radius: 6px; }

/* ── Exchange chip ──────────────────────────────────────────────────────────
   Small colored dot + short code so the listing venue is scannable at a
   glance across a table, instead of only readable as plain text — also
   makes it visually obvious when a symbol resolves to a less-common venue
   (e.g. CBOE/BATS) rather than the NASDAQ default. */
.exch-chip {
    display: inline-flex; align-items: center; gap: 4px;
    font-family: var(--font-body); font-size: 0.62rem; letter-spacing: 0.05em;
    color: var(--text-1); text-transform: uppercase;
}
.exch-chip .exch-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }

/* ── Sparkline ──────────────────────────────────────────────────────────── */
.spark-wrap { display: inline-block; vertical-align: middle; line-height: 0; }

/* Draw-in reveal for sparkline polylines — pairs with the `pathLength="100"`
   attribute _sparkline_svg() now sets on every polyline, which normalizes
   the line's total length to 100 regardless of its actual point count/span.
   That lets a single fixed dasharray/keyframe work for every sparkline
   without computing each polyline's real pixel length in Python. Runs once
   on mount (new DOM node each Streamlit rerun), not looping — a live table
   that kept re-drawing itself every render would be more distracting than
   informative. */
.spark-draw {
    stroke-dasharray: 100;
    animation: sparkDraw 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
@keyframes sparkDraw {
    from { stroke-dashoffset: 100; }
    to   { stroke-dashoffset: 0; }
}
@media (prefers-reduced-motion: reduce) {
    .spark-draw { animation: none; stroke-dashoffset: 0; }
}

/* ── Market-table row hover ─────────────────────────────────────────────────
   Scoped via the .mkt-live-table-scope marker (see _render_live_market_table
   in tab_ml_predictions.py) using a general-sibling selector, so only the
   per-symbol data rows get the hover affordance — not the header row above
   them (rendered before the marker) or unrelated st.columns() layouts
   elsewhere on the page (outside this marker's sibling scope). */
.mkt-live-table-scope ~ div[data-testid="stHorizontalBlock"] {
    border-radius: var(--radius-sm);
    transition: background 0.15s ease, box-shadow 0.15s ease;
}
.mkt-live-table-scope ~ div[data-testid="stHorizontalBlock"]:hover {
    background: var(--bg-2);
    box-shadow: inset 3px 0 0 0 var(--cyan-border);
}

/* ── Ticker tape ────────────────────────────────────────────────────────────
   Continuously scrolling strip of symbols/changes — ambient "trading floor"
   texture. Pure CSS keyframe translate, duplicated content so the loop is
   seamless (second copy picks up exactly where the first's translate ends). */
.ticker-tape-outer {
    overflow: hidden; width: 100%; white-space: nowrap;
    background: var(--bg-1); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 7px 0;
    mask-image: linear-gradient(90deg, transparent 0%, black 4%, black 96%, transparent 100%);
    -webkit-mask-image: linear-gradient(90deg, transparent 0%, black 4%, black 96%, transparent 100%);
}
.ticker-tape-track {
    display: inline-flex; align-items: center;
    animation: tickerScroll linear infinite;
    animation-duration: var(--ticker-duration, 30s);
}
.ticker-tape-item {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: var(--font-body); font-size: 0.72rem; letter-spacing: 0.03em;
    padding: 0 20px; flex-shrink: 0;
}
.ticker-tape-item .tt-sym  { color: var(--text-0); font-weight: 700; }
.ticker-tape-item .tt-sep  { color: var(--border-mid); }
@keyframes tickerScroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* ── Radial gauge ───────────────────────────────────────────────────────────
   Lightweight SVG arc gauge (no Plotly figure overhead) for a single
   percentage value shown at "instrument panel" prominence — see
   radial_gauge_svg() below. Track + fill are both plain <circle> elements
   with a dash-offset trick, animated in on first paint like .bar-fill. */
.gauge-wrap { position: relative; display: inline-flex; align-items: center; justify-content: center; }
.gauge-fill {
    transform-origin: center; transform: rotate(-90deg);
    animation: gaugeGrow 0.9s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
@keyframes gaugeGrow {
    from { stroke-dashoffset: var(--gauge-start); }
    to   { stroke-dashoffset: var(--gauge-end); }
}
.gauge-label {
    position: absolute; text-align: center;
    font-family: var(--font-body); font-weight: 700; color: var(--text-0);
}

/* ── Correlation heatmap ────────────────────────────────────────────────────
   Pairs with correlation_heatmap_svg() below. Plain SVG grid, not a Plotly
   heatmap — see that function's docstring for why. Cells fade in with a
   quick stagger so a 10x10 grid doesn't just pop in as a flat block. */
.corr-heatmap-wrap { display: inline-block; overflow-x: auto; max-width: 100%; }
.corr-heatmap-label {
    font-family: var(--font-body); font-size: 10px; fill: var(--text-2);
}
.corr-heatmap-value {
    font-family: var(--font-body); font-size: 10px; font-weight: 600;
    pointer-events: none;
}
.corr-heatmap-cell {
    animation: corrCellIn 0.4s ease-out backwards;
    transition: stroke 0.15s ease;
}
.corr-heatmap-cell:hover { stroke: var(--cyan-border); stroke-width: 1.5; }
@keyframes corrCellIn {
    from { opacity: 0; transform: scale(0.85); transform-origin: center; }
    to   { opacity: 1; transform: scale(1); }
}

/* ── Boot sequence ──────────────────────────────────────────────────────────
   One-shot "connecting to feed" flicker shown before the header on a
   session's first render only (gated in Python via session_state — see
   render_boot_sequence()). Collapses itself via max-height/opacity so it
   never leaves a permanent gap once the animation finishes. */
.boot-sequence {
    overflow: hidden; font-family: 'DM Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: var(--cyan);
    opacity: 0; max-height: 22px; margin-bottom: 8px;
    animation: bootReveal 1.9s ease-in-out forwards;
}
.boot-sequence::before { content: '> '; opacity: 0.6; }
.boot-cursor {
    display: inline-block; width: 6px; height: 11px; background: var(--cyan);
    margin-left: 2px; vertical-align: -1px;
    animation: bootBlink 0.5s steps(1) 4;
}
@keyframes bootReveal {
    0%   { opacity: 0; max-height: 22px; }
    10%  { opacity: 1; }
    70%  { opacity: 1; max-height: 22px; }
    100% { opacity: 0; max-height: 0; margin-bottom: 0; }
}
@keyframes bootBlink {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0; }
}

/* ── Leaderboard rank medals ────────────────────────────────────────────────
   Used by the Daily Winners leaderboard rows in place of the old plain
   pandas-styled dataframe (its default `.background_gradient` colormap
   rendered light-green cells that broke the dark terminal look). */
.rank-medal {
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 50%; font-size: 0.85rem;
    font-family: var(--font-display); font-weight: 700; flex-shrink: 0;
}
.rank-medal.gold   { background: rgba(224,168,60,0.18); border: 1px solid rgba(224,168,60,0.5); }
.rank-medal.silver { background: rgba(203,213,225,0.14); border: 1px solid rgba(203,213,225,0.4); }
.rank-medal.bronze { background: rgba(217,119,6,0.16);  border: 1px solid rgba(217,119,6,0.42); }
.rank-medal.plain  { background: var(--bg-3); border: 1px solid var(--border-mid); color: var(--text-2); font-size: 0.72rem; }

.leaderboard-row {
    display: flex; align-items: center; gap: 14px;
    padding: 10px 14px; margin-bottom: 6px;
    background: var(--bg-2); border: 1px solid var(--border);
    border-radius: var(--radius-sm); transition: border-color 0.15s, transform 0.15s;
}
.leaderboard-row:hover { border-color: var(--cyan-border); transform: translateX(2px); }
.leaderboard-row.lb-top1 { border-color: rgba(224,168,60,0.4); background: linear-gradient(90deg, rgba(224,168,60,0.06), transparent 60%); }


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

/* ── Countdown urgency ──────────────────────────────────────────────────────
   #mit-countdown is styled identically whether it reads "3h 58m" or "0:45" —
   these two states make the last stretch before a session transition
   (market open/close, after-hours end) actually read as urgent instead of
   being just another quiet monospace label. Thresholds are applied in JS
   (inject_live_clock_script) against the same countdown target it's already
   computing; Python's first-paint value gets the matching class too, set in
   dashboard.py, so there's no flash of the wrong color before the JS ticker
   takes over a moment later. */
#mit-countdown.cd-warn {
    color: var(--amber-bright) !important;
}
#mit-countdown.cd-critical {
    color: var(--red-bright) !important;
    animation: countdownCritical 1s ease-in-out infinite;
}
@keyframes countdownCritical {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.55; }
}
@media (prefers-reduced-motion: reduce) {
    #mit-countdown.cd-critical { animation: none; }
}
/* One-shot pulse fired by _LIVE_CLOCK_JS the instant the countdown crosses
   a round minute marker (5:00, 1:00) — a brief bigger flash on top of the
   steady-state warn/critical color, so hitting those marks reads as an
   event rather than just another ticking second. Self-removing (JS strips
   the class after the animation completes), so it never lingers. */
#mit-countdown.cd-tick-flash {
    animation: countdownTickFlash 0.6s ease-out !important;
}
@keyframes countdownTickFlash {
    0%   { text-shadow: 0 0 0 transparent; transform: scale(1); }
    30%  { text-shadow: 0 0 10px currentColor; transform: scale(1.12); }
    100% { text-shadow: 0 0 0 transparent; transform: scale(1); }
}
@media (prefers-reduced-motion: reduce) {
    #mit-countdown.cd-tick-flash { animation: none !important; }
}

/* ── Glitch flash (rare, on live data refresh) ─────────────────────────────
   Toggled briefly by the price-scramble script's MutationObserver whenever
   a batch of prices actually updates — see GLITCH_CHANCE there for the odds.
   A quick RGB-split + scanline flicker instead of a plain refresh, styled
   to match the CRT/terminal conceit rather than looking like a rendering
   bug. Purely decorative, self-removing after ~200ms. */
.mit-glitch-flash {
    animation: mitGlitch 0.22s steps(3, end);
    position: relative;
}
.mit-glitch-flash::after {
    content: '';
    position: absolute; inset: 0;
    background: repeating-linear-gradient(
        0deg, rgba(224,168,60,0.05) 0px, rgba(224,168,60,0.05) 1px,
        transparent 1px, transparent 3px
    );
    pointer-events: none;
    animation: mitGlitchScan 0.22s steps(3, end);
}
@keyframes mitGlitch {
    0%   { transform: translate(0, 0); filter: none; }
    20%  { transform: translate(-2px, 0); filter: drop-shadow(2px 0 0 rgba(239,68,68,0.5)) drop-shadow(-2px 0 0 rgba(52,211,153,0.5)); }
    40%  { transform: translate(2px, 0); }
    60%  { transform: translate(-1px, 0); filter: drop-shadow(1px 0 0 rgba(239,68,68,0.4)) drop-shadow(-1px 0 0 rgba(52,211,153,0.4)); }
    100% { transform: translate(0, 0); filter: none; }
}
@keyframes mitGlitchScan {
    0%   { opacity: 0; }
    30%  { opacity: 1; }
    100% { opacity: 0; }
}
@media (prefers-reduced-motion: reduce) {
    .mit-glitch-flash, .mit-glitch-flash::after { animation: none !important; }
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

/* ── Form section label — small tick + trailing rule ───────────────────────
   Lighter-weight cousin of `.section-header-num` above, sized for labeling
   a form rather than a whole tab section ("Configure your simulation"),
   so the backtest form reads as a deliberately designed control panel
   instead of a bare st.markdown("**bold text**"). */
.form-section-label {
    display: flex; align-items: center; gap: 10px;
    margin: 2px 0 10px;
    font-family: var(--font-body); font-size: 0.74rem; font-weight: 500;
    letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-2);
}
.form-section-label::before {
    content: ""; width: 8px; height: 8px; flex-shrink: 0;
    background: var(--cyan); border-radius: 2px; opacity: 0.85;
    box-shadow: 0 0 8px var(--cyan-glow);
}
.form-section-label::after {
    content: ""; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border-mid), transparent 90%);
}

/* ── Data / search / warning cards ─────────────────────────────────────── */
.data-card {
    background: var(--bg-2); border: 1px solid var(--border-mid);
    border-radius: var(--radius); padding: 16px 18px;
    transition: border-color 0.15s, background 0.15s, box-shadow 0.15s, transform 0.15s ease-out;
    box-shadow: 0 1px 3px rgba(0,0,0,0.28);
    position: relative;
    transform: perspective(900px) rotateX(var(--rx, 0deg)) rotateY(var(--ry, 0deg));
    transform-style: preserve-3d;
}
.data-card:hover {
    border-color: var(--cyan-border);
    background: var(--bg-3);
    box-shadow: 0 10px 24px rgba(0,0,0,0.38);
}
/* Cursor-driven tilt — --rx/--ry are set (in deg, per-card) by the same
   mousemove handler that already drives the --cx/--cy glow position below.
   Capped to ±1.6deg: enough to read as "physically present" without numbers
   swimming or text becoming hard to parse in a data-dense finance UI — a
   full game-UI tilt (5-10deg) was tried and reads as gimmicky here, so this
   stays deliberately subtle. Disabled outright for touch/no-hover and
   reduced-motion, same as the glow. */
@media (hover: none), (prefers-reduced-motion: reduce) {
    .data-card { transform: none !important; }
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
   the page-level cursor glow above and (see .data-card --rx/--ry) the
   card's tilt. Kept as a separate layer from the tilt itself since the
   glow position is in local %, while the tilt is in deg. */
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

/* ── Primary CTA submit (e.g. "Run Backtest") ──────────────────────────────
   The one submit button per form that actually kicks off real work
   (fetching bars, running the simulation) — everything else on the page
   is just shaping the config for this moment. Filled instead of ghost,
   taller, and a size step up in weight/letter-spacing so it reads as
   "the button" the instant you scan the form, without reaching for the
   raised-chip/drop-shadow/saturated-glow combo the rest of this file
   deliberately avoids (see the tab-nav comment above). The glow only
   shows up on hover/focus, not at rest — emphasis you notice when you're
   about to press it, not a standing visual claim on the page. */
.st-key-cta_run_backtest .stFormSubmitButton > button,
.st-key-cta_run_comparison .stFormSubmitButton > button {
    position: relative !important;
    overflow: hidden !important;
    background: linear-gradient(115deg, #7c3aed 0%, #a855f7 32%, #e0a83c 68%, #f0c05e 100%) !important;
    background-size: 220% 220% !important;
    background-position: 0% 50% !important;
    border: 1px solid rgba(168, 85, 247, 0.55) !important;
    color: #0a0710 !important;
    font-family: var(--font-display) !important;
    font-weight: 800 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.09em !important;
    padding: 13px 20px !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.35) inset,
                0 6px 20px -6px rgba(139, 92, 246, 0.55),
                0 6px 20px -8px rgba(224, 168, 60, 0.4) !important;
    transition: background-position 0.4s ease, box-shadow 0.2s ease,
        transform 0.12s ease, border-color 0.15s ease !important;
}
/* subtle diagonal sheen that sweeps across on hover */
.st-key-cta_run_backtest .stFormSubmitButton > button::before,
.st-key-cta_run_comparison .stFormSubmitButton > button::before {
    content: "" !important;
    position: absolute !important;
    top: 0; left: -60%; width: 40%; height: 100%;
    background: linear-gradient(115deg, transparent 0%, rgba(255,255,255,0.55) 50%, transparent 100%) !important;
    transform: skewX(-20deg) !important;
    transition: left 0.55s ease !important;
    pointer-events: none !important;
}
.st-key-cta_run_backtest .stFormSubmitButton > button:hover,
.st-key-cta_run_comparison .stFormSubmitButton > button:hover {
    background-position: 100% 50% !important;
    border-color: rgba(224, 168, 60, 0.75) !important;
    color: #0a0710 !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.45) inset,
                0 0 0 1px rgba(224, 168, 60, 0.4),
                0 10px 28px -6px rgba(139, 92, 246, 0.65),
                0 10px 30px -8px rgba(224, 168, 60, 0.55),
                0 0 40px rgba(168, 85, 247, 0.25) !important;
    transform: translateY(-2px) scale(1.01) !important;
}
.st-key-cta_run_backtest .stFormSubmitButton > button:hover::before,
.st-key-cta_run_comparison .stFormSubmitButton > button:hover::before {
    left: 120% !important;
}
.st-key-cta_run_backtest .stFormSubmitButton > button:active,
.st-key-cta_run_comparison .stFormSubmitButton > button:active {
    transform: translateY(0) scale(0.99) !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.25) inset,
                0 3px 12px -6px rgba(139, 92, 246, 0.5) !important;
}
.st-key-cta_run_backtest .stFormSubmitButton > button:focus-visible,
.st-key-cta_run_comparison .stFormSubmitButton > button:focus-visible {
    outline: 2px solid rgba(168, 85, 247, 0.7) !important;
    outline-offset: 2px !important;
}

/* ── Plotly charts ──────────────────────────────────────────────────────── */
div[data-testid="stPlotlyChart"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important; border: 1px solid var(--border-mid) !important;
}
/* Thin crosshair cursor over the plot area only (not the modebar/legend) —
   Plotly already draws its own hover spike lines on most of these charts,
   so this just makes the cursor itself agree with that "instrument, not a
   picture" framing instead of showing a generic pointer/arrow. */
div[data-testid="stPlotlyChart"] .nsewdrag {
    cursor: crosshair !important;
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

/* Signature readout strip — the kind of throwaway build/session/uptime
   line a real terminal app has that nobody asked for but that sells the
   "instrument, not a webpage" conceit. Sits under .app-footer, smaller and
   quieter still, monospace, wide-tracked, easy to skip past. */
.mit-signature {
    margin-top: 10px; text-align: center;
    font-family: var(--font-body); font-size: 0.6rem;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--text-3); opacity: 0.55;
}
.mit-signature span { margin: 0 8px; }


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
div[data-testid="stMetricValue"] { font-size: clamp(0.8rem, 15cqw, 1.15rem) !important; white-space: normal !important; overflow-wrap: anywhere !important; }
.stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"] { gap: 2px !important; }
.stTabs [role="tab"], .stTabs [data-baseweb="tab"] { padding: 8px 14px 8px 26px !important; }
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


# Per-venue dot color for exchange_chip_html — deliberately distinct hues
# so NASDAQ/NYSE/AMEX/BATS-CBOE/OTC are each recognizable at a glance
# rather than reading as the same gray text everywhere.
_EXCHANGE_DOT_COLORS = {
    "NASDAQ":   "#38bdf8",
    "NYSE":     "#e0a83c",
    "AMEX":     "#2dd4bf",
    "BATS":     "#a78bfa",
    "CBOE":     "#a78bfa",
    "OTC":      "#8695ab",
}


def exchange_chip_html(exchange: str) -> str:
    """Small colored-dot + code chip for a listing venue, e.g. for the
    "Stock" column of a live table. Falls back to a neutral gray dot for
    any venue not in _EXCHANGE_DOT_COLORS rather than guessing a color.

        st.markdown(exchange_chip_html("CBOE"), unsafe_allow_html=True)
    """
    ex = str(exchange or "").strip().upper()
    color = _EXCHANGE_DOT_COLORS.get(ex, "var(--text-3)")
    return (
        f'<span class="exch-chip">'
        f'<span class="exch-dot" style="background:{color};"></span>{ex or "—"}'
        f'</span>'
    )


def ticker_tape_html(items, duration_s=30):
    """Continuously scrolling ticker-tape strip of (symbol, change_pct)
    tuples — ambient "trading floor" texture, pairs with `.ticker-tape-*`
    CSS above. Purely decorative; not a data table, so no sort/filter.

    items: list of (symbol, change_pct) — change_pct may be None.
    duration_s: full loop duration; longer = slower scroll.

        st.markdown(
            ticker_tape_html([("AAPL", 1.24), ("TSLA", -2.1)]),
            unsafe_allow_html=True,
        )
    """
    if not items:
        return ""

    def _cell(sym, chg):
        if chg is None:
            chg_html = '<span style="color:var(--text-2);">—</span>'
        else:
            color = "var(--green-bright)" if chg >= 0 else "var(--red-bright)"
            arrow = "▲" if chg >= 0 else "▼"
            chg_html = f'<span style="color:{color};">{arrow} {abs(chg):.2f}%</span>'
        return (
            f'<span class="ticker-tape-item">'
            f'<span class="tt-sym">{sym}</span>{chg_html}'
            f'<span class="tt-sep">/</span></span>'
        )

    cells = "".join(_cell(s, c) for s, c in items)
    # Duplicated once so the -50% translateX loop is seamless.
    return (
        f'<div class="ticker-tape-outer">'
        f'<div class="ticker-tape-track" style="--ticker-duration:{duration_s}s;">'
        f'{cells}{cells}'
        f'</div></div>'
    )


def radial_gauge_svg(percent, size=76, stroke=7, color=None, label=None) -> str:
    """Lightweight SVG arc gauge for a single 0-100 value — an
    "instrument panel" alternative to `.bar-fill` for the one number a
    section most wants to emphasize (e.g. model confidence on a top pick).
    No Plotly figure overhead, so it's cheap to render many of in a list.

        st.markdown(radial_gauge_svg(82.4), unsafe_allow_html=True)
    """
    import math
    pct = max(0.0, min(100.0, float(percent)))
    if color is None:
        color = (
            "var(--green-bright)" if pct >= 70
            else "var(--cyan)" if pct >= 50
            else "var(--amber-bright)"
        )
    r = (size - stroke) / 2
    cx = cy = size / 2
    circumference = 2 * math.pi * r
    filled = circumference * (pct / 100.0)
    gap = circumference - filled
    label_text = label if label is not None else f"{pct:.0f}%"

    return (
        f'<div class="gauge-wrap" style="width:{size}px;height:{size}px;">'
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
        f'stroke="var(--bg-4)" stroke-width="{stroke}"/>'
        f'<circle class="gauge-fill" cx="{cx}" cy="{cy}" r="{r}" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-dasharray="{circumference:.2f}" '
        f'style="--gauge-start:{circumference:.2f};--gauge-end:{gap:.2f};" '
        f'stroke-dashoffset="{gap:.2f}"/>'
        f'</svg>'
        f'<div class="gauge-label" style="font-size:{size*0.2:.0f}px;">{label_text}</div>'
        f'</div>'
    )


def correlation_heatmap_svg(symbols, matrix, cell=40, label_col=56) -> str:
    """
    Symbol x symbol correlation grid — diverging brass/emerald scale so it
    matches the accent palette instead of a generic red/blue heatmap. Takes
    a plain list-of-lists / 2D array `matrix` of Pearson correlations in
    [-1, 1] aligned to `symbols` (matrix[i][j] = corr(symbols[i], symbols[j])).
    Deliberately NOT a Plotly heatmap: this is meant to sit inline next to
    other small SVG widgets (the sparklines / radial gauges above) with the
    same crisp, no-toolbar, no-hover-lag feel — a real Plotly figure here
    would drag in its own font/hover chrome and look like a different app
    bolted on next to the hand-drawn widgets.

    Color logic: 0 correlation renders as the plain card background (no
    signal), positive correlation ramps toward emerald, negative ramps
    toward red — same convention already used for gain/loss everywhere
    else, so a trader reads "green cell = moves together, red cell = moves
    opposite" without a legend.

        st.markdown(
            correlation_heatmap_svg(["AAPL", "MSFT", "TSLA"], corr_matrix),
            unsafe_allow_html=True,
        )
    """
    n = len(symbols)
    if n < 2 or not matrix or len(matrix) != n:
        return ""

    def _color(v):
        v = max(-1.0, min(1.0, float(v)))
        if v >= 0:
            # 0 -> transparent bg-4, 1 -> full emerald
            return f"rgba(52,211,153,{0.06 + 0.62 * v:.3f})"
        return f"rgba(239,68,68,{0.06 + 0.62 * -v:.3f})"

    w = label_col + cell * n
    h = label_col + cell * n
    parts = [f'<div class="corr-heatmap-wrap"><svg width="{w}" height="{h}" '
             f'viewBox="0 0 {w} {h}" class="corr-heatmap-svg">']

    # Column labels (rotated) and row labels
    for j, sym in enumerate(symbols):
        cx = label_col + cell * j + cell / 2
        parts.append(
            f'<text x="{cx}" y="{label_col - 8}" text-anchor="middle" '
            f'transform="rotate(-40 {cx} {label_col - 8})" class="corr-heatmap-label">{sym}</text>'
        )
    for i, sym in enumerate(symbols):
        cy = label_col + cell * i + cell / 2 + 4
        parts.append(
            f'<text x="{label_col - 8}" y="{cy}" text-anchor="end" class="corr-heatmap-label">{sym}</text>'
        )

    # Cells — only render below/on the diagonal (correlation matrices are
    # symmetric; showing both triangles just doubles the visual noise for
    # zero extra information).
    for i in range(n):
        for j in range(n):
            if j > i:
                continue
            v = matrix[i][j]
            x = label_col + cell * j
            y = label_col + cell * i
            is_diag = i == j
            fill = "var(--bg-4)" if is_diag else _color(v)
            txt = "1.00" if is_diag else f"{v:+.2f}"
            txt_color = "var(--text-3)" if is_diag else ("var(--green-bright)" if v >= 0 else "var(--red-bright)")
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="3" '
                f'fill="{fill}" stroke="var(--border)" stroke-width="1" class="corr-heatmap-cell"/>'
                f'<text x="{x + cell/2}" y="{y + cell/2 + 4}" text-anchor="middle" '
                f'class="corr-heatmap-value" fill="{txt_color}">{txt}</text>'
            )

    parts.append('</svg></div>')
    return "".join(parts)



# partner reloading the app across days sees some variety instead of the
# exact same string every single time. All phrased in the same
# "connecting to X" register so none of them reads as more/less important
# than the others; this is flavor text, not a real status log.
_BOOT_LINES = [
    "Connecting to market data feed",
    "Syncing order book",
    "Calibrating model weights",
    "Establishing terminal uplink",
    "Reconciling overnight signals",
    "Warming up the screening engine",
]

# Rare easter-egg lines — picked instead of the normal pool roughly 1 in 50
# sessions. Deliberately break character slightly (the normal pool is all
# straight "connecting to X" phrasing); a wink for anyone who reloads
# enough times to notice, never so often it dilutes the terminal conceit.
_BOOT_LINES_RARE = [
    "Definitely not checking Reddit first",
    "Bribing the RNG for a green day",
    "Politely asking the market to cooperate",
    "Recalling why we do this to ourselves",
]


def render_boot_sequence(key="_boot_shown", lines=None):
    """One-shot "connecting to market data feed"-style flicker, shown only
    on a session's very first render (gated via st.session_state[key]) —
    pure cosmetic flavor for the "Market Intelligence Terminal" conceit.
    Safe to call on every rerun; it's a no-op after the first.

    `lines`, if given, pins the exact text shown instead of picking one at
    random from `_BOOT_LINES` (with a small chance of `_BOOT_LINES_RARE`
    instead) — the random pick itself is also frozen into session_state
    once made, so a mid-session rerun never shows a second, different line.

        render_boot_sequence()   # call once, near the very top of main()
    """
    import random
    import streamlit as st
    if st.session_state.get(key):
        return
    st.session_state[key] = True
    if lines:
        text = lines
    elif random.random() < 0.02:
        text = random.choice(_BOOT_LINES_RARE)
    else:
        text = random.choice(_BOOT_LINES)
    st.markdown(
        f'<div class="boot-sequence">{text}<span class="boot-cursor"></span></div>',
        unsafe_allow_html=True,
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
    var pending = false, lastX = null, lastY = null, hoveredCard = null, lastTiltedCard = null;
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

      // Per-card local glow position + tilt, only while a card is hovered.
      // Tilt is capped at TILT_MAX_DEG (see .data-card CSS comment for why
      // this stays small) and inverted on the Y axis so the card leans
      // *toward* the cursor rather than away from it, matching the glow's
      // "light source follows pointer" logic.
      if (hoveredCard) {
        var rect = hoveredCard.getBoundingClientRect();
        var cx = ((lastX - rect.left) / rect.width) * 100;
        var cy = ((lastY - rect.top) / rect.height) * 100;
        hoveredCard.style.setProperty('--cx', cx.toFixed(1) + '%');
        hoveredCard.style.setProperty('--cy', cy.toFixed(1) + '%');

        var TILT_MAX_DEG = 1.6;
        var nx = (cx - 50) / 50; // -1 .. 1 across card width
        var ny = (cy - 50) / 50; // -1 .. 1 down card height
        var rx = clamp(-ny * TILT_MAX_DEG, -TILT_MAX_DEG, TILT_MAX_DEG);
        var ry = clamp(nx * TILT_MAX_DEG, -TILT_MAX_DEG, TILT_MAX_DEG);
        hoveredCard.style.setProperty('--rx', rx.toFixed(2) + 'deg');
        hoveredCard.style.setProperty('--ry', ry.toFixed(2) + 'deg');
        if (hoveredCard !== lastTiltedCard) {
          if (lastTiltedCard) {
            lastTiltedCard.style.setProperty('--rx', '0deg');
            lastTiltedCard.style.setProperty('--ry', '0deg');
          }
          lastTiltedCard = hoveredCard;
        }
      } else if (lastTiltedCard) {
        // Cursor left every card this frame — relax the last one back flat
        // instead of leaving it frozen at its final tilt angle.
        lastTiltedCard.style.setProperty('--rx', '0deg');
        lastTiltedCard.style.setProperty('--ry', '0deg');
        lastTiltedCard = null;
      }
    }
    doc.addEventListener('mousemove', function(e) {
      lastX = e.clientX; lastY = e.clientY;
      hoveredCard = e.target.closest ? e.target.closest('.data-card') : null;
      if (!pending) { pending = true; window.parent.requestAnimationFrame(apply); }
    }, { passive: true });

    // Idle drift — flips body.mit-idle on after IDLE_MS of no mouse/key
    // activity (pairs with the `body.mit-idle` CSS above), clears it on
    // the next interaction. A single shared timer rather than one per
    // effect since nothing here needs sub-second precision.
    var IDLE_MS = 20000;
    var idleTimer = null;
    function markActive() {
      doc.body.classList.remove('mit-idle');
      if (idleTimer) window.parent.clearTimeout(idleTimer);
      idleTimer = window.parent.setTimeout(function() {
        doc.body.classList.add('mit-idle');
      }, IDLE_MS);
    }
    doc.addEventListener('mousemove', markActive, { passive: true });
    doc.addEventListener('keydown', markActive, { passive: true });
    doc.addEventListener('click', markActive, { passive: true });
    markActive();

    // Konami code easter egg — up up down down left right left right b a.
    // Flips body.mit-matrix-mode on for 10s (CSS swap below), then reverts
    // on its own. Pure novelty, no functional effect, resets progress on
    // any wrong key so it can't trigger by accident.
    var KONAMI = ['ArrowUp','ArrowUp','ArrowDown','ArrowDown','ArrowLeft','ArrowRight','ArrowLeft','ArrowRight','b','a'];
    var konamiPos = 0, matrixTimer = null;
    doc.addEventListener('keydown', function(e) {
      var key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      konamiPos = (key === KONAMI[konamiPos]) ? konamiPos + 1 : (key === KONAMI[0] ? 1 : 0);
      if (konamiPos === KONAMI.length) {
        konamiPos = 0;
        doc.body.classList.add('mit-matrix-mode');
        if (matrixTimer) window.parent.clearTimeout(matrixTimer);
        matrixTimer = window.parent.setTimeout(function() {
          doc.body.classList.remove('mit-matrix-mode');
        }, 10000);
      }
    });
  } catch (e) { /* cross-origin or DOM not ready — effects just stay static */ }
})();
</script>
"""


def inject_mouse_glow_script():
    """Call once per page render to wire up all cursor-driven effects:
    the page-level background glow, the ambient-blob parallax lean, the
    per-card proximity glow, and the per-card tilt (see DASHBOARD_CSS: the
    `[data-testid="stAppViewContainer"]::after/::before` rules and
    `.data-card::before` / `.data-card` transform). No-op for
    reduced-motion or touch/no-hover users."""
    import streamlit.components.v1 as components
    components.html(_MOUSE_GLOW_JS, height=0, width=0)


# Live ET clock + session countdown. Deliberately NOT based on the visitor's
# local clock/timezone — everything here is computed against a fixed
# "America/New_York" wall clock via Intl, ticking every second client-side
# (Streamlit only re-runs Python on interaction, so a per-second Python
# re-render isn't an option). Ports the same NYSE holiday math as the Python
# `_nyse_holidays()` above, plus the same hand-maintained EARLY_CLOSE_DATES
# half-day table as dashboard.py, so the countdown target skips
# weekends/holidays and uses the right close time on early-close days too.
# Same iframe -> window.parent.document pattern as the other injected
# scripts.
#
# State machine: whichever session is currently active is never counted
# down to (that would be counting down to something already happening) —
# instead the countdown always targets the *next* transition:
#   Market Open       -> counts down to today's close (16:00, or the
#                         early-close time on a half day)
#   Pre-Market         -> counts down to today's 09:30 open
#   After-Hours        -> counts down to today's 20:00 after-hours end
#   Closed/weekend/holiday -> counts down to the next trading day's 04:00
#                             pre-market open
_LIVE_CLOCK_JS = """
<script>
(function() {
  try {
    var doc = window.parent.document;
    var TZ = 'America/New_York';
    var holidayCache = {};
    var lastTickFlashSec = null; // last remaining-seconds value that fired the round-number flash

    // Mirrors Python's EARLY_CLOSE_DATES / EARLY_CLOSE_TABLE_THROUGH in
    // dashboard.py — keep these two in sync when the table is updated.
    // Values are [hour, minute] of the early close, in ET, keyed 'YYYY-MM-DD'.
    var EARLY_CLOSE_DATES = {
      '2025-07-03':  [13, 0],
      '2025-11-28':  [13, 0],
      '2025-12-24':  [13, 0],
      '2026-11-27':  [13, 0],
      '2026-12-24':  [13, 0],
      '2027-11-26':  [13, 0]
    };
    var EARLY_CLOSE_TABLE_THROUGH = Date.UTC(2027, 11, 31); // pseudo-UTC, ET wall-clock space
    var EARLY_CLOSE_WARN_WINDOW_MS = 90 * 86400000;

    function pad2(n) { return n < 10 ? '0' + n : '' + n; }
    function ymdUTC(d) {
      return d.getUTCFullYear() + '-' + pad2(d.getUTCMonth() + 1) + '-' + pad2(d.getUTCDate());
    }
    function earlyCloseFor(d) {
      return EARLY_CLOSE_DATES[ymdUTC(d)] || null;
    }

    // weekday: JS convention, Sunday=0 ... Saturday=6.
    function nthWeekday(year, month, weekday, n) {
      var d = new Date(Date.UTC(year, month - 1, 1));
      var offset = (weekday - d.getUTCDay() + 7) % 7;
      return new Date(Date.UTC(year, month - 1, 1 + offset + 7 * (n - 1)));
    }
    function lastWeekday(year, month, weekday) {
      var d = new Date(Date.UTC(year, month, 0)); // last day of `month`
      var offset = (d.getUTCDay() - weekday + 7) % 7;
      d.setUTCDate(d.getUTCDate() - offset);
      return d;
    }
    function easterSunday(year) {
      var a = year % 19, b = Math.floor(year / 100), c = year % 100;
      var d = Math.floor(b / 4), e = b % 4, f = Math.floor((b + 8) / 25);
      var g = Math.floor((b - f + 1) / 3), h = (19 * a + b - d - g + 15) % 30;
      var i = Math.floor(c / 4), k = c % 4;
      var l = (32 + 2 * e + 2 * i - h - k) % 7;
      var m = Math.floor((a + 11 * h + 22 * l) / 451);
      var month = Math.floor((h + l - 7 * m + 114) / 31);
      var day = ((h + l - 7 * m + 114) % 31) + 1;
      return new Date(Date.UTC(year, month - 1, day));
    }
    function observed(d) {
      var wd = d.getUTCDay(), r = new Date(d);
      if (wd === 6) r.setUTCDate(r.getUTCDate() - 1); // Sat -> preceding Fri
      if (wd === 0) r.setUTCDate(r.getUTCDate() + 1); // Sun -> following Mon
      return r;
    }
    function nyseHolidays(year) {
      if (holidayCache[year]) return holidayCache[year];
      var goodFriday = easterSunday(year);
      goodFriday.setUTCDate(goodFriday.getUTCDate() - 2);
      var list = [
        observed(new Date(Date.UTC(year, 0, 1))),   // New Year's Day
        nthWeekday(year, 1, 1, 3),                  // MLK Day
        nthWeekday(year, 2, 1, 3),                  // Presidents' Day
        goodFriday,                                 // Good Friday
        lastWeekday(year, 5, 1),                    // Memorial Day
        observed(new Date(Date.UTC(year, 5, 19))),  // Juneteenth
        observed(new Date(Date.UTC(year, 6, 4))),   // Independence Day
        nthWeekday(year, 9, 1, 1),                  // Labor Day
        nthWeekday(year, 11, 4, 4),                 // Thanksgiving
        observed(new Date(Date.UTC(year, 11, 25))), // Christmas
      ];
      var set = {};
      list.forEach(function(d) { set[ymdUTC(d)] = true; });
      holidayCache[year] = set;
      return set;
    }
    function isTradingDay(d) {
      var dow = d.getUTCDay();
      if (dow === 0 || dow === 6) return false;
      return !nyseHolidays(d.getUTCFullYear())[ymdUTC(d)];
    }

    // Reads the current wall-clock time in America/New_York and returns it
    // as a Date object constructed via Date.UTC from those fields. It is
    // NOT a real UTC instant — it's a "pretend UTC" stand-in so we can do
    // plain date arithmetic entirely in ET wall-clock space (weekday
    // checks, adding days, HH:MM comparisons, ms diffs for the countdown).
    var etFmt = new Intl.DateTimeFormat('en-US', {
      timeZone: TZ, hour12: false, year: 'numeric', month: '2-digit',
      day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
    function nowET() {
      var parts = {};
      etFmt.formatToParts(new Date()).forEach(function(p) {
        if (p.type !== 'literal') parts[p.type] = parseInt(p.value, 10);
      });
      if (parts.hour === 24) parts.hour = 0; // midnight edge case in some engines
      return new Date(Date.UTC(parts.year, parts.month - 1, parts.day, parts.hour, parts.minute, parts.second));
    }

    // Next trading-day 04:00 ET at/after `fromMs` (strictly after `fromMs`
    // if `fromMs` itself lands exactly on a candidate).
    function nextPreMarketOpen(fromMs) {
      var d = new Date(fromMs);
      var candidate = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate(), 4, 0, 0);
      if (candidate <= fromMs) {
        var nd = new Date(candidate);
        nd.setUTCDate(nd.getUTCDate() + 1);
        candidate = Date.UTC(nd.getUTCFullYear(), nd.getUTCMonth(), nd.getUTCDate(), 4, 0, 0);
      }
      while (!isTradingDay(new Date(candidate))) {
        var nd2 = new Date(candidate);
        nd2.setUTCDate(nd2.getUTCDate() + 1);
        candidate = Date.UTC(nd2.getUTCFullYear(), nd2.getUTCMonth(), nd2.getUTCDate(), 4, 0, 0);
      }
      return candidate;
    }

    function fmtCountdown(ms) {
      if (ms < 0) ms = 0;
      var totalSec = Math.floor(ms / 1000);
      var days = Math.floor(totalSec / 86400);
      var hours = Math.floor((totalSec % 86400) / 3600);
      var mins = Math.floor((totalSec % 3600) / 60);
      var secs = totalSec % 60;
      if (days > 0) return days + 'd ' + pad2(hours) + ':' + pad2(mins) + ':' + pad2(secs);
      return pad2(hours) + ':' + pad2(mins) + ':' + pad2(secs);
    }

    var WEEKDAY_ABBR = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
    var MONTH_ABBR = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

    function update() {
      var dotEl = doc.getElementById('mit-status-dot');
      var labelEl = doc.getElementById('mit-status-label');
      var dateEl = doc.getElementById('mit-date');
      var timeEl = doc.getElementById('mit-time');
      var cdEl = doc.getElementById('mit-countdown');
      var warnEl = doc.getElementById('mit-stale-warning');
      if (!dotEl || !labelEl || !dateEl || !timeEl) return; // not rendered (yet)

      var et = nowET();
      var etMs = et.getTime();
      var dow = et.getUTCDay();
      var hour = et.getUTCHours(), minute = et.getUTCMinutes();
      var tradingDay = isTradingDay(et);
      var earlyClose = tradingDay ? earlyCloseFor(et) : null;
      var closeHour = earlyClose ? earlyClose[0] : 16;
      var closeMinute = earlyClose ? earlyClose[1] : 0;

      var afterOpen = (hour > 9) || (hour === 9 && minute >= 30);
      var beforeClose = (hour < closeHour) || (hour === closeHour && minute < closeMinute);
      var marketOpen = tradingDay && afterOpen && beforeClose;
      var preMarket = tradingDay && ((hour >= 4 && hour < 9) || (hour === 9 && minute < 30));
      var afterHoursStartMin = closeHour * 60 + closeMinute;
      var nowMin = hour * 60 + minute;
      var afterHours = tradingDay && (nowMin >= afterHoursStartMin && nowMin < 20 * 60);
      var isHoliday = (dow !== 0 && dow !== 6) && !tradingDay;

      var dotCls, label, color, nextMs, nextLabel;
      if (marketOpen) {
        dotCls = 'live'; label = earlyClose ? 'Open (Early Close)' : 'Open'; color = 'var(--green-bright)';
        nextMs = Date.UTC(et.getUTCFullYear(), et.getUTCMonth(), et.getUTCDate(), closeHour, closeMinute, 0);
        nextLabel = 'Closes in';
      } else if (preMarket) {
        dotCls = 'warning'; label = 'Pre-Market'; color = 'var(--amber-bright)';
        nextMs = Date.UTC(et.getUTCFullYear(), et.getUTCMonth(), et.getUTCDate(), 9, 30, 0);
        nextLabel = 'Opens in';
      } else if (afterHours) {
        dotCls = 'warning'; label = 'After Hours'; color = 'var(--amber-bright)';
        nextMs = Date.UTC(et.getUTCFullYear(), et.getUTCMonth(), et.getUTCDate(), 20, 0, 0);
        nextLabel = 'Ends in';
      } else if (isHoliday) {
        dotCls = 'idle'; label = 'Holiday'; color = 'var(--text-2)';
        nextMs = nextPreMarketOpen(etMs);
        nextLabel = 'Pre-market in';
      } else {
        dotCls = 'idle'; label = 'Closed'; color = 'var(--text-2)';
        nextMs = nextPreMarketOpen(etMs);
        nextLabel = 'Pre-market in';
      }

      dotEl.className = 'status-dot ' + dotCls;
      labelEl.textContent = label;
      labelEl.style.color = color;
      // Session-aware palette hook — exposes the exact same state the
      // status dot already uses as a body attribute, so CSS elsewhere can
      // shift ambient tone (glow intensity, background warmth) between
      // live/pre-post/closed without a second parallel state machine.
      doc.body.setAttribute('data-market-session', dotCls);
      dateEl.textContent = WEEKDAY_ABBR[dow] + ' ' + pad2(et.getUTCDate()) + ' ' + MONTH_ABBR[et.getUTCMonth()] + ' ' + et.getUTCFullYear();
      timeEl.innerHTML = pad2(hour) + ':' + pad2(minute) + ':' + pad2(et.getUTCSeconds()) +
        ' <span style="font-size:0.75rem;color:var(--text-2);">ET</span>';
      if (cdEl) {
        var remainMs = nextMs - etMs;
        cdEl.textContent = nextLabel + ' ' + fmtCountdown(remainMs);
        // Urgency only makes sense counting down to something imminent
        // (open/close/after-hours-end) — not the "next trading day" case,
        // which can legitimately span a whole weekend and shouldn't glow
        // red for 60 hours straight.
        cdEl.classList.remove('cd-warn', 'cd-critical');
        if (nextLabel !== 'Pre-market in') {
          if (remainMs <= 5 * 60 * 1000)       cdEl.classList.add('cd-critical');
          else if (remainMs <= 15 * 60 * 1000) cdEl.classList.add('cd-warn');

          // Round-number tick flash — fires once, right as the countdown
          // crosses 5:00 or 1:00, rather than on every render (checked
          // against the previous tick's remaining-seconds so a re-render
          // with the same second doesn't re-fire the animation).
          var remainSec = Math.floor(remainMs / 1000);
          if ((remainSec === 300 || remainSec === 60) && remainSec !== lastTickFlashSec) {
            lastTickFlashSec = remainSec;
            cdEl.classList.remove('cd-tick-flash');
            void cdEl.offsetWidth; // restart animation if class is re-added
            cdEl.classList.add('cd-tick-flash');
          }
        }
      }
      if (warnEl) {
        warnEl.style.display = (EARLY_CLOSE_TABLE_THROUGH - etMs) <= EARLY_CLOSE_WARN_WINDOW_MS ? 'flex' : 'none';
      }
    }

    update();
    setInterval(update, 1000);
  } catch (e) { /* cross-origin or DOM not ready — clock just stays static */ }
})();
</script>
"""


def inject_live_clock_script():
    """Call once per page render to drive the header's ET clock: ticks the
    displayed time every second, and keeps the market-status pill + session
    countdown (see #mit-date / #mit-time / #mit-status-dot /
    #mit-status-label / #mit-countdown in dashboard.py) live without
    requiring a Streamlit rerun. Always shows America/New_York time
    regardless of the visitor's own timezone."""
    import streamlit.components.v1 as components
    components.html(_LIVE_CLOCK_JS, height=0, width=0)


# Digit-scramble reveal for live price cells. Fires on any element carrying
# `.price-flash-up` or `.price-flash-down` (see chart_utils/tab_ml_predictions
# — those classes are only ever applied by Python when a symbol's live price
# actually changed since the previous render, so this never fires on an
# unrelated rerun). Streamlit re-renders the cell with its *final* text
# already in place — there's no earlier value to animate from client-side —
# so this fakes the "settling" motion by holding on random digits for a few
# short frames before revealing the real (already-correct) text, rather than
# trying to interpolate between two known numbers like the count-up script
# does for st.metric.
_PRICE_SCRAMBLE_JS = """
<script>
(function() {
  var SCRAMBLE_FRAMES = 5, SCRAMBLE_MS = 45;
  function scramble(el) {
    if (el._scrDone) return;
    el._scrDone = true;
    var final = el.textContent;
    if (!/[0-9]/.test(final)) return;
    var i = 0;
    var iv = setInterval(function() {
      i++;
      if (i >= SCRAMBLE_FRAMES) {
        el.textContent = final;
        clearInterval(iv);
        return;
      }
      el.textContent = final.replace(/[0-9]/g, function() {
        return Math.floor(Math.random() * 10);
      });
    }, SCRAMBLE_MS);
  }
  function scan(root) {
    try {
      root.querySelectorAll('.price-flash-up, .price-flash-down').forEach(scramble);
    } catch (e) {}
  }
  try {
    var doc = window.parent.document;
    // Rare glitch flash — whenever a batch of price flashes lands (i.e. a
    // real data refresh happened), there's a small chance the live-quotes
    // table gets a brief RGB-split/scanline flicker instead of a plain
    // settle. Deliberately rare (GLITCH_CHANCE) and short (GLITCH_MS) —
    // this is meant to read as "ooh, neat" once in a while, not as
    // something wrong with the page.
    var GLITCH_CHANCE = 0.08, GLITCH_MS = 220;
    function maybeGlitch() {
      if (Math.random() > GLITCH_CHANCE) return;
      var table = doc.querySelector('.mkt-live-table-scope');
      var target = table ? table.parentElement : null;
      if (!target || target._glitching) return;
      target._glitching = true;
      target.classList.add('mit-glitch-flash');
      window.parent.setTimeout(function() {
        target.classList.remove('mit-glitch-flash');
        target._glitching = false;
      }, GLITCH_MS);
    }
    new MutationObserver(function(muts) {
      var sawFlash = false;
      muts.forEach(function(m) {
        m.addedNodes && m.addedNodes.forEach(function(n) {
          if (n.nodeType !== 1) return;
          if (n.matches && (n.matches('.price-flash-up') || n.matches('.price-flash-down'))) { scramble(n); sawFlash = true; }
          if (n.querySelectorAll) scan(n);
        });
      });
      if (sawFlash) maybeGlitch();
    }).observe(doc.body, { childList: true, subtree: true });
    scan(doc);
  } catch (e) { /* cross-origin or DOM not ready — cells just show final value */ }
})();
</script>
"""


def inject_price_scramble_script():
    """Call once per page render to make live price cells that just ticked
    (`.price-flash-up` / `.price-flash-down`, set in
    _render_live_market_table) settle in through a brief digit-scramble
    instead of just silently being a different number than a moment ago."""
    import streamlit.components.v1 as components
    components.html(_PRICE_SCRAMBLE_JS, height=0, width=0)


def render_signature_footer():
    """
    Small monospace "build info" strip — a session id, a fake-but-stable
    build revision derived from today's date, and a rough uptime clock.
    None of it is real infrastructure data (there's no actual deploy
    pipeline stamping a build number here); it's a hand-placed detail in
    the same spirit as a terminal's version banner, purely to sell the
    "instrument panel" conceit. Session id and boot time are frozen into
    st.session_state on first render so they stay stable for the session
    rather than changing on every rerun.

        render_signature_footer()   # call once, near the very bottom of main()
    """
    import random
    import string
    import time
    import streamlit as st

    if "_mit_session_id" not in st.session_state:
        st.session_state["_mit_session_id"] = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=6)
        )
    if "_mit_boot_time" not in st.session_state:
        st.session_state["_mit_boot_time"] = time.time()

    session_id = st.session_state["_mit_session_id"]
    uptime_s = int(time.time() - st.session_state["_mit_boot_time"])
    mins, secs = divmod(uptime_s, 60)
    hrs, mins = divmod(mins, 60)
    uptime_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
    build_str = "rev." + __import__("datetime").date.today().strftime("%Y.%m.%d") + "-a"

    st.markdown(
        f'<div class="mit-signature">'
        f'<span>{build_str}</span>'
        f'<span>session {session_id}</span>'
        f'<span>up {uptime_str}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


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
