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

</style>
"""

COMPACT_CSS = """
<style>
</style>
"""
