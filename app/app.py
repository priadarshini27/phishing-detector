"""Streamlit live URL checker (dashboard layout).

Run from the repo root:  streamlit run app/app.py
Optional deep link:      http://localhost:8501/?url=example.com/login   (pre-fills and checks)

Safety: the app only parses the text of the URL. It never fetches, opens or resolves it.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.explain import MODEL_PATH, analyze_url, load_model  # noqa: E402

st.set_page_config(page_title="Phishing URL Checker", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------------------------
# Styling. Colours: green = legitimate, red = phishing, amber = suspicious/warnings,
# blue = neutral information and progress.
# Material icons (e.g. the expander arrow) keep their icon font, otherwise their name is drawn
# as text. Feature names and numbers use a monospace font.
# ----------------------------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg: #0d1117; --card: #161b22; --card-2: #1c222b; --border: #2a313c; --border-soft: #222831;
  --text: #e6edf3; --muted: #8b949e;
  --green: #3fb950; --red: #f85149; --amber: #d29922; --blue: #58a6ff;
  --radius: 10px; --pad: 1.1rem; --gap: 1.1rem;
  --display: 'Space Grotesk', 'Inter', system-ui, sans-serif;   /* title, headings, key labels */
  --sans: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;  /* body and UI */
  --mono: 'JetBrains Mono', ui-monospace, Consolas, monospace;  /* technical values */
}

/* Typography.
   1. Inter for everything (wrapped in :where so it has low specificity and the rules below win;
      !important still beats Streamlit's own font rules).
   2. Space Grotesk for the title, section headings and prominent labels.
   3. JetBrains Mono for technical values.
   4. Material icons keep their icon font, otherwise the icon name is drawn as text. */
.stApp :where(*:not([data-testid="stIconMaterial"]):not([class*="material"])) {
  font-family: var(--sans) !important;
}
.stApp .app-header .t, .stApp .sec-title, .stApp .verdict .v-title, .stApp .stat .v,
.stApp [data-testid="stExpander"] summary p, .stApp .stFormSubmitButton button p {
  font-family: var(--display) !important; letter-spacing: -0.01em;
}
.stApp code, .stApp .mono, .stApp .mono * { font-family: var(--mono) !important; letter-spacing: 0; }
.stApp [data-testid="stIconMaterial"], .stApp [class*="material"] {
  font-family: 'Material Symbols Rounded' !important;
}
html, body, .stApp { font-size: 15px; color: var(--text); }
.stApp { background: var(--bg); }

/* Page frame */
header[data-testid="stHeader"] { background: transparent; height: 2.6rem; }
.block-container { padding: 2.4rem 2.2rem 1.2rem 2.2rem !important; max-width: 100% !important; }
div[data-testid="stVerticalBlock"] { gap: 0.6rem; }
div[data-testid="stHorizontalBlock"] { gap: var(--gap); }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }

/* Header */
.app-header { display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; margin: 0 0 0.9rem 0; }
.app-header .t { font-size: 2rem; font-weight: 700; line-height: 1.15; letter-spacing: .2px; }
.app-header .s { font-size: 1.05rem; color: var(--muted); }

/* Cards (bordered st.container) */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > .st-key-url_card),
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > .st-key-ex_card),
.st-key-url_card, .st-key-ex_card {
  background: var(--card); border: 1px solid var(--border-soft) !important; border-radius: var(--radius);
}
.st-key-url_card, .st-key-ex_card { padding: 0.95rem var(--pad) 1rem var(--pad); gap: 0.7rem;
  flex: 1 1 auto; }
/* Make the two cards in row 1 the same height */
div[data-testid="stHorizontalBlock"]:has(.st-key-url_card) { align-items: stretch; }
div[data-testid="stColumn"]:has(.st-key-url_card) > div,
div[data-testid="stColumn"]:has(.st-key-ex_card) > div { height: 100%; }
div[data-testid="stLayoutWrapper"]:has(> .st-key-url_card),
div[data-testid="stLayoutWrapper"]:has(> .st-key-ex_card) { flex: 1 1 auto; display: flex; }
/* Streamlit pulls markdown blocks up with a negative bottom margin; our HTML blocks have no
   trailing paragraph margin, so that would clip them. */
div[data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
.st-key-ex_card div[data-testid="stHorizontalBlock"] { gap: 0.6rem; }

/* Section headings */
.sec-title { font-size: 1.3rem; font-weight: 700; margin: 0; line-height: 1.2; }
.sec-sub { font-size: 0.93rem; color: var(--muted); margin: 0.15rem 0 0 0; line-height: 1.3; }

/* Form, inputs, buttons */
div[data-testid="stForm"] { border: none; padding: 0; }
.stTextInput input {
  font-size: 1.05rem !important; background: var(--card-2) !important; border-radius: 8px !important;
  height: 2.6rem;
}
.stTextInput div[data-baseweb="input"] { border: 1px solid var(--border) !important; border-radius: 8px !important; }
.stButton button, .stFormSubmitButton button {
  height: 2.6rem; min-height: 2.6rem; border-radius: 8px; font-size: 1rem !important; white-space: nowrap;
}
.stButton button { background: var(--card-2); border: 1px solid var(--border); }
.stButton button:hover { border-color: var(--blue); color: var(--blue); }
.stFormSubmitButton button { background: var(--blue) !important; border-color: var(--blue) !important;
  color: #0d1117 !important; }
.stFormSubmitButton button p { font-weight: 700 !important; font-size: 1.05rem; color: #0d1117; }
.stFormSubmitButton button:hover { filter: brightness(1.1); }
.hint { font-size: 0.88rem; color: var(--muted); margin-top: 0.1rem; }

/* Result panels */
.panel { background: var(--card); border: 1px solid var(--border-soft); border-radius: var(--radius);
  padding: 0.95rem var(--pad) var(--pad) var(--pad); height: 100%; }
div[data-testid="stHorizontalBlock"]:has(.panel) { align-items: stretch; }
div[data-testid="stColumn"]:has(.panel) div[data-testid="stMarkdownContainer"],
div[data-testid="stColumn"]:has(.panel) div[data-testid="stMarkdown"],
div[data-testid="stColumn"]:has(.panel) div[data-testid="stElementContainer"],
div[data-testid="stColumn"]:has(.panel) > div,
div[data-testid="stColumn"]:has(.panel) div[data-testid="stMarkdown"] > div { height: 100%; }
.panel .sec-sub { margin-bottom: 0.8rem; }

/* Verdict */
.verdict { border-left: 5px solid var(--c); background: var(--card-2); border-radius: 8px;
  padding: 0.75rem 0.95rem; }
.verdict .v-title { font-size: 1.65rem; font-weight: 700; color: var(--c); line-height: 1.15; }
.verdict .v-sub { margin-top: 0.35rem; line-height: 1.38; font-size: 0.98rem; }
.verdict .v-host { margin-top: 0.55rem; color: var(--muted); font-size: 0.95rem; }
.verdict .v-host code { color: var(--blue); background: transparent; padding: 0; font-size: 0.85rem; }
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 0.7rem; margin-top: 0.8rem; }
.stat { background: var(--card-2); border-radius: 8px; padding: 0.5rem 0.8rem; }
.stat .k { font-size: 0.9rem; color: var(--muted); }
.stat .v { font-size: 1.6rem; font-weight: 600; line-height: 1.2; }
.bar { position: relative; height: 8px; background: var(--card-2); border-radius: 99px; margin-top: 0.85rem; }
.bar .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 99px; background: var(--blue); }
.bar .thr { position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--text); opacity: .7; }
.bar-legend { display: flex; justify-content: space-between; font-size: 0.82rem; color: var(--muted);
  margin-top: 0.3rem; }

/* Reason and red-flag rows */
.row { display: flex; align-items: center; gap: 0.65rem; padding: 0.42rem 0.7rem; margin-bottom: 0.4rem;
  background: var(--card-2); border-radius: 7px; line-height: 1.3; font-size: 0.98rem; }
.row:last-child { margin-bottom: 0; }
.row .ic { flex: none; width: 1rem; text-align: center; font-size: 0.8rem; }
.row .txt { flex: 1; }
.row .pct { flex: none; min-width: 3.6rem; text-align: right; font-size: 0.82rem; }
.up .ic, .up .pct { color: var(--red); }
.down .ic, .down .pct { color: var(--green); }
.flag { border-left: 3px solid var(--amber); }
.flag .ic { color: var(--amber); }
.ok { border-left: 3px solid var(--green); }
.ok .ic { color: var(--green); }
.legend-up { color: var(--red); } .legend-down { color: var(--green); }

/* Feature grid */
.feat-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.35rem 1.4rem; }
.feat { display: flex; align-items: baseline; gap: 0.5rem; padding: 0.22rem 0; font-size: 0.82rem; }
.feat .lead { flex: 1; border-bottom: 1px dotted var(--border); transform: translateY(-0.2rem); }
.feat .n { color: var(--muted); } .feat .val { color: var(--text); font-weight: 600; }
div[data-testid="stExpander"] details { background: var(--card); border: 1px solid var(--border-soft) !important;
  border-radius: var(--radius); }
div[data-testid="stExpander"] summary p { font-size: 1.05rem; font-weight: 600; }

/* Notices */
div[data-testid="stAlert"] { border-radius: var(--radius); }

/* Narrow screens: stack columns, 2-column feature grid */
@media (max-width: 1000px) {
  .feat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .block-container { padding: 2.4rem 1rem 1rem 1rem !important; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

EXAMPLES = {
    "Legitimate": "https://github.com/scikit-learn/scikit-learn",
    "Phishing-looking": "http://secure-paypal-login.verify-account.xyz/@signin?session=88213",
    "Free-hosting lure": "https://emerald-exch-0utlook.firebaseapp.com/",
    "'@' trick": "http://paypal.com@evil-site.ru/confirm",
    "Big-site subdomain": "https://en.wikipedia.org/wiki/Phishing",
}

GREEN, RED, AMBER, BLUE = "#3fb950", "#f85149", "#d29922", "#58a6ff"


@st.cache_resource(show_spinner="Loading model ...")
def get_model():
    return load_model()


def heading(title: str, sub: str) -> str:
    return f'<div class="sec-title">{title}</div><div class="sec-sub">{sub}</div>'


# ---------------------------------------------------------------------------------------------- #
# Result panels (pure HTML so the three columns share one consistent card style)
# ---------------------------------------------------------------------------------------------- #
def verdict_panel(res: dict) -> str:
    p, thr = res["proba_phishing"], res["threshold"]
    if res["red_flag_override"]:
        color, title = AMBER, "⚠ Suspicious"
        sub = (f"The hostname alone looks ordinary to the model ({p:.0%}), but the URL hides its "
               "real destination behind an '@'. Treat it as phishing.")
    elif res["is_phishing"]:
        color, title = RED, "⚠ Phishing"
        sub = f"The model gives this URL a {p:.0%} phishing probability, above the {thr:.0%} threshold."
    elif res["known_domain"] and res["model_is_phishing"]:
        color, title = GREEN, "✓ Legitimate (known domain)"
        sub = (f"<b>{html.escape(res['registered_domain'])}</b> is on the well-known-domains allowlist. "
               f"The model alone scored it {p:.0%}: it over-flags subdomains of big sites because "
               "of a bias in its training data.")
    else:
        color, title = GREEN, "✓ Legitimate"
        sub = f"The model gives this URL a {p:.0%} phishing probability, below the {thr:.0%} threshold."
    fill = min(max(p, 0.0), 1.0) * 100
    return (
        f'<div class="panel">{heading("Verdict", "Model score combined with the app&#39;s rules.")}'
        f'<div class="verdict" style="--c:{color}">'
        f'<div class="v-title">{title}</div><div class="v-sub">{sub}</div>'
        f'<div class="v-host">Analysed hostname: <code>{html.escape(res["host"])}</code></div></div>'
        f'<div class="stats">'
        f'<div class="stat"><div class="k">Phishing probability</div><div class="v">{p:.1%}</div></div>'
        f'<div class="stat"><div class="k">Decision threshold</div><div class="v">{thr:.1%}</div></div>'
        f'</div>'
        f'<div class="bar"><div class="fill" style="width:{fill:.1f}%"></div>'
        f'<div class="thr" style="left:{thr * 100:.1f}%"></div></div>'
        f'<div class="bar-legend"><span>0% · legitimate</span><span>threshold</span>'
        f'<span>phishing · 100%</span></div>'
        f'</div>'
    )


def reasons_panel(res: dict) -> str:
    rows = "".join(
        f'<div class="row {"up" if r["direction"] == "phishing" else "down"}">'
        f'<span class="ic">{"▲" if r["direction"] == "phishing" else "▼"}</span>'
        f'<span class="txt">{html.escape(r["text"])}</span>'
        f'<span class="pct mono">{r["contribution"]:+.1%}</span></div>'
        for r in res["reasons"]
    ) or '<div class="row"><span class="txt">No single factor stood out.</span></div>'
    sub = (f"Starting from a base rate of {res['base_rate']:.0%}. "
           '<span class="legend-up">▲ towards phishing</span> · '
           '<span class="legend-down">▼ towards legitimate</span>')
    return f'<div class="panel">{heading("Why the model thinks so", sub)}{rows}</div>'


def flags_panel(res: dict) -> str:
    rows = "".join(
        f'<div class="row flag"><span class="ic">⚠</span><span class="txt">{html.escape(w)}</span></div>'
        for w in res["warnings"]
    ) or ('<div class="row ok"><span class="ic">✓</span>'
          '<span class="txt">No red flags in the path, query or scheme.</span></div>')
    sub = "Rule-based checks on scheme, path and query. Not used by the model."
    return f'<div class="panel">{heading("URL red flags", sub)}{rows}</div>'


def features_grid(feats: dict) -> str:
    cells = "".join(
        f'<div class="feat"><span class="n mono">{html.escape(k)}</span><span class="lead"></span>'
        f'<span class="val mono">{html.escape(str(v))}</span></div>'
        for k, v in feats.items()
    )
    return f'<div class="feat-grid">{cells}</div>'


# ---------------------------------------------------------------------------------------------- #
# Page
# ---------------------------------------------------------------------------------------------- #
def main() -> None:
    st.markdown(
        '<div class="app-header"><span class="t">🛡️ Phishing URL Checker</span>'
        '<span class="s">Random Forest trained on the PhiUSIIL dataset. It reads the URL text '
        'only and <b>never opens the link</b>.</span></div>',
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
        st.error("Model file not found. Run `python -m src.data_prep` then `python -m src.train`.")
        st.stop()
    pipe, meta = get_model()

    # Session state: the text box is keyed "url". Example buttons (drawn after the box) store a
    # pending URL and rerun, so the box is filled before it is drawn.
    if "url" not in st.session_state:
        st.session_state.url = ""
        qp = st.query_params.get("url")
        if qp:
            st.session_state.url = qp
            st.session_state.run = True
    if "pending_url" in st.session_state:
        st.session_state.url = st.session_state.pop("pending_url")

    # Row 1: URL input (65%) | examples (35%)
    c_url, c_ex = st.columns([0.65, 0.35])
    with c_url:
        with st.container(key="url_card"):
            st.markdown(heading("URL to check", "Paste any http/https link. The scheme is optional."),
                        unsafe_allow_html=True)
            with st.form("check", border=False):
                c_in, c_btn = st.columns([5, 1], vertical_alignment="center")
                url = c_in.text_input("URL to check", key="url", label_visibility="collapsed",
                                      placeholder="e.g. https://example.com/login")
                submitted = c_btn.form_submit_button("Check URL", type="primary",
                                                     use_container_width=True)
            st.markdown('<div class="hint">🔒 Only the text is analysed. The link is never visited, '
                        'resolved or downloaded.</div>', unsafe_allow_html=True)
    with c_ex:
        with st.container(key="ex_card"):
            st.markdown(heading("Try an example", "One click fills the box and runs the check."),
                        unsafe_allow_html=True)
            items = list(EXAMPLES.items())
            for chunk in (items[:3], items[3:]):
                cols = st.columns(3)
                for col, (label, ex_url) in zip(cols, chunk):
                    if col.button(label, use_container_width=True, key=f"ex_{label}"):
                        st.session_state.pending_url = ex_url
                        st.session_state.run = True
                        st.rerun()

    if not (submitted or st.session_state.pop("run", False)):
        st.info("Paste a URL and press **Check URL**, or pick one of the examples.")
        return

    try:
        res = analyze_url(url, pipe, meta)
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        return

    # Row 2: verdict (33%) | explanation (37%) | red flags (30%)
    c1, c2, c3 = st.columns([0.33, 0.37, 0.30])
    c1.markdown(verdict_panel(res), unsafe_allow_html=True)
    c2.markdown(reasons_panel(res), unsafe_allow_html=True)
    c3.markdown(flags_panel(res), unsafe_allow_html=True)

    # Row 3: features
    with st.expander("Extracted hostname features"):
        st.markdown(features_grid(res["features"]), unsafe_allow_html=True)


main()
