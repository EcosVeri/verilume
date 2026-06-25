"""Streamlit CSS for the Verilume desktop app."""

from __future__ import annotations

import streamlit as st


def inject_styles(appearance: str = "dark") -> None:
    theme = _theme_tokens(appearance)
    css = """
<style>
:root {
__THEME_VARIABLES__
}

.stApp {
  background:
    __APP_GRADIENT__,
    var(--veri-bg);
  color: var(--veri-text);
}
"""
    css += _BASE_CSS
    css = css.replace("__THEME_VARIABLES__", theme["variables"])
    css = css.replace("__APP_GRADIENT__", theme["app_gradient"])
    st.markdown(
        css,
        unsafe_allow_html=True,
    )


def _theme_tokens(appearance: str) -> dict[str, str]:
    normalized = "light" if str(appearance or "").strip().lower() == "light" else "dark"
    if normalized == "light":
        tokens = {
            "veri-bg": "#f8f9fb",
            "veri-panel": "#ffffff",
            "veri-panel-2": "#f3f5f8",
            "veri-sidebar": "#f3f5f8",
            "veri-line": "#d9dee6",
            "veri-text": "#1c2430",
            "veri-muted": "#556070",
            "veri-input-bg": "#ffffff",
            "veri-input-text": "#1c2430",
            "veri-header-bg": "linear-gradient(180deg, rgba(255,255,255,.98), rgba(248,249,251,.94))",
            "veri-bottom-bg": (
                "linear-gradient(180deg, rgba(248,249,251,0), rgba(248,249,251,.96) 28%), "
                "rgba(248,249,251,.96)"
            ),
            "veri-card-soft": "rgba(255,255,255,.84)",
            "veri-track": "rgba(217,222,230,.88)",
            "veri-amber": "#c78a1a",
            "veri-teal": "#0066cc",
            "veri-coral": "#d94b3d",
            "veri-green": "#2d9d44",
            "veri-button-text": "#1c2430",
            "veri-primary-text": "#ffffff",
            "veri-local-text": "#176b34",
            "veri-web-text": "#075ba8",
            "veri-ai-text": "#6540a0",
            "veri-amber-text": "#825400",
            "veri-evidence-text": "#9f2f24",
        }
        gradient = "linear-gradient(180deg, rgba(199, 138, 26, 0.08) 0%, rgba(248, 249, 251, 0) 280px)"
    else:
        tokens = {
            "veri-bg": "#0b0d10",
            "veri-panel": "#14171d",
            "veri-panel-2": "#191d24",
            "veri-sidebar": "#101319",
            "veri-line": "#2b303a",
            "veri-text": "#f5f2e8",
            "veri-muted": "#9ca6b5",
            "veri-input-bg": "#12161c",
            "veri-input-text": "#f5f2e8",
            "veri-header-bg": "linear-gradient(180deg, rgba(14, 15, 17, .98), rgba(14, 15, 17, .92))",
            "veri-bottom-bg": (
                "linear-gradient(180deg, rgba(11, 13, 16, 0), rgba(11, 13, 16, .96) 28%), "
                "rgba(11, 13, 16, .96)"
            ),
            "veri-card-soft": "rgba(20, 23, 29, .78)",
            "veri-track": "rgba(43, 48, 58, .78)",
            "veri-amber": "#ffc857",
            "veri-teal": "#36d1c4",
            "veri-coral": "#ff6b5f",
            "veri-green": "#7bd88f",
            "veri-button-text": "#f5f2e8",
            "veri-primary-text": "#101319",
            "veri-local-text": "#b9f3c5",
            "veri-web-text": "#b9d9ff",
            "veri-ai-text": "#dcc8ff",
            "veri-amber-text": "#ffe3a3",
            "veri-evidence-text": "#ffc7c2",
        }
        gradient = "linear-gradient(180deg, rgba(255, 200, 87, 0.08) 0%, rgba(11, 13, 16, 0) 280px)"
    variables = "\n".join(f"  --{name}: {value};" for name, value in tokens.items())
    return {"variables": variables, "app_gradient": gradient}


_BASE_CSS = """

[data-testid="stSidebar"] {
  background: var(--veri-sidebar) !important;
  border-right: 1px solid var(--veri-line) !important;
  min-width: 320px;
  max-width: 320px;
}

[data-testid="stSidebar"] > div:first-child {
  min-width: 320px;
  max-width: 320px;
}

[data-testid="stSidebar"] * {
  color: var(--veri-text);
}

.block-container {
  margin-left: auto !important;
  margin-right: auto !important;
  max-width: 1120px !important;
  padding: 2.2rem 2rem 7.4rem 2rem !important;
  width: 100%;
}

h1, h2, h3 {
  letter-spacing: 0;
}

.veri-header {
  position: sticky;
  top: 2.8rem;
  z-index: 20;
  background: var(--veri-header-bg);
  backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--veri-line);
  padding: .75rem 0 1.1rem 0;
  margin-bottom: 1rem;
}

.veri-theme-toggle-wrap {
  margin-top: .7rem;
  text-align: right;
}

.veri-theme-toggle-wrap + div button,
.veri-theme-toggle-wrap + div [data-testid="baseButton-secondary"] {
  background: var(--veri-panel-2) !important;
  border: 1px solid var(--veri-line) !important;
  border-radius: 999px !important;
  color: var(--veri-text) !important;
  min-height: 2.3rem !important;
  min-width: 2.6rem !important;
}

button[kind="secondary"][data-testid="baseButton-secondary"] {
  color: var(--veri-text) !important;
}

.veri-brand,
.veri-side-brand {
  font-size: .78rem;
  line-height: 1;
  font-weight: 800;
  letter-spacing: .12rem;
  text-transform: uppercase;
  color: var(--veri-amber);
}

.veri-side-brand {
  position: sticky;
  top: .75rem;
  z-index: 10;
  background: var(--veri-sidebar);
  padding: .35rem 0 1rem 0;
  margin-bottom: .35rem;
}

[data-testid="stSidebar"] [data-testid="stExpander"] {
  border: 1px solid var(--veri-line);
  border-radius: 12px;
  background: var(--veri-panel) !important;
  overflow: hidden;
}

[data-testid="stSidebar"] [data-testid="stExpander"] details {
  background: transparent;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary {
  background: var(--veri-panel-2) !important;
  border-bottom: 1px solid var(--veri-line);
  font-weight: 700;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
  color: var(--veri-amber);
}

.veri-active-model {
  border: 1px solid rgba(54, 209, 196, .34);
  border-radius: 8px;
  background:
    linear-gradient(180deg, rgba(54, 209, 196, .12), var(--veri-panel));
  margin: .75rem 0 .25rem 0;
  padding: .68rem .72rem;
}

.veri-active-model-label {
  color: var(--veri-teal);
  font-size: .72rem;
  font-weight: 820;
  letter-spacing: .06rem;
  line-height: 1.2;
  text-transform: uppercase;
}

.veri-active-model-name {
  color: var(--veri-text);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: .82rem;
  font-weight: 740;
  line-height: 1.35;
  margin-top: .38rem;
  overflow-wrap: anywhere;
  white-space: normal;
}

.veri-title {
  margin-top: .24rem;
  font-size: 2rem;
  line-height: 1.05;
  font-weight: 780;
  color: var(--veri-text);
}

.veri-subtitle {
  margin-top: .35rem;
  color: var(--veri-muted);
  font-size: .96rem;
}

.veri-pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: .5rem;
  margin-top: .75rem;
}

.veri-pill {
  display: inline-flex;
  align-items: center;
  gap: .4rem;
  border: 1px solid var(--veri-line);
  background: var(--veri-panel-2);
  border-radius: 999px;
  padding: .34rem .62rem;
  font-size: .78rem;
  color: var(--veri-muted);
}

.veri-dot {
  width: .48rem;
  height: .48rem;
  border-radius: 99px;
  background: var(--veri-teal);
}

.veri-dot.amber { background: var(--veri-amber); }
.veri-dot.coral { background: var(--veri-coral); }
.veri-dot.green { background: var(--veri-green); }

[data-testid="stMetric"] {
  background: var(--veri-panel) !important;
  border: 1px solid var(--veri-line) !important;
  border-radius: 8px;
  padding: .8rem .9rem;
}

[data-testid="stMetricLabel"] {
  color: var(--veri-muted) !important;
}

[data-testid="stMetricValue"] {
  color: var(--veri-text) !important;
  font-size: 1.35rem;
}

[data-testid="stChatMessage"] {
  background: var(--veri-card-soft) !important;
  border: 1px solid var(--veri-line) !important;
  border-radius: 8px;
  overflow: hidden;
  width: 100%;
}

[data-testid="stChatMessage"] p {
  color: var(--veri-text) !important;
}

[data-testid="stChatMessageContent"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] {
  max-width: none !important;
  color: var(--veri-text) !important;
}

[data-testid="stChatInput"] {
  background: var(--veri-input-bg) !important;
  border: 1px solid var(--veri-line) !important;
  border-radius: 999px !important;
  margin-left: auto;
  margin-right: auto;
  max-width: 1120px;
  transition: border-color .16s ease, box-shadow .16s ease, background-color .16s ease;
  width: calc(100vw - 380px);
}

[data-testid="stChatInput"]:hover,
[data-testid="stChatInput"]:focus-within {
  border-color: var(--veri-amber) !important;
}

[data-testid="stChatInput"]:focus-within {
  box-shadow: 0 0 0 3px rgba(255, 200, 87, .14) !important;
}

[data-testid="stBottom"] {
  background: var(--veri-input-bg) !important;
  border-top: 0;
}

[data-testid="stBottom"] > div {
  background: var(--veri-input-bg) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  max-width: 1120px !important;
  padding: .75rem 2rem 1rem 2rem !important;
  width: 100%;
}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] form,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="textarea"] > div {
  background: var(--veri-input-bg) !important;
  border-radius: 999px !important;
}

[data-testid="stChatInput"] textarea {
  background-color: var(--veri-input-bg) !important;
  border: 0 !important;
  border-radius: 999px !important;
  color: var(--veri-input-text) !important;
  min-height: 3rem !important;
  padding-top: .86rem !important;
  padding-bottom: .86rem !important;
  padding-left: 1.15rem !important;
  padding-right: 3.25rem !important;
  -webkit-text-fill-color: var(--veri-input-text) !important;
  box-shadow: none !important;
  outline: none !important;
}

[data-testid="stChatInput"] textarea:hover {
  border-color: transparent !important;
}

[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible {
  border-color: transparent !important;
  box-shadow: none !important;
  outline: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
  color: var(--veri-muted) !important;
  opacity: .82;
}

[data-testid="stChatInput"] button {
  align-items: center !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: 999px !important;
  color: var(--veri-amber) !important;
  display: inline-flex !important;
  height: 2.35rem !important;
  justify-content: center !important;
  margin-right: .32rem !important;
  transition: background-color .16s ease, color .16s ease, transform .16s ease;
  width: 2.35rem !important;
}

[data-testid="stChatInput"] button:hover {
  background: var(--veri-amber) !important;
  color: #ffffff !important;
  transform: translateY(-1px);
}

[data-testid="stChatInput"] button svg {
  color: currentColor !important;
  fill: currentColor !important;
}

.veri-answer-heading {
  font-size: 1.02rem;
  font-weight: 760;
  color: var(--veri-text);
}

.veri-answer-timestamp {
  margin: -.15rem 0 .6rem 0;
  color: var(--veri-muted);
  font-size: .82rem;
}

.veri-answer-origin {
  display: flex;
  flex-wrap: wrap;
  gap: .5rem;
  margin: .2rem 0 .75rem 0;
}

.veri-answer-origin span {
  display: inline-flex;
  align-items: center;
  border: 1px solid rgba(54, 209, 196, .22);
  border-radius: 999px;
  background: var(--veri-panel-2);
  color: var(--veri-muted);
  font-size: .78rem;
  font-weight: 700;
  line-height: 1.2;
  padding: .34rem .58rem;
}

.veri-answer-origin-local span {
  border-color: rgba(123, 216, 143, .36);
  background: rgba(123, 216, 143, .1);
  color: var(--veri-local-text);
}

.veri-answer-origin-web span {
  border-color: rgba(91, 173, 255, .38);
  background: rgba(91, 173, 255, .1);
  color: var(--veri-web-text);
}

.veri-answer-origin-ai span {
  border-color: rgba(184, 132, 255, .38);
  background: rgba(184, 132, 255, .1);
  color: var(--veri-ai-text);
}

.veri-answer-origin-hybrid span {
  border-color: rgba(255, 200, 87, .42);
  background: rgba(255, 200, 87, .12);
  color: var(--veri-amber-text);
}

.veri-answer-origin-evidence span {
  border-color: rgba(255, 107, 95, .36);
  background: rgba(255, 107, 95, .1);
  color: var(--veri-evidence-text);
}

.veri-answer-origin-detail {
  flex-basis: 100%;
  display: flex;
  flex-wrap: wrap;
  gap: .45rem;
}

.veri-answer-origin-detail span:first-child {
  border-color: transparent;
  background: transparent;
  color: var(--veri-muted);
  padding-left: 0;
}

.veri-evidence-badges {
  display: flex;
  flex-wrap: wrap;
  gap: .45rem;
  margin: -.25rem 0 .85rem 0;
}

.veri-evidence-badges span {
  display: inline-flex;
  align-items: center;
  border: 1px solid rgba(255, 200, 87, .24);
  border-radius: 999px;
  background: rgba(255, 200, 87, .08);
  color: var(--veri-amber-text);
  font-size: .76rem;
  font-weight: 720;
  line-height: 1.2;
  padding: .3rem .55rem;
}

.veri-evidence-summary {
  border-top: 1px solid var(--veri-line);
  margin: .95rem 0 .85rem 0;
  padding-top: .72rem;
}

.veri-evidence-summary-title {
  color: var(--veri-muted);
  font-size: .78rem;
  font-weight: 800;
  letter-spacing: .08rem;
  margin-bottom: .48rem;
  text-transform: uppercase;
}

.veri-evidence-summary-badges {
  display: flex;
  flex-wrap: wrap;
  gap: .45rem;
  margin-bottom: .62rem;
}

.veri-evidence-summary-badges span {
  display: inline-flex;
  align-items: center;
  border: 1px solid rgba(255, 200, 87, .28);
  border-radius: 999px;
  background: rgba(255, 200, 87, .08);
  color: var(--veri-amber-text);
  font-size: .76rem;
  font-weight: 720;
  line-height: 1.2;
  padding: .3rem .55rem;
}

.veri-source-strengths {
  display: grid;
  gap: .38rem;
  max-width: 460px;
}

.veri-source-strength-row {
  align-items: center;
  display: grid;
  gap: .55rem;
  grid-template-columns: 3.8rem minmax(7rem, 1fr) 3rem;
}

.veri-source-strength-label,
.veri-source-strength-value {
  color: var(--veri-muted);
  font-size: .78rem;
  font-weight: 720;
}

.veri-source-strength-value {
  text-align: right;
}

.veri-source-strength-track {
  background: var(--veri-track);
  border-radius: 999px;
  height: .5rem;
  overflow: hidden;
}

.veri-source-strength-fill {
  display: block;
  height: 100%;
}

.veri-source-strength-local { background: var(--veri-local-text); }
.veri-source-strength-web { background: var(--veri-web-text); }
.veri-source-strength-ai { background: var(--veri-ai-text); }

.veri-history-label {
  margin: 1rem 0 .65rem 0;
  color: var(--veri-muted);
  font-size: .82rem;
  letter-spacing: .08rem;
  text-transform: uppercase;
}

.veri-inline-source-heading {
  border-top: 1px solid var(--veri-line);
  color: var(--veri-amber);
  font-size: .82rem;
  font-weight: 760;
  margin: .85rem 0 .45rem 0;
  padding-top: .65rem;
}

.veri-inline-source-heading-local {
  color: var(--veri-local-text);
}

.veri-inline-source-heading-web {
  color: var(--veri-web-text);
}

.veri-source-section {
  border-left: 3px solid var(--veri-amber);
  color: var(--veri-text);
  font-size: .9rem;
  font-weight: 780;
  letter-spacing: .01rem;
  margin: .9rem 0 .45rem 0;
  padding: .18rem 0 .18rem .65rem;
}

.veri-source-section-local {
  border-left-color: var(--veri-local-text);
}

.veri-source-section-web {
  border-left-color: var(--veri-web-text);
}

.veri-empty-state {
  border: 1px solid rgba(54, 209, 196, .22);
  border-radius: 8px;
  background: rgba(54, 209, 196, .055);
  margin: .7rem 0 1rem 0;
  padding: .85rem .95rem;
}

.veri-empty-state-title {
  color: var(--veri-text);
  font-weight: 780;
  margin-bottom: .25rem;
}

.veri-empty-state-body {
  color: var(--veri-muted);
  font-size: .9rem;
  line-height: 1.45;
}

.veri-recommendation-card {
  border: 1px solid rgba(255, 200, 87, 0.32);
  border-radius: 12px;
  background:
    linear-gradient(180deg, rgba(255, 200, 87, 0.08), var(--veri-panel));
  padding: 1rem 1rem .9rem 1rem;
  margin-bottom: .65rem;
}

.veri-recommendation-kicker {
  color: var(--veri-amber);
  font-size: .76rem;
  font-weight: 800;
  letter-spacing: .08rem;
  text-transform: uppercase;
  margin-bottom: .35rem;
}

.veri-recommendation-title {
  color: var(--veri-text);
  font-size: 1.14rem;
  font-weight: 760;
  margin-bottom: .45rem;
}

.veri-recommendation-body {
  color: var(--veri-text);
  margin-bottom: .55rem;
}

.veri-recommendation-list {
  margin: 0;
  padding-left: 1.05rem;
  color: var(--veri-muted);
}

.veri-upload-card {
  border: 1px solid rgba(255, 200, 87, 0.2);
  border-radius: 12px;
  background:
    linear-gradient(180deg, rgba(255, 200, 87, 0.08), var(--veri-panel));
  padding: 1rem 1rem .95rem 1rem;
  margin-bottom: .75rem;
}

.veri-upload-title {
  color: var(--veri-text);
  font-size: 1rem;
  font-weight: 760;
  margin-bottom: .55rem;
}

.veri-upload-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: .75rem;
}

.veri-upload-label {
  color: var(--veri-muted);
  font-size: .74rem;
  font-weight: 700;
  letter-spacing: .05rem;
  text-transform: uppercase;
  margin-bottom: .22rem;
}

.veri-upload-value {
  color: var(--veri-text);
  font-size: .88rem;
  line-height: 1.45;
}

.stButton > button,
.stDownloadButton > button {
  border-radius: 8px !important;
  border: 1px solid var(--veri-line) !important;
  background: var(--veri-panel-2) !important;
  color: var(--veri-button-text) !important;
  min-height: 2.35rem;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
  border-color: var(--veri-amber);
  color: var(--veri-amber);
}

.stButton > button[kind="primary"] {
  background: linear-gradient(90deg, #ffc857, #36d1c4);
  color: var(--veri-primary-text) !important;
  border: 0;
  font-weight: 700;
}

.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextArea textarea {
  background-color: var(--veri-input-bg) !important;
  border-color: var(--veri-line) !important;
  color: var(--veri-input-text) !important;
  border-radius: 8px;
  -webkit-text-fill-color: var(--veri-input-text) !important;
  caret-color: var(--veri-amber);
  box-shadow: none !important;
}

.stSelectbox div[data-baseweb="select"] *,
.stMultiSelect div[data-baseweb="select"] * {
  color: var(--veri-input-text) !important;
  -webkit-text-fill-color: var(--veri-input-text) !important;
}

.stSelectbox div[data-baseweb="select"] input,
.stMultiSelect div[data-baseweb="select"] input {
  color: var(--veri-input-text) !important;
  -webkit-text-fill-color: var(--veri-input-text) !important;
}

.stSelectbox div[data-baseweb="select"] svg,
.stMultiSelect div[data-baseweb="select"] svg {
  color: var(--veri-muted) !important;
  fill: var(--veri-muted) !important;
}

.stMultiSelect [data-baseweb="tag"] {
  background-color: rgba(54, 209, 196, .16) !important;
  border: 1px solid rgba(54, 209, 196, .32) !important;
  color: var(--veri-text) !important;
}

[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] ul,
div[data-baseweb="popover"],
div[data-baseweb="menu"] {
  background-color: var(--veri-panel) !important;
  border: 1px solid var(--veri-line) !important;
  color: var(--veri-text) !important;
}

[data-baseweb="popover"] [role="option"],
div[data-baseweb="menu"] li {
  background-color: var(--veri-panel) !important;
  color: var(--veri-text) !important;
  -webkit-text-fill-color: var(--veri-text) !important;
}

[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"],
div[data-baseweb="menu"] li:hover {
  background-color: var(--veri-panel-2) !important;
  color: var(--veri-amber) !important;
  -webkit-text-fill-color: var(--veri-amber) !important;
}

.stTextInput input:-webkit-autofill,
.stTextInput input:-webkit-autofill:hover,
.stTextInput input:-webkit-autofill:focus {
  -webkit-box-shadow: 0 0 0 1000px var(--veri-input-bg) inset !important;
  -webkit-text-fill-color: var(--veri-input-text) !important;
  transition: background-color 9999s ease-in-out 0s;
}

[data-baseweb="input"] {
  background-color: var(--veri-input-bg) !important;
  border-color: var(--veri-line) !important;
}

[data-baseweb="input"] button {
  background-color: var(--veri-panel-2) !important;
  color: var(--veri-input-text) !important;
}

[data-testid="stFileUploader"] {
  background: var(--veri-panel);
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  padding: .5rem;
}

[data-testid="stFileUploaderDropzone"] {
  background: var(--veri-input-bg) !important;
  border: 1px dashed var(--veri-line) !important;
  border-radius: 12px !important;
  color: var(--veri-text) !important;
  padding: 1.1rem .9rem !important;
}

[data-testid="stFileUploaderDropzone"] * {
  color: var(--veri-text) !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
  display: none !important;
}

[data-testid="stFileUploaderDropzoneInstructions"]::before {
  content: "Build your knowledge base";
  display: block;
  color: var(--veri-text);
  font-size: 1rem;
  font-weight: 760;
  margin-bottom: .55rem;
}

[data-testid="stFileUploaderDropzoneInstructions"]::after {
  content: "Supported:\\A PDF • Scanned PDF • DOCX • PPTX • Images • TXT • MD • CSV\\A\\A Maximum size:\\A 200 MB per file";
  white-space: pre-line;
  display: block;
  color: var(--veri-muted);
  font-size: .88rem;
  line-height: 1.45;
}

[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
  color: var(--veri-muted) !important;
}

[data-testid="stFileUploaderDropzone"] button {
  background: var(--veri-panel-2) !important;
  border: 1px solid var(--veri-line) !important;
  color: var(--veri-button-text) !important;
  border-radius: 8px !important;
  font-weight: 700 !important;
}

[data-testid="stFileUploaderFile"] {
  background: var(--veri-panel-2) !important;
  border-radius: 8px !important;
  color: var(--veri-text) !important;
}

[data-testid="stFileUploaderFile"] * {
  color: var(--veri-text) !important;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
  background: var(--veri-panel) !important;
  border: 1px solid var(--veri-line) !important;
  color: var(--veri-text) !important;
}

[data-testid="stDataFrame"] *,
[data-testid="stTable"] * {
  color: var(--veri-text) !important;
}

.veri-source-block {
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  background: var(--veri-panel);
  padding: .75rem;
  margin-top: .5rem;
}

.veri-source-title {
  color: var(--veri-amber);
  font-weight: 720;
  margin-bottom: .35rem;
}

a {
  color: var(--veri-teal) !important;
}

@media (max-width: 900px) {
  [data-testid="stSidebar"] {
    min-width: 280px;
    max-width: 280px;
  }

  [data-testid="stSidebar"] > div:first-child {
    min-width: 280px;
    max-width: 280px;
  }

  .block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }

  [data-testid="stBottom"] > div {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }

  [data-testid="stChatInput"] {
    width: calc(100vw - 2rem);
  }
}
</style>
"""
