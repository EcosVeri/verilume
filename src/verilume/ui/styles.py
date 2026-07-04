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
            "veri-panel-strong": "#eef2f6",
            "veri-sidebar": "#f3f5f8",
            "veri-line": "#d9dee6",
            "veri-text": "#1c2430",
            "veri-muted": "#556070",
            "veri-input-bg": "#ffffff",
            "veri-input-text": "#1c2430",
            "veri-shadow": "0 18px 48px rgba(25, 35, 50, .10)",
            "veri-focus": "0 0 0 3px rgba(0, 102, 204, .18)",
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
            "veri-sidebar-width": "320px",
            "veri-command-width": "min(860px, calc(100vw - var(--veri-sidebar-width) - 96px))",
            "veri-command-bottom": "28px",
            "veri-command-height": "46px",
            "veri-command-shadow": "0 8px 26px rgba(31, 41, 55, .14)",
            "veri-command-focus-shadow": (
                "0 0 0 3px rgba(199, 138, 26, .18), "
                "0 8px 26px rgba(31, 41, 55, .16)"
            ),
            "veri-command-border": "rgba(0, 102, 204, .38)",
            "veri-tooltip-bg": "#ffffff",
            "veri-tooltip-text": "#1c2430",
            "veri-tooltip-border": "#cfd6df",
        }
        gradient = "linear-gradient(180deg, rgba(199, 138, 26, 0.08) 0%, rgba(248, 249, 251, 0) 280px)"
    else:
        tokens = {
            "veri-bg": "#0b0d10",
            "veri-panel": "#20252f",
            "veri-panel-2": "#262c38",
            "veri-panel-strong": "#303947",
            "veri-sidebar": "#101319",
            "veri-line": "#353c48",
            "veri-text": "#f5f2e8",
            "veri-muted": "#9ca6b5",
            "veri-input-bg": "#12161c",
            "veri-input-text": "#f5f2e8",
            "veri-shadow": "0 22px 58px rgba(0, 0, 0, .38)",
            "veri-focus": "0 0 0 3px rgba(54, 209, 196, .18)",
            "veri-header-bg": "linear-gradient(180deg, rgba(14, 15, 17, .98), rgba(14, 15, 17, .92))",
            "veri-bottom-bg": (
                "linear-gradient(180deg, rgba(11, 13, 16, 0), rgba(11, 13, 16, .96) 28%), "
                "rgba(11, 13, 16, .96)"
            ),
            "veri-card-soft": "rgba(25, 29, 37, .82)",
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
            "veri-sidebar-width": "320px",
            "veri-command-width": "min(860px, calc(100vw - var(--veri-sidebar-width) - 96px))",
            "veri-command-bottom": "28px",
            "veri-command-height": "46px",
            "veri-command-shadow": "0 8px 30px rgba(0, 0, 0, .35)",
            "veri-command-focus-shadow": (
                "0 0 0 3px rgba(255, 200, 87, .16), "
                "0 8px 30px rgba(0, 0, 0, .28)"
            ),
            "veri-command-border": "rgba(54, 209, 196, .42)",
            "veri-tooltip-bg": "#14171d",
            "veri-tooltip-text": "#f5f2e8",
            "veri-tooltip-border": "#ffc857",
        }
        gradient = "linear-gradient(180deg, rgba(255, 200, 87, 0.08) 0%, rgba(11, 13, 16, 0) 280px)"
    variables = "\n".join(f"  --{name}: {value};" for name, value in tokens.items())
    return {"variables": variables, "app_gradient": gradient}


_BASE_CSS = """

@keyframes veri-fade-up {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes veri-soft-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(255, 200, 87, .18); }
  50% { box-shadow: 0 0 0 4px rgba(255, 200, 87, .08); }
}

@keyframes veri-progress {
  from { transform: translateX(-55%); }
  to { transform: translateX(125%); }
}

html {
  scroll-behavior: smooth;
}

.stApp * {
  box-sizing: border-box;
}

[data-testid="stHeader"] {
  background: transparent !important;
}

[data-testid="stSidebar"] {
  background: var(--veri-sidebar) !important;
  border-right: 1px solid var(--veri-line) !important;
  min-width: 320px;
  max-width: 320px;
  box-shadow: 8px 0 36px rgba(0, 0, 0, .12);
}

[data-testid="stSidebar"] > div:first-child {
  min-width: 320px;
  max-width: 320px;
  padding-top: 1.25rem;
  padding-bottom: 1.25rem;
}

[data-testid="stSidebar"] * {
  color: var(--veri-text);
}

/* ── Tooltips (consolidated) ─────────────────────────────────────────────────
   Use :where() to avoid specificity wars; a single !important per property
   instead of repeating across 6 selectors. Survives Streamlit minor updates
   because it targets semantic roles, not internal data-testid values.        */

:where([role="tooltip"], [data-baseweb="tooltip"]) {
  background: var(--veri-tooltip-bg, #14171d) !important;
  border: 1px solid var(--veri-tooltip-border, #ffc857) !important;
  border-radius: 8px !important;
  box-shadow: 0 16px 38px rgba(0, 0, 0, .28) !important;
  color: var(--veri-tooltip-text, #f5f2e8) !important;
  height: auto !important;
  line-height: 1.35 !important;
  max-width: min(28rem, calc(100vw - 2rem)) !important;
  min-height: 0 !important;
  overflow: visible !important;
  -webkit-text-fill-color: var(--veri-tooltip-text, #f5f2e8) !important;
}

:where([role="tooltip"], [data-baseweb="tooltip"]):empty,
:where([data-testid="stTooltip"], [data-testid="stTooltipContent"], [data-testid="stMarkdownTooltip"]):empty {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  display: none !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}

:where([data-testid="stTooltip"], [data-testid="stTooltipContent"], [data-testid="stMarkdownTooltip"]) {
  background: transparent !important;
  border: 0 !important;
  height: auto !important;
  line-height: 1.35 !important;
  min-height: 0 !important;
  overflow: visible !important;
  padding: .48rem .68rem !important;
}

:where([role="tooltip"], [data-baseweb="tooltip"]) :where(div, p, span, [data-testid="stMarkdownContainer"]) {
  background: transparent !important;
  color: var(--veri-tooltip-text, #f5f2e8) !important;
  line-height: 1.35 !important;
  margin: 0 !important;
  -webkit-text-fill-color: var(--veri-tooltip-text, #f5f2e8) !important;
}

:where([role="tooltip"], [data-baseweb="tooltip"]) [data-popper-arrow] {
  color: var(--veri-tooltip-bg, #14171d) !important;
}

.block-container {
  max-width: 1280px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding-top: 2rem !important;
  padding-right: 2rem !important;
  padding-bottom: 180px !important;
  padding-left: 2rem !important;
  width: 100%;
}

h1, h2, h3 {
  letter-spacing: 0;
}

.veri-header {
  position: sticky;
  top: .9rem;
  z-index: 20;
  background: var(--veri-header-bg);
  backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--veri-line);
  padding: .85rem 0 1.05rem 0;
  margin-bottom: 1rem;
}

.veri-theme-toggle-wrap {
  margin-top: .7rem;
  text-align: right;
}

.veri-dashboard-toggle-wrap {
  margin: .15rem 0 .25rem 0;
  text-align: right;
}

.veri-dark-button-anchor {
  height: 0;
  margin: 0;
  padding: 0;
}

.veri-prompt-button-wrap {
  margin-top: .44rem;
}

.veri-theme-toggle-wrap + div button,
.veri-theme-toggle-wrap + div [data-testid="baseButton-secondary"],
.veri-dashboard-toggle-wrap + div button,
.veri-dashboard-toggle-wrap + div [data-testid="baseButton-secondary"],
div:has(.veri-dark-button-anchor) + div button,
div:has(.veri-dark-button-anchor) + div [data-testid="baseButton-secondary"],
div:has(.veri-dark-button-anchor) + div [data-testid="stButton"] button {
  background: var(--veri-panel-2) !important;
  background-color: var(--veri-panel-2) !important;
  border: 1px solid var(--veri-line) !important;
  border-radius: 999px !important;
  color: var(--veri-text) !important;
  min-height: 2.3rem !important;
  min-width: 2.6rem !important;
  -webkit-text-fill-color: var(--veri-text) !important;
}

.veri-dashboard-toggle-wrap + div button,
.veri-dashboard-toggle-wrap + div [data-testid="baseButton-secondary"],
div:has(.veri-dark-button-anchor) + div button,
div:has(.veri-dark-button-anchor) + div [data-testid="baseButton-secondary"],
div:has(.veri-dark-button-anchor) + div [data-testid="stButton"] button {
  box-shadow: 0 10px 24px rgba(0, 0, 0, .22);
  font-size: .84rem !important;
  font-weight: 760 !important;
}

.veri-dashboard-toggle-wrap + div button *,
.veri-dashboard-toggle-wrap + div [data-testid="baseButton-secondary"] *,
div:has(.veri-dark-button-anchor) + div button *,
div:has(.veri-dark-button-anchor) + div [data-testid="baseButton-secondary"] *,
div:has(.veri-dark-button-anchor) + div [data-testid="stButton"] button * {
  color: inherit !important;
  -webkit-text-fill-color: currentColor !important;
}

.veri-dashboard-toggle-wrap + div button:hover,
.veri-dashboard-toggle-wrap + div [data-testid="baseButton-secondary"]:hover,
div:has(.veri-dark-button-anchor) + div button:hover,
div:has(.veri-dark-button-anchor) + div [data-testid="baseButton-secondary"]:hover,
div:has(.veri-dark-button-anchor) + div [data-testid="stButton"] button:hover {
  border-color: var(--veri-amber) !important;
  color: var(--veri-amber) !important;
  -webkit-text-fill-color: var(--veri-amber) !important;
}

[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button,
button[kind="secondary"][data-testid="baseButton-secondary"] {
  background: var(--veri-panel-2) !important;
  background-color: var(--veri-panel-2) !important;
  border: 1px solid var(--veri-line) !important;
  color: var(--veri-text) !important;
  -webkit-text-fill-color: var(--veri-text) !important;
}

button[kind="secondary"][data-testid="baseButton-secondary"] *,
button[kind="secondary"][data-testid="baseButton-secondary"] [data-testid="stMarkdownContainer"] p,
.stButton > button *,
.stButton > button [data-testid="stMarkdownContainer"] p,
.stDownloadButton > button *,
.stDownloadButton > button [data-testid="stMarkdownContainer"] p,
[data-testid="stButton"] button *,
[data-testid="stButton"] button [data-testid="stMarkdownContainer"] p,
[data-testid="stDownloadButton"] button *,
[data-testid="stDownloadButton"] button [data-testid="stMarkdownContainer"] p {
  color: inherit !important;
  -webkit-text-fill-color: currentColor !important;
}

[data-testid="stButton"] button:hover,
[data-testid="stDownloadButton"] button:hover,
button[kind="secondary"][data-testid="baseButton-secondary"]:hover {
  border-color: var(--veri-amber) !important;
  color: var(--veri-amber) !important;
  -webkit-text-fill-color: var(--veri-amber) !important;
}

.veri-brand,
.veri-sidebar-brand {
  font-size: .78rem;
  line-height: 1;
  font-weight: 800;
  letter-spacing: .12rem;
  text-transform: uppercase;
  color: var(--veri-amber);
}

.veri-sidebar-brand {
  position: sticky;
  top: .75rem;
  z-index: 10;
  background: var(--veri-sidebar);
  padding: .35rem 0 .75rem 0;
  margin-bottom: .25rem;
}

.veri-sidebar-panel {
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  background:
    linear-gradient(180deg, rgba(54, 209, 196, .08), rgba(255, 200, 87, .05)),
    var(--veri-panel);
  box-shadow: var(--veri-shadow);
  animation: veri-fade-up .28s ease both;
  margin: .2rem 0 .9rem 0;
  padding: .86rem .9rem .9rem .9rem;
}

.veri-sidebar-brandline {
  color: var(--veri-amber);
  font-size: .72rem;
  font-weight: 840;
  letter-spacing: .12rem;
  line-height: 1.2;
  margin-bottom: .3rem;
  text-transform: uppercase;
}

.veri-sidebar-title {
  color: var(--veri-text);
  font-size: 1.12rem;
  font-weight: 800;
  line-height: 1.2;
  margin-bottom: .22rem;
}

.veri-sidebar-subtitle,
.veri-sidebar-version {
  color: var(--veri-muted);
  font-size: .74rem;
  font-weight: 650;
  line-height: 1.25;
}

.veri-sidebar-version {
  margin: .16rem 0 .7rem 0;
}

.veri-sidebar-group-title {
  color: var(--veri-amber);
  font-size: .68rem;
  font-weight: 820;
  letter-spacing: .08rem;
  line-height: 1.1;
  margin: .82rem 0 .34rem 0;
  text-transform: uppercase;
}

.veri-sidebar-group-title:first-child {
  margin-top: .1rem;
}

.veri-field-help {
  align-items: center;
  display: inline-flex;
  gap: .35rem;
  line-height: 1.2;
  margin: .1rem 0 .3rem 0;
  position: relative;
  z-index: 30;
}

.veri-field-help-label {
  color: var(--veri-text);
  font-size: .86rem;
  font-weight: 760;
}

.veri-field-help-dot {
  align-items: center;
  border: 1px solid var(--veri-line);
  border-radius: 999px;
  color: var(--veri-muted);
  cursor: help;
  display: inline-flex;
  flex: 0 0 auto;
  font-size: .7rem;
  font-weight: 820;
  height: 1rem;
  justify-content: center;
  line-height: 1;
  outline: none;
  width: 1rem;
  -webkit-text-fill-color: var(--veri-muted);
}

.veri-field-help-dot:hover,
.veri-field-help-dot:focus {
  border-color: var(--veri-tooltip-border, #ffc857);
  color: var(--veri-tooltip-border, #ffc857);
  -webkit-text-fill-color: var(--veri-tooltip-border, #ffc857);
}

.veri-field-help-bubble {
  background: var(--veri-tooltip-bg, #14171d);
  border: 1px solid var(--veri-tooltip-border, #ffc857);
  border-radius: 8px;
  box-shadow: 0 16px 38px rgba(0, 0, 0, .28);
  color: var(--veri-tooltip-text, #f5f2e8);
  display: none;
  font-size: .78rem;
  font-weight: 650;
  bottom: calc(100% + .38rem);
  left: 0;
  line-height: 1.35;
  max-width: min(22rem, calc(100vw - 2rem));
  min-width: min(18rem, calc(100vw - 2rem));
  padding: .58rem .68rem;
  position: absolute;
  top: auto;
  white-space: normal;
  z-index: 9999;
  -webkit-text-fill-color: var(--veri-tooltip-text, #f5f2e8);
}

.veri-field-help-dot:hover + .veri-field-help-bubble,
.veri-field-help-dot:focus + .veri-field-help-bubble,
.veri-field-help:focus-within .veri-field-help-bubble {
  display: block;
}

.veri-benchmark-compare {
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  background: var(--veri-panel-2);
  margin: .42rem 0 .25rem 0;
  padding: .56rem .62rem;
}

.veri-benchmark-compare strong {
  color: var(--veri-muted);
  display: block;
  font-size: .68rem;
  font-weight: 820;
  letter-spacing: .06rem;
  line-height: 1.1;
  margin-bottom: .42rem;
  text-transform: uppercase;
}

.veri-benchmark-compare div {
  display: flex;
  flex-wrap: wrap;
  gap: .34rem;
}

.veri-benchmark-compare span {
  border: 1px solid var(--veri-line);
  border-radius: 999px;
  color: var(--veri-text);
  display: inline-flex;
  font-size: .72rem;
  font-weight: 760;
  line-height: 1;
  padding: .28rem .48rem;
}

.veri-sidebar-status {
  display: grid;
  gap: .44rem;
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.veri-sidebar-stat {
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  background: rgba(255, 255, 255, .03);
  min-width: 0;
  padding: .48rem .44rem;
}

.veri-sidebar-stat strong,
.veri-sidebar-stat span,
.veri-sidebar-stat em {
  display: block;
  font-style: normal;
  text-align: center;
}

.veri-sidebar-stat span {
  font-size: .9rem;
  line-height: 1;
  margin-bottom: .24rem;
}

.veri-sidebar-stat strong {
  color: var(--veri-text);
  font-size: .96rem;
  font-weight: 800;
  line-height: 1.05;
  overflow-wrap: anywhere;
}

.veri-sidebar-stat em {
  color: var(--veri-muted);
  font-size: .68rem;
  font-weight: 740;
  letter-spacing: .04rem;
  line-height: 1.1;
  margin-top: .25rem;
  text-transform: uppercase;
}

[data-testid="stSidebar"] [data-testid="stExpander"] {
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  background: var(--veri-panel) !important;
  box-shadow: 0 10px 28px rgba(0, 0, 0, .08);
  overflow: visible;
  transition: border-color .16s ease, box-shadow .16s ease, transform .16s ease;
}

[data-testid="stSidebar"] [data-testid="stExpander"]:hover {
  border-color: rgba(255, 200, 87, .42);
  box-shadow: 0 14px 34px rgba(0, 0, 0, .12);
  transform: translateY(-1px);
}

[data-testid="stSidebar"] [data-testid="stExpander"] details[open] > div {
  animation: veri-fade-up .22s ease both;
  overflow: visible;
}

[data-testid="stSidebar"] [data-testid="stExpander"] details {
  background: transparent;
  overflow: visible;
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
  font-size: 2.08rem;
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
.veri-dot.muted { background: var(--veri-muted); }

[data-testid="stMetric"] {
  background: var(--veri-panel) !important;
  border: 1px solid var(--veri-line) !important;
  border-radius: 8px;
  padding: .8rem .9rem;
  box-shadow: 0 10px 28px rgba(0, 0, 0, .08);
}

[data-testid="stMetricLabel"] {
  color: var(--veri-muted) !important;
}

[data-testid="stMetricValue"] {
  color: var(--veri-text) !important;
  font-size: 1.35rem;
}

.veri-metric-grid {
  display: grid;
  gap: .75rem;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  margin: .25rem 0 .85rem 0;
}

/* Text-valued metric cards (model names) get a smaller, wrappable value. */
.veri-metric-value-text {
  font-size: .92rem !important;
  line-height: 1.25 !important;
  overflow-wrap: anywhere;
}

.veri-metric-card {
  align-items: center;
  animation: veri-fade-up .28s ease both;
  background: var(--veri-panel);
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  box-shadow: 0 10px 28px rgba(0, 0, 0, .08);
  display: grid;
  grid-template-columns: 3rem 1fr;
  min-height: 5.1rem;
  padding: .8rem .92rem;
  transition: border-color .16s ease, box-shadow .16s ease, transform .16s ease;
}

.veri-metric-card:hover,
.veri-workspace-card:hover,
.veri-source-card:hover,
[data-testid="stChatMessage"]:hover {
  border-color: rgba(54, 209, 196, .42) !important;
  box-shadow: var(--veri-shadow);
  transform: translateY(-2px);
}

.veri-metric-icon {
  align-items: center;
  background: var(--veri-panel-strong);
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  display: flex;
  font-size: 1.24rem;
  height: 2.5rem;
  justify-content: center;
  width: 2.5rem;
}

.veri-metric-value {
  color: var(--veri-text);
  font-size: 1.45rem;
  font-weight: 820;
  line-height: 1;
}

.veri-metric-label {
  border-top: 1px solid var(--veri-line);
  color: var(--veri-muted);
  font-size: .76rem;
  font-weight: 780;
  grid-column: 1 / -1;
  letter-spacing: .06rem;
  margin-top: .72rem;
  padding-top: .56rem;
  text-transform: uppercase;
}

.veri-workspace-grid {
  display: grid;
  gap: .75rem;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin: 0 0 1rem 0;
}

.veri-workspace-card {
  animation: veri-fade-up .34s ease both;
  background: var(--veri-panel);
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  min-height: 8.2rem;
  padding: .84rem .88rem;
  transition: border-color .16s ease, box-shadow .16s ease, transform .16s ease;
}

.veri-workspace-kicker {
  color: var(--veri-amber);
  font-size: .72rem;
  font-weight: 820;
  letter-spacing: .08rem;
  margin-bottom: .58rem;
  text-transform: uppercase;
}

.veri-mini-row {
  align-items: center;
  display: grid;
  gap: .5rem;
  grid-template-columns: 1.3rem minmax(0, 1fr);
  margin-bottom: .46rem;
}

.veri-mini-row strong,
.veri-mini-muted {
  color: var(--veri-muted);
  font-size: .8rem;
  font-weight: 650;
  line-height: 1.25;
  overflow-wrap: anywhere;
}

.veri-prompt-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: .42rem;
}

.veri-prompt-chip-row span {
  border: 1px solid var(--veri-line);
  border-radius: 999px;
  color: var(--veri-muted);
  font-size: .74rem;
  font-weight: 720;
  line-height: 1.15;
  padding: .3rem .46rem;
}

[data-testid="stChatMessage"] {
  background: var(--veri-card-soft) !important;
  border: 1px solid var(--veri-line) !important;
  border-radius: 8px;
  box-shadow: 0 12px 34px rgba(0, 0, 0, .10);
  overflow: hidden;
  transition: border-color .16s ease, box-shadow .16s ease, transform .16s ease;
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

.veri-command-dock {
  bottom: var(--veri-command-bottom);
  left: calc(var(--veri-sidebar-width) + ((100vw - var(--veri-sidebar-width)) / 2));
  pointer-events: auto;
  position: fixed;
  transform: translateX(-50%);
  width: var(--veri-command-width);
  z-index: 9999;
}

[data-testid="stChatInput"] {
  background: var(--veri-input-bg) !important;
  border: 1px solid var(--veri-command-border) !important;
  border-radius: 999px !important;
  animation: veri-soft-pulse 3.6s ease-in-out infinite;
  bottom: var(--veri-command-bottom) !important;
  box-shadow: var(--veri-command-shadow);
  left: calc(var(--veri-sidebar-width) + ((100vw - var(--veri-sidebar-width)) / 2)) !important;
  margin: 0 !important;
  max-width: var(--veri-command-width) !important;
  position: fixed !important;
  pointer-events: auto !important;
  transform: translateX(-50%) !important;
  transition: border-color .16s ease, box-shadow .16s ease, background-color .16s ease;
  width: var(--veri-command-width) !important;
  z-index: 9999 !important;
}

[data-testid="stChatInput"]:hover,
[data-testid="stChatInput"]:focus-within {
  border-color: var(--veri-amber) !important;
}

[data-testid="stChatInput"]:focus-within {
  box-shadow: var(--veri-command-focus-shadow) !important;
}

[data-testid="stBottom"] {
  background: var(--veri-bottom-bg) !important;
  border-top: 0;
  min-height: calc(var(--veri-command-height) + var(--veri-command-bottom) + 24px);
  pointer-events: none;
}

[data-testid="stBottom"] > div {
  background: transparent !important;
  margin-left: auto !important;
  margin-right: auto !important;
  max-width: 1280px !important;
  padding: 0 !important;
  pointer-events: none;
  width: 100%;
}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] form,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="textarea"] > div {
  background: var(--veri-input-bg) !important;
  border-radius: 999px !important;
  pointer-events: auto;
}

[data-testid="stChatInput"] textarea {
  background-color: var(--veri-input-bg) !important;
  border: 0 !important;
  border-radius: 999px !important;
  color: var(--veri-input-text) !important;
  line-height: 1.3 !important;
  max-height: 140px !important;
  min-height: var(--veri-command-height) !important;
  padding: .7rem 3.2rem .7rem 1rem !important;
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
  height: 2rem !important;
  justify-content: center !important;
  margin-bottom: 3px !important;
  margin-right: 8px !important;
  pointer-events: auto !important;
  transition: background-color .16s ease, color .16s ease, transform .16s ease;
  width: 2rem !important;
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

.veri-answer-stats {
  display: flex;
  flex-wrap: wrap;
  gap: .4rem;
  margin: .1rem 0 .1rem 0;
}

.veri-answer-stats span {
  background: var(--veri-panel-2);
  border: 1px solid var(--veri-line);
  border-radius: 999px;
  color: var(--veri-muted);
  font-size: .8rem;
  font-weight: 700;
  padding: .12rem .62rem;
}

.veri-answer-stats span.veri-answer-stat-conf {
  color: var(--veri-text);
}

.veri-answer-divider {
  border: none;
  border-top: 1px solid var(--veri-line);
  margin: .55rem 0 .7rem 0;
}

.veri-benchmark-teaser {
  color: var(--veri-muted);
  font-size: .8rem;
  font-weight: 720;
  margin: .35rem 0 .3rem 0;
}

/* The answer body dominates the card — larger, airier than diagnostics. */
[class*="st-key-veri-answer-body"] [data-testid="stMarkdownContainer"] p,
[class*="st-key-veri-answer-body"] [data-testid="stMarkdownContainer"] li {
  font-size: 1.16rem;
  line-height: 1.72;
}

[class*="st-key-veri-answer-body"] [data-testid="stMarkdownContainer"] p {
  margin-bottom: .72rem;
}

[class*="st-key-veri-answer-body"] [data-testid="stMarkdownContainer"] h1,
[class*="st-key-veri-answer-body"] [data-testid="stMarkdownContainer"] h2,
[class*="st-key-veri-answer-body"] [data-testid="stMarkdownContainer"] h3 {
  font-size: 1.32rem;
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
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  background: linear-gradient(180deg, rgba(255, 200, 87, .06), transparent), var(--veri-panel);
  margin: .95rem 0 .85rem 0;
  padding: .82rem .9rem .9rem .9rem;
  animation: veri-fade-up .24s ease both;
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

.veri-evidence-summary-rows {
  display: grid;
  gap: .45rem;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin: .55rem 0 .72rem 0;
}

.veri-evidence-summary-row {
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  background: rgba(255, 255, 255, .025);
  min-width: 0;
  padding: .48rem .52rem;
}

.veri-evidence-summary-row span,
.veri-evidence-summary-row strong {
  display: block;
  line-height: 1.15;
  overflow-wrap: anywhere;
}

.veri-evidence-summary-row span {
  color: var(--veri-muted);
  font-size: .68rem;
  font-weight: 780;
  letter-spacing: .05rem;
  margin-bottom: .28rem;
  text-transform: uppercase;
}

.veri-evidence-summary-row strong {
  color: var(--veri-text);
  font-size: .8rem;
  font-weight: 780;
}

.veri-evidence-analysis-grid {
  grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
}

.veri-knowledge-used {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: .55rem;
  margin: .1rem 0 .3rem 0;
}

.veri-knowledge-label {
  color: var(--veri-muted);
  font-size: .68rem;
  font-weight: 800;
  letter-spacing: .05rem;
  text-transform: uppercase;
}

.veri-knowledge-chips {
  display: flex;
  flex-wrap: wrap;
  gap: .38rem;
}

.veri-knowledge-chip {
  border: 1px solid var(--veri-line);
  border-radius: 999px;
  font-size: .76rem;
  font-weight: 760;
  padding: .12rem .55rem;
}

.veri-knowledge-chip.on {
  background: rgba(123, 216, 143, .14);
  border-color: rgba(123, 216, 143, .4);
  color: var(--veri-local-text);
}

.veri-knowledge-chip.off {
  color: var(--veri-muted);
  opacity: .7;
}

.veri-evidence-reasons {
  border: 1px solid rgba(54, 209, 196, .22);
  border-radius: 8px;
  background: rgba(54, 209, 196, .045);
  margin: .52rem 0 .72rem 0;
  padding: .62rem .7rem;
}

.veri-evidence-reasons strong {
  color: var(--veri-text);
  display: block;
  font-size: .76rem;
  font-weight: 820;
  letter-spacing: .05rem;
  margin-bottom: .4rem;
  text-transform: uppercase;
}

.veri-evidence-reasons ul {
  display: flex;
  flex-wrap: wrap;
  gap: .36rem;
  list-style: none;
  margin: 0;
  padding: 0;
}

.veri-evidence-reasons li {
  border: 1px solid var(--veri-line);
  border-radius: 999px;
  color: var(--veri-muted);
  font-size: .75rem;
  font-weight: 720;
  line-height: 1.2;
  padding: .28rem .48rem;
}

.veri-source-strength-row {
  border: 1px solid var(--veri-line);
  border-radius: 9px;
  background: rgba(255, 255, 255, .02);
  display: grid;
  gap: .4rem;
  padding: .55rem .62rem;
}

.veri-source-strength-head {
  align-items: center;
  display: flex;
  justify-content: space-between;
}

.veri-source-strength-label {
  color: var(--veri-muted);
  font-size: .68rem;
  font-weight: 800;
  letter-spacing: .06rem;
}

.veri-source-strength-grade {
  border-radius: 999px;
  font-size: .68rem;
  font-weight: 800;
  letter-spacing: .03rem;
  padding: .1rem .46rem;
  text-transform: uppercase;
}

.veri-source-strength-grade-local {
  background: rgba(123, 216, 143, .16);
  color: var(--veri-local-text);
}
.veri-source-strength-grade-web {
  background: rgba(185, 217, 255, .16);
  color: var(--veri-web-text);
}
.veri-source-strength-grade-ai {
  background: rgba(220, 200, 255, .16);
  color: var(--veri-ai-text);
}

.veri-source-strength-meter {
  align-items: center;
  display: grid;
  gap: .6rem;
  grid-template-columns: minmax(7rem, 1fr) 3.2rem;
}

.veri-source-strength-value {
  color: var(--veri-text);
  font-size: 1.18rem;
  font-weight: 820;
  text-align: right;
}

.veri-source-strength-track {
  background: var(--veri-track);
  border-radius: 999px;
  height: .6rem;
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
  align-items: center;
  border: 1px solid var(--veri-line);
  border-left: 3px solid var(--veri-amber);
  border-radius: 8px;
  background: var(--veri-panel);
  color: var(--veri-text);
  display: flex;
  font-size: .9rem;
  font-weight: 780;
  letter-spacing: .01rem;
  margin: .9rem 0 .45rem 0;
  min-height: 2.35rem;
  padding: .28rem .72rem;
}

.veri-source-section-local {
  border-left-color: var(--veri-local-text);
}

.veri-source-section-web {
  border-left-color: var(--veri-web-text);
}

.veri-source-group-heading {
  color: var(--veri-text);
  font-size: .86rem;
  font-weight: 800;
  line-height: 1.2;
  margin: .72rem 0 .1rem 0;
}

.veri-source-stars {
  margin-left: .5rem;
  color: var(--veri-gold, #ffc857);
  font-size: .78rem;
  font-weight: 600;
  letter-spacing: .5px;
  vertical-align: baseline;
}

.veri-empty-state {
  border: 1px solid rgba(54, 209, 196, .22);
  border-radius: 8px;
  background:
    linear-gradient(180deg, rgba(54, 209, 196, .09), rgba(255, 200, 87, .045)),
    var(--veri-panel);
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

/* Big first-run upload CTA under the empty state. */
.veri-upload-cta-wrap {
  height: 0;
  margin: 0;
  padding: 0;
}

div:has(.veri-upload-cta-wrap) + div [data-testid="stButton"] button {
  border-radius: 10px !important;
  font-size: 1.02rem !important;
  font-weight: 780 !important;
  min-height: 3.1rem !important;
}

.veri-recommendation-card {
  border: 1px solid rgba(255, 200, 87, 0.32);
  border-radius: 8px;
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
  border-radius: 8px;
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
  background-color: var(--veri-panel-2) !important;
  color: var(--veri-button-text) !important;
  font-weight: 740 !important;
  min-height: 2.35rem;
  transition: border-color .16s ease, box-shadow .16s ease, color .16s ease, transform .16s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
  border-color: var(--veri-amber);
  color: var(--veri-amber);
  transform: translateY(-1px);
}

.stButton > button:focus,
.stDownloadButton > button:focus {
  box-shadow: var(--veri-focus) !important;
}

.stButton > button[kind="primary"] {
  background: linear-gradient(90deg, var(--veri-amber), var(--veri-teal)) !important;
  color: var(--veri-primary-text) !important;
  border: 0 !important;
  font-weight: 700;
}

.stButton > button[kind="primary"] *,
.stButton > button[kind="primary"] [data-testid="stMarkdownContainer"] p {
  color: var(--veri-primary-text) !important;
  -webkit-text-fill-color: var(--veri-primary-text) !important;
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

.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus,
.stSelectbox div[data-baseweb="select"] > div:focus-within,
.stMultiSelect div[data-baseweb="select"] > div:focus-within {
  border-color: var(--veri-teal) !important;
  box-shadow: var(--veri-focus) !important;
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
  border-radius: 8px !important;
  color: var(--veri-text) !important;
  padding: 1.1rem .9rem !important;
}

.veri-upload-info {
  border: 1px solid rgba(255, 200, 87, .18);
  border-radius: 8px;
  background: rgba(255, 200, 87, .055);
  color: var(--veri-text);
  font-size: .88rem;
  line-height: 1.42;
  margin-bottom: .7rem;
  padding: .8rem .82rem;
}

.veri-upload-info strong {
  color: var(--veri-text);
  font-size: .98rem;
}

.veri-upload-types {
  display: grid;
  gap: .38rem;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: .7rem;
}

.veri-upload-types span {
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  color: var(--veri-text);
  font-size: .76rem;
  font-weight: 760;
  line-height: 1.1;
  padding: .42rem .44rem;
}

.veri-upload-limit {
  color: var(--veri-amber);
  font-size: .72rem;
  font-weight: 800;
  letter-spacing: .07rem;
  margin-top: .62rem;
  text-transform: uppercase;
}

[data-testid="stFileUploaderDropzone"] * {
  color: var(--veri-text) !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
  display: none !important;
}

[data-testid="stFileUploaderDropzoneInstructions"]::before {
  content: "Drop files here";
  display: block;
  color: var(--veri-text);
  font-size: .94rem;
  font-weight: 760;
  margin-bottom: .55rem;
}

[data-testid="stFileUploaderDropzoneInstructions"]::after {
  content: "or browse from your Mac";
  white-space: pre-line;
  display: block;
  color: var(--veri-muted);
  font-size: .82rem;
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
  max-width: 100% !important;
  overflow-x: auto !important;
  overflow-y: auto !important;
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

.veri-source-card-grid {
  display: grid;
  gap: .62rem;
  margin-top: .5rem;
}

.veri-source-card {
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  background: var(--veri-panel);
  padding: .78rem .85rem;
  transition: border-color .16s ease, box-shadow .16s ease, transform .16s ease;
}

.veri-source-card-top {
  align-items: flex-start;
  display: flex;
  gap: .65rem;
  justify-content: space-between;
}

.veri-source-card-title {
  align-items: center;
  color: var(--veri-text);
  display: inline-flex;
  gap: .42rem;
  font-size: .92rem;
  font-weight: 780;
  line-height: 1.25;
  overflow-wrap: anywhere;
}

.veri-source-card-rank {
  align-items: center;
  background: var(--veri-teal);
  border-radius: 6px;
  color: var(--veri-primary-text);
  display: inline-flex;
  flex: none;
  font-size: .74rem;
  font-weight: 800;
  height: 1.15rem;
  justify-content: center;
  min-width: 1.15rem;
  padding: 0 .2rem;
}

.veri-document-explorer {
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  background: var(--veri-panel);
  margin: .72rem 0 .8rem 0;
  padding: .74rem .78rem;
}

.veri-document-explorer-title {
  color: var(--veri-amber);
  font-size: .72rem;
  font-weight: 820;
  letter-spacing: .08rem;
  margin-bottom: .55rem;
  text-transform: uppercase;
}

.veri-document-row {
  align-items: center;
  display: grid;
  gap: .46rem;
  grid-template-columns: 1.3rem minmax(0, 1fr);
  margin-bottom: .4rem;
}

.veri-document-row strong {
  color: var(--veri-muted);
  font-size: .78rem;
  font-weight: 680;
  line-height: 1.2;
  overflow-wrap: anywhere;
}

.veri-welcome {
  animation: veri-fade-up .28s ease both;
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  background:
    linear-gradient(180deg, rgba(54, 209, 196, .08), rgba(255, 200, 87, .045)),
    var(--veri-panel);
  margin: .75rem 0 .8rem 0;
  padding: 1rem;
}

.veri-welcome-kicker {
  color: var(--veri-amber);
  font-size: .76rem;
  font-weight: 820;
  letter-spacing: .08rem;
  text-transform: uppercase;
}

.veri-welcome-title {
  color: var(--veri-text);
  font-size: 1.15rem;
  font-weight: 780;
  line-height: 1.25;
  margin: .34rem 0 .9rem 0;
}

.veri-welcome-grid {
  display: grid;
  gap: .62rem;
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.veri-welcome-grid div {
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  padding: .68rem .7rem;
}

.veri-welcome-grid strong,
.veri-welcome-grid span {
  display: block;
}

.veri-welcome-grid strong {
  color: var(--veri-text);
  font-size: .84rem;
  margin-bottom: .28rem;
}

.veri-welcome-grid span {
  color: var(--veri-muted);
  font-size: .76rem;
  line-height: 1.3;
}

.veri-welcome-hint {
  color: var(--veri-muted);
  font-size: .82rem;
  margin-top: -.4rem;
}

/* Welcome action cards — clickable example prompts under the welcome panel. */
.veri-welcome-action-wrap + div [data-testid="stButton"] button,
div:has(.veri-welcome-action-wrap) + div [data-testid="stButton"] button {
  border-radius: 8px !important;
  justify-content: flex-start !important;
  min-height: 2.7rem !important;
  text-align: left !important;
}

/* Welcome cell — structured title + description blocks (fix 4B) */
.veri-welcome-cell {
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  padding: .68rem .7rem;
}

.veri-welcome-cell-title {
  color: var(--veri-text);
  display: block;
  font-size: .84rem;
  font-weight: 700;
  line-height: 1.2;
  margin-bottom: .28rem;
}

.veri-welcome-cell-desc {
  color: var(--veri-muted);
  display: block;
  font-size: .76rem;
  line-height: 1.3;
}

/* Dashboard divider — visual connector for the toggle (fix 4D) */
.veri-dashboard-divider {
  align-items: center;
  border-top: 1px solid var(--veri-line);
  display: flex;
  gap: .6rem;
  margin: .9rem 0 .5rem 0;
}

.veri-dashboard-divider-label {
  color: var(--veri-muted);
  font-size: .7rem;
  font-weight: 760;
  letter-spacing: .06rem;
  text-transform: uppercase;
  white-space: nowrap;
}

.veri-loading-panel {
  animation: veri-fade-up .18s ease both;
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  background: var(--veri-panel);
  margin-bottom: .75rem;
  padding: .74rem .8rem;
}

.veri-loading-label {
  color: var(--veri-text);
  font-size: .84rem;
  font-weight: 760;
  margin-bottom: .58rem;
}

.veri-loading-steps {
  display: grid;
  gap: .48rem;
  grid-template-columns: repeat(5, minmax(0, 1fr));
}

.veri-loading-step span {
  color: var(--veri-muted);
  display: block;
  font-size: .72rem;
  font-weight: 760;
  margin-bottom: .28rem;
}

.veri-loading-step i {
  background: var(--veri-track);
  border-radius: 999px;
  display: block;
  height: .38rem;
  overflow: hidden;
  position: relative;
}

.veri-loading-step.done i {
  background: var(--veri-teal);
}

.veri-loading-step.active i::after {
  animation: veri-progress 1.15s ease-in-out infinite;
  background: linear-gradient(90deg, transparent, var(--veri-amber), var(--veri-teal), transparent);
  content: "";
  inset: 0;
  position: absolute;
  width: 75%;
}

.veri-source-card-badge {
  border: 1px solid rgba(54, 209, 196, .28);
  border-radius: 999px;
  color: var(--veri-teal);
  flex: 0 0 auto;
  font-size: .72rem;
  font-weight: 800;
  line-height: 1.15;
  padding: .24rem .45rem;
}

.veri-source-card-meta {
  color: var(--veri-muted);
  display: flex;
  flex-wrap: wrap;
  font-size: .78rem;
  gap: .38rem .7rem;
  line-height: 1.3;
  margin-top: .46rem;
}

.veri-source-card-preview {
  color: var(--veri-muted);
  font-size: .86rem;
  line-height: 1.45;
  margin-top: .48rem;
}

.veri-specialized-panel {
  animation: veri-fade-up .24s ease both;
  border: 1px solid rgba(54, 209, 196, .22);
  border-radius: 8px;
  background: rgba(54, 209, 196, .035);
  margin: .9rem 0 .85rem 0;
  padding: .78rem .85rem;
}

.veri-specialized-title {
  color: var(--veri-amber);
  font-size: .76rem;
  font-weight: 840;
  letter-spacing: .08rem;
  margin-bottom: .62rem;
  text-transform: uppercase;
}

.veri-specialized-card {
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  background: var(--veri-panel);
  padding: .78rem .85rem;
}

.veri-specialized-body {
  display: grid;
  gap: .38rem;
  margin-top: .52rem;
}

.veri-specialized-body p {
  color: var(--veri-muted);
  font-size: .82rem;
  line-height: 1.4;
  margin: 0;
  overflow-wrap: anywhere;
}

.veri-specialized-body strong {
  color: var(--veri-text);
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
  :root {
    --veri-sidebar-width: 0px;
    --veri-command-width: calc(100vw - 32px);
    --veri-command-bottom: 18px;
  }

  [data-testid="stSidebar"] {
    min-width: 280px;
    max-width: 280px;
  }

  [data-testid="stSidebar"] > div:first-child {
    min-width: 280px;
    max-width: 280px;
  }

  .block-container {
    padding-top: 1.35rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }

  [data-testid="stBottom"] > div {
    padding: 0 !important;
  }
}

.veri-onboard {
  background: var(--veri-panel-2);
  border: 1px solid var(--veri-line);
  border-radius: 12px;
  margin: 0 0 .9rem 0;
  padding: .85rem 1rem;
}

.veri-onboard-head {
  align-items: center;
  display: flex;
  justify-content: space-between;
  margin-bottom: .6rem;
}

.veri-onboard-kicker {
  color: var(--veri-amber);
  font-size: .74rem;
  font-weight: 820;
  letter-spacing: .08rem;
  text-transform: uppercase;
}

.veri-onboard-progress {
  color: var(--veri-text-dim, #9aa3b2);
  font-size: .76rem;
  font-weight: 700;
}

.veri-onboard-steps {
  display: grid;
  gap: .4rem;
}

.veri-onboard-step {
  align-items: baseline;
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  display: flex;
  gap: .55rem;
  padding: .5rem .65rem;
}

.veri-onboard-mark {
  font-weight: 820;
}

.veri-onboard-mark.done {
  color: var(--veri-green, #7bd88f);
}

.veri-onboard-mark.todo {
  color: var(--veri-text-dim, #9aa3b2);
}

.veri-onboard-step-title {
  color: var(--veri-text);
  font-weight: 720;
}

.veri-onboard-step-detail {
  color: var(--veri-text-dim, #9aa3b2);
  font-size: .8rem;
  margin-left: auto;
}
</style>
"""
