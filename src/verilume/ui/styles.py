"""Streamlit CSS for the Verilume desktop app."""

from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
<style>
:root {
  --veri-bg: #0b0d10;
  --veri-panel: #14171d;
  --veri-panel-2: #191d24;
  --veri-line: #2b303a;
  --veri-text: #f5f2e8;
  --veri-muted: #9ca6b5;
  --veri-amber: #ffc857;
  --veri-teal: #36d1c4;
  --veri-coral: #ff6b5f;
  --veri-green: #7bd88f;
}

.stApp {
  background:
    linear-gradient(180deg, rgba(255, 200, 87, 0.08) 0%, rgba(11, 13, 16, 0) 280px),
    var(--veri-bg);
  color: var(--veri-text);
}

[data-testid="stSidebar"] {
  background: #101319;
  border-right: 1px solid var(--veri-line);
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
  padding-top: 3.2rem;
  padding-bottom: 2rem;
  max-width: 1400px;
}

h1, h2, h3 {
  letter-spacing: 0;
}

.veri-header {
  position: sticky;
  top: 2.8rem;
  z-index: 20;
  background: linear-gradient(180deg, rgba(14, 15, 17, .98), rgba(14, 15, 17, .92));
  backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--veri-line);
  padding: .75rem 0 1.1rem 0;
  margin-bottom: 1rem;
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
  background: #101319;
  padding: .35rem 0 1rem 0;
  margin-bottom: .35rem;
}

[data-testid="stSidebar"] [data-testid="stExpander"] {
  border: 1px solid var(--veri-line);
  border-radius: 12px;
  background: rgba(20, 23, 29, 0.92);
  overflow: hidden;
}

[data-testid="stSidebar"] [data-testid="stExpander"] details {
  background: transparent;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary {
  background: rgba(26, 31, 39, 0.9);
  border-bottom: 1px solid rgba(43, 48, 58, 0.65);
  font-weight: 700;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
  color: var(--veri-amber);
}

.veri-active-model {
  border: 1px solid rgba(54, 209, 196, .34);
  border-radius: 8px;
  background:
    linear-gradient(180deg, rgba(54, 209, 196, .12), rgba(20, 23, 29, .92));
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
  background: rgba(20, 23, 29, 0.88);
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
  background: var(--veri-panel);
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  padding: .8rem .9rem;
}

[data-testid="stMetricLabel"] {
  color: var(--veri-muted);
}

[data-testid="stMetricValue"] {
  color: var(--veri-text);
  font-size: 1.35rem;
}

[data-testid="stChatMessage"] {
  background: rgba(20, 23, 29, 0.78);
  border: 1px solid rgba(43, 48, 58, 0.85);
  border-radius: 8px;
}

[data-testid="stChatMessage"] p {
  color: var(--veri-text);
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
  background: rgba(20, 23, 29, .88);
  color: var(--veri-muted);
  font-size: .78rem;
  font-weight: 700;
  line-height: 1.2;
  padding: .34rem .58rem;
}

.veri-answer-origin-local span {
  border-color: rgba(123, 216, 143, .36);
  background: rgba(123, 216, 143, .1);
  color: #b9f3c5;
}

.veri-answer-origin-web span {
  border-color: rgba(91, 173, 255, .38);
  background: rgba(91, 173, 255, .1);
  color: #b9d9ff;
}

.veri-answer-origin-ai span {
  border-color: rgba(184, 132, 255, .38);
  background: rgba(184, 132, 255, .1);
  color: #dcc8ff;
}

.veri-answer-origin-hybrid span {
  border-color: rgba(255, 200, 87, .42);
  background: rgba(255, 200, 87, .12);
  color: #ffe3a3;
}

.veri-answer-origin-evidence span {
  border-color: rgba(255, 107, 95, .36);
  background: rgba(255, 107, 95, .1);
  color: #ffc7c2;
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
  color: #ffe3a3;
  font-size: .76rem;
  font-weight: 720;
  line-height: 1.2;
  padding: .3rem .55rem;
}

.veri-history-label {
  margin: 1rem 0 .65rem 0;
  color: var(--veri-muted);
  font-size: .82rem;
  letter-spacing: .08rem;
  text-transform: uppercase;
}

.veri-inline-source-heading {
  border-top: 1px solid rgba(43, 48, 58, .75);
  color: var(--veri-amber);
  font-size: .82rem;
  font-weight: 760;
  margin: .85rem 0 .45rem 0;
  padding-top: .65rem;
}

.veri-inline-source-heading-local {
  color: #6ee7a8;
}

.veri-inline-source-heading-web {
  color: #7cc7ff;
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
  border-left-color: #6ee7a8;
}

.veri-source-section-web {
  border-left-color: #7cc7ff;
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
    linear-gradient(180deg, rgba(255, 200, 87, 0.08), rgba(20, 23, 29, 0.92));
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
    linear-gradient(180deg, rgba(255, 200, 87, 0.08), rgba(20, 23, 29, 0.94));
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
  border-radius: 8px;
  border: 1px solid var(--veri-line);
  background: #1a1f27;
  color: var(--veri-text);
  min-height: 2.35rem;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
  border-color: var(--veri-amber);
  color: var(--veri-amber);
}

.stButton > button[kind="primary"] {
  background: linear-gradient(90deg, #ffc857, #36d1c4);
  color: #101319;
  border: 0;
  font-weight: 700;
}

.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextArea textarea {
  background-color: #12161c !important;
  border-color: var(--veri-line) !important;
  color: var(--veri-text) !important;
  border-radius: 8px;
  -webkit-text-fill-color: var(--veri-text) !important;
  caret-color: var(--veri-amber);
  box-shadow: none !important;
}

.stTextInput input:-webkit-autofill,
.stTextInput input:-webkit-autofill:hover,
.stTextInput input:-webkit-autofill:focus {
  -webkit-box-shadow: 0 0 0 1000px #12161c inset !important;
  -webkit-text-fill-color: var(--veri-text) !important;
  transition: background-color 9999s ease-in-out 0s;
}

[data-baseweb="input"] {
  background-color: #12161c !important;
  border-color: var(--veri-line) !important;
}

[data-baseweb="input"] button {
  background-color: #1a1f27 !important;
  color: var(--veri-text) !important;
}

[data-testid="stFileUploader"] {
  background: var(--veri-panel);
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  padding: .5rem;
}

[data-testid="stFileUploaderDropzone"] {
  background: #12161c !important;
  border: 1px dashed #46505f !important;
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
  content: "Supported:\\A PDF • DOCX • TXT • MD • CSV\\A\\A Maximum size:\\A 200 MB per file";
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
  background: #202734 !important;
  border: 1px solid #596575 !important;
  color: var(--veri-text) !important;
  border-radius: 8px !important;
  font-weight: 700 !important;
}

[data-testid="stFileUploaderFile"] {
  background: #10151d !important;
  border-radius: 8px !important;
  color: var(--veri-text) !important;
}

[data-testid="stFileUploaderFile"] * {
  color: var(--veri-text) !important;
}

.veri-source-block {
  border: 1px solid var(--veri-line);
  border-radius: 8px;
  background: rgba(20, 23, 29, .6);
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
</style>
        """,
        unsafe_allow_html=True,
    )
