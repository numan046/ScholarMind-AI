"""ScholarMind AI — Premium AI Research Assistant.

Summarizes research papers, extracts key concepts, generates organized reports.
Powered by Groq.

Run with: streamlit run app.py
"""
from __future__ import annotations

import html
import io
import json
import os
import re
import uuid
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

# Optional PDF support
try:
    import pymupdf
    PDF_OK = True
except Exception:
    PDF_OK = False

APP_NAME = "ScholarMind AI"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Model fallbacks — TPM-safe order
# ---------------------------------------------------------------------------
MODEL_FALLBACKS = (
    "llama-3.1-8b-instant",         # highest TPM — safest fallback
    "llama-3.3-70b-versatile",      # best quality
    "moonshotai/kimi-k2-instruct",  # large context
    "openai/gpt-oss-20b",           # low TPM but small
    "qwen/qwen3-32b",               # small TPM
    "openai/gpt-oss-120b",          # absolute last
)

# Approximate token-per-minute limits (Groq free tier)
MODEL_TPM_LIMITS = {
    "llama-3.1-8b-instant": 30000,
    "llama-3.3-70b-versatile": 12000,
    "moonshotai/kimi-k2-instruct": 10000,
    "openai/gpt-oss-20b": 8000,
    "qwen/qwen3-32b": 6000,
    "openai/gpt-oss-120b": 8000,
}

# Rough chars-per-token for safety margin
CHARS_PER_TOKEN = 3.2

# Input safety limits (based on lowest-TPM model we might fall back to)
SAFE_INPUT_CHARS = 20000
HARD_INPUT_CHARS = 45000

MAX_PAPERS = 20
MAX_CHAT_TURNS = 12

# ---------------------------------------------------------------------------
# Research modes
# ---------------------------------------------------------------------------
RESEARCH_MODES = {
    "skim": {
        "label": "⚡ Skim",
        "hint": "Quick 1-minute overview — title, main claim, verdict.",
        "depth": (
            "Provide a HIGH-LEVEL, fast summary. Focus on: what the paper is about, "
            "its main claim, and whether it's worth reading. Keep the whole response short."
        ),
    },
    "standard": {
        "label": "📄 Standard",
        "hint": "Balanced analysis — all sections, moderate depth.",
        "depth": (
            "Provide a BALANCED analysis. Cover all sections with reasonable depth. "
            "Focus on the paper's contribution, method, and key findings."
        ),
    },
    "deep": {
        "label": "🔬 Deep Analysis",
        "hint": "Exhaustive breakdown — methodology, formulas, limitations.",
        "depth": (
            "Provide an EXHAUSTIVE, technical analysis. Include methodology details, "
            "specific metrics/numbers, formulas if mentioned, statistical claims, "
            "limitations, threats to validity, and future work. Cite specific sections "
            "or page numbers from the paper when possible."
        ),
    },
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# PREMIUM CSS — Amber/Rose theme
# ===========================================================================
PREMIUM_CSS = """
<style>
:root {
  --sm-ink: #0f172a;
  --sm-ink-2: #475569;
  --sm-ink-3: #94a3b8;
  --sm-bg: #fffbf5;
  --sm-surface: #ffffff;
  --sm-surface-2: #fef3c7;
  --sm-border: #fde68a;
  --sm-border-2: #fcd34d;
  --sm-primary: #f59e0b;
  --sm-primary-2: #d97706;
  --sm-primary-3: #fbbf24;
  --sm-soft: #fffbeb;
  --sm-soft-2: #fef3c7;
  --sm-accent: #e11d48;
  --sm-accent-2: #be123c;
  --sm-success: #10b981;
  --sm-warning: #f59e0b;
  --sm-danger: #dc2626;
  --sm-info: #0891b2;
  --sm-shadow-sm: 0 1px 2px rgba(15,23,42,.05);
  --sm-shadow-md: 0 4px 14px rgba(15,23,42,.08);
  --sm-shadow-lg: 0 12px 36px rgba(15,23,42,.10);
  --sm-radius-sm: 8px;
  --sm-radius-md: 12px;
  --sm-radius-lg: 18px;
  --sm-radius-xl: 26px;
}

[data-testid="stAppViewContainer"] { background: var(--sm-bg); }
[data-testid="stHeader"] {
  background: rgba(255,251,245,.85);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--sm-border);
}
.block-container { padding: 1.5rem 2.5rem 4rem; max-width: 1500px; }

html, body, .stApp, [data-testid="stAppViewContainer"] {
  color: var(--sm-ink);
}
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"],
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] * {
  color: var(--sm-ink);
}
[data-testid="stCaptionContainer"] *,
[data-testid="stAppViewContainer"] small {
  color: var(--sm-ink-2) !important;
}

.stTextInput input, .stTextArea textarea, .stNumberInput input,
.stChatInput textarea, [data-testid="stChatInput"] textarea,
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea,
[data-baseweb="select"] > div, .stSelectbox div[data-baseweb="select"] > div {
  background-color: #ffffff !important;
  color: var(--sm-ink) !important;
  -webkit-text-fill-color: var(--sm-ink) !important;
  caret-color: var(--sm-primary) !important;
  border: 1px solid var(--sm-border-2) !important;
  border-radius: var(--sm-radius-sm) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus,
[data-testid="stChatInput"] textarea:focus {
  border-color: var(--sm-primary) !important;
  box-shadow: 0 0 0 3px var(--sm-soft-2) !important;
}

[data-baseweb="popover"], [role="listbox"], [role="option"] {
  background-color: #ffffff !important;
  color: var(--sm-ink) !important;
}
[role="option"]:hover, [aria-selected="true"] {
  background-color: var(--sm-soft) !important;
  color: var(--sm-primary-2) !important;
}

.stButton > button, .stDownloadButton > button {
  background: #ffffff !important;
  color: var(--sm-ink) !important;
  -webkit-text-fill-color: var(--sm-ink) !important;
  border: 1px solid var(--sm-border-2) !important;
  border-radius: var(--sm-radius-sm) !important;
  font-weight: 600 !important;
  padding: 0.5rem 1rem !important;
  transition: all .15s ease !important;
  box-shadow: var(--sm-shadow-sm) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--sm-primary) !important;
  color: var(--sm-primary-2) !important;
  -webkit-text-fill-color: var(--sm-primary-2) !important;
  background: var(--sm-soft) !important;
  transform: translateY(-1px) !important;
  box-shadow: var(--sm-shadow-md) !important;
}
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
  background: var(--sm-primary) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  border-color: var(--sm-primary) !important;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
  background: var(--sm-primary-2) !important;
  border-color: var(--sm-primary-2) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #451a03 0%, #78350f 100%) !important;
  border-right: 1px solid rgba(255,255,255,.08) !important;
}
section[data-testid="stSidebar"] * {
  color: #fef3c7 !important;
  -webkit-text-fill-color: #fef3c7 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] strong {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  font-weight: 700 !important;
}
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
  color: #fcd34d !important;
  -webkit-text-fill-color: #fcd34d !important;
}
section[data-testid="stSidebar"] hr {
  border-color: rgba(255,255,255,.12) !important;
  margin: 1rem 0 !important;
}
section[data-testid="stSidebar"] .stButton > button {
  background: rgba(255,255,255,.08) !important;
  color: #fef3c7 !important;
  -webkit-text-fill-color: #fef3c7 !important;
  border: 1px solid rgba(255,255,255,.15) !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(255,255,255,.16) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  border-color: var(--sm-primary-3) !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: var(--sm-primary) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  border-color: var(--sm-primary) !important;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: rgba(255,255,255,.08) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  border: 1px solid rgba(255,255,255,.15) !important;
}
section[data-testid="stSidebar"] [data-baseweb="popover"],
section[data-testid="stSidebar"] [role="option"] {
  background: #78350f !important;
  color: #fef3c7 !important;
  -webkit-text-fill-color: #fef3c7 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
  background-color: rgba(255,255,255,.06) !important;
  border: 2px dashed rgba(255,255,255,.2) !important;
  border-radius: var(--sm-radius-md) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] {
  color: #fef3c7 !important;
  -webkit-text-fill-color: #fef3c7 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * {
  color: #fcd34d !important;
  -webkit-text-fill-color: #fcd34d !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
  background-color: rgba(255,255,255,.06) !important;
  border: 1px solid rgba(255,255,255,.10) !important;
  border-radius: 8px !important;
}
section[data-testid="stSidebar"] [role="slider"] {
  background: var(--sm-primary-3) !important;
}

[data-testid="stMetric"] {
  background: var(--sm-surface) !important;
  border: 1px solid var(--sm-border) !important;
  border-radius: var(--sm-radius-md) !important;
  padding: 1rem 1.15rem !important;
  box-shadow: var(--sm-shadow-sm) !important;
}
[data-testid="stMetricLabel"] * {
  color: var(--sm-ink-2) !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
  font-size: 0.72rem !important;
}
[data-testid="stMetricValue"] * {
  color: var(--sm-ink) !important;
  font-weight: 750 !important;
  font-size: 1.5rem !important;
}

[data-testid="stTabs"] [role="tablist"] {
  border-bottom: 1px solid var(--sm-border) !important;
  gap: 0.25rem !important;
}
[data-testid="stTabs"] button[role="tab"] {
  color: var(--sm-ink-2) !important;
  -webkit-text-fill-color: var(--sm-ink-2) !important;
  font-weight: 600 !important;
  border-radius: var(--sm-radius-sm) var(--sm-radius-sm) 0 0 !important;
  padding: 0.6rem 1rem !important;
  border: none !important;
  background: transparent !important;
}
[data-testid="stTabs"] button[role="tab"]:hover {
  color: var(--sm-primary-2) !important;
  -webkit-text-fill-color: var(--sm-primary-2) !important;
  background: var(--sm-soft) !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: var(--sm-primary-2) !important;
  -webkit-text-fill-color: var(--sm-primary-2) !important;
  background: var(--sm-soft) !important;
  border-bottom: 2px solid var(--sm-primary) !important;
}

[data-testid="stExpander"] {
  border: 1px solid var(--sm-border) !important;
  border-radius: var(--sm-radius-md) !important;
  background: var(--sm-surface) !important;
  box-shadow: var(--sm-shadow-sm) !important;
  overflow: hidden !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
  color: var(--sm-ink) !important;
  -webkit-text-fill-color: var(--sm-ink) !important;
  font-weight: 600 !important;
}

[data-testid="stAlert"] {
  background: var(--sm-surface) !important;
  border-radius: var(--sm-radius-md) !important;
  border: 1px solid var(--sm-border) !important;
}
[data-testid="stAlert"] * { color: var(--sm-ink) !important; }

[data-testid="stChatMessage"] {
  border-radius: var(--sm-radius-lg) !important;
  padding: 1rem 1.15rem !important;
  margin-bottom: 0.85rem !important;
  border: 1px solid var(--sm-border) !important;
  background: var(--sm-surface) !important;
  box-shadow: var(--sm-shadow-sm) !important;
}
[data-testid="stChatMessage"] * { color: var(--sm-ink) !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%) !important;
  border-color: var(--sm-border-2) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
  background: #ffffff !important;
  border-left: 3px solid var(--sm-primary) !important;
}

pre, code, [data-testid="stCode"] * {
  background: #fef3c7 !important;
  color: #451a03 !important;
  border-radius: var(--sm-radius-sm) !important;
}

[data-testid="stDataFrame"] * {
  color: var(--sm-ink) !important;
  background: #ffffff !important;
}

.sm-hero {
  position: relative;
  overflow: hidden;
  padding: 2.25rem 2.5rem;
  margin-bottom: 1.5rem;
  border-radius: var(--sm-radius-xl);
  background:
    radial-gradient(circle at top right, rgba(245,158,11,.18), transparent 50%),
    radial-gradient(circle at bottom left, rgba(225,29,72,.10), transparent 50%),
    linear-gradient(135deg, #ffffff 0%, #fffbeb 100%);
  border: 1px solid var(--sm-border);
  box-shadow: var(--sm-shadow-lg);
}
.sm-hero::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 4px;
  background: linear-gradient(90deg, #f59e0b, #fbbf24, #e11d48);
}
.sm-kicker {
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--sm-primary-2) !important;
  -webkit-text-fill-color: var(--sm-primary-2) !important;
  background: var(--sm-soft-2);
  padding: 0.4rem 0.85rem;
  border-radius: 999px;
  border: 1px solid var(--sm-border-2);
  margin-bottom: 1rem;
}
.sm-title {
  font-size: clamp(1.9rem, 3.4vw, 2.75rem);
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.1;
  margin: 0 0 0.65rem;
  color: var(--sm-ink) !important;
  -webkit-text-fill-color: var(--sm-ink) !important;
}
.sm-title .accent {
  background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 50%, #e11d48 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent !important;
  color: transparent !important;
}
.sm-subtitle {
  font-size: 1.05rem;
  line-height: 1.65;
  color: var(--sm-ink-2) !important;
  -webkit-text-fill-color: var(--sm-ink-2) !important;
  max-width: 800px;
  margin: 0;
}

.sm-card {
  background: var(--sm-surface);
  border: 1px solid var(--sm-border);
  border-radius: var(--sm-radius-lg);
  padding: 1.5rem 1.6rem;
  margin-bottom: 1.25rem;
  box-shadow: var(--sm-shadow-sm);
  transition: box-shadow .2s ease;
}
.sm-card:hover { box-shadow: var(--sm-shadow-md); }
.sm-card.accent-amber { border-top: 3px solid var(--sm-primary); }
.sm-card.accent-rose  { border-top: 3px solid var(--sm-accent); }
.sm-card.accent-green { border-top: 3px solid var(--sm-success); }
.sm-card.accent-cyan  { border-top: 3px solid var(--sm-info); }
.sm-card.accent-purple{ border-top: 3px solid #7c3aed; }

.sm-card-head {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.85rem;
}
.sm-card-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 36px; height: 36px;
  border-radius: 10px;
  background: var(--sm-soft);
  font-size: 1.1rem;
}
.sm-card-title {
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--sm-primary-2) !important;
  -webkit-text-fill-color: var(--sm-primary-2) !important;
  margin: 0;
}
.sm-card-body {
  font-size: 1rem;
  line-height: 1.65;
  color: var(--sm-ink) !important;
  -webkit-text-fill-color: var(--sm-ink) !important;
}

.sm-concept {
  padding: 0.9rem 1.05rem;
  background: #fffbeb;
  border-left: 3px solid var(--sm-primary);
  border-radius: var(--sm-radius-sm);
  margin-bottom: 0.55rem;
  color: var(--sm-ink) !important;
  -webkit-text-fill-color: var(--sm-ink) !important;
}
.sm-concept .name {
  font-weight: 700;
  color: var(--sm-primary-2) !important;
  -webkit-text-fill-color: var(--sm-primary-2) !important;
  font-size: 0.95rem;
}
.sm-concept .def {
  font-size: 0.9rem;
  margin-top: 0.25rem;
  line-height: 1.55;
}

.sm-finding {
  padding: 0.85rem 1rem;
  background: #f0fdf4;
  border-left: 3px solid var(--sm-success);
  border-radius: var(--sm-radius-sm);
  margin-bottom: 0.5rem;
  color: var(--sm-ink) !important;
  -webkit-text-fill-color: var(--sm-ink) !important;
  font-size: 0.92rem;
  line-height: 1.55;
}

.sm-limit {
  padding: 0.85rem 1rem;
  background: #fff1f2;
  border-left: 3px solid var(--sm-accent);
  border-radius: var(--sm-radius-sm);
  margin-bottom: 0.5rem;
  color: var(--sm-ink) !important;
  -webkit-text-fill-color: var(--sm-ink) !important;
  font-size: 0.92rem;
  line-height: 1.55;
}

.sm-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}
.sm-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.4rem 0.85rem;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 600;
  background: var(--sm-soft-2);
  border: 1px solid var(--sm-border-2);
  color: var(--sm-primary-2) !important;
  -webkit-text-fill-color: var(--sm-primary-2) !important;
}

.sm-info-card {
  padding: 1.35rem 1.4rem;
  border-radius: var(--sm-radius-lg);
  background: var(--sm-surface);
  border: 1px solid var(--sm-border);
  box-shadow: var(--sm-shadow-sm);
  height: 100%;
  transition: transform .2s ease, box-shadow .2s ease;
}
.sm-info-card:hover {
  transform: translateY(-3px);
  box-shadow: var(--sm-shadow-md);
}
.sm-info-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 44px; height: 44px;
  border-radius: 12px;
  background: var(--sm-soft-2);
  color: var(--sm-primary-2) !important;
  -webkit-text-fill-color: var(--sm-primary-2) !important;
  font-size: 1.3rem;
  margin-bottom: 0.85rem;
}
.sm-info-card h4 {
  color: var(--sm-ink) !important;
  -webkit-text-fill-color: var(--sm-ink) !important;
  font-size: 1.05rem;
  font-weight: 700;
  margin: 0 0 0.5rem;
}
.sm-info-card p {
  color: var(--sm-ink-2) !important;
  -webkit-text-fill-color: var(--sm-ink-2) !important;
  font-size: 0.92rem;
  line-height: 1.6;
  margin: 0;
}

#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

@media (max-width: 768px) {
  .block-container { padding: 1rem 1rem 3rem !important; }
  .sm-hero { padding: 1.5rem 1.25rem; border-radius: var(--sm-radius-lg); }
  .sm-title { font-size: 1.7rem; }
}
</style>
"""
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


# ===========================================================================
# API helpers
# ===========================================================================

def resolve_api_key() -> Optional[str]:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("GROQ_API_KEY", "")).strip() or None
    except Exception:
        return None


def get_groq_client() -> OpenAI:
    key = resolve_api_key()
    if not key:
        raise RuntimeError(
            "Groq API key is missing. Set GROQ_API_KEY in your environment "
            "or in .streamlit/secrets.toml."
        )
    return OpenAI(api_key=key, base_url=GROQ_BASE_URL)


def _classify_groq_error(exc: Exception) -> tuple[str, str]:
    msg = str(exc).lower()
    if any(w in msg for w in ("401", "unauthorized", "invalid api key", "authentication")):
        return ("auth", "Groq authentication failed. Your GROQ_API_KEY is invalid or expired.")
    if any(w in msg for w in ("402", "insufficient", "payment required")):
        return ("billing", "Groq account has insufficient balance or quota.")
    if any(w in msg for w in ("413", "request too large", "tokens per minute",
                              "tpm", "rate_limit_exceeded")):
        return ("tpm", "Request too large — exceeded the model's TPM limit. "
                       "Try a shorter input or a higher-TPM model (llama-3.1-8b-instant).")
    if any(w in msg for w in ("429", "rate limit", "too many requests")):
        return ("rate_limit", "Groq rate limit reached. Please wait a moment and retry.")
    if any(w in msg for w in ("timeout", "deadline", "connection", "network", "unreachable")):
        return ("network", "Could not reach Groq — network or DNS issue.")
    if any(w in msg for w in ("404", "model_not_found", "does not exist",
                              "decommissioned", "no such model", "invalid model")):
        return ("model", "Model unavailable.")
    if any(w in msg for w in ("context length", "too many tokens", "maximum context", "token limit")):
        return ("context", "Input is too long for this model. Trim the text or switch models.")
    if any(w in msg for w in ("400", "bad request")):
        return ("bad_request", f"Groq rejected the request: {str(exc)[:200]}")
    return ("unknown", f"Groq request failed: {str(exc)[:200]}")


def test_groq(model: str) -> tuple[bool, str]:
    if not resolve_api_key():
        return False, "❌ GROQ_API_KEY not set."
    try:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with just: OK"}],
            max_tokens=5,
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return False, f"⚠️ Empty response from `{model}`."
        return True, f"✅ `{model}` is working. Response: {text[:30]}"
    except Exception as exc:
        cat, msg = _classify_groq_error(exc)
        return False, f"❌ [{cat}] {msg}"


def call_groq_json(
    prompt: str,
    system_instruction: str,
    model: str,
    max_tokens: int = 3000,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Call Groq expecting JSON output. TPM-aware model selection + fallbacks."""
    client = get_groq_client()

    # Estimate input tokens to pre-filter models
    input_chars = len(prompt) + len(system_instruction)
    estimated_input_tokens = int(input_chars / CHARS_PER_TOKEN)
    total_estimated = estimated_input_tokens + max_tokens

    candidates = [model] + [m for m in MODEL_FALLBACKS if m != model]

    # Prefer models whose TPM can handle this request
    def model_fits(m: str) -> bool:
        tpm = MODEL_TPM_LIMITS.get(m, 8000)
        return tpm >= total_estimated

    fitting = [m for m in candidates if model_fits(m)]
    non_fitting = [m for m in candidates if not model_fits(m)]
    ordered = fitting + non_fitting

    last_error = "No models tried."
    tpm_failures = 0

    for model_name in ordered:
        try:
            kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_instruction.strip()},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
            try:
                resp = client.chat.completions.create(
                    **kwargs,
                    response_format={"type": "json_object"},
                )
            except Exception:
                resp = client.chat.completions.create(**kwargs)

            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("Empty response.")

            data = _parse_json_safe(text)
            if isinstance(data, dict):
                try:
                    st.session_state["active_model"] = model_name
                    usage = getattr(resp, "usage", None)
                    if usage is not None:
                        st.session_state["last_usage"] = {
                            "prompt": getattr(usage, "prompt_tokens", 0) or 0,
                            "completion": getattr(usage, "completion_tokens", 0) or 0,
                            "total": getattr(usage, "total_tokens", 0) or 0,
                        }
                except Exception:
                    pass
                return data
            raise RuntimeError(f"Model returned non-JSON: {text[:150]}")
        except Exception as exc:
            cat, msg = _classify_groq_error(exc)
            last_error = f"[{cat}] {msg}"
            if cat in ("model", "context", "tpm"):
                if cat == "tpm":
                    tpm_failures += 1
                continue
            raise RuntimeError(msg)

    if tpm_failures >= max(1, len(ordered) - 1):
        raise RuntimeError(
            f"All models exceeded their TPM (tokens-per-minute) limit. "
            f"Input is ~{estimated_input_tokens:,} tokens + {max_tokens:,} output = ~{total_estimated:,} total. "
            f"Solutions:\n"
            f"1. Switch to `llama-3.1-8b-instant` (30,000 TPM) in the sidebar\n"
            f"2. Paste only abstract + intro + method (not full paper)\n"
            f"3. Wait 60 seconds and retry"
        )

    raise RuntimeError(f"All Groq models failed. Last: {last_error}")


def call_groq_text(
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int = 1200,
    temperature: float = 0.5,
) -> str:
    """Text-only chat call (for Q&A)."""
    client = get_groq_client()
    candidates = [model] + [m for m in MODEL_FALLBACKS if m != model]
    for model_name in candidates:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            out = (resp.choices[0].message.content or "").strip()
            if out:
                try:
                    st.session_state["active_model"] = model_name
                except Exception:
                    pass
                return out
        except Exception as exc:
            cat, _ = _classify_groq_error(exc)
            if cat in ("model", "context", "tpm"):
                continue
            raise RuntimeError(str(exc))
    return "⚠️ No response from any model."


def _parse_json_safe(text: str) -> Any:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(cleaned[s:e + 1])
        except Exception:
            pass
    return None


# ===========================================================================
# File extraction (only PDF/TXT/MD — no URL)
# ===========================================================================

def extract_from_pdf(data: bytes) -> str:
    if not PDF_OK:
        raise RuntimeError("PyMuPDF not installed. Run: pip install PyMuPDF")
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        parts = []
        for page in doc:
            text = page.get_text("text")
            if text:
                parts.append(text)
        return "\n\n".join(parts)
    finally:
        doc.close()


def extract_from_txt(data: bytes) -> str:
    for enc in ("utf-8", "latin-1", "utf-16"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_from_upload(uploaded_file: Any) -> str:
    data = uploaded_file.getvalue()
    name = (getattr(uploaded_file, "name", "") or "").lower()
    if name.endswith(".pdf"):
        return extract_from_pdf(data)
    return extract_from_txt(data)


# ===========================================================================
# Research analysis
# ===========================================================================

RESEARCH_SYSTEM = """You are ScholarMind AI, an expert research analyst.

You read research papers and produce structured, factual JSON summaries.

Return ONLY valid JSON (no markdown fences, no commentary) matching this schema:

{
  "title": "paper title (reconstructed if unclear)",
  "authors": ["Author 1", "Author 2"],
  "year": "YYYY or empty string",
  "venue": "conference/journal name or empty string",
  "field": "research field / subfield",
  "abstract_summary": "3-5 sentence plain-language summary of the abstract and main claim",
  "key_concepts": [
    {"name": "concept name", "definition": "1-2 sentence explanation"}
  ],
  "research_question": "the primary question the paper tries to answer",
  "methodology": "3-5 sentence summary of what they did — data, models, experiments",
  "key_findings": [
    "specific factual finding (include numbers/metrics if available)"
  ],
  "contributions": [
    "novel contribution claimed by the authors"
  ],
  "limitations": [
    "limitation stated or implied by the paper"
  ],
  "future_work": [
    "future direction mentioned"
  ],
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "takeaways": [
    "1-sentence memorable takeaway for a reader"
  ],
  "critical_review": "2-3 sentence neutral critical assessment: what's strong, what's weak",
  "reading_time_minutes": null
}

Rules:
- NEVER invent authors, numbers, or findings. If a field is unclear, use empty string / empty array.
- Preserve exact numbers, dataset names, model names, and technical terms from the source.
- key_findings should be specific (e.g., "Achieved 94.2% accuracy on MNIST"), not vague.
- takeaways must be standalone (understandable without reading the paper).
- critical_review should be neutral and evidence-based, not hype-driven."""


def _smart_truncate_paper(text: str, max_chars: int) -> str:
    """Intelligently truncate a paper while keeping title + abstract + intro + method."""
    if len(text) <= max_chars:
        return text
    first_part = text[:max_chars]
    last_period = max(
        first_part.rfind(". "),
        first_part.rfind(".\n"),
        first_part.rfind("!\n"),
        first_part.rfind("?\n"),
    )
    if last_period > max_chars * 0.7:
        first_part = first_part[:last_period + 1]
    return first_part.strip()


def analyze_paper(
    text: str,
    mode_key: str = "standard",
    language: str = "English",
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Paper text is empty.")

    text = text.strip()
    original_len = len(text)

    # Smart truncation to stay within TPM limits
    if len(text) > SAFE_INPUT_CHARS:
        text = _smart_truncate_paper(text, SAFE_INPUT_CHARS)

    truncated_note = ""
    if original_len > SAFE_INPUT_CHARS:
        truncated_note = (
            f"\n\n[Note: original input was {original_len:,} chars; "
            f"only the first {len(text):,} chars were analyzed to fit API limits.]"
        )

    mode = RESEARCH_MODES.get(mode_key, RESEARCH_MODES["standard"])
    depth_note = f"\n\nDEPTH: {mode['depth']}"

    lang_note = ""
    if language == "Roman Urdu":
        lang_note = (
            "\n\nLANGUAGE: Write all text fields (abstract_summary, definitions, findings, "
            "takeaways, critical_review, etc.) in ROMAN URDU (Urdu written in English letters). "
            "Keep JSON keys in English. Preserve technical terms and numbers as-is."
        )

    prompt = f"""Analyze the following research paper text and produce the structured JSON summary.

PAPER TEXT:
---
{text}
---

Return the JSON schema as specified. Be faithful to the source — no inventions.
If a field genuinely isn't present in the source, use empty string / empty array.{truncated_note}"""

    system = RESEARCH_SYSTEM + depth_note + lang_note
    data = call_groq_json(prompt, system, model=model, max_tokens=3000, temperature=0.2)
    return _normalize_paper(data, text)


def _normalize_paper(data: dict[str, Any], raw_text: str) -> dict[str, Any]:
    def as_list(x: Any) -> list:
        if isinstance(x, list):
            return x
        if x is None:
            return []
        return [x]

    def as_str(x: Any, default: str = "") -> str:
        if x is None:
            return default
        return str(x).strip()

    out: dict[str, Any] = {
        "title": as_str(data.get("title")) or _guess_title(raw_text),
        "authors": [as_str(a) for a in as_list(data.get("authors")) if as_str(a)],
        "year": as_str(data.get("year")),
        "venue": as_str(data.get("venue")),
        "field": as_str(data.get("field")),
        "abstract_summary": as_str(data.get("abstract_summary")),
        "key_concepts": [],
        "research_question": as_str(data.get("research_question")),
        "methodology": as_str(data.get("methodology")),
        "key_findings": [as_str(x) for x in as_list(data.get("key_findings")) if as_str(x)],
        "contributions": [as_str(x) for x in as_list(data.get("contributions")) if as_str(x)],
        "limitations": [as_str(x) for x in as_list(data.get("limitations")) if as_str(x)],
        "future_work": [as_str(x) for x in as_list(data.get("future_work")) if as_str(x)],
        "keywords": [as_str(x) for x in as_list(data.get("keywords")) if as_str(x)],
        "takeaways": [as_str(x) for x in as_list(data.get("takeaways")) if as_str(x)],
        "critical_review": as_str(data.get("critical_review")),
        "reading_time_minutes": data.get("reading_time_minutes") if isinstance(data.get("reading_time_minutes"), int) else None,
        "raw_text": raw_text,
        "analyzed_at": datetime.now().isoformat(),
    }

    for kc in as_list(data.get("key_concepts")):
        if isinstance(kc, dict):
            name = as_str(kc.get("name"))
            if name:
                out["key_concepts"].append({
                    "name": name,
                    "definition": as_str(kc.get("definition")),
                })
        elif isinstance(kc, str) and kc.strip():
            out["key_concepts"].append({"name": kc.strip(), "definition": ""})

    return out


def _guess_title(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:10]:
        if 20 <= len(ln) <= 200 and not ln.lower().startswith(("abstract", "arxiv:", "doi:")):
            return ln[:120]
    return (lines[0][:120] if lines else "Untitled Paper")


def generate_citation(paper: dict[str, Any], style: str = "APA") -> str:
    title = paper.get("title", "Untitled")
    authors = paper.get("authors", [])
    year = paper.get("year", "n.d.")
    venue = paper.get("venue", "")

    def fmt_authors_apa() -> str:
        if not authors:
            return "Unknown Author"
        if len(authors) == 1:
            return authors[0]
        if len(authors) <= 6:
            return ", ".join(authors[:-1]) + ", & " + authors[-1]
        return authors[0] + " et al."

    def fmt_authors_mla() -> str:
        if not authors:
            return "Unknown Author"
        if len(authors) == 1:
            return authors[0]
        if len(authors) == 2:
            return authors[0] + ", and " + authors[1]
        return authors[0] + ", et al."

    if style == "APA":
        parts = [fmt_authors_apa(), f"({year}).", f"{title}."]
        if venue:
            parts.append(f"{venue}.")
        return " ".join(parts)

    if style == "MLA":
        parts = [fmt_authors_mla() + ".", f'"{title}."']
        if venue:
            parts.append(f"{venue},")
        parts.append(f"{year}.")
        return " ".join(parts)

    if style == "Chicago":
        author_str = fmt_authors_apa()
        parts = [author_str + ".", f"{year}.", f'"{title}."']
        if venue:
            parts.append(f"{venue}.")
        return " ".join(parts)

    if style == "BibTeX":
        key = re.sub(r"\W+", "", (authors[0].split()[-1] if authors else "unknown").lower()) + year
        bib_authors = " and ".join(authors) if authors else "Unknown"
        lines = [
            f"@article{{{key},",
            f"  title = {{{title}}},",
            f"  author = {{{bib_authors}}},",
            f"  year = {{{year}}},",
        ]
        if venue:
            lines.append(f"  journal = {{{venue}}},")
        lines.append("}")
        return "\n".join(lines)

    return f"{fmt_authors_apa()} ({year}). {title}."


# ===========================================================================
# Q&A on paper
# ===========================================================================

QA_SYSTEM = """You are ScholarMind AI, answering questions about a specific research paper.

You have:
1. A structured analysis of the paper (summary, concepts, findings).
2. The user's prior conversation (if any).

Rules:
- Answer using ONLY the paper analysis and prior conversation.
- If the answer isn't in the analysis, say so clearly.
- Match the user's language (English or Roman Urdu).
- Be concise — under 200 words unless more detail is requested.
- Do NOT invent numbers, authors, or findings."""


def answer_question(
    question: str,
    paper: dict[str, Any],
    history: list[dict[str, str]],
    language: str = "English",
) -> str:
    if not question.strip():
        return "Please ask a question."

    ctx_parts = [f"Title: {paper.get('title', '')}"]
    if paper.get("authors"):
        ctx_parts.append(f"Authors: {', '.join(paper['authors'])}")
    if paper.get("field"):
        ctx_parts.append(f"Field: {paper['field']}")
    if paper.get("abstract_summary"):
        ctx_parts.append(f"Summary: {paper['abstract_summary']}")
    if paper.get("research_question"):
        ctx_parts.append(f"RQ: {paper['research_question']}")
    if paper.get("methodology"):
        ctx_parts.append(f"Method: {paper['methodology']}")
    if paper.get("key_concepts"):
        concepts = "; ".join(f"{c['name']}: {c['definition']}" for c in paper["key_concepts"][:6])
        ctx_parts.append(f"Concepts: {concepts}")
    if paper.get("key_findings"):
        ctx_parts.append("Findings:\n- " + "\n- ".join(paper["key_findings"][:8]))
    if paper.get("limitations"):
        ctx_parts.append("Limitations:\n- " + "\n- ".join(paper["limitations"][:5]))

    context = "\n\n".join(ctx_parts)

    messages: list[dict[str, str]] = [{"role": "system", "content": QA_SYSTEM}]
    messages.append({"role": "system", "content": f"PAPER ANALYSIS:\n{context}"})
    for turn in history[-MAX_CHAT_TURNS:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    try:
        return call_groq_text(
            messages,
            model=st.session_state.preferred_model,
            max_tokens=700,
            temperature=0.4,
        )
    except Exception as e:
        return f"⚠️ Could not answer: {str(e)[:200]}"


# ===========================================================================
# Exports
# ===========================================================================

def build_markdown_report(paper: dict[str, Any]) -> str:
    p = paper
    lines = [f"# 🔬 {p.get('title', 'Untitled Paper')}", ""]
    if p.get("authors"):
        lines.append(f"**Authors:** {', '.join(p['authors'])}  ")
    if p.get("year"):
        lines.append(f"**Year:** {p['year']}  ")
    if p.get("venue"):
        lines.append(f"**Venue:** {p['venue']}  ")
    if p.get("field"):
        lines.append(f"**Field:** {p['field']}  ")
    lines += ["", "---", ""]

    if p.get("abstract_summary"):
        lines += ["## 📝 Summary", "", p["abstract_summary"], ""]
    if p.get("research_question"):
        lines += ["## ❓ Research Question", "", p["research_question"], ""]
    if p.get("key_concepts"):
        lines.append("## 💡 Key Concepts")
        lines.append("")
        for c in p["key_concepts"]:
            lines.append(f"- **{c['name']}** — {c.get('definition', '')}")
        lines.append("")
    if p.get("methodology"):
        lines += ["## 🔬 Methodology", "", p["methodology"], ""]
    if p.get("key_findings"):
        lines.append("## 🎯 Key Findings")
        lines.append("")
        for f in p["key_findings"]:
            lines.append(f"- {f}")
        lines.append("")
    if p.get("contributions"):
        lines.append("## ⭐ Contributions")
        lines.append("")
        for c in p["contributions"]:
            lines.append(f"- {c}")
        lines.append("")
    if p.get("limitations"):
        lines.append("## ⚠️ Limitations")
        lines.append("")
        for l in p["limitations"]:
            lines.append(f"- {l}")
        lines.append("")
    if p.get("future_work"):
        lines.append("## 🔮 Future Work")
        lines.append("")
        for f in p["future_work"]:
            lines.append(f"- {f}")
        lines.append("")
    if p.get("takeaways"):
        lines.append("## 🎓 Key Takeaways")
        lines.append("")
        for t in p["takeaways"]:
            lines.append(f"- {t}")
        lines.append("")
    if p.get("critical_review"):
        lines += ["## 🧐 Critical Review", "", p["critical_review"], ""]
    if p.get("keywords"):
        lines.append("## 🏷️ Keywords")
        lines.append("")
        lines.append(", ".join(p["keywords"]))
        lines.append("")

    lines += ["---", "", "## 📚 Citation", ""]
    lines.append(f"**APA:** {generate_citation(p, 'APA')}")
    lines.append("")
    lines.append(f"**MLA:** {generate_citation(p, 'MLA')}")
    lines.append("")
    lines.append("```bibtex")
    lines.append(generate_citation(p, "BibTeX"))
    lines.append("```")
    lines.append("")
    lines.append(f"*Generated by ScholarMind AI on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    return "\n".join(lines)


def build_json_report(paper: dict[str, Any]) -> str:
    out = {k: v for k, v in paper.items() if k != "raw_text"}
    return json.dumps(out, ensure_ascii=False, indent=2)


def build_flashcards_csv(paper: dict[str, Any]) -> str:
    rows = []
    for c in paper.get("key_concepts", []):
        rows.append({"Front": c.get("name", ""), "Back": c.get("definition", "")})
    for f in paper.get("key_findings", []):
        rows.append({"Front": f"Key finding: {f[:50]}…", "Back": f})
    for t in paper.get("takeaways", []):
        rows.append({"Front": f"Takeaway: {t[:50]}…", "Back": t})
    if not rows:
        return "Front,Back\n"
    return pd.DataFrame(rows).to_csv(index=False)


def _safe_pdf(s: Any) -> str:
    return str(s).encode("latin-1", "ignore").decode("latin-1")


def build_pdf_report(paper: dict[str, Any]) -> bytes:
    try:
        from fpdf import FPDF
    except Exception:
        return b""

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    PAGE_W = 210
    LM = 10
    RM = 10
    CW = PAGE_W - LM - RM

    def reset_x():
        pdf.set_x(LM)

    def multi(text: str, height: float = 6, font_size: int = 11,
              bold: bool = False, italic: bool = False):
        reset_x()
        style = "B" if bold else ("I" if italic else "")
        pdf.set_font("Helvetica", style, font_size)
        pdf.multi_cell(CW, height, _safe_pdf(text), new_x="LMARGIN", new_y="NEXT")
        reset_x()

    def bullet(text: str, font_size: int = 11):
        reset_x()
        pdf.set_font("Helvetica", "", font_size)
        pdf.multi_cell(CW, 6, _safe_pdf("- " + text), new_x="LMARGIN", new_y="NEXT")
        reset_x()

    def section(title: str):
        reset_x()
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(217, 119, 6)
        pdf.multi_cell(CW, 8, _safe_pdf(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(15, 23, 42)
        reset_x()
        pdf.ln(1)

    def divider():
        reset_x()
        pdf.set_draw_color(253, 230, 138)
        pdf.line(LM, pdf.get_y(), PAGE_W - RM, pdf.get_y())
        pdf.ln(3)

    pdf.set_fill_color(245, 158, 11)
    pdf.rect(0, 0, PAGE_W, 30, "F")
    pdf.set_xy(LM, 8)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(CW, 8, _safe_pdf("ScholarMind AI - Research Report"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(LM)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(CW, 6, _safe_pdf("AI-generated paper summary"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(15, 23, 42)
    reset_x()
    pdf.set_y(38)

    multi(paper.get("title", "Untitled"), height=9, font_size=16, bold=True)
    pdf.ln(2)

    meta = []
    if paper.get("authors"):
        meta.append(f"Authors: {', '.join(paper['authors'])}")
    if paper.get("year"):
        meta.append(f"Year: {paper['year']}")
    if paper.get("venue"):
        meta.append(f"Venue: {paper['venue']}")
    if meta:
        multi("  |  ".join(meta), font_size=10, italic=True)
    pdf.ln(3)
    divider()

    if paper.get("abstract_summary"):
        section("Summary")
        multi(paper["abstract_summary"])
        pdf.ln(2)
    if paper.get("research_question"):
        section("Research Question")
        multi(paper["research_question"])
        pdf.ln(2)
    if paper.get("key_concepts"):
        section("Key Concepts")
        for c in paper["key_concepts"]:
            reset_x()
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(CW, 6, _safe_pdf(c["name"]), new_x="LMARGIN", new_y="NEXT")
            reset_x()
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(CW, 6, _safe_pdf(c.get("definition", "")), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        pdf.ln(1)
    if paper.get("methodology"):
        section("Methodology")
        multi(paper["methodology"])
        pdf.ln(2)
    if paper.get("key_findings"):
        section("Key Findings")
        for f in paper["key_findings"]:
            bullet(f)
        pdf.ln(2)
    if paper.get("contributions"):
        section("Contributions")
        for c in paper["contributions"]:
            bullet(c)
        pdf.ln(2)
    if paper.get("limitations"):
        section("Limitations")
        for l in paper["limitations"]:
            bullet(l)
        pdf.ln(2)
    if paper.get("future_work"):
        section("Future Work")
        for f in paper["future_work"]:
            bullet(f)
        pdf.ln(2)
    if paper.get("takeaways"):
        section("Key Takeaways")
        for t in paper["takeaways"]:
            bullet(t)
        pdf.ln(2)
    if paper.get("critical_review"):
        section("Critical Review")
        multi(paper["critical_review"])
        pdf.ln(2)
    if paper.get("keywords"):
        section("Keywords")
        multi(", ".join(paper["keywords"]))
        pdf.ln(2)

    divider()
    section("Citation (APA)")
    multi(generate_citation(paper, "APA"))
    pdf.ln(2)
    divider()
    multi(
        f"Generated by ScholarMind AI on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        height=5, font_size=9, italic=True,
    )

    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1", "ignore")
    return bytes(out)


# ===========================================================================
# Session state
# ===========================================================================

def init_state() -> None:
    defaults: dict[str, Any] = {
        "papers": {},
        "active_id": None,
        "mode": "standard",
        "language": "English",
        "preferred_model": DEFAULT_MODEL,
        "show_debug": False,
        "connection_test": None,
        "active_model": None,
        "last_usage": None,
        "last_error": None,
        "input_text": "",
        "last_upload_name": None,
        "input_version": 0,
        "chat_history": {},
        "input_source": "paste",  # paste | upload
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def save_paper(paper: dict[str, Any]) -> str:
    pid = str(uuid.uuid4())[:8]
    st.session_state.papers[pid] = paper
    st.session_state.active_id = pid
    if len(st.session_state.papers) > MAX_PAPERS:
        oldest = sorted(
            st.session_state.papers.items(),
            key=lambda kv: kv[1].get("analyzed_at", ""),
        )[:1]
        for oid, _ in oldest:
            del st.session_state.papers[oid]
    return pid


def get_active_paper() -> Optional[dict[str, Any]]:
    aid = st.session_state.active_id
    if aid and aid in st.session_state.papers:
        return st.session_state.papers[aid]
    return None


# ===========================================================================
# Rendering
# ===========================================================================

def esc(s: Any) -> str:
    return html.escape(str(s) if s is not None else "")


def render_hero() -> None:
    st.markdown(
        """
        <div class="sm-hero">
          <div class="sm-kicker">AI-Powered Research Analysis</div>
          <div class="sm-title">🔬 ScholarMind <span class="accent">AI</span></div>
          <div class="sm-subtitle">
            Turn dense research papers into organized reports — key concepts,
            methodology, findings, limitations, and citations. Powered by Groq.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_landing() -> None:
    st.markdown("### How ScholarMind works")
    a, b, c, d = st.columns(4)
    cards = [
        ("📥", "Input a paper", "Paste the text or upload a PDF."),
        ("🧠", "AI analysis", "Groq extracts concepts, methodology, findings, and limitations."),
        ("📊", "Organized report", "Clean cards and tables — a research brief at a glance."),
        ("📤", "Export anywhere", "Markdown, JSON, flashcards CSV, or a styled PDF report."),
    ]
    for col, (icon, title, desc) in zip([a, b, c, d], cards):
        with col:
            st.markdown(
                f'<div class="sm-info-card">'
                f'<div class="sm-info-icon">{icon}</div>'
                f'<h4>{title}</h4>'
                f'<p>{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown("")
    st.info(
        "💡 **Tip:** Works best with papers that include an **abstract**, **introduction**, "
        "**methodology**, **results**, and **conclusion** sections. "
        "Large papers are auto-truncated to fit API limits."
    )


def render_paper(paper: dict[str, Any]) -> None:
    p = paper

    st.markdown(f"## {esc(p.get('title', 'Untitled'))}")

    cols = st.columns(4)
    with cols[0]:
        st.metric("Authors", len(p.get("authors", [])))
    with cols[1]:
        st.metric("Year", p.get("year") or "—")
    with cols[2]:
        st.metric("Concepts", len(p.get("key_concepts", [])))
    with cols[3]:
        st.metric("Findings", len(p.get("key_findings", [])))

    if p.get("authors"):
        authors_chips = "".join(
            f'<span class="sm-chip">👤 {esc(a)}</span>' for a in p["authors"]
        )
        st.markdown(
            f'<div style="margin-top:0.5rem;"><div class="sm-chips">{authors_chips}</div></div>',
            unsafe_allow_html=True,
        )

    if p.get("venue") or p.get("field"):
        bits = []
        if p.get("venue"):
            bits.append(f"📖 {esc(p['venue'])}")
        if p.get("field"):
            bits.append(f"🏷️ {esc(p['field'])}")
        st.caption("  ·  ".join(bits))

    if p.get("abstract_summary"):
        st.markdown(
            f'<div class="sm-card accent-amber">'
            f'<div class="sm-card-head">'
            f'<div class="sm-card-icon">📝</div>'
            f'<p class="sm-card-title">Summary</p>'
            f'</div>'
            f'<div class="sm-card-body">{esc(p["abstract_summary"])}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if p.get("research_question"):
        st.markdown(
            f'<div class="sm-card accent-rose">'
            f'<div class="sm-card-head">'
            f'<div class="sm-card-icon">❓</div>'
            f'<p class="sm-card-title">Research Question</p>'
            f'</div>'
            f'<div class="sm-card-body">{esc(p["research_question"])}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if p.get("key_concepts"):
        blocks = ""
        for c in p["key_concepts"]:
            blocks += (
                f'<div class="sm-concept">'
                f'<div class="name">{esc(c["name"])}</div>'
                f'<div class="def">{esc(c.get("definition", ""))}</div>'
                f'</div>'
            )
        st.markdown(
            f'<div class="sm-card accent-amber">'
            f'<div class="sm-card-head">'
            f'<div class="sm-card-icon">💡</div>'
            f'<p class="sm-card-title">Key Concepts</p>'
            f'</div>'
            f'{blocks}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if p.get("methodology"):
        st.markdown(
            f'<div class="sm-card accent-purple">'
            f'<div class="sm-card-head">'
            f'<div class="sm-card-icon">🔬</div>'
            f'<p class="sm-card-title">Methodology</p>'
            f'</div>'
            f'<div class="sm-card-body">{esc(p["methodology"])}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if p.get("key_findings"):
        items = "".join(f'<div class="sm-finding">🎯 {esc(f)}</div>' for f in p["key_findings"])
        st.markdown(
            f'<div class="sm-card accent-green">'
            f'<div class="sm-card-head">'
            f'<div class="sm-card-icon">🎯</div>'
            f'<p class="sm-card-title">Key Findings</p>'
            f'</div>'
            f'{items}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if p.get("contributions"):
        items = "".join(f"<li>{esc(c)}</li>" for c in p["contributions"])
        st.markdown(
            f'<div class="sm-card accent-amber">'
            f'<div class="sm-card-head">'
            f'<div class="sm-card-icon">⭐</div>'
            f'<p class="sm-card-title">Contributions</p>'
            f'</div>'
            f'<div class="sm-card-body"><ul style="margin:0;padding-left:1.25rem;">{items}</ul></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if p.get("limitations"):
        items = "".join(f'<div class="sm-limit">⚠️ {esc(l)}</div>' for l in p["limitations"])
        st.markdown(
            f'<div class="sm-card accent-rose">'
            f'<div class="sm-card-head">'
            f'<div class="sm-card-icon">⚠️</div>'
            f'<p class="sm-card-title">Limitations</p>'
            f'</div>'
            f'{items}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if p.get("future_work"):
        items = "".join(f"<li>{esc(f)}</li>" for f in p["future_work"])
        st.markdown(
            f'<div class="sm-card accent-cyan">'
            f'<div class="sm-card-head">'
            f'<div class="sm-card-icon">🔮</div>'
            f'<p class="sm-card-title">Future Work</p>'
            f'</div>'
            f'<div class="sm-card-body"><ul style="margin:0;padding-left:1.25rem;">{items}</ul></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if p.get("takeaways"):
        items = "".join(f'<div class="sm-finding">🎓 {esc(t)}</div>' for t in p["takeaways"])
        st.markdown(
            f'<div class="sm-card accent-green">'
            f'<div class="sm-card-head">'
            f'<div class="sm-card-icon">🎓</div>'
            f'<p class="sm-card-title">Key Takeaways</p>'
            f'</div>'
            f'{items}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if p.get("critical_review"):
        st.markdown(
            f'<div class="sm-card accent-purple">'
            f'<div class="sm-card-head">'
            f'<div class="sm-card-icon">🧐</div>'
            f'<p class="sm-card-title">Critical Review</p>'
            f'</div>'
            f'<div class="sm-card-body">{esc(p["critical_review"])}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if p.get("keywords"):
        chips = "".join(f'<span class="sm-chip">🏷️ {esc(k)}</span>' for k in p["keywords"])
        st.markdown(
            f'<div class="sm-card accent-amber">'
            f'<div class="sm-card-head">'
            f'<div class="sm-card-icon">🏷️</div>'
            f'<p class="sm-card-title">Keywords</p>'
            f'</div>'
            f'<div class="sm-chips">{chips}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="sm-card accent-rose">'
        f'<div class="sm-card-head">'
        f'<div class="sm-card-icon">📚</div>'
        f'<p class="sm-card-title">Citations</p>'
        f'</div>'
        f'<div class="sm-card-body">'
        f'<p><strong>APA:</strong> {esc(generate_citation(p, "APA"))}</p>'
        f'<p><strong>MLA:</strong> {esc(generate_citation(p, "MLA"))}</p>'
        f'<p><strong>Chicago:</strong> {esc(generate_citation(p, "Chicago"))}</p>'
        f'<pre style="background:#fef3c7;padding:0.75rem;border-radius:8px;font-size:0.8rem;overflow-x:auto;">{esc(generate_citation(p, "BibTeX"))}</pre>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.get("show_debug"):
        with st.expander("🐛 Raw extracted JSON"):
            st.json({k: v for k, v in p.items() if k != "raw_text"})


def render_qa(paper: dict[str, Any], paper_id: str) -> None:
    st.subheader("💬 Ask about this paper")
    st.caption("Follow-up questions grounded in the extracted analysis.")

    if paper_id not in st.session_state.chat_history:
        st.session_state.chat_history[paper_id] = []
    history = st.session_state.chat_history[paper_id]

    if not history:
        st.markdown("**Try asking:**")
        suggestions = [
            "What is the main contribution of this paper?",
            "How did they evaluate their method?",
            "What are the biggest limitations?",
            "Explain the key concept in simple terms.",
            "What dataset did they use?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(s, key=f"qa_sugg_{paper_id}_{i}", use_container_width=True):
                    history.append({"role": "user", "content": s})
                    reply = answer_question(s, paper, history[:-1], st.session_state.language)
                    history.append({"role": "assistant", "content": reply})
                    st.rerun()

    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    q = st.chat_input("Ask a question about this paper…", key=f"qa_input_{paper_id}")
    if q:
        q = q.strip()
        if q:
            history.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    reply = answer_question(q, paper, history[:-1], st.session_state.language)
                st.markdown(reply)
            history.append({"role": "assistant", "content": reply})
            st.rerun()

    if history:
        if st.button("🗑️ Clear this Q&A", key=f"clear_qa_{paper_id}"):
            st.session_state.chat_history[paper_id] = []
            st.rerun()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    init_state()
    render_hero()

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.markdown("### 📥 Input Source")
        source = st.radio(
            "Source",
            ["✍️ Paste text", "📎 Upload PDF"],
            index=0,
            label_visibility="collapsed",
            key="input_source_radio",
        )

        if st.session_state.input_source != source:
            st.session_state.input_source = source
            st.session_state.input_text = ""
            st.session_state.last_upload_name = None
            st.session_state.input_version += 1

        if source == "📎 Upload PDF":
            uploaded_file = st.file_uploader(
                "Upload PDF",
                type=["pdf", "txt", "md"],
                label_visibility="collapsed",
                key="paper_uploader",
            )
            if uploaded_file:
                if st.session_state.last_upload_name != uploaded_file.name:
                    try:
                        text = extract_from_upload(uploaded_file)
                        if not text or not text.strip():
                            st.warning("⚠️ Could not extract text from this file.")
                        else:
                            st.session_state.input_text = text
                            st.session_state.last_upload_name = uploaded_file.name
                            st.session_state.input_version += 1
                            st.success(f"✅ Loaded {len(text):,} chars")
                    except Exception as e:
                        st.error(f"❌ {e}")

        st.markdown("---")
        st.markdown("### 🎯 Analysis Depth")
        mode_labels = {k: v["label"] for k, v in RESEARCH_MODES.items()}
        current_m = st.session_state.get("mode", "standard")
        m_label = st.selectbox(
            "Mode",
            options=list(mode_labels.values()),
            index=list(mode_labels.keys()).index(current_m) if current_m in mode_labels else 1,
            label_visibility="collapsed",
        )
        st.session_state.mode = [k for k, v in mode_labels.items() if v == m_label][0]
        st.caption(RESEARCH_MODES[st.session_state.mode]["hint"])

        st.markdown("---")
        st.markdown("### 🌐 Language")
        st.selectbox(
            "Output language",
            options=["English", "Roman Urdu"],
            index=0,
            key="language",
        )

        st.markdown("### ⚙️ Model")
        st.selectbox(
            "Model",
            options=list(MODEL_FALLBACKS),
            index=0,
            key="preferred_model",
            help=(
                "llama-3.1-8b-instant — 30,000 TPM (safest for large papers)\n"
                "llama-3.3-70b-versatile — 12,000 TPM (best quality)\n"
                "moonshotai/kimi-k2-instruct — 10,000 TPM\n"
                "others — lower TPM"
            ),
        )

        st.markdown("---")
        st.markdown("### 📚 Paper Library")
        if st.session_state.papers:
            papers_sorted = sorted(
                st.session_state.papers.items(),
                key=lambda kv: kv[1].get("analyzed_at", ""),
                reverse=True,
            )[:MAX_PAPERS]
            for pid, pp in papers_sorted:
                is_active = pid == st.session_state.active_id
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    title = (pp.get("title", "Paper") or "Paper")[:26]
                    if st.button(
                        title,
                        key=f"paper_{pid}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        st.session_state.active_id = pid
                        st.rerun()
                with col_b:
                    if st.button("✕", key=f"delpaper_{pid}", use_container_width=True):
                        del st.session_state.papers[pid]
                        if st.session_state.active_id == pid:
                            remaining = list(st.session_state.papers.keys())
                            st.session_state.active_id = remaining[0] if remaining else None
                        st.rerun()
            if st.button("🗑️ Clear library", use_container_width=True):
                st.session_state.papers = {}
                st.session_state.active_id = None
                st.session_state.chat_history = {}
                st.rerun()
        else:
            st.caption("No papers analyzed yet.")

        st.markdown("---")
        st.markdown("### 🔌 API Status")
        if resolve_api_key():
            st.success("Groq API key detected")
            col_t, col_c = st.columns(2)
            with col_t:
                if st.button("Test", use_container_width=True):
                    with st.spinner("Testing…"):
                        ok, msg = test_groq(st.session_state.get("preferred_model", DEFAULT_MODEL))
                        st.session_state.connection_test = (ok, msg)
            with col_c:
                if st.button("Clear", use_container_width=True):
                    st.session_state.connection_test = None
            if st.session_state.get("connection_test"):
                ok, msg = st.session_state["connection_test"]
                (st.success if ok else st.error)(msg)
            if st.session_state.get("active_model"):
                st.caption(f"Last used: `{st.session_state['active_model']}`")
            usage = st.session_state.get("last_usage")
            if usage:
                st.caption(f"Tokens: p={usage.get('prompt',0)} c={usage.get('completion',0)}")
        else:
            st.warning("Groq API key not configured")

        st.markdown("---")
        st.markdown("### 🐛 Debug")
        st.checkbox("Show debug info", key="show_debug")
        if st.session_state.get("last_error") and st.session_state.get("show_debug"):
            with st.expander("⚠️ Last error", expanded=True):
                st.code(st.session_state["last_error"])

    # ---------------- Main ----------------
    tab_input, tab_report, tab_qa, tab_export = st.tabs(
        ["📝 Input", "📋 Report", "💬 Q&A", "📤 Export"]
    )

    # ---------- Input tab ----------
    with tab_input:
        st.subheader("Research paper text")
        st.caption(
            f"Paste the paper or upload a PDF. Max {HARD_INPUT_CHARS:,} characters. "
            f"Inputs over {SAFE_INPUT_CHARS:,} chars are auto-truncated to fit API limits. "
            "Include the abstract, introduction, and methodology for best results."
        )

        text_key = f"paper_input_v{st.session_state.input_version}"
        paper_text = st.text_area(
            "Paper text",
            value=st.session_state.input_text,
            height=420,
            key=text_key,
            placeholder=(
                "Paste the paper's abstract + introduction + methodology here…\n\n"
                "Or switch to '📎 Upload PDF' in the sidebar to upload the paper file.\n\n"
                "Tip: For very long papers, pasting just the abstract + intro + method "
                "gives the best results and avoids TPM limits."
            ),
            label_visibility="collapsed",
        )
        st.session_state.input_text = paper_text

        # Live character counter with TPM warnings
        char_count = len(paper_text or "")
        if char_count > 0:
            estimated_tokens = int(char_count / CHARS_PER_TOKEN)
            ccount1, ccount2 = st.columns([1, 3])
            with ccount1:
                st.metric("Characters", f"{char_count:,}")
            with ccount2:
                if estimated_tokens > 25000:
                    st.error(
                        f"⚠️ Very large input (~{estimated_tokens:,} tokens). Will be "
                        f"auto-truncated to {SAFE_INPUT_CHARS:,} chars. For full analysis, "
                        f"use **llama-3.1-8b-instant** model."
                    )
                elif estimated_tokens > 12000:
                    st.warning(
                        f"ℹ️ Large input (~{estimated_tokens:,} tokens). May hit TPM limits on "
                        f"some models. If you get an error, switch to **llama-3.1-8b-instant**."
                    )
                elif estimated_tokens > 100:
                    st.success(f"✅ Input size OK (~{estimated_tokens:,} tokens)")

        c1, c2 = st.columns([1, 1])
        with c1:
            run = st.button("🔬 Analyze Paper", type="primary", use_container_width=True)
        with c2:
            if st.button("🗑️ Clear input", use_container_width=True):
                st.session_state.input_text = ""
                st.session_state.last_upload_name = None
                st.session_state.input_version += 1
                st.rerun()

        if run:
            text_to_use = (paper_text or "").strip()
            if not text_to_use:
                st.error("Please paste or upload a paper first.")
            elif len(text_to_use) < 200:
                st.error("Text is too short. Paste more of the paper.")
            elif not resolve_api_key():
                st.error("Groq API key missing.")
            else:
                with st.spinner("Analyzing paper… (may take 15-40 seconds)"):
                    try:
                        paper = analyze_paper(
                            text=text_to_use,
                            mode_key=st.session_state.mode,
                            language=st.session_state.language,
                            model=st.session_state.preferred_model,
                        )
                        save_paper(paper)
                        st.success(f"✅ Analyzed: **{paper['title']}**")
                        st.info("Check the **Report** tab.")
                    except Exception as exc:
                        err = str(exc)
                        st.session_state.last_error = err
                        st.error(f"Analysis failed: {err}")

    # ---------- Report tab ----------
    with tab_report:
        paper = get_active_paper()
        if not paper:
            st.info("No paper analyzed yet. Add one in the Input tab.")
            render_landing()
        else:
            render_paper(paper)

    # ---------- Q&A tab ----------
    with tab_qa:
        paper = get_active_paper()
        if not paper:
            st.info("Analyze a paper first to enable Q&A.")
        else:
            render_qa(paper, st.session_state.active_id or "default")

    # ---------- Export tab ----------
    with tab_export:
        paper = get_active_paper()
        if not paper:
            st.info("Analyze a paper first, then export from here.")
        else:
            st.subheader(f"Export — {paper.get('title', 'Paper')}")

            md = build_markdown_report(paper)
            js = build_json_report(paper)
            fc = build_flashcards_csv(paper)
            pdf_bytes = build_pdf_report(paper)
            citation_apa = generate_citation(paper, "APA")
            citation_bib = generate_citation(paper, "BibTeX")

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "📥 Markdown report (.md)",
                    md,
                    file_name="research_report.md",
                    mime="text/markdown",
                    type="primary",
                    use_container_width=True,
                )
            with c2:
                st.download_button(
                    "🧾 Structured JSON",
                    js,
                    file_name="research_data.json",
                    mime="application/json",
                    use_container_width=True,
                )

            c3, c4 = st.columns(2)
            with c3:
                st.download_button(
                    "🎴 Flashcards (.csv)",
                    fc,
                    file_name="flashcards.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with c4:
                if pdf_bytes:
                    st.download_button(
                        "📄 Report (.pdf)",
                        pdf_bytes,
                        file_name="research_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.button("📄 PDF unavailable", disabled=True, use_container_width=True)

            c5, c6 = st.columns(2)
            with c5:
                st.download_button(
                    "📚 Citation — APA (.txt)",
                    citation_apa,
                    file_name="citation_apa.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with c6:
                st.download_button(
                    "📚 Citation — BibTeX (.bib)",
                    citation_bib,
                    file_name="citation.bib",
                    mime="text/plain",
                    use_container_width=True,
                )

            st.markdown("---")
            st.markdown("#### Preview — Markdown report")
            with st.expander("Click to expand", expanded=False):
                st.code(md, language="markdown")

    st.markdown("---")
    st.caption(
        "🔒 **Privacy** — Paper text is sent to Groq for AI analysis. "
        "Analyzed papers are kept in memory for this session only."
    )


if __name__ == "__main__":
    main()