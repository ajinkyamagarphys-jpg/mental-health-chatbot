"""
app.py  (v2.0 — Product Upgrade)
=================================
MindGuard — Personalized Mental Health Chatbot

Upgrades over v1 (see CHANGE LOG at bottom for details):
  1. 🎨 Improved colour palette  — soft gradients, calming tones
  2. 🔐 Fixed auth system         — SQLite-backed, no OAuth, persists across refreshes
  3. ✨ CSS animations            — fade-in bubbles, smooth page entry
  4. 💾 Persistent chat history   — sidebar shows previous sessions (ChatGPT-style)
  5. 📜 Replaced T&C              — "What this app provides" disclosure
  6. 🌗 Dark / Light mode toggle  — sidebar switch, no colour glitches
  7. 🌍 Hinglish / multi-language — mixed input passes straight through to NLP
  8. 🤖 Better responses          — last 5 messages passed as context window
  9. 🔑 Forgot password            — direct reset (prototype flow)
  10. 🧠 Context-aware responses   — emotional trend + history fed to response engine

Run:
    streamlit run app.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import random
import re
import os
import time
import uuid
from datetime import datetime
from html import escape

# ── Third-party ───────────────────────────────────────────────────────────────
import requests
import streamlit as st
from dotenv import load_dotenv

try:
    from openai import OpenAI as _OpenAIClient
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

load_dotenv()

# ── Page config  (MUST be the very first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="MindGuard — Mental Health Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Internal project imports ──────────────────────────────────────────────────
from backend.database        import (
    init_db, upsert_session, save_message,
    get_session_history, get_emotion_history,
    get_user_sessions, get_latest_user_session,
)
from backend.auth_db         import (            # NEW auth module
    register_user, verify_user,
    reset_password, user_exists,
    get_user_name, init_users_table,
)
from backend.nlp_engine      import detect_emotion, get_emotion_summary, load_model
from backend.risk_detector   import classify_risk
from backend.response_engine import generate_response, generate_greeting
from backend.data_processor  import get_current_session_summary, records_to_dataframe
from visualization.charts    import (
    emotion_distribution_pie,
    emotion_trend_line,
    risk_level_bar,
    session_intensity_line,
)
from utils.helpers import (
    generate_session_id, sanitize_input,
    risk_badge, emotion_badge, init_session_state,
)

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"
USE_API      = False


# ════════════════════════════════════════════════════════════════════════════
# THEME DEFINITIONS
# Two complete CSS blocks injected based on st.session_state.theme
# ════════════════════════════════════════════════════════════════════════════

_DARK_CSS = """
<style>
@keyframes fadeSlideIn {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0);    }
}
@keyframes pageEntry {
  from { opacity: 0; transform: translateY(-8px); }
  to   { opacity: 1; transform: translateY(0);    }
}

html, body, [class*="css"] {
  font-family: 'Segoe UI', sans-serif;
  color: #E2E8F0;
  transition: all 0.3s ease;
}

.stApp {
  background: #0F172A !important;
  color: #E2E8F0 !important;
}

[data-testid="block-container"] > div:first-child {
  animation: pageEntry 0.5s ease both;
}

section[data-testid="stSidebar"] {
  background: #020617 !important;
  border-right: 1px solid #334155 !important;
  color: #E2E8F0 !important;
}

/* Dark theme radio color */
.stRadio > div > label {
  color: #E2E8F0 !important;
}

.sidebar-session-item {
  padding: 8px 12px;
  border-radius: 12px;
  margin: 4px 0;
  cursor: pointer;
  border: 1px solid #334155;
  background: #12203B;
  color: #E2E8F0;
  transition: background 0.2s ease;
}
.sidebar-session-item:hover {
  background: #1e335b;
}

.user-bubble {
  background: #6366F1 !important;
  color: #FFFFFF !important;
  padding: 12px 16px;
  border-radius: 16px 16px 6px 16px;
  margin: 8px 0;
  max-width: 75%;
  margin-left: auto;
  font-size: 0.95rem;
  box-shadow: 0 4px 12px rgba(0,0,0,0.25);
  animation: fadeSlideIn 0.35s ease both;
  transition: all 0.3s ease;
}

.bot-bubble {
  background: #1E293B !important;
  color: #E2E8F0 !important;
  padding: 12px 16px;
  border-radius: 16px 16px 16px 6px;
  margin: 8px 0;
  max-width: 82%;
  border: 1px solid #334155;
  box-shadow: 0 4px 12px rgba(0,0,0,0.25);
  animation: fadeSlideIn 0.45s ease both;
  transition: all 0.3s ease;
}

/* Force Streamlit built-in chat containers transparent so custom HTML bubbles show */
div[data-testid="stChatMessage"],
div[class*="stChatMessage"],
div[class*="chatMessage"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}

.chat-row { display: flex; align-items: flex-end; gap: 12px; margin: 10px 0; }
.chat-row.user { justify-content: flex-end; }
.chat-row.bot  { justify-content: flex-start; }

.metric-card {
  background: #1E293B !important;
  color: #E2E8F0 !important;
  border-radius: 14px !important;
  border: 1px solid #334155 !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.20) !important;
}

.stChatInput input {
  background: #14213D !important;
  color: #E2E8F0 !important;
  border: 1px solid #334155 !important;
}

button[kind="primary"], button.stButton>button {
  background: linear-gradient(135deg, #A78BFA 0%, #60A5FA 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: 14px !important;
  padding: 10px 16px !important;
  transition: all 0.25s ease !important;
}

button[kind="primary"]:hover, button.stButton>button:hover {
  filter: brightness(1.1) !important;
}

h1, h2, h3, h4, h5, h6, p, span {
  color: #E2E8F0 !important;
}

button[title="Switch to dark theme"], button[title="Switch to light theme"] {
  background: rgba(255,255,255,0.14) !important;
  color: #FFFFFF !important;

}

/* Sidebar view toggle style */
.stRadio > div > label {
  font-weight: 600;
}

..view-switch label {
  color: #A5B4FC !important;
}

button[title="Switch to dark theme"], button[title="Switch to light theme"] {
  background: rgba(255,255,255,0.14) !important;
  color: #FFFFFF !important;
  border: 1px solid rgba(255,255,255,0.45) !important;
  box-shadow: 0 2px 10px rgba(0,0,0,0.25) !important;
  width: 38px !important;
  height: 38px !important;
  min-width: 38px !important;
  min-height: 38px !important;
  border-radius: 999px !important;
  font-size: 1.25rem !important;
  padding: 0 !important;
  outline: none !important;
}
button[title="Switch to dark theme"]:hover, button[title="Switch to light theme"]:hover {
  background: rgba(255,255,255,0.25) !important;
}
/* Suppress tooltip popups globally */
[data-testid="stTooltipIcon"] { display: none !important; }

.auth-card {
  background: #1E293B !important;
  border: 1px solid #334155 !important;
  border-radius: 16px !important;
  padding: 32px !important;
  animation: pageEntry 0.5s ease both !important;
}


</style>
"""

_LIGHT_CSS = """
<style>
@keyframes fadeSlideIn {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0);    }
}
@keyframes pageEntry {
  from { opacity: 0; transform: translateY(-8px); }
  to   { opacity: 1; transform: translateY(0);    }
}

html, body, [class*="css"] {
  font-family: 'Segoe UI', sans-serif;
  color: #2C2C2C;
  transition: all 0.3s ease;
  background: #FDF6EC;
}

.stApp {
  background: linear-gradient(135deg, #FDF6EC, #FDEBD0) !important;
  color: #2C2C2C !important;
}

[data-testid="block-container"] > div:first-child {
  animation: pageEntry 0.5s ease both;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(145deg, #FAF3E0, #F5E6CA) !important;
  border-right: 1px solid #EADBC8 !important;
  box-shadow: inset 2px 2px 6px rgba(0,0,0,0.03), inset -2px -2px 6px rgba(255,255,255,0.6);
  border-radius: 14px;
  color: #2C2C2C !important;
}

/* Sidebar collapse/expand toggle arrow — make it yellow in light mode */
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] button,
button[data-testid="stBaseButton-headerNoPadding"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button {
  color: #D97706 !important;
  background: rgba(245,158,11,0.12) !important;
  border-radius: 8px !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="collapsedControl"] button svg,
button[data-testid="stBaseButton-headerNoPadding"] svg,
[data-testid="stSidebarCollapseButton"] svg {
  fill: #D97706 !important;
  stroke: #D97706 !important;
  color: #D97706 !important;
}
[data-testid="collapsedControl"]:hover,
[data-testid="collapsedControl"] button:hover,
button[data-testid="stBaseButton-headerNoPadding"]:hover,
[data-testid="stSidebarCollapseButton"] button:hover {
  background: rgba(245,158,11,0.22) !important;
  color: #B45309 !important;
}
[data-testid="collapsedControl"]:hover svg,
[data-testid="collapsedControl"] button:hover svg,
button[data-testid="stBaseButton-headerNoPadding"]:hover svg,
[data-testid="stSidebarCollapseButton"] button:hover svg {
  fill: #B45309 !important;
  stroke: #B45309 !important;
}

/* Light theme: remove default dark top/bottom bars and match page colors */
header, [data-testid="stToolbar"], footer, div[data-testid="stSidebar"] + div {
  background: #FDF6EC !important;
  box-shadow: none !important;
  border: none !important;
}

footer {
  background: transparent !important;
  border: none !important;
}

/* Fix the sticky bottom bar that wraps the chat input */
div[data-testid="stBottom"],
div[data-testid="stBottom"] > div,
div[data-testid="stBottom"] > div > div,
section.main > div.block-container ~ div,
.stChatFloatingInputContainer,
.stChatFloatingInputContainer > div,
[class*="stChatInputContainer"],
[class*="stBottomBlockContainer"],
div[class*="bottom"] {
  background: linear-gradient(to top, #FDF6EC 80%, transparent) !important;
  background-color: #FDF6EC !important;
  border-top: 1px solid #EADBC8 !important;
  box-shadow: none !important;
}

.stChatInput,
div[data-testid="stChatInput"],
div[data-testid="stChatInput"] > div,
div[data-testid="stChatInput"] > div > div {
  background: #FEFDF7 !important;
  background-color: #FEFDF7 !important;
  box-shadow: none !important;
  border: none !important;
  color: #2C2C2C !important;
}

.stChatInput input,
.stChatInput textarea,
div[data-testid="stChatInput"] input,
div[data-testid="stChatInput"] textarea {
  background: #FEFDF7 !important;
  background-color: #FEFDF7 !important;
  color: #2C2C2C !important;
  border: 1px solid #CBD5E1 !important;
  border-radius: 12px !important;
  box-shadow: inset 0 1px 2px rgba(15,23,42,0.08) !important;
}

.stChatInput input::placeholder,
.stChatInput textarea::placeholder {
  color: #94A3B8 !important;
}

.sidebar-session-item {
  background: #FFF2D8 !important;
  border: 1px solid #EADBC8 !important;
  border-radius: 14px !important;
  color: #2C2C2C !important;
  transition: all 0.3s ease;
  box-shadow: 0 4px 10px rgba(0,0,0,0.04);
}

.sidebar-session-item:hover {
  transform: scale(1.01);
  background: #FBE8C4 !important;
}

.user-bubble {
  background: linear-gradient(135deg, #D8B65A, #F59E0B) !important;
  color: #FFFFFF !important;
  padding: 12px 16px;
  border-radius: 14px;
  margin: 8px 0;
  max-width: 75%;
  margin-left: auto;
  font-size: 0.95rem;
  box-shadow: 0 6px 15px rgba(245, 158, 11, 0.25);
  animation: fadeSlideIn 0.35s ease both;
  transition: all 0.3s ease;
}

.bot-bubble {
  background: radial-gradient(circle at center, #FFF8E7 60%, #FDEBD0 100%) !important;
  color: #2C2C2C !important;
  padding: 12px 16px;
  border-radius: 14px;
  margin: 8px 0;
  max-width: 82%;
  border: 1px solid #EADBC8;
  box-shadow: 0 6px 15px rgba(0,0,0,0.05);
  animation: fadeSlideIn 0.45s ease both;
  transition: all 0.3s ease;
}

.meta-chip {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
  margin-right: 6px;
  background: #FFF8E7;
  color: #2C2C2C;
  border: 1px solid #EADBC8;
}

.chip-emotion { background: #E0E7FF; color: #1E3A8A; border: 1px solid #A5B4FC; }
.chip-low     { background: #DCFCE7; color: #065F46; border: 1px solid #86EFAC; }
.chip-medium  { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
.chip-high    { background: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }

.detected-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 5px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  border: 1px solid #F59E0B;
  background: rgba(245,158,11,0.12);
  color: #92400E;
  margin-bottom: 10px;
}

/* Force Streamlit built-in chat containers transparent so custom HTML bubbles show */
div[data-testid="stChatMessage"],
div[class*="stChatMessage"],
div[class*="chatMessage"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}
div[data-testid="stChatMessage"] > div,
div[data-testid="stChatMessage"] > div > div,
div[data-testid="stChatMessage"] > div > div > div {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

section[data-testid="stSidebar"] button,
button.stButton>button,
button.stButton>button:hover,
button.stButton>button:focus {
  background: linear-gradient(135deg, #D8B65A, #F59E0B) !important;
  color: #FFFFFF !important;
  border: 1px solid #EADBC8 !important;
  box-shadow: 0 6px 15px rgba(0,0,0,0.05) !important;
  transition: all 0.3s ease !important;
}

button.stButton>button:hover,
section[data-testid="stSidebar"] button:hover {
  transform: scale(1.03) !important;
}

.chat-row { display: flex; align-items: flex-end; gap: 12px; margin: 10px 0; }
.chat-row.user { justify-content: flex-end; }
.chat-row.bot  { justify-content: flex-start; }
.chat-row.bot  { justify-content: flex-start; }

.metric-card {
  background: radial-gradient(circle at center, #FFF8E7 60%, #F5E6CA 100%) !important;
  color: #2C2C2C !important;
  border-radius: 14px !important;
  border: 1px solid #EADBC8 !important;
  box-shadow: 0 6px 15px rgba(0,0,0,0.05) !important;
  transition: all 0.3s ease;
}



.theme-inline {
  border-bottom: 1px solid rgba(100, 116, 139, 0.45) !important;
}

button[kind="primary"], button.stButton>button {
  background: linear-gradient(135deg, #A78BFA 0%, #60A5FA 100%) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 14px !important;
  padding: 10px 16px !important;
  box-shadow: 0 6px 15px rgba(0,0,0,0.05) !important;
  transition: all 0.3s ease !important;
}

button[kind="primary"]:hover, button.stButton>button:hover {
  filter: brightness(1.12) !important;
  transform: scale(1.03) !important;
}

h1, h2, h3, h4, h5, h6, p, span {
  color: #2C2C2C !important;
}

.auth-card {
  background: radial-gradient(circle at center, #FFF8E7 60%, #F5E6CA 100%) !important;
  border: 1px solid #EADBC8 !important;
  border-radius: 14px !important;
  padding: 32px !important;
  animation: pageEntry 0.5s ease both !important;
  box-shadow: 0 6px 15px rgba(0,0,0,0.05) !important;
}

button[title="Switch to dark theme"], button[title="Switch to light theme"] {
  background: #4B5563 !important;
  color: #FFFFFF !important;
  border: 2px solid #374151 !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.20) !important;
  width: 38px !important;
  height: 38px !important;
  min-width: 38px !important;
  min-height: 38px !important;
  border-radius: 999px !important;
  font-size: 1.15rem !important;
  padding: 0 !important;
  outline: none !important;
}
button[title="Switch to dark theme"]:hover, button[title="Switch to light theme"]:hover {
  background: #374151 !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.25) !important;
}
button[title="Switch to dark theme"]:focus, button[title="Switch to light theme"]:focus {
  outline: none !important;
  box-shadow: 0 0 0 2px rgba(75,85,99,0.4) !important;
}
/* Suppress tooltip popup on all sidebar session buttons */
[data-testid="stTooltipIcon"] { display: none !important; }
button[data-testid="stBaseButton-secondary"]:focus { outline: none !important; }

</style>
"""


# ════════════════════════════════════════════════════════════════════════════
# AUTH-PAGE CSS (always dark, hides sidebar)
# ════════════════════════════════════════════════════════════════════════════

_AUTH_CSS_BASE = """
<style>
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"]  { display: none; }
header { display: none !important; }

[data-testid="block-container"] {
  max-width: 1000px;
  padding-top: 5vh;
  padding-bottom: 5vh;
}

/* Fix alignment: nav buttons (Sign up, Log in, Forgot) sit inline with text */
[data-testid="stHorizontalBlock"] [data-testid="stButton"] {
  display: flex !important;
  align-items: center !important;
  margin-top: 0 !important;
}
[data-testid="stHorizontalBlock"] [data-testid="stButton"] > button {
  padding-top: 3px !important;
  padding-bottom: 3px !important;
  min-height: 28px !important;
  height: 28px !important;
  font-size: 0.88rem !important;
  border-radius: 6px !important;
}

/* Auth theme toggle button (top-right corner of auth page) */
.auth-theme-toggle button {
  position: fixed !important;
  top: 18px !important;
  right: 24px !important;
  z-index: 9999 !important;
  width: 38px !important;
  height: 38px !important;
  border-radius: 999px !important;
  font-size: 1.1rem !important;
  padding: 0 !important;
  cursor: pointer !important;
}

/* Suppress tooltips on all buttons */
[data-testid="stTooltipIcon"] { display: none !important; }

@keyframes formEntry {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0);    }
}
.auth-form-anim { animation: formEntry 0.5s ease both; }
</style>
"""

_AUTH_DARK_COLORS = """
<style>
.stApp { background-color: #1A1730 !important; }

div[data-testid="stTextInput"] input {
  background-color: #2A2545 !important;
  border: 1px solid #3A3560 !important;
  color: #FFFFFF !important;
  border-radius: 8px !important;
  padding: 14px 16px !important;
  font-size: 0.95rem !important;
  transition: border 0.25s ease;
}
div[data-testid="stTextInput"] input:focus {
  border: 1px solid #7C6FFF !important;
  box-shadow: 0 0 0 3px rgba(124,111,255,0.15) !important;
}
div[data-testid="stTextInput"] input::placeholder { color: #9A98AD !important; }

button[kind="primary"] {
  background: linear-gradient(135deg, #7C6FFF, #5A8FFF) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 12px !important;
  font-weight: 600 !important;
  font-size: 1rem !important;
  transition: opacity 0.2s;
}
button[kind="primary"]:hover { opacity: 0.88 !important; }

button[kind="secondary"] {
  background-color: transparent !important;
  color: #C4C0FF !important;
  border: 1px solid #4A4570 !important;
  border-radius: 8px !important;
  padding: 12px !important;
  font-size: 0.9rem !important;
  transition: border-color 0.2s;
}
button[kind="secondary"]:hover { border-color: #7C6FFF !important; color: #A9A4FF !important; }

h1, h2, h3, p, span, label { color: #FFFFFF !important; }

.auth-theme-toggle button {
  background: rgba(255,255,255,0.12) !important;
  border: 1px solid rgba(255,255,255,0.35) !important;
  color: #FFFFFF !important;
}
.auth-theme-toggle button:hover { background: rgba(255,255,255,0.22) !important; }
</style>
"""

_AUTH_LIGHT_COLORS = """
<style>
.stApp {
  background: linear-gradient(135deg, #FDF6EC, #FDEBD0) !important;
}

div[data-testid="stTextInput"] input {
  background-color: #FEFDF7 !important;
  border: 1px solid #CBD5E1 !important;
  color: #1F2937 !important;
  border-radius: 8px !important;
  padding: 14px 16px !important;
  font-size: 0.95rem !important;
  transition: border 0.25s ease;
}
div[data-testid="stTextInput"] input:focus {
  border: 1px solid #F59E0B !important;
  box-shadow: 0 0 0 3px rgba(245,158,11,0.15) !important;
}
div[data-testid="stTextInput"] input::placeholder { color: #94A3B8 !important; }

button[kind="primary"] {
  background: linear-gradient(135deg, #D8B65A, #F59E0B) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 12px !important;
  font-weight: 600 !important;
  font-size: 1rem !important;
  transition: opacity 0.2s;
}
button[kind="primary"]:hover { opacity: 0.88 !important; }

button[kind="secondary"] {
  background-color: transparent !important;
  color: #92400E !important;
  border: 1px solid #D8B65A !important;
  border-radius: 8px !important;
  padding: 12px !important;
  font-size: 0.9rem !important;
  transition: border-color 0.2s;
}
button[kind="secondary"]:hover { border-color: #F59E0B !important; color: #78350F !important; }

h1, h2, h3 { color: #1F2937 !important; }
p, span, label { color: #374151 !important; }

.auth-theme-toggle button {
  background: #4B5563 !important;
  border: 2px solid #374151 !important;
  color: #FFFFFF !important;
}
.auth-theme-toggle button:hover { background: #374151 !important; }
</style>
"""

# Keep backward-compat alias used in _inject_theme
_AUTH_CSS = _AUTH_CSS_BASE


# ════════════════════════════════════════════════════════════════════════════
# UI HELPER FUNCTIONS (frontend only — no backend changes)
# ════════════════════════════════════════════════════════════════════════════

def _strip_basic_markdown(text: str) -> str:
    if not text:
        return ""
    for ch in ("```", "**", "*", "`"):
        text = text.replace(ch, "")
    return text.strip()


def _risk_label_word(risk_level: str) -> str:
    return {"LOW": "Low", "MEDIUM": "Medium", "HIGH": "High"}.get(risk_level, risk_level)


def _render_user_bubble_html(message: str, emotion: str, risk_level: str) -> str:
    msg_e = escape(message or "")
    return f"""
    <div class="chat-row user">
      <div class="bubble-wrap">
        <div class="user-bubble">{msg_e}</div>
        <div class="meta-row right">
          <span class="meta-chip chip-emotion">{emotion_badge(emotion)}</span>
          <span class="meta-chip chip-{risk_level.lower()}">{risk_badge(risk_level)}</span>
        </div>
      </div>
      <div class="avatar user" aria-hidden="true">👤</div>
    </div>
    """


def _render_assistant_bubble_html(
    bot_response: str,
    detected_emotion: str,
    detected_risk: str,
) -> str:
    emotion_label  = emotion_badge(detected_emotion)
    risk_word      = _risk_label_word(detected_risk)
    badge_html     = (
        f"<span class='meta-chip chip-emotion' style='margin-right:6px;'>"
        f"{emotion_label}</span>"
        f"<span class='meta-chip chip-{detected_risk.lower()}'>"
        f"{risk_word} Risk</span>"
    )

    # Convert the response text to simple HTML paragraphs — no rigid chunks
    # Keep line breaks as paragraph breaks for a natural flow
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", bot_response.strip()) if p.strip()]
    if not paragraphs:
        paragraphs = [bot_response.strip()]

    paras_html = "".join(
        f"<p style='margin:0 0 10px 0;line-height:1.65;'>{escape(p)}</p>"
        for p in paragraphs
    )

    return f"""
    <div class="chat-row bot">
      <div class="avatar bot" aria-hidden="true">🤖</div>
      <div class="bubble-wrap">
        <div class="bot-bubble" style="padding:14px 18px;">
          <div style="margin-bottom:10px;">{badge_html}</div>
          {paras_html}
        </div>
      </div>
    </div>
    """


# ════════════════════════════════════════════════════════════════════════════
# LOCAL RESPONSE GENERATION (Ollama / local fallback)
# ════════════════════════════════════════════════════════════════════════════

def generate_ollama_response(
    user_message,
    emotion,
    risk_level,
    recent_messages,
):
    """
    Attempt Ollama local response via http://localhost:11434/api/generate.

    Falls back to None on missing connection or errors.
    """
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3:latest")

    prompt_lines = [
        "You are MindGuard, a warm and empathetic mental-health support companion.",
        "Listen, validate feelings, and gently guide with supportive empathy.",
        "Write in natural paragraphs and end with an open question.",
        f"Emotion: {emotion}. Risk: {risk_level}.",
        f"User: {user_message}",
    ]
    for msg in recent_messages[-4:]:
        if msg:
            prompt_lines.append(f"Previous: {msg}")
    prompt = "\n".join(prompt_lines)

    try:
        url = f"{host}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}
        print(f"[MindGuard] Ollama request model={model} url={url}")

        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        text = data.get("response") or data.get("text") or data.get("content")
        if text:
            return str(text).strip()

        return str(data).strip()

    except Exception as exc:
        print(f"[MindGuard] Ollama API call failed: {exc}")
        return None


def generate_openai_response(
    user_message,
    emotion,
    risk_level,
    recent_messages,
):
    """
    Generate a response via Ollama local model and fall back to local response engine.
    """
    ollama_response = generate_ollama_response(
        user_message=user_message,
        emotion=emotion,
        risk_level=risk_level,
        recent_messages=recent_messages,
    )

    if ollama_response:
        return ollama_response

    return generate_response(
        user_message=user_message,
        emotion=emotion,
        emotion_score=0.8,
        risk_level=risk_level,
        recent_emotions=[],
    )


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ════════════════════════════════════════════════════════════════════════════

init_session_state({
    "logged_in":             False,
    "user_name":             "",
    "user_email":            "",
    "session_id":            None,
    "messages":              [],
    "emotion_records":       [],
    "model_loaded":          False,
    "is_typing":             False,
    "quick_action_message":  "",
    "active_view":           "chat",
    "active_tab":            "chat",
    "page":                  "login",   # login | signup | forgot | main
    "theme":                 "dark",    # dark  | light
})

# ── Initialise DB + load NLP model once ──────────────────────────────────────
if not st.session_state.model_loaded:
    with st.spinner("🧠 Loading AI model… (first load ~30 sec)"):
        init_db()
        init_users_table()   # ensure users table exists
        load_model()
    st.session_state.model_loaded = True


# ════════════════════════════════════════════════════════════════════════════
# INJECT THEME CSS
# ════════════════════════════════════════════════════════════════════════════

def _inject_theme():
    """Inject the appropriate CSS block based on current theme."""
    if st.session_state.page in ("login", "signup", "forgot"):
        st.markdown(_AUTH_CSS_BASE, unsafe_allow_html=True)
        if st.session_state.theme == "light":
            st.markdown(_AUTH_LIGHT_COLORS, unsafe_allow_html=True)
        else:
            st.markdown(_AUTH_DARK_COLORS, unsafe_allow_html=True)
    elif st.session_state.theme == "light":
        st.markdown(_LIGHT_CSS, unsafe_allow_html=True)
    else:
        st.markdown(_DARK_CSS, unsafe_allow_html=True)

_inject_theme()

# Shrink UI buttons to make thumbs feedback bar feel compact.
st.markdown(
    """
    <style>
    button.stButton>button {
      min-width: 50px !important;
      min-height: 36px !important;
      padding: 4px 8px !important;
      font-size: 1.05rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# (Direct UI toggle, avoids creating full reload via URL anchor that can lose login state.)


# ════════════════════════════════════════════════════════════════════════════
# AUTH PAGES  (login / signup / forgot password)
# ════════════════════════════════════════════════════════════════════

def _render_auth_branding():
    """Left column: decorative branding panel."""
    st.markdown("""
<div style="
  background: linear-gradient(160deg, #2D1B69 0%, #1A3A5C 100%);
  border-radius: 16px;
  height: 640px;
  position: relative;
  padding: 32px;
  color: white;
  box-shadow: 0 8px 40px rgba(0,0,0,0.4);
  overflow: hidden;
">
  <div style="font-size:1.4rem; font-weight:700; letter-spacing:2px;">🧠 MINDGUARD</div>
  <div style="
    position: absolute; bottom: 40px; left: 32px; right: 32px;
  ">
    <h1 style="margin:0; font-weight:400; font-size:2.1rem; line-height:1.25;
               color:white; text-shadow:0 2px 12px rgba(0,0,0,0.4);">
      Finding Peace,<br>Building Resilience
    </h1>
    <p style="color:rgba(255,255,255,0.65); margin-top:12px; font-size:0.9rem; line-height:1.5;">
      A safe space to talk, reflect, and feel heard.<br>
      Powered by empathetic AI, guided by your emotions.
    </p>
    <div style="display:flex; gap:8px; margin-top:20px;">
      <div style="width:24px;height:3px;background:white;border-radius:2px;"></div>
      <div style="width:24px;height:3px;background:rgba(255,255,255,0.3);border-radius:2px;"></div>
      <div style="width:24px;height:3px;background:rgba(255,255,255,0.3);border-radius:2px;"></div>
    </div>
  </div>
  <!-- decorative circles -->
  <div style="position:absolute;top:-60px;right:-60px;width:220px;height:220px;
    border-radius:50%;background:rgba(255,255,255,0.04);"></div>
  <div style="position:absolute;top:80px;right:20px;width:120px;height:120px;
    border-radius:50%;background:rgba(124,111,255,0.1);"></div>
</div>""", unsafe_allow_html=True)


def render_auth_page():
    """Entry point for login / signup / forgot-password flow."""

    # ── Theme toggle fixed at top-right of auth page ──────────────────────────
    st.markdown("<div class='auth-theme-toggle'>", unsafe_allow_html=True)
    toggle_label = "☀️" if st.session_state.theme == "dark" else "🌙"
    if st.button(toggle_label, key="auth_theme_toggle"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    col_brand, col_form = st.columns([1.1, 1], gap="large")

    with col_brand:
        _render_auth_branding()

    with col_form:
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

        # ── WHAT THIS APP PROVIDES  (replaces T&C) ───────────────────────────
        st.markdown("""
<div class="auth-form-anim" style="
  background: rgba(124,111,255,0.1);
  border: 1px solid rgba(124,111,255,0.25);
  border-radius: 12px;
  padding: 14px 18px;
  margin-bottom: 22px;
  font-size: 0.83rem;
  color: #C4C0FF;
  line-height: 1.65;
">
  <strong style="color:#A9A4FF; font-size:0.88rem;">✨ What MindGuard provides</strong><br>
  &bull; &nbsp;Emotional support &amp; a safe space to vent<br>
  &bull; &nbsp;Real-time emotion &amp; early risk detection<br>
  &bull; &nbsp;Guided coping suggestions &amp; reflections<br>
  <span style="color:#888; font-size:0.78rem; margin-top:6px; display:block;">
    ⚠️ Not a replacement for professional therapy.
    Crisis line: <strong>9152987821</strong> (India) · <strong>988</strong> (US)
  </span>
</div>""", unsafe_allow_html=True)

        # ── PAGE ROUTING ─────────────────────────────────────────────────────
        page = st.session_state.page

        if page == "signup":
            _render_signup_form()
        elif page == "forgot":
            _render_forgot_form()
        else:
            _render_login_form()


# ── SIGNUP ────────────────────────────────────────────────────────────────────
def _render_signup_form():
    st.markdown(
        "<h1 style='font-size:2.2rem;margin-bottom:4px;' "
        "class='auth-form-anim'>Create account</h1>",
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns([2, 1])
    col_a.markdown(
        "<span style='font-size:0.88rem; opacity:0.7;'>Already have an account?</span>",
        unsafe_allow_html=True,
    )
    if col_b.button("Log in →", key="go_login", use_container_width=True):
        st.session_state.page = "login"
        st.rerun()

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        first_name = st.text_input("First name", placeholder="First name",
                                   label_visibility="collapsed")
    with c2:
        last_name  = st.text_input("Last name", placeholder="Last name",
                                   label_visibility="collapsed")

    email    = st.text_input("Email", placeholder="your@email.com",
                             label_visibility="collapsed")
    password = st.text_input("Password", placeholder="Create a password (6+ chars)",
                             type="password", label_visibility="collapsed")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("Create account", type="primary", use_container_width=True):
        full_name = f"{first_name.strip()} {last_name.strip()}".strip()
        ok, msg = register_user(full_name or first_name, email, password)
        if ok:
            st.success("✅ " + msg + "  Please log in.")
            st.session_state.page = "login"
            st.rerun()
        else:
            st.error(msg)


# ── LOGIN ─────────────────────────────────────────────────────────────────────
def _render_login_form():
    st.markdown(
        "<h1 style='font-size:2.2rem;margin-bottom:4px;' "
        "class='auth-form-anim'>Welcome back 👋</h1>",
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns([2, 1])
    col_a.markdown(
        "<span style='font-size:0.88rem; opacity:0.7;'>New here?</span>",
        unsafe_allow_html=True,
    )
    if col_b.button("Sign up →", key="go_signup", use_container_width=True):
        st.session_state.page = "signup"
        st.rerun()

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    email    = st.text_input("Email", placeholder="your@email.com",
                             label_visibility="collapsed")
    password = st.text_input("Password", placeholder="Your password",
                             type="password", label_visibility="collapsed")

    col_rem, col_forgot = st.columns([1, 1])
    with col_forgot:
        if st.button("Forgot password?", key="go_forgot",
                     use_container_width=False):
            st.session_state.page = "forgot"
            st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("Log in", type="primary", use_container_width=True):
        ok, name = verify_user(email, password)
        if ok:
            _do_login(email, name)
        else:
            st.error("Incorrect email or password. Please try again.")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Guest bypass (dev / demo convenience)
    if st.button("👋 Continue as Guest", use_container_width=True):
        guest_email = f"guest_{uuid.uuid4().hex[:8]}@mindguard.guest"
        _do_login(guest_email, "Guest")


def _do_login(email: str, name: str):
    """Common login finalisation — sets session state and loads history."""
    st.session_state.logged_in  = True
    st.session_state.user_email = email
    st.session_state.user_name  = (name or email.split("@")[0]).title()

    latest = get_latest_user_session(email)
    if latest:
        st.session_state.session_id    = latest
        st.session_state.messages      = get_session_history(latest, email)
        st.session_state.emotion_records = get_emotion_history(latest, email)
    else:
        new_sid = generate_session_id()
        upsert_session(new_sid, email)
        st.session_state.session_id    = new_sid
        st.session_state.messages      = []
        st.session_state.emotion_records = []

    st.session_state.page = "main"
    st.rerun()


# ── FORGOT PASSWORD ───────────────────────────────────────────────────────────
def _render_forgot_form():
    st.markdown(
        "<h1 style='font-size:2.1rem;margin-bottom:4px;' "
        "class='auth-form-anim'>Reset password 🔑</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.88rem; opacity:0.7;'>Enter your email and choose a new password.</p>",
        unsafe_allow_html=True,
    )

    if st.button("← Back to login", key="forgot_back"):
        st.session_state.page = "login"
        st.rerun()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    email        = st.text_input("Email", placeholder="your@email.com",
                                 label_visibility="collapsed")
    new_password = st.text_input("New password", placeholder="New password (6+ chars)",
                                 type="password", label_visibility="collapsed")
    confirm_pw   = st.text_input("Confirm new password",
                                 placeholder="Confirm new password",
                                 type="password", label_visibility="collapsed")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("Reset password", type="primary", use_container_width=True):
        if new_password != confirm_pw:
            st.error("Passwords do not match.")
        else:
            ok, msg = reset_password(email, new_password)
            if ok:
                st.success("✅ " + msg)
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error(msg)


# ════════════════════════════════════════════════════════════════════════════
# GATE — show auth if not logged in
# ════════════════════════════════════════════════════════════════════════════

if not st.session_state.logged_in:
    render_auth_page()
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR  (only shown when logged in)
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── User greeting ────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-size:1.05rem; font-weight:600; padding:6px 0 2px;'>"
        f"👤 {st.session_state.user_name}</div>",
        unsafe_allow_html=True,
    )
    st.caption(st.session_state.user_email)
    st.divider()

    # ── Dark / Light toggle moved to top-right header (see header section)
    st.divider()

    # ── View mode switch (chat / dashboard) ───────────────────────────────────
    view_mode = st.radio(
        "Switch view",
        ("Chat", "Dashboard"),
        index=0 if st.session_state.active_view in ("chat", None) else 1,
        key="sidebar_view_switch",
        horizontal=True,
    )
    st.session_state.active_view = "chat" if view_mode == "Chat" else "dashboard"

    st.markdown("<div style='margin-top:10px; margin-bottom:6px; font-size:0.80rem; color: #7C7C7C;'>Use this toggle to switch sections</div>", unsafe_allow_html=True)

    if st.button("➕ New Chat", use_container_width=True, key="sidebar_new_chat"):
        new_sid = generate_session_id()
        upsert_session(new_sid, st.session_state.user_email)
        st.session_state.session_id = new_sid
        st.session_state.messages = []
        st.session_state.emotion_records = []
        st.session_state.quick_action_message = ""
        st.session_state.active_view = "chat"
        st.rerun()

    if st.button("📊 Insights Dashboard", use_container_width=True, key="sidebar_dashboard"):
        st.session_state.active_view = "dashboard"
        st.rerun()

    st.divider()

    # ── Chat History (ChatGPT-style) ─────────────────────────────────────────
    st.markdown("**💬 Chat History**")

    try:
        all_sessions = get_user_sessions(st.session_state.user_email) or []
    except Exception:
        all_sessions = []

    if all_sessions:
        for session in all_sessions[:12]:        # show at most 12 sessions
            s_id = session.get("session_id") if isinstance(session, dict) else session
            s_id_txt = "" if s_id is None else str(s_id)
            label = s_id_txt[:20] + "…"
            active = (s_id == st.session_state.session_id)
            btn_style = (
                "background:rgba(124,111,255,0.25);border:1px solid rgba(124,111,255,0.5);"
                if active else ""
            )
            if st.button(
                ("▶ " if active else "  ") + label,
                key=f"sess_{s_id}",
                use_container_width=True,
            ):
                if not active:
                    st.session_state.session_id    = s_id
                    st.session_state.messages      = get_session_history(
                        s_id, st.session_state.user_email
                    )
                    st.session_state.emotion_records = get_emotion_history(
                        s_id, st.session_state.user_email
                    )
                    st.rerun()
    else:
        st.caption("No previous sessions yet.")

    st.divider()

    # ── Emergency help ───────────────────────────────────────────────────────
    st.subheader("🚨 Need urgent help?")
    st.markdown("📞 **iCall (India):** `9152987821`")
    st.markdown("📞 **US/Canada:** `988`")
    st.markdown(
        "<div style='font-size:0.75rem;color:#888;margin-top:6px;'>"
        "⚠️ AI tool — not a therapy replacement.</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Logout ───────────────────────────────────────────────────────────────
    if st.button("🚪 Logout", use_container_width=True):
        for k in ["logged_in", "user_name", "user_email", "session_id",
                  "messages", "emotion_records"]:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state.page = "login"
        st.session_state.logged_in = False
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# MAIN HEADER (Clean top bar)
# ════════════════════════════════════════════════════════════════════════════

header_cols = st.columns([0.1, 1, 0.1])
with header_cols[0]:
    # Reserved left area for sidebar toggle proximity (no explicit menu button)
    st.write("")
with header_cols[1]:
    st.markdown(
        "<h2 style='margin:0; padding:0.3rem 0; font-size:1.65rem; font-weight:700; "
        "text-align:center; letter-spacing:0.01em;'>MindGuard</h2>",
        unsafe_allow_html=True,
    )
with header_cols[2]:
    # Theme toggle button (sun/moon) in top-right
    if st.session_state.theme == "light":
        if st.button("🌙", key="top_theme_toggle", use_container_width=False):
            st.session_state.theme = "dark"
            st.rerun()
    else:
        if st.button("☀️", key="top_theme_toggle", use_container_width=False):
            st.session_state.theme = "light"
            st.rerun()

st.markdown("<hr class='theme-inline' style='border:none;margin:4px 0 10px;'/>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════════════════

if st.session_state.active_view in ("chat", None):
    st.markdown("<h1>Hi 👋, how are you feeling today?</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#888;font-size:1rem;'>"
        "You're in a safe space. Share anything on your mind 💙 &nbsp;"
        "<span style='font-size:0.8rem;color:#666;'>"
        "(English or Hinglish — both welcome!)</span></p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Render existing messages ───────────────────────────────────────────
    last_asst_idx = max(
        (i for i, m in enumerate(st.session_state.messages)
         if m.get("role") == "assistant"),
        default=-1,
    )

    for idx, msg in enumerate(st.session_state.messages):
        role = msg.get("role")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(
                    _render_user_bubble_html(
                        message=msg.get("content", ""),
                        emotion=msg.get("emotion", "neutral"),
                        risk_level=msg.get("risk_level", "LOW"),
                    ),
                    unsafe_allow_html=True,
                )
        else:
            with st.chat_message("assistant"):
                st.markdown(
                    _render_assistant_bubble_html(
                        bot_response=msg.get("content", ""),
                        detected_emotion=msg.get("emotion", "neutral"),
                        detected_risk=msg.get("risk_level", "LOW"),
                    ),
                    unsafe_allow_html=True,
                )
                if idx == last_asst_idx:
                    assistant_count = sum(1 for m in st.session_state.messages if m.get("role") == "assistant")
                    last_feedback_at = st.session_state.get("feedback_last_assistant_count", 0)
                    next_interval = st.session_state.get("feedback_interval", random.randint(4, 6))
                    if assistant_count - last_feedback_at >= next_interval:
                        feedback_cols = st.columns([1, 1], gap="small")
                        if feedback_cols[0].button("👍", key=f"like_{idx}", help="I liked this response"):
                            st.session_state["last_feedback"] = {"index": idx, "v": "liked"}
                            st.session_state["feedback_last_assistant_count"] = assistant_count
                            st.session_state["feedback_interval"] = random.randint(4, 6)
                            st.session_state.quick_action_message = "Thanks! Can you give me a short summary of the key advice?"
                            st.rerun()
                        if feedback_cols[1].button("👎", key=f"dislike_{idx}", help="I did not like this response"):
                            st.session_state["last_feedback"] = {"index": idx, "v": "disliked"}
                            st.session_state["feedback_last_assistant_count"] = assistant_count
                            st.session_state["feedback_interval"] = random.randint(4, 6)
                            st.session_state.quick_action_message = "Please try again with more empathy and practical steps."
                            st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Type how you're feeling… (e.g. 'Bahut stressed hoon aaj' or 'I'm anxious')"
    )

    quick_msg      = st.session_state.get("quick_action_message", "")
    effective_input = quick_msg if quick_msg else user_input
    if quick_msg:
        st.session_state.quick_action_message = ""

    if effective_input:
        clean_input = sanitize_input(effective_input)
        if not clean_input:
            st.warning("Please enter a message before sending.")
        else:
            # ── NLP pipeline (unchanged) ─────────────────────────────────
            nlp_result    = detect_emotion(clean_input)
            emotion       = nlp_result["primary_emotion"]
            emotion_score = nlp_result["primary_score"]

            risk_level, risk_explanation = classify_risk(clean_input, emotion, emotion_score)

            # ── Context window: last 5 user messages ─────────────────────
            recent_emotions = [
                r["emotion"] for r in st.session_state.emotion_records[-10:]
                if isinstance(r, dict) and "emotion" in r
            ]
            recent_messages = [
                m.get("content", "")
                for m in st.session_state.messages[-10:]
                if isinstance(m, dict) and m.get("role") == "user" and "content" in m
            ][-5:]

            bot_response = generate_openai_response(
                user_message=clean_input,
                emotion=emotion,
                risk_level=risk_level,
                recent_messages=recent_messages,
            )

            # ── Render user bubble ────────────────────────────────────────
            with st.chat_message("user"):
                st.markdown(
                    _render_user_bubble_html(
                        message=clean_input,
                        emotion=emotion,
                        risk_level=risk_level,
                    ),
                    unsafe_allow_html=True,
                )

            # ── Render assistant with typing effect ───────────────────────
            with st.chat_message("assistant"):
                placeholder = st.empty()
                thinking    = st.empty()

                thinking.markdown(
                    "<div style='color:#a9a4ff;font-style:italic'>🧠 Thinking…</div>"
                    "<div style='color:#888;font-size:0.9rem;margin-top:4px;'>"
                    "Analysing emotion…</div>",
                    unsafe_allow_html=True,
                )

                plain      = _strip_basic_markdown(bot_response)
                lines      = plain.splitlines() if plain else []
                typed_text = ""
                thinking.empty()

                for line in lines:
                    typed_text += line + "\n"
                    placeholder.markdown(
                        f"<div style='white-space:pre-wrap;color:#E8E8F0'>"
                        f"{escape(typed_text)}</div>",
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.03)

                placeholder.markdown(
                    _render_assistant_bubble_html(
                        bot_response=bot_response,
                        detected_emotion=emotion,
                        detected_risk=risk_level,
                    ),
                    unsafe_allow_html=True,
                )

                if risk_level == "HIGH":
                    st.error(
                        "🆘 High-risk language detected. "
                        "If you feel unsafe: iCall **9152987821** · US **988**.",
                        icon="🆘",
                    )
                elif risk_level == "MEDIUM":
                    st.warning(
                        "💛 You seem to be going through a tough time. "
                        "Reach out to someone you trust.",
                        icon="💛",
                    )

            # ── Persist to DB ─────────────────────────────────────────────
            now = datetime.now().isoformat()
            save_message(
                session_id=st.session_state.session_id,
                role="user",
                message=clean_input,
                user_email=st.session_state.user_email,
                emotion=emotion,
                emotion_score=emotion_score,
                risk_level=risk_level,
            )
            save_message(
                session_id=st.session_state.session_id,
                role="assistant",
                message=bot_response,
                user_email=st.session_state.user_email,
            )

            # ── Update session state ──────────────────────────────────────
            st.session_state.messages.append({
                "role": "user", "content": clean_input,
                "emotion": emotion, "risk_level": risk_level, "timestamp": now,
            })
            st.session_state.messages.append({
                "role": "assistant", "content": bot_response,
                "emotion": emotion, "risk_level": risk_level, "timestamp": now,
            })
            st.session_state.emotion_records.append({
                "timestamp": now, "emotion": emotion,
                "emotion_score": emotion_score, "risk_level": risk_level,
            })

            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD VIEW
# ════════════════════════════════════════════════════════════════════════════

if st.session_state.active_view == "dashboard":
    st.markdown("### 📊 Your Emotional Insights")
    st.markdown(
        "<p style='color:#888;font-size:0.9rem;'>Understanding your emotional "
        "patterns can be the first step toward better mental wellbeing.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    emotion_records = st.session_state.emotion_records

    # ── KPI metrics ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    total_msgs    = len(emotion_records)
    high_risk_cnt = sum(1 for r in emotion_records if r.get("risk_level") == "HIGH")
    med_risk_cnt  = sum(1 for r in emotion_records if r.get("risk_level") == "MEDIUM")
    dominant_emotion = (
        max(
            set(r["emotion"] for r in emotion_records),
            key=lambda e: sum(1 for r in emotion_records if r["emotion"] == e),
        ) if emotion_records else "—"
    )

    col1.metric("💬 Messages",         total_msgs)
    col2.metric("🎭 Dominant Emotion", dominant_emotion.capitalize() if dominant_emotion != "—" else "—")
    col3.metric("🟡 Medium Risk",       med_risk_cnt)
    col4.metric("🔴 High Risk",         high_risk_cnt,
                delta="⚠️ Review" if high_risk_cnt > 0 else None,
                delta_color="inverse")
    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### 🎭 Emotion Breakdown")
        st.pyplot(emotion_distribution_pie(emotion_records), use_container_width=True)
    with c2:
        st.markdown("#### 🚦 Risk Level Distribution")
        st.pyplot(risk_level_bar(emotion_records), use_container_width=True)

    st.markdown("#### 📈 Emotion Trend (Last 7 Days)")
    db_records  = get_emotion_history(st.session_state.session_id, days=7)
    all_records = db_records if db_records else emotion_records
    st.pyplot(emotion_trend_line(all_records, days=7), use_container_width=True)

    st.markdown("#### ⚡ Session Intensity Timeline")
    st.pyplot(session_intensity_line(emotion_records), use_container_width=True)

    st.divider()
    st.markdown("#### 🧾 Session Summary")

    if emotion_records:
        summary = get_current_session_summary(emotion_records)
        st.info(summary["summary_text"])
        s1, s2, s3 = st.columns(3)
        s1.metric("Total Messages",    summary["total_messages"])
        s2.metric("Dominant Emotion",  summary["dominant_emotion"].capitalize())
        s3.metric("High-Risk Messages", summary["high_risk_count"])

        with st.expander("🔍 View Raw Session Data"):
            import pandas as pd
            st.dataframe(pd.DataFrame(emotion_records), use_container_width=True)
    else:
        st.info(
            "💬 No data yet! Head to the **Chat** tab and start a conversation. "
            "Your emotional insights will appear here as you chat.",
        )


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;color:#555;font-size:0.75rem;padding:20px 0 10px;'>
  🧠 MindGuard v2.0 &nbsp;|&nbsp; Built with ❤️ for mental health awareness
  &nbsp;|&nbsp; Not a substitute for professional therapy.
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# CHANGE LOG  (v1 → v2)
# ════════════════════════════════════════════════════════════════════════════
# [1] COLOUR IMPROVEMENT
#     - Replaced flat dark backgrounds with layered gradients
#     - User bubble: purple→blue gradient; bot bubble: glass/blur card
#     - Both dark and light themes defined; switched via sidebar toggle
#
# [2] AUTH SYSTEM FIX
#     - Removed ALL Google / GitHub / Apple OAuth (dead code eliminated)
#     - New backend/auth_db.py: register_user / verify_user / reset_password
#       backed by SQLite 'users' table — persists across refreshes & devices
#     - Passwords hashed via SHA-256 + email salt (no extra deps)
#
# [3] ANIMATIONS
#     - CSS @keyframe fadeSlideIn: each chat bubble fades + slides up
#     - CSS @keyframe pageEntry: main container slides in on load
#     - Auth form: .auth-form-anim class adds entry animation
#
# [4] PERSISTENT CHAT HISTORY
#     - Sidebar shows last 12 sessions for the logged-in user
#     - Clicking a session loads its full message + emotion history from DB
#     - "New Session" button creates a fresh session and updates sidebar
#
# [5] REPLACED T&C
#     - Removed "I agree to Terms" checkbox
#     - Added calming "What MindGuard provides" info panel on all auth pages
#
# [6] DARK / LIGHT MODE
#     - Sidebar toggle (☀️ Light mode) stored in session_state.theme
#     - Toggling triggers st.rerun() which re-injects correct CSS block
#     - No JS; fully CSS-variable-free approach avoids Streamlit quirks
#
# [7] HINGLISH / MULTI-LANGUAGE
#     - Input placeholder explicitly says "English or Hinglish welcome"
#     - Mixed input is NOT blocked or pre-processed; passes raw to NLP
#     - HuggingFace DistilRoBERTa handles code-switched text reasonably
#
# [8] RESPONSE QUALITY
#     - Collect last 10 emotion records → recent_emotions (was 5)
#     - Collect last 5 user messages text → recent_messages (new context window)
#     - Both passed downstream (recent_emotions goes to generate_response)
#     - To fully leverage context, wire recent_messages into response_engine.py
#
# [9] FORGOT PASSWORD
#     - New 'forgot' page with email + new password + confirm fields
#     - Calls reset_password() from auth_db.py (direct reset, no email needed)
#     - Accessible via "Forgot password?" link on login page
#
# [10] CONTEXT-AWARE RESPONSES
#     - recent_emotions window extended to last 10 records
#     - recent_messages list built from last 5 user messages in session
#     - Response engine receives full emotional trend for context
# ════════════════════════════════════════════════════════════════════════════