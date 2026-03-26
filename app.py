"""
app.py
======
Main Streamlit application for the Mental Health Chatbot.
Provides a two-tab interface:
    Tab 1 — 💬 Chat      : Real-time conversation with the AI chatbot
    Tab 2 — 📊 Dashboard : Emotion trends, distribution charts, session summary

Architecture:
    This app calls the FastAPI backend (running separately on port 8000).
    If the backend is unreachable, it falls back to direct module imports
    so the demo still works with just `streamlit run app.py`.

Run:
    Terminal 1: uvicorn backend.api:app --reload --port 8000
    Terminal 2: streamlit run app.py
"""

import streamlit as st
import json
import time
import random
import re
import os
import uuid
import base64
import hashlib
import secrets
from html import escape
from datetime import datetime
from urllib.parse import urlencode, parse_qs
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── OAuth Configuration ──────────────────────────────────────────────────────
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
GITHUB_CLIENT_ID = os.getenv('GITHUB_CLIENT_ID')
GITHUB_CLIENT_SECRET = os.getenv('GITHUB_CLIENT_SECRET')
OAUTH_REDIRECT_URI = os.getenv('OAUTH_REDIRECT_URI', 'http://localhost:8501/oauth/callback')

# ── OAuth Helper Functions ───────────────────────────────────────────────────
def generate_code_verifier():
    """Generate a code verifier for PKCE."""
    return secrets.token_urlsafe(32)

def generate_code_challenge(verifier):
    """Generate a code challenge from verifier."""
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip('=')

def get_google_auth_url():
    """Generate Google OAuth authorization URL."""
    state = secrets.token_urlsafe(32)
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)
    
    # Store state and verifier in session
    st.session_state.oauth_state = state
    st.session_state.code_verifier = code_verifier
    st.session_state.oauth_provider = 'google'
    
    params = {
        'client_id': GOOGLE_CLIENT_ID,
        'redirect_uri': OAUTH_REDIRECT_URI,
        'scope': 'openid email profile',
        'response_type': 'code',
        'state': state,
        'code_challenge': code_challenge,
        'code_challenge_method': 'S256',
        'access_type': 'offline',
        'prompt': 'consent'
    }
    
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"

def get_github_auth_url():
    """Generate GitHub OAuth authorization URL."""
    state = secrets.token_urlsafe(32)
    
    # Store state in session
    st.session_state.oauth_state = state
    st.session_state.oauth_provider = 'github'
    
    params = {
        'client_id': GITHUB_CLIENT_ID,
        'redirect_uri': OAUTH_REDIRECT_URI,
        'scope': 'user:email',
        'state': state,
        'allow_signup': 'true'
    }
    
    return f"https://github.com/login/oauth/authorize?{urlencode(params)}"

def exchange_code_for_token(provider, code, code_verifier=None):
    """Exchange authorization code for access token."""
    if provider == 'google':
        token_url = 'https://oauth2.googleapis.com/token'
        data = {
            'client_id': GOOGLE_CLIENT_ID,
            'client_secret': GOOGLE_CLIENT_SECRET,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': OAUTH_REDIRECT_URI,
            'code_verifier': code_verifier
        }
    elif provider == 'github':
        token_url = 'https://github.com/login/oauth/access_token'
        data = {
            'client_id': GITHUB_CLIENT_ID,
            'client_secret': GITHUB_CLIENT_SECRET,
            'code': code,
            'redirect_uri': OAUTH_REDIRECT_URI
        }
    else:
        return None
    
    headers = {'Accept': 'application/json'}
    response = requests.post(token_url, data=data, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    return None

def get_user_info(provider, access_token):
    """Get user information from OAuth provider."""
    if provider == 'google':
        # Get user info from Google
        headers = {'Authorization': f'Bearer {access_token}'}
        response = requests.get('https://www.googleapis.com/oauth2/v2/userinfo', headers=headers)
        
        if response.status_code == 200:
            user_data = response.json()
            return {
                'email': user_data.get('email'),
                'name': user_data.get('name'),
                'provider': 'google',
                'provider_id': user_data.get('id')
            }
    
    elif provider == 'github':
        # Get user info from GitHub
        headers = {'Authorization': f'token {access_token}'}
        
        # Get basic user info
        user_response = requests.get('https://api.github.com/user', headers=headers)
        if user_response.status_code != 200:
            return None
            
        user_data = user_response.json()
        
        # Get user email (GitHub may have private emails)
        email_response = requests.get('https://api.github.com/user/emails', headers=headers)
        email = user_data.get('email')
        
        if email_response.status_code == 200:
            emails = email_response.json()
            # Prefer primary, verified email
            for email_info in emails:
                if email_info.get('primary') and email_info.get('verified'):
                    email = email_info.get('email')
                    break
        
        return {
            'email': email,
            'name': user_data.get('name') or user_data.get('login'),
            'provider': 'github',
            'provider_id': str(user_data.get('id'))
        }
    
    return None



def handle_oauth_callback():
    """Handle OAuth callback from URL parameters."""
    query_params = st.query_params
    
    if 'code' not in query_params or 'state' not in query_params:
        return False, "Missing OAuth parameters"
    
    code = query_params['code']
    state = query_params['state']
    
    # Verify state to prevent CSRF
    if state != st.session_state.get('oauth_state'):
        return False, "Invalid OAuth state"
    
    provider = st.session_state.get('oauth_provider')
    if not provider:
        return False, "No OAuth provider specified"
    
    # Exchange code for token
    code_verifier = st.session_state.get('code_verifier') if provider == 'google' else None
    token_data = exchange_code_for_token(provider, code, code_verifier)
    
    if not token_data or 'access_token' not in token_data:
        return False, "Failed to exchange code for token"
    
    # Get user info
    user_info = get_user_info(provider, token_data['access_token'])
    
    if not user_info or not user_info.get('email'):
        return False, "Failed to get user information"
    
    # Create or update user account
    email = user_info['email']
    name = user_info['name'] or email.split('@')[0]
    
    # Store OAuth user in the users dict
    if 'oauth_users' not in st.session_state:
        st.session_state.oauth_users = {}
    
    st.session_state.oauth_users[email] = {
        'name': name,
        'provider': provider,
        'provider_id': user_info['provider_id']
    }
    
    # Log the user in
    st.session_state.logged_in = True
    st.session_state.user_email = email
    st.session_state.user_name = name.title()
    
    # Get or create a session for this user
    latest_session = get_latest_user_session(email)
    if latest_session:
        # Reuse latest session and load its history
        st.session_state.session_id = latest_session
        st.session_state.messages = get_session_history(latest_session, email)
        st.session_state.emotion_records = get_emotion_history(latest_session, email)
    else:
        # Create new session for this user
        st.session_state.session_id = generate_session_id()
        upsert_session(st.session_state.session_id, email)
        st.session_state.messages = []
        st.session_state.emotion_records = []
    
    # Clear OAuth state
    for key in ['oauth_state', 'code_verifier', 'oauth_provider']:
        if key in st.session_state:
            del st.session_state[key]
    
    # Clear URL parameters
    st.query_params.clear()
    
    return True, f"Successfully logged in with {provider.title()}!"

# ── Internal imports (fallback mode) ─────────────────────────────────────────
# Import function to get latest user session
from backend.database        import init_db, upsert_session, save_message, \
                                    get_session_history, get_emotion_history, \
                                    get_user_sessions, get_latest_user_session
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
    generate_session_id, sanitize_input, risk_badge,
    emotion_badge, init_session_state
)

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"
USE_API      = False   # Set True if FastAPI is running separately

# ── Page config (must be first Streamlit command) ─────────────────────────────
st.set_page_config(
    page_title="MindEase — Mental Health Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* ── Chat bubbles ── */
.user-bubble {
    background: linear-gradient(135deg, #6C63FF, #8B85FF);
    color: white;
    padding: 10px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0;
    max-width: 75%;
    margin-left: auto;
    text-align: right;
    font-size: 0.95rem;
    box-shadow: 0 2px 8px rgba(108,99,255,0.3);
}
.bot-bubble {
    background: #1E1E35;
    color: #E8E8F0;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 0;
    max-width: 80%;
    font-size: 0.93rem;
    border-left: 3px solid #6C63FF;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.meta-chip {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    margin: 2px 3px 6px 0;
    font-weight: 600;
}
.chip-emotion { background: #2A2A45; color: #A9A4FF; border: 1px solid #4A44AA; }
.chip-low    { background: #1A3028; color: #68D391; border: 1px solid #2D6A4F; }
.chip-medium { background: #3A3010; color: #F6E05E; border: 1px solid #856A00; }
.chip-high   { background: #3A1010; color: #FC8181; border: 1px solid #8B2020; }

/* ── Metric cards ── */
.metric-card {
    background: #1A1A2E;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #2A2A45;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #6C63FF; }
.metric-label { font-size: 0.8rem; color: #888; margin-top: 4px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #12122A !important;
}

/* ── Chat input ── */
.stChatInput input {
    background: #1E1E35 !important;
    color: #E8E8F0 !important;
    border-color: #6C63FF !important;
}

/* ── Scrollable chat container ── */
.chat-container { max-height: 520px; overflow-y: auto; padding-right: 8px; }

/* ── Chat layout (avatars + alignment) ── */
.chat-row {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    margin: 8px 0;
}
.chat-row.user { justify-content: flex-end; }
.chat-row.bot  { justify-content: flex-start; }

.bubble-wrap { display: flex; flex-direction: column; }

.avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex: 0 0 34px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
}
.avatar.user { background: rgba(108,99,255,0.22); border: 1px solid rgba(108,99,255,0.45); }
.avatar.bot  { background: rgba(108,99,255,0.10); border: 1px solid rgba(108,99,255,0.25); }

.meta-row {
    margin-top: 6px;
    display: flex;
    gap: 8px;
    align-items: center;
}
.meta-row.left  { justify-content: flex-start; }
.meta-row.right { justify-content: flex-end; }

.detected-line {
    font-size: 0.8rem;
    color: #A9A4FF;
    margin-top: 6px;
}

/* Detected emotion badge inside assistant bubble */
.detected-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    border: 1px solid rgba(167,139,250,0.25);
    background: rgba(167,139,250,0.10);
    color: #A9A4FF;
    margin-bottom: 10px;
}

/* Assistant message chunk styling */
.chunk-box {
    white-space: pre-wrap;
    line-height: 1.45;
    margin-top: 6px;
}
.chunk-empathy {
    color: rgba(233, 229, 255, 0.92);
    font-style: italic;
}
.chunk-support {
    margin-top: 12px;
    padding: 10px 12px;
    border-radius: 12px;
    background: rgba(167,139,250,0.12);
    border: 1px solid rgba(167,139,250,0.22);
}
.chunk-question {
    margin-top: 12px;
    font-weight: 650;
}

/* Assistant variation styles */
.assistant-variant-normal .bot-bubble { border-left: 0px solid transparent; }
.assistant-variant-quote .bot-bubble  { border-left: 4px solid #a78bfa; }
.assistant-variant-indent .bot-bubble { margin-left: 10px; }

/* ── Typography ── */
h1 { font-size: 2.5rem; color: #a78bfa; margin: 0 0 0.25rem 0; }
</style>
""", unsafe_allow_html=True)

 
# ════════════════════════════════════════════════════════════════════════════
# UI helpers (frontend-only; does not touch backend/NLP/DB logic)
# ════════════════════════════════════════════════════════════════════════════
def _strip_basic_markdown(text: str) -> str:
    """
    Best-effort cleanup so we can safely place model output inside HTML blocks
    without rendering markdown markers (** * `).
    """
    if not text:
        return ""
    text = text.replace("```", "")
    text = text.replace("**", "")
    text = text.replace("*", "")
    text = text.replace("`", "")
    return text.strip()


def _split_response_into_parts(bot_response: str) -> tuple[str, str, str]:
    """
    Split assistant response into 2–3 logical chunks:
      1) emotional acknowledgement
      2) suggestion/support
      3) follow-up question (or last paragraph)
    """
    plain = _strip_basic_markdown(bot_response)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", plain) if p.strip()]
    if not paragraphs:
        return "", "", ""
    if len(paragraphs) == 1:
        return paragraphs[0], "", ""

    empathy = paragraphs[0]
    question = paragraphs[-1]
    support = "\n\n".join(paragraphs[1:-1]).strip()
    if not support and len(paragraphs) >= 2:
        support = paragraphs[1].strip()
    return empathy, support, question


def _risk_label_word(risk_level: str) -> str:
    return {"LOW": "Low", "MEDIUM": "Medium", "HIGH": "High"}.get(risk_level, risk_level)


def _render_user_bubble_html(message: str, emotion: str, risk_level: str) -> str:
    msg_escaped = escape(message or "")
    return f"""
    <div class="chat-row user">
      <div class="bubble-wrap">
        <div class="user-bubble">{msg_escaped}</div>
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
    variant: int,
) -> str:
    empathy, support, question = _split_response_into_parts(bot_response)

    empathy_e = escape(empathy or "")
    support_e = escape(support or "")
    question_e = escape(question or "")

    emotion_label = emotion_badge(detected_emotion)
    risk_word = _risk_label_word(detected_risk)
    detected_badge = f"[ {emotion_label} | {risk_word} Risk ]"

    variant_class = {
        0: "assistant-variant-normal",
        1: "assistant-variant-quote",
        2: "assistant-variant-indent",
    }.get(variant, "assistant-variant-normal")

    empathy_block = (
        f"<div class=\"chunk-box chunk-empathy\"><em>{empathy_e}</em></div>"
        if empathy_e else ""
    )
    support_block = (
        f"<div class=\"chunk-box chunk-support\">💡 {support_e}</div>"
        if support_e else ""
    )
    question_block = (
        f"<div class=\"chunk-box chunk-question\"><strong>{question_e}</strong></div>"
        if question_e else ""
    )

    return f"""
    <div class="chat-row bot">
      <div class="avatar bot" aria-hidden="true">🤖</div>
      <div class="bubble-wrap {variant_class}">
        <div class="bot-bubble">
          <div class="detected-badge">{escape(detected_badge)}</div>
          {empathy_block}
          {support_block}
          {question_block}
        </div>
      </div>
    </div>
    """

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

    # ════════════════════════════════════════════════════════════════════════════
# Session State Initialization
# ════════════════════════════════════════════════════════════════════════════

init_session_state({
    "logged_in":        False,
    "user_name":        "",
    "user_email":       "",
    "session_id":       None,
    "messages":         [],      # {role, content, emotion, risk_level, timestamp}
    "emotion_records":  [],      # Emotion data for charts
    "model_loaded":     False,
    "is_typing":        False,
    "quick_action_message": "",
    "active_tab":       "chat",
})

# ── Initialize DB and load NLP model on first run ─────────────────────────────
if not st.session_state.model_loaded:
    with st.spinner("🧠 Loading AI model... (first load may take ~30 seconds)"):
        init_db()
        load_model()
    st.session_state.model_loaded = True

# 🔐 LOGIN GATE
# ════════════════════════════════════════════════════════════════════════════
# AUTH SYSTEM (Login + Signup)
# ════════════════════════════════════════════════════════════════════════════

if "page" not in st.session_state:
    st.session_state.page = "signup"

if "users" not in st.session_state:
    st.session_state.users = {"demo@example.com": "password123"}

def render_auth_page():
    # Handle OAuth callback first
    if 'code' in st.query_params and 'state' in st.query_params:
        success, message = handle_oauth_callback()
        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(f"OAuth authentication failed: {message}")
        return
    
    # 1. Inject Custom CSS to match the image's dark purple aesthetic
    st.markdown("""
    <style>
    /* Hide the sidebar and top header during authentication */
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }
    header { display: none !important; }
    
    /* Global background matching the outer dark grey/purple area */
    .stApp {
        background-color: #353542 !important;
    }
    
    /* Center the main container to act like the card bounds */
    [data-testid="block-container"] {
        max-width: 1050px;
        padding-top: 5vh;
        padding-bottom: 5vh;
    }

    /* Style text inputs to match the dark form fields */
    div[data-testid="stTextInput"] input {
        background-color: #403D58 !important;
        border: 1px solid #403D58 !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        padding: 14px 16px !important;
        font-size: 0.95rem !important;
        transition: border 0.2s ease;
    }
    div[data-testid="stTextInput"] input:focus {
        border: 1px solid #7E57C2 !important;
        box-shadow: none !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #9A98AD !important;
    }

    /* Primary Button (Create Account / Log in) */
    button[kind="primary"] {
        background-color: #7A5AF8 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 12px !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        transition: background-color 0.2s;
    }
    button[kind="primary"]:hover {
        background-color: #6941C6 !important;
    }

    /* Secondary Buttons (Google / GitHub / Apple) */
    button[kind="secondary"] {
        background-color: transparent !important;
        color: #FFFFFF !important;
        border: 1px solid #5A5872 !important;
        border-radius: 6px !important;
        font-weight: 400 !important;
        padding: 12px !important;
        font-size: 0.9rem !important;
    }
    button[kind="secondary"]:hover {
        border-color: #7A5AF8 !important;
        color: #7A5AF8 !important;
    }

    /* Checkbox text */
    .stCheckbox span {
        color: #D1D0DC !important;
        font-size: 0.85rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 2. Structure the layout with two columns exactly like the image
    # The gap="large" helps separate the image card from the form
    col_img, col_form = st.columns([1.1, 1], gap="large")

    # -- LEFT COLUMN: Image & Branding --
    with col_img:
        # We remove indentation here so Streamlit doesn't render it as a markdown code block
        st.markdown("""
<div style="
background-image: url('https://images.unsplash.com/photo-1542272201-b1ca555f8505?q=80&w=1000&auto=format&fit=crop');
background-size: cover;
background-position: center;
border-radius: 12px;
height: 650px;
position: relative;
padding: 30px;
color: white;
box-shadow: inset 0 0 0 2000px rgba(53, 40, 90, 0.45);
">
<div style="display: flex; justify-content: space-between; align-items: center;">
<h3 style="margin:0; font-family: sans-serif; letter-spacing: 1px; font-weight: 600;">MINDEASE</h3>
<span style="font-size: 0.85rem; background: rgba(255,255,255,0.15); padding: 6px 14px; border-radius: 20px; cursor: pointer;">Back to website →</span>
</div>

<div style="position: absolute; bottom: 40px; left: 30px;">
<h1 style="margin:0; font-weight: 400; font-size: 2.2rem; line-height: 1.2;">Finding Peace,<br>Building Resilience</h1>

<div style="display: flex; gap: 6px; margin-top: 25px;">
<div style="width: 20px; height: 3px; background: white; border-radius: 2px;"></div>
<div style="width: 20px; height: 3px; background: rgba(255,255,255,0.3); border-radius: 2px;"></div>
<div style="width: 20px; height: 3px; background: rgba(255,255,255,0.3); border-radius: 2px;"></div>
</div>
</div>
</div>
        """, unsafe_allow_html=True)

    # -- RIGHT COLUMN: Authentication Form --
    with col_form:
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True) # Vertical centering buffer
        
        if st.session_state.page == "signup":
            # --- SIGNUP STATE ---
            st.markdown("<h1 style='color: white; font-size: 2.4rem; margin-bottom: 5px;'>Create an account</h1>", unsafe_allow_html=True)
            
            # Switch to Login
            st.markdown("<span style='color: #9A98AD; font-size: 0.9rem;'>Already have an account? </span>", unsafe_allow_html=True)
            if st.button("Log in", key="switch_to_login", use_container_width=False):
                st.session_state.page = "login"
                st.rerun()
            
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

            # Name inputs side-by-side
            c_fn, c_ln = st.columns(2)
            with c_fn:
                name = st.text_input("First Name", placeholder="First name", label_visibility="collapsed")
            with c_ln:
                last_name = st.text_input("Last Name", placeholder="Last name", label_visibility="collapsed")
            
            email = st.text_input("Email", placeholder="Email", label_visibility="collapsed")
            password = st.text_input("Password", placeholder="Enter your password", type="password", label_visibility="collapsed")

            st.checkbox("I agree to the Terms & Conditions")

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button("Create account", type="primary", use_container_width=True):
                if email and password and name:
                    if email not in st.session_state.users:
                        st.session_state.users[email] = password
                        st.session_state.page = "login"
                        st.rerun()
                    else:
                        st.error("Email already exists")
                else:
                    st.warning("Please fill all required fields")

            # Divider
            st.markdown("""
            <div style='display: flex; align-items: center; text-align: center; color: #7B7894; margin: 25px 0 15px;'>
                <div style='flex: 1; height: 1px; background: #5A5872;'></div>
                <span style='padding: 0 15px; font-size: 0.8rem;'>Or register with</span>
                <div style='flex: 1; height: 1px; background: #5A5872;'></div>
            </div>
            """, unsafe_allow_html=True)

            # Social Buttons
            col_g, col_gh, col_a = st.columns(3)
            with col_g:
                if st.button("🔴 Google", use_container_width=True, key="su_g"):
                    if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
                        auth_url = get_google_auth_url()
                        st.markdown(f'<meta http-equiv="refresh" content="0; url={auth_url}">', unsafe_allow_html=True)
                    else:
                        st.error("Google OAuth not configured. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env")
            with col_gh:
                if st.button("🐙 GitHub", use_container_width=True, key="su_gh"):
                    if GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET:
                        auth_url = get_github_auth_url()
                        st.markdown(f'<meta http-equiv="refresh" content="0; url={auth_url}">', unsafe_allow_html=True)
                    else:
                        st.error("GitHub OAuth not configured. Please set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET in .env")
            with col_a:
                st.button("🍎 Apple", use_container_width=True, key="su_a")

        else:
            # --- LOGIN STATE ---
            st.markdown("<h1 style='color: white; font-size: 2.4rem; margin-bottom: 5px;'>Welcome back</h1>", unsafe_allow_html=True)
            
            # Switch to Signup
            st.markdown("<span style='color: #9A98AD; font-size: 0.9rem;'>Don't have an account? </span>", unsafe_allow_html=True)
            if st.button("Sign up", key="switch_to_signup", use_container_width=False):
                st.session_state.page = "signup"
                st.rerun()

            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

            email = st.text_input("Email", placeholder="Email", label_visibility="collapsed")
            password = st.text_input("Password", placeholder="Enter your password", type="password", label_visibility="collapsed")

            st.checkbox("Remember me")

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            
            if st.button("Log in", type="primary", use_container_width=True):
                if email in st.session_state.users and st.session_state.users[email] == password:
                    st.session_state.logged_in = True
                    st.session_state.user_email = email  # Store the email for data isolation
                    st.session_state.user_name = email.split('@')[0].title()
                    
                    # Get or create a session for this user
                    latest_session = get_latest_user_session(email)
                    if latest_session:
                        # Reuse latest session and load its history
                        st.session_state.session_id = latest_session
                        st.session_state.messages = get_session_history(latest_session, email)
                        st.session_state.emotion_records = get_emotion_history(latest_session, email)
                    else:
                        # Create new session for this user
                        st.session_state.session_id = generate_session_id()
                        upsert_session(st.session_state.session_id, email)
                        st.session_state.messages = []
                        st.session_state.emotion_records = []
                    
                    st.session_state.page = "main"
                    st.rerun()
                else:
                    st.error("Invalid credentials (Demo: demo@example.com / password123)")

            # Divider
            st.markdown("""
            <div style='display: flex; align-items: center; text-align: center; color: #7B7894; margin: 25px 0 15px;'>
                <div style='flex: 1; height: 1px; background: #5A5872;'></div>
                <span style='padding: 0 15px; font-size: 0.8rem;'>Or log in with</span>
                <div style='flex: 1; height: 1px; background: #5A5872;'></div>
            </div>
            """, unsafe_allow_html=True)

            # Social Buttons
            col_g, col_gh, col_a = st.columns(3)
            with col_g:
                if st.button("🔴 Google", use_container_width=True, key="li_g"):
                    if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
                        auth_url = get_google_auth_url()
                        st.markdown(f'<meta http-equiv="refresh" content="0; url={auth_url}">', unsafe_allow_html=True)
                    else:
                        st.error("Google OAuth not configured. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env")
            with col_gh:
                if st.button("🐙 GitHub", use_container_width=True, key="li_gh"):
                    if GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET:
                        auth_url = get_github_auth_url()
                        st.markdown(f'<meta http-equiv="refresh" content="0; url={auth_url}">', unsafe_allow_html=True)
                    else:
                        st.error("GitHub OAuth not configured. Please set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET in .env")
            with col_a:
                st.button("🍎 Apple", use_container_width=True, key="li_a")
            
            # Guest bypass for easy testing
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button("👋 Continue as Guest", use_container_width=True):
                st.session_state.logged_in = True
                guest_email = f"guest_{random.randint(100000, 999999)}"
                st.session_state.user_email = guest_email
                st.session_state.user_name = f"Guest"
                st.session_state.session_id = generate_session_id()
                upsert_session(st.session_state.session_id, guest_email)
                st.session_state.messages = []
                st.session_state.emotion_records = []
                st.session_state.page = "main"
                st.rerun()


# 🔐 LOGIN GATE
if not st.session_state.logged_in:
    render_auth_page()
    st.stop()

# (The rest of your sidebar and chat tabs go exactly here...)

# ════════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════════
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_name = ""
        st.session_state.page = "login"
        st.rerun()

with st.sidebar:
    st.title(f"🤖 {st.session_state.user_name}")
    st.caption("Your AI-powered emotional support companion")
    st.divider()

    # ── Emotion summary ──────────────────────────────────────────────────
    st.subheader("🧭 Emotion Summary")
    if st.session_state.emotion_records:
        summary = get_current_session_summary(st.session_state.emotion_records)
        today_mood = summary["dominant_emotion"]

        # Trend: compare last 5 vs previous 5 using mapped risk intensity
        risk_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        last_chunk = st.session_state.emotion_records[-5:]
        prev_chunk = st.session_state.emotion_records[-10:-5] if len(st.session_state.emotion_records) >= 10 else []

        last_avg = sum(risk_map.get(r.get("risk_level", "LOW"), 1) for r in last_chunk) / max(1, len(last_chunk))
        prev_avg = sum(risk_map.get(r.get("risk_level", "LOW"), 1) for r in prev_chunk) / max(1, len(prev_chunk)) if prev_chunk else last_avg

        if last_avg > prev_avg + 0.01:
            trend = "↑ Increasing"
        elif last_avg < prev_avg - 0.01:
            trend = "↓ Decreasing"
        else:
            trend = "→ Stable"

        st.markdown(f"**Today’s Mood:** {emotion_badge(today_mood)}")
        st.markdown(f"**Trend:** {trend}")
    else:
        st.info("No mood data yet. Start chatting to unlock your summary.")

    st.divider()

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_name = ""
        st.session_state.user_email = ""
        _set_page_query('login')
        st.rerun()

    # ── Quick actions ─────────────────────────────────────────────────────
    st.subheader("⚡ Quick Actions")
    if st.button("🧘 Calm me down", use_container_width=True):
        st.session_state.quick_action_message = (
            "I'm feeling overwhelmed. Calm me down with a few grounding steps and slow breathing guidance."
        )
        st.rerun()

    if st.button("😴 Help me relax", use_container_width=True):
        st.session_state.quick_action_message = (
            "Help me relax right now. Give me a short calming routine I can do in the next 2 minutes."
        )
        st.rerun()

    if st.button("📊 Show my trends", use_container_width=True):
        st.session_state.quick_action_message = (
            "Show me my mental health trends from this session and suggest what I should focus on next."
        )
        st.rerun()

    st.divider()

    # ── Emergency section ────────────────────────────────────────────────
    st.subheader("🚨 Need urgent help?")
    st.markdown("📞 **Call:** `9152987821`")
    st.markdown("If in the US/Canada, you can also call **988**.")

    st.markdown(
        "<div style='font-size:0.75rem; color:#666; margin-top:10px'>"
        "⚠️ This is an AI support tool, not a replacement for therapy."
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Session meta (kept small at the bottom)
    st.caption("Session")
    if st.session_state.session_id:
        st.code(str(st.session_state.session_id)[:18] + "...", language=None)
    else:
        st.code("No active session", language=None)

    if st.button("🔄 New Session", use_container_width=True):
        # Create a new session for the current user
        st.session_state.session_id = generate_session_id()
        upsert_session(st.session_state.session_id, st.session_state.user_email)
        st.session_state.messages = []
        st.session_state.emotion_records = []
        st.session_state.quick_action_message = ""
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# Main content — Tabs
# ════════════════════════════════════════════════════════════════════════════

# ── Navigation Header ────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("← Back to Login", key="nav_back"):
        st.session_state.logged_in = False
        st.session_state.user_name = ""
        st.session_state.user_email = ""
        st.session_state.page = "login"
        st.rerun()

with col2:
    st.markdown("<h2 style='text-align: center; margin: 0;'>🧠 MindEase Dashboard</h2>", unsafe_allow_html=True)

with col3:
    if st.button("🚪 Logout", key="nav_logout"):
        st.session_state.logged_in = False
        st.session_state.user_name = ""
        st.session_state.user_email = ""
        st.session_state.page = "login"
tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Insights Dashboard"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════════════════

with tab_chat:
    # ── Chat Navigation ───────────────────────────────────────────────────
    chat_nav_col1, chat_nav_col2 = st.columns([3, 1])
    with chat_nav_col1:
        st.markdown("<h1>Hi 👋, how are you feeling today?</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#888; font-size:1.0rem'>"
            "You're in a safe space. Share anything on your mind 💙"
            "</p>",
            unsafe_allow_html=True,
        )
    with chat_nav_col2:
        if st.button("📊 View Dashboard", key="chat_to_dashboard"):
            st.session_state.active_tab = "dashboard"
            st.rerun()

    st.markdown("---")

    # ── Chat history display (GPT-style) ──────────────────────────────────────
    last_assistant_index = max(
        (i for i, m in enumerate(st.session_state.messages) if m.get("role") == "assistant"),
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
            detected_emotion = msg.get("emotion", "neutral")
            detected_risk = msg.get("risk_level", "LOW")

            # Stable pseudo-random variant (less repetitive, but consistent per message)
            variant = random.Random(idx).randrange(3)

            with st.chat_message("assistant"):
                st.markdown(
                    _render_assistant_bubble_html(
                        bot_response=msg.get("content", ""),
                        detected_emotion=detected_emotion,
                        detected_risk=detected_risk,
                        variant=variant,
                    ),
                    unsafe_allow_html=True,
                )

                # Micro-interactions for the latest assistant response
                if idx == last_assistant_index:
                    b_cols = st.columns(3)
                    if b_cols[0].button("👍 Helpful", key=f"helpful_{idx}"):
                        st.session_state["last_feedback"] = {"message_index": idx, "value": "helpful"}
                    if b_cols[1].button("💬 Continue", key=f"continue_{idx}"):
                        st.session_state.quick_action_message = (
                            "Continue from your last response and ask me one follow-up question to help me reflect."
                        )
                        st.rerun()
                    if b_cols[2].button("🔁 Another suggestion", key=f"another_{idx}"):
                        st.session_state.quick_action_message = (
                            "Give me another supportive suggestion based on my detected emotion, and end with a question."
                        )
                        st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Type how you're feeling... (e.g. 'I've been really anxious lately')"
    )

    quick_msg = st.session_state.get("quick_action_message", "")
    effective_input = quick_msg if quick_msg else user_input
    if quick_msg:
        st.session_state.quick_action_message = ""

    if effective_input:
        clean_input = sanitize_input(effective_input)
        if not clean_input:
            st.warning("Please enter a message before sending.")
        else:
            # Run NLP pipeline (direct module call) ─ backend logic unchanged
            nlp_result = detect_emotion(clean_input)
            emotion = nlp_result["primary_emotion"]
            emotion_score = nlp_result["primary_score"]

            risk_level, risk_explanation = classify_risk(clean_input, emotion, emotion_score)

            recent_emotions = [r["emotion"] for r in st.session_state.emotion_records[-5:]]
            bot_response = generate_response(
                user_message=clean_input,
                emotion=emotion,
                emotion_score=emotion_score,
                risk_level=risk_level,
                recent_emotions=recent_emotions,
            )

            # ── Show user message immediately ────────────────────────────────
            with st.chat_message("user"):
                st.markdown(
                    _render_user_bubble_html(
                        message=clean_input,
                        emotion=emotion,
                        risk_level=risk_level,
                    ),
                    unsafe_allow_html=True,
                )

            # ── Show assistant with typing effect ────────────────────────────
            with st.chat_message("assistant"):
                variant = random.Random(len(st.session_state.messages)).randrange(3)
                assistant_placeholder = st.empty()

                # AI thinking state
                thinking = st.empty()
                thinking.markdown(
                    "<div style='color:#a9a4ff; font-style:italic'>🧠 Thinking...</div>"
                    "<div style='color:#888; font-size:0.9rem; margin-top:4px'>Analyzing emotion...</div>",
                    unsafe_allow_html=True,
                )

                # Typing animation: line-by-line (delay per line)
                plain = _strip_basic_markdown(bot_response)
                lines = plain.splitlines() if plain else []
                typed_text = ""
                thinking.empty()

                for line in lines:
                    typed_text += line + "\n"
                    assistant_placeholder.markdown(
                        f"<div style='white-space:pre-wrap; color:#E8E8F0'>{escape(typed_text)}</div>",
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.03)

                # Final chunked + styled assistant bubble (empathy/support/question + badge)
                assistant_placeholder.markdown(
                    _render_assistant_bubble_html(
                        bot_response=bot_response,
                        detected_emotion=emotion,
                        detected_risk=risk_level,
                        variant=variant,
                    ),
                    unsafe_allow_html=True,
                )

                # Risk alert banner (UI only)
                if risk_level == "HIGH":
                    st.error(
                        "🆘 High-risk language detected. "
                        "If you feel unsafe, please consider calling iCall: **9152987821** or **988**.",
                        icon="🆘",
                    )
                elif risk_level == "MEDIUM":
                    st.warning(
                        "💛 It sounds like you're going through a tough time. "
                        "Reach out to someone you trust.",
                        icon="💛",
                    )

            # ── Persist to DB (logic unchanged) ──────────────────────────────
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

            # ── Update session state (logic unchanged) ───────────────────────
            st.session_state.messages.append({
                "role": "user",
                "content": clean_input,
                "emotion": emotion,
                "risk_level": risk_level,
                "timestamp": now,
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_response,
                "emotion": emotion,
                "risk_level": risk_level,
                "timestamp": now,
            })
            st.session_state.emotion_records.append({
                "timestamp": now,
                "emotion": emotion,
                "emotion_score": emotion_score,
                "risk_level": risk_level,
            })

            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

with tab_dashboard:
    # ── Dashboard Navigation ──────────────────────────────────────────────
    dash_nav_col1, dash_nav_col2 = st.columns([3, 1])
    with dash_nav_col1:
        st.markdown("### 📊 Your Emotional Insights")
        st.markdown(
            "<p style='color:#888; font-size:0.9rem'>Understanding your emotional patterns "
            "can be the first step toward better mental wellbeing.</p>",
            unsafe_allow_html=True
        )
    with dash_nav_col2:
        if st.button("💬 Back to Chat", key="dashboard_to_chat"):
            st.session_state.active_tab = "chat"
            st.rerun()

    st.markdown("---")

    emotion_records = st.session_state.emotion_records

    # ── Top KPI metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    total_msgs    = len(emotion_records)
    high_risk_cnt = sum(1 for r in emotion_records if r.get("risk_level") == "HIGH")
    med_risk_cnt  = sum(1 for r in emotion_records if r.get("risk_level") == "MEDIUM")

    if emotion_records:
        dominant_emotion = max(
            set(r["emotion"] for r in emotion_records),
            key=lambda e: sum(1 for r in emotion_records if r["emotion"] == e)
        )
    else:
        dominant_emotion = "—"

    with col1:
        st.metric("💬 Messages", total_msgs)
    with col2:
        st.metric("🎭 Dominant Emotion", dominant_emotion.capitalize() if dominant_emotion != "—" else "—")
    with col3:
        st.metric("🟡 Medium Risk", med_risk_cnt)
    with col4:
        st.metric("🔴 High Risk", high_risk_cnt,
                  delta="⚠️ Review" if high_risk_cnt > 0 else None,
                  delta_color="inverse")

    st.divider()

    # ── Charts row 1: Distribution + Risk ────────────────────────────────────
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### 🎭 Emotion Breakdown")
        fig_pie = emotion_distribution_pie(emotion_records)
        st.pyplot(fig_pie, use_container_width=True)

    with c2:
        st.markdown("#### 🚦 Risk Level Distribution")
        fig_risk = risk_level_bar(emotion_records)
        st.pyplot(fig_risk, use_container_width=True)

    # ── Charts row 2: Trend line ──────────────────────────────────────────────
    st.markdown("#### 📈 Emotion Trend (This Session → Last 7 Days)")
    # Merge session data with DB history for richer trend
    db_records = get_emotion_history(st.session_state.session_id, days=7)
    all_records = db_records if db_records else emotion_records
    fig_trend = emotion_trend_line(all_records, days=7)
    st.pyplot(fig_trend, use_container_width=True)

    # ── Chart row 3: Session intensity ───────────────────────────────────────
    st.markdown("#### ⚡ Session Intensity Timeline")
    fig_intensity = session_intensity_line(emotion_records)
    st.pyplot(fig_intensity, use_container_width=True)

    # ── Session summary text ──────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🧾 Session Summary")

    if emotion_records:
        summary = get_current_session_summary(emotion_records)
        st.info(summary["summary_text"])
        cols = st.columns(3)
        cols[0].metric("Total User Messages", summary["total_messages"])
        cols[1].metric("Dominant Emotion",    summary["dominant_emotion"].capitalize())
        cols[2].metric("High-Risk Messages",  summary["high_risk_count"])
    else:
        st.info(
            "💬 No data yet! Head to the **Chat** tab and start a conversation. "
            "Your emotional insights will appear here as you chat.",
        )

    # ── Raw data expander ─────────────────────────────────────────────────────
    if emotion_records:
        with st.expander("🔍 View Raw Session Data"):
            import pandas as pd
            df_raw = pd.DataFrame(emotion_records)
            st.dataframe(df_raw, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# Footer
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; color:#444; font-size:0.75rem; padding: 20px 0 10px'>
    🧠 MindEase v1.0 &nbsp;|&nbsp; Built with ❤️ for mental health awareness &nbsp;|&nbsp;
    Not a substitute for professional therapy.
</div>
""", unsafe_allow_html=True)
