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
import httpx
import json
import time
from datetime import datetime

# ── Internal imports (fallback mode) ─────────────────────────────────────────
from backend.database        import init_db, upsert_session, save_message, \
                                    get_session_history, get_emotion_history
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
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Session State Initialization
# ════════════════════════════════════════════════════════════════════════════

init_session_state({
    "session_id":       None,
    "messages":         [],      # {role, content, emotion, risk_level, timestamp}
    "emotion_records":  [],      # Emotion data for charts
    "model_loaded":     False,
    "is_typing":        False,
})

# ── Initialize DB and load NLP model on first run ─────────────────────────────
if not st.session_state.model_loaded:
    with st.spinner("🧠 Loading AI model... (first load may take ~30 seconds)"):
        init_db()
        load_model()
    st.session_state.model_loaded = True

# ── Create a new session if we don't have one ─────────────────────────────────
if st.session_state.session_id is None:
    sid = generate_session_id()
    st.session_state.session_id = sid
    upsert_session(sid)

    # Add bot greeting
    greeting = generate_greeting()
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   greeting,
        "emotion":   "neutral",
        "risk_level":"LOW",
        "timestamp": datetime.now().isoformat(),
    })


# ════════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 MindEase")
    st.markdown("*Your AI-powered emotional support companion*")
    st.divider()

    # Session info
    st.markdown("**Session ID**")
    st.code(st.session_state.session_id[:18] + "...", language=None)
    st.markdown(f"**Messages:** {len(st.session_state.messages)}")

    st.divider()

    # Quick emotion summary
    if st.session_state.emotion_records:
        summary = get_current_session_summary(st.session_state.emotion_records)
        st.markdown("**Today's Mood Summary**")
        st.markdown(summary["summary_text"])
        st.divider()

    # New Session button
    if st.button("🔄 New Session", use_container_width=True):
        for key in ["session_id", "messages", "emotion_records"]:
            st.session_state[key] = None if key == "session_id" else []
        st.rerun()

    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem; color:#666; text-align:center'>
    ⚠️ This is an AI support tool,<br>
    not a replacement for therapy.<br><br>
    🆘 <b>Crisis?</b> Call iCall: 9152987821<br>
    (India) or 988 (US)
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Main content — Tabs
# ════════════════════════════════════════════════════════════════════════════

tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Insights Dashboard"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.markdown("### 💬 How are you feeling today?")
    st.markdown(
        "<p style='color:#888; font-size:0.9rem'>This is a safe, judgment-free space. "
        "Share what's on your mind.</p>",
        unsafe_allow_html=True
    )

    # ── Chat history display ─────────────────────────────────────────────────
    chat_area = st.container()

    with chat_area:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="user-bubble">{msg['content']}</div>
                <div style='text-align:right'>
                  <span class="meta-chip chip-emotion">{emotion_badge(msg.get('emotion','neutral'))}</span>
                  <span class="meta-chip chip-{msg.get('risk_level','LOW').lower()}">{risk_badge(msg.get('risk_level','LOW'))}</span>
                </div>
                """, unsafe_allow_html=True)

            else:   # assistant
                st.markdown(f"""
                <div class="bot-bubble">{msg['content']}</div>
                """, unsafe_allow_html=True)

    # ── Typing indicator placeholder ─────────────────────────────────────────
    typing_placeholder = st.empty()

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Type how you're feeling... (e.g. 'I've been really anxious lately')"
    )

    if user_input:
        clean_input = sanitize_input(user_input)

        if not clean_input:
            st.warning("Please enter a message before sending.")
        else:
            # ── Typing indicator ──────────────────────────────────────────────
            with typing_placeholder:
                st.markdown(
                    "<div style='color:#888; font-style:italic; font-size:0.85rem'>🤔 Analyzing...</div>",
                    unsafe_allow_html=True
                )

            # ── Run NLP pipeline (direct module call) ─────────────────────────
            nlp_result    = detect_emotion(clean_input)
            emotion       = nlp_result["primary_emotion"]
            emotion_score = nlp_result["primary_score"]

            risk_level, risk_explanation = classify_risk(
                clean_input, emotion, emotion_score
            )

            # Fetch recent context for trend-aware response
            recent_emotions = [
                r["emotion"] for r in st.session_state.emotion_records[-5:]
            ]

            bot_response = generate_response(
                user_message=clean_input,
                emotion=emotion,
                emotion_score=emotion_score,
                risk_level=risk_level,
                recent_emotions=recent_emotions,
            )

            # ── Persist to DB ─────────────────────────────────────────────────
            now = datetime.now().isoformat()
            save_message(
                session_id=st.session_state.session_id,
                role="user",
                message=clean_input,
                emotion=emotion,
                emotion_score=emotion_score,
                risk_level=risk_level,
            )
            save_message(
                session_id=st.session_state.session_id,
                role="assistant",
                message=bot_response,
            )

            # ── Update session state ──────────────────────────────────────────
            st.session_state.messages.append({
                "role":       "user",
                "content":    clean_input,
                "emotion":    emotion,
                "risk_level": risk_level,
                "timestamp":  now,
            })
            st.session_state.messages.append({
                "role":      "assistant",
                "content":   bot_response,
                "emotion":   "neutral",
                "risk_level":"LOW",
                "timestamp": now,
            })
            st.session_state.emotion_records.append({
                "timestamp":     now,
                "emotion":       emotion,
                "emotion_score": emotion_score,
                "risk_level":    risk_level,
            })

            # ── Clear typing indicator ────────────────────────────────────────
            typing_placeholder.empty()

            # ── Show risk alert banner for HIGH risk ──────────────────────────
            if risk_level == "HIGH":
                st.error(
                    "🆘 **High-risk message detected.** "
                    "Please consider calling iCall: **9152987821** or **988** (US/Canada). "
                    "You are not alone. 💙",
                    icon="🆘"
                )
            elif risk_level == "MEDIUM":
                st.warning(
                    "💛 It sounds like you're going through a tough time. "
                    "Reach out to someone you trust.",
                    icon="💛"
                )

            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

with tab_dashboard:
    st.markdown("### 📊 Your Emotional Insights")
    st.markdown(
        "<p style='color:#888; font-size:0.9rem'>Understanding your emotional patterns "
        "can be the first step toward better mental wellbeing.</p>",
        unsafe_allow_html=True
    )

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
