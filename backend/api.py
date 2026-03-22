"""
api.py
======
FastAPI backend for the Mental Health Chatbot.
Exposes a clean REST API that the Streamlit frontend calls via HTTP.

Endpoints:
    POST /chat              → Process a user message (NLP + risk + response)
    GET  /history/{sid}     → Fetch chat history for a session
    GET  /emotions/{sid}    → Fetch emotion records for trend analysis
    GET  /summary/{sid}     → Get session summary (dominant emotion, risk count)
    GET  /health            → Simple health-check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uuid

# ── Internal modules ──────────────────────────────────────────────────────────
from backend.database       import init_db, upsert_session, save_message, \
                                   get_session_history, get_emotion_history
from backend.nlp_engine     import detect_emotion, get_emotion_summary
from backend.risk_detector  import classify_risk
from backend.response_engine import generate_response, generate_greeting
from backend.data_processor import get_current_session_summary

# ── App initialization ────────────────────────────────────────────────────────
app = FastAPI(
    title="Mental Health Chatbot API",
    description="NLP-powered chatbot for emotional support and early risk detection.",
    version="1.0.0",
)

# Allow Streamlit (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB tables on startup
@app.on_event("startup")
def startup_event():
    init_db()
    # Pre-load NLP model so first chat is fast
    from backend.nlp_engine import load_model
    load_model()
    print("[API] Startup complete. NLP model loaded.")


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str  = Field(..., description="Unique session identifier")
    message:    str  = Field(..., min_length=1, max_length=2000,
                             description="User's message text")


class ChatResponse(BaseModel):
    session_id:     str
    user_message:   str
    bot_response:   str
    emotion:        str
    emotion_score:  float
    emotion_summary: str
    risk_level:     str
    risk_explanation: str


class NewSessionResponse(BaseModel):
    session_id: str
    greeting:   str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Simple endpoint to verify the API is alive."""
    return {"status": "ok", "service": "Mental Health Chatbot API"}


@app.post("/session/new", response_model=NewSessionResponse)
def create_session():
    """
    Create a new chat session.
    Returns a unique session_id and an opening greeting message.
    """
    session_id = str(uuid.uuid4())
    upsert_session(session_id)
    greeting = generate_greeting()

    # Store the bot's greeting in chat history
    save_message(
        session_id=session_id,
        role="assistant",
        message=greeting,
        emotion="neutral",
        emotion_score=1.0,
        risk_level="LOW",
    )

    return NewSessionResponse(session_id=session_id, greeting=greeting)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint. Full pipeline:
        1. Save user message → DB (placeholder)
        2. Run emotion detection via NLP
        3. Classify risk level
        4. Generate empathetic response
        5. Persist both user + bot messages to DB
        6. Return full analysis to frontend
    """

    # ── Guard: non-empty message ──────────────────────────────────────────────
    user_msg = request.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # ── Ensure session exists ─────────────────────────────────────────────────
    upsert_session(request.session_id)

    # ── Step 1: Emotion detection ─────────────────────────────────────────────
    nlp_result     = detect_emotion(user_msg)
    emotion        = nlp_result["primary_emotion"]
    emotion_score  = nlp_result["primary_score"]
    emotion_summary = get_emotion_summary(emotion, emotion_score)

    # ── Step 2: Risk classification ───────────────────────────────────────────
    risk_level, risk_explanation = classify_risk(user_msg, emotion, emotion_score)

    # ── Step 3: Fetch recent emotion context (for trend-aware responses) ──────
    recent_records  = get_emotion_history(request.session_id, days=1)
    recent_emotions = [r["emotion"] for r in recent_records[-5:]]   # Last 5

    # ── Step 4: Generate response ─────────────────────────────────────────────
    bot_response = generate_response(
        user_message=user_msg,
        emotion=emotion,
        emotion_score=emotion_score,
        risk_level=risk_level,
        recent_emotions=recent_emotions,
    )

    # ── Step 5: Persist to DB ─────────────────────────────────────────────────
    save_message(
        session_id=request.session_id,
        role="user",
        message=user_msg,
        emotion=emotion,
        emotion_score=emotion_score,
        risk_level=risk_level,
    )
    save_message(
        session_id=request.session_id,
        role="assistant",
        message=bot_response,
        emotion="neutral",      # Bot responses tagged as neutral
        emotion_score=1.0,
        risk_level="LOW",
    )

    return ChatResponse(
        session_id=request.session_id,
        user_message=user_msg,
        bot_response=bot_response,
        emotion=emotion,
        emotion_score=emotion_score,
        emotion_summary=emotion_summary,
        risk_level=risk_level,
        risk_explanation=risk_explanation,
    )


@app.get("/history/{session_id}")
def get_history(session_id: str, limit: int = 50):
    """
    Retrieve recent chat history for a session.

    Args:
        session_id: The session to query.
        limit:      Max messages to return (default 50).
    """
    history = get_session_history(session_id, limit=limit)
    return {"session_id": session_id, "messages": history}


@app.get("/emotions/{session_id}")
def get_emotions(session_id: str, days: int = 7):
    """
    Retrieve emotion history for a session (user messages only).

    Args:
        session_id: The session to query.
        days:       How many past days to include.
    """
    records = get_emotion_history(session_id, days=days)
    return {"session_id": session_id, "records": records, "count": len(records)}


@app.get("/summary/{session_id}")
def get_summary(session_id: str):
    """
    Get a natural-language summary and stats for the current session.
    """
    records = get_emotion_history(session_id, days=1)
    summary = get_current_session_summary(records)
    return {"session_id": session_id, **summary}
