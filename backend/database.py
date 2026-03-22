"""
database.py
===========
Handles all SQLite database operations for the Mental Health Chatbot.
Stores messages, detected emotions, risk levels, and timestamps.

Tables:
    - sessions   : Tracks individual user sessions
    - chat_logs  : Stores every message + NLP analysis result
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional

# ── Path to the SQLite database file ─────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chatbot.db")


def get_connection() -> sqlite3.Connection:
    """Return a thread-safe SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row   # Rows accessible as dicts
    return conn


def init_db() -> None:
    """
    Create database tables if they don't already exist.
    Call this once at application startup.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)  # Ensure /data/ exists

    with get_connection() as conn:
        cursor = conn.cursor()

        # ── sessions table ────────────────────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                started_at   TEXT NOT NULL,
                last_active  TEXT NOT NULL
            )
        """)

        # ── chat_logs table ───────────────────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT    NOT NULL,
                role         TEXT    NOT NULL,   -- 'user' or 'assistant'
                message      TEXT    NOT NULL,
                emotion      TEXT    DEFAULT 'neutral',
                emotion_score REAL   DEFAULT 0.0,
                risk_level   TEXT    DEFAULT 'LOW',
                timestamp    TEXT    NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        conn.commit()
    print("[DB] Database initialized at:", DB_PATH)


def upsert_session(session_id: str) -> None:
    """
    Create a new session or update its last_active timestamp.

    Args:
        session_id: Unique identifier for the user's session.
    """
    now = datetime.now().isoformat()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO sessions (session_id, started_at, last_active)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET last_active = excluded.last_active
        """, (session_id, now, now))
        conn.commit()


def save_message(
    session_id: str,
    role: str,
    message: str,
    emotion: str = "neutral",
    emotion_score: float = 0.0,
    risk_level: str = "LOW"
) -> int:
    """
    Save a single chat message and its NLP metadata to the database.

    Args:
        session_id    : Session identifier.
        role          : 'user' or 'assistant'.
        message       : Raw message text.
        emotion       : Detected emotion label (e.g., 'sadness').
        emotion_score : Confidence score from the NLP model (0.0–1.0).
        risk_level    : 'LOW', 'MEDIUM', or 'HIGH'.

    Returns:
        The auto-incremented row ID of the inserted record.
    """
    now = datetime.now().isoformat()
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO chat_logs
                (session_id, role, message, emotion, emotion_score, risk_level, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, role, message, emotion, emotion_score, risk_level, now))
        conn.commit()
        return cursor.lastrowid


def get_session_history(session_id: str, limit: int = 50) -> list[dict]:
    """
    Fetch recent chat history for a given session.

    Args:
        session_id : Session to retrieve messages for.
        limit      : Maximum number of recent messages to return.

    Returns:
        List of dicts, each representing one chat log row.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT role, message, emotion, risk_level, timestamp
            FROM chat_logs
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit)).fetchall()

    # Reverse so oldest messages appear first
    return [dict(row) for row in reversed(rows)]


def get_emotion_history(session_id: str, days: int = 7) -> list[dict]:
    """
    Fetch emotion + timestamp data for user messages only (for trend charts).

    Args:
        session_id : Session to query.
        days       : How many past days to include.

    Returns:
        List of dicts with keys: timestamp, emotion, emotion_score, risk_level.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT timestamp, emotion, emotion_score, risk_level
            FROM chat_logs
            WHERE session_id = ?
              AND role = 'user'
              AND timestamp >= datetime('now', ? || ' days')
            ORDER BY timestamp ASC
        """, (session_id, f"-{days}")).fetchall()

    return [dict(row) for row in rows]


def get_all_emotion_history(days: int = 7) -> list[dict]:
    """
    Fetch emotion history across ALL sessions (for aggregate dashboard).

    Args:
        days: How many past days to include.

    Returns:
        List of dicts with emotion data.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT session_id, timestamp, emotion, emotion_score, risk_level
            FROM chat_logs
            WHERE role = 'user'
              AND timestamp >= datetime('now', ? || ' days')
            ORDER BY timestamp ASC
        """, (f"-{days}",)).fetchall()

    return [dict(row) for row in rows]
