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
                user_email   TEXT    NOT NULL,
                started_at   TEXT    NOT NULL,
                last_active  TEXT    NOT NULL
            )
        """)

        # ── chat_logs table ───────────────────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT    NOT NULL,
                user_email   TEXT    NOT NULL,
                role         TEXT    NOT NULL,   -- 'user' or 'assistant'
                message      TEXT    NOT NULL,
                emotion      TEXT    DEFAULT 'neutral',
                emotion_score REAL   DEFAULT 0.0,
                risk_level   TEXT    DEFAULT 'LOW',
                timestamp    TEXT    NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        # Add user_email column to existing tables if they don't have it
        cursor.execute("PRAGMA table_info(sessions)")
        columns = [col[1] for col in cursor.fetchall()]
        if "user_email" not in columns:
            cursor.execute("ALTER TABLE sessions ADD COLUMN user_email TEXT DEFAULT 'unknown'")
        
        cursor.execute("PRAGMA table_info(chat_logs)")
        columns = [col[1] for col in cursor.fetchall()]
        if "user_email" not in columns:
            cursor.execute("ALTER TABLE chat_logs ADD COLUMN user_email TEXT DEFAULT 'unknown'")

        conn.commit()
    print("[DB] Database initialized at:", DB_PATH)


def upsert_session(session_id: str, user_email: str = "unknown") -> None:
    """
    Create a new session or update its last_active timestamp.

    Args:
        session_id: Unique identifier for the user's session.
        user_email: Email of the logged-in user.
    """
    now = datetime.now().isoformat()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO sessions (session_id, user_email, started_at, last_active)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET last_active = excluded.last_active
        """, (session_id, user_email, now, now))
        conn.commit()


def save_message(
    session_id: str,
    role: str,
    message: str,
    user_email: str = "unknown",
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
        user_email    : Email of the user associated with this session.
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
                (session_id, user_email, role, message, emotion, emotion_score, risk_level, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, user_email, role, message, emotion, emotion_score, risk_level, now))
        conn.commit()
        return cursor.lastrowid


def get_session_history(session_id: str, user_email: str = "", limit: int = 50) -> list[dict]:
    """
    Fetch recent chat history for a given session (filtered by user email for security).

    Args:
        session_id : Session to retrieve messages for.
        user_email : Email of the user (for permission verification).
        limit      : Maximum number of recent messages to return.

    Returns:
        List of dicts, each representing one chat log row.
    """
    with get_connection() as conn:
        if user_email:
            rows = conn.execute("""
                SELECT role, message, emotion, risk_level, timestamp
                FROM chat_logs
                WHERE session_id = ? AND user_email = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, user_email, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT role, message, emotion, risk_level, timestamp
                FROM chat_logs
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit)).fetchall()

    # Reverse so oldest messages appear first
    return [dict(row) for row in reversed(rows)]


def get_emotion_history(session_id: str, user_email: str = "", days: int = 7) -> list[dict]:
    """
    Fetch emotion + timestamp data for user messages only (for trend charts).

    Args:
        session_id : Session to query.
        user_email : Email of the user (for permission verification).
        days       : How many past days to include.

    Returns:
        List of dicts with keys: timestamp, emotion, emotion_score, risk_level.
    """
    with get_connection() as conn:
        if user_email:
            rows = conn.execute("""
                SELECT timestamp, emotion, emotion_score, risk_level
                FROM chat_logs
                WHERE session_id = ? AND user_email = ?
                  AND role = 'user'
                  AND timestamp >= datetime('now', ? || ' days')
                ORDER BY timestamp ASC
            """, (session_id, user_email, f"-{days}")).fetchall()
        else:
            rows = conn.execute("""
                SELECT timestamp, emotion, emotion_score, risk_level
                FROM chat_logs
                WHERE session_id = ?
                  AND role = 'user'
                  AND timestamp >= datetime('now', ? || ' days')
                ORDER BY timestamp ASC
            """, (session_id, f"-{days}")).fetchall()

    return [dict(row) for row in rows]


def get_user_sessions(user_email: str) -> list[dict]:
    """
    Fetch all sessions for a specific user email.

    Args:
        user_email: Email of the user.

    Returns:
        List of session dicts with session_id, started_at, last_active.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT session_id, started_at, last_active
            FROM sessions
            WHERE user_email = ?
            ORDER BY last_active DESC
        """, (user_email,)).fetchall()

    return [dict(row) for row in rows]


def get_latest_user_session(user_email: str) -> Optional[str]:
    """
    Get the most recently active session ID for a user.

    Args:
        user_email: Email of the user.

    Returns:
        Session ID string, or None if user has no sessions.
    """
    sessions = get_user_sessions(user_email)
    return sessions[0]['session_id'] if sessions else None
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
