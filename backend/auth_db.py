"""
backend/auth_db.py
==================
SQLite-backed user authentication helpers.

Provides:
  - register_user(name, email, password)  → (ok: bool, msg: str)
  - verify_user(email, password)          → (ok: bool, name: str | None)
  - reset_password(email, new_password)   → (ok: bool, msg: str)
  - user_exists(email)                    → bool
  - get_user_name(email)                  → str | None

Passwords are stored as SHA-256 hashes (salted with email).
No extra dependencies — uses only stdlib sqlite3 + hashlib.

The users table is created alongside the existing tables in init_db().
If init_db() is not called before these helpers, the helpers call it
themselves via a lazy import.
"""

import sqlite3
import hashlib
import os

# ── Re-use the same DB path as the rest of the app ───────────────────────────
_DB_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH  = os.path.join(_DB_DIR, "chatbot.db")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    os.makedirs(_DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _hash_password(email: str, password: str) -> str:
    """
    Derive a deterministic per-user hash.
    Using email as a salt keeps it simple without storing a separate salt column.
    """
    raw = f"{email.lower().strip()}::{password}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def init_users_table() -> None:
    """Create the users table if it does not exist yet."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                name         TEXT    NOT NULL,
                email        TEXT    UNIQUE NOT NULL,
                password_hash TEXT   NOT NULL,
                created_at   TEXT    DEFAULT (datetime('now'))
            )
        """)
        conn.commit()


# ── Public API ────────────────────────────────────────────────────────────────

def user_exists(email: str) -> bool:
    init_users_table()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM users WHERE LOWER(email) = LOWER(?)", (email.strip(),)
        ).fetchone()
        return row is not None


def register_user(name: str, email: str, password: str) -> tuple[bool, str]:
    """
    Register a new user.
    Returns (True, "ok") on success or (False, reason) on failure.
    """
    init_users_table()
    email = email.strip().lower()
    name  = name.strip()

    if not name or not email or not password:
        return False, "All fields are required."
    if "@" not in email or "." not in email:
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    if user_exists(email):
        return False, "An account with this email already exists."

    pw_hash = _hash_password(email, password)
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, pw_hash),
            )
            conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."


def verify_user(email: str, password: str) -> tuple[bool, str | None]:
    """
    Verify login credentials.
    Returns (True, display_name) on success or (False, None) on failure.
    """
    init_users_table()
    email = email.strip().lower()
    pw_hash = _hash_password(email, password)

    with _get_conn() as conn:
        row = conn.execute(
            "SELECT name FROM users WHERE LOWER(email) = ? AND password_hash = ?",
            (email, pw_hash),
        ).fetchone()
        if row:
            return True, row["name"]
        return False, None


def reset_password(email: str, new_password: str) -> tuple[bool, str]:
    """
    Reset a user's password (no email step — direct reset for prototype).
    """
    init_users_table()
    email = email.strip().lower()

    if not user_exists(email):
        return False, "No account found with that email address."
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters."

    pw_hash = _hash_password(email, new_password)
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE LOWER(email) = ?",
            (pw_hash, email),
        )
        conn.commit()
    return True, "Password updated successfully. You can now log in."


def get_user_name(email: str) -> str | None:
    """Return the stored display name for an email, or None."""
    init_users_table()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT name FROM users WHERE LOWER(email) = LOWER(?)", (email.strip(),)
        ).fetchone()
        return row["name"] if row else None
