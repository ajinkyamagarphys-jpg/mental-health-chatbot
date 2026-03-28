"""
utils/helpers.py
================
General-purpose utility functions shared across the application.
"""

import uuid
import re
import streamlit as st
from datetime import datetime


def generate_session_id() -> str:
    """Generate a new unique session ID using UUID4."""
    return str(uuid.uuid4())


def format_timestamp(iso_string: str) -> str:
    """
    Convert an ISO datetime string to a human-friendly format.

    Args:
        iso_string: e.g. '2024-03-15T14:32:11.456789'

    Returns:
        e.g. 'Mar 15, 2:32 PM'
    """
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%b %d, %I:%M %p")
    except (ValueError, TypeError):
        return iso_string or "Unknown time"


def sanitize_input(text: str) -> str:
    """
    Lightly sanitize user input:
      - Strip leading/trailing whitespace
      - Collapse multiple spaces/newlines into one
      - Remove null bytes

    Args:
        text: Raw user input.

    Returns:
        Cleaned string.
    """
    if not text:
        return ""
    text = text.strip()
    text = text.replace("\x00", "")
    text = re.sub(r"\s{3,}", "  ", text)       # 3+ spaces → 2 spaces
    text = re.sub(r"\n{3,}", "\n\n", text)     # 3+ newlines → 2
    return text


def risk_badge(risk_level: str) -> str:
    """
    Return a colored emoji badge for a given risk level.

    Args:
        risk_level: 'LOW', 'MEDIUM', or 'HIGH'

    Returns:
        e.g. '🟢 LOW'
    """
    badges = {
        "LOW":    "🟢 LOW",
        "MEDIUM": "🟡 MEDIUM",
        "HIGH":   "🔴 HIGH",
    }
    return badges.get(risk_level, "⚪ UNKNOWN")


def emotion_badge(emotion: str) -> str:
    """
    Return an emoji + label for a detected emotion.

    Args:
        emotion: Mapped emotion label.

    Returns:
        e.g. '😢 Sadness'
    """
    badges = {
        "sadness":   "😢 Sadness",
        "anxiety":   "😰 Anxiety",
        "anger":     "😠 Anger",
        "happiness": "😊 Happiness",
        "neutral":   "😐 Neutral",
    }
    return badges.get(emotion, f"🔵 {emotion.capitalize()}")


def init_session_state(defaults: dict) -> None:
    """
    Initialize Streamlit session state keys if they don't already exist.

    Args:
        defaults: Dict of {key: default_value} to set if missing.

    Usage:
        init_session_state({"messages": [], "session_id": None})
    """
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def truncate_text(text: str, max_chars: int = 80) -> str:
    """
    Truncate text to max_chars and append ellipsis if needed.

    Args:
        text     : String to truncate.
        max_chars: Maximum character length.

    Returns:
        Truncated string.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."
