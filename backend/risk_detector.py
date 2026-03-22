"""
risk_detector.py
================
Detects emotional risk level in user messages using a two-layer approach:

  Layer 1 — Keyword matching:
      Scans for explicit self-harm / crisis / high-distress language.

  Layer 2 — Emotion + score weighting:
      Combines NLP emotion signals with keyword results for a final score.

Risk levels:
    LOW    → Normal conversation, mild emotions.
    MEDIUM → Moderate distress; suggest coping strategies.
    HIGH   → Potential self-harm / crisis; show crisis resources immediately.

IMPORTANT: This is a detection aid, NOT a clinical tool.
"""

import re
from typing import Tuple

# ── Keyword libraries ─────────────────────────────────────────────────────────

# HIGH-RISK: phrases strongly associated with self-harm or suicidal ideation
HIGH_RISK_KEYWORDS = [
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\bwant to die\b",
    r"\bsuicid\w*\b",
    r"\bself.?harm\b",
    r"\bcut myself\b",
    r"\bhurt myself\b",
    r"\bno reason to live\b",
    r"\bnot worth living\b",
    r"\bbetter off dead\b",
    r"\bcan't go on\b",
    r"\bcan't take it anymore\b",
    r"\boverdose\b",
    r"\bwant to disappear\b",
    r"\bending it all\b",
    r"\btake my life\b",
]

# MEDIUM-RISK: phrases indicating significant emotional distress
MEDIUM_RISK_KEYWORDS = [
    r"\bfeel hopeless\b",
    r"\bfeel worthless\b",
    r"\bgive up\b",
    r"\bno hope\b",
    r"\bnumb\b",
    r"\bempty inside\b",
    r"\bbreakdown\b",
    r"\bcan't cope\b",
    r"\bpanic attack\b",
    r"\bextremely anxious\b",
    r"\bterrified\b",
    r"\bhate myself\b",
    r"\bfeel alone\b",
    r"\bno one cares\b",
    r"\bwhat's the point\b",
    r"\bwhat is the point\b",
    r"\bexhausted\b",
    r"\bdesperate\b",
]

# Emotions that raise risk level when combined with distress keywords
HIGH_RISK_EMOTIONS   = {"sadness", "anger", "anxiety"}
MEDIUM_RISK_EMOTIONS = {"sadness", "anxiety"}


def _scan_keywords(text: str, patterns: list[str]) -> bool:
    """
    Check whether any pattern in `patterns` matches the lowercased text.

    Args:
        text     : Input text to scan.
        patterns : List of regex pattern strings.

    Returns:
        True if any pattern matches, False otherwise.
    """
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in patterns)


def classify_risk(
    message: str,
    emotion: str,
    emotion_score: float
) -> Tuple[str, str]:
    """
    Classify the risk level of a user message.

    Algorithm:
        1. Check for HIGH-risk keywords → immediately return HIGH.
        2. Check for MEDIUM-risk keywords → start at MEDIUM.
        3. Use emotion + score to refine the final level:
           - Strong negative emotion (score > 0.75) raises MEDIUM → HIGH.
           - Moderate negative emotion can raise LOW → MEDIUM.

    Args:
        message       : Raw user message.
        emotion       : Detected primary emotion label.
        emotion_score : Confidence score (0.0–1.0) from NLP model.

    Returns:
        Tuple of (risk_level, explanation_string).
            risk_level ∈ {"LOW", "MEDIUM", "HIGH"}
    """

    # ── Layer 1: Explicit high-risk language ─────────────────────────────────
    if _scan_keywords(message, HIGH_RISK_KEYWORDS):
        return (
            "HIGH",
            "High-risk language detected. Crisis support resources shown."
        )

    # ── Layer 2: Medium-risk language ────────────────────────────────────────
    medium_keyword_hit = _scan_keywords(message, MEDIUM_RISK_KEYWORDS)

    if medium_keyword_hit:
        # Strong negative emotion + distress keywords = escalate to HIGH
        if emotion in HIGH_RISK_EMOTIONS and emotion_score > 0.75:
            return (
                "HIGH",
                "Significant distress language combined with strong negative emotion."
            )
        return (
            "MEDIUM",
            "Distress-related language detected. Coping strategies suggested."
        )

    # ── Layer 3: Emotion-based risk without keywords ──────────────────────────
    if emotion in MEDIUM_RISK_EMOTIONS and emotion_score > 0.80:
        return (
            "MEDIUM",
            "Strong negative emotion detected without explicit crisis language."
        )

    # ── Default: LOW ──────────────────────────────────────────────────────────
    return ("LOW", "No significant risk indicators detected.")


# ── Crisis resource strings (shown only on HIGH risk) ─────────────────────────
CRISIS_RESOURCES = """
🆘 **It looks like you may be going through something really difficult.**

Please know you are not alone, and help is available **right now**:

| Resource | Contact |
|---|---|
| **iCall (India)** | 9152987821 |
| **Vandrevala Foundation** | 1860-2662-345 (24/7) |
| **iCall (Email)** | icall@tiss.edu |
| **International Association for Suicide Prevention** | https://www.iasp.info/resources/Crisis_Centres/ |
| **Crisis Text Line (US)** | Text HOME to 741741 |

💙 *This chatbot is not a substitute for professional help. Please reach out to a trusted person or counselor.*
"""

MEDIUM_RISK_NOTE = "💛 I can hear that things feel tough right now. I'm here with you."
