"""
nlp_engine.py
=============
NLP Emotion Detection using HuggingFace Transformers.

Model: j-hartmann/emotion-english-distilroberta-base
  - Lightweight DistilRoBERTa fine-tuned on 6 emotion datasets
  - Labels: anger, disgust, fear, joy, neutral, sadness, surprise
  - Fast inference, runs on CPU without issues

We map the raw model labels → our 5 mental-health-focused categories:
    anger    → anger
    disgust  → anger       (similar valence for our purposes)
    fear     → anxiety
    joy      → happiness
    neutral  → neutral
    sadness  → sadness
    surprise → neutral     (ambiguous; treat as neutral)
"""

from transformers import pipeline
from typing import Optional
import torch

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# Map HuggingFace labels → our internal emotion vocabulary
LABEL_MAP = {
    "anger":    "anger",
    "disgust":  "anger",
    "fear":     "anxiety",
    "joy":      "happiness",
    "neutral":  "neutral",
    "sadness":  "sadness",
    "surprise": "neutral",
}

# Emoji for each emotion (used in UI responses)
EMOTION_EMOJI = {
    "anger":     "😠",
    "anxiety":   "😰",
    "happiness": "😊",
    "neutral":   "😐",
    "sadness":   "😢",
}

# ── Global model singleton (loaded once on first call) ────────────────────────
_emotion_pipeline = None


def load_model() -> None:
    """
    Load the HuggingFace emotion-classification pipeline into memory.
    Uses CPU by default (compatible with Streamlit Cloud free tier).
    Call this at app startup to avoid cold-start delays during chat.
    """
    global _emotion_pipeline
    if _emotion_pipeline is None:
        print(f"[NLP] Loading emotion model: {MODEL_NAME} ...")
        device = 0 if torch.cuda.is_available() else -1   # GPU if available, else CPU
        _emotion_pipeline = pipeline(
            task="text-classification",
            model=MODEL_NAME,
            return_all_scores=True,   # Return scores for all labels
            device=device,
        )
        print("[NLP] Model loaded successfully.")


def detect_emotion(text: str) -> dict:
    """
    Run emotion detection on a user message.

    Args:
        text: The user's raw message string.

    Returns:
        A dict with:
            - primary_emotion (str)      : Top mapped emotion label
            - primary_score   (float)    : Confidence score 0.0–1.0
            - all_scores      (dict)     : {mapped_emotion: score} for all categories
            - raw_label       (str)      : Original HuggingFace label (for debugging)
    """
    # ── Guard: empty or very short input ─────────────────────────────────────
    text = text.strip()
    if len(text) < 2:
        return {
            "primary_emotion": "neutral",
            "primary_score": 1.0,
            "all_scores": {e: 0.0 for e in EMOTION_EMOJI},
            "raw_label": "neutral"
        }

    # ── Ensure model is loaded ────────────────────────────────────────────────
    load_model()

    # ── Run inference ─────────────────────────────────────────────────────────
    raw_output = _emotion_pipeline(text[:512])

    # ── Normalize output format ───────────────────────────────────────────────
    # Older transformers: [[{"label":..., "score":...}, ...]]
    # Newer transformers: [{"label":..., "score":...}, ...]  OR
    #                     just the top result as a single dict
    if isinstance(raw_output, list) and len(raw_output) > 0:
        first = raw_output[0]
        if isinstance(first, list):
            # Nested list format (older): [[{...}, {...}]]
            raw_results = first
        elif isinstance(first, dict):
            # Flat list format (newer): [{...}, {...}]
            raw_results = raw_output
        else:
            raw_results = []
    else:
        raw_results = []

    # ── Fallback: if we still got nothing usable, return neutral ─────────────
    if not raw_results:
        return {
            "primary_emotion": "neutral",
            "primary_score": 1.0,
            "all_scores": {e: 0.0 for e in EMOTION_EMOJI},
            "raw_label": "neutral"
        }

    # ── Aggregate scores by mapped emotion category ───────────────────────────
    aggregated: dict = {e: 0.0 for e in EMOTION_EMOJI}
    best_raw_label = "neutral"
    best_raw_score = 0.0

    for item in raw_results:
        # Handle both dict items and string items defensively
        if isinstance(item, dict):
            raw_label = item.get("label", "neutral")
            score     = float(item.get("score", 0.0))
        elif isinstance(item, str):
            # Some versions return just the label string
            raw_label = item
            score     = 1.0
        else:
            continue

        mapped = LABEL_MAP.get(raw_label.lower(), "neutral")

        if score > best_raw_score:
            best_raw_score = score
            best_raw_label = raw_label

        aggregated[mapped] += score

    # ── Find the dominant mapped emotion ─────────────────────────────────────
    primary_emotion = max(aggregated, key=aggregated.get)
    primary_score   = round(aggregated[primary_emotion], 4)

    return {
        "primary_emotion": primary_emotion,
        "primary_score":   primary_score,
        "all_scores":      {k: round(v, 4) for k, v in aggregated.items()},
        "raw_label":       best_raw_label,
    }


def get_emotion_summary(emotion: str, score: float) -> str:
    """
    Return a human-readable summary line for the detected emotion.

    Args:
        emotion : Mapped emotion label.
        score   : Confidence score.

    Returns:
        A short summary string, e.g. "😢 Sadness detected (87% confidence)"
    """
    emoji = EMOTION_EMOJI.get(emotion, "🔵")
    pct   = int(score * 100)
    label = emotion.capitalize()
    return f"{emoji} {label} detected ({pct}% confidence)"