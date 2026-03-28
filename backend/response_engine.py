"""
response_engine.py
==================
Generates empathetic, human-like chatbot responses based on:
  - Detected emotion
  - Risk level
  - Conversation context (last user message)

Design principles:
  - Never dismissive; always validating first
  - Offer concrete coping strategies, not just platitudes
  - HIGH risk → crisis resources + warm redirection to professional help
  - Responses feel varied (random selection from pools) to avoid robotic repetition
"""

import random
from backend.risk_detector import CRISIS_RESOURCES, MEDIUM_RISK_NOTE

# ──────────────────────────────────────────────────────────────────────────────
# Response pools: opening acknowledgements per emotion
# ──────────────────────────────────────────────────────────────────────────────

OPENING_BY_EMOTION = {
    "sadness": [
        "I can hear that you're carrying a lot right now, and I want you to know that's completely valid. 💙",
        "It sounds like things have been really heavy for you. I'm here and I'm listening. 🫂",
        "Thank you for sharing that with me. Sadness can feel so isolating — but you don't have to face it alone.",
        "What you're feeling matters deeply. It's okay to let yourself feel sad — it's part of being human. 💙",
    ],
    "anxiety": [
        "I can sense there's a lot weighing on your mind right now. That feeling of worry can be really exhausting. 🫧",
        "Anxiety is incredibly hard to live with. What you're feeling is real, and you're not overreacting.",
        "It sounds like your nervous system is working overtime. Let's slow things down together. 🌿",
        "When everything feels uncertain or overwhelming, it makes complete sense to feel anxious. I'm with you.",
    ],
    "anger": [
        "It sounds like something (or someone) has really gotten to you, and that frustration is absolutely valid. 🔥",
        "Your anger is telling you something important. I hear you — you deserve to be heard.",
        "Feeling angry often means a boundary has been crossed or something deeply unfair has happened. Let's talk about it.",
        "It takes courage to name your anger. I'm not here to tell you to calm down — just to listen. 💛",
    ],
    "happiness": [
        "It's wonderful to hear some positivity from you! 😊 What's been bringing you joy?",
        "That sounds like a bright spot! I love hearing this — what happened?",
        "It's so good to check in when things are going well too. Keep that energy! ✨",
        "That's genuinely lovely to hear. Savoring good moments is so important for wellbeing. 🌟",
    ],
    "neutral": [
        "Thanks for sharing that with me. How are you feeling overall today?",
        "I'm here and listening. Is there something specific on your mind?",
        "Sometimes it's hard to put feelings into words. That's completely okay — we can just talk.",
        "Tell me more — I'm fully present with you right now.",
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# Coping strategies per emotion
# ──────────────────────────────────────────────────────────────────────────────

COPING_STRATEGIES = {
    "sadness": [
        "**🌊 Ride the wave:** Let yourself feel it without judgment. Emotions, like waves, rise and fall.",
        "**📓 Write it out:** Try journaling — even 3 sentences about what you're feeling can help externalize the pain.",
        "**🚶 Gentle movement:** A slow 10-minute walk, even indoors, can shift your emotional state.",
        "**💬 Reach out:** Text one person you trust, even just: *'Hey, having a rough day.'*",
        "**🎵 Mood music:** Create a playlist that matches or gently lifts your current emotion.",
    ],
    "anxiety": [
        "**🌬️ 4-7-8 Breathing:** Inhale 4 sec → Hold 7 sec → Exhale slowly 8 sec. Repeat 4×. Activates the parasympathetic system.",
        "**🧊 5-4-3-2-1 Grounding:** Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
        "**📱 Limit doom-scrolling:** Set a 15-minute timer and step away from social media or news.",
        "**✍️ Brain dump:** Write down every worry — no filter. Then ask: *'Is this in my control right now?'*",
        "**🫧 Box breathing:** Inhale 4 → Hold 4 → Exhale 4 → Hold 4. Repeat 5 times.",
    ],
    "anger": [
        "**🏃 Physical release:** 10 jumping jacks, a brisk walk, or punching a pillow — safe physical outlets help.",
        "**⏱ Take a pause:** Give yourself 90 seconds before responding to whatever triggered you. The emotion peak passes.",
        "**🖊️ Anger journaling:** Write a letter you'll never send. Say EVERYTHING. Then delete/burn/shred it.",
        "**🎶 Cold water trick:** Splash cold water on your face or hold ice — quickly activates the dive reflex and slows heart rate.",
        "**🗣️ 'I feel' framing:** When ready: *'I feel [x] when [y] because [z].'* — helps communicate without escalation.",
    ],
    "happiness": [
        "**📸 Capture the moment:** Write down 3 specific things that made today good — this trains your brain to notice positivity.",
        "**🙏 Gratitude practice:** Share this good mood with someone who lifted you up recently.",
        "**🏗️ Build on it:** Use this positive energy to tackle one thing you've been putting off.",
    ],
    "neutral": [
        "**🧘 Mindfulness check-in:** Take 2 minutes to just notice your breathing without changing it.",
        "**💧 Hydrate & step outside:** Simple basics often make a subtle but real difference.",
        "**📋 List 3 small wins:** What did you do today, even tiny things, that you can acknowledge yourself for?",
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# Follow-up check-in questions per emotion
# ──────────────────────────────────────────────────────────────────────────────

FOLLOW_UP_QUESTIONS = {
    "sadness":   "Would you like to talk about what's been making you feel this way?",
    "anxiety":   "Can you tell me more about what's making you feel anxious right now?",
    "anger":     "What happened that brought this on, if you'd like to share?",
    "happiness": "What's been the highlight of your day so far?",
    "neutral":   "Is there something specific you'd like to explore or talk through today?",
}


def generate_response(
    user_message: str,
    emotion: str,
    emotion_score: float,
    risk_level: str,
    recent_emotions: list[str] | None = None,
) -> str:
    """
    Build a complete empathetic chatbot response.

    Args:
        user_message    : The user's latest message.
        emotion         : Detected primary emotion.
        emotion_score   : Confidence score for the emotion.
        risk_level      : 'LOW', 'MEDIUM', or 'HIGH'.
        recent_emotions : List of the last N emotion labels from session history
                          (used for personalized summary lines).

    Returns:
        A formatted multi-line Markdown string to display in the chat UI.
    """

    parts: list[str] = []

    # ── 1. Crisis response (HIGH risk) ──────────────────────────────────────
    if risk_level == "HIGH":
        parts.append("💙 I hear you, and I want you to know that what you're feeling matters deeply to me.")
        parts.append(CRISIS_RESOURCES)
        parts.append("I'm still here to talk. You don't have to go through this alone.")
        return "\n\n".join(parts)

    # ── 2. Empathetic opening ─────────────────────────────────────────────────
    opening_pool = OPENING_BY_EMOTION.get(emotion, OPENING_BY_EMOTION["neutral"])
    parts.append(random.choice(opening_pool))

    # ── 3. Medium-risk warm note ──────────────────────────────────────────────
    if risk_level == "MEDIUM":
        parts.append(MEDIUM_RISK_NOTE)

    # ── 4. Trend-aware personalization ───────────────────────────────────────
    if recent_emotions and len(recent_emotions) >= 3:
        dominant = max(set(recent_emotions), key=recent_emotions.count)
        if dominant == emotion and emotion in ("sadness", "anxiety", "anger"):
            parts.append(
                f"I've noticed this has been a recurring feeling for you, and that makes total sense given what you're sharing. "
                "If you'd like, we can explore small steps that might create a little breathing room."
            )

    # ── 5. Coping strategy (single natural paragraph) ─────────────────────────
    strategy_pool = COPING_STRATEGIES.get(emotion, COPING_STRATEGIES["neutral"])
    strategy = random.choice(strategy_pool)
    strategy_text = strategy.replace("**", "").replace("> ", "")

    if emotion == "happiness":
        parts.append(f"That's wonderful to hear — {strategy_text}")
    else:
        parts.append(f"One gentle idea might be: {strategy_text}")

    # ── 6. Follow-up question ─────────────────────────────────────────────────
    follow_up = FOLLOW_UP_QUESTIONS.get(emotion, FOLLOW_UP_QUESTIONS["neutral"])
    parts.append(follow_up)

    # ── 7. Disclaimer footer (subtle) ────────────────────────────────────────
    parts.append(
        "*Remember: I'm an AI support tool, not a replacement for professional care. "
        "If things feel unmanageable, please speak with a counselor or therapist. 💙*"
    )

    return "\n\n".join(p for p in parts if p is not None and p.strip() != "")


def generate_greeting() -> str:
    """Return a warm opening greeting for a new session."""
    greetings = [
        "Hello! 👋 I'm here to listen and support you. How are you feeling today?",
        "Hi there 💙 This is a safe space for you. What's on your mind?",
        "Welcome! I'm glad you're here. How's your day going so far?",
        "Hey! Take a breath — you've got a safe space here. How are you feeling right now?",
    ]
    return random.choice(greetings)
