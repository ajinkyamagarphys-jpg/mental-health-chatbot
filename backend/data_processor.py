"""
data_processor.py
=================
Uses Pandas to process and aggregate raw emotion data from SQLite.
Produces ready-to-plot DataFrames for the dashboard visualizations.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


# ── Canonical emotion ordering (for consistent chart colors) ──────────────────
EMOTION_ORDER = ["happiness", "neutral", "anxiety", "sadness", "anger"]

EMOTION_COLORS = {
    "happiness": "#FFD700",   # Gold
    "neutral":   "#A0AEC0",   # Grey-blue
    "anxiety":   "#F6AD55",   # Amber
    "sadness":   "#63B3ED",   # Sky blue
    "anger":     "#FC8181",   # Coral red
}

RISK_COLORS = {
    "LOW":    "#68D391",   # Green
    "MEDIUM": "#F6E05E",   # Yellow
    "HIGH":   "#FC8181",   # Red
}


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """
    Convert a list of raw database rows into a clean DataFrame.

    Ensures:
        - 'timestamp' is parsed as a proper datetime column.
        - 'date' column (date-only) is added for daily aggregations.
        - 'emotion_score' is cast to float.

    Args:
        records: Output from database.get_emotion_history() or similar.

    Returns:
        A cleaned Pandas DataFrame, or an empty DataFrame if no data.
    """
    if not records:
        return pd.DataFrame(columns=["timestamp", "date", "emotion", "emotion_score", "risk_level"])

    df = pd.DataFrame(records)

    # Parse timestamp string → datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Add a date-only column for grouping by day
    df["date"] = df["timestamp"].dt.date

    # Ensure numeric score
    if "emotion_score" in df.columns:
        df["emotion_score"] = pd.to_numeric(df["emotion_score"], errors="coerce").fillna(0.0)

    # Fill missing emotion / risk_level with defaults
    df["emotion"]    = df.get("emotion", pd.Series(dtype=str)).fillna("neutral")
    df["risk_level"] = df.get("risk_level", pd.Series(dtype=str)).fillna("LOW")

    return df.sort_values("timestamp").reset_index(drop=True)


def get_emotion_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count occurrences of each emotion across all records.

    Args:
        df: Cleaned emotion DataFrame from records_to_dataframe().

    Returns:
        DataFrame with columns [emotion, count, percentage].
    """
    if df.empty:
        return pd.DataFrame(columns=["emotion", "count", "percentage"])

    counts = (
        df["emotion"]
        .value_counts()
        .reindex(EMOTION_ORDER, fill_value=0)
        .reset_index()
    )
    counts.columns = ["emotion", "count"]
    counts["percentage"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts


def get_daily_emotion_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate emotions by day → produces data for the trend line chart.

    Strategy: For each day, return the most frequent emotion and the
    average emotion score (a proxy for emotional intensity).

    Args:
        df: Cleaned emotion DataFrame.

    Returns:
        DataFrame with columns [date, dominant_emotion, avg_score, message_count].
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "dominant_emotion", "avg_score", "message_count"])

    # Group by date
    grouped = df.groupby("date")

    daily_records = []
    for date_val, group in grouped:
        dominant_emotion = group["emotion"].mode()[0]   # Most common
        avg_score        = group["emotion_score"].mean()
        message_count    = len(group)
        daily_records.append({
            "date":             date_val,
            "dominant_emotion": dominant_emotion,
            "avg_score":        round(avg_score, 3),
            "message_count":    message_count,
        })

    result = pd.DataFrame(daily_records)
    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values("date").reset_index(drop=True)


def get_risk_summary(df: pd.DataFrame) -> dict:
    """
    Summarize risk level distribution.

    Args:
        df: Cleaned emotion DataFrame.

    Returns:
        Dict: {"LOW": count, "MEDIUM": count, "HIGH": count, "total": count}
    """
    if df.empty:
        return {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "total": 0}

    counts = df["risk_level"].value_counts().to_dict()
    return {
        "LOW":    counts.get("LOW",    0),
        "MEDIUM": counts.get("MEDIUM", 0),
        "HIGH":   counts.get("HIGH",   0),
        "total":  len(df),
    }


def get_current_session_summary(emotion_records: list[dict]) -> dict:
    """
    Generate a short natural-language summary for the current session.

    Args:
        emotion_records: Recent emotion records for this session.

    Returns:
        Dict with keys:
            - summary_text     (str) : Human-readable summary
            - dominant_emotion (str) : Most common emotion in session
            - high_risk_count  (int) : Number of HIGH risk messages
            - total_messages   (int) : Total user messages
    """
    if not emotion_records:
        return {
            "summary_text":     "No messages yet in this session.",
            "dominant_emotion": "neutral",
            "high_risk_count":  0,
            "total_messages":   0,
        }

    df = records_to_dataframe(emotion_records)

    dominant = df["emotion"].mode()[0] if not df.empty else "neutral"
    high_risk_count = int((df["risk_level"] == "HIGH").sum())
    total = len(df)

    # Build a natural-language summary
    if dominant == "happiness":
        summary = f"😊 You've seemed mostly **happy** today — that's wonderful!"
    elif dominant == "sadness":
        summary = f"😢 You've been expressing a lot of **sadness** today. I hope things improve."
    elif dominant == "anxiety":
        summary = f"😰 **Anxiety** seems to be a theme today. Remember to breathe."
    elif dominant == "anger":
        summary = f"😠 It seems like **frustration** has come up a lot today. That can be draining."
    else:
        summary = f"😐 Your tone has been mostly **neutral** today."

    if high_risk_count > 0:
        summary += f"\n\n⚠️ *{high_risk_count} message(s) flagged for high-risk language. Please check in with a professional.*"

    return {
        "summary_text":     summary,
        "dominant_emotion": dominant,
        "high_risk_count":  high_risk_count,
        "total_messages":   total,
    }


def fill_missing_dates(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """
    Ensure the trend DataFrame has an entry for every day in the past `days`.
    Missing days get filled with neutral/0 values so charts don't have gaps.

    Args:
        df   : Output from get_daily_emotion_trend().
        days : How many days back to fill.

    Returns:
        A complete DataFrame with one row per day for the past `days`.
    """
    if df.empty:
        date_range = pd.date_range(
            start=datetime.now() - timedelta(days=days - 1),
            periods=days,
            freq="D"
        )
        return pd.DataFrame({
            "date":             date_range,
            "dominant_emotion": ["neutral"] * days,
            "avg_score":        [0.0] * days,
            "message_count":    [0] * days,
        })

    date_range = pd.date_range(
        start=datetime.now() - timedelta(days=days - 1),
        end=datetime.now(),
        freq="D"
    )
    full_df = pd.DataFrame({"date": date_range})
    full_df["date"] = pd.to_datetime(full_df["date"]).dt.normalize()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    merged = full_df.merge(df, on="date", how="left")
    merged["dominant_emotion"] = merged["dominant_emotion"].fillna("neutral")
    merged["avg_score"]        = merged["avg_score"].fillna(0.0)
    merged["message_count"]    = merged["message_count"].fillna(0)

    return merged.reset_index(drop=True)
