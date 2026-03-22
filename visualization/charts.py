"""
charts.py
=========
All Matplotlib / Seaborn chart generation for the dashboard.
Each function returns a matplotlib Figure that Streamlit renders via st.pyplot().

Charts:
    1. emotion_distribution_pie   → Donut chart of emotion breakdown
    2. emotion_trend_line         → Daily emotion trend over time
    3. risk_level_bar             → Risk distribution bar chart
    4. emotion_heatmap_by_hour    → When during the day emotions peak
    5. session_score_line         → Emotion intensity over the session
"""

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend; required for server-side rendering

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from datetime import datetime

from backend.data_processor import (
    EMOTION_COLORS, RISK_COLORS, EMOTION_ORDER,
    get_emotion_distribution, get_daily_emotion_trend,
    get_risk_summary, fill_missing_dates, records_to_dataframe,
)

# ── Global plot style ─────────────────────────────────────────────────────────
DARK_BG     = "#0F0F1A"
CARD_BG     = "#1A1A2E"
TEXT_COLOR  = "#E8E8F0"
ACCENT      = "#6C63FF"
GRID_COLOR  = "#2A2A40"

def _apply_dark_style(fig, ax_or_axes):
    """Apply a consistent dark theme to a figure and its axes."""
    fig.patch.set_facecolor(DARK_BG)
    axes = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
    for ax in np.array(axes).flatten():
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        if hasattr(ax, "title"):
            ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)


# ── Chart 1: Emotion Distribution Donut ──────────────────────────────────────

def emotion_distribution_pie(emotion_records: list[dict]) -> plt.Figure:
    """
    Donut chart showing percentage breakdown of detected emotions.

    Args:
        emotion_records: List of dicts from database.get_emotion_history().

    Returns:
        Matplotlib Figure.
    """
    df = records_to_dataframe(emotion_records)
    dist = get_emotion_distribution(df)

    fig, ax = plt.subplots(figsize=(5, 4))
    _apply_dark_style(fig, ax)

    if dist.empty or dist["count"].sum() == 0:
        ax.text(0.5, 0.5, "No data yet\nSend some messages!", ha="center",
                va="center", color=TEXT_COLOR, fontsize=13, transform=ax.transAxes)
        ax.axis("off")
        fig.suptitle("Emotion Distribution", color=TEXT_COLOR, fontsize=13, fontweight="bold")
        return fig

    # Filter out zero-count emotions
    dist = dist[dist["count"] > 0]
    colors = [EMOTION_COLORS.get(e, "#888") for e in dist["emotion"]]

    wedges, texts, autotexts = ax.pie(
        dist["count"],
        labels=dist["emotion"].str.capitalize(),
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"width": 0.55, "edgecolor": DARK_BG, "linewidth": 2},
    )

    for text in texts + autotexts:
        text.set_color(TEXT_COLOR)
        text.set_fontsize(9)

    ax.set_title("Emotion Distribution", color=TEXT_COLOR, fontsize=13,
                 fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


# ── Chart 2: Emotion Trend Line (7 days) ─────────────────────────────────────

def emotion_trend_line(emotion_records: list[dict], days: int = 7) -> plt.Figure:
    """
    Line chart showing how emotional intensity has changed over the past N days.
    One line per detected emotion category.

    Args:
        emotion_records: List of dicts from database.get_emotion_history().
        days           : Number of days to display.

    Returns:
        Matplotlib Figure.
    """
    df = records_to_dataframe(emotion_records)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    _apply_dark_style(fig, ax)

    if df.empty:
        ax.text(0.5, 0.5, "No trend data yet.\nKeep chatting to see your emotional journey!",
                ha="center", va="center", color=TEXT_COLOR, fontsize=11,
                transform=ax.transAxes)
        ax.axis("off")
        fig.suptitle(f"Emotion Trend (Last {days} Days)", color=TEXT_COLOR,
                     fontsize=13, fontweight="bold")
        return fig

    # Pivot: date × emotion → avg score
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    pivot = (
        df.groupby(["date", "emotion"])["emotion_score"]
        .mean()
        .unstack(fill_value=0)
        .reset_index()
    )

    for emotion in EMOTION_ORDER:
        if emotion in pivot.columns:
            ax.plot(
                pivot["date"],
                pivot[emotion],
                marker="o",
                markersize=5,
                linewidth=2,
                color=EMOTION_COLORS[emotion],
                label=emotion.capitalize(),
            )

    ax.set_xlabel("Date", color=TEXT_COLOR)
    ax.set_ylabel("Avg. Intensity Score", color=TEXT_COLOR)
    ax.set_title(f"Emotion Trend — Last {days} Days", color=TEXT_COLOR,
                 fontsize=13, fontweight="bold")

    # Format x-axis dates
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.tick_params(axis="x", labelsize=8)

    legend = ax.legend(
        loc="upper left", framealpha=0.2, labelcolor=TEXT_COLOR,
        facecolor=CARD_BG, edgecolor=GRID_COLOR, fontsize=8
    )
    fig.tight_layout()
    return fig


# ── Chart 3: Risk Level Bar Chart ────────────────────────────────────────────

def risk_level_bar(emotion_records: list[dict]) -> plt.Figure:
    """
    Horizontal bar chart of risk level distribution.

    Args:
        emotion_records: List of dicts from database.get_emotion_history().

    Returns:
        Matplotlib Figure.
    """
    df = records_to_dataframe(emotion_records)
    summary = get_risk_summary(df)

    fig, ax = plt.subplots(figsize=(5, 2.5))
    _apply_dark_style(fig, ax)

    levels  = ["LOW", "MEDIUM", "HIGH"]
    counts  = [summary[l] for l in levels]
    colors  = [RISK_COLORS[l] for l in levels]

    bars = ax.barh(levels, counts, color=colors, height=0.5, edgecolor=DARK_BG)

    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(
                bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", color=TEXT_COLOR, fontsize=10
            )

    ax.set_xlabel("Number of Messages", color=TEXT_COLOR)
    ax.set_title("Risk Level Distribution", color=TEXT_COLOR,
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(counts) * 1.3 if max(counts) > 0 else 5)
    fig.tight_layout()
    return fig


# ── Chart 4: Session Intensity Timeline ──────────────────────────────────────

def session_intensity_line(emotion_records: list[dict]) -> plt.Figure:
    """
    Scatter + line showing emotion intensity throughout the current chat session.
    Each point is one user message, colored by emotion.

    Args:
        emotion_records: List of dicts from database.get_emotion_history().

    Returns:
        Matplotlib Figure.
    """
    df = records_to_dataframe(emotion_records)

    fig, ax = plt.subplots(figsize=(7, 3))
    _apply_dark_style(fig, ax)

    if df.empty or len(df) < 2:
        ax.text(0.5, 0.5, "Keep chatting to see your session timeline!",
                ha="center", va="center", color=TEXT_COLOR, fontsize=11,
                transform=ax.transAxes)
        ax.axis("off")
        fig.suptitle("Session Emotion Intensity", color=TEXT_COLOR,
                     fontsize=13, fontweight="bold")
        return fig

    x      = range(len(df))
    scores = df["emotion_score"].values
    colors = [EMOTION_COLORS.get(e, "#888") for e in df["emotion"]]

    ax.plot(x, scores, color=ACCENT, linewidth=1.5, alpha=0.4, zorder=1)
    ax.scatter(x, scores, c=colors, s=60, zorder=2, edgecolors=DARK_BG, linewidth=0.5)

    # Legend patches
    present_emotions = df["emotion"].unique()
    patches = [
        mpatches.Patch(color=EMOTION_COLORS.get(e, "#888"), label=e.capitalize())
        for e in present_emotions
    ]
    ax.legend(handles=patches, loc="upper right", framealpha=0.2,
              labelcolor=TEXT_COLOR, facecolor=CARD_BG, edgecolor=GRID_COLOR,
              fontsize=8)

    ax.set_xlabel("Message #", color=TEXT_COLOR)
    ax.set_ylabel("Emotion Score", color=TEXT_COLOR)
    ax.set_title("Session Emotion Intensity", color=TEXT_COLOR,
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    return fig
