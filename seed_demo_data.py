"""
seed_demo_data.py
=================
Seeds the SQLite database with sample data from data/sample_data.csv.
Run this to populate the dashboard with demo trends before presenting.

Usage:
    python seed_demo_data.py
"""

import sqlite3
import pandas as pd
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from backend.database import init_db, DB_PATH

def seed():
    """Load sample CSV and insert into chat_logs table."""
    csv_path = os.path.join("data", "sample_data.csv")

    if not os.path.exists(csv_path):
        print(f"[SEED] Sample data not found at {csv_path}")
        return

    print("[SEED] Initializing database...")
    init_db()

    df = pd.read_csv(csv_path)
    print(f"[SEED] Loaded {len(df)} sample records from CSV.")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    # Create demo session
    session_id = "demo-session-001"
    conn.execute("""
        INSERT OR IGNORE INTO sessions (session_id, started_at, last_active)
        VALUES (?, '2024-03-10T09:00:00', '2024-03-15T08:00:00')
    """, (session_id,))

    # Insert each row as a 'user' message
    for _, row in df.iterrows():
        conn.execute("""
            INSERT OR IGNORE INTO chat_logs
                (session_id, role, message, emotion, emotion_score, risk_level, timestamp)
            VALUES (?, 'user', ?, ?, ?, ?, ?)
        """, (
            row["session_id"],
            row["message"],
            row["emotion"],
            float(row["emotion_score"]),
            row["risk_level"],
            row["timestamp"],
        ))

    conn.commit()
    conn.close()
    print(f"[SEED] ✅ Demo data seeded successfully into {DB_PATH}")
    print("[SEED] Start the app and use session ID: demo-session-001 to see trends.")


if __name__ == "__main__":
    seed()
