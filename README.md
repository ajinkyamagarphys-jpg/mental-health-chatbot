# 🧠 MindEase — Personalized Mental Health Chatbot

> An AI-powered emotional support chatbot using NLP for real-time emotion detection,  
> risk assessment, empathetic responses, and mood trend visualization.

---

## 📁 Project Structure

```
mental_health_chatbot/
│
├── app.py                    ← Main Streamlit UI (chat + dashboard)
│
├── backend/
│   ├── __init__.py
│   ├── api.py                ← FastAPI REST backend (optional, standalone mode)
│   ├── nlp_engine.py         ← HuggingFace emotion detection
│   ├── risk_detector.py      ← Keyword + NLP risk classification (LOW/MEDIUM/HIGH)
│   ├── response_engine.py    ← Empathetic response generation
│   ├── database.py           ← SQLite CRUD operations
│   └── data_processor.py     ← Pandas aggregation & trend analysis
│
├── visualization/
│   ├── __init__.py
│   └── charts.py             ← Matplotlib chart generators for dashboard
│
├── utils/
│   ├── __init__.py
│   └── helpers.py            ← Shared utility functions
│
├── data/
│   ├── sample_data.csv       ← Demo dataset for seeding
│   └── chatbot.db            ← SQLite database (auto-created on first run)
│
├── .streamlit/
│   └── config.toml           ← Dark theme configuration
│
├── seed_demo_data.py          ← Script to populate DB with sample data
├── requirements.txt
└── README.md
```

---

## 🔧 System Flow (Pipeline)

```
User Input
    ↓
Streamlit Chat UI (app.py)
    ↓
NLP Emotion Detection (nlp_engine.py)
  → HuggingFace: j-hartmann/emotion-english-distilroberta-base
  → Output: {emotion, confidence_score}
    ↓
Risk Classification (risk_detector.py)
  → Keyword scan + emotion severity
  → Output: LOW / MEDIUM / HIGH
    ↓
Response Generation (response_engine.py)
  → Empathetic opening + coping strategy + follow-up
    ↓
Persist to SQLite (database.py)
  → session_id, message, emotion, risk_level, timestamp
    ↓
Pandas Aggregation (data_processor.py)
  → Daily trends, distribution stats, session summary
    ↓
Matplotlib Charts (charts.py)
  → Donut, trend line, bar, intensity scatter
    ↓
Streamlit Dashboard Display
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10 or 3.11
- pip
- ~2 GB disk space (for HuggingFace model download)
- Internet connection on first run (model auto-downloads)

---

### Step 1 — Clone / Download the project

```bash
# If using git:
git clone <your-repo-url>
cd mental_health_chatbot

# Or simply unzip the project folder and cd into it
```

---

### Step 2 — Create a virtual environment

```bash
# Create venv
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> ⏱️ First install takes 3–5 minutes (PyTorch + Transformers are large).

---

### Step 4 — (Optional) Seed demo data

Populate the database with sample emotion records so the dashboard  
shows meaningful charts right away:

```bash
python seed_demo_data.py
```

---

### Step 5 — Run the application

#### Option A: Streamlit only (Recommended for hackathon demo)

The app runs fully standalone — no separate backend needed.

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

#### Option B: Full stack (Streamlit + FastAPI)

Run both services in separate terminals:

**Terminal 1 — FastAPI backend:**
```bash
uvicorn backend.api:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

**Terminal 2 — Streamlit frontend:**
```bash
# In app.py, set USE_API = True at the top of the file
streamlit run app.py
```

---

## 🌐 Deploying to Streamlit Cloud

1. Push code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set **Main file path** to `app.py`
4. Add requirements.txt — Streamlit Cloud installs automatically
5. Click **Deploy** 🚀

> Note: The HuggingFace model downloads automatically on first cold start (~60 sec on cloud).

---

## 🔌 FastAPI Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/session/new` | Create session, get greeting |
| POST | `/chat` | Process message → emotion + response |
| GET | `/history/{session_id}` | Fetch chat history |
| GET | `/emotions/{session_id}` | Fetch emotion records for charts |
| GET | `/summary/{session_id}` | Get session mood summary |

**Example API call:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123", "message": "I feel so anxious today"}'
```

---

## 🎭 Emotion Labels

| HuggingFace Label | Mapped To |
|---|---|
| joy | happiness 😊 |
| sadness | sadness 😢 |
| fear | anxiety 😰 |
| anger | anger 😠 |
| disgust | anger 😠 |
| neutral | neutral 😐 |
| surprise | neutral 😐 |

---

## 🚦 Risk Levels

| Level | Criteria | Action |
|---|---|---|
| 🟢 LOW | No distress signals | Normal empathetic response |
| 🟡 MEDIUM | Distress keywords OR strong negative emotion | Coping strategies + warm note |
| 🔴 HIGH | Self-harm keywords OR crisis language | Crisis resources shown immediately |

---

## 💡 Key Design Decisions

- **Standalone mode**: Works without FastAPI running — ideal for demos
- **Model singleton**: NLP model loaded once at startup, not per request
- **Session tracking**: UUID-based sessions stored in SQLite
- **Dark theme**: Calming purple/dark palette appropriate for mental health context
- **Varied responses**: Random selection from response pools avoids robotic repetition

---

## ⚠️ Disclaimer

This application is a **hackathon prototype** and an **early support/detection tool only**.  
It is **NOT** a replacement for licensed therapists, psychiatrists, or mental health professionals.

If you or someone you know is in crisis:
- 🇮🇳 **iCall (India):** 9152987821
- 🇺🇸 **988 Suicide & Crisis Lifeline (US):** Call or text 988
- 🌍 **International:** https://www.iasp.info/resources/Crisis_Centres/

---

## 📄 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend API | FastAPI + Uvicorn |
| NLP Model | HuggingFace Transformers (DistilRoBERTa) |
| Database | SQLite (built-in Python) |
| Data Processing | Pandas + NumPy |
| Visualization | Matplotlib + Seaborn |
| Deployment | Streamlit Cloud |
