# 🤖 NexusHire — Enterprise AI Interview Platform

> **An intelligent, voice-enabled interview system that dynamically generates and evaluates technical questions using the Groq API (LLaMA 3.3), Whisper speech-to-text, and ElevenLabs / gTTS for voice synthesis.**

---

## 🧭 Overview

**NexusHire** is an interactive, end-to-end AI interview platform designed for **enterprise-level candidate evaluation**.

Upload your resume, select your target role, and begin a live interview session where the AI:
- Reads your resume & job description  
- Dynamically generates contextual, role-based interview questions  
- Evaluates your responses in real-time using structured rubrics (relevance, depth, clarity, correctness)  
- Provides immediate scoring, AI feedback, and tailored follow-up questions  
- Creates a **final report** with hiring recommendations  

All in a sleek, enterprise-grade web interface. ✨

---

## 🚀 Key Features

| Type | Feature | Description |
|------|----------|-------------|
| 🧠 **Dynamic AI Generation** | Uses Groq’s **LLaMA 3.3 70B Versatile** to generate context-aware interview questions and feedback |
| 📄 **Resume-Aware** | Parses uploaded resumes (PDF, DOCX, TXT) to tailor interview flow |
| 🎙️ **Voice Interaction** | Supports **microphone-based voice answers** (through browser) |
| 🗣️ **TTS Playback** | Plays spoken version of each question using ElevenLabs or gTTS |
| ⚙️ **Structured Evaluation** | Evaluates correctness, clarity, relevance, and depth on a 1–10 scale |
| 🧾 **Final Report** | Generates detailed PDF-friendly reports highlighting strengths and gaps |
| 🧩 **Frontend + API** | Single-page frontend (HTML/CSS/JS) with FastAPI backend |
| 💬 **Self-Learning Flow** | Follows up on weak areas with dynamic probes or example-seeking questions |

---

## 🛠️ Tech Stack

| Category | Technology |
|-----------|-------------|
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Backend** | Python 3.11+, [FastAPI](https://fastapi.tiangolo.com/) |
| **AI Model** | [Groq API](https://groq.com) — `llama-3.3-70b-versatile` |
| **Audio** | Whisper (Speech Recognition) + ElevenLabs / gTTS (Text-to-Speech) |
| **Database** | (Optional) SQLite for session caching |
| **Deployment** | Render (Docker, free tier) |
| **Dependencies** | `fastapi`, `groq`, `pydub`, `aiofiles`, `python-dotenv`, `uvicorn` |

---

## ⚙️ How It Works

### 1️⃣ Resume Processing
- PDF/DOCX/TXT resume is uploaded via the frontend.
- FastAPI extracts & anonymizes the text for privacy.
- Key information (skills, experience) is passed to the Groq model to initialize context.

### 2️⃣ Dynamic Question Generation
- Every new question is generated dynamically by Groq based on:
  - Past responses
  - Resume content
  - Current interview phase (Intro, Technical, Behavioral, Analytical)
- Question includes a rationale tag (“Why this question?”).

### 3️⃣ Real-Time Evaluation
When you answer:
- If **typed** → Text sent directly to model.
- If **spoken** → Browser records via `MediaRecorder`, encoded in Base64, and sent to server → Whisper transcribes.
- Groq evaluates response → returns:
  ```json
  {
    "overall": 8.2,
    "feedback": "Clear example with measurable impact.",
    "strengths": ["Strong structure"],
    "weaknesses": ["Could mention trade-offs"],
    "probe": "How would you optimize this design for scalability?"
  }
  ```

### 4️⃣ Voice Feedback & Follow-up
- ElevenLabs or fallback **gTTS** converts the follow-up question or feedback into playable audio.
- UI displays evaluation card + “Next Question” option.

### 5️⃣ Final Report Generation
At the end:
- Aggregates all answers, strengths, weaknesses
- Calls Groq once more to produce a hiring recommendation summary
- Shows dynamic charts, scores, and next steps

---

## 💻 Getting Started Locally

Clone and run the app locally.

### 1️⃣ Clone this repo
```bash
git clone https://github.com/abhaythegamer2-cmd/nexushire.git
cd nexushire
```

### 2️⃣ Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate         # On Windows
# OR
source venv/bin/activate      # On Mac/Linux
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Create `.env` file
```bash
GROQ_API_KEY=your_groq_api_key_here
ELEVEN_API_KEY=your_elevenlabs_api_key_here
```

### 5️⃣ Start FastAPI server
```bash
uvicorn api_server:app --reload
```

Server runs at → [http://localhost:8000](http://localhost:8000)

### 6️⃣ Open Frontend
Now simply open your local `index.html` in any modern browser.

---


## 🧩 Project Structure


nexushire/
│
├── api_server.py         # FastAPI server (main entry)
├── your_agent.py         # AI logic, Groq integration, evaluation engine
├── index.html            # Frontend interface
├── requirements.txt      # Dependencies
├── Dockerfile            # Render / Docker deployment
├── .env.example          # Environment config sample
└── README.md             # You're here :)

---

## 🧠 Understanding the AI Logic

| Stage | Function | Description |
|--------|-----------|-------------|
| `evaluate_answer()` | Uses Groq model to rate and explain responses |
| `_safe_evaluate()` | Adds retry guards + fallback evaluation models |
| `generate_dynamic_question()` | Produces next question based on prior feedback |
| `gen_audio_tts()` | Uses ElevenLabs or gTTS for spoken follow-ups |
| `extract_pdf_text()` | Cleans and anonymizes uploaded resume |
| `_robust_parse()` | Safely parses model responses (handles invalid JSON) |

---

## 📊 Architecture Diagram


[Browser UI]
↓
Submit voice/text
↓
[FastAPI Backend]
├── Groq 70B → Evaluates answer
├── Whisper → Converts voice→text
├── ElevenLabs/gTTS → Voice output
└── Aggregates scores → Report
↓
[Rendered report sent back to user]

---

## 🧠 AI Notes

- **Groq** models are extremely fast and cost-efficient (≈3¢ per 1000 tokens)
- Audio handled locally or via **gTTS**
- No external database needed — all in-memory
- Fully open-source and free to fork

---

## 💡 Ideas for Future Enhancements

- 👥 Multi-interviewer panel simulation  
- 💼 Multi-language interview support  
- 📊 Analytics dashboard for hiring teams  
- 🧾 Export reports as styled PDFs  
- 🧠 Fine-tuned question generation by company/position type  
- ⚙️ Integration with ATS (Greenhouse / Lever)

---

## 🧑‍💻 Author

**Abhay [@abhaythegamer2-cmd](https://github.com/abhaythegamer2-cmd)**  
🤖 Passionate about AI automation, backend systems, and intelligent recruiting tools.

---

## 🪪 License

MIT License © 2026 — [Abhay](https://github.com/abhaythegamer2-cmd)  

Feel free to fork, modify, and deploy your own version of NexusHire.  
Just credit the original repo if you go public!

---

## ⭐ Support the Project

If **NexusHire** helped you learn or build something cool:
- ⭐ Star this repo on GitHub
- 🍴 Fork it and share your custom interviews
- ☕ [Buy Abhay a coffee](https://buymeacoffee.com) to keep AI projects open-source

---

> _"Let AI handle the interviews, so you can focus on hiring great humans."_ 💼
