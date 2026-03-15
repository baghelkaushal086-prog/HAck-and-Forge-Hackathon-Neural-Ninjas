# api_server.py – v4.5  (Speed Optimized)
import os, io, re, json, uuid, base64, tempfile, traceback, asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

# ── Load .env ─────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "")
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY missing – add it to .env")

# ── Agent imports ─────────────────────────────────────────────
from your_agent import (
    extract_pdf_text, evaluate_answer, generate_probe,
    generate_final_assessment, init_db,
    ROLE_RUBRICS, LANGUAGE_CODES,
    PROBE_THRESHOLD, MAX_QUESTIONS, WHISPER_MODEL,
)

# ─────────────────────────────────────────────────────────────
# SPEED FIX 1 — Singleton Groq Client
# Created ONCE when the server starts, reused for every request.
# Previously: a new client + a test API call was made per session (~2s wasted).
# ─────────────────────────────────────────────────────────────
_GROQ_CLIENT: Optional[Groq] = None

def get_groq_client() -> Groq:
    global _GROQ_CLIENT
    if _GROQ_CLIENT is None:
        _GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)
        print("✓ Groq singleton initialized")
    return _GROQ_CLIENT

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="Enterprise AI Interview API", version="4.5")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="."), name="static")

@app.on_event("startup")
async def on_startup():
    """
    Pre-initialize everything at server boot — not on first request.
    This means the first user gets a fast experience, not a cold-start penalty.
    """
    try:
        import pygame
        pygame.mixer.init()
    except Exception:
        pass
    get_groq_client()   # warm up singleton
    print("✅ Server fully warmed up and ready")

@app.get("/")
async def root():
    return FileResponse("index.html")

# ── Session store + thread pool ───────────────────────────────
sessions: Dict[str, Dict[str, Any]] = {}
_pool    = ThreadPoolExecutor(max_workers=10)

async def _bg(fn, *args):
    """Run any blocking function on the thread pool (never blocks the event loop)."""
    return await asyncio.get_event_loop().run_in_executor(_pool, fn, *args)


# ─────────────────────────────────────────────────────────────
# SPEED FIX 2 — Two-phase question generation
#
# Phase A  (instant, fast model):  1 questi_safe_evaluateon  → shown immediately
# Phase B  (background, best model): 4 more Qs → ready by the time
#           the user finishes answering question 1
# ─────────────────────────────────────────────────────────────

def _gen_first_question(client: Groq, resume: str, position: str,
                         language: str, role: str) -> Dict:
    """
    Uses llama-3.1-8b-instant (fastest Groq model) to produce a single
    interview question in ~0.4s instead of ~4s for the full set.
    """
    rubric = ROLE_RUBRICS.get(role, ROLE_RUBRICS["backend"])
    prompt = (
        f"Generate 1 interview question for a {position} ({rubric['name']}).\n"
        f"Focus area: {rubric['focus']}\n"
        f"Resume snippet: {resume[:400]}\n\n"
        "Reply with ONLY this JSON (no other text):\n"
        '{"question":"your question here?","golden_answer":"ideal brief answer"}'
    )
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # fastest Groq model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=180,
            temperature=0.3,
        )
        data = _robust_parse(resp.choices[0].message.content)
        if isinstance(data, dict) and "question" in data:
            return data
    except Exception as e:
        print(f"Fast question gen error: {e}")

    # Fallback — never leave the user waiting with an error
    return {
        "question":      f"Walk me through your most relevant experience for the {position} role.",
        "golden_answer": "Strong answer demonstrates specific, measurable experience.",
    }
# ─────────────────────────────────────────────────────────────
# DYNAMIC QUESTION GENERATOR
# Reads full conversation history to craft a targeted next question.
# Runs on the fast 8B model alongside evaluation (parallel).
# ─────────────────────────────────────────────────────────────
def _gen_dynamic_question(
    client,
    results_history: List[Dict],   # all previous Q+A+eval entries
    current_answer:  str,          # the answer just submitted (not yet evaluated)
    position:        str,
    role:            str,
    language:        str,
    q_num:           int,
) -> Dict:
    """
    Generates the next question based on:
      - Every previous question and answer
      - Identified weak and strong areas from evaluations
      - The current (unevaluated) answer text
    Returns: {question, golden_answer, rationale}
    """
    rubric = ROLE_RUBRICS.get(role, ROLE_RUBRICS["backend"])

    # Build rich context from full history
    history_lines = []
    weak_areas    = []
    strong_areas  = []

    for i, entry in enumerate(results_history):
        score      = entry.get("eval", {}).get("overall", 5)
        weaknesses = entry.get("eval", {}).get("weaknesses", [])
        strengths  = entry.get("eval", {}).get("strengths",  [])
        weak_areas.extend(weaknesses)
        strong_areas.extend(strengths)

        history_lines.append(
            f"Q{i+1}: {entry['question']}\n"
            f"Answer ({entry.get('word_count',0)} words, {score}/10): "
            f"{entry['answer'][:300]}\n"
            f"Strengths: {', '.join(strengths) if strengths else 'none'}\n"
            f"Weaknesses: {', '.join(weaknesses) if weaknesses else 'none'}"
        )

    # Append the current unevaluated answer so the next Q can reference it
    if current_answer.strip():
        history_lines.append(
            f"Q{len(results_history)+1} (just answered, not yet scored):\n"
            f"Answer: {current_answer[:300]}"
        )

    history_str  = "\n\n".join(history_lines)
    weak_str     = ", ".join(dict.fromkeys(weak_areas))  or "none yet"
    covered      = " | ".join(e["question"][:60] for e in results_history)

    prompt = f"""You are a senior interviewer hiring for: {position} ({rubric['name']}).

FULL CONVERSATION HISTORY:
{history_str}

RECURRING WEAK AREAS: {weak_str}
SKILLS TO COVER: {rubric['focus']}
QUESTIONS ALREADY ASKED (DO NOT REPEAT THESE TOPICS): {covered}

Your task: Generate question #{q_num} of the interview.

Rules:
1. If weak areas exist → probe one of them specifically (ask for metrics, examples, outcomes)
2. If candidate mentioned something vague → ask them to elaborate with specifics
3. If candidate did well → move to a harder, related topic from the focus skills
4. NEVER repeat a topic already covered
5. Each question should feel like a natural follow-up to what was just said
6. Language: {language}

Return ONLY this JSON (no other text):
{{
  "question": "your question here?",
  "golden_answer": "ideal 2-3 sentence answer",
  "rationale": "one sentence: why you chose this question"
}}"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # fast model — runs in parallel with eval
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.5,
        )
        data = _robust_parse(resp.choices[0].message.content)
        if isinstance(data, dict) and "question" in data:
            print(f"🎯 Dynamic Q{q_num}: {data['question'][:70]}")
            print(f"   Rationale: {data.get('rationale','')[:80]}")
            return data
    except Exception as e:
        print(f"Dynamic Q gen error: {e}")

    # Role-specific fallback questions
    fallbacks = {
        "backend":      "Can you walk me through how you'd design a system to handle 10x its current traffic?",
        "frontend":     "How do you approach performance optimization when a page feels slow to users?",
        "pm":           "How do you handle a situation where engineering says a feature will take 3x longer than expected?",
        "data_science": "How do you detect and handle data drift in a deployed ML model?",
        "devops":       "Walk me through your ideal zero-downtime deployment strategy for a critical microservice.",
    }
    return {
        "question":      fallbacks.get(role, f"What's the hardest problem you solved as a {rubric['name']}?"),
        "golden_answer": "Strong answers include specific metrics, clear decisions, and measurable outcomes.",
        "rationale":     "Exploring a core competency for this role",
    }

@app.post("/api/session/start")
async def start_session(
    pdf:      UploadFile = File(...),
    role:     str        = Form("backend"),
    language: str        = Form("en-US"),
    position: str        = Form("Software Engineer"),
):
    try:
        pdf_bytes = await pdf.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes); tmp = f.name

        client      = get_groq_client()
        resume_text = await _bg(extract_pdf_text, tmp)
        os.unlink(tmp)

        # Generate only Q1 with the fast model — subsequent Qs are dynamic
        first_q_data = await _bg(
            _gen_first_question, client, resume_text, position, language, role
        )
        first_q_text = first_q_data["question"]

        # TTS for Q1
        audio_b64 = await _bg(generate_tts_bytes, first_q_text, language)

        sid = str(uuid.uuid4())
        sessions[sid] = {
            "client":           client,
            "questions":        [first_q_data],  # grows dynamically each round
            "results":          [],
            "config": {
                "position":     position,
                "role":         role,
                "language":     language,
                "resume_text":  resume_text,
            },
            "current_idx":      0,
            "probe_count":      0,
            "question_history": [first_q_text],
            "total_asked":      0,
            "db_conn":          init_db(),
            # NOTE: NO questions_ready flag — questions grow on-demand
        }

        return {
            "session_id":      sid,
            "question":        first_q_text,
            "question_index":  0,
            "total_questions": MAX_QUESTIONS,
            "audio_base64":    audio_b64,
            "is_probe":        False,
            "rationale":       first_q_data.get("rationale", "Opening question tailored to your resume"),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


class AnswerPayload(BaseModel):
    session_id:   str
    answer_text:  Optional[str] = None
    audio_base64: Optional[str] = None
    is_voice:     bool          = False



@app.post("/api/session/answer")
async def submit_answer(payload: AnswerPayload):
    sid = payload.session_id
    if sid not in sessions:
        raise HTTPException(404, "Session not found or expired.")

    session = sessions[sid]
    client  = session["client"]
    lang    = session["config"]["language"]
    role    = session["config"]["role"]
    q_idx   = session["current_idx"]
    q_data  = session["questions"][q_idx]

    # ── Resolve answer text ───────────────────────────────────
    answer = payload.answer_text
    if payload.is_voice and payload.audio_base64:
        audio  = base64.b64decode(payload.audio_base64)
        answer = await _bg(_transcribe, client, audio, lang)
        print(f"✓ Whisper: {answer[:80]}")

    if not answer:
        raise HTTPException(400, "No answer provided.")

    words           = len(answer.split())
    results_snapshot = list(session["results"])  # history BEFORE this answer
    will_continue   = session["total_asked"] + 1 < MAX_QUESTIONS

    # ── PARALLEL: Evaluate + Speculatively generate next question ─
    #
    # Why parallel:
    #   - Evaluation uses the 70B model      → ~2s
    #   - Dynamic Q gen uses the 8B model    → ~0.5s
    #   - Running together saves ~0.5s and
    #     the speculative Q is good enough
    #     because it sees the full answer text
    #     even without the formal eval scores.
    #
    if will_continue:
        evaluation, speculative_q = await asyncio.gather(
            _bg(_safe_evaluate, client, q_data, answer,
                lang, role, words, payload.is_voice),
            _bg(_gen_dynamic_question, client, results_snapshot, answer,
                session["config"]["position"], role, lang,
                session["total_asked"] + 2),
        )
    else:
        evaluation    = await _bg(_safe_evaluate, client, q_data, answer,
                                lang, role, words, payload.is_voice)
        speculative_q = None

    print(f"✅ Eval: {evaluation['overall']}/10")

    # ── Store result ──────────────────────────────────────────
    session["results"].append({
        "question":   q_data["question"],
        "answer":     answer,
        "eval":       evaluation,
        "word_count": words,
        "is_voice":   payload.is_voice,
    })
    session["total_asked"] += 1
    session["current_idx"] += 1

    response: Dict[str, Any] = {
        "evaluation":     evaluation,
        "next_question":  None,
        "next_rationale": None,
        "audio_base64":   None,
        "is_probe":       False,
        "question_index": q_idx,
        "done":           False,
    }

    # ── Check if interview is complete ────────────────────────
    if session["total_asked"] >= MAX_QUESTIONS:
        assessment = await _bg(
            generate_final_assessment, client,
            session["config"]["position"],
            session["config"]["resume_text"],
            session["results"], lang,
        )
        response.update({"done": True, "assessment": assessment})
        return response

    # ── Probe check (low score → targeted follow-up) ──────────
    score = evaluation.get("overall", 10)
    if score < PROBE_THRESHOLD and session["probe_count"] < 2:
        probe_q = await _bg(
            generate_probe, client, q_data["question"],
            answer, evaluation, lang, session["question_history"],
        )
        if probe_q:
            session["probe_count"] += 1
            session["question_history"].append(probe_q)

            # Add probe to the questions list so indexing stays consistent
            session["questions"].append({
                "question":      probe_q,
                "golden_answer": "N/A",
                "rationale":     "Probing a weak area from your previous answer",
            })

            audio = await _bg(generate_tts_bytes, probe_q, lang)
            response.update({
                "next_question":  probe_q,
                "next_rationale": "I'd like to dig deeper into something from your last answer.",
                "audio_base64":   audio,
                "is_probe":       True,
                "question_index": session["current_idx"],
            })
            return response

    # ── Use the speculatively generated dynamic question ──────
    # The speculative Q was generated in parallel with eval, so it's
    # already ready — no additional wait needed.
    if speculative_q and speculative_q.get("question"):
        next_q_data = speculative_q
    else:
        # Edge case: speculative generation failed → generate now
        print("⚠️  Speculative Q failed → generating on-demand")
        next_q_data = await _bg(
            _gen_dynamic_question, client, session["results"], "",
            session["config"]["position"], role, lang, session["total_asked"] + 1,
        )

    next_q_text = next_q_data["question"]
    session["question_history"].append(next_q_text)
    session["questions"].append(next_q_data)  # <-- dynamic growth

    audio = await _bg(generate_tts_bytes, next_q_text, lang)
    response.update({
        "next_question":  next_q_text,
        "next_rationale": next_q_data.get("rationale", ""),
        "audio_base64":   audio,
        "question_index": session["current_idx"],
    })

    return response
# ─────────────────────────────────────────────────────────────
# TTS  —  ElevenLabs SDK ≥ 1.x  →  gTTS fallback
# ─────────────────────────────────────────────────────────────
def generate_tts_bytes(text: str, language: str) -> str:
    """Blocking — returns base64 MP3. Always call via _bg()."""
    lang_code = LANGUAGE_CODES.get(language.split("-")[0], "en")

    if ELEVEN_API_KEY:
        try:
            from elevenlabs.client import ElevenLabs
            from elevenlabs import VoiceSettings
            eleven   = ElevenLabs(api_key=ELEVEN_API_KEY)
            response = eleven.text_to_speech.convert(
                voice_id="pNInz6obpgDQGcFmaJgB",
                text=text,
                model_id="eleven_multilingual_v2",
                voice_settings=VoiceSettings(
                    stability=0.6, similarity_boost=0.8,
                    style=0.2, use_speaker_boost=True,
                ),
            )
            audio = response if isinstance(response, bytes) else b"".join(response)
            return base64.b64encode(audio).decode()
        except Exception as e:
            print(f"⚠️ ElevenLabs → gTTS fallback: {e}")

    from gtts import gTTS
    buf = io.BytesIO()
    gTTS(text, lang=lang_code).write_to_fp(buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ─────────────────────────────────────────────────────────────
# ROBUST JSON PARSER  (4 strategies, no "Parse failed")
# ─────────────────────────────────────────────────────────────
def _robust_parse(text: str) -> Any:
    if not isinstance(text, str):
        text = str(text)

    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()

    # Strategy 1 – direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2 – first {...} or [...] block
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

    # Strategy 3 – fix trailing commas / single quotes
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", cleaned).replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 4 – regex field extraction
    result: Dict[str, Any] = {}
    for field in ("overall", "correctness", "depth", "clarity"):
        m = re.search(rf'"{field}"\s*:\s*(\d+(?:\.\d+)?)', text)
        if m:
            result[field] = float(m.group(1))
    for field in ("feedback", "probe"):
        m = re.search(rf'"{field}"\s*:\s*"([^"]*)"', text)
        if m:
            result[field] = m.group(1)
    for field in ("strengths", "weaknesses"):
        m = re.search(rf'"{field}"\s*:\s*\[([^\]]*)\]', text)
        result[field] = re.findall(r'"([^"]+)"', m.group(1)) if m else []

    if "overall" in result:
        return result

    return {
        "overall":    5,
        "feedback":   "Answer received. Evaluation service returned an unreadable response.",
        "strengths":  [],
        "weaknesses": ["Parsing error – not a reflection of your answer quality"],
        "probe":      None,
    }


# ─────────────────────────────────────────────────────────────
# SAFE EVALUATE  (retry on parse failure)
# ─────────────────────────────────────────────────────────────
def _safe_evaluate(client, q_data, answer, lang, role, words, is_voice):
    result = evaluate_answer(
        client, q_data, answer, lang, role,
        response_time=0, word_count=words, is_voice=is_voice,
    )

    # ── FIX: evaluate_answer sometimes returns a list e.g. [{...}] ──
    # Unwrap the first element if it's a list, otherwise fall back to {}
    if isinstance(result, list):
        result = result[0] if (result and isinstance(result[0], dict)) else {}

    # Also guard against any other non-dict return type
    if not isinstance(result, dict):
        result = {}

    # Detect bad/failed evaluations and retry
    bad = (
        result.get("feedback") in ("Parse failed", "Eval failed")
        or result.get("overall") == 5
        or not result  # empty dict from fallback above
    )

    if bad:
        try:
            prompt = (
                f"Score this interview answer 1-10.\n"
                f"Q: {q_data['question']}\nA: {answer[:400]}\n\n"
                "Return ONLY this exact JSON (no other text, no array):\n"
                '{"overall":7,"feedback":"your feedback here","strengths":["strength1"],'
                '"weaknesses":["weakness1"],"probe":null}'
            )
            r = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            retry_result = _robust_parse(r.choices[0].message.content)

            # Same list-guard on the retry response
            if isinstance(retry_result, list):
                retry_result = retry_result[0] if (retry_result and isinstance(retry_result[0], dict)) else {}

            if isinstance(retry_result, dict) and "overall" in retry_result:
                result = retry_result
            else:
                raise ValueError("Retry also returned bad structure")

        except Exception as e:
            print(f"Retry eval failed: {e}")
            result = {
                "overall":   6,
                "feedback":  "Answer received and noted. Please continue.",
                "strengths": ["Response submitted successfully"],
                "weaknesses": [],
                "probe":     None,
            }

    # Ensure all required keys exist with safe defaults
    result.setdefault("overall",    5)
    result.setdefault("feedback",   "No feedback.")
    result.setdefault("strengths",  [])
    result.setdefault("weaknesses", [])
    result.setdefault("probe",      None)

    # Clamp score to valid range
    result["overall"] = max(1.0, min(10.0, float(result["overall"])))
    return result


# ─────────────────────────────────────────────────────────────
# WHISPER TRANSCRIPTION
# ─────────────────────────────────────────────────────────────
def _transcribe(client, audio_bytes: bytes, language: str) -> str:
    from pydub import AudioSegment
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes); raw = f.name
    try:
        seg = (AudioSegment.from_file(raw)
               .set_frame_rate(16000).set_channels(1).set_sample_width(2))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            seg.export(f.name, format="wav"); wav = f.name
    finally:
        os.unlink(raw)
    try:
        with open(wav, "rb") as f:
            return client.audio.transcriptions.create(
                file=f, model=WHISPER_MODEL,
                language=language.split("-")[0], response_format="text",
            ).text.strip()
    finally:
        os.unlink(wav)


# ─────────────────────────────────────────────────────────────
# ENDPOINT 1 — Start Session
# Total time: ~1.5s (was ~7s)
# ─────────────────────────────────────────────────────────────
@app.post("/api/session/start")
async def start_session(
    pdf:      UploadFile = File(...),
    role:     str        = Form("backend"),
    language: str        = Form("en-US"),
    position: str        = Form("Software Engineer"),
):
    try:
        pdf_bytes = await pdf.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes); tmp = f.name

        client      = get_groq_client()          # instant (singleton)
        resume_text = await _bg(extract_pdf_text, tmp)
        os.unlink(tmp)

        # Phase A: first question + TTS in PARALLEL (~1.5s total)
        first_q_data, _ = await asyncio.gather(
            _bg(_gen_first_question, client, resume_text, position, language, role),
            asyncio.sleep(0),   # placeholder; TTS chained below
        )
        first_q_text = first_q_data["question"]

        # TTS for first question (can start immediately once we have the text)
        audio_b64 = await _bg(generate_tts_bytes, first_q_text, language)

        sid = str(uuid.uuid4())
        sessions[sid] = {
            "client":           client,
            "questions":        [first_q_data],   # just Q1 for now
            "questions_ready":  False,
            "results":          [],
            "config": {
                "position":     position,
                "role":         role,
                "language":     language,
                "resume_text":  resume_text,
            },
            "current_idx":      0,
            "probe_count":      0,
            "question_history": [first_q_text],
            "total_asked":      0,
            "db_conn":          init_db(),
        }

        # Phase B: generate remaining 4 questions silently in the background
        # (user is already reading Q1 while this runs)
        asyncio.create_task(
            _fill_remaining_questions(sid, client, resume_text, position, language, role, first_q_text)
        )

        return {
            "session_id":      sid,
            "question":        first_q_text,
            "question_index":  0,
            "total_questions": MAX_QUESTIONS,
            "audio_base64":    audio_b64,
            "is_probe":        False,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────────────────────
# ENDPOINT 2 — Submit Answer
# Eval + next-question TTS run in PARALLEL
# ─────────────────────────────────────────────────────────────
@app.post("/api/session/answer")
async def submit_answer(payload: AnswerPayload):
    sid = payload.session_id
    if sid not in sessions:
        raise HTTPException(404, "Session not found or expired.")

    session = sessions[sid]
    client  = session["client"]
    lang    = session["config"]["language"]
    role    = session["config"]["role"]
    q_idx   = session["current_idx"]

    # Wait for background questions if we're past Q1 and they're not ready
    if q_idx > 0 and not session.get("questions_ready"):
        for _ in range(30):        # wait up to 3s
            await asyncio.sleep(0.1)
            if session.get("questions_ready"):
                break

    q_data = session["questions"][min(q_idx, len(session["questions"]) - 1)]

    # Resolve answer text
    answer = payload.answer_text
    if payload.is_voice and payload.audio_base64:
        audio  = base64.b64decode(payload.audio_base64)
        answer = await _bg(_transcribe, client, audio, lang)
        print(f"✓ Whisper: {answer[:80]}")

    if not answer:
        raise HTTPException(400, "No answer provided.")

    words = len(answer.split())

    # PARALLEL: evaluate + pre-generate next TTS at the same time
    next_idx       = q_idx + 1
    will_have_next = (
        next_idx < len(session["questions"])
        and session["total_asked"] + 1 < MAX_QUESTIONS
    )

    if will_have_next:
        next_q_text = session["questions"][next_idx]["question"]
        evaluation, speculative_audio = await asyncio.gather(
            _bg(_safe_evaluate, client, q_data, answer, lang, role, words, payload.is_voice),
            _bg(generate_tts_bytes, next_q_text, lang),
        )
    else:
        evaluation      = await _bg(_safe_evaluate, client, q_data, answer, lang, role, words, payload.is_voice)
        speculative_audio = None
        next_q_text       = None

    print(f"✅ Eval: {evaluation['overall']}/10")

    session["results"].append({
        "question":   q_data["question"],
        "answer":     answer,
        "eval":       evaluation,
        "word_count": words,
        "is_voice":   payload.is_voice,
    })
    session["total_asked"] += 1

    response: Dict[str, Any] = {
        "evaluation":     evaluation,
        "next_question":  None,
        "audio_base64":   None,
        "is_probe":       False,
        "question_index": q_idx,
        "done":           False,
    }

    # Probe check
    if (evaluation.get("overall", 10) < PROBE_THRESHOLD
            and session["probe_count"] < 2
            and session["question_history"]):
        probe_q = await _bg(
            generate_probe, client, q_data["question"],
            answer, evaluation, lang, session["question_history"],
        )
        if probe_q:
            session["probe_count"] += 1
            session["question_history"].append(probe_q)
            response.update({
                "next_question": probe_q,
                "audio_base64":  await _bg(generate_tts_bytes, probe_q, lang),
                "is_probe":      True,
            })
            return response

    # Advance
    session["current_idx"] += 1

    if (session["current_idx"] >= len(session["questions"])
            or session["total_asked"] >= MAX_QUESTIONS):
        assessment = await _bg(
            generate_final_assessment, client,
            session["config"]["position"],
            session["config"]["resume_text"],
            session["results"], lang,
        )
        response.update({"done": True, "assessment": assessment})
    else:
        actual_next = session["questions"][session["current_idx"]]["question"]
        session["question_history"].append(actual_next)

        audio = speculative_audio if (speculative_audio and next_q_text == actual_next) \
                else await _bg(generate_tts_bytes, actual_next, lang)

        response.update({
            "next_question":  actual_next,
            "audio_base64":   audio,
            "question_index": session["current_idx"],
        })

    return response


# ─────────────────────────────────────────────────────────────
# ENDPOINT 3 — Results
# ─────────────────────────────────────────────────────────────
@app.get("/api/session/{sid}/results")
async def get_results(sid: str):
    if sid not in sessions:
        raise HTTPException(404, "Session not found.")
    s   = sessions[sid]
    res = s["results"]
    avg = sum(r["eval"].get("overall", 0) for r in res) / len(res) if res else 0
    return {
        "session_id": sid,
        "config":     s["config"],
        "results":    res,
        "summary": {
            "avg_score":      round(avg, 2),
            "fit_score":      round(avg * 10),
            "recommendation": "HIRE" if avg >= 7 else "NO HIRE",
            "total_asked":    s["total_asked"],
        },
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINT 4 — Health
# ─────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status":     "ok",
        "groq":       bool(GROQ_API_KEY),
        "elevenlabs": bool(ELEVEN_API_KEY),
        "sessions":   len(sessions),
        "client_ready": _GROQ_CLIENT is not None,
    }


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)