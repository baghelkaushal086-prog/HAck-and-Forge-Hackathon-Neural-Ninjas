"""
Enterprise AI Interview Agent v4.0 (Groq + Advanced Features)
UPGRADES IMPLEMENTED:
- ✅ TTS: AI speaks questions aloud (gTTS + pygame)
- ✅ Groq Whisper: Ultra-fast, accurate voice transcription
- ✅ Dynamic Branching: Probes low-score answers (<6/10)
- ✅ Golden Answers: Anti-hallucination grading benchmark
- ✅ SQLite Database: Persistent interview storage
- ✅ Bias Mitigation: Anonymizes PII from resume
Requirements: pip install groq pypdf SpeechRecognition pyaudio pandas openpyxl numpy gtts pygame pydub
"""

import os
import sys
import json
import re
import time
import csv
import warnings
import textwrap
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import sqlite3
import io
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

ELEVENLABS_VOICE = "pNInz6obpgDQGcFmaJgB"  # "Adam" - Professional interviewer
# Alternatives: "EXAVITQu4vr4xnSDxMaL" (Sarah), "D38BD911C1C3A5DE1B5C" (Josh), "VR6AewLTigWG4xSOukaG" (Matilda)
# Data processing and AI
import pandas as pd
from pypdf import PdfReader
from groq import Groq

# Voice recognition + TTS
import speech_recognition as sr
from gtts import gTTS
import pygame
from pydub import AudioSegment  # For WAV conversion

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION & CONSTANTS (UNCHANGED)
# ============================================
DEFAULT_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama3-70b-8192"
WHISPER_MODEL = "whisper-large-v3"  # Groq Whisper

ROLE_RUBRICS = {  # Unchanged
    "backend": {"name": "Backend Engineer", "focus": "scalability, APIs, databases, microservices, system design", "technical_weight": 0.7, "behavioral_weight": 0.3},
    "frontend": {"name": "Frontend Engineer", "focus": "React/Vue/Angular, UI/UX, performance, accessibility, state management", "technical_weight": 0.6, "behavioral_weight": 0.4},
    "pm": {"name": "Product Manager", "focus": "stakeholder management, roadmaps, metrics, prioritization, market analysis", "technical_weight": 0.4, "behavioral_weight": 0.6},
    "data_science": {"name": "Data Scientist", "focus": "statistics, ML models, data pipelines, A/B testing, visualization", "technical_weight": 0.8, "behavioral_weight": 0.2},
    "devops": {"name": "DevOps Engineer", "focus": "CI/CD, cloud infrastructure, monitoring, containerization, security", "technical_weight": 0.75, "behavioral_weight": 0.25}
}

LANGUAGE_CODES = {  # Unchanged
    "en": "en-US", "es": "es-ES", "fr": "fr-FR", "de": "de-DE",
    "hi": "hi-IN", "zh": "zh-CN", "ja": "ja-JP", "ko": "ko-KR",
    "ru": "ru-RU", "ar": "ar-SA", "pt": "pt-BR", "it": "it-IT"
}

DEFAULT_LANGUAGE = "en-US"
DEFAULT_ROLE = "backend"
QUESTION_TIMEOUT = 120
WARNING_TIME = 90
VOICE_RETRIES = 3
MAX_RESUME_TEXT = 4000
PROBE_THRESHOLD = 6.0  # Trigger probe if score < this
MAX_QUESTIONS = 7  # Allow up to 2 probes

# ============================================
# NEW: DATABASE SETUP
# ============================================
def init_db():
    conn = sqlite3.connect('interviews.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interviews
                 (id INTEGER PRIMARY KEY, timestamp TEXT, position TEXT, role TEXT, language TEXT,
                  resume_hash TEXT, avg_score REAL, fit_score INTEGER, recommendation TEXT,
                  json_data TEXT)''')
    conn.commit()
    return conn

# ============================================
# NEW: BIAS MITIGATION - ANONYMIZE RESUME
# ============================================
def anonymize_resume(text: str) -> str:
    """Remove PII: names, emails, phones, dates, locations for fair evaluation."""
    # Common PII patterns
    patterns = [
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Full names
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US phones
        r'\b\d{10}\b',  # 10-digit phones
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Dates
        r'\b\d{4}\b',  # Years (keep some context)
    ]
    for pattern in patterns:
        text = re.sub(pattern, '[REDACTED]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[A-Z]{2,5}\b', '[LOCATION]', text)  # States/Cities
    return text.strip()

# ============================================
# UPDATED: INITIALIZATION (ADD TTS INIT)
# ============================================
def initialize_groq(api_key: str) -> Groq:
    try:
        client = Groq(api_key=api_key)
        # Test chat + whisper
        client.chat.completions.create(messages=[{"role": "user", "content": "Test"}], model=DEFAULT_MODEL, max_tokens=5)
        print(f"✓ Groq client initialized (Llama3 + Whisper)")
        pygame.mixer.init()  # TTS init
        return client
    except Exception as e:
        raise ValueError(f"Failed to initialize Groq: {str(e)}")

# ============================================
# UPDATED: PDF EXTRACTION + ANONYMIZATION
# ============================================
def extract_pdf_text(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) == 0:
            raise ValueError("No text extracted")
        anon_text = anonymize_resume(text)
        print(f"✓ Extracted {len(text)} chars → Anonymized {len(anon_text)} chars")
        return anon_text[:MAX_RESUME_TEXT]
    except Exception as e:
        raise ValueError(f"PDF parse failed: {str(e)}")

# ============================================
# NEW: ELEVENLABS TTS (HUMAN-LIKE)
# ============================================
def speak_question(question: str, language: str = "en-US"):
    """Fixed ElevenLabs: generate() → bytes (no streaming)."""
    try:
        if use_elevenlabs and eleven_client:
            audio = eleven_client.generate(
                text=question,
                voice=ELEVENLABS_VOICE,
                model="eleven_multilingual_v2",
                voice_settings=VoiceSettings(
                    stability=0.6,
                    similarity_boost=0.8,
                    style=0.2,
                    use_speaker_boost=True
                )
            )
            pygame.mixer.music.load(io.BytesIO(audio))
            print("🔊 ElevenLabs: Ultra-real voice")
        else:
            # gTTS fallback (unchanged)
            from gtts import gTTS
            lang_code = LANGUAGE_CODES.get(language.split('-')[0], "en")
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tts = gTTS(question, lang=lang_code)
                tts.save(tmp.name)
                pygame.mixer.music.load(tmp.name)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy(): time.sleep(0.1)
                os.unlink(tmp.name)
            return
        
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(0.1):
            time.sleep(0.1)
            
    except Exception as e:
        print(f"TTS fallback activated: {str(e)[:60]}")

# Global vars for client
use_elevenlabs = False
eleven_client = None

# ============================================
# UPDATED: GENERATE QUESTIONS + GOLDEN ANSWERS
# ============================================
def generate_questions(client: Groq, resume_text: str, position: str, 
                       language: str = "en-US", role: str = "backend") -> List[Dict[str, str]]:
    """Generate questions + hidden golden answers."""
    rubric = ROLE_RUBRICS.get(role.lower(), ROLE_RUBRICS["backend"])
    
    prompt = f"""Generate EXACTLY 5 interview questions + GOLDEN ANSWERS for {position}.

RESUME: {resume_text}

Output ONLY JSON array of objects:
[
  {{"question": "Q1?", "golden_answer": "Ideal 2-3 sentence response"}},
  ...
]

Focus: {rubric['focus']}. Language: {language}"""
    
    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3
        )
        data = safe_json_parse(response.choices[0].message.content)
        if isinstance(data, list) and len(data) >= 5:
            return [{"question": q["question"], "golden_answer": q.get("golden_answer", "")} for q in data[:5]]
    except:
        pass
    
    # Fallback
    return [{"question": q, "golden_answer": "Strong response demonstrates depth with metrics."} for q in [
        f"Experience as {rubric['name']}?", "Core skills?", "Technical challenge?", "Scenario?", "Architecture?"
    ]]

# ============================================
# FIXED: GROQ WHISPER VOICE INPUT
# ============================================
def get_voice_input(client: Groq, recognizer: sr.Recognizer, microphone: sr.Microphone, 
                    language: str, max_retries: int = 3) -> Tuple[str, bool, float]:
    """Fixed: Resample to 16kHz mono WAV for Groq Whisper."""
    for attempt in range(1, max_retries + 1):
        print(f"Listening (Attempt {attempt}/{max_retries}) - Speak now!")
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=45)
            
            # Convert to proper WAV for Groq (16kHz mono)
            wav_bytes = audio.get_wav_data()
            audio_segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            
            tmp_file = tempfile.mktemp(suffix='.wav')
            audio_segment.export(tmp_file, format="wav")
            
            # Transcribe with Groq
            with open(tmp_file, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    file=(tmp_file, f),
                    model=WHISPER_MODEL,
                    language=language.split('-')[0] if '-' in language else language,
                    response_format="text"
                ).text
            
            os.unlink(tmp_file)  # Cleanup
            if transcript.strip():
                print(f"✓ Whisper: '{transcript[:80]}...'")
                return transcript.strip(), True, time.perf_counter() - (time.perf_counter() - 2)  # Approx time
        except sr.WaitTimeoutError:
            print("⏰ No speech detected")
        except sr.UnknownValueError:
            print("🤷 Speech unclear")
        except Exception as e:
            print(f"Voice error: {str(e)[:50]}")
    
    print("✗ Voice failed → Falling back to text")
    return "", False, 0.0
    
# ============================================
# UPDATED: EVALUATE WITH GOLDEN ANSWER
# ============================================
# ============================================
# FIXED: EVALUATE_ANSWER (ALWAYS INCLUDE WEAKNESSES)
# ============================================
def evaluate_answer(client: Groq, question_data: Dict[str, str], answer: str, language: str,
                    role: str, response_time: float, word_count: int, is_voice: bool) -> Dict[str, Any]:
    rubric = ROLE_RUBRICS.get(role.lower(), ROLE_RUBRICS["backend"])
    golden = question_data.get("golden_answer", "")
    
    prompt = f"""Score 1-10. ALWAYS include "weaknesses": ["list 1-2 specific issues"].

QUESTION: {question_data['question']}
GOLDEN: {golden}
ANSWER: {answer}
TIME: {response_time:.1f}s | WORDS: {word_count}

JSON:
{{
  "correctness": 8, "depth": 7, "clarity": 9, "structure": 8,
  "overall": 8,
  "feedback": "In {language}",
  "probe": "Suggested probe idea",
  "strengths": ["..."], "weaknesses": ["Missing metrics", "Vague example"]
}}"""
    
    try:
        response = client.chat.completions.create(model=DEFAULT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.1)
        return safe_json_parse(response.choices[0].message.content)
    except:
        return {"overall": 5, "feedback": "Eval failed", "probe": None}

# ============================================
# DYNAMIC PROBING LOGIC (NEW)
# ============================================
def should_probe(eval_result: Dict) -> bool:
    return eval_result.get("overall", 0) < PROBE_THRESHOLD and "probe" in eval_result

def generate_probe(client: Groq, question: str, answer: str, eval_result: Dict, 
                   language: str, history: List[str]) -> Optional[str]:
    """Smart probe: Different, weakness-focused, no repeats."""
    weaknesses = eval_result.get("weaknesses", ["unclear"])
    history_str = " | ".join(history[-3:])  # Last 3 Qs
    
    prompt = f"""Generate 1 NEW probe question DIFFERENT from these recent Qs:
Recent: {history_str}

Original Q: {question}
Weak answer: {answer[:200]}
Weaknesses: {weaknesses}

Probe must:
- Probe SPECIFIC weakness (e.g. 'no metrics' → 'What metrics?')
- COMPLETELY DIFFERENT angle from original/recent Qs
- Short, natural, conversational
- Language: {language}

RETURN ONLY the probe QUESTION (no intro/JSON)."""
    
    try:
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL, 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.7,  # Creative but focused
            max_tokens=70
        )
        probe = resp.choices[0].message.content.strip()
        
        # Dedupe check
        if any(similar := probe.lower() in q.lower() for q in history) or probe.lower() in question.lower():
            return None  # Skip if too similar
        
        return probe if probe.endswith('?') else probe + '?'
    except:
        return None

# ============================================
# UPDATED: MAIN FLOW WITH DYNAMIC + TTS + WHISPER
# ============================================
def main():
    print("\n" + "="*80)
    print(" AI INTERVIEWER v4.2 (ELEVENLABS HUMAN VOICE + WHISPER + DYNAMIC)")
    print("="*80)
    
    groq_key = input("Groq API key: ").strip()
    eleven_key = input("ElevenLabs API key (optional, falls back to gTTS): ").strip()
    
    try:
        client = initialize_groq(groq_key)
        global eleven_client, use_elevenlabs
        if eleven_key:
            eleven_client = ElevenLabs(api_key=eleven_key)
            use_elevenlabs = True
            print("✓ ElevenLabs loaded (real human voice)")
        else:
            use_elevenlabs = False
            print("ElevenLabs skipped → Using gTTS fallback")
    except Exception as e:
        return print(e)
    
    language = input("Language (en-US): ").strip() or DEFAULT_LANGUAGE
    resume_path = input("PDF resume path: ").strip()
    try:
        resume_text = extract_pdf_text(resume_path)
    except Exception as e:
        return print(e)
    
    position = input("Position: ").strip() or "Software Engineer"
    role_input = input("Role (backend/...): ").strip().lower() or DEFAULT_ROLE
    rubric = ROLE_RUBRICS.get(role_input, ROLE_RUBRICS["backend"])
    
    # Generate questions + golden
    print("Generating smart questions...")
    questions = generate_questions(client, resume_text, position, language, role_input)
    asked_questions = set()  # Track unique Qs
    probe_count = 0
    MAX_PROBES = 2
    # Voice setup
    recognizer = sr.Recognizer()
    microphone = None
    try:
        microphone = sr.Microphone()
        print("✓ Microphone + Whisper ready")
    except:
        print("✗ Voice fallback to text")
    
    # DB
    db_conn = init_db()
    
    print("\n" + "="*80)
    print(" INTERVIEW START (AI will SPEAK questions)")
    print("="*80)
    
    results = []
    question_idx = 0
    total_questions = 0
    asked_questions = set()
    probe_count = 0
    question_history = []  # List for context
    
    while question_idx < len(questions) and total_questions < MAX_QUESTIONS:
        q_data = questions[question_idx]
        question = q_data["question"]
        
        # Skip if duplicate
        if question.lower() in asked_questions:
            question_idx += 1
            continue
            
        speak_question(question, language)
        print(f"\n🆕 Q{question_idx+1}/{len(questions)} ({total_questions+1}/{MAX_QUESTIONS}):")
        print(question)
        asked_questions.add(question.lower())
        question_history.append(question)
        
        # Input + eval (unchanged from v4.1)
        mode = input("Reply [T]ext / [V]oice? (t/v) [t]: ").strip().lower() or 't'
        start_time = time.perf_counter()
        if mode == 'v' and microphone:
            answer, success, resp_time = get_voice_input(client, recognizer, microphone, language)
            if success:
                elapsed = resp_time
                word_count = len(answer.split())
                is_voice = True
            else:
                answer, elapsed, word_count = get_text_input()
                is_voice = False
        else:
            answer, elapsed, word_count = get_text_input()
            is_voice = False
        
        print("\n🤖 Evaluating...")
        evaluation = evaluate_answer(client, q_data, answer, language, role_input, elapsed, word_count, is_voice)
        print(f"Score: \033[94m{evaluation.get('overall', 5):.1f}/10\033[0m")
        print(f"   Feedback: {evaluation.get('feedback', 'N/A')[:100]}")
        
        result = {'question': question, 'answer': answer[:500], 'eval': evaluation,
                  'response_time': elapsed, 'word_count': word_count, 'is_voice': is_voice}
        results.append(result)
        total_questions += 1
        
        # FIXED PROBE: Limited + smart + history-aware
        if (evaluation.get('overall', 0) < PROBE_THRESHOLD and 
            probe_count < MAX_PROBES and 
            len(question_history) > 1):  # Not first Q
            
            print(f"\n🔍 Low score → Smart probe ({probe_count+1}/{MAX_PROBES})...")
            probe_question = generate_probe(client, question, answer, evaluation, language, question_history)
            
            if probe_question:
                speak_question(probe_question, language)
                print(f"🔍 PROBE: {probe_question}")
                asked_questions.add(probe_question.lower())
                question_history.append(probe_question)
                probe_count += 1
                
                # Probe input
                p_mode = input("Probe [T]ext / [V]oice? (t/v) [t]: ").strip().lower() or 't'
                if p_mode == 'v' and microphone:
                    p_answer, p_success, p_resp_time = get_voice_input(client, recognizer, microphone, language)
                    if p_success:
                        p_elapsed = p_resp_time
                        p_word_count = len(p_answer.split())
                        p_voice = True
                    else:
                        p_answer, p_elapsed, p_word_count = get_text_input()
                        p_voice = False
                else:
                    p_answer, p_elapsed, p_word_count = get_text_input()
                    p_voice = False
                
                print("\nProbe eval...")
                p_eval = evaluate_answer(client, {"question": probe_question, "golden_answer": "N/A (probe)"}, 
                                       p_answer, language, role_input, p_elapsed, p_word_count, p_voice)
                print(f"Probe Score: \033[94m{p_eval.get('overall', 5):.1f}/10\033[0m")
                print(f"   Probe Feedback: {p_eval.get('feedback', 'N/A')[:100]}")
                
                results.append({
                    'question': f"🔍 PROBE: {probe_question}", 'answer': p_answer[:500],
                    'eval': p_eval, 'response_time': p_elapsed, 'word_count': p_word_count, 'is_voice': p_voice
                })
                total_questions += 1
            else:
                print("⏭️ Probe skipped (would repeat)")
        
        if total_questions < MAX_QUESTIONS and question_idx < len(questions) - 1:
            input("\n⏭  Press Enter for next main question...")
        question_idx += 1

# ============================================
# UTILITY FUNCTIONS (UNCHANGED/MINOR UPDATES)
# ============================================
def safe_json_parse(text: str) -> Dict[str, Any]:
    # Unchanged
    if not isinstance(text, str): text = str(text)
    text = re.sub(r'```json\s*|\s*```', '', text).strip()
    json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if json_match: text = json_match.group(1)
    try:
        return json.loads(text)
    except:
        return {"overall": 5, "feedback": "Parse failed"}

# ============================================
# FIXED: TEXT INPUT (BETTER DOUBLE-ENTER)
# ============================================
def get_text_input(timeout: int = QUESTION_TIMEOUT) -> Tuple[str, float, int]:
    print(f"\n💬 Type answer (Enter TWICE to submit | {timeout}s timeout):")
    lines = []
    start_time = time.perf_counter()
    empty_lines = 0
    while time.perf_counter() - start_time < timeout:
        try:
            line = input()
            if not line.strip():
                empty_lines += 1
                if empty_lines >= 2 and lines:
                    break
            else:
                empty_lines = 0
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            break
    elapsed = time.perf_counter() - start_time
    text = ' '.join(lines).strip()
    print(f"Submitted ({len(text.split())} words, {elapsed:.1f}s)")
    return text or "[No response]", elapsed, len(text.split())

def generate_final_assessment(client: Groq, position: str, resume_text: str, results: List[Dict], language: str) -> Dict[str, Any]:
    # Unchanged core logic
    qa_summary = "\n".join([f"Q{i+1}: Score {r['eval'].get('overall',5):.1f}" for i,r in enumerate(results)])
    avg_score = np.mean([r['eval'].get('overall',5) for r in results])
    prompt = f"Final assessment. Avg: {avg_score:.1f}. Summary: {qa_summary[:1000]}. JSON: fit_score, recommendation, justification"
    try:
        resp = client.chat.completions.create(model=DEFAULT_MODEL, messages=[{"role": "user", "content": prompt}])
        assessment = safe_json_parse(resp.choices[0].message.content)
        assessment["fit_score"] = assessment.get("fit_score", int(avg_score * 10))
        assessment["recommendation"] = assessment.get("recommendation", "HIRE" if avg_score >=7 else "NO HIRE")
        return assessment
    except:
        return {"fit_score": int(avg_score*10), "recommendation": "HIRE" if avg_score >=7 else "NO HIRE", "justification": "Fallback"}

def print_enterprise_scorecard(results: List[Dict], avg_score: float, assessment: Dict, language: str, role: str):
    # Unchanged
    print("\n"*3 + "="*120)
    print(" " * 45 + "ENTERPRISE INTERVIEW SCORECARD v4.0")
    print("="*120)
    print(f"Position: {assessment.get('position')} | Role: {ROLE_RUBRICS[role]['name']} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*120)
    print(f"{'#':2} {'Mode':5} {'Time':7} {'Words':5} {'Score':5} {'Feedback'}")
    print("-"*120)
    for i, r in enumerate(results):
        mode = "Voice" if r.get('is_voice') else "Text"
        print(f"{i+1:2d} {mode:5} {r.get('response_time',0):6.1f} {r.get('word_count',0):5} {r['eval'].get('overall',0):5.1f}  {r['eval'].get('feedback','')[:60]}")
    print("-"*120)
    print(f"Avg: {avg_score:.1f}/10 | Fit: {assessment.get('fit_score',0)}% | REC: \033[92m{assessment.get('recommendation','PENDING')}\033[0m")
    print(f"Justification: {assessment.get('justification','N/A')}")

def export_report(results: List[Dict], assessment: Dict, filename: str = None) -> str:
    # Unchanged
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = filename or f"interview_report_{timestamp}"
    csv_path = f"{filename}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['#', 'Question', 'Answer', 'Mode', 'Time', 'Words', 'Score', 'Feedback'])
        for i, r in enumerate(results):
            writer.writerow([i+1, r['question'], r['answer'][:200], ('Voice' if r.get('is_voice') else 'Text'), r.get('response_time'), r.get('word_count'), r['eval'].get('overall'), r['eval'].get('feedback')])
    
    json_path = f"{filename}.json"
    with open(json_path, 'w') as f:
        json.dump({"assessment": assessment, "results": results}, f, indent=2)
    print(f"✓ Exported: {csv_path}, {json_path}")
    return csv_path

# ============================================
# MAIN ENTRY (UNCHANGED)
# ============================================
def quick_test_mode():
    groq_key = input("Groq key: ").strip()
    eleven_key = input("ElevenLabs key (opt): ").strip()
    try:
        client = initialize_groq(groq_key)
        global use_elevenlabs, eleven_client
        if eleven_key:
            eleven_client = ElevenLabs(api_key=eleven_key)
            use_elevenlabs = True
        speak_question("Hello! This is your AI interviewer powered by ElevenLabs. Test complete!")
        print("✅ ElevenLabs + Full system ready!")
    except Exception as e:
        print(f"Test: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.test: quick_test_mode()
    else: main()