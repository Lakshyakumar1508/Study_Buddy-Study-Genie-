import os
import re
import json
import hashlib
from typing import TypedDict, List, Dict, Any, Optional
from django.conf import settings
from dotenv import load_dotenv

load_dotenv()

import markdown
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# --- STATE DEFINITION FOR LANGGRAPH ---
class StudyBuddyState(TypedDict):
    task: str                 # 'explain', 'summarize', 'quiz', 'flashcards'
    user_input: str           # raw input from user/pdf
    num_items: int            # for quiz/flashcards
    language: str             # 'Auto', 'English', 'Hinglish', 'Hindi', etc.
    cleaned_input: str        # token-optimized & trimmed input
    raw_response: str         # LLM raw text
    formatted_html: str       # clean, styled HTML output (no raw **)
    clean_text: str           # plain text without markdown for TTS / speech
    structured_data: Any      # parsed list/dict for interactive quiz/flashcards
    tokens_saved_estimate: int
    is_cached: bool
    error: Optional[str]

# --- IN-MEMORY CACHE TO SAVE 100% TOKENS ON REPEAT QUERIES ---
CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_MAX_ENTRIES = 200

def _generate_cache_key(task: str, text: str, num_items: int, language: str) -> str:
    normalized = f"{task}:{language}:{num_items}:{text.strip().lower()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def _extract_text_from_response(content: Any) -> str:
    """
    Safely extracts clean plain text from LangChain / Google GenAI response content.
    Handles str, list of dicts (e.g. [{'type': 'text', 'text': '...', 'extras': ...}]),
    and objects with a text attribute. Strips out raw signatures/metadata.
    """
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                pieces.append(str(part.get("text", "")))
            elif hasattr(part, "text"):
                pieces.append(str(getattr(part, "text", "")))
            else:
                pieces.append(str(part))
        text = "".join(pieces)
    elif hasattr(content, "text"):
        text = str(getattr(content, "text", ""))
    else:
        text = str(content)

    # Extra safety: If string starts like a Python literal representation of list of dicts
    if text.startswith("[{'type': 'text'") or text.startswith('[{"type": "text"'):
        try:
            import ast
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in parsed)
        except Exception:
            pass

    return text.strip()

def _clean_markdown_to_plain_text(md_text: str) -> str:
    """Strips all markdown formatting (*, #, _, `, etc.) for clean Text-to-Speech."""
    md_text = _extract_text_from_response(md_text)
    text = re.sub(r'#+\s*', '', md_text)
    text = re.sub(r'\*{1,3}([^*]+?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}([^_]+?)_{1,3}', r'\1', text)
    text = re.sub(r'`{1,3}([^`]+?)`{1,3}', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'^[*\-+\d.]\s+', '', text, flags=re.MULTILINE)
    text = text.replace('**', '').replace('*', '')
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()

def _clean_markdown_html(md_text: str) -> str:
    """
    Cleans raw markdown formatting quirks, ensures NO raw asterisks exist,
    and converts to modern, rich semantic HTML.
    """
    md_text = _extract_text_from_response(md_text)
    
    # Pre-cleaning: Fix whitespace issues around bold asterisks (e.g. '** word **' -> '**word**')
    cleaned = re.sub(r'\*\*\s+([^*\n]+?)\s+\*\*', r'**\1**', md_text)
    cleaned = re.sub(r'\*\*\s+([^*\n]+?)\*\*', r'**\1**', cleaned)
    cleaned = re.sub(r'\*\*([^*\n]+?)\s+\*\*', r'**\1**', cleaned)
    # Fix solitary unclosed bold at end of lines
    cleaned = re.sub(r'\*\*([^*\n]+)$', r'**\1**', cleaned, flags=re.MULTILINE)
    
    # Convert markdown to rich semantic HTML
    html = markdown.markdown(
        cleaned,
        extensions=['extra', 'tables', 'nl2br', 'sane_lists']
    )
    
    # Post-cleaning: Catch any unparsed **word** left in HTML and convert to <strong>
    html = re.sub(r'\*\*([^*\n<]+?)\*\*', r'<strong>\1</strong>', html)
    # Remove any remaining stray asterisks entirely so user NEVER sees raw **
    html = html.replace('**', '')
    html = re.sub(r'\\(\*|_)', r'\1', html)
    
    return html

def _parse_quiz_items(raw_text: str, num_requested: int) -> List[Dict[str, Any]]:
    """Robust parser that extracts quiz items as structured objects."""
    # Attempt 1: Extract JSON array directly
    match = re.search(r'\[\s*\{.*\}\s*\]', raw_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list) and len(data) > 0:
                normalized = []
                for idx, q in enumerate(data[:num_requested]):
                    opts = q.get("options", [])
                    # Clean options
                    clean_opts = [re.sub(r'^[A-Da-d][\)\.\:\-]\s*', '', str(opt)).replace('**', '').strip() for opt in opts]
                    normalized.append({
                        "id": idx + 1,
                        "question": str(q.get("question", "")).replace('**', '').strip(),
                        "options": clean_opts if len(clean_opts) == 4 else opts,
                        "answer": str(q.get("answer", "A")).replace('**', '').strip()[:1].upper(),
                        "explanation": str(q.get("explanation", "")).replace('**', '').strip()
                    })
                if len(normalized) > 0:
                    return normalized
        except Exception:
            pass

    # Attempt 2: Fallback Regex parser for plain formatted text
    questions = []
    blocks = re.split(r'\n(?=(?:Q\d+[:.]|\d+[\.)]))', raw_text)
    for idx, b in enumerate(blocks):
        b = b.strip()
        if not b:
            continue
        lines = [l.strip() for l in b.split('\n') if l.strip()]
        if not lines:
            continue
        q_text = lines[0]
        q_text = re.sub(r'^(?:Q\d+[:.]|\d+[\.)])\s*', '', q_text).replace('**', '')
        options = []
        ans = "A"
        expl = ""
        for line in lines[1:]:
            opt_match = re.match(r'^([A-Da-d])[\)\.]\s*(.+)', line)
            ans_match = re.search(r'(?:Answer|Ans|Correct)[:\s]+([A-Da-d])', line, re.IGNORECASE)
            expl_match = re.search(r'(?:Explanation|Reason)[:\s]+(.+)', line, re.IGNORECASE)
            if opt_match:
                options.append(opt_match.group(2).replace('**', '').strip())
            if ans_match:
                ans = ans_match.group(1).upper()
            if expl_match:
                expl = expl_match.group(1).replace('**', '').strip()
        if q_text and len(options) >= 2:
            questions.append({
                "id": idx + 1,
                "question": q_text,
                "options": options,
                "answer": ans,
                "explanation": expl or "High-yield exam concept."
            })
    return questions[:num_requested]

def _parse_flashcard_items(raw_text: str, num_requested: int) -> List[Dict[str, Any]]:
    """Robust parser that extracts flashcards as structured objects."""
    # Attempt 1: Extract JSON array directly
    match = re.search(r'\[\s*\{.*\}\s*\]', raw_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list) and len(data) > 0:
                normalized = []
                for idx, c in enumerate(data[:num_requested]):
                    front = str(c.get("front", c.get("question", ""))).replace('**', '').strip()
                    back = str(c.get("back", c.get("answer", ""))).replace('**', '').strip()
                    hint = str(c.get("hint", "")).replace('**', '').strip()
                    if front and back:
                        normalized.append({
                            "id": idx + 1,
                            "front": front,
                            "back": back,
                            "hint": hint or "Key memory trigger for revision"
                        })
                if len(normalized) > 0:
                    return normalized
        except Exception:
            pass

    # Attempt 2: Fallback Regex parser
    cards = []
    blocks = re.split(r'\n(?=(?:Card\s*\d+[:.]|\d+[\.)]))', raw_text)
    for idx, b in enumerate(blocks):
        lines = [l.strip() for l in b.split('\n') if l.strip()]
        front, back, hint = "", "", ""
        for line in lines:
            if re.search(r'^(?:Q|Front|Question)[:\s]', line, re.IGNORECASE):
                front = re.sub(r'^(?:Q|Front|Question)[:\s]+', '', line, flags=re.IGNORECASE).replace('**', '').strip()
            elif re.search(r'^(?:A|Back|Answer)[:\s]', line, re.IGNORECASE):
                back = re.sub(r'^(?:A|Back|Answer)[:\s]+', '', line, flags=re.IGNORECASE).replace('**', '').strip()
            elif re.search(r'^(?:Hint|Tip)[:\s]', line, re.IGNORECASE):
                hint = re.sub(r'^(?:Hint|Tip)[:\s]+', '', line, flags=re.IGNORECASE).replace('**', '').strip()
        if front and back:
            cards.append({
                "id": idx + 1,
                "front": front,
                "back": back,
                "hint": hint or "Core concept"
            })
    return cards[:num_requested]


# --- LANGGRAPH NODE IMPLEMENTATIONS ---

def input_optimizer_node(state: StudyBuddyState) -> Dict[str, Any]:
    """
    Node 1: Token Saver & Sanitizer
    - Trims redundant whitespace, repeated blank lines, and boilerplate.
    - Limits character boundary to save LLM tokens.
    - Checks cache to skip LLM entirely if identical prompt was computed.
    """
    raw_text = state.get("user_input", "").strip()
    original_len = len(raw_text)
    
    # Check cache
    cache_key = _generate_cache_key(
        state.get("task", "explain"),
        raw_text,
        state.get("num_items", 5),
        state.get("language", "Auto")
    )
    if cache_key in CACHE:
        cached_val = CACHE[cache_key]
        return {
            "cleaned_input": raw_text,
            "raw_response": cached_val.get("raw_response", ""),
            "formatted_html": cached_val.get("formatted_html", ""),
            "clean_text": cached_val.get("clean_text", ""),
            "structured_data": cached_val.get("structured_data"),
            "tokens_saved_estimate": original_len // 4 + 400, # Saved entire input + output
            "is_cached": True,
            "error": None
        }

    # Clean whitespace & repetitive newlines
    cleaned = re.sub(r"[ \t]+", " ", raw_text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    
    # Token-saving cap: Max 7,000 characters (~1,750 tokens)
    if len(cleaned) > 7000:
        cleaned = cleaned[:7000] + "\n\n[Note: Text clipped to maintain optimal study focus]"

    chars_trimmed = max(0, original_len - len(cleaned))
    saved_tokens = chars_trimmed // 4

    return {
        "cleaned_input": cleaned,
        "tokens_saved_estimate": saved_tokens,
        "is_cached": False,
        "error": None
    }


def study_engine_node(state: StudyBuddyState) -> Dict[str, Any]:
    """
    Node 2: LangChain + Google Gemini AI Core
    - Uses gemini-2.5-flash for maximum speed, accuracy, and token efficiency.
    - Strict system instructions avoid fluff ('Sure, here is your answer...').
    - Outputs in target language (English, Hindi, Hinglish, Telugu, Bhojpuri).
    """
    if state.get("is_cached"):
        return {}

    task = state.get("task", "explain")
    content = state.get("cleaned_input", "")
    num_items = state.get("num_items", 5)
    lang_pref = state.get("language", "Auto")

    lang_instruction = (
        "Respond in the exact same language/style as the input. "
        "If input is Hinglish (Hindi in Roman English script), explain in natural, student-friendly Hinglish. "
        "If input is Hindi, explain in Hindi. If English, explain in clear English."
    )
    if lang_pref and lang_pref != "Auto":
        lang_instruction = f"Strictly respond in {lang_pref} (if Hinglish, write Hindi words in Latin/English alphabets)."

    system_prompts = {
        "explain": (
            "You are Study Buddy, an elite exam tutor for Indian students preparing for board exams, JEE, NEET, and university exams.\n"
            f"{lang_instruction}\n\n"
            "TEACHING PHILOSOPHY & QUALITY GUIDELINES:\n"
            "- Make complex concepts instantly understandable, engaging, and memorable.\n"
            "- Explain with high pedagogical clarity: start from intuitive everyday analogies, then move into precise technical definitions.\n"
            "- No greetings, no pleasantries ('Sure, I will explain...'). Start directly with the main title.\n"
            "- STRICT MARKDOWN FORMATTING: Never produce loose or dangling asterisks (**). Every bold word must be strictly formatted like **Keyword** with no extra space inside.\n\n"
            "STRUCTURE YOUR ANSWER IN THIS EXACT ORDER:\n"
            "## 🎯 60-Second Intuition & Overview\n"
            "> A 2-line plain, relatable explanation that anyone can grasp immediately.\n\n"
            "## 📖 Formal Definition & Core Principles\n"
            "- Academic/scientific definition.\n"
            "- Mathematical laws, equations, or scientific statements.\n\n"
            "## 🔍 In-Depth Breakdown & How It Works\n"
            "- Step-by-step mechanism or key properties.\n"
            "- Start each bullet point with a bold keyword, e.g. '- **Property Name**: Detailed explanation.'\n\n"
            "## 🌍 Relatable Everyday Examples\n"
            "- 2-3 vivid real-world examples (e.g. cricket, walking, smartphone sensors, car brakes, rockets, cooking).\n\n"
            "## ⚠️ Common Mistakes & Exam Traps\n"
            "- Highlight common misconceptions where students lose marks.\n\n"
            "## 💡 High-Yield Exam Cheat Sheet\n"
            "- Formulas, mnemonics, or 1-line golden rules to remember."
        ),
        "summarize": (
            "You are Study Buddy, a master exam tutor creating top-tier revision notes and cheat-sheets for students.\n"
            f"{lang_instruction}\n\n"
            "SUMMARY QUALITY & FORMATTING RULES:\n"
            "- Provide a crisp, high-yield, structured revision breakdown.\n"
            "- No fluff or chatter. Start directly with the summary.\n"
            "- STRICT MARKDOWN: Never leave stray or unclosed ** asterisks. Tightly wrap all bold text: **Concept**.\n\n"
            "STRUCTURE YOUR SUMMARY AS FOLLOWS:\n"
            "## 📌 Quick Executive TL;DR\n"
            "> A punchy 2-sentence summary capturing the core essence of the topic.\n\n"
            "## 🔑 Core Concepts & Essential Definitions\n"
            "- Every crucial term explained in 1 crisp bullet.\n"
            "- Start each point with '- **Concept Name**: Definition and role.'\n\n"
            "## ⚡ Formulas, Equations & Processes\n"
            "- High-yield mathematical formulas, reactions, or step-by-step workflows.\n\n"
            "## 💡 Exam Cheat-Sheet & Mnemonics\n"
            "- Most frequently tested facts and quick memory tricks."
        ),
        "quiz": (
            f"Generate exactly {num_items} multiple choice questions from this text. "
            f"{lang_instruction} "
            "You MUST respond ONLY with a valid JSON array of objects. No markdown backticks, no text before or after. "
            "Format:\n"
            '[\n'
            '  {\n'
            '    "question": "Question text here?",\n'
            '    "options": ["Option A", "Option B", "Option C", "Option D"],\n'
            '    "answer": "A",\n'
            '    "explanation": "Clear 1-line reason why this option is correct."\n'
            '  }\n'
            ']\n'
            "Strictly follow this JSON format."
        ),
        "flashcards": (
            f"Generate exactly {num_items} flashcards from this text. "
            f"{lang_instruction} "
            "You MUST respond ONLY with a valid JSON array of objects. No markdown backticks, no text before or after. "
            "Format:\n"
            '[\n'
            '  {\n'
            '    "front": "Clear question or concept prompt",\n'
            '    "back": "Concise, precise answer",\n'
            '    "hint": "Brief memory trigger / mnemonic"\n'
            '  }\n'
            ']\n'
            "Strictly follow this JSON format."
        )
    }

    sys_msg = system_prompts.get(task, system_prompts["explain"])
    api_key = os.getenv("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", "")

    try:
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
            max_output_tokens=2048,
        )
        response = llm.invoke([
            SystemMessage(content=sys_msg),
            HumanMessage(content=content)
        ])
        raw_text = _extract_text_from_response(response.content)
        return {"raw_response": raw_text}
    except Exception as e:
        err_msg = str(e)
        if any(w in err_msg.lower() for w in ["429", "quota", "resource_exhausted"]):
            err_msg = "Daily API quota limit reached. Please try again shortly."
        elif "leaked" in err_msg.lower() or "permission_denied" in err_msg.lower():
            err_msg = "Gemini API Key is invalid or was revoked/reported leaked. Please get a fresh API key from Google AI Studio (https://aistudio.google.com/app/apikey)."
        return {"raw_response": "", "error": f"Study Buddy Engine: {err_msg}"}


def format_cleaner_node(state: StudyBuddyState) -> Dict[str, Any]:
    """
    Node 3: Format & Presentation Optimizer
    - Converts raw markdown into pristine semantic HTML.
    - Strips all raw ** and markdown clutter so the UI looks ultra-clean.
    - Generates structured items for interactive quizzes and 3D flashcards.
    - Generates clean plain text for Text-to-Speech (TTS).
    - Caches output for 100% token savings on repeat queries.
    """
    if state.get("is_cached"):
        return {}

    error = state.get("error")
    if error:
        return {
            "formatted_html": f'<div class="error-banner">⚠️ {error}</div>',
            "clean_text": error,
            "structured_data": None
        }

    raw = _extract_text_from_response(state.get("raw_response", ""))
    task = state.get("task", "explain")
    num_items = state.get("num_items", 5)

    structured_data = None
    if task == "quiz":
        structured_data = _parse_quiz_items(raw, num_items)
        clean_text = "Generated Quiz. Practice the questions on screen."
        formatted_html = "" # Rendered via interactive template
    elif task == "flashcards":
        structured_data = _parse_flashcard_items(raw, num_items)
        clean_text = "Generated Flashcards. Flip the cards to test your knowledge."
        formatted_html = "" # Rendered via interactive template
    else:
        # Explain or Summarize
        formatted_html = _clean_markdown_html(raw)
        clean_text = _clean_markdown_to_plain_text(raw)

    # Save into cache
    cache_key = _generate_cache_key(
        task,
        state.get("user_input", ""),
        num_items,
        state.get("language", "Auto")
    )
    if len(CACHE) >= CACHE_MAX_ENTRIES:
        # Remove oldest entry
        CACHE.pop(next(iter(CACHE)))
    CACHE[cache_key] = {
        "raw_response": raw,
        "formatted_html": formatted_html,
        "clean_text": clean_text,
        "structured_data": structured_data
    }

    return {
        "formatted_html": formatted_html,
        "clean_text": clean_text,
        "structured_data": structured_data
    }


# --- COMPILE LANGGRAPH WORKFLOW ---
def _build_study_graph():
    graph = StateGraph(StudyBuddyState)
    graph.add_node("input_optimizer", input_optimizer_node)
    graph.add_node("study_engine", study_engine_node)
    graph.add_node("format_cleaner", format_cleaner_node)

    graph.add_edge(START, "input_optimizer")
    graph.add_edge("input_optimizer", "study_engine")
    graph.add_edge("study_engine", "format_cleaner")
    graph.add_edge("format_cleaner", END)
    return graph.compile()

study_buddy_graph = _build_study_graph()


# --- PRIMARY CLASS INTERFACE FOR DJANGO VIEWS ---
class StudyGenie:
    """
    Modern Study Buddy engine powered by LangChain + LangGraph.
    Offers:
    - ⚡ Token optimization & smart caching
    - 🌐 Multi-lingual support (English, Hindi, Hinglish, Telugu, Bhojpuri)
    - 🎨 Zero raw asterisks: outputs clean styled HTML and interactive components
    - 🔊 Text-To-Speech (TTS) clean text generation
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def process(self, task: str, user_input: str, num_items: int = 5, language: str = "Auto") -> Dict[str, Any]:
        initial_state: StudyBuddyState = {
            "task": task,
            "user_input": user_input,
            "num_items": num_items,
            "language": language,
            "cleaned_input": "",
            "raw_response": "",
            "formatted_html": "",
            "clean_text": "",
            "structured_data": None,
            "tokens_saved_estimate": 0,
            "is_cached": False,
            "error": None
        }
        result = study_buddy_graph.invoke(initial_state)
        return {
            "html": result.get("formatted_html", ""),
            "raw": result.get("raw_response", ""),
            "clean_text": result.get("clean_text", ""),
            "structured": result.get("structured_data"),
            "tokens_saved": result.get("tokens_saved_estimate", 0),
            "cached": result.get("is_cached", False),
            "error": result.get("error")
        }

    # Backward compatibility methods
    def explain_topic(self, topic: str, language: str = "Auto") -> str:
        res = self.process("explain", topic, language=language)
        return res["html"] or res["raw"]

    def summarize_text(self, notes: str, language: str = "Auto") -> str:
        res = self.process("summarize", notes, language=language)
        return res["html"] or res["raw"]

    def generate_quiz(self, material: str, num_questions: int = 5, language: str = "Auto") -> str:
        res = self.process("quiz", material, num_items=num_questions, language=language)
        return res["raw"]

    def generate_flashcards(self, material: str, num_cards: int = 5, language: str = "Auto") -> str:
        res = self.process("flashcards", material, num_items=num_cards, language=language)
        return res["raw"]

# Singleton instance
genie_instance = StudyGenie()

# Top-level backward compatibility wrappers
def explain_topic(topic: str) -> str:
    return genie_instance.explain_topic(topic)

def summarize_text(notes: str) -> str:
    return genie_instance.summarize_text(notes)

def generate_quiz(material: str, num_questions: int = 5) -> str:
    return genie_instance.generate_quiz(material, num_questions)

def generate_flashcards(material: str, num_cards: int = 5) -> str:
    return genie_instance.generate_flashcards(material, num_cards)
