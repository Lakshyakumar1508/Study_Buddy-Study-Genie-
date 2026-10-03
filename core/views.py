import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .ai_utils import StudyGenie
import pdfplumber

# Single shared AI instance powered by LangGraph + LangChain
genie = StudyGenie()

def extract_pdf_text(pdf_file, max_chars: int = 25000):
    """
    Extracts text from an uploaded PDF file safely and efficiently.
    Limits character count to optimize token consumption.
    """
    try:
        text_parts = []
        with pdfplumber.open(pdf_file) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text.strip())
                # Stop reading if we already reached a solid chunk of text to save memory & tokens
                if sum(len(p) for p in text_parts) >= max_chars:
                    break
        
        full_text = "\n\n".join(text_parts).strip()
        if not full_text:
            return None, "No readable text found in PDF. Make sure it is not a scanned image PDF."
        
        return full_text, None
    except Exception as e:
        return None, f"PDF Extraction Error: {str(e)}"


# ---------------- Home ----------------
def home_view(request):
    return render(request, "home.html")


# ---------------- Fast AJAX API Endpoint for Lightning Fast Responses ----------------
def api_study_view(request):
    """
    Ultra-fast API endpoint that allows the frontend to request AI study notes
    via asynchronous fetch() without full-page reloads.
    Supports both text inputs and PDF uploads.
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "POST method required"}, status=405)

    try:
        task = request.POST.get("task", "explain")
        user_input = request.POST.get("user_input", "").strip()
        language = request.POST.get("language", "Auto")
        
        try:
            num_items = int(request.POST.get("num_items", 5))
        except (ValueError, TypeError):
            num_items = 5

        # Handle PDF upload if provided
        if "pdf_file" in request.FILES:
            pdf_file = request.FILES["pdf_file"]
            extracted_text, error = extract_pdf_text(pdf_file)
            if error:
                return JsonResponse({"success": False, "error": error}, status=400)
            user_input = extracted_text

        if not user_input:
            return JsonResponse({"success": False, "error": "Please provide topic text or upload a PDF."}, status=400)

        # Run LangGraph pipeline
        result = genie.process(
            task=task,
            user_input=user_input,
            num_items=num_items,
            language=language
        )

        if result.get("error"):
            return JsonResponse({"success": False, "error": result["error"]}, status=500)

        return JsonResponse({
            "success": True,
            "task": task,
            "html": result.get("html", ""),
            "clean_text": result.get("clean_text", ""),
            "structured": result.get("structured"),
            "tokens_saved": result.get("tokens_saved", 0),
            "cached": result.get("cached", False)
        })

    except Exception as e:
        return JsonResponse({"success": False, "error": f"Internal Server Error: {str(e)}"}, status=500)


# ---------------- Explain View ----------------
def explain_view(request):
    context = {
        "explanation_html": "",
        "clean_tts_text": "",
        "tokens_saved": 0,
        "is_cached": False,
        "language": "Auto",
        "error": "",
        "user_input": ""
    }

    if request.method == "POST":
        text = request.POST.get("user_input", "").strip()
        language = request.POST.get("language", "Auto")
        context["language"] = language
        context["user_input"] = text
        pdf = request.FILES.get("pdf_file")

        if pdf:
            extracted, error = extract_pdf_text(pdf)
            if error:
                context["error"] = error
                return render(request, "explain.html", context)
            text = extracted
            context["user_input"] = text[:300] + ("..." if len(text) > 300 else "")

        if not text:
            context["error"] = "Please enter a topic or upload a PDF."
        else:
            res = genie.process("explain", text, language=language)
            if res.get("error"):
                context["error"] = res["error"]
            else:
                context["explanation_html"] = res["html"]
                context["clean_tts_text"] = res["clean_text"]
                context["tokens_saved"] = res["tokens_saved"]
                context["is_cached"] = res["cached"]

    return render(request, "explain.html", context)


# ---------------- Summarize View ----------------
def summarize_view(request):
    context = {
        "summary_html": "",
        "clean_tts_text": "",
        "tokens_saved": 0,
        "is_cached": False,
        "language": "Auto",
        "error": "",
        "user_input": ""
    }

    if request.method == "POST":
        text = request.POST.get("user_input", "").strip()
        language = request.POST.get("language", "Auto")
        context["language"] = language
        context["user_input"] = text
        pdf = request.FILES.get("pdf_file")

        if pdf:
            extracted, error = extract_pdf_text(pdf)
            if error:
                context["error"] = error
                return render(request, "summarize.html", context)
            text = extracted
            context["user_input"] = text[:300] + ("..." if len(text) > 300 else "")

        if not text:
            context["error"] = "Please enter text or upload a PDF to summarize."
        else:
            res = genie.process("summarize", text, language=language)
            if res.get("error"):
                context["error"] = res["error"]
            else:
                context["summary_html"] = res["html"]
                context["clean_tts_text"] = res["clean_text"]
                context["tokens_saved"] = res["tokens_saved"]
                context["is_cached"] = res["cached"]

    return render(request, "summarize.html", context)


# ---------------- Quiz View ----------------
def quiz_view(request):
    context = {
        "quiz_items": [],
        "quiz_json": "[]",
        "clean_tts_text": "",
        "tokens_saved": 0,
        "is_cached": False,
        "num_questions": 5,
        "language": "Auto",
        "error": "",
        "user_input": ""
    }

    if request.method == "POST":
        text = request.POST.get("user_input", "").strip()
        language = request.POST.get("language", "Auto")
        pdf = request.FILES.get("pdf_file")
        try:
            num_questions = int(request.POST.get("num_questions", 5))
        except ValueError:
            num_questions = 5

        context["num_questions"] = num_questions
        context["language"] = language
        context["user_input"] = text

        if pdf:
            extracted, error = extract_pdf_text(pdf)
            if error:
                context["error"] = error
                return render(request, "quiz.html", context)
            text = extracted
            context["user_input"] = text[:300] + ("..." if len(text) > 300 else "")

        if not text:
            context["error"] = "Please enter study notes or upload a PDF to generate a quiz."
        else:
            res = genie.process("quiz", text, num_items=num_questions, language=language)
            if res.get("error"):
                context["error"] = res["error"]
            else:
                items = res.get("structured") or []
                context["quiz_items"] = items
                context["quiz_json"] = json.dumps(items)
                context["clean_tts_text"] = res["clean_text"]
                context["tokens_saved"] = res["tokens_saved"]
                context["is_cached"] = res["cached"]

    return render(request, "quiz.html", context)


# ---------------- Flashcards View ----------------
def flashcards_view(request):
    context = {
        "flashcard_items": [],
        "flashcards_json": "[]",
        "clean_tts_text": "",
        "tokens_saved": 0,
        "is_cached": False,
        "num_cards": 5,
        "language": "Auto",
        "error": "",
        "user_input": ""
    }

    if request.method == "POST":
        text = request.POST.get("user_input", "").strip()
        language = request.POST.get("language", "Auto")
        pdf = request.FILES.get("pdf_file")
        try:
            num_cards = int(request.POST.get("num_cards", 5))
        except ValueError:
            num_cards = 5

        context["num_cards"] = num_cards
        context["language"] = language
        context["user_input"] = text

        if pdf:
            extracted, error = extract_pdf_text(pdf)
            if error:
                context["error"] = error
                return render(request, "flashcards.html", context)
            text = extracted
            context["user_input"] = text[:300] + ("..." if len(text) > 300 else "")

        if not text:
            context["error"] = "Please enter notes or upload a PDF to generate flashcards."
        else:
            res = genie.process("flashcards", text, num_items=num_cards, language=language)
            if res.get("error"):
                context["error"] = res["error"]
            else:
                items = res.get("structured") or []
                context["flashcard_items"] = items
                context["flashcards_json"] = json.dumps(items)
                context["clean_tts_text"] = res["clean_text"]
                context["tokens_saved"] = res["tokens_saved"]
                context["is_cached"] = res["cached"]

    return render(request, "flashcards.html", context)
