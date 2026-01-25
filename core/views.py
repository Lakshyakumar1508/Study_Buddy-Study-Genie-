from django.shortcuts import render
from .ai_utils import StudyGenie
import pdfplumber

# Single shared AI instance
genie = StudyGenie()


# ---------- Utility: Extract text from PDF ----------
def extract_pdf_text(pdf_file):
    """
    Extracts text from a PDF file.
    Returns a tuple: (text, error)
    """
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join(
                page.extract_text() or ""
                for page in pdf.pages
            )
        return text, None
    except Exception as e:
        return None, f"PDF Read Error: {e}"


# ---------------- Home ----------------
def home_view(request):
    return render(request, "home.html")


# ---------------- Explain ----------------
def explain_view(request):
    context = {"explanation": "", "error": ""}

    if request.method == "POST":
        text = request.POST.get("user_input", "").strip()
        pdf = request.FILES.get("pdf_file")

        if pdf:
            extracted, error = extract_pdf_text(pdf)
            if error:
                context["error"] = error
                return render(request, "explain.html", context)
            text = extracted

        if not text:
            context["error"] = "Please enter a topic or upload a PDF."
        else:
            context["explanation"] = genie.explain_topic(text)

    return render(request, "explain.html", context)


# ---------------- Summarize ----------------
def summarize_view(request):
    context = {"summary": "", "error": ""}

    if request.method == "POST":
        text = request.POST.get("user_input", "").strip()
        pdf = request.FILES.get("pdf_file")

        if pdf:
            extracted, error = extract_pdf_text(pdf)
            if error:
                context["error"] = error
                return render(request, "summarize.html", context)
            text = extracted

        if not text:
            context["error"] = "Please enter text or upload a PDF."
        else:
            context["summary"] = genie.summarize_text(text)

    return render(request, "summarize.html", context)


# ---------------- Quiz ----------------
def quiz_view(request):
    context = {"quiz": "", "error": ""}

    if request.method == "POST":
        text = request.POST.get("user_input", "").strip()
        try:
            num_questions = int(request.POST.get("num_questions", 5))
        except ValueError:
            num_questions = 5

        if not text:
            context["error"] = "Please enter text to generate a quiz."
        else:
            context["quiz"] = genie.generate_quiz(text, num_questions)

    return render(request, "quiz.html", context)


# ---------------- Flashcards ----------------
def flashcards_view(request):
    context = {"flashcards": "", "error": ""}

    if request.method == "POST":
        text = request.POST.get("user_input", "").strip()
        try:
            num_cards = int(request.POST.get("num_cards", 5))
        except ValueError:
            num_cards = 5

        if not text:
            context["error"] = "Please enter text to generate flashcards."
        else:
            context["flashcards"] = genie.generate_flashcards(text, num_cards)

    return render(request, "flashcards.html", context)
