from google import genai
from google.genai import types
from django.conf import settings

class StudyGenie:
    """
    Bug-free, multilingual Study Genie.
    Features:
    - English, Hindi, Hinglish, Telugu, Bhojpuri support
    - Explain, summarize, quiz, flashcards
    - Hinglish optimized for students
    - Single API call per request (stable)
    """

    def __init__(self, api_key: str = None):
        self.client = genai.Client(api_key=api_key or settings.GEMINI_API_KEY)
        self.config = types.GenerateContentConfig(
            system_instruction=(
                "You are Study Genie, a friendly AI tutor for Indian students. "
                "You can respond in English, Hindi, Hinglish (Hindi words in Latin script), Telugu, or Bhojpuri. "
                "Use headings, bullet points, short paragraphs, simple examples. "
                "Exam-focused, student-friendly, avoid unnecessary blank lines or markdown clutter. "
                "Output in the same language as the input. Hinglish must be simple and easy to read."
            ),
            temperature=0.7,
        )

    def _call_model(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=self.config
            )
            return response.text
        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ["429", "quota", "resource_exhausted"]):
                return "Genie is resting 😴 Daily limit reached. Try again later."
            return f"Genie Error: {str(e)}"

    def explain_topic(self, topic: str) -> str:
        prompt = (
            f"Explain the topic '{topic}' clearly for students. "
            "Include headings, bullet points, and examples. "
            "Respond in the same language as input. "
            "If input is Hinglish, explain in student-friendly Hinglish."
        )
        return self._call_model(prompt)

    def summarize_text(self, notes: str) -> str:
        prompt = (
            f"Summarize these notes into key points for quick revision:\n\n{notes}\n\n"
            "Respond in the same language as input. "
            "If input is Hinglish, summarize in Hinglish."
        )
        return self._call_model(prompt)

    def generate_quiz(self, material: str, num_questions: int = 5) -> str:
        prompt = (
            f"Create a {num_questions}-question multiple choice quiz with answers based on this material:\n\n{material}\n\n"
            "Respond in the same language as input. "
            "If input is Hinglish, provide quiz in Hinglish."
        )
        return self._call_model(prompt)

    def generate_flashcards(self, material: str, num_cards: int = 5) -> str:
        prompt = (
            f"Create {num_cards} flashcards in Q&A format from this material:\n\n{material}\n\n"
            "Respond in the same language as input. "
            "If input is Hinglish, provide flashcards in Hinglish."
        )
        return self._call_model(prompt)

# ---------------- SINGLETON INSTANCE ----------------
genie_instance = StudyGenie()

# ---------------- BACKWARD COMPATIBILITY ----------------
def explain_topic(topic: str) -> str:
    return genie_instance.explain_topic(topic)

def summarize_text(notes: str) -> str:
    return genie_instance.summarize_text(notes)

def generate_quiz(material: str, num_questions: int = 5) -> str:
    return genie_instance.generate_quiz(material, num_questions)

def generate_flashcards(material: str, num_cards: int = 5) -> str:
    return genie_instance.generate_flashcards(material, num_cards)
