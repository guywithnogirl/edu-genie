import os

from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

if GEMINI_API_KEY:
    client = genai.Client(api_key="GEMINI_API_KEY")

def answer_question_with_gemini(question: str) -> str:
    """Answer an academic/general-knowledge question using Gemini."""
    if not GEMINI_API_KEY:
        return "Error in QnA: GEMINI_API_KEY is not configured."

    try:
        model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        response = model.generate_content(question)
        return response.text.strip()
    except Exception as e:
        return f"Error in QnA: {e}"
