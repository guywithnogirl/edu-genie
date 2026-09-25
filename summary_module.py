import os

import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def summarize_text(text: str) -> str:
    """Summarize educational text in simple language."""
    if not GEMINI_API_KEY:
        return "Error in Summary: GEMINI_API_KEY is not configured."

    try:
        model = genai.GenerativeModel(model_name=GEMINI_MODEL)

        prompt = (
            "Summarize the following text in simple language. "
            "Keep the important information and remove unnecessary repetition.\n\n"
            f"{text}"
        )

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"Error in Summary: {e}"
