import os

from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def summarize_text(text: str) -> str:
    """Summarize educational text in simple language."""

    if client is None:
        return "Error in Summary: GEMINI_API_KEY is not configured."

    try:
        prompt = (
            "Summarize the following text in simple language. "
            "Keep the important information and remove unnecessary repetition.\n\n"
            f"{text}"
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        if not response.text:
            return "Error in Summary: Gemini returned an empty response."

        return response.text.strip()

    except Exception as e:
        return f"Error in Summary: {e}"