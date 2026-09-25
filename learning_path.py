import os

from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def get_learning_recommendations(topic: str) -> str:
    """Generate a structured beginner-to-advanced learning path."""

    if client is None:
        return "Error: GEMINI_API_KEY is not configured."

    prompt = f"""
You are an AI tutor. The student wants to learn about: {topic}

Suggest a structured and practical learning path including:
- Beginner level
- Intermediate level
- Advanced level
- Key topics in each level
- Suggested learning order
- Useful resources such as videos, articles, or books
- Approximate timelines where appropriate
- Practical exercises or projects

Keep the guidance clear and adaptable to the learner's level.
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        if response.text:
            return response.text.strip()

        return "Could not extract content from Gemini response."

    except Exception as e:
        return f"Error occurred: {e}"