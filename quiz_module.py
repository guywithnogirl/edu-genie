import json
import os
import re

from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def clean_json_block(text: str) -> str:
    """Remove Markdown code fences around a JSON response."""
    return re.sub(
        r"```(?:json)?\s*|\s*```",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def generate_quiz(text: str) -> list:
    """Generate three MCQs with four options each."""

    if client is None:
        return [{"error": "GEMINI_API_KEY is not configured."}]

    try:
        prompt = f"""
You are a quiz generator.

From the following passage or topic, create exactly 3 multiple-choice questions.
Each question must include:
- A "question" field
- An "options" field containing exactly 4 options
- An "answer" field containing the correct answer, exactly matching one option

Return only valid JSON. Do not include Markdown or explanations.

Example format:
[
  {{
    "question": "What is ...?",
    "options": ["A", "B", "C", "D"],
    "answer": "A"
  }}
]

Passage/topic:
{text}
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        quiz_text = response.text.strip()
        cleaned_text = clean_json_block(quiz_text)

        quiz = json.loads(cleaned_text)

        if not isinstance(quiz, list):
            raise ValueError("Gemini did not return a JSON list.")

        return quiz

    except Exception as e:
        return [{"error": f"Quiz generation failed: {e}"}]