import json
import os
import re

import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def clean_json_block(text: str) -> str:
    """Remove Markdown code fences around a JSON response."""
    return re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.IGNORECASE).strip()


def generate_quiz(text: str) -> list:
    """Generate three MCQs with four options each."""
    if not GEMINI_API_KEY:
        return [{"error": "GEMINI_API_KEY is not configured."}]

    try:
        model = genai.GenerativeModel(model_name=GEMINI_MODEL)

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

        response = model.generate_content(prompt)
        quiz_text = response.text.strip()
        cleaned_text = clean_json_block(quiz_text)

        quiz = json.loads(cleaned_text)

        if not isinstance(quiz, list):
            raise ValueError("Gemini did not return a JSON list.")

        return quiz

    except Exception as e:
        return [{"error": f"Quiz generation failed: {e}"}]
