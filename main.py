import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from explanation_module import explain_topic
from qna import answer_question_with_gemini
from quiz_module import generate_quiz
from summary_module import summarize_text
from learning_path import get_learning_recommendations

app = FastAPI(title="EduGenie - Google Gemini Powered Learning Assistant")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/qa")
async def qa(question: str):
    question = question.strip()
    if not question:
        return JSONResponse(
            content={"error": "Please provide a question."},
            status_code=400,
        )

    answer = answer_question_with_gemini(question)
    return {"question": question, "answer": answer}


@app.post("/explain")
async def explain(request: Request):
    data = await request.json()
    topic = str(data.get("topic", "")).strip()

    if not topic:
        return JSONResponse(
            content={"error": "Please provide a topic."},
            status_code=400,
        )

    explanation = explain_topic(topic)
    return {"topic": topic, "explanation": explanation}


@app.post("/summarize")
async def summarize(request: Request):
    data = await request.json()
    text = str(data.get("text", "")).strip()

    if not text:
        return JSONResponse(
            content={"error": "Please provide text to summarize."},
            status_code=400,
        )

    summary = summarize_text(text)
    return {"summary": summary}


@app.post("/quiz")
async def quiz(request: Request):
    data = await request.json()
    text = str(data.get("text", "")).strip()

    if not text:
        return JSONResponse(
            content={"error": "Please provide text or a topic for the quiz."},
            status_code=400,
        )

    quiz_data = generate_quiz(text)
    return {"quiz": quiz_data}


@app.get("/learn/recommendations")
async def learning_recommendations(topic: str):
    topic = topic.strip()

    if not topic:
        return JSONResponse(
            content={"error": "Please provide a learning topic."},
            status_code=400,
        )

    recommendation = get_learning_recommendations(topic)
    return {"topic": topic, "recommendation": recommendation}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
