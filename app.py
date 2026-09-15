"""
Веб-API для RAG-чат-бота со стримингом ответа в реальном времени.

Запуск:
    pip install fastapi uvicorn
    uvicorn app:app --host 0.0.0.0 --port 8000

Использует классы и функции из rag_chatbot_v2.py — оставьте оба файла
(и glossary.json) в одной папке.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from main import (
    ClaudeLLM,
    DocumentIndex,
    OllamaLLM,
    expand_query,
    expand_query_with_glossary,
    load_glossary,
)

app = FastAPI()

# Ограничьте allow_origins доменом вашего сайта в проде — сейчас открыто для теста
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Индекс и словарь грузятся один раз при старте сервера, а не на каждый запрос
index = DocumentIndex()
if index.exists_on_disk():
    index.load()
else:
    raise RuntimeError("Индекс не найден — сначала соберите его через rag_chatbot_v2.py")

glossary = load_glossary()

# Переключение backend'а — достаточно поменять эту строку:
# llm = ClaudeLLM(model="claude-haiku-4-5-20251001")  # использует переменную окружения ANTHROPIC_API_KEY
llm = OllamaLLM(model_name="mistral")   # локальный/арендованный вариант


class ChatRequest(BaseModel):
    question: str


def build_context(question: str) -> tuple[str, list[dict]]:
    glossary_variants = expand_query_with_glossary(question, glossary)
    queries = [question] + glossary_variants if glossary_variants else expand_query(question, llm)

    seen = set()
    results = []
    for q in queries:
        for r in index.search(q, k=3):
            key = r["text"][:80]
            if key not in seen:
                seen.add(key)
                results.append(r)
    results = results[:5]
    context = "\n---\n".join(r["text"] for r in results)
    return context, results


def stream_answer(question: str):
    """Генератор SSE-сообщений — работает одинаково что с Ollama, что с Claude API,
    потому что оба класса реализуют одинаковый метод stream_answer(question, context_chunks)."""
    _, results = build_context(question)
    for token in llm.stream_answer(question, results):
        yield f"data: {token}\n\n"
    yield "data: [DONE]\n\n"

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
def chat(req: ChatRequest):
    return StreamingResponse(stream_answer(req.question), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)