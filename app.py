"""
Веб-API для RAG-чат-бота со стримингом ответа в реальном времени.

Запуск:
    pip install fastapi uvicorn
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import LLM_MODEL, LLM_PROVIDER
from debug_logger import DebugLogger
from llm import BaseLLM, build_prompt, create_llm
from pipeline import build_context
from retrieval import DocumentIndex, load_glossary

app = FastAPI()

# Ограничьте allow_origins доменом вашего сайта в проде — сейчас открыто для теста
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Индекс, словарь и LLM создаются один раз при старте сервера, а не на каждый запрос
index = DocumentIndex()
if not index.exists_on_disk():
    raise RuntimeError("Индекс не найден — сначала соберите его через main.py")
index.load()

glossary = load_glossary()
llm: BaseLLM = create_llm(LLM_PROVIDER, LLM_MODEL)
debug_logger = DebugLogger()


class ChatRequest(BaseModel):
    question: str


def stream_answer(question: str):
    """Генератор SSE-сообщений — работает одинаково для любого backend'а
    (Claude/OpenAI/Ollama), все они реализуют общий интерфейс BaseLLM."""
    ctx = build_context(question, index, glossary, llm)
    prompt = build_prompt(question, ctx.context_text)

    tokens: list[str] = []
    for token in llm.stream_answer(question, ctx.chunks):
        tokens.append(token)
        yield f"data: {token}\n\n"
    yield "data: [DONE]\n\n"

    debug_logger.log(
        question=question,
        detected_device=ctx.detected_device,
        expanded_queries=ctx.expanded_queries,
        retrieved_chunks=ctx.chunks,
        prompt=prompt,
        answer="".join(tokens),
    )


@app.post("/chat")
def chat(req: ChatRequest):
    return StreamingResponse(stream_answer(req.question), media_type="text/event-stream")

@app.get("/")
def root():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
