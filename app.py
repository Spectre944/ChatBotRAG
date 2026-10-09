import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import LLM_MODEL, LLM_PROVIDER
from debug_logger import DebugLogger
from llm import BaseLLM, build_prompt, create_llm
from pipeline import build_context
from retrieval import DocumentIndex, load_glossary

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

index = DocumentIndex()
if not index.exists_on_disk():
    raise RuntimeError("Индекс не найден — сначала соберите его через main.py")
index.load()

glossary = load_glossary()
llm: BaseLLM = create_llm(LLM_PROVIDER, LLM_MODEL)
debug_logger = DebugLogger()


class ChatRequest(BaseModel):
    question: str
    device: str | None = None  # Явная передача прибора со страницы


class RatingRequest(BaseModel):
    session_id: str
    rating: int = Field(..., ge=1, le=5)  # Оценка от 1 до 5
    comment: str | None = None


def stream_answer(question: str, device: str | None = None):
    # Дополняем контекст выбранным прибором, если он передан
    query_for_context = f"Прибор: {device}. Вопрос: {question}" if device else question
    
    ctx = build_context(query_for_context, index, glossary, llm)
    
    # Если прибор передан явно, фиксируем его
    if device:
        ctx.detected_device = device

    prompt = build_prompt(question, ctx.context_text)

    tokens: list[str] = []
    for token in llm.stream_answer(question, ctx.chunks):
        tokens.append(token)

        # Экранируем \n, чтобы SSE не воспринимал их как разделители пакетов
        safe_token = token.replace("\n", "\\n")
        yield f"data: {safe_token}\n\n"

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
    return StreamingResponse(
        stream_answer(req.question, req.device), media_type="text/event-stream"
    )


@app.post("/rate")
def rate_answer(req: RatingRequest):
    # Сохранение или логирование оценки для анализа и отладки
    debug_logger.log(
        session_id=req.session_id,
        rating=req.rating,
        comment=req.comment
    )


@app.get("/")
def root():
    return FileResponse("index.html")

@app.get("/spectra")
def get_spectra():
    return FileResponse("pages/spectra.html")

@app.get("/guarder")
def get_guarder():
    return FileResponse("pages/guarder.html")

@app.get("/mksum")
def get_guarder():
    return FileResponse("pages/mksum.html")

app.mount("/", StaticFiles(directory="pages"), name="pages")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)