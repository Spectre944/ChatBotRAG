import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal

from config import LLM_MODEL, LLM_PROVIDER
from debug_logger import DebugLogger
from llm import BaseLLM, build_prompt, create_llm
from pipeline import build_context
from retrieval import DocumentIndex, load_glossary

DEVICE_NAMES = {
    "spectra": "Spectra",   # подставьте реальные названия
    "guarder": "Guarder",
    "mksum":   "MKSUM",
}

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
    device: str | None = None
    session_id: str | None = None


class RatingRequest(BaseModel):
    message_id: str
    session_id: str | None = None
    rating: Literal[-1, 1]          # дизлайк / лайк
    comment: str | None = None


def stream_answer(question, device_name=None, session_id=None):
    message_id = uuid.uuid4().hex
    yield f"data: [ID:{message_id}]\n\n"

    ctx = build_context(question, index, glossary, llm, device=device_name)
    prompt = build_prompt(question, ctx.context_text)

    tokens = []
    try:
        for token in llm.stream_answer(question, ctx.chunks):
            tokens.append(token)
            yield f"data: {token.replace(chr(10), '\\n')}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        debug_logger.log({
            "type": "answer",
            "message_id": message_id,
            "session_id": session_id,
            "question": question,
            "device": ctx.detected_device,
            "expanded_queries": ctx.expanded_queries,
            "retrieved_chunks": [
                {
                    "source_file": c.get("source_file"),
                    "device": c.get("device"),
                    "type": c.get("type"),
                    "text_preview": c.get("text", "")[:200],
                }
                for c in ctx.chunks
            ],
            "prompt": prompt,
            "answer": "".join(tokens),
        })

DEVICE_SLUGS = {
    "spectra": "МКС-11 Spectra",
    "guarder": "ДКГ-24 Guarder",
    "mksum":   "МКС-УМ",
}

def resolve_device(value: str | None) -> str | None:
    if not value:
        return None
    known = index.get_known_devices()
    if value in known:                      # пришло полное название
        return value
    name = DEVICE_SLUGS.get(value.lower())  # пришёл слаг
    return name if name in known else None

@app.post("/chat")
def chat(req: ChatRequest):
    print("device from client:", repr(req.device))
    print("known devices:", index.get_known_devices())
    return StreamingResponse(
        stream_answer(req.question, resolve_device(req.device), req.session_id),
        media_type="text/event-stream",
    )


@app.post("/rate")
def rate_answer(req: RatingRequest):
    debug_logger.log({
        "type": "rating",
        "message_id": req.message_id,
        "session_id": req.session_id,
        "rating": req.rating,
        "comment": req.comment,
    })

    return {"status": "ok"}


@app.get("/")
def root():
    return FileResponse("pages/index.html")

@app.get("/spectra")
def get_spectra():
    return FileResponse("pages/spectra.html")

@app.get("/guarder")
def get_guarder():
    return FileResponse("pages/guarder.html")

@app.get("/mksum")
def get_mksum():
    return FileResponse("pages/mksum.html")

app.mount("/", StaticFiles(directory="pages"), name="pages")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)