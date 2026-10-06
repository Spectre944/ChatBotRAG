"""
Центральная конфигурация проекта.

Все пути, названия моделей и параметры поиска собраны здесь — чтобы не
разыскивать "магические числа" по всему коду и менять поведение бота в
одном месте, а не в десятке файлов.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Пути ---
INDEX_DIR = PROJECT_ROOT / "index_data"
FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "chunks_meta.pkl"
GLOSSARY_PATH = PROJECT_ROOT / "glossary.json"
API_KEY_FILE = PROJECT_ROOT / "api_key.txt"
LOG_DIR = PROJECT_ROOT / "logs"
DEBUG_LOG_PATH = LOG_DIR / "debug.jsonl"

# --- LLM ---
# Небольшая/бюджетная модель по умолчанию. Для ответов по уже найденному
# контексту "ума" топовой модели обычно не требуется — основное качество
# определяет retrieval, а не размер модели (см. всю историю отладки проекта).
LLM_PROVIDER = "openai"  # "claude" | "openai" | "ollama"
LLM_MODEL = "GPT-5 nano"
OLLAMA_BASE_URL = "http://localhost:11434"

# --- Эмбеддинги и поиск ---
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
CHUNK_TARGET_SIZE = 500
CHUNK_OVERLAP_SENTENCES = 1
SEARCH_K = 5
USE_MMR = True
MMR_LAMBDA = 0.7  # ближе к 1 — приоритет релевантности, ближе к 0 — разнообразия

# --- Отладка ---
# При DEBUG_MODE = True каждый вопрос (с найденными чанками, промптом и
# ответом) дописывается в LOG_DIR/debug.jsonl для последующего разбора.
DEBUG_MODE = True
