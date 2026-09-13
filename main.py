"""
RAG-чат-бот v2 для документации по приборам.

Что добавлено по сравнению с первой версией:
1. Индекс сохраняется на диск — не пересчитывается при перезапуске
2. Документы по новым приборам добавляются в индекс инкрементально,
   с меткой (метаданными), к какому прибору они относятся
3. Эмбеддинги — многоязычная модель (важно для русской документации)
4. Более строгий промпт + возможность посмотреть, что реально нашёл поиск
   (для отладки "бредовых" ответов)

Установка:
    pip install sentence-transformers faiss-cpu pymupdf python-docx requests
"""

import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("./index_data")
INDEX_DIR.mkdir(exist_ok=True)
FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "chunks_meta.pkl"

# Многоязычная модель — важно, если документация на русском.
# Английская all-MiniLM хуже ищет релевантные куски в русских текстах.
EMBED_MODEL = "intfloat/multilingual-e5-base"


# ---------- Загрузка и разбиение документов ----------
def normalize_text(text: str) -> str:
    """
    Чинит типичные артефакты извлечения текста из PDF:
    - слово, разорванное дефисом на конце строки ("слово-\nпродовження" -> "словопродовження")
    - лишние пробелы и переносы строк между буквами/словами
    """
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_documents(folder: str) -> list[tuple[str, str]]:
    """Возвращает список (текст, имя_файла)."""
    result = []
    for path in Path(folder).glob("**/*"):
        if path.suffix.lower() in [".txt", ".md"]:
            result.append((normalize_text(path.read_text(encoding="utf-8")), path.name))
        # elif path.suffix.lower() == ".pdf":
        #     import fitz  # PyMuPDF — извлекает текст надёжнее pypdf на сложных макетах
        #     doc = fitz.open(str(path))
        #     text = "\n".join(page.get_text() for page in doc)
        #     result.append((normalize_text(text), path.name))
        elif path.suffix.lower() == ".docx":
            # DOCX — это XML с реальной структурой текста, парсится намного
            # надёжнее PDF (нет проблемы "угадать порядок символов")
            import docx
            document = docx.Document(str(path))
            parts = [p.text for p in document.paragraphs if p.text.strip()]
            for table in document.tables:
                for row in table.rows:
                    parts.append(" | ".join(cell.text for cell in row.cells))
            text = "\n".join(parts)
            result.append((normalize_text(text), path.name))
    return result


def chunk_text(text: str, target_size: int = 500, overlap_sentences: int = 1) -> list[str]:
    """
    Режем по предложениям, а не по "сырым" символам — иначе в один чанк
    попадает хвост одной мысли и начало другой (это размывает эмбеддинг
    и портит поиск). Собираем предложения в чанк, пока не наберём
    примерно target_size символов, затем начинаем новый с небольшим
    перехлёстом (overlap_sentences), чтобы не терять связку на границе.
    """
    # Разбиение на предложения по украинской/русской пунктуации
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        current.append(sentence)
        current_len += len(sentence)
        if current_len >= target_size:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) for s in current)

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------- Индекс с метаданными и персистентностью ----------
class DocumentIndex:
    def __init__(self, model_name: str = EMBED_MODEL):
        self.embedder = SentenceTransformer(model_name)
        self.chunks: list[str] = []
        self.metadata: list[dict] = []  # {"device": ..., "source_file": ...}
        self.index = None

    def exists_on_disk(self) -> bool:
        return FAISS_PATH.exists() and META_PATH.exists()

    def load(self):
        self.index = faiss.read_index(str(FAISS_PATH))
        with open(META_PATH, "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]
        print(f"Загружен индекс: {len(self.chunks)} чанков")

    def save(self):
        faiss.write_index(self.index, str(FAISS_PATH))
        with open(META_PATH, "wb") as f:
            pickle.dump({"chunks": self.chunks, "metadata": self.metadata}, f)

    def add_documents(self, folder: str, device_name: str):
        """Добавляет документацию по конкретному прибору в индекс (не пересоздавая его)."""
        docs = load_documents(folder)
        new_chunks = []
        new_meta = []
        for text, filename in docs:
            for chunk in chunk_text(text):
                new_chunks.append(chunk)
                new_meta.append({"device": device_name, "source_file": filename})

        if not new_chunks:
            print(f"В папке {folder} не нашлось документов")
            return

        # Модель e5 обучена с префиксом "passage: " для индексируемых текстов
        # и "query: " для вопросов — без этого эмбеддинги существенно хуже
        prefixed = [f"passage: {c}" for c in new_chunks]
        embeddings = self.embedder.encode(
            prefixed, show_progress_bar=True, normalize_embeddings=True
        )
        embeddings = np.array(embeddings).astype("float32")

        if self.index is None:
            dim = embeddings.shape[1]
            # IndexFlatIP на нормализованных векторах = косинусное сходство,
            # это метрика, под которую обучена e5 (не L2!)
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        self.metadata.extend(new_meta)
        self.save()
        print(f"Добавлено {len(new_chunks)} чанков по прибору «{device_name}»")

    def search(self, query: str, k: int = 5, device_filter: str | None = None) -> list[dict]:
        q_emb = self.embedder.encode(
            [f"query: {query}"], normalize_embeddings=True
        ).astype("float32")
        search_k = k * 5 if device_filter else k
        _, idx = self.index.search(q_emb, min(search_k, len(self.chunks)))

        results = []
        for i in idx[0]:
            if device_filter and self.metadata[i]["device"] != device_filter:
                continue
            results.append({"text": self.chunks[i], **self.metadata[i]})
            if len(results) >= k:
                break
        return results


# ---------- LLM через Ollama (рекомендуемый вариант) ----------
class OllamaLLM:
    """
    Требует запущенной Ollama (локально: `ollama serve`, модель: `ollama pull mistral`).
    На сервере — тот же код, меняется только base_url.
    """
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "mistral"):
        import requests
        self.requests = requests
        self.base_url = base_url
        self.model_name = model_name

    def answer(self, question: str, context_chunks: list[dict]) -> str:
        context = "\n---\n".join(c["text"] for c in context_chunks)
        prompt = build_prompt(question, context)
        resp = self.requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},  # низкая температура для технических ответов
            },
        )
        return resp.json()["response"].strip()


def build_prompt(question: str, context: str) -> str:
    return f"""Ты — технический помощник по документации приборов.
СТРОГО следуй правилам:
- Отвечай ТОЛЬКО на основе контекста ниже.
- Если в контексте нет ответа — прямо скажи "В документации нет информации по этому вопросу", не выдумывай.
- Не додумывай технические характеристики, которых нет в тексте.

Контекст:
{context}

Вопрос: {question}
Ответ:"""


# ---------- Основной сценарий ----------
def main():
    index = DocumentIndex()

    if index.exists_on_disk():
        index.load()
    else:
        print("Индекс не найден, создаю новый и добавляю документы...")
        index.add_documents("./docs/MKS_UM", device_name="МКС-УМ")

    llm = OllamaLLM(model_name="mistral")

    print("\nГотово! Команды: 'exit' — выход, 'add <папка> <прибор>' — добавить документацию")
    while True:
        question = input("\nВопрос: ").strip()
        if question.lower() == "exit":
            break
        if question.lower().startswith("add "):
            _, folder, device = question.split(maxsplit=2)
            index.add_documents(folder, device)
            continue

        results = index.search(question, k=3)

        # Отладка: если ответы "бредовые" — сначала смотрим сюда,
        # действительно ли найденные куски релевантны вопросу
        print("\n[Найденные фрагменты]")
        for r in results:
            print(f"  ({r['device']} / {r['source_file']}): {r['text'][:100]}...")

        answer = llm.answer(question, results)
        print(f"\nОтвет: {answer}")


if __name__ == "__main__":
    main()