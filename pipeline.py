"""
Общий пайплайн "вопрос -> найденный контекст", используемый и в main.py
(CLI), и в app.py (веб-API) — чтобы логика поиска не дублировалась и не
расходилась между двумя интерфейсами при будущих правках.
"""
from dataclasses import dataclass, field

from config import SEARCH_K
from llm.base import BaseLLM
from retrieval import build_search_queries, detect_device

PER_QUERY_K = 5   # сколько результатов берём с каждого запроса
RRF_K = 60        # константа Reciprocal Rank Fusion


@dataclass
class RetrievedContext:
    question: str
    detected_device: str | None
    expanded_queries: list[str]
    chunks: list[dict] = field(default_factory=list)

    @property
    def context_text(self) -> str:
        return "\n---\n".join(c["text"] for c in self.chunks)


def build_context(question, index, glossary, llm: BaseLLM, k: int = SEARCH_K, device: str | None = None):
    if not device:
        device = detect_device(question, index.get_known_devices())

    queries = build_search_queries(question, glossary, llm)

    # RRF: фрагмент, найденный по нескольким запросам или высоко по одному, поднимается выше.
    scores: dict[str, float] = {}
    chunks_by_key: dict[str, dict] = {}
    for q in queries:
        for rank, r in enumerate(index.search(q, k=PER_QUERY_K, device_filter=device)):
            key = r["text"][:80]
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            chunks_by_key.setdefault(key, r)

    ranked = sorted(scores, key=scores.get, reverse=True)
    return RetrievedContext(
        question=question,
        detected_device=device,
        expanded_queries=queries,
        chunks=[chunks_by_key[key] for key in ranked[:k]],
    )