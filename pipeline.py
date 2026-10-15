"""
Общий пайплайн "вопрос -> найденный контекст", используемый и в main.py
(CLI), и в app.py (веб-API) — чтобы логика поиска не дублировалась и не
расходилась между двумя интерфейсами при будущих правках.
"""
from dataclasses import dataclass, field

from config import SEARCH_K
from llm.base import BaseLLM
from retrieval import detect_device, expand_query, expand_query_with_glossary


@dataclass
class RetrievedContext:
    question: str
    device_filter: str | None
    device_source: str | None          # "ui" | "auto" | None
    expanded_queries: list[str]
    chunks: list[dict] = field(default_factory=list)

    @property
    def context_text(self) -> str:
        return "\n---\n".join(c["text"] for c in self.chunks)


def build_context(question, index, glossary, llm: BaseLLM, k: int = SEARCH_K, device: str | None = None):
    if device:
        device_source = "ui"
    else:
        device = detect_device(question, index.get_known_devices())
        device_source = "auto" if device else None

    glossary_variants = expand_query_with_glossary(question, glossary)
    queries = [question] + glossary_variants if glossary_variants else expand_query(question, llm)

    seen_texts, results = set(), []
    for q in queries:
        for r in index.search(q, k=3, device_filter=device):
            key = r["text"][:80]
            if key not in seen_texts:
                seen_texts.add(key)
                results.append(r)

    return RetrievedContext(
        question=question,
        device_filter=device,
        device_source=device_source,
        expanded_queries=queries,
        chunks=results[:k],
    )