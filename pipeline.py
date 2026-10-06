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
    detected_device: str | None
    expanded_queries: list[str]
    chunks: list[dict] = field(default_factory=list)

    @property
    def context_text(self) -> str:
        return "\n---\n".join(c["text"] for c in self.chunks)


def build_context(question: str, index, glossary: dict, llm: BaseLLM, k: int = SEARCH_K) -> RetrievedContext:
    """
    1) Определяем прибор по вопросу (если упомянут) — чтобы искать только
       в его документации, а не по всей базе.
    2) Сначала пробуем словарь терминов (быстро, предсказуемо); LLM для
       переформулировки вопроса подключаем только если словарь не помог.
    3) Ищем по всем вариантам запроса, объединяя результаты без дублей.
    """
    device = detect_device(question, index.get_known_devices())

    glossary_variants = expand_query_with_glossary(question, glossary)
    queries = [question] + glossary_variants if glossary_variants else expand_query(question, llm)

    seen_texts = set()
    results = []
    for q in queries:
        for r in index.search(q, k=3, device_filter=device):
            key = r["text"][:80]
            if key not in seen_texts:
                seen_texts.add(key)
                results.append(r)
    results = results[:k]

    return RetrievedContext(
        question=question,
        detected_device=device,
        expanded_queries=queries,
        chunks=results,
    )
