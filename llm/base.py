"""
Общий интерфейс для всех LLM-провайдеров (Claude / OpenAI / Ollama).

Да, классы Claude/OpenAI/Ollama очень похожи по структуре — это осознанно:
любой новый провайдер, реализующий эти три метода, подставляется в
main.py/app.py без единой правки остального кода. Абстрактный класс здесь
не бюрократия ради бюрократии — он гарантирует, что при добавлении нового
провайдера вы не забудете реализовать какой-то из методов (Python сразу
откажется создавать экземпляр с недостающим методом), и документирует
контракт в одном месте, а не по памяти "как было у предыдущего класса".
"""
from abc import ABC, abstractmethod
from typing import Iterator


class BaseLLM(ABC):
    @abstractmethod
    def answer(self, question: str, context_chunks: list[dict]) -> str:
        """Полный (нестриминговый) ответ на вопрос по уже найденному контексту."""
        raise NotImplementedError

    @abstractmethod
    def stream_answer(self, question: str, context_chunks: list[dict]) -> Iterator[str]:
        """Тот же ответ, но токенами по мере генерации — для live-вывода в чате."""
        raise NotImplementedError

    @abstractmethod
    def raw_generate(self, prompt: str, max_tokens: int = 150) -> str:
        """Произвольный запрос без RAG-обвязки — используется, например, в expand_query."""
        raise NotImplementedError


SYSTEM_PROMPT = (
    "Ты — технический помощник по документации приборов. "
    "Отвечай ТОЛЬКО на основе переданного контекста. Если в контексте "
    "нет ответа — прямо скажи, что информации нет, не выдумывай."
    "Отвечай на вопрос на том же языке на котором был задан вопрос."
)


def build_prompt(question: str, context: str) -> str:
    """Собирает промпт в "плоском" виде — нужен провайдерам без отдельного
    параметра под системный промпт (например, Ollama)."""
    return f"""{SYSTEM_PROMPT}

Контекст:
{context}

Вопрос: {question}
Ответ:"""
