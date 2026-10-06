from typing import Iterator

from api_keys import get_api_key

from .base import SYSTEM_PROMPT, BaseLLM


class ClaudeLLM(BaseLLM):
    def __init__(self, model: str, api_key: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key or get_api_key("ANTHROPIC_API_KEY"))
        self.model = model

    @staticmethod
    def _user_message(question: str, context: str) -> str:
        return f"Контекст:\n{context}\n\nВопрос: {question}"

    def answer(self, question: str, context_chunks: list[dict]) -> str:
        context = "\n---\n".join(c["text"] for c in context_chunks)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self._user_message(question, context)}],
        )
        return resp.content[0].text

    def stream_answer(self, question: str, context_chunks: list[dict]) -> Iterator[str]:
        context = "\n---\n".join(c["text"] for c in context_chunks)
        with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self._user_message(question, context)}],
        ) as stream:
            yield from stream.text_stream

    def raw_generate(self, prompt: str, max_tokens: int = 150) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
