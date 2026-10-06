from typing import Iterator

from api_keys import get_api_key

from .base import SYSTEM_PROMPT, BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model: str, api_key: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or get_api_key("OPENAI_API_KEY"))
        self.model = model

    @staticmethod
    def _messages(question: str, context: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
        ]

    def answer(self, question: str, context_chunks: list[dict]) -> str:
        context = "\n---\n".join(c["text"] for c in context_chunks)
        resp = self.client.chat.completions.create(
            model=self.model, messages=self._messages(question, context)
        )
        return resp.choices[0].message.content

    def stream_answer(self, question: str, context_chunks: list[dict]) -> Iterator[str]:
        context = "\n---\n".join(c["text"] for c in context_chunks)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(question, context),
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def raw_generate(self, prompt: str, max_tokens: int = 150) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content
