import json
from typing import Iterator

import requests

from .base import BaseLLM, build_prompt


class OllamaLLM(BaseLLM):
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = model

    def _generate(self, prompt: str, max_tokens: int = 512, repeat_penalty: float = 1.1) -> str:
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": max_tokens,  # жёсткий потолок длины — защита от зацикливания
                    "repeat_penalty": repeat_penalty,
                },
            },
        )
        return resp.json()["response"].strip()

    def answer(self, question: str, context_chunks: list[dict]) -> str:
        context = "\n---\n".join(c["text"] for c in context_chunks)
        return self._generate(build_prompt(question, context))

    def stream_answer(self, question: str, context_chunks: list[dict]) -> Iterator[str]:
        context = "\n---\n".join(c["text"] for c in context_chunks)
        prompt = build_prompt(question, context)
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": 0.2, "repeat_penalty": 1.1},
            },
            stream=True,
        )
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("response", "")
            if token:
                yield token
            if chunk.get("done"):
                break

    def raw_generate(self, prompt: str, max_tokens: int = 150) -> str:
        # Более высокий repeat_penalty — специально для вспомогательных
        # генераций вроде expand_query, где слабые модели склонны зацикливаться
        return self._generate(prompt, max_tokens=max_tokens, repeat_penalty=1.3)
