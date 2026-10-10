"""
Дебаг-режим: на каждый вопрос в файл logs/debug.jsonl дописывается одна
JSON-строка со всей цепочкой обработки — сам вопрос, определённый прибор,
сгенерированные варианты запроса, найденные чанки, итоговый промпт и ответ
модели. Формат JSONL (по объекту на строку) выбран специально — его легко
дочитывать построчно, не загружая весь файл в память, и легко парсить
скриптом для анализа истории вопросов.
"""
import json, time
from datetime import datetime, timezone
from pathlib import Path

from config import DEBUG_LOG_PATH, DEBUG_MODE


class DebugLogger:
    def __init__(self, enabled: bool = DEBUG_MODE, path: Path = DEBUG_LOG_PATH):
        self.enabled = enabled
        self.path = path
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        question: str,
        detected_device: str | None,
        expanded_queries: list[str],
        retrieved_chunks: list[dict],
        prompt: str,
        answer: str,
        message_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        entry = {
            "type": "answer",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_id": message_id,
            "session_id": session_id,
            "question": question,
            "detected_device": detected_device,
            "expanded_queries": expanded_queries,
            "retrieved_chunks": [
                {
                    "source_file": c.get("source_file"),
                    "device": c.get("device"),
                    "type": c.get("type"),
                    "text_preview": c.get("text", "")[:200],
                }
                for c in retrieved_chunks
            ],
            "prompt": prompt,
            "answer": answer,
        }
        self._write(entry)

    def log_rating(
        self,
        message_id: str,
        rating: int,
        session_id: str | None = None,
        comment: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        self._write({
            "type": "rating",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_id": message_id,
            "session_id": session_id,
            "rating": rating,
            "comment": comment,
        })

    def _write(self, entry: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")