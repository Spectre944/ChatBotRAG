"""Нарезка обычного текста на чанки для индексации."""
import re

from config import CHUNK_OVERLAP_SENTENCES, CHUNK_TARGET_SIZE


def chunk_text(
    text: str,
    target_size: int = CHUNK_TARGET_SIZE,
    overlap_sentences: int = CHUNK_OVERLAP_SENTENCES,
) -> list[str]:
    """
    Режем по предложениям, а не по "сырым" символам — иначе в один чанк
    попадает хвост одной мысли и начало другой (это размывает эмбеддинг
    и портит поиск). Собираем предложения в чанк, пока не наберём примерно
    target_size символов, затем начинаем новый с небольшим перехлёстом
    (overlap_sentences), чтобы не терять связку на границе.
    """
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
