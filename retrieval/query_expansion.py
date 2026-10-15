"""
Сглаживание расхождения между тем, как спрашивает пользователь ("сбросить
дозу"), и тем, как написано в документации ("обнулити АЕД"):

1. expand_query_with_glossary — быстрый, предсказуемый словарь терминов.
2. expand_query — LLM-переформулировка, используется только если словарь
   ничего не подсказал (дороже и менее предсказуемо, поэтому запасной вариант).
3. detect_device — определяет, о каком приборе идёт речь (точное совпадение
   + нечёткое сравнение для опечаток), чтобы искать только в его документации.
"""
from __future__ import annotations  # иначе BaseLLM под TYPE_CHECKING даст NameError

import difflib
import json
import re
from typing import TYPE_CHECKING

from config import GLOSSARY_PATH

# Кириллица, визуально похожая на латиницу (частая "опечатка" при вводе)
_HOMOGLYPHS = str.maketrans("aceopxyk", "асеорхук")  # lat -> cyr; применяем к обеим сторонам

FUZZY_THRESHOLD = 0.82
MIN_FUZZY_LEN = 5          # короткие алиасы ("мкс") сравниваем только точно
AMBIGUITY_MARGIN = 0.05    # если второй прибор почти так же хорош — не гадаем

DEVICE_ALIASES = {
    "МКС-11 Spectra": ["spectra", "спектра", "мкс-11", "мкс11"],
    "ДКГ-24 Guarder": ["guarder", "гардер", "дкг-24", "дкг24"],
    "МКС-УМ":         ["мкс-ум", "мксум", "mksum", "mks-um"],
    "Стора-АБГ":      ["стора", "stora", "ркс-01"],
}

if TYPE_CHECKING:
    from llm.base import BaseLLM

def load_glossary() -> dict[str, list[str]]:
    if not GLOSSARY_PATH.exists():
        return {}
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.pop("_readme", None)
    return data


def expand_query_with_glossary(question: str, glossary: dict[str, list[str]]) -> list[str]:
    """Подставляет официальные термины из словаря вместо бытовых — без LLM."""
    variants = []
    q_lower = question.lower()
    for term, synonyms in glossary.items():
        if term.lower() in q_lower:
            for syn in synonyms:
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                variants.append(pattern.sub(syn, question))
    return variants


def expand_query(question: str, llm: "BaseLLM") -> list[str]:
    """
    Запасной вариант, когда словарь не сработал: просим LLM предложить
    альтернативные формулировки с вероятной официальной терминологией.
    Генерация жёстко ограничена по длине и штрафу за повтор на стороне
    LLM-класса — небольшие модели склонны зацикливаться на этой задаче.
    """
    prompt = f"""Переформулируй вопрос ниже РОВНО 2 альтернативными способами,
используя вероятные официальные технические термины, которые могли бы
использоваться в инструкции к прибору (например, вместо "сбросить дозу"
может быть "обнулити АЕД" или "обнуление накопленной дозы").

Правила:
- Выведи ТОЛЬКО 2 строки, без нумерации и пояснений
- Формулировки должны быть на том же языке, что и вопрос ниже
- Не добавляй ничего, кроме самих 2 формулировок

Вопрос: {question}"""
    raw = llm.raw_generate(prompt, max_tokens=150)
    lines = [line.strip(" -.") for line in raw.split("\n") if line.strip()]
    lines = [re.sub(r"^\d+\.\s*", "", line) for line in lines][:3]
    return [question] + lines


def _tokens(text: str) -> list[str]:
    text = text.lower().replace("ё", "е")
    return re.findall(r"[a-zа-я0-9]+", text)  # дефисы и пробелы -> границы токенов


def _canon(s: str) -> str:
    """Для сравнения: без дефисов/пробелов, гомоглифы приведены к одному алфавиту."""
    return "".join(_tokens(s)).translate(_HOMOGLYPHS)


def _windows(tokens: list[str], max_n: int = 2):
    """Слова и склейки соседних слов: ['мкс','11'] -> 'мкс', '11', 'мкс11'."""
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            yield "".join(tokens[i:i + n]).translate(_HOMOGLYPHS)


def _score(window: str, alias: str) -> float:
    if window == alias:
        return 1.0
    if len(alias) >= MIN_FUZZY_LEN and len(window) >= MIN_FUZZY_LEN:
        ratio = difflib.SequenceMatcher(None, window, alias).ratio()
        if ratio >= FUZZY_THRESHOLD:
            return ratio
    return 0.0


def detect_device(question: str, known_devices: list[str]) -> str | None:
    windows = set(_windows(_tokens(question)))
    if not windows:
        return None

    scores: dict[str, float] = {}
    for device, aliases in DEVICE_ALIASES.items():
        if device not in known_devices:
            continue
        best = max(
            (_score(w, _canon(a)) for a in aliases for w in windows),
            default=0.0,
        )
        if best > 0:
            scores[device] = best

    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < AMBIGUITY_MARGIN:
        return None  # вопрос про два прибора сразу — ищем по всей базе
    return ranked[0][0]