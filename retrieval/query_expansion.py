"""
Сглаживание расхождения между тем, как спрашивает пользователь ("сбросить
дозу"), и тем, как написано в документации ("обнулити АЕД"):

1. expand_query_with_glossary — быстрый, предсказуемый словарь терминов.
2. expand_query — LLM-переформулировка, используется только если словарь
   ничего не подсказал (дороже и менее предсказуемо, поэтому запасной вариант).
3. detect_device — определяет, о каком приборе идёт речь (точное совпадение
   + нечёткое сравнение для опечаток), чтобы искать только в его документации.
"""
import difflib
import json
import re
from typing import TYPE_CHECKING

from config import GLOSSARY_PATH

DEVICE_ALIASES = {
    "МКС-11 Spectra": ["spectra", "спектра", "мкс-11", "мкс11"],
    "ДКГ-24 Guarder": ["guarder", "гардер", "дкг-24", "дкг24"],
    "МКС-УМ":         ["мкс-ум", "мксум", "mksum", "mks-um"],
    "Стора-АБГ":      ["стора", "stora"],
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


def detect_device(question: str, known_devices: list[str]) -> str | None:
    q = question.lower()
    for device, aliases in DEVICE_ALIASES.items():
        if device in known_devices and any(a in q for a in aliases):
            return device
    return None