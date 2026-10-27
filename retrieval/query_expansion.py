"""
Сглаживание расхождения между тем, как спрашивает пользователь ("сбросить
дозу"), и тем, как написано в документации ("обнулити АЕД").

Порядок работы (см. build_context):
1. detect_device                  — о каком приборе вопрос (искать только в его документации).
2. expand_query_with_glossary     — быстрый, предсказуемый словарь понятий (glossary.json).
3. expand_query                   — LLM-переформулировка; запасной вариант, если словарь
                                    ничего не подсказал (дороже и менее предсказуемо).

Публичный API: load_glossary, expand_query_with_glossary, expand_query, detect_device.
"""
from __future__ import annotations  # иначе BaseLLM под TYPE_CHECKING даст NameError

import difflib
import json
import logging
import re
from typing import TYPE_CHECKING

from config import GLOSSARY_PATH

if TYPE_CHECKING:
    from llm.base import BaseLLM

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# 1. Настройки
# ════════════════════════════════════════════════════════════════════

# --- словарь понятий ---
MAX_CONCEPTS = 3        # сколько понятий максимум берём из одного вопроса
MAX_TERMS = 3           # сколько вариантов (doc_terms) подставляем для одного понятия

# Служебные слова не участвуют в сопоставлении со словарём ("как", "что", "чи", ...).
# Отрицания ("не", "нет") сюда НЕ входят: они меняют смысл.
STOPWORDS = {
    "как", "что", "где", "когда", "почему", "можно", "ли", "в", "на", "по", "из", "для", "с", "к", "и", "или",
    "як", "що", "де", "коли", "чому", "можна", "чи", "у", "до", "із", "та", "й", "це", "это", "мне", "мені", "мой", "мій",
    "how", "what", "the", "a", "to", "do", "i", "my", "is", "of",
}

NEGATIONS = {"не", "нет", "ні", "немає", "без"}

# --- определение прибора ---
FUZZY_THRESHOLD = 0.82
MIN_FUZZY_LEN = 5          # короткие алиасы ("мкс") сравниваем только точно
AMBIGUITY_MARGIN = 0.05    # если второй прибор почти так же хорош — не гадаем

DEVICE_ALIASES = {
    "МКС-11 Spectra": ["spectra", "спектра", "мкс-11", "мкс11"],
    "ДКГ-24 Guarder": ["guarder", "гардер", "дкг-24", "дкг24"],
    "МКС-УМ":         ["мкс-ум", "мксум", "mksum", "mks-um"],
    "Стора-АБГ":      ["стора", "stora", "ркс-01"],
}

# Кириллица, визуально похожая на латиницу (частая "опечатка" при вводе)
_HOMOGLYPHS = str.maketrans("aceopxyk", "асеорхук")  # lat -> cyr; применяем к обеим сторонам


# ════════════════════════════════════════════════════════════════════
# 2. Словарь понятий: загрузка
# ════════════════════════════════════════════════════════════════════

def load_glossary() -> dict[str, dict]:
    """Формат: {id_понятия: {"doc_terms": [...], "synonyms": {"ru": [...], ...}, ...}}.
    Служебные ключи (начинаются с "_", например "_readme") и некорректные записи пропускаются."""
    if not GLOSSARY_PATH.exists():
        return {}
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    glossary = {}
    for cid, concept in data.items():
        if cid.startswith("_"):
            continue
        if not isinstance(concept, dict) or not concept.get("doc_terms") or not concept.get("synonyms"):
            log.warning("glossary: пропущено понятие %r (нет doc_terms или synonyms)", cid)
            continue
        glossary[cid] = concept
    return glossary


# ════════════════════════════════════════════════════════════════════
# 3. Словарь понятий: токенизация и индекс
# ════════════════════════════════════════════════════════════════════

try:  # необязательно: лучшая нормализация для ru/uk, если установлен pymorphy3
    import pymorphy3
    _MORPH = [pymorphy3.MorphAnalyzer(lang="ru"), pymorphy3.MorphAnalyzer(lang="uk")]
except Exception:
    _MORPH = []

_GLOSSARY_WORD = re.compile(r"[\w']+", re.UNICODE)   # буквы любого алфавита (в т.ч. і, ї, є, ґ) и апостроф


def _glossary_tokens(text: str) -> list[str]:
    """Слова вопроса/фразы без служебных слов."""
    text = text.lower().replace("ё", "е").replace("’", "'")
    return [w for w in _GLOSSARY_WORD.findall(text) if w not in STOPWORDS]


def _stem(word: str) -> str:
    """Лёгкий стеммер-запасной вариант: доза/дозу/дозы -> доз, включить/включается -> включ."""
    if len(word) >= 6:
        return word[:5]
    if len(word) >= 4:
        return word[:-1]
    return word


def _forms(word: str) -> set[str]:
    """Все допустимые 'формы' слова для сопоставления."""
    if _MORPH:
        forms = {m.parse(word)[0].normal_form for m in _MORPH}
        forms.add(word)
        return forms
    return {_stem(word)}


def build_glossary_index(glossary: dict) -> list[tuple[list[set[str]], str]]:
    """Список (формы_слов_каждой_фразы, id_понятия) по всем synonyms всех языков."""
    index = []
    for cid, concept in glossary.items():
        for phrases in concept.get("synonyms", {}).values():
            for phrase in phrases:
                tokens = _glossary_tokens(phrase)
                if tokens:
                    index.append(([_forms(t) for t in tokens], cid))
    return index


_INDEX_CACHE: dict[int, tuple[dict, list]] = {}


def _get_index(glossary: dict) -> list:
    """Индекс строится один раз на словарь (кэш по идентичности объекта)."""
    cached = _INDEX_CACHE.get(id(glossary))
    if cached is None or cached[0] is not glossary:
        cached = (glossary, build_glossary_index(glossary))
        _INDEX_CACHE[id(glossary)] = cached
    return cached[1]


# ════════════════════════════════════════════════════════════════════
# 4. Расширение запроса: словарь (основной путь)
# ════════════════════════════════════════════════════════════════════

def _question_words(question: str) -> list[tuple[int, int, set[str], str]]:
    """Значимые слова вопроса: (начало, конец, формы, слово). Позиции нужны для подстановки."""
    words = []
    for m in _GLOSSARY_WORD.finditer(question):
        w = m.group().lower().replace("ё", "е").replace("’", "'")
        if w not in STOPWORDS:
            words.append((m.start(), m.end(), _forms(w), w))
    return words


def _match_positions(phrase_forms: list[set[str]], q_words: list) -> list[int] | None:
    """Индексы слов вопроса, которыми покрыта фраза (каждое слово фразы — своё слово вопроса).
    None, если какого-то слова фразы в вопросе нет. Порядок слов не важен."""
    used: set[int] = set()
    positions: list[int] = []
    for pf in phrase_forms:
        for i, (_, _, qf, _) in enumerate(q_words):
            if i not in used and pf & qf:
                used.add(i)
                positions.append(i)
                break
        else:
            return None
    return sorted(positions)


def _substitute(question: str, q_words: list, positions: list[int], term: str) -> str:
    """Заменяет найденные слова вопроса на термин документации, остальной вопрос не трогает:
    "Как сбросить дозу на ДКГ?" + "обнулити АЕД" -> "Как обнулити АЕД на ДКГ?"."""
    spans = [q_words[i][:2] for i in positions]
    parts = [question[:spans[0][0]], term]
    last = spans[0][1]
    for start, end in spans[1:]:          # слова фразы, разделённые другими словами, убираем
        parts.append(question[last:start].rstrip())
        last = end
    parts.append(question[last:])
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _match_concepts(question: str, glossary: dict) -> tuple[list, list[tuple[str, list[int]]]]:
    """Понятия из словаря, найденные в вопросе: (q_words, [(id_понятия, позиции_слов), ...])."""
    q_words = _question_words(question)
    if not q_words:
        return q_words, []

    # Для каждого понятия берём самую длинную (самую точную) найденную фразу.
    matches: dict[str, list[int]] = {}
    for phrase_forms, cid in _get_index(glossary):
        positions = _match_positions(phrase_forms, q_words)
        if positions and len(positions) > len(matches.get(cid, [])):
            matches[cid] = positions

    # Понятие нельзя подставлять, если рядом с его словами стоит "не", которое оно не покрывает:
    # "прибор не включается" -> "увімкнення приладу не" означало бы обратное.
    def _loses_negation(positions: list[int]) -> bool:
        near = {j for i in positions for j in (i - 1, i + 1) if 0 <= j < len(q_words)} - set(positions)
        return any(q_words[j][3] in NEGATIONS for j in near)

    # Понятие, чьи слова целиком покрыты более точным понятием, отбрасываем
    # ("доза" внутри "сбросить дозу" — лишний шум).
    selected: list[tuple[str, list[int]]] = []
    covered: set[int] = set()
    for cid, positions in sorted(matches.items(), key=lambda kv: -len(kv[1])):
        if set(positions) <= covered or _loses_negation(positions):
            continue
        selected.append((cid, positions))
        covered |= set(positions)
        if len(selected) == MAX_CONCEPTS:
            break
    return q_words, selected


def expand_query_with_glossary(question: str, glossary: dict) -> list[str]:
    """СТАРЫЙ путь (подстановка слов в предложение). Оставлен для совместимости;
    основной путь теперь build_search_queries. Поле devices здесь не используется."""
    q_words, selected = _match_concepts(question, glossary)
    if not selected:
        return []

    # Сначала первый термин каждого понятия, потом вторые и т.д.: так при обрезке
    # результатов поиска каждое понятие остаётся представленным.
    variants: list[str] = []
    for n in range(MAX_TERMS):
        for cid, positions in selected:
            terms = glossary[cid]["doc_terms"]
            if n < len(terms):
                variants.append(_substitute(question, q_words, positions, terms[n]))
    return list(dict.fromkeys(v for v in variants if v != question))   # без дублей


# ════════════════════════════════════════════════════════════════════
# 4b. Основной путь: словарь как подсказка + одна LLM-переформулировка
# ════════════════════════════════════════════════════════════════════

MAX_QUERIES = 4   # оригинал + 2 от LLM + терминный запрос


def _clean_llm_lines(raw: str) -> list[str]:
    lines = []
    for ln in raw.split("\n"):
        ln = re.sub(r"^\s*(\d+[.)]|[-•*])\s*", "", ln).strip(" .")
        ln = re.sub(r"^(строка|рядок|line)\s*\d*\s*[:.-]\s*", "", ln, flags=re.I)
        if ln:
            lines.append(ln)
    return lines


def build_search_queries(question: str, glossary: dict, llm: "BaseLLM") -> list[str]:
    """[оригінал, до 2 переформулювань українською, термінний запит].
    Словник знаходить поняття й дає LLM точні терміни; LLM виправляє одруки та перекладає
    мовою документації. Термінний запит зі словника додається завжди як страховка."""
    q_words, selected = _match_concepts(question, glossary)

    hints = []
    for cid, positions in selected:
        said = " ".join(q_words[i][3] for i in positions)
        terms = ", ".join(glossary[cid]["doc_terms"][:MAX_TERMS])
        hints.append(f"- «{said}» = {terms}")
    hint_block = ("Терміни з документації (використовуй їх точно):\n" + "\n".join(hints)) if hints else ""

    prompt = f"""Ти допомагаєш шукати відповідь у технічній документації приладу. Документація українською мовою.
Перетвори запитання користувача на пошукові запити українською.

Правила:
- Виправ одруки та переклади запитання українською.
- Побутові слова замінюй термінами документації зі списку нижче. Не вигадуй інших термінів.
- Назви приладів, кнопок і індикаторів (RAYS/RESTART, ON/SAVE, CHARGE, ALARM, GPS) залиш без змін.
- Рядок 1: повне запитання українською.
- Рядок 2: 3-6 ключових слів українською, без питальних слів.
- Виведи ТІЛЬКИ ці 2 рядки, без нумерації та пояснень.

{hint_block}

Запитання: {question}"""

    queries = [question]
    try:
        raw = llm.raw_generate(prompt, max_tokens=120)
        queries += _clean_llm_lines(raw)[:4]
    except Exception:
        log.exception("query rewrite failed, using glossary only")

    if selected:
        covered = {i for _, pos in selected for i in pos}
        rest = [w[3] for i, w in enumerate(q_words) if i not in covered]
        queries.append(" ".join([glossary[cid]["doc_terms"][0] for cid, _ in selected] + rest))

    return list(dict.fromkeys(q for q in queries if q.strip()))[:MAX_QUERIES]


# ════════════════════════════════════════════════════════════════════
# 5. Расширение запроса: LLM (запасной путь)
# ════════════════════════════════════════════════════════════════════

def expand_query(question: str, llm: "BaseLLM") -> list[str]:
    """
    Запасной вариант, когда словарь не сработал: просим LLM предложить
    альтернативные формулировки с вероятной официальной терминологией.
    Генерация жёстко ограничена по длине и штрафу за повтор на стороне
    LLM-класса — небольшие модели склонны зацикливаться на этой задаче.
    Возвращает [question, формулировка1, формулировка2].
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


# ════════════════════════════════════════════════════════════════════
# 6. Определение прибора
# ════════════════════════════════════════════════════════════════════

def _device_tokens(text: str) -> list[str]:
    """Слова для сравнения названий приборов: дефисы и пробелы — границы токенов."""
    text = text.lower().replace("ё", "е")
    return re.findall(r"[a-zа-яіїєґ0-9]+", text)


def _canon(s: str) -> str:
    """Для сравнения: без дефисов/пробелов, гомоглифы приведены к одному алфавиту."""
    return "".join(_device_tokens(s)).translate(_HOMOGLYPHS)


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
    """Название прибора из вопроса или None, если не упомянут / неоднозначно."""
    windows = set(_windows(_device_tokens(question)))
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