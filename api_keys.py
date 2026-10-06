"""
Загрузка API-ключей.

Порядок приоритета:
1) переменная окружения (ANTHROPIC_API_KEY / OPENAI_API_KEY)
2) файл api_key.txt в корне проекта (формат "ИМЯ=значение", по одному ключу
   на строку — см. api_key.txt.example)

api_key.txt добавлен в .gitignore и никогда не должен попадать в git —
используйте api_key.txt.example как образец для новых разработчиков.
"""
import os

from config import API_KEY_FILE


def _load_api_key_file() -> dict[str, str]:
    if not API_KEY_FILE.exists():
        return {}
    keys = {}
    for line in API_KEY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        keys[key.strip()] = value.strip()
    return keys


_FILE_KEYS = _load_api_key_file()


def get_api_key(env_var: str) -> str:
    """Возвращает ключ из переменной окружения или из api_key.txt."""
    value = os.environ.get(env_var) or _FILE_KEYS.get(env_var)
    if not value:
        raise RuntimeError(
            f"Не найден {env_var}. Задайте переменную окружения или добавьте "
            f"строку '{env_var}=...' в api_key.txt в корне проекта "
            f"(см. api_key.txt.example)."
        )
    return value
