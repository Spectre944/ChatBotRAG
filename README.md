# RAG-чат-бот по документации приборов

## Структура проекта

```
config.py              # все настройки (пути, модель, параметры поиска)
api_keys.py             # загрузка API-ключей (env -> api_key.txt)
debug_logger.py         # дебаг-режим: лог вопрос/промпт/ответ в logs/debug.jsonl
llm/                     # LLM-провайдеры (Claude / OpenAI / Ollama) за единым интерфейсом
retrieval/               # загрузка документов, чанкинг, векторный индекс, поиск
pipeline.py              # общая логика "вопрос -> контекст" для CLI и веб-API
main.py                  # CLI
app.py                   # веб-API (FastAPI) со стримингом
glossary.json            # словарь бытовых терминов -> официальные из документации
```

## Установка

```bash
pip install -r requirements.txt
```

## Настройка API-ключа

```bash
cp api_key.txt.example api_key.txt
# впишите в api_key.txt реальный ключ, например:
# ANTHROPIC_API_KEY=sk-ant-...
```

`api_key.txt` уже в `.gitignore` — в коммиты не попадёт.

## Выбор модели и провайдера

Настраивается в `config.py`:

```python
LLM_PROVIDER = "claude"                       # "claude" | "openai" | "ollama"
LLM_MODEL = "claude-haiku-4-5-20251001"
```

## Запуск

Первый запуск сам соберёт индекс из `./docs/device_a` и `./docs/device_b` —
замените на реальные папки с документацией, либо добавляйте документы
командой `add <папка> <прибор> [manual|tu]` прямо в чате.

```bash
python main.py
```

Веб-версия:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Дебаг-режим

Включён по умолчанию (`config.DEBUG_MODE = True`). Каждый вопрос со всей
цепочкой обработки (найденный прибор, варианты запроса, найденные чанки,
итоговый промпт, ответ) дописывается в `logs/debug.jsonl` — по одной
JSON-строке на вопрос, удобно разбирать построчно или скриптом.
