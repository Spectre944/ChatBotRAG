"""
Пакетний прогін питань через RAG-пайплайн: питання з JSON, результат у JSON.

Запуск:
    python run_eval.py                          # питання з questions.json
    python run_eval.py -q my_questions.json     # свій файл питань
    python run_eval.py -o results/run1.json     # свій вихідний файл

Формати questions.json (допустимі обидва, можна змішувати):
    ["Питання 1", "Питання 2"]
    [{"question": "Питання 1", "expected_device": "МКС-УМ"}, ...]
    {"questions": [ ...один зі списків вище... ]}
Додаткові поля об'єкта (expected_device, tags тощо) копіюються в результат.

Результат: один JSON-файл {"meta": {...}, "results": [...]}.
Файл перезаписується атомарно після кожного питання, тож при збої дані не губляться.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent   # .../проект/test
ROOT = HERE.parent                       # .../проект

sys.path.insert(0, str(ROOT))            # чтобы работал `from config import ...`
os.chdir(ROOT)                           # чтобы работали относительные пути (индекс, ./docs, глоссарий)

import argparse
import json
import time
from datetime import datetime

from config import LLM_MODEL, LLM_PROVIDER
from llm import BaseLLM, build_prompt, create_llm
from pipeline import build_context
from retrieval import DocumentIndex, load_glossary


def read_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("questions", [])
    items = []
    for item in data:
        if isinstance(item, str):
            item = {"question": item}
        if isinstance(item, dict) and str(item.get("question", "")).strip():
            items.append(item)
    return items


def save_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run_one(item: dict, index, glossary, llm: BaseLLM) -> dict:
    question = item["question"].strip()
    record = dict(item)
    record["question"] = question
    t0 = time.perf_counter()
    try:
        ctx = build_context(question, index, glossary, llm)
        t_retrieval = time.perf_counter()
        answer = llm.answer(question, ctx.chunks)
        t_end = time.perf_counter()

        record.update({
            "detected_device": ctx.detected_device,
            "expanded_queries": ctx.expanded_queries,
            "chunks": [
                {
                    "device": c.get("device"),
                    "source_file": c.get("source_file"),
                    "doc_type": c.get("doc_type"),
                    "text": c.get("text"),
                }
                for c in ctx.chunks
            ],
            "prompt": build_prompt(question, ctx.context_text),
            "answer": answer,
            "retrieval_sec": round(t_retrieval - t0, 2),
            "llm_sec": round(t_end - t_retrieval, 2),
            "error": None,
        })
    except Exception as exc:  # один збійний запит не має валити весь прогін
        record.update({"answer": None, "error": f"{type(exc).__name__}: {exc}"})
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Пакетний прогін питань через RAG")
    parser.add_argument("-q", "--questions", default=HERE / "questions.json", type=Path)
    parser.add_argument("-o", "--out", default=None, type=Path, help="вихідний JSON (за замовчуванням results/eval_<дата-час>.json)")
    args = parser.parse_args()

    if not args.questions.exists():
        sys.exit(f"Файл з питаннями не знайдено: {args.questions}")
    try:
        questions = read_questions(args.questions)
    except json.JSONDecodeError as exc:
        sys.exit(f"Некоректний JSON у {args.questions}: {exc}")
    if not questions:
        sys.exit("У файлі немає питань.")

    index = DocumentIndex()
    if not index.exists_on_disk():
        sys.exit("Індекс не знайдено. Спочатку запустіть main.py, щоб його створити.")
    index.load()

    glossary = load_glossary()
    llm: BaseLLM = create_llm(LLM_PROVIDER, LLM_MODEL)

    out_path = args.out or HERE / "results" / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "provider": LLM_PROVIDER,
            "model": LLM_MODEL,
            "questions_file": str(args.questions),
            "total": len(questions),
        },
        "results": [],
    }

    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {item['question']}")
        rec = run_one(item, index, glossary, llm)
        rec["id"] = item.get("id", i)
        rec["timestamp"] = datetime.now().isoformat(timespec="seconds")
        payload["results"].append(rec)
        save_json_atomic(out_path, payload)

    payload["meta"]["finished_at"] = datetime.now().isoformat(timespec="seconds")
    payload["meta"]["errors"] = sum(1 for r in payload["results"] if r["error"])
    save_json_atomic(out_path, payload)
    print(f"\nГотово. Результат: {out_path}")


if __name__ == "__main__":
    main()
