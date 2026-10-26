"""
CLI для RAG-чат-бота з документації приладів.

Запуск:
    python main.py
Команди всередині чату:
    exit                           — вихід
    add <папка> <прилад> [тип]     — додати документацію (тип: manual|tu, за замовчуванням manual)
"""
from config import LLM_MODEL, LLM_PROVIDER
from debug_logger import DebugLogger
from llm import BaseLLM, build_prompt, create_llm
from pipeline import build_context
from retrieval import DocumentIndex, load_glossary


def main() -> None:
    index = DocumentIndex()
    if index.exists_on_disk():
        index.load()
    else:
        print("Індекс не знайдено, створюю новий і додаю документи...")
        index.add_documents("./docs/MKS_UM", device_name="МКС-УМ")
        index.add_documents("./docs/Spectra", device_name="МКС-11 Spectra")
        index.add_documents("./docs/Guarder", device_name="ДКГ-24 Guarder")
        index.add_documents("./docs/StoraABG", device_name="Стора-АБГ")
        # index.add_documents("./docs/-", device_name="ДРГ-Т")
        # index.add_documents("./docs/-", device_name="МКС-05 Терра")
        # index.add_documents("./docs/-", device_name="МКС-05 Терра П")
        # index.add_documents("./docs/-", device_name="МКС-05 Терра П+")
        # index.add_documents("./docs/-", device_name="ДКГ-21M")
        # index.add_documents("./docs/-", device_name="ДКГ-21")
        # index.add_documents("./docs/-", device_name="ДКГ-23")

    glossary = load_glossary()
    llm: BaseLLM = create_llm(LLM_PROVIDER, LLM_MODEL)
    debug_logger = DebugLogger()

    print("\nГотово! Команди: 'exit' — вихід, 'add <папка> <прилад> [manual|tu]' — додати документацію")
    while True:
        question = input("\nПитання: ").strip()
        if not question:
            continue

        ctx = build_context(question, index, glossary, llm)

        if ctx.detected_device:
            print(f"[Прилад]: {ctx.detected_device}")
        print(f"[Формулювання пошуку]: {ctx.expanded_queries}")
        print("[Знайдені фрагменти]")
        for r in ctx.chunks:
            print(f"  ({r['device']} / {r['source_file']} / {r.get('doc_type', '?')}): {r['text'][:100]}...")

        prompt = build_prompt(question, ctx.context_text)
        answer = llm.answer(question, ctx.chunks)
        print(f"\nВідповідь: {answer}")

        debug_logger.log({
            "type": "answer",
            "question": question,
            "device": ctx.detected_device,
            "expanded_queries": ctx.expanded_queries,
            "prompt": prompt,
            "answer": answer,
        })


if __name__ == "__main__":
    main()
