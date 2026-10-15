"""
CLI для RAG-чат-бота по документации приборов.

Запуск:
    python main.py
Команды внутри чата:
    exit                           — выход
    add <папка> <прибор> [тип]     — добавить документацию (тип: manual|tu, по умолчанию manual)
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
        print("Индекс не найден, создаю новый и добавляю документы...")
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

    print("\nГотово! Команды: 'exit' — выход, 'add <папка> <прибор> [manual|tu]' — добавить документацию")
    while True:
        question = input("\nВопрос: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            break
        if question.lower().startswith("add "):
            parts = question.split(maxsplit=3)
            folder, device = parts[1], parts[2]
            doc_type = parts[3] if len(parts) > 3 else "manual"
            index.add_documents(folder, device, doc_type)
            continue

        ctx = build_context(question, index, glossary, llm)

        if ctx.detected_device:
            print(f"[Определён прибор]: {ctx.detected_device}")
        print(f"[Формулировки поиска]: {ctx.expanded_queries}")
        print("[Найденные фрагменты]")
        for r in ctx.chunks:
            print(f"  ({r['device']} / {r['source_file']} / {r.get('doc_type', '?')}): {r['text'][:100]}...")

        prompt = build_prompt(question, ctx.context_text)
        answer = llm.answer(question, ctx.chunks)
        print(f"\nОтвет: {answer}")

        debug_logger.log(
            question=question,
            detected_device=ctx.detected_device,
            expanded_queries=ctx.expanded_queries,
            retrieved_chunks=ctx.chunks,
            prompt=prompt,
            answer=answer,
        )


if __name__ == "__main__":
    main()
