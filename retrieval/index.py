"""
Векторный индекс документации: построение, сохранение/загрузка с диска,
поиск с MMR-переранжированием для разнообразия результатов.
"""
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL_NAME, FAISS_PATH, INDEX_DIR, META_PATH, MMR_LAMBDA
from .chunking import chunk_text
from .documents import load_documents


class DocumentIndex:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.embedder = SentenceTransformer(model_name)
        self.chunks: list[str] = []
        self.metadata: list[dict] = []  # {"device", "source_file", "type", "doc_type"}
        self.index: faiss.Index | None = None
        self._devices = None

    # --- Персистентность ---
    def exists_on_disk(self) -> bool:
        return FAISS_PATH.exists() and META_PATH.exists()

    def load(self) -> None:
        self.index = faiss.read_index(str(FAISS_PATH))
        with open(META_PATH, "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]
        print(f"Загружен индекс: {len(self.chunks)} чанков")
        self._build_cache()

    def save(self) -> None:
        INDEX_DIR.mkdir(exist_ok=True)
        faiss.write_index(self.index, str(FAISS_PATH))
        with open(META_PATH, "wb") as f:
            pickle.dump({"chunks": self.chunks, "metadata": self.metadata}, f)

    # --- Добавление документов ---
    def add_documents(self, folder: str, device_name: str, doc_type: str = "manual") -> None:
        """
        doc_type — произвольная метка источника, например "manual" (руководство
        по эксплуатации) или "tu" (технические условия). Полезна для отладки и,
        при необходимости, для приоритизации нужного типа документа под
        конкретные вопросы.
        """
        docs = load_documents(folder)
        new_chunks: list[str] = []
        new_meta: list[dict] = []

        for doc in docs:
            for chunk in chunk_text(doc["text"]):
                new_chunks.append(chunk)
                new_meta.append({
                    "device": device_name, "source_file": doc["filename"],
                    "type": "text", "doc_type": doc_type,
                })

            # Таблицы — отдельными атомарными чанками, см. documents.py
            for table_chunk in doc["tables"]:
                new_chunks.append(table_chunk)
                new_meta.append({
                    "device": device_name, "source_file": doc["filename"],
                    "type": "table", "doc_type": doc_type,
                })

        if not new_chunks:
            print(f"В папке {folder} не нашлось документов")
            return

        # Модель e5 обучена с префиксом "passage: " для индексируемых текстов
        # и "query: " для вопросов — без этого эмбеддинги существенно хуже
        prefixed = [f"passage: {c}" for c in new_chunks]
        embeddings = self.embedder.encode(prefixed, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype("float32")

        if self.index is None:
            dim = embeddings.shape[1]
            # IndexFlatIP на нормализованных векторах = косинусное сходство —
            # метрика, под которую обучена e5 (не L2!)
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        self.metadata.extend(new_meta)
        self._build_cache()
        self.save()
        print(f"Добавлено {len(new_chunks)} чанков по прибору «{device_name}» ({doc_type})")

    # --- Вспомогательное ---
    def get_known_devices(self) -> list[str]:
        """Уникальный список приборов, которые уже есть в индексе."""
        return sorted({m["device"] for m in self.metadata})

    # --- Поиск ---
    def _build_cache(self):
        self._embs = self.index.reconstruct_n(0, self.index.ntotal)   # вызвать в load() и после add_documents()
        self._devices = np.array([m["device"] for m in self.metadata])

    def search(self, query, k=5, device_filter=None, use_mmr=True):
        q = self.embedder.encode([f"query: {query}"], normalize_embeddings=True).astype("float32")[0]
        if device_filter:
            ids = np.where(self._devices == device_filter)[0]
            if len(ids) == 0:
                return []
            sims = self._embs[ids] @ q
            top = np.argsort(-sims)[:max(k * 6, 20)]
            candidates = [{"i": int(ids[j]), "sim_to_query": float(sims[j])} for j in top]
        else:
            distances, idx = self.index.search(q[None, :], min(max(k * 6, 20), len(self.chunks)))
            candidates = [{"i": int(i), "sim_to_query": float(d)} for i, d in zip(idx[0], distances[0]) if i >= 0]
        if not candidates:
            return []
        chosen = self._mmr_select(candidates, k) if use_mmr else candidates[:k]
        return [{"text": self.chunks[c["i"]], **self.metadata[c["i"]]} for c in chosen]

    def _mmr_select(self, candidates: list[dict], k: int, lambda_param: float = MMR_LAMBDA) -> list[dict]:
        """
        Maximal Marginal Relevance: на каждом шаге выбираем кандидата,
        который одновременно (а) релевантен запросу и (б) не похож на уже
        выбранные — не даёт топ-k схлопнуться в несколько почти одинаковых
        по смыслу чанков.
        """
        if len(candidates) <= k:
            return candidates

        cand_embeddings = np.array([self.index.reconstruct(int(c["i"])) for c in candidates])

        selected = [candidates[0]]
        selected_embs = [cand_embeddings[0]]
        remaining = list(zip(candidates[1:], cand_embeddings[1:]))

        while len(selected) < k and remaining:
            best_score, best_idx = -1e9, 0
            for idx_r, (cand, emb) in enumerate(remaining):
                max_sim_to_selected = max(float(np.dot(emb, sel_emb)) for sel_emb in selected_embs)
                score = lambda_param * cand["sim_to_query"] - (1 - lambda_param) * max_sim_to_selected
                if score > best_score:
                    best_score, best_idx = score, idx_r
            chosen_cand, chosen_emb = remaining.pop(best_idx)
            selected.append(chosen_cand)
            selected_embs.append(chosen_emb)

        return selected
