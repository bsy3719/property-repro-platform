from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"
VECTORSTORE_ROOT = PROJECT_ROOT / "vectorstore"


class LocalVectorDB:
    def __init__(self, embedding_model: str | None = None) -> None:
        load_dotenv(DOTENV_PATH)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        VECTORSTORE_ROOT.mkdir(parents=True, exist_ok=True)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.embedding_model, input=texts)
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)

    def _embed_query(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(model=self.embedding_model, input=text)
        return np.array(response.data[0].embedding, dtype=np.float32)

    @staticmethod
    def chunk_markdown(markdown: str, chunk_size: int = 1200, overlap: int = 200) -> list[dict[str, Any]]:
        text = markdown.strip()
        if not text:
            return []

        chunks: list[dict[str, Any]] = []
        start = 0
        chunk_id = 0
        step = max(1, chunk_size - overlap)

        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "start": start,
                    "end": end,
                    "text": chunk_text,
                }
            )
            chunk_id += 1
            if end == len(text):
                break
            start += step

        return chunks

    def _metadata_path(self, store_dir: Path) -> Path:
        return store_dir / "metadata.json"

    def build_from_markdown(self, markdown: str, source_id: str | None = None) -> dict[str, Any]:
        chunks = self.chunk_markdown(markdown)
        if not chunks:
            raise ValueError("벡터DB를 만들 markdown 텍스트가 비어 있습니다.")

        markdown_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        source_key = source_id or markdown_hash[:16]
        store_dir = VECTORSTORE_ROOT / source_key
        store_dir.mkdir(parents=True, exist_ok=True)

        chunks_path = store_dir / "chunks.json"
        vectors_path = store_dir / "embeddings.npy"
        meta_path = self._metadata_path(store_dir)

        if chunks_path.exists() and vectors_path.exists() and meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if (
                meta.get("markdown_hash") == markdown_hash
                and meta.get("embedding_model") == self.embedding_model
                and int(meta.get("num_chunks", -1)) == len(chunks)
            ):
                vectors = np.load(vectors_path)
                return {
                    "source_id": source_key,
                    "store_dir": str(store_dir),
                    "num_chunks": len(chunks),
                    "embedding_dim": int(vectors.shape[1]),
                    "cached": True,
                }

        texts = [c["text"] for c in chunks]
        vectors = self._embed_texts(texts)

        np.save(vectors_path, vectors)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_id": source_key,
                    "markdown_hash": markdown_hash,
                    "embedding_model": self.embedding_model,
                    "num_chunks": len(chunks),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return {
            "source_id": source_key,
            "store_dir": str(store_dir),
            "num_chunks": len(chunks),
            "embedding_dim": int(vectors.shape[1]),
            "cached": False,
        }

    def retrieve(self, source_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        store_dir = VECTORSTORE_ROOT / source_id
        chunks_path = store_dir / "chunks.json"
        vectors_path = store_dir / "embeddings.npy"

        if not chunks_path.exists() or not vectors_path.exists():
            raise FileNotFoundError(f"벡터DB를 찾을 수 없습니다: {store_dir}")

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        vectors = np.load(vectors_path)
        query_vec = self._embed_query(query)

        vec_norm = np.linalg.norm(vectors, axis=1)
        q_norm = np.linalg.norm(query_vec)
        denom = (vec_norm * q_norm) + 1e-12
        scores = np.dot(vectors, query_vec) / denom

        top_idx = np.argsort(scores)[::-1][:top_k]
        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(top_idx, start=1):
            c = chunks[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "score": float(scores[int(idx)]),
                    "chunk_id": c["chunk_id"],
                    "start": c["start"],
                    "end": c["end"],
                    "text": c["text"],
                }
            )

        return results
