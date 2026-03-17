from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"
VECTORSTORE_ROOT = PROJECT_ROOT / "vectorstore"
CHUNKING_VERSION = "semantic_markdown_v1"


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
    def _tokenize_for_bm25(text: str) -> list[str]:
        return [token.lower() for token in re.findall(r"[A-Za-z0-9]+(?:[._/+:-][A-Za-z0-9]+)*", str(text))]

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        if np.isclose(max_score, min_score):
            return np.ones_like(scores, dtype=np.float32) if max_score > 0 else np.zeros_like(scores, dtype=np.float32)
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized.astype(np.float32)

    @staticmethod
    def _bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        tokenized_docs = [LocalVectorDB._tokenize_for_bm25(document) for document in documents]
        query_tokens = LocalVectorDB._tokenize_for_bm25(query)
        if not documents or not query_tokens:
            return np.zeros(len(documents), dtype=np.float32)

        doc_lengths = np.array([len(tokens) for tokens in tokenized_docs], dtype=np.float32)
        avgdl = float(np.mean(doc_lengths)) if len(doc_lengths) else 0.0
        if avgdl <= 0.0:
            return np.zeros(len(documents), dtype=np.float32)

        doc_freqs: dict[str, int] = {}
        term_freqs: list[dict[str, int]] = []
        for tokens in tokenized_docs:
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            term_freqs.append(tf)
            for token in set(tokens):
                doc_freqs[token] = doc_freqs.get(token, 0) + 1

        num_docs = len(documents)
        scores = np.zeros(num_docs, dtype=np.float32)
        for index, tf in enumerate(term_freqs):
            doc_len = float(doc_lengths[index])
            score = 0.0
            for token in query_tokens:
                freq = tf.get(token, 0)
                if freq <= 0:
                    continue
                df = doc_freqs.get(token, 0)
                idf = np.log(1.0 + ((num_docs - df + 0.5) / (df + 0.5)))
                denominator = freq + k1 * (1.0 - b + b * (doc_len / avgdl))
                score += idf * ((freq * (k1 + 1.0)) / max(denominator, 1e-12))
            scores[index] = float(score)
        return scores

    @staticmethod
    def estimate_tokens(text: str) -> int:
        stripped = str(text).strip()
        if not stripped:
            return 0
        words = re.findall(r"\S+", stripped)
        return max(1, int(round(max(len(words) * 1.25, len(stripped) / 4.0))))

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t]+\n", "\n", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    @staticmethod
    def parse_markdown_blocks(markdown: str) -> list[dict[str, Any]]:
        text = LocalVectorDB._normalize_text(markdown)
        if not text:
            return []

        raw_blocks = re.split(r"\n\s*\n+", text)
        blocks: list[dict[str, Any]] = []
        for block_id, raw_block in enumerate(raw_blocks):
            block_text = raw_block.strip()
            if not block_text:
                continue
            block_type = LocalVectorDB.classify_block(block_text)
            block: dict[str, Any] = {
                "block_id": len(blocks),
                "source_index": block_id,
                "text": block_text,
                "type": block_type,
                "approx_tokens": LocalVectorDB.estimate_tokens(block_text),
            }
            if block_type == "heading":
                match = re.match(r"^(#{1,6})\s+(.*)$", block_text)
                if match:
                    block["heading_level"] = len(match.group(1))
                    block["heading_text"] = match.group(2).strip()
            blocks.append(block)
        return blocks

    @staticmethod
    def _is_markdown_table(lines: list[str]) -> bool:
        if len(lines) < 2:
            return False
        first, second = lines[0], lines[1]
        if "|" not in first or "|" not in second:
            return False
        return bool(re.match(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$", second))

    @staticmethod
    def classify_block(block_text: str) -> str:
        lines = [line.rstrip() for line in block_text.splitlines() if line.strip()]
        first_line = lines[0].strip() if lines else ""
        lowered = first_line.lower()

        if re.fullmatch(r"<!--\s*\[Page\s+\d+\]\s*-->", first_line, re.IGNORECASE) or re.fullmatch(
            r"\[Page\s+\d+\]", first_line, re.IGNORECASE
        ):
            return "page_marker"
        if re.match(r"^#{1,6}\s+\S", first_line):
            return "heading"
        if LocalVectorDB._is_markdown_table(lines):
            return "table"
        if re.match(r"^(figure|fig\.|scheme|table)\s+\d+\b", lowered, re.IGNORECASE):
            return "figure_like"
        if lines and all(
            re.match(r"^(?:[-*•]\s+|\d+[.)]\s+|\(?[a-zA-Z0-9]+\)\s+)", line.strip()) for line in lines
        ):
            return "list"
        return "paragraph"

    @staticmethod
    def _normalize_section_title(title: str) -> str:
        normalized = re.sub(r"^[■•\s]+", "", title).strip()
        return re.sub(r"\s+", " ", normalized)

    @staticmethod
    def _section_label(section_path: list[str]) -> str:
        if not section_path:
            return ""
        return LocalVectorDB._normalize_section_title(section_path[-1]).lower()

    @staticmethod
    def _infer_unit_type(section_path: list[str], block_type: str) -> str:
        label = LocalVectorDB._section_label(section_path)
        if block_type == "table":
            return "table"
        if block_type == "figure_like":
            return "figure_like"
        if label == "abstract":
            return "abstract_like"
        if label in {"conclusion", "conclusions", "summary", "summary and conclusions"}:
            return "conclusion_like"
        return block_type

    @staticmethod
    def _longest_common_section_path(paths: list[list[str]]) -> list[str]:
        if not paths:
            return []
        common = list(paths[0])
        for path in paths[1:]:
            max_len = min(len(common), len(path))
            index = 0
            while index < max_len and common[index] == path[index]:
                index += 1
            common = common[:index]
            if not common:
                break
        return common

    @staticmethod
    def build_semantic_units(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []
        pending_prefix_blocks: list[dict[str, Any]] = []
        current_section_path: list[str] = []

        for block in blocks:
            block_type = str(block.get("type", "paragraph"))
            if block_type == "page_marker":
                pending_prefix_blocks.append(block)
                continue

            if block_type == "heading":
                heading_level = int(block.get("heading_level", 1))
                heading_text = LocalVectorDB._normalize_section_title(str(block.get("heading_text", block.get("text", ""))))
                current_section_path = [*current_section_path[: max(heading_level - 1, 0)], heading_text]
                pending_prefix_blocks.append(block)
                continue

            prefix_text = "\n\n".join(prefix_block["text"] for prefix_block in pending_prefix_blocks).strip()
            body_text = str(block.get("text", "")).strip()
            combined_text = "\n\n".join([part for part in [prefix_text, body_text] if part]).strip()
            block_types = [str(prefix_block.get("type", "paragraph")) for prefix_block in pending_prefix_blocks]
            block_types.append(block_type)
            paragraph_count = 1 if body_text else 0
            source_block_start = pending_prefix_blocks[0]["block_id"] if pending_prefix_blocks else block["block_id"]
            source_block_end = block["block_id"]

            units.append(
                {
                    "unit_type": LocalVectorDB._infer_unit_type(current_section_path, block_type),
                    "section_path": list(current_section_path),
                    "prefix_text": prefix_text,
                    "body_text": body_text,
                    "text": combined_text,
                    "block_types": block_types,
                    "paragraph_count": paragraph_count,
                    "approx_tokens": LocalVectorDB.estimate_tokens(combined_text),
                    "source_block_start": source_block_start,
                    "source_block_end": source_block_end,
                    "has_overlap_prefix": False,
                }
            )
            pending_prefix_blocks = []

        if pending_prefix_blocks:
            prefix_text = "\n\n".join(prefix_block["text"] for prefix_block in pending_prefix_blocks).strip()
            units.append(
                {
                    "unit_type": "heading",
                    "section_path": list(current_section_path),
                    "prefix_text": prefix_text,
                    "body_text": "",
                    "text": prefix_text,
                    "block_types": [str(prefix_block.get("type", "heading")) for prefix_block in pending_prefix_blocks],
                    "paragraph_count": 0,
                    "approx_tokens": LocalVectorDB.estimate_tokens(prefix_text),
                    "source_block_start": pending_prefix_blocks[0]["block_id"],
                    "source_block_end": pending_prefix_blocks[-1]["block_id"],
                    "has_overlap_prefix": False,
                }
            )

        return units

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|(?<=\.)\s+(?=\d)", normalized)
        cleaned = [sentence.strip() for sentence in sentences if sentence.strip()]
        return cleaned or [normalized]

    @staticmethod
    def _split_large_text(text: str, max_tokens: int, overlap_sentences: int = 2) -> list[tuple[str, bool]]:
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
        segments: list[str] = []
        current = ""

        def flush_current() -> None:
            nonlocal current
            if current.strip():
                segments.append(current.strip())
            current = ""

        for paragraph in paragraphs:
            paragraph_tokens = LocalVectorDB.estimate_tokens(paragraph)
            if paragraph_tokens > max_tokens:
                flush_current()
                sentences = LocalVectorDB._split_sentences(paragraph)
                sentence_buffer: list[str] = []
                for sentence in sentences:
                    candidate = " ".join([*sentence_buffer, sentence]).strip()
                    if sentence_buffer and LocalVectorDB.estimate_tokens(candidate) > max_tokens:
                        segments.append(" ".join(sentence_buffer).strip())
                        overlap = sentence_buffer[-overlap_sentences:] if overlap_sentences > 0 else []
                        sentence_buffer = [*overlap, sentence]
                    else:
                        sentence_buffer.append(sentence)
                if sentence_buffer:
                    segments.append(" ".join(sentence_buffer).strip())
                continue

            candidate = "\n\n".join([part for part in [current, paragraph] if part]).strip()
            if current and LocalVectorDB.estimate_tokens(candidate) > max_tokens:
                flush_current()
                current = paragraph
            else:
                current = candidate

        flush_current()
        if not segments:
            return []

        with_overlap: list[tuple[str, bool]] = []
        previous_sentences: list[str] = []
        for index, segment in enumerate(segments):
            if index == 0:
                with_overlap.append((segment, False))
                previous_sentences = LocalVectorDB._split_sentences(segment)
                continue
            overlap_prefix = previous_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
            prefixed = " ".join([*overlap_prefix, segment]).strip()
            with_overlap.append((prefixed, bool(overlap_prefix)))
            previous_sentences = LocalVectorDB._split_sentences(segment)
        return with_overlap

    @staticmethod
    def _split_large_unit(unit: dict[str, Any], max_tokens: int) -> list[dict[str, Any]]:
        if LocalVectorDB.estimate_tokens(unit.get("text", "")) <= max_tokens:
            return [unit]

        prefix_text = str(unit.get("prefix_text", "")).strip()
        body_text = str(unit.get("body_text", "")).strip()
        if not body_text:
            return [unit]

        split_parts = LocalVectorDB._split_large_text(body_text, max_tokens=max_tokens)
        if len(split_parts) <= 1:
            return [unit]

        split_units: list[dict[str, Any]] = []
        for index, (segment_text, has_overlap) in enumerate(split_parts):
            combined_text = "\n\n".join(
                [part for part in [prefix_text if index == 0 else "", segment_text] if part]
            ).strip()
            split_units.append(
                {
                    **unit,
                    "text": combined_text,
                    "body_text": segment_text,
                    "prefix_text": prefix_text if index == 0 else "",
                    "paragraph_count": max(1, len([part for part in re.split(r"\n\s*\n", segment_text) if part.strip()])),
                    "approx_tokens": LocalVectorDB.estimate_tokens(combined_text),
                    "has_overlap_prefix": has_overlap,
                }
            )
        return split_units

    @staticmethod
    def _make_chunk(chunk_id: int, units: list[dict[str, Any]]) -> dict[str, Any]:
        chunk_text = "\n\n".join([str(unit.get("text", "")).strip() for unit in units if str(unit.get("text", "")).strip()]).strip()
        section_paths = [list(unit.get("section_path", [])) for unit in units if unit.get("section_path") is not None]
        block_types: list[str] = []
        for unit in units:
            for block_type in unit.get("block_types", []):
                if block_type not in block_types:
                    block_types.append(block_type)
        source_block_start = min(int(unit.get("source_block_start", 0)) for unit in units)
        source_block_end = max(int(unit.get("source_block_end", 0)) for unit in units)
        return {
            "chunk_id": chunk_id,
            "text": chunk_text,
            "section_path": LocalVectorDB._longest_common_section_path(section_paths),
            "block_types": block_types,
            "paragraph_count": sum(int(unit.get("paragraph_count", 0)) for unit in units),
            "approx_tokens": LocalVectorDB.estimate_tokens(chunk_text),
            "has_overlap_prefix": any(bool(unit.get("has_overlap_prefix", False)) for unit in units),
            "source_block_start": source_block_start,
            "source_block_end": source_block_end,
            "start": source_block_start,
            "end": source_block_end,
        }

    @staticmethod
    def assemble_chunks(
        units: list[dict[str, Any]],
        target_token_range: tuple[int, int] = (400, 800),
        target_paragraph_range: tuple[int, int] = (2, 4),
    ) -> list[dict[str, Any]]:
        min_tokens, max_tokens = target_token_range
        min_paragraphs, max_paragraphs = target_paragraph_range

        expanded_units: list[dict[str, Any]] = []
        for unit in units:
            expanded_units.extend(LocalVectorDB._split_large_unit(unit, max_tokens=max_tokens))

        chunks: list[dict[str, Any]] = []
        current_units: list[dict[str, Any]] = []
        current_section_path: list[str] = []

        def flush_current() -> None:
            nonlocal current_units, current_section_path
            if current_units:
                chunks.append(LocalVectorDB._make_chunk(len(chunks), current_units))
            current_units = []
            current_section_path = []

        for unit in expanded_units:
            unit_type = str(unit.get("unit_type", "paragraph"))
            is_standalone = unit_type in {"table", "figure_like"}
            unit_section_path = list(unit.get("section_path", []))

            if is_standalone:
                flush_current()
                chunks.append(LocalVectorDB._make_chunk(len(chunks), [unit]))
                continue

            if not current_units:
                current_units = [unit]
                current_section_path = unit_section_path
                continue

            same_section = current_section_path == unit_section_path
            candidate_units = [*current_units, unit]
            candidate_tokens = LocalVectorDB.estimate_tokens("\n\n".join(candidate["text"] for candidate in candidate_units))
            candidate_paragraphs = sum(int(candidate.get("paragraph_count", 0)) for candidate in candidate_units)
            current_tokens = LocalVectorDB.estimate_tokens("\n\n".join(candidate["text"] for candidate in current_units))
            current_paragraphs = sum(int(candidate.get("paragraph_count", 0)) for candidate in current_units)
            current_is_sized = current_tokens >= min_tokens or current_paragraphs >= min_paragraphs

            if (not same_section) or (
                current_is_sized and (candidate_tokens > max_tokens or candidate_paragraphs > max_paragraphs)
            ):
                flush_current()
                current_units = [unit]
                current_section_path = unit_section_path
                continue

            current_units.append(unit)

        flush_current()
        return chunks

    @staticmethod
    def chunk_markdown(markdown: str) -> list[dict[str, Any]]:
        blocks = LocalVectorDB.parse_markdown_blocks(markdown)
        if not blocks:
            return []
        units = LocalVectorDB.build_semantic_units(blocks)
        return LocalVectorDB.assemble_chunks(units)

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
                and meta.get("chunking_version") == CHUNKING_VERSION
                and int(meta.get("num_chunks", -1)) == len(chunks)
            ):
                vectors = np.load(vectors_path)
                return {
                    "source_id": source_key,
                    "store_dir": str(store_dir),
                    "num_chunks": len(chunks),
                    "embedding_dim": int(vectors.shape[1]),
                    "chunking_version": CHUNKING_VERSION,
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
                    "chunking_version": CHUNKING_VERSION,
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
            "chunking_version": CHUNKING_VERSION,
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
        chunk_texts = [str(chunk.get("text", "")) for chunk in chunks]

        vec_norm = np.linalg.norm(vectors, axis=1)
        q_norm = np.linalg.norm(query_vec)
        denom = (vec_norm * q_norm) + 1e-12
        cosine_scores = np.dot(vectors, query_vec) / denom
        bm25_scores = self._bm25_scores(query, chunk_texts)

        cosine_norm = self._normalize_scores(cosine_scores)
        bm25_norm = self._normalize_scores(bm25_scores)
        ensemble_scores = (0.5 * cosine_norm) + (0.5 * bm25_norm)

        top_idx = sorted(
            range(len(chunks)),
            key=lambda index: (float(ensemble_scores[index]), float(bm25_norm[index]), float(cosine_norm[index])),
            reverse=True,
        )[:top_k]
        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(top_idx, start=1):
            c = chunks[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "score": float(ensemble_scores[int(idx)]),
                    "cosine_score": float(cosine_scores[int(idx)]),
                    "bm25_score": float(bm25_scores[int(idx)]),
                    "chunk_id": c["chunk_id"],
                    "start": c["start"],
                    "end": c["end"],
                    "text": c["text"],
                }
            )

        return results
