from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import src.services.vector_db_service as vector_db_module
from src.services.vector_db_service import CHUNKING_VERSION, LocalVectorDB


EXAMPLE_MARKDOWN = """
<!-- [Page 1] -->

# Integrating Data Science and Machine Learning to Chemistry Education

Published as part of Journal of Chemical Education.

## ABSTRACT

Artificial intelligence (AI) and data science (DS) are receiving a lot of attention in various fields. The Random Forest model accurately predicted whether the type of compound in the unknown dataset was hydrocarbons, alcohols, or amines.

**KEYWORDS:** High School, Introductory Chemistry

## INTRODUCTION

As digital technology takes root throughout society, humanity is entering an era of digital transformation. The era of digital transformation utilizes digital technologies such as data science (DS) and artificial intelligence (AI), and AI provides insights from big data.

In the educational field, there is a need to explore ways to integrate AI, machine learning and data science in each subject. Recently, AI tools corresponding to the first and second types have appeared.

<!-- [Page 2] -->

Figure 1. Example workflow for Orange3 drag and drop modeling.

| Model | MAE |
| --- | --- |
| Random Forest | 19.876 |

## Conclusion

The activity provides meaningful implications for how AI/DS technology could be integrated into each domain. The regression results were useful for predicting boiling points of arbitrary compounds.
""".strip()


class SemanticChunkingTests(unittest.TestCase):
    def test_chunk_markdown_creates_semantic_chunks(self) -> None:
        chunks = LocalVectorDB.chunk_markdown(EXAMPLE_MARKDOWN)

        self.assertGreaterEqual(len(chunks), 4)
        self.assertTrue(any("## ABSTRACT" in chunk["text"] for chunk in chunks))
        self.assertTrue(any("| Model | MAE |" in chunk["text"] for chunk in chunks))
        self.assertTrue(any("Figure 1. Example workflow" in chunk["text"] for chunk in chunks))
        self.assertTrue(any("## Conclusion" in chunk["text"] for chunk in chunks))

    def test_chunk_markdown_does_not_split_only_by_page_markers(self) -> None:
        chunks = LocalVectorDB.chunk_markdown(EXAMPLE_MARKDOWN)
        self.assertTrue(any("<!-- [Page 2] -->" in chunk["text"] for chunk in chunks))
        self.assertFalse(any(chunk["text"].strip() == "<!-- [Page 2] -->" for chunk in chunks))

    def test_chunk_markdown_preserves_each_content_block_once_except_overlap(self) -> None:
        blocks = LocalVectorDB.parse_markdown_blocks(EXAMPLE_MARKDOWN)
        chunks = LocalVectorDB.chunk_markdown(EXAMPLE_MARKDOWN)

        covered = set()
        for chunk in chunks:
            for block_index in range(chunk["source_block_start"], chunk["source_block_end"] + 1):
                covered.add(block_index)

        self.assertEqual(covered, {block["block_id"] for block in blocks})

    def test_chunk_markdown_splits_large_sections_with_overlap(self) -> None:
        long_paragraph = " ".join(["This sentence explains the boiling point model." for _ in range(120)])
        markdown = "\n\n".join(
            [
                "# Title",
                "## INTRODUCTION",
                long_paragraph,
            ]
        )

        chunks = LocalVectorDB.chunk_markdown(markdown)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any(chunk["has_overlap_prefix"] for chunk in chunks[1:]))

    def test_chunk_markdown_marks_table_and_figure_as_standalone_candidates(self) -> None:
        chunks = LocalVectorDB.chunk_markdown(EXAMPLE_MARKDOWN)

        table_chunk = next(chunk for chunk in chunks if "| Model | MAE |" in chunk["text"])
        figure_chunk = next(chunk for chunk in chunks if "Figure 1. Example workflow" in chunk["text"])

        self.assertEqual(table_chunk["block_types"], ["table"])
        self.assertIn("figure_like", figure_chunk["block_types"])
        self.assertTrue(any("**KEYWORDS:**" in chunk["text"] and "## ABSTRACT" in chunk["text"] for chunk in chunks))


class VectorDBIntegrationTests(unittest.TestCase):
    def _build_fake_vdb(self) -> LocalVectorDB:
        vdb = LocalVectorDB.__new__(LocalVectorDB)
        vdb.client = None
        vdb.embedding_model = "test-embedding-model"
        return vdb

    def test_build_from_markdown_uses_chunking_version_for_cache(self) -> None:
        markdown = EXAMPLE_MARKDOWN
        vdb = self._build_fake_vdb()

        def fake_embed_texts(texts: list[str]) -> np.ndarray:
            return np.array([[float(index + 1), 0.0] for index, _ in enumerate(texts)], dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            with patch.object(vector_db_module, "VECTORSTORE_ROOT", temp_root):
                with patch.object(LocalVectorDB, "_embed_texts", side_effect=fake_embed_texts):
                    first = vdb.build_from_markdown(markdown, source_id="paper")
                    second = vdb.build_from_markdown(markdown, source_id="paper")
                    self.assertFalse(first["cached"])
                    self.assertTrue(second["cached"])

                with patch.object(vector_db_module, "CHUNKING_VERSION", CHUNKING_VERSION + "_v2"):
                    with patch.object(LocalVectorDB, "_embed_texts", side_effect=fake_embed_texts):
                        third = vdb.build_from_markdown(markdown, source_id="paper")
                        self.assertFalse(third["cached"])

    def test_retrieve_keeps_existing_result_shape(self) -> None:
        markdown = EXAMPLE_MARKDOWN
        vdb = self._build_fake_vdb()

        def fake_embed_texts(texts: list[str]) -> np.ndarray:
            rows = []
            for index, _ in enumerate(texts):
                rows.append([1.0 if index == 0 else 0.1, float(index)])
            return np.array(rows, dtype=np.float32)

        def fake_embed_query(_: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            with patch.object(vector_db_module, "VECTORSTORE_ROOT", temp_root):
                with patch.object(LocalVectorDB, "_embed_texts", side_effect=fake_embed_texts):
                    vdb.build_from_markdown(markdown, source_id="paper")
                with patch.object(LocalVectorDB, "_embed_query", side_effect=fake_embed_query):
                    rows = vdb.retrieve(source_id="paper", query="abstract", top_k=2)

        self.assertEqual(len(rows), 2)
        self.assertIn("chunk_id", rows[0])
        self.assertIn("score", rows[0])
        self.assertIn("text", rows[0])
        self.assertIn("start", rows[0])
        self.assertIn("end", rows[0])
        self.assertIn("cosine_score", rows[0])
        self.assertIn("bm25_score", rows[0])

    def test_retrieve_uses_bm25_and_cosine_ensemble(self) -> None:
        vdb = self._build_fake_vdb()

        def fake_embed_query(_: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            with patch.object(vector_db_module, "VECTORSTORE_ROOT", temp_root):
                store_dir = temp_root / "paper"
                store_dir.mkdir(parents=True, exist_ok=True)
                chunks = [
                    {"chunk_id": 0, "start": 0, "end": 0, "text": "General chemistry education and regression modeling overview."},
                    {"chunk_id": 1, "start": 1, "end": 1, "text": "Orange3 drag and drop workflow is used for boiling point prediction."},
                ]
                (store_dir / "chunks.json").write_text(vector_db_module.json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
                np.save(store_dir / "embeddings.npy", np.array([[1.0, 0.0], [0.2, 0.0]], dtype=np.float32))
                with patch.object(LocalVectorDB, "_embed_query", side_effect=fake_embed_query):
                    rows = vdb.retrieve(source_id="paper", query="Orange3 workflow", top_k=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["chunk_id"], 1)
        self.assertGreaterEqual(rows[0]["bm25_score"], 0.0)
        self.assertGreaterEqual(rows[0]["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
