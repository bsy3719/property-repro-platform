from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import END, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"


class RetrieverSummaryState(TypedDict, total=False):
    retrieved_by_topic: dict[str, list[dict[str, Any]]]
    summary_markdown: str
    error: str


class RetrieverTableAgent:
    """Builds a report-oriented summary from multi-topic retrieval results."""

    def __init__(self, model_name: str = "gpt-5-mini") -> None:
        load_dotenv(DOTENV_PATH)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(RetrieverSummaryState)
        graph.add_node("make_summary", self._make_summary)
        graph.set_entry_point("make_summary")
        graph.add_edge("make_summary", END)
        return graph.compile()

    def _make_summary(self, state: RetrieverSummaryState) -> RetrieverSummaryState:
        retrieved_by_topic = state.get("retrieved_by_topic", {})
        if not retrieved_by_topic:
            return {"error": "정리할 리트리버 결과가 없습니다."}

        payload: dict[str, Any] = {}
        for topic, rows in retrieved_by_topic.items():
            payload[topic] = [
                {
                    "rank": r.get("rank"),
                    "score": r.get("score"),
                    "chunk_id": r.get("chunk_id"),
                    "text": r.get("text", "")[:1600],
                }
                for r in rows
            ]

        prompt = (
            "You are a research assistant preparing evidence for a reproducibility report.\n"
            "Summarize what the paper did for each topic: model, feature, hyperparameter, training, metrics.\n"
            "Focus on factual, implementation-relevant details from retrieved evidence.\n\n"
            "Output format (markdown only):\n"
            "# Retrieved Paper Method Summary\n"
            "## model\n"
            "- What paper did: ...\n"
            "- Key values/settings: ...\n"
            "- Evidence chunks: ...\n"
            "- Evidence snippets: ...\n"
            "## feature\n"
            "... same structure for all topics ...\n"
            "## hyperparameter\n"
            "## training\n"
            "## metrics\n"
            "## Report Comparison Keys\n"
            "- Bullet list of fields to compare with reproduced run results.\n\n"
            "Rules:\n"
            "1) Do not invent values. If missing, write 'Not found'.\n"
            "2) Keep each topic concise but specific enough for final comparison report writing.\n"
            "3) Preserve critical numbers exactly when present.\n"
            "4) Mention chunk references as chunk-<id>.\n"
            "5) Return markdown only.\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
        )
        summary_md = (response.output_text or "").strip()
        if not summary_md:
            return {"error": "요약 에이전트가 결과를 반환하지 않았습니다."}

        return {"summary_markdown": summary_md}

    def invoke(self, retrieved_by_topic: dict[str, list[dict[str, Any]]]) -> RetrieverSummaryState:
        return self.graph.invoke({"retrieved_by_topic": retrieved_by_topic})
