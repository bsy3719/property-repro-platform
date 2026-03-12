from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.utils import (
    build_paper_method_summary_markdown,
    create_openai_client,
    extract_json_object,
    normalize_paper_method_spec,
    run_text_response,
)


class RetrieverSummaryState(TypedDict, total=False):
    retrieved_by_topic: dict[str, list[dict[str, Any]]]
    summary_markdown: str
    paper_method_spec: dict[str, Any]
    error: str


class RetrieverTableAgent:
    """Builds a report-oriented summary and structured spec from multi-topic retrieval results."""

    def __init__(self, model_name: str = "gpt-5.2") -> None:
        self.client, self.model_name = create_openai_client(model_name)
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

        payload = self._build_payload(retrieved_by_topic)
        prompt = self._build_prompt(payload)
        try:
            raw_response = run_text_response(self.client, self.model_name, prompt, "리트리버 structured spec 생성")
            parsed_response = extract_json_object(raw_response)
        except (RuntimeError, ValueError) as exc:
            return {"error": str(exc)}

        paper_method_spec = normalize_paper_method_spec(parsed_response)
        summary_markdown = build_paper_method_summary_markdown(paper_method_spec)
        return {
            "summary_markdown": summary_markdown,
            "paper_method_spec": paper_method_spec,
        }

    def invoke(self, retrieved_by_topic: dict[str, list[dict[str, Any]]]) -> RetrieverSummaryState:
        return self.graph.invoke({"retrieved_by_topic": retrieved_by_topic})

    def _build_payload(self, retrieved_by_topic: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for topic, rows in retrieved_by_topic.items():
            payload[topic] = [
                {
                    "rank": row.get("rank"),
                    "score": row.get("score"),
                    "chunk_id": row.get("chunk_id"),
                    "query": row.get("query"),
                    "text": row.get("text", "")[:1800],
                }
                for row in rows
            ]
        return payload

    def _build_prompt(self, payload: dict[str, Any]) -> str:
        schema = {
            "selection_basis": {
                "summary": "Why this single boiling point regression model was selected as the best/final reported model.",
                "evidence_chunks": ["chunk-0"],
                "evidence_snippet": "Short evidence snippet.",
            },
            "preprocessing": {
                "summary": "Short factual summary.",
                "key_values": "invalid_smiles=drop, duplicates=drop",
                "invalid_smiles": "drop",
                "missing_target": "drop",
                "missing_features": "median_impute",
                "duplicates": "drop",
                "scaling": None,
                "evidence_chunks": ["chunk-0"],
                "evidence_snippet": "Short evidence snippet.",
            },
            "feature": {
                "summary": "Short factual summary.",
                "key_values": "Morgan fingerprint(radius=2, n_bits=2048)",
                "method": "descriptor",
                "radius": None,
                "n_bits": None,
                "use_rdkit_descriptors": None,
                "evidence_chunks": ["chunk-0"],
                "evidence_snippet": "Short evidence snippet.",
            },
            "model": {
                "summary": "Short factual summary.",
                "key_values": "RandomForestRegressor",
                "name": "RandomForestRegressor",
                "evidence_chunks": ["chunk-0"],
                "evidence_snippet": "Short evidence snippet.",
            },
            "hyperparameters": {
                "summary": "Short factual summary.",
                "key_values": "n_estimators=300, max_depth=20",
                "values": {"n_estimators": 300},
                "evidence_chunks": ["chunk-0"],
                "evidence_snippet": "Short evidence snippet.",
            },
            "training": {
                "summary": "Short factual summary.",
                "key_values": "split=train_test_split, test_size=0.2, random_state=42",
                "split_strategy": "train_test_split",
                "test_size": 0.2,
                "random_state": 42,
                "evidence_chunks": ["chunk-0"],
                "evidence_snippet": "Short evidence snippet.",
            },
            "metrics": {
                "summary": "Short factual summary.",
                "key_values": "MAE=0.12, RMSE=0.20, R2=0.91",
                "reported": {"MAE": 0.12, "RMSE": 0.20, "R2": 0.91},
                "evidence_chunks": ["chunk-0"],
                "evidence_snippet": "Short evidence snippet.",
            },
        }
        return (
            "You are a research assistant preparing evidence for a reproducibility system.\n"
            "Your job is to extract one single structured method spec for boiling point regression from retrieved evidence.\n"
            "If multiple boiling point regression models appear, select only the single best-performing or final reported model.\n"
            "All sections must stay aligned to that same selected model. Do not mix settings across different models.\n"
            "Ignore classification, ranking, or other target properties.\n\n"
            "Output rules:\n"
            "1) Return exactly one JSON object and nothing else.\n"
            "2) Use the schema below exactly.\n"
            "3) model.name must be one of: RandomForestRegressor, Ridge, Lasso, ElasticNet, SVR, KNeighborsRegressor, MLPRegressor, Not found.\n"
            "4) feature.method must be one of: descriptor, morgan, combined, Not found.\n"
            "5) Preserve numeric values exactly when evidence gives them.\n"
            "6) If a detail is missing, use 'Not found' for text fields and null for unknown numeric/boolean fields.\n"
            "7) evidence_chunks must reference the supporting chunk ids such as chunk-3.\n"
            "8) selection_basis.summary must explain why the chosen model is the best/final boiling point regression model in the paper.\n"
            "9) metrics.reported may include only MAE, RMSE, MSE, R2.\n\n"
            f"JSON schema example:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
            f"Input retrieval JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )
