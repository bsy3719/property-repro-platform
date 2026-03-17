from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.utils import canonical_model_name, create_openai_client, extract_json_object, run_text_response
from src.utils.spec_builder import chunk_refs, summarize_evidence_snippet


class ModelSelectionState(TypedDict, total=False):
    retrieved_by_topic: dict[str, list[dict[str, Any]]]
    selection_result: dict[str, Any]
    error: str


class ModelSelectionAgent:
    def __init__(self, model_name: str = "gpt-5.2") -> None:
        self.client, self.model_name = create_openai_client(model_name)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ModelSelectionState)
        graph.add_node("select_model", self._select_model)
        graph.set_entry_point("select_model")
        graph.add_edge("select_model", END)
        return graph.compile()

    def invoke(self, retrieved_by_topic: dict[str, list[dict[str, Any]]]) -> ModelSelectionState:
        return self.graph.invoke({"retrieved_by_topic": retrieved_by_topic})

    def _select_model(self, state: ModelSelectionState) -> ModelSelectionState:
        retrieved_by_topic = state.get("retrieved_by_topic", {})
        payload = {
            "model": self._build_payload_rows(retrieved_by_topic.get("model", [])),
            "metrics": self._build_payload_rows(retrieved_by_topic.get("metrics", [])),
            "training": self._build_payload_rows(retrieved_by_topic.get("training", [])),
        }
        try:
            raw = run_text_response(self.client, self.model_name, self._build_prompt(payload), "최종 모델 선택")
            parsed = extract_json_object(raw)
        except (RuntimeError, ValueError):
            parsed = self._fallback_result(retrieved_by_topic)

        model = parsed.get("model", {}) if isinstance(parsed.get("model"), dict) else {}
        model["name"] = canonical_model_name(model.get("name"))
        selection_basis = parsed.get("selection_basis", {}) if isinstance(parsed.get("selection_basis"), dict) else {}
        selected_terms = parsed.get("selected_model_terms", [])
        if not selected_terms:
            selected_terms = [model.get("name", "")]

        return {
            "selection_result": {
                "selection_basis": selection_basis,
                "model": model,
                "selected_model_terms": selected_terms,
            }
        }

    def _build_payload_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "rank": row.get("rank"),
                "chunk_id": row.get("chunk_id"),
                "matched_queries": row.get("matched_queries", []),
                "matched_keywords": row.get("matched_keywords", []),
                "text": row.get("text", "")[:1600],
            }
            for row in rows
        ]

    def _build_prompt(self, payload: dict[str, Any]) -> str:
        schema = {
            "selection_basis": {
                "summary": "Why this is the single final or best boiling point regression model.",
                "key_values": "best final model selection rationale",
                "evidence_chunks": ["chunk-0"],
                "evidence_snippet": "Short evidence snippet.",
            },
            "model": {
                "summary": "Short factual summary of the selected model.",
                "key_values": "Random forest regressor",
                "name": "RandomForestRegressor",
                "evidence_chunks": ["chunk-0"],
                "evidence_snippet": "Short evidence snippet.",
            },
            "selected_model_terms": ["random forest", "randomforestregressor"],
        }
        return (
            "You are selecting one final model for boiling point regression from retrieved paper evidence.\n"
            "Choose only the single final reported or best-performing boiling point regression model.\n"
            "Do not mix settings across multiple models.\n"
            "Ignore other target properties and ignore classification tasks.\n\n"
            "Output rules:\n"
            "1) Return exactly one JSON object.\n"
            "2) model.name must be one of: RandomForestRegressor, Ridge, Lasso, ElasticNet, SVR, KNeighborsRegressor, MLPRegressor, Not found.\n"
            "3) selected_model_terms should contain lower-variance aliases or phrases useful for filtering other evidence rows to the same model.\n"
            "4) evidence_chunks must point to the supporting chunk ids.\n\n"
            f"JSON schema example:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
            f"Retrieved evidence:\n{json.dumps(payload, ensure_ascii=False)}"
        )

    def _fallback_result(self, retrieved_by_topic: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        model_rows = retrieved_by_topic.get("model", []) or []
        top_row = model_rows[0] if model_rows else {}
        model_text = str(top_row.get("text", ""))
        model_name = canonical_model_name(model_text)
        evidence_chunks = chunk_refs(model_rows)
        evidence_snippet = summarize_evidence_snippet(model_rows)
        return {
            "selection_basis": {
                "summary": "Fallback selection used the top retrieved model evidence row.",
                "key_values": "fallback top model row",
                "evidence_chunks": evidence_chunks,
                "evidence_snippet": evidence_snippet,
            },
            "model": {
                "summary": model_text[:220] if model_text else "Not found",
                "key_values": model_name,
                "name": model_name,
                "evidence_chunks": evidence_chunks,
                "evidence_snippet": evidence_snippet,
            },
            "selected_model_terms": [model_name.lower()] if model_name and model_name != "Not found" else [],
        }
