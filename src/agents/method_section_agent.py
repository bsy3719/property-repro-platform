from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.utils import create_openai_client, extract_json_object, run_text_response
from src.utils.spec_builder import chunk_refs, summarize_evidence_snippet


class MethodSectionState(TypedDict, total=False):
    selected_model_name: str
    retrieved_by_topic: dict[str, list[dict[str, Any]]]
    method_result: dict[str, Any]
    error: str


class MethodSectionAgent:
    def __init__(self, model_name: str = "gpt-5.2") -> None:
        self.client, self.model_name = create_openai_client(model_name)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(MethodSectionState)
        graph.add_node("extract_method_sections", self._extract_method_sections)
        graph.set_entry_point("extract_method_sections")
        graph.add_edge("extract_method_sections", END)
        return graph.compile()

    def invoke(self, selected_model_name: str, retrieved_by_topic: dict[str, list[dict[str, Any]]]) -> MethodSectionState:
        return self.graph.invoke({"selected_model_name": selected_model_name, "retrieved_by_topic": retrieved_by_topic})

    def _extract_method_sections(self, state: MethodSectionState) -> MethodSectionState:
        selected_model_name = state.get("selected_model_name", "Not found")
        retrieved_by_topic = state.get("retrieved_by_topic", {})
        payload = {"selected_model_name": selected_model_name}
        payload.update(
            {
                topic: [
                    {
                        "rank": row.get("rank"),
                        "chunk_id": row.get("chunk_id"),
                        "matched_queries": row.get("matched_queries", []),
                        "matched_keywords": row.get("matched_keywords", []),
                        "text": row.get("text", "")[:1800],
                    }
                    for row in rows
                ]
                for topic, rows in retrieved_by_topic.items()
            }
        )

        try:
            raw = run_text_response(self.client, self.model_name, self._build_prompt(payload), "방법론 section 추출")
            parsed = extract_json_object(raw)
        except (RuntimeError, ValueError):
            parsed = self._fallback_result(retrieved_by_topic)

        return {
            "method_result": {
                "preprocessing": parsed.get("preprocessing", {}),
                "hyperparameters": parsed.get("hyperparameters", {}),
                "training": parsed.get("training", {}),
                "metrics": parsed.get("metrics", {}),
            }
        }

    def _build_prompt(self, payload: dict[str, Any]) -> str:
        schema = {
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
            "You are extracting preprocessing, hyperparameters, training, and metrics for one selected boiling point regression model.\n"
            "All returned details must align to that same model. Do not mix rows from different models.\n"
            "Use 'Not found' for missing text fields and null for missing numeric/boolean fields.\n"
            "metrics.reported may include only MAE, RMSE, MSE, R2.\n"
            "Return exactly one JSON object.\n\n"
            f"JSON schema example:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
            f"Input payload:\n{json.dumps(payload, ensure_ascii=False)}"
        )

    def _fallback_result(self, retrieved_by_topic: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        training_rows = retrieved_by_topic.get("training", []) or []
        metrics_rows = retrieved_by_topic.get("metrics", []) or []
        preprocessing_rows = retrieved_by_topic.get("preprocessing", []) or training_rows
        hyperparameter_rows = retrieved_by_topic.get("hyperparameter", []) or []

        training_text = "\n".join([str(row.get("text", "")) for row in training_rows])
        metrics_text = "\n".join([str(row.get("text", "")) for row in metrics_rows])

        test_size = None
        test_match = re.search(r"test[_ -]?size\s*[:=]?\s*(0?\.\d+)", training_text.lower())
        if test_match:
            test_size = float(test_match.group(1))

        random_state = None
        seed_match = re.search(r"(?:random[_ -]?state|seed)\s*[:=]?\s*(\d+)", training_text.lower())
        if seed_match:
            random_state = int(seed_match.group(1))

        reported: dict[str, float] = {}
        for metric_name in ["MAE", "RMSE", "MSE", "R2"]:
            match = re.search(rf"\b{metric_name}\b\s*[:=]?\s*(-?\d+(?:\.\d+)?)", metrics_text, re.IGNORECASE)
            if match:
                reported[metric_name] = float(match.group(1))

        return {
            "preprocessing": {
                "summary": summarize_evidence_snippet(preprocessing_rows),
                "key_values": "Not found",
                "invalid_smiles": "Not found",
                "missing_target": "Not found",
                "missing_features": "Not found",
                "duplicates": "Not found",
                "scaling": None,
                "evidence_chunks": chunk_refs(preprocessing_rows),
                "evidence_snippet": summarize_evidence_snippet(preprocessing_rows),
            },
            "hyperparameters": {
                "summary": summarize_evidence_snippet(hyperparameter_rows),
                "key_values": "Not found",
                "values": {},
                "evidence_chunks": chunk_refs(hyperparameter_rows),
                "evidence_snippet": summarize_evidence_snippet(hyperparameter_rows),
            },
            "training": {
                "summary": summarize_evidence_snippet(training_rows),
                "key_values": "Not found",
                "split_strategy": "Not found",
                "test_size": test_size,
                "random_state": random_state,
                "evidence_chunks": chunk_refs(training_rows),
                "evidence_snippet": summarize_evidence_snippet(training_rows),
            },
            "metrics": {
                "summary": summarize_evidence_snippet(metrics_rows),
                "key_values": "Not found",
                "reported": reported,
                "evidence_chunks": chunk_refs(metrics_rows),
                "evidence_snippet": summarize_evidence_snippet(metrics_rows),
            },
        }
