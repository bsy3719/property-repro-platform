from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.utils import create_openai_client, extract_json_object, run_text_response


class ColumnSelectionState(TypedDict, total=False):
    dataset_profile: dict[str, Any]
    target_property: str
    suggestion: dict[str, Any]
    error: str


class ColumnSelectionAgent:
    """Suggests dataset columns for SMILES and target property selection."""

    def __init__(self, model_name: str = "gpt-5.2") -> None:
        self.client, self.model_name = create_openai_client(model_name)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ColumnSelectionState)
        graph.add_node("select_columns", self._select_columns)
        graph.set_entry_point("select_columns")
        graph.add_edge("select_columns", END)
        return graph.compile()

    def _select_columns(self, state: ColumnSelectionState) -> ColumnSelectionState:
        dataset_profile = state.get("dataset_profile", {})
        if not dataset_profile:
            return {"error": "컬럼 추천에 필요한 데이터 프로필이 없습니다."}

        prompt = self._build_prompt(
            dataset_profile=dataset_profile,
            target_property=state.get("target_property", "boiling point"),
        )
        try:
            raw_response = run_text_response(self.client, self.model_name, prompt, "컬럼 자동 추천")
            parsed = extract_json_object(raw_response)
        except (RuntimeError, ValueError) as exc:
            return {"error": str(exc)}

        return {"suggestion": self._normalize_suggestion(parsed, dataset_profile)}

    def invoke(self, dataset_profile: dict[str, Any], target_property: str = "boiling point") -> ColumnSelectionState:
        return self.graph.invoke({"dataset_profile": dataset_profile, "target_property": target_property})

    def _build_prompt(self, dataset_profile: dict[str, Any], target_property: str) -> str:
        schema = {
            "smiles_column": "SMILES_COL or null",
            "target_column": "TARGET_COL or null",
            "smiles_confidence": "high",
            "target_confidence": "medium",
            "reasoning": {
                "smiles": "Why this column is the SMILES column.",
                "target": f"Why this column is the {target_property} column.",
            },
        }
        return (
            "You are a data-column selection agent for a chemistry regression app.\n"
            "Choose the most likely SMILES column and the most likely target-property column.\n"
            "The target property is boiling point.\n"
            "Use only the provided dataset profile.\n\n"
            "Rules:\n"
            "1) Return exactly one JSON object and nothing else.\n"
            "2) smiles_column and target_column must be one of the provided column names or null.\n"
            "3) Prefer columns whose sample values and dtype support the choice.\n"
            "4) SMILES columns usually contain molecular string patterns such as C, N, O, Cl, Br, =, #, parentheses, or brackets.\n"
            "5) Boiling point columns are usually numeric and may have names like boiling point, bp, tb, temp, or similar.\n"
            "6) If uncertain, choose the best candidate and lower the confidence.\n"
            "7) Use confidence values only from: high, medium, low.\n"
            "8) Keep reasoning short.\n\n"
            f"JSON schema example:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
            f"Dataset profile JSON:\n{json.dumps(dataset_profile, ensure_ascii=False)}"
        )

    def _normalize_suggestion(self, parsed: dict[str, Any], dataset_profile: dict[str, Any]) -> dict[str, Any]:
        valid_columns = {str(column) for column in dataset_profile.get("columns", [])}

        def normalize_column(value: Any) -> str | None:
            if value is None:
                return None
            text = str(value).strip()
            return text if text in valid_columns else None

        def normalize_confidence(value: Any) -> str:
            text = str(value).strip().lower() if value is not None else ""
            if text in {"high", "medium", "low"}:
                return text
            return "low"

        reasoning = parsed.get("reasoning", {}) if isinstance(parsed.get("reasoning"), dict) else {}
        return {
            "smiles_column": normalize_column(parsed.get("smiles_column")),
            "target_column": normalize_column(parsed.get("target_column")),
            "smiles_confidence": normalize_confidence(parsed.get("smiles_confidence")),
            "target_confidence": normalize_confidence(parsed.get("target_confidence")),
            "reasoning": {
                "smiles": str(reasoning.get("smiles", "")).strip(),
                "target": str(reasoning.get("target", "")).strip(),
            },
        }
