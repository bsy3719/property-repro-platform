from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.utils import normalize_paper_method_spec
from src.utils.spec_builder import assemble_paper_method_spec


class RetrieverSummaryState(TypedDict, total=False):
    selection_result: dict[str, Any]
    feature_result: dict[str, Any]
    method_result: dict[str, Any]
    filtered_by_topic: dict[str, list[dict[str, Any]]]
    feature_rows: list[dict[str, Any]]
    paper_method_spec: dict[str, Any]
    error: str


class RetrieverTableAgent:
    """Deterministically assembles only a codegen-ready paper_method_spec."""

    def __init__(self, model_name: str = "gpt-5.2") -> None:
        self.model_name = model_name
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(RetrieverSummaryState)
        graph.add_node("assemble_spec", self._assemble_spec)
        graph.set_entry_point("assemble_spec")
        graph.add_edge("assemble_spec", END)
        return graph.compile()

    def _assemble_spec(self, state: RetrieverSummaryState) -> RetrieverSummaryState:
        paper_method_spec = assemble_paper_method_spec(
            selection_result=state.get("selection_result", {}),
            feature_result=state.get("feature_result", {}),
            method_result=state.get("method_result", {}),
            filtered_by_topic=state.get("filtered_by_topic", {}),
            feature_rows=state.get("feature_rows", []),
        )
        normalized = normalize_paper_method_spec(paper_method_spec)
        return {"paper_method_spec": normalized}

    def invoke(
        self,
        *,
        selection_result: dict[str, Any],
        feature_result: dict[str, Any],
        method_result: dict[str, Any],
        filtered_by_topic: dict[str, list[dict[str, Any]]],
        feature_rows: list[dict[str, Any]] | None = None,
    ) -> RetrieverSummaryState:
        return self.graph.invoke(
            {
                "selection_result": selection_result,
                "feature_result": feature_result,
                "method_result": method_result,
                "filtered_by_topic": filtered_by_topic,
                "feature_rows": feature_rows or [],
            }
        )
