from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.utils import analyze_feature_text, create_openai_client, extract_json_object, run_text_response
from src.utils.spec_builder import chunk_refs, summarize_evidence_snippet


class FeatureEvidenceState(TypedDict, total=False):
    selected_model_name: str
    feature_rows: list[dict[str, Any]]
    feature_result: dict[str, Any]
    error: str


class FeatureEvidenceAgent:
    def __init__(self, model_name: str = "gpt-5.2") -> None:
        self.client, self.model_name = create_openai_client(model_name)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(FeatureEvidenceState)
        graph.add_node("extract_feature_evidence", self._extract_feature_evidence)
        graph.set_entry_point("extract_feature_evidence")
        graph.add_edge("extract_feature_evidence", END)
        return graph.compile()

    def invoke(self, selected_model_name: str, feature_rows: list[dict[str, Any]]) -> FeatureEvidenceState:
        return self.graph.invoke({"selected_model_name": selected_model_name, "feature_rows": feature_rows})

    def _extract_feature_evidence(self, state: FeatureEvidenceState) -> FeatureEvidenceState:
        feature_rows = state.get("feature_rows", []) or []
        selected_model_name = state.get("selected_model_name", "Not found")
        payload = {
            "selected_model_name": selected_model_name,
            "feature_rows": [
                {
                    "rank": row.get("rank"),
                    "chunk_id": row.get("chunk_id"),
                    "matched_queries": row.get("matched_queries", []),
                    "matched_keywords": row.get("matched_keywords", []),
                    "text": row.get("text", "")[:1800],
                }
                for row in feature_rows
            ],
        }
        try:
            raw = run_text_response(self.client, self.model_name, self._build_prompt(payload), "화학 feature 근거 추출")
            parsed = extract_json_object(raw)
        except (RuntimeError, ValueError):
            parsed = self._fallback_result(feature_rows)

        feature_section = parsed.get("feature", {}) if isinstance(parsed.get("feature"), dict) else {}
        return {"feature_result": {"feature": feature_section}}

    def _build_prompt(self, payload: dict[str, Any]) -> str:
        schema = {
            "feature": {
                "summary": "Short factual summary of the selected model feature representation.",
                "key_values": "MolWt, TPSA, atom count, ECFP4 2048 bits, mw, polararea, C/N/O number",
                "raw_feature_mentions": ["MolWt", "TPSA", "atom count", "mw", "polararea", "C/N/O number"],
                "raw_tool_mentions": ["RDKit descriptors", "ECFP4"],
                "evidence_chunks": ["chunk-0"],
                "evidence_snippet": "Short evidence snippet.",
            }
        }
        return (
            "You are extracting chemistry feature evidence for one already-selected boiling point regression model.\n"
            "Extract only the feature representation actually used by that selected model.\n"
            "Keep raw feature mentions exactly as written in the evidence where possible.\n"
            "If the paper describes dataset characteristics, transformed variables, or discretized labels, include those exact names in raw_feature_mentions.\n"
            "Examples include cmpdname, mw, mf, polararea, heavycnt, hbondacc, iso smiles, C number, N number, O number, C/N/O number, side chain number, hydrocarbon, alcohol, amine.\n"
            "If an exact feature list appears inside parentheses or a table description, enumerate every item individually instead of collapsing it into generic phrases like '12 characteristics'.\n"
            "Treat data cleaning, data transformation, data discretization, and data integration rows as valid upstream evidence for the final regression input features.\n"
            "Use generic count phrases like '12 characteristics' only when no exact list is visible in the evidence.\n"
            "Do not silently normalize unsupported external chemistry tools.\n\n"
            "Output rules:\n"
            "1) Return exactly one JSON object.\n"
            "2) raw_feature_mentions should include descriptors, dataset columns, transformed count-style variables, discretized class labels, and representation terms explicitly mentioned in evidence.\n"
            "3) raw_tool_mentions should include tool or family names like RDKit, ECFP4, MACCS, Mordred, PaDEL, Dragon.\n"
            "4) evidence_chunks must reference supporting chunk ids.\n\n"
            f"JSON schema example:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
            f"Input payload:\n{json.dumps(payload, ensure_ascii=False)}"
        )

    def _fallback_result(self, feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
        combined_text = "\n".join([str(row.get("text", "")) for row in feature_rows])
        analysis = analyze_feature_text(combined_text)
        raw_terms = analysis["feature_terms"]
        if analysis["fingerprint_family"]:
            raw_terms.append(analysis["fingerprint_family"])
        return {
            "feature": {
                "summary": combined_text[:260] if combined_text else "Not found",
                "key_values": ", ".join(raw_terms) if raw_terms else "Not found",
                "raw_feature_mentions": raw_terms,
                "raw_tool_mentions": analysis["unresolved_feature_terms"],
                "evidence_chunks": chunk_refs(feature_rows),
                "evidence_snippet": summarize_evidence_snippet(feature_rows),
            }
        }
