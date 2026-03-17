from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.feature_evidence_agent import FeatureEvidenceAgent
from src.agents.method_section_agent import MethodSectionAgent
from src.agents.model_selection_agent import ModelSelectionAgent
from src.agents.retriever_table_agent import RetrieverTableAgent
from src.services.vector_db_service import LocalVectorDB
from src.utils.spec_builder import (
    build_selected_model_terms,
    build_spec_build_trace,
    filter_rows_for_selected_model,
    validate_paper_method_spec_contract,
)


TOPIC_QUERY_VARIANTS = {
    "model": [
        "For boiling point regression only, identify the single best-performing or final reported model. Ignore classification or other property tasks.",
        "Which regression model is finally selected or reported as best for boiling point prediction, such as Random Forest, SVR, Ridge, KNN, MLP, XGBoost, or other regressors?",
    ],
    "feature": [
        "For the same single best-performing or final reported boiling point regression model, what feature representation, molecular representation, or input features are used? Ignore classification or other property tasks.",
        "Look for feature engineering terms for the final boiling point regression model: molecular descriptors, physicochemical descriptors, constitutional descriptors, topological descriptors, electronic descriptors, RDKit descriptors, Mordred, PaDEL, Dragon.",
        "Look for fingerprint terms for the final boiling point regression model: Morgan fingerprint, ECFP, circular fingerprint, MACCS keys, atom pair fingerprint, topological torsion fingerprint, hashed fingerprint, molecular fingerprint.",
        "Look for count-based or structure-based features for the final boiling point regression model: atom counts, bond counts, ring counts, functional group counts, element counts, composition vectors, adjacency matrix, bond matrix, graph node or edge features.",
        "Look for exact feature names, dataset characteristics, cleaned variables, transformed variables, and derived chemistry counts used by the final boiling point regression model, including phrases like cmpdname, mw, mf, polararea, heavycnt, hbondacc, iso smiles, carbon number, nitrogen number, oxygen number, C/N/O number, side chain number.",
        "Look for feature selection or transformation statements for the final boiling point regression model, such as only necessary characteristics kept, new characteristics added, discretized labels, hydrocarbons, alcohols, amines, or compound class indicators.",
        "Look for dataset feature construction evidence used upstream of the final boiling point regression model: data cleaning, data transformation, data discretization, data integration, Table 1(a), Table 1(b), blue box, yellow box, green box, red box, and any parenthetical characteristic lists.",
    ],
    "preprocessing": [
        "For the same single best-performing or final boiling point regression model, what preprocessing steps are reported? Look for invalid SMILES handling, duplicate removal, missing target handling, missing value imputation, and scaling.",
        "Find preprocessing and data cleaning details for the final boiling point regression model, including descriptor imputation, feature scaling, normalization, duplicate removal, and invalid molecule filtering.",
        "Find data cleaning, data transformation, and data discretization details for the final boiling point regression model, including selected characteristics, added count variables, and class labeling steps.",
        "Find dataset construction details that define the final regression input table, including retained columns, added variables, discretized labels, Table 1(a), Table 1(b), and data integration of boiling point values.",
    ],
    "hyperparameter": [
        "For the same single best-performing or final boiling point regression model, what hyperparameters are reported? Ignore classification or other property tasks.",
        "Find parameter settings for the final boiling point regression model, such as n_estimators, max_depth, C, gamma, alpha, neighbors, hidden layers, fingerprint radius, or fingerprint bit length.",
    ],
    "training": [
        "For the same single best-performing or final boiling point regression model, what training setup is used (split, seed, optimizer, lr, epoch, batch, loss, validation, early stopping)? Ignore classification or other property tasks.",
        "Find data split, cross-validation, random seed, train/validation/test protocol, preprocessing, and scaling details for the final boiling point regression model.",
    ],
    "metrics": [
        "For the same single best-performing or final boiling point regression model, what regression metrics and reported values are provided? Ignore classification metrics or other property tasks.",
        "Find reported boiling point regression performance for the final model, including MAE, RMSE, MSE, R2, test error, validation error, and comparison table values.",
    ],
}

TOPIC_KEYWORDS = {
    "feature": [
        "descriptor",
        "descriptors",
        "fingerprint",
        "fingerprints",
        "morgan",
        "ecfp",
        "maccs",
        "atom pair",
        "topological torsion",
        "rdkit",
        "mordred",
        "padel",
        "dragon",
        "atom count",
        "bond count",
        "ring count",
        "functional group",
        "element count",
        "adjacency matrix",
        "bond matrix",
        "graph",
        "characteristic",
        "characteristics",
        "feature selection",
        "data transformation",
        "derived feature",
        "derived features",
        "cmpdname",
        "mw",
        "mf",
        "polararea",
        "heavycnt",
        "hbondacc",
        "iso smiles",
        "isosmiles",
        "c/n/o number",
        "carbon number",
        "nitrogen number",
        "oxygen number",
        "side chain number",
        "hydrocarbon",
        "alcohol",
        "amine",
        "table 1",
        "table 1(a)",
        "table 1(b)",
        "blue box",
        "yellow box",
        "green box",
        "red box",
        "data cleaning",
        "data transformation",
        "data discretization",
        "data integration",
    ],
    "preprocessing": [
        "preprocessing",
        "data cleaning",
        "data transformation",
        "data discretization",
        "data integration",
        "invalid smiles",
        "duplicate",
        "missing value",
        "imputation",
        "scaling",
        "normalization",
        "standardization",
        "characteristics",
        "labeling compounds",
    ],
}


class RagState(TypedDict, total=False):
    markdown: str
    source_id: str
    top_k: int
    vector_info: dict[str, Any]
    retrieved: list[dict[str, Any]]
    retrieved_by_topic: dict[str, list[dict[str, Any]]]
    selection_result: dict[str, Any]
    selected_model_name: str
    selected_model_terms: list[str]
    filtered_by_topic: dict[str, list[dict[str, Any]]]
    feature_result: dict[str, Any]
    feature_evidence_rows: list[dict[str, Any]]
    method_result: dict[str, Any]
    paper_method_spec: dict[str, Any]
    spec_build_trace: dict[str, Any]
    error: str


def _matched_keywords(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in keywords if keyword in lowered]


def _list_pattern_bonus(text: str) -> float:
    lowered = text.lower()
    bonus = 0.0
    patterns = [
        r"(?:characteristics?|features?|columns?)\s*\(([^)]{1,320})\)",
        r"(?:added|adding).{0,80}\d+\s+characteristics?\s*\(([^)]{1,320})\)",
        r"(?:labeling compounds as|labeled as|classified as)\s+[a-z ,/()-]{3,240}",
    ]
    if any(re.search(pattern, lowered, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns):
        bonus += 0.03
    if any(token in lowered for token in ["table 1", "table 1(a)", "table 1(b)", "blue box", "yellow box", "green box", "red box"]):
        bonus += 0.015
    return bonus


def _dedupe_rows_by_chunk(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_chunk: dict[int, dict[str, Any]] = {}
    for row in rows:
        chunk_id = row.get("chunk_id")
        if not isinstance(chunk_id, int):
            continue
        current = best_by_chunk.get(chunk_id)
        current_score = float(current.get("score", 0.0)) if current else float("-inf")
        row_score = float(row.get("score", 0.0))
        if current is None or row_score > current_score:
            best_by_chunk[chunk_id] = row
    return [best_by_chunk[chunk_id] for chunk_id in sorted(best_by_chunk)]


def _collect_feature_evidence_rows(state: RagState) -> list[dict[str, Any]]:
    filtered = state.get("filtered_by_topic", {})
    seed_rows = [
        *(filtered.get("feature", []) or []),
        *(filtered.get("preprocessing", []) or []),
    ]
    all_rows = _dedupe_rows_by_chunk(state.get("retrieved", []) or [])
    rows_by_chunk = {
        row["chunk_id"]: row for row in all_rows if isinstance(row.get("chunk_id"), int)
    }
    expanded = list(seed_rows)
    for row in seed_rows:
        chunk_id = row.get("chunk_id")
        if not isinstance(chunk_id, int):
            continue
        for offset in (-1, 1):
            adjacent = rows_by_chunk.get(chunk_id + offset)
            if adjacent:
                expanded.append(adjacent)
    return _dedupe_rows_by_chunk(expanded)


def _merge_topic_rows(topic: str, query_rows: list[tuple[str, list[dict[str, Any]]]], top_k: int) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    keywords = TOPIC_KEYWORDS.get(topic, [])

    for query_index, (query, rows) in enumerate(query_rows):
        for row in rows:
            chunk_id = int(row.get("chunk_id", -1))
            if chunk_id < 0:
                continue

            matched_keywords = _matched_keywords(row.get("text", ""), keywords)
            keyword_bonus = min(len(matched_keywords), 4) * 0.015
            pattern_bonus = _list_pattern_bonus(row.get("text", "")) if topic in {"feature", "preprocessing"} else 0.0
            query_bonus = max(0.0, 0.006 - (query_index * 0.001))
            adjusted_score = float(row.get("score", 0.0)) + keyword_bonus + pattern_bonus + query_bonus

            current = merged.get(chunk_id)
            if current is None or adjusted_score > float(current.get("adjusted_score", -1.0)):
                merged[chunk_id] = {
                    **row,
                    "query": query,
                    "matched_queries": [query],
                    "matched_keywords": matched_keywords,
                    "adjusted_score": adjusted_score,
                }
                continue

            current_queries = current.get("matched_queries", [])
            if query not in current_queries:
                current["matched_queries"] = [*current_queries, query]
            combined_keywords = set(current.get("matched_keywords", []))
            combined_keywords.update(matched_keywords)
            current["matched_keywords"] = sorted(combined_keywords)

    ranked_rows = sorted(
        merged.values(),
        key=lambda row: (float(row.get("adjusted_score", 0.0)), float(row.get("score", 0.0))),
        reverse=True,
    )[:top_k]

    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank
        row.pop("adjusted_score", None)
    return ranked_rows


def _build_vector_db(state: RagState) -> RagState:
    if state.get("error"):
        return state

    vdb = LocalVectorDB(embedding_model="text-embedding-3-large")
    info = vdb.build_from_markdown(
        markdown=state.get("markdown", ""),
        source_id=state.get("source_id"),
    )
    return {"vector_info": info, "source_id": info["source_id"]}


def _retrieve_evidence(state: RagState) -> RagState:
    if state.get("error"):
        return state

    vdb = LocalVectorDB(embedding_model="text-embedding-3-large")
    top_k = int(state.get("top_k", 5))

    retrieved_by_topic: dict[str, list[dict[str, Any]]] = {}
    flattened: list[dict[str, Any]] = []

    for topic, queries in TOPIC_QUERY_VARIANTS.items():
        per_query_top_k = max(top_k + 3, 8) if topic in {"feature", "preprocessing"} else top_k
        query_rows = [(query, vdb.retrieve(source_id=state["source_id"], query=query, top_k=per_query_top_k)) for query in queries]
        topic_rows = _merge_topic_rows(topic=topic, query_rows=query_rows, top_k=top_k)
        enriched_rows: list[dict[str, Any]] = []
        for row in topic_rows:
            enriched = dict(row)
            enriched["topic"] = topic
            enriched_rows.append(enriched)
            flattened.append(enriched)
        retrieved_by_topic[topic] = enriched_rows

    return {"retrieved_by_topic": retrieved_by_topic, "retrieved": flattened}


def _select_final_model(state: RagState) -> RagState:
    if state.get("error"):
        return state

    agent = ModelSelectionAgent(model_name="gpt-5.2")
    result = agent.invoke(retrieved_by_topic=state.get("retrieved_by_topic", {}))
    if result.get("error"):
        return {"error": result["error"]}

    selection_result = result.get("selection_result", {})
    selected_model_name = str(selection_result.get("model", {}).get("name", "Not found"))
    selected_model_terms = build_selected_model_terms(
        selected_model_name,
        selection_result.get("selected_model_terms", []),
    )
    filtered_by_topic = {
        topic: filter_rows_for_selected_model(rows, selected_model_name, selected_model_terms, topic=topic)
        for topic, rows in state.get("retrieved_by_topic", {}).items()
    }

    return {
        "selection_result": selection_result,
        "selected_model_name": selected_model_name,
        "selected_model_terms": selected_model_terms,
        "filtered_by_topic": filtered_by_topic,
    }


def _extract_feature_evidence(state: RagState) -> RagState:
    if state.get("error"):
        return state

    feature_rows = _collect_feature_evidence_rows(state)
    agent = FeatureEvidenceAgent(model_name="gpt-5.2")
    result = agent.invoke(
        selected_model_name=state.get("selected_model_name", "Not found"),
        feature_rows=feature_rows,
    )
    if result.get("error"):
        return {"error": result["error"]}
    return {
        "feature_result": result.get("feature_result", {}),
        "feature_evidence_rows": feature_rows,
    }


def _extract_method_sections(state: RagState) -> RagState:
    if state.get("error"):
        return state

    agent = MethodSectionAgent(model_name="gpt-5.2")
    method_topics = {
        topic: rows
        for topic, rows in state.get("filtered_by_topic", {}).items()
        if topic in {"preprocessing", "hyperparameter", "training", "metrics"}
    }
    result = agent.invoke(
        selected_model_name=state.get("selected_model_name", "Not found"),
        retrieved_by_topic=method_topics,
    )
    if result.get("error"):
        return {"error": result["error"]}
    return {"method_result": result.get("method_result", {})}


def _normalize_spec(state: RagState) -> RagState:
    if state.get("error"):
        return state

    agent = RetrieverTableAgent(model_name="gpt-5.2")
    result = agent.invoke(
        selection_result=state.get("selection_result", {}),
        feature_result=state.get("feature_result", {}),
        method_result=state.get("method_result", {}),
        filtered_by_topic=state.get("filtered_by_topic", {}),
        feature_rows=state.get("feature_evidence_rows", []),
    )
    if result.get("error"):
        return {"error": result["error"]}

    return {"paper_method_spec": result.get("paper_method_spec", {})}


def _validate_spec(state: RagState) -> RagState:
    if state.get("error"):
        return state

    validation = validate_paper_method_spec_contract(
        spec=state.get("paper_method_spec", {}),
        selected_model_name=state.get("selected_model_name", "Not found"),
        selected_model_terms=state.get("selected_model_terms", []),
        filtered_by_topic=state.get("filtered_by_topic", {}),
    )
    spec_build_trace = build_spec_build_trace(
        selected_model_name=state.get("selected_model_name", "Not found"),
        selected_model_terms=state.get("selected_model_terms", []),
        retrieved_by_topic=state.get("retrieved_by_topic", {}),
        filtered_by_topic=state.get("filtered_by_topic", {}),
        validation=validation,
    )
    if not validation.get("is_valid", False):
        return {
            "error": "; ".join(validation.get("issues", [])) or "Spec validation failed.",
            "spec_build_trace": spec_build_trace,
        }
    return {"spec_build_trace": spec_build_trace}


def build_graph():
    graph = StateGraph(RagState)
    graph.add_node("build_vector_db", _build_vector_db)
    graph.add_node("retrieve_evidence", _retrieve_evidence)
    graph.add_node("select_final_model", _select_final_model)
    graph.add_node("extract_feature_evidence", _extract_feature_evidence)
    graph.add_node("extract_method_sections", _extract_method_sections)
    graph.add_node("normalize_spec", _normalize_spec)
    graph.add_node("validate_spec", _validate_spec)

    graph.set_entry_point("build_vector_db")
    graph.add_edge("build_vector_db", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "select_final_model")
    graph.add_edge("select_final_model", "extract_feature_evidence")
    graph.add_edge("extract_feature_evidence", "extract_method_sections")
    graph.add_edge("extract_method_sections", "normalize_spec")
    graph.add_edge("normalize_spec", "validate_spec")
    graph.add_edge("validate_spec", END)

    return graph.compile()


RAG_GRAPH = build_graph()


def run_rag_pipeline(markdown: str, source_id: str | None = None, top_k: int = 5) -> RagState:
    return RAG_GRAPH.invoke(
        {
            "markdown": markdown,
            "source_id": source_id,
            "top_k": top_k,
        }
    )
