from __future__ import annotations

import re
from typing import Any

from .chemistry_features import augment_feature_payload, merge_unique, normalize_string_list
from .paper_method_spec import has_meaningful_paper_method_spec, normalize_paper_method_spec

MODEL_ALIAS_TERMS = {
    "RandomForestRegressor": ["random forest", "randomforest", "randomforestregressor", "rf"],
    "Ridge": ["ridge", "ridgeregression"],
    "Lasso": ["lasso"],
    "ElasticNet": ["elastic net", "elasticnet"],
    "SVR": ["svr", "support vector regression", "support vector regressor"],
    "KNeighborsRegressor": ["knn", "k-nearest neighbors", "kneighborsregressor", "k neighbors regressor"],
    "MLPRegressor": ["mlp", "mlpregressor", "multi layer perceptron", "multilayer perceptron", "neural network"],
}

DATASET_FEATURE_CONTEXT_TERMS = [
    "data cleaning",
    "data transformation",
    "data discretization",
    "data integration",
    "characteristic",
    "characteristics",
    "selected characteristics",
    "necessary characteristics",
    "added",
    "labeling compounds",
    "retained columns",
    "input variables",
    "feature set",
    "table 1",
    "table 1(a)",
    "table 1(b)",
    "blue box",
    "yellow box",
    "green box",
    "red box",
    "cmpdname",
    "mw",
    "mf",
    "polararea",
    "heavycnt",
    "hbondacc",
    "iso smiles",
    "iso-smiles",
    "c number",
    "n number",
    "o number",
    "c/n/o number",
    "side chain number",
    "hydrocarbon",
    "alcohol",
    "amine",
]


def chunk_ref(row: dict[str, Any]) -> str:
    chunk_id = row.get("chunk_id")
    if isinstance(chunk_id, int):
        return f"chunk-{chunk_id}"
    text = str(chunk_id).strip()
    return text if text.startswith("chunk-") else f"chunk-{text}" if text else "chunk-unknown"


def chunk_refs(rows: list[dict[str, Any]], limit: int = 3) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for row in rows:
        ref = chunk_ref(row)
        if ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
        if len(refs) >= limit:
            break
    return refs


def summarize_evidence_snippet(rows: list[dict[str, Any]], limit: int = 320) -> str:
    for row in rows:
        text = str(row.get("text", "")).strip()
        if text:
            return text[:limit]
    return "Not found"


def build_selected_model_terms(model_name: str, raw_terms: list[str] | None = None) -> list[str]:
    terms = [str(term).strip().lower() for term in normalize_string_list(raw_terms or [])]
    for alias in MODEL_ALIAS_TERMS.get(model_name, []):
        terms.append(alias.lower())
    if model_name and model_name != "Not found":
        terms.append(model_name.lower())
    return merge_unique(terms)


def row_mentions_selected_model(row: dict[str, Any], selected_model_terms: list[str]) -> bool:
    if not selected_model_terms:
        return False
    lowered = str(row.get("text", "")).lower()
    for term in selected_model_terms:
        if not term:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", lowered):
            return True
    return False


def row_mentions_other_model(row: dict[str, Any], selected_model_name: str) -> bool:
    lowered = str(row.get("text", "")).lower()
    for canonical_name, aliases in MODEL_ALIAS_TERMS.items():
        if canonical_name == selected_model_name:
            continue
        for alias in aliases:
            if re.search(rf"(?<![a-z0-9]){re.escape(alias.lower())}(?![a-z0-9])", lowered):
                return True
    return False


def row_has_dataset_feature_context(row: dict[str, Any]) -> bool:
    lowered = str(row.get("text", "")).lower()
    if any(term in lowered for term in DATASET_FEATURE_CONTEXT_TERMS):
        return True
    list_patterns = [
        r"(?:characteristics?|features?|columns?)\s*\(([^)]{1,280})\)",
        r"(?:added|adding).{0,80}\d+\s+characteristics?\s*\(([^)]{1,280})\)",
        r"(?:labeling compounds as|labeled as|classified as)\s+[a-z ,/()-]{3,240}",
    ]
    return any(re.search(pattern, lowered, flags=re.IGNORECASE | re.DOTALL) for pattern in list_patterns)


def filter_rows_for_selected_model(
    rows: list[dict[str, Any]],
    model_name: str,
    selected_model_terms: list[str],
    *,
    topic: str = "",
) -> list[dict[str, Any]]:
    if not rows or not model_name or model_name == "Not found":
        return list(rows)
    matched = [row for row in rows if row_mentions_selected_model(row, selected_model_terms)]
    if not matched:
        return list(rows)
    if topic not in {"feature", "preprocessing"}:
        return matched

    preserved = list(matched)
    seen_chunk_ids = {row.get("chunk_id") for row in preserved}
    for row in rows:
        chunk_id = row.get("chunk_id")
        if chunk_id in seen_chunk_ids:
            continue
        if row_mentions_other_model(row, model_name):
            continue
        if row_has_dataset_feature_context(row):
            preserved.append(row)
            seen_chunk_ids.add(chunk_id)
    return preserved or matched


def ensure_section_evidence(section: dict[str, Any] | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = dict(section or {})
    if not payload.get("evidence_chunks"):
        payload["evidence_chunks"] = chunk_refs(rows)
    if payload.get("evidence_snippet") in {None, "", "Not found"}:
        payload["evidence_snippet"] = summarize_evidence_snippet(rows)
    if payload.get("summary") in {None, ""}:
        payload["summary"] = "Not found"
    if payload.get("key_values") in {None, ""}:
        payload["key_values"] = "Not found"
    return payload


def normalize_feature_section(raw_feature_section: dict[str, Any] | None, feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = ensure_section_evidence(raw_feature_section, feature_rows)
    text_sources = [
        *normalize_string_list(payload.get("raw_feature_mentions")),
        *normalize_string_list(payload.get("raw_tool_mentions")),
        *[str(row.get("text", "")) for row in feature_rows],
    ]
    normalized = augment_feature_payload(payload, text_sources=text_sources)
    normalized.pop("raw_feature_mentions", None)
    normalized.pop("raw_tool_mentions", None)
    return normalized


def assemble_paper_method_spec(
    selection_result: dict[str, Any] | None,
    feature_result: dict[str, Any] | None,
    method_result: dict[str, Any] | None,
    filtered_by_topic: dict[str, list[dict[str, Any]]] | None,
    feature_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    filtered_by_topic = filtered_by_topic or {}
    selection_result = selection_result or {}
    feature_result = feature_result or {}
    method_result = method_result or {}

    selection_basis = ensure_section_evidence(
        selection_result.get("selection_basis"),
        [*(filtered_by_topic.get("model", []) or []), *(filtered_by_topic.get("metrics", []) or [])],
    )
    model = ensure_section_evidence(selection_result.get("model"), filtered_by_topic.get("model", []) or [])
    feature = normalize_feature_section(feature_result.get("feature"), feature_rows or filtered_by_topic.get("feature", []) or [])
    preprocessing_rows = filtered_by_topic.get("preprocessing", []) or filtered_by_topic.get("training", []) or []
    preprocessing = ensure_section_evidence(method_result.get("preprocessing"), preprocessing_rows)
    hyperparameters = ensure_section_evidence(method_result.get("hyperparameters"), filtered_by_topic.get("hyperparameter", []) or [])
    training = ensure_section_evidence(method_result.get("training"), filtered_by_topic.get("training", []) or [])
    metrics = ensure_section_evidence(method_result.get("metrics"), filtered_by_topic.get("metrics", []) or [])

    raw_spec = {
        "selection_basis": selection_basis,
        "preprocessing": preprocessing,
        "feature": feature,
        "model": model,
        "hyperparameters": hyperparameters,
        "training": training,
        "metrics": metrics,
    }
    return normalize_paper_method_spec(raw_spec)


def detect_conflicting_model_mentions(text: str, selected_model_name: str) -> list[str]:
    lowered = str(text).lower()
    conflicts: list[str] = []
    for canonical_name, aliases in MODEL_ALIAS_TERMS.items():
        if canonical_name == selected_model_name:
            continue
        if any(alias in lowered for alias in aliases):
            conflicts.append(canonical_name)
    return merge_unique(conflicts)


def validate_paper_method_spec_contract(
    spec: dict[str, Any] | None,
    selected_model_name: str,
    selected_model_terms: list[str],
    filtered_by_topic: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    normalized = normalize_paper_method_spec(spec or {})
    filtered_by_topic = filtered_by_topic or {}

    issues: list[str] = []
    warnings: list[str] = []

    if not has_meaningful_paper_method_spec(normalized):
        issues.append("No meaningful structured spec could be assembled from retrieved evidence.")

    normalized_model_name = str(normalized.get("model", {}).get("name", "Not found"))
    if selected_model_name and selected_model_name != "Not found" and normalized_model_name != selected_model_name:
        issues.append(f"Selected model mismatch: expected {selected_model_name}, got {normalized_model_name}.")

    feature = normalized.get("feature", {})
    if feature.get("unresolved_feature_terms"):
        warnings.append(
            "Unsupported external chemistry feature tools were preserved as unresolved terms: "
            + ", ".join(feature.get("unresolved_feature_terms", []))
        )

    section_to_topic = {
        "preprocessing": "preprocessing",
        "feature": "feature",
        "hyperparameters": "hyperparameter",
        "training": "training",
        "metrics": "metrics",
    }
    for section_name, topic_name in section_to_topic.items():
        section = normalized.get(section_name, {})
        inspection_text = " ".join(
            [
                str(section.get("summary", "")),
                str(section.get("key_values", "")),
                str(section.get("evidence_snippet", "")),
            ]
        )
        conflicts = detect_conflicting_model_mentions(inspection_text, selected_model_name)
        if conflicts:
            warnings.append(f"{section_name} evidence may mention other models: {', '.join(conflicts)}.")

        filtered_rows = filtered_by_topic.get(topic_name, []) or []
        filtered_chunk_ids = set(chunk_refs(filtered_rows, limit=max(len(filtered_rows), 1)))
        evidence_chunks = set(str(chunk).strip() for chunk in section.get("evidence_chunks", []) if str(chunk).strip())
        if filtered_chunk_ids and evidence_chunks and not evidence_chunks.issubset(filtered_chunk_ids):
            warnings.append(f"{section_name} evidence chunks include rows outside the selected-model filtered set.")

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "selected_model_name": selected_model_name,
        "selected_model_terms": selected_model_terms,
        "filtered_chunk_ids_by_topic": {
            topic: chunk_refs(rows, limit=max(len(rows), 1)) for topic, rows in filtered_by_topic.items()
        },
    }


def build_spec_build_trace(
    selected_model_name: str,
    selected_model_terms: list[str],
    retrieved_by_topic: dict[str, list[dict[str, Any]]],
    filtered_by_topic: dict[str, list[dict[str, Any]]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "selected_model": {
            "name": selected_model_name,
            "terms": selected_model_terms,
        },
        "retrieved_chunk_ids_by_topic": {
            topic: chunk_refs(rows, limit=max(len(rows), 1)) for topic, rows in retrieved_by_topic.items()
        },
        "filtered_chunk_ids_by_topic": {
            topic: chunk_refs(rows, limit=max(len(rows), 1)) for topic, rows in filtered_by_topic.items()
        },
        "validation": validation,
    }
