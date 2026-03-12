from __future__ import annotations

import ast
import json
import re
from typing import Any

MODEL_ALIASES = {
    "random forest": "RandomForestRegressor",
    "randomforest": "RandomForestRegressor",
    "randomforestregressor": "RandomForestRegressor",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elastic net": "ElasticNet",
    "elasticnet": "ElasticNet",
    "svr": "SVR",
    "support vector regression": "SVR",
    "support vector regressor": "SVR",
    "supportvectorregression": "SVR",
    "supportvectorregressor": "SVR",
    "k-nearest neighbors": "KNeighborsRegressor",
    "kneighbors": "KNeighborsRegressor",
    "kneighborsregressor": "KNeighborsRegressor",
    "mlp": "MLPRegressor",
    "mlpregressor": "MLPRegressor",
    "multi-layer perceptron": "MLPRegressor",
    "multilayerperceptron": "MLPRegressor",
}
CANONICAL_TO_INTERNAL = {
    "RandomForestRegressor": "random_forest",
    "Ridge": "ridge",
    "Lasso": "lasso",
    "ElasticNet": "elasticnet",
    "SVR": "svr",
    "KNeighborsRegressor": "knn",
    "MLPRegressor": "mlp",
}


def _text(value: Any, default: str = "Not found") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _first_available(*values: Any, default: str = "Not found") -> str:
    for value in values:
        text = _text(value, default="")
        if text:
            return text
    return default


def _number(value: Any) -> int | float | None:
    if value is None or value == "" or value == "Not found":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if isinstance(value, float) and value.is_integer() else value
    text = str(value).strip().rstrip("%")
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "" or value == "Not found":
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def extract_json_object(text: str) -> dict[str, Any]:
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])

    decoder = json.JSONDecoder()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        try:
            parsed = ast.literal_eval(candidate)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("JSON 객체를 추출하지 못했습니다.")


def canonical_model_name(value: Any) -> str:
    text = _text(value)
    if text == "Not found":
        return text
    lowered = text.lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    for alias, canonical_name in MODEL_ALIASES.items():
        alias_compact = re.sub(r"[^a-z0-9]+", "", alias)
        if alias in lowered or alias_compact == compact:
            return canonical_name
    return text


def internal_model_name(value: Any) -> str:
    canonical_name = canonical_model_name(value)
    return CANONICAL_TO_INTERNAL.get(canonical_name, "")


def normalize_feature_method(value: Any) -> str:
    text = _text(value, default="").lower()
    if not text:
        return "Not found"
    has_morgan = "morgan" in text or "fingerprint" in text
    has_descriptor = "descriptor" in text or "rdkit" in text
    if has_morgan and has_descriptor:
        return "combined"
    if has_morgan:
        return "morgan"
    if has_descriptor:
        return "descriptor"
    if text in {"descriptor", "morgan", "combined"}:
        return text
    return "Not found"


def normalize_chunk_refs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[,;\n]+", str(value))

    refs: list[str] = []
    for raw_item in raw_items:
        item = str(raw_item).strip()
        if not item:
            continue
        lowered = item.lower()
        if lowered.startswith("chunk-"):
            refs.append(lowered)
            continue
        if lowered.startswith("chunk"):
            suffix = lowered.replace("chunk", "", 1).strip(" -:#")
            refs.append(f"chunk-{suffix}" if suffix else "chunk")
            continue
        if item.isdigit():
            refs.append(f"chunk-{item}")
            continue
        refs.append(item)
    return refs


def normalize_metric_dict(raw_metrics: Any) -> dict[str, float]:
    if not isinstance(raw_metrics, dict):
        return {}
    normalized: dict[str, float] = {}
    for metric_name in ["MAE", "RMSE", "MSE", "R2"]:
        value = raw_metrics.get(metric_name)
        number = _number(value)
        if number is not None:
            normalized[metric_name] = float(number)
    return normalized


def normalize_hyperparameter_values(raw_values: Any) -> dict[str, Any]:
    if isinstance(raw_values, dict):
        return {str(key): value for key, value in raw_values.items() if str(key).strip()}
    if isinstance(raw_values, list):
        return {f"param_{index}": value for index, value in enumerate(raw_values)}
    return {}


def _normalize_section_base(section: Any) -> dict[str, Any]:
    payload = section if isinstance(section, dict) else {}
    return {
        "summary": _text(payload.get("summary")),
        "key_values": _text(payload.get("key_values")),
        "evidence_chunks": normalize_chunk_refs(payload.get("evidence_chunks", [])),
        "evidence_snippet": _text(payload.get("evidence_snippet")),
    }


def normalize_paper_method_spec(raw_spec: dict[str, Any] | None) -> dict[str, Any]:
    spec = raw_spec if isinstance(raw_spec, dict) else {}

    selection_basis = _normalize_section_base(spec.get("selection_basis", {}))

    preprocessing_payload = spec.get("preprocessing", {}) if isinstance(spec.get("preprocessing"), dict) else {}
    preprocessing = {
        **_normalize_section_base(preprocessing_payload),
        "invalid_smiles": _text(preprocessing_payload.get("invalid_smiles")) if preprocessing_payload.get("invalid_smiles") not in {None, ""} else "Not found",
        "missing_target": _text(preprocessing_payload.get("missing_target")) if preprocessing_payload.get("missing_target") not in {None, ""} else "Not found",
        "missing_features": _text(preprocessing_payload.get("missing_features")) if preprocessing_payload.get("missing_features") not in {None, ""} else "Not found",
        "duplicates": _text(preprocessing_payload.get("duplicates")) if preprocessing_payload.get("duplicates") not in {None, ""} else "Not found",
        "scaling": _bool_or_none(preprocessing_payload.get("scaling")),
    }

    feature_payload = spec.get("feature", {}) if isinstance(spec.get("feature"), dict) else {}
    feature = {
        **_normalize_section_base(feature_payload),
        "method": normalize_feature_method(feature_payload.get("method")),
        "radius": _number(feature_payload.get("radius")),
        "n_bits": _number(feature_payload.get("n_bits")),
        "use_rdkit_descriptors": _bool_or_none(feature_payload.get("use_rdkit_descriptors")),
    }

    model_payload = spec.get("model", {}) if isinstance(spec.get("model"), dict) else {}
    model = {
        **_normalize_section_base(model_payload),
        "name": canonical_model_name(model_payload.get("name")),
    }

    hyperparameter_payload = spec.get("hyperparameters", {}) if isinstance(spec.get("hyperparameters"), dict) else {}
    hyperparameters = {
        **_normalize_section_base(hyperparameter_payload),
        "values": normalize_hyperparameter_values(hyperparameter_payload.get("values")),
    }

    training_payload = spec.get("training", {}) if isinstance(spec.get("training"), dict) else {}
    training = {
        **_normalize_section_base(training_payload),
        "split_strategy": _text(training_payload.get("split_strategy")) if training_payload.get("split_strategy") not in {None, ""} else "Not found",
        "test_size": _number(training_payload.get("test_size")),
        "random_state": _number(training_payload.get("random_state")),
    }

    metric_payload = spec.get("metrics", {}) if isinstance(spec.get("metrics"), dict) else {}
    metrics = {
        **_normalize_section_base(metric_payload),
        "reported": normalize_metric_dict(metric_payload.get("reported")),
    }

    return {
        "selection_basis": selection_basis,
        "preprocessing": preprocessing,
        "feature": feature,
        "model": model,
        "hyperparameters": hyperparameters,
        "training": training,
        "metrics": metrics,
    }


def has_meaningful_paper_method_spec(spec: dict[str, Any] | None) -> bool:
    normalized = normalize_paper_method_spec(spec or {})
    if normalized["model"].get("name") not in {None, "", "Not found"}:
        return True
    if normalized["feature"].get("method") not in {None, "", "Not found"}:
        return True
    if normalized["hyperparameters"].get("values"):
        return True
    if normalized["training"].get("split_strategy") not in {None, "", "Not found"}:
        return True
    if normalized["metrics"].get("reported"):
        return True
    return False


def _preprocessing_label(section: dict[str, Any]) -> str:
    explicit_values = []
    for key in ["invalid_smiles", "missing_target", "missing_features", "duplicates"]:
        value = section.get(key)
        if value not in {None, "", "Not found"}:
            explicit_values.append(f"{key}={value}")
    if section.get("scaling") is not None:
        explicit_values.append(f"scaling={section['scaling']}")
    return _first_available(section.get("key_values"), ", ".join(explicit_values), section.get("summary"))


def format_feature_label(section: dict[str, Any]) -> str:
    method = normalize_feature_method(section.get("method"))
    radius = section.get("radius")
    n_bits = section.get("n_bits")
    extras = []
    if radius is not None:
        extras.append(f"radius={radius}")
    if n_bits is not None:
        extras.append(f"n_bits={n_bits}")
    suffix = f" ({', '.join(extras)})" if extras else ""
    if method == "descriptor":
        return "RDKit descriptors"
    if method == "morgan":
        return f"Morgan fingerprint{suffix}"
    if method == "combined":
        return f"Morgan fingerprint + RDKit descriptors{suffix}"
    return _first_available(section.get("key_values"), section.get("summary"))


def _hyperparameter_label(section: dict[str, Any]) -> str:
    values = section.get("values", {})
    if values:
        return ", ".join([f"{key}={value}" for key, value in values.items()])
    return _first_available(section.get("key_values"), section.get("summary"))


def _training_label(section: dict[str, Any]) -> str:
    parts = []
    if section.get("split_strategy") not in {None, "", "Not found"}:
        parts.append(f"split={section['split_strategy']}")
    if section.get("test_size") is not None:
        parts.append(f"test_size={section['test_size']}")
    if section.get("random_state") is not None:
        parts.append(f"random_state={section['random_state']}")
    return _first_available(section.get("key_values"), ", ".join(parts), section.get("summary"))


def _metric_label(section: dict[str, Any]) -> str:
    reported = section.get("reported", {})
    if reported:
        return ", ".join([f"{key}={value}" for key, value in reported.items()])
    return _first_available(section.get("key_values"), section.get("summary"))


def build_paper_method_summary_markdown(spec: dict[str, Any]) -> str:
    normalized = normalize_paper_method_spec(spec)
    rows = [
        ["Topic", "What Paper Did", "Key Values or Settings", "Evidence Chunks", "Evidence Snippets"],
        [
            "model",
            normalized["model"]["summary"],
            _first_available(normalized["model"].get("name"), normalized["model"].get("key_values"), normalized["model"].get("summary")),
            ", ".join(normalized["model"].get("evidence_chunks", [])) or "Not found",
            normalized["model"].get("evidence_snippet", "Not found"),
        ],
        [
            "feature",
            normalized["feature"]["summary"],
            format_feature_label(normalized["feature"]),
            ", ".join(normalized["feature"].get("evidence_chunks", [])) or "Not found",
            normalized["feature"].get("evidence_snippet", "Not found"),
        ],
        [
            "hyperparameter",
            normalized["hyperparameters"]["summary"],
            _hyperparameter_label(normalized["hyperparameters"]),
            ", ".join(normalized["hyperparameters"].get("evidence_chunks", [])) or "Not found",
            normalized["hyperparameters"].get("evidence_snippet", "Not found"),
        ],
        [
            "training",
            normalized["training"]["summary"],
            _training_label(normalized["training"]),
            ", ".join(normalized["training"].get("evidence_chunks", [])) or "Not found",
            normalized["training"].get("evidence_snippet", "Not found"),
        ],
        [
            "metrics",
            normalized["metrics"]["summary"],
            _metric_label(normalized["metrics"]),
            ", ".join(normalized["metrics"].get("evidence_chunks", [])) or "Not found",
            normalized["metrics"].get("evidence_snippet", "Not found"),
        ],
    ]
    safe_rows = [[str(cell).replace("|", "/") for cell in row] for row in rows]
    header = "| " + " | ".join(safe_rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(safe_rows[0])) + " |"
    body = ["| " + " | ".join(row) + " |" for row in safe_rows[1:]]
    return "\n".join([header, separator, *body])


def paper_method_spec_to_comparison_spec(spec: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_paper_method_spec(spec)
    return {
        "selection_basis": _first_available(
            normalized["selection_basis"].get("summary"),
            normalized["selection_basis"].get("key_values"),
        ),
        "preprocessing": _preprocessing_label(normalized["preprocessing"]),
        "feature": format_feature_label(normalized["feature"]),
        "model": _first_available(
            normalized["model"].get("name"),
            normalized["model"].get("key_values"),
            normalized["model"].get("summary"),
        ),
        "hyperparameters": _hyperparameter_label(normalized["hyperparameters"]),
        "training": _training_label(normalized["training"]),
        "metrics": normalized["metrics"].get("reported", {}),
    }

