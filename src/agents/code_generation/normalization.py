from __future__ import annotations

from typing import Any

from src.utils import normalize_paper_method_spec

from .defaults import (
    default_hyperparameters,
    fill_dataset_details,
    fill_feature_details,
    fill_model_details,
    fill_preprocessing_details,
    fill_training_details,
)
from .raw_parsing import normalize_dict, parse_raw_string


def parse_paper_info(raw_paper_info: dict[str, Any] | str) -> tuple[dict[str, Any], list[str]]:
    assumptions: list[str] = []
    if isinstance(raw_paper_info, str):
        parsed = parse_raw_string(raw_paper_info)
    elif isinstance(raw_paper_info, dict):
        parsed = normalize_dict(raw_paper_info)
    else:
        parsed = {}
        assumptions.append("입력 형식이 불명확하여 기본 회귀 파이프라인을 사용했습니다.")

    normalized = {
        "dataset": parsed.get("dataset", {}),
        "target_property": parsed.get("target_property", "boiling_point"),
        "task_type": "regression",
        "preprocessing": parsed.get("preprocessing", {}),
        "feature": parsed.get("feature", {}),
        "model": parsed.get("model", {}),
        "hyperparameters": parsed.get("hyperparameters", {}),
        "training": parsed.get("training", {}),
        "metrics": parsed.get("metrics", ["MAE", "RMSE", "MSE", "R2"]),
        "paper_method_spec": normalize_paper_method_spec(parsed.get("paper_method_spec", {})),
    }
    return normalized, assumptions


def fill_missing_details(normalized_spec: dict[str, Any], assumptions: list[str]) -> tuple[dict[str, Any], list[str]]:
    updated_spec = dict(normalized_spec or {})
    updated_assumptions = list(assumptions or [])
    paper_method_spec = normalize_paper_method_spec(updated_spec.get("paper_method_spec", {}))
    updated_spec["paper_method_spec"] = paper_method_spec

    updated_spec["dataset"] = fill_dataset_details(updated_spec.get("dataset", {}), updated_assumptions)
    updated_spec["preprocessing"] = fill_preprocessing_details(_merge_dict(updated_spec.get("preprocessing", {}), paper_method_spec.get("preprocessing", {})))
    updated_spec["feature"] = fill_feature_details(_merge_feature(updated_spec.get("feature", {}), paper_method_spec.get("feature", {})), updated_assumptions)
    updated_spec["model"] = fill_model_details(_merge_model(updated_spec.get("model", {}), paper_method_spec.get("model", {})), updated_assumptions)
    updated_spec["hyperparameters"] = default_hyperparameters(
        updated_spec["model"]["name"],
        _merge_hyperparameters(updated_spec.get("hyperparameters", {}), paper_method_spec.get("hyperparameters", {})),
        updated_assumptions,
    )
    updated_spec["training"] = fill_training_details(
        _merge_training(updated_spec.get("training", {}), paper_method_spec.get("training", {})),
        updated_spec["model"]["name"],
        updated_assumptions,
    )
    updated_spec["metrics"] = ["MAE", "RMSE", "MSE", "R2"]
    return updated_spec, updated_assumptions


def build_code_spec(normalized_spec: dict[str, Any]) -> dict[str, Any]:
    dataset_spec = dict(normalized_spec.get("dataset", {}))
    feature_spec = dict(normalized_spec.get("feature", {}))
    training_spec = dict(normalized_spec.get("training", {}))

    return {
        "task_type": "regression",
        "target_property": normalized_spec.get("target_property", "boiling_point"),
        "paper_method_spec": normalized_spec.get("paper_method_spec", {}),
        "dataset": dataset_spec,
        "feature_pipeline": {
            "feature_source": "smiles_only",
            "input_column": dataset_spec.get("smiles_column", "smiles"),
            "method": feature_spec.get("method", "descriptor"),
            "exact_smiles_features": feature_spec.get("exact_smiles_features", []),
            "descriptor_names": feature_spec.get("descriptor_names", []),
            "count_feature_names": feature_spec.get("count_feature_names", []),
            # fingerprint removed (fingerprint_family, radius, n_bits)
            "class_label_names": feature_spec.get("class_label_names", []),
            "feature_terms": feature_spec.get("feature_terms", []),
            "unresolved_feature_terms": feature_spec.get("unresolved_feature_terms", []),
        },
        "preprocessing_pipeline": {
            "invalid_smiles": normalized_spec["preprocessing"].get("invalid_smiles", "drop"),
            "missing_target": normalized_spec["preprocessing"].get("missing_target", "drop"),
            "missing_features": normalized_spec["preprocessing"].get("missing_features", "median_impute"),
            "duplicates": normalized_spec["preprocessing"].get("duplicates", "drop"),
            "scaling": training_spec.get("scaling", False),
        },
        "model": {
            "name": normalized_spec.get("model", {}).get("name", "random_forest"),
            "hyperparameters": normalized_spec.get("hyperparameters", {}),
        },
        "training": {
            "split_strategy": training_spec.get("split_strategy", "train_test_split"),
            "test_size": training_spec.get("test_size", 0.2),
            "random_state": training_spec.get("random_state", 42),
        },
        "metrics": ["MAE", "RMSE", "MSE", "R2"],
    }


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    for key, value in dict(override or {}).items():
        if value is None or value == "" or value == "Not found" or value == [] or value == {}:
            continue
        merged[key] = value
    return merged


def _merge_feature(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    for key in [
        "summary",
        "key_values",
        "evidence_snippet",
        "evidence_chunks",
        "method",
        "descriptor_names",
        "count_feature_names",
        # fingerprint removed (fingerprint_family, radius, n_bits)
        "retained_input_features",
        "derived_feature_names",
        "class_label_names",
        "feature_terms",
        "unresolved_feature_terms",
    ]:
        value = dict(override or {}).get(key)
        if value is None or value == "" or value == "Not found" or value == [] or value == {}:
            continue
        merged[key] = value
    return merged


def _merge_model(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    if dict(override or {}).get("name") not in {None, "", "Not found"}:
        merged["name"] = override["name"]
    return merged


def _merge_hyperparameters(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    merged.update(dict(override or {}).get("values", {}))
    return merged


def _merge_training(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    for key in ["split_strategy", "test_size", "random_state", "scaling"]:
        value = dict(override or {}).get(key)
        if value in {None, "", "Not found"}:
            continue
        merged[key] = value
    return merged
