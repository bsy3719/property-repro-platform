from __future__ import annotations

import json
from typing import Any

from src.utils import internal_model_name, normalize_paper_method_spec

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
        parsed_info = parse_raw_string(raw_paper_info)
    elif isinstance(raw_paper_info, dict):
        parsed_info = normalize_dict(raw_paper_info)
    else:
        parsed_info = {}
        assumptions.append("raw_paper_info 형식이 불명확하여 일반적인 회귀 파이프라인 기본값을 사용했습니다.")

    normalized_spec = {
        "dataset": parsed_info.get("dataset", {}),
        "target_property": parsed_info.get("target_property", "boiling_point"),
        "task_type": "regression",
        "preprocessing": parsed_info.get("preprocessing", {}),
        "feature": parsed_info.get("feature", {}),
        "model": parsed_info.get("model", {}),
        "hyperparameters": parsed_info.get("hyperparameters", {}),
        "training": parsed_info.get("training", {}),
        "metrics": parsed_info.get("metrics", []),
        "paper_markdown": parsed_info.get("paper_markdown", ""),
        "model_anchor_summary": parsed_info.get("model_anchor_summary", ""),
        "paper_method_spec": normalize_paper_method_spec(parsed_info.get("paper_method_spec", {})),
    }
    return normalized_spec, assumptions


def fill_missing_details(normalized_spec: dict[str, Any], assumptions: list[str]) -> tuple[dict[str, Any], list[str]]:
    updated_spec = dict(normalized_spec)
    updated_assumptions = list(assumptions)
    updated_spec["paper_method_spec"] = normalize_paper_method_spec(updated_spec.get("paper_method_spec", {}))
    paper_method_spec = updated_spec["paper_method_spec"]

    updated_spec["dataset"] = fill_dataset_details(updated_spec.get("dataset", {}), updated_assumptions)
    updated_spec["preprocessing"] = fill_preprocessing_details(_merge_preprocessing(updated_spec.get("preprocessing", {}), paper_method_spec))
    updated_spec["feature"] = fill_feature_details(
        _merge_feature(updated_spec.get("feature", {}), paper_method_spec),
        combine_context_text(updated_spec),
        updated_assumptions,
    )
    updated_spec["model"] = fill_model_details(
        _merge_model(updated_spec.get("model", {}), paper_method_spec),
        combine_context_text(updated_spec),
    )
    updated_spec["hyperparameters"] = default_hyperparameters(
        updated_spec["model"]["name"],
        _merge_hyperparameters(updated_spec.get("hyperparameters", {}), paper_method_spec),
        updated_assumptions,
    )
    updated_spec["training"] = fill_training_details(
        _merge_training(updated_spec.get("training", {}), paper_method_spec),
        updated_spec["model"]["name"],
        updated_assumptions,
    )
    updated_spec["metrics"] = merge_metrics(updated_spec.get("metrics", []), paper_method_spec)
    return updated_spec, updated_assumptions


def build_code_spec(normalized_spec: dict[str, Any]) -> dict[str, Any]:
    dataset_spec = dict(normalized_spec.get("dataset", {}))
    feature_spec = dict(normalized_spec.get("feature", {}))
    model_spec = dict(normalized_spec.get("model", {}))
    training_spec = dict(normalized_spec.get("training", {}))
    hyperparameter_spec = dict(normalized_spec.get("hyperparameters", {}))

    return {
        "task_type": "regression",
        "target_property": normalized_spec.get("target_property", "boiling_point"),
        "goal": "Run the machine learning model described in the paper and obtain the regression metrics reported by the paper.",
        "paper_markdown": normalized_spec.get("paper_markdown", ""),
        "model_anchor_summary": normalized_spec.get("model_anchor_summary", ""),
        "paper_method_spec": normalized_spec.get("paper_method_spec", {}),
        "dataset": dataset_spec,
        "feature_pipeline": {
            "method": feature_spec.get("method", "descriptor"),
            "radius": feature_spec.get("radius", 2),
            "n_bits": feature_spec.get("n_bits", 2048),
            "use_rdkit_descriptors": feature_spec.get("use_rdkit_descriptors", True),
            "excluded_feature_columns": dataset_spec.get("excluded_feature_columns", []),
            "allowed_feature_columns": dataset_spec.get("allowed_feature_columns", []),
        },
        "preprocessing_pipeline": {
            "invalid_smiles": normalized_spec["preprocessing"].get("invalid_smiles", "drop"),
            "missing_target": normalized_spec["preprocessing"].get("missing_target", "drop"),
            "missing_features": normalized_spec["preprocessing"].get("missing_features", "median_impute"),
            "duplicates": normalized_spec["preprocessing"].get("duplicates", "drop"),
            "scaling": training_spec.get("scaling", False),
        },
        "model": {"name": model_spec.get("name", "random_forest"), "hyperparameters": hyperparameter_spec},
        "training": {
            "split_strategy": training_spec.get("split_strategy", "train_test_split"),
            "test_size": training_spec.get("test_size", 0.2),
            "random_state": training_spec.get("random_state", 42),
        },
        "metrics": normalized_spec.get("metrics", ["MAE", "RMSE", "MSE", "R2"]),
    }


def combine_context_text(normalized_spec: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            str(normalized_spec.get("paper_markdown", "")),
            str(normalized_spec.get("model_anchor_summary", "")),
            json.dumps(normalized_spec.get("paper_method_spec", {}), ensure_ascii=False),
        ]
    ).strip()


def merge_metrics(metrics: list[Any], paper_method_spec: dict[str, Any]) -> list[str]:
    merged_metrics: list[str] = []
    for metric_name in ["MAE", "RMSE", "MSE", "R2"]:
        if metric_name not in merged_metrics:
            merged_metrics.append(metric_name)
    for metric in metrics or []:
        normalized_metric = str(metric).upper()
        if normalized_metric not in merged_metrics:
            merged_metrics.append(normalized_metric)
    for metric_name in paper_method_spec.get("metrics", {}).get("reported", {}).keys():
        if metric_name not in merged_metrics:
            merged_metrics.append(metric_name)
    return merged_metrics


def _merge_preprocessing(preprocessing: dict[str, Any], paper_method_spec: dict[str, Any]) -> dict[str, Any]:
    merged = dict(preprocessing)
    source = paper_method_spec.get("preprocessing", {})
    for key in ["invalid_smiles", "missing_target", "missing_features", "duplicates"]:
        value = source.get(key)
        if value not in {None, "", "Not found"}:
            merged[key] = value
    if source.get("scaling") is not None:
        merged["scaling"] = source["scaling"]
    return merged


def _merge_feature(feature: dict[str, Any], paper_method_spec: dict[str, Any]) -> dict[str, Any]:
    merged = dict(feature)
    source = paper_method_spec.get("feature", {})
    if source.get("method") not in {None, "", "Not found"}:
        merged["method"] = source["method"]
    for key in ["radius", "n_bits"]:
        value = source.get(key)
        if value is not None:
            merged[key] = value
    if source.get("use_rdkit_descriptors") is not None:
        merged["use_rdkit_descriptors"] = source["use_rdkit_descriptors"]
    return merged


def _merge_model(model: dict[str, Any], paper_method_spec: dict[str, Any]) -> dict[str, Any]:
    merged = dict(model)
    internal_name = internal_model_name(paper_method_spec.get("model", {}).get("name"))
    if internal_name:
        merged["name"] = internal_name
    return merged


def _merge_hyperparameters(hyperparameters: dict[str, Any], paper_method_spec: dict[str, Any]) -> dict[str, Any]:
    merged = dict(hyperparameters)
    merged.update(paper_method_spec.get("hyperparameters", {}).get("values", {}))
    return merged


def _merge_training(training: dict[str, Any], paper_method_spec: dict[str, Any]) -> dict[str, Any]:
    merged = dict(training)
    source = paper_method_spec.get("training", {})
    if source.get("split_strategy") not in {None, "", "Not found"}:
        merged["split_strategy"] = source["split_strategy"]
    for key in ["test_size", "random_state"]:
        value = source.get(key)
        if value is not None:
            merged[key] = value
    return merged
