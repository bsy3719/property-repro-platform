from __future__ import annotations

import json
from typing import Any


def parse_raw_string(raw_text: str) -> dict[str, Any]:
    return {
        "dataset": {},
        "target_property": "boiling_point",
        "preprocessing": {},
        "feature": {},
        "model": {},
        "hyperparameters": {},
        "training": {},
        "metrics": ["MAE", "RMSE", "MSE", "R2"],
        "paper_markdown": raw_text,
        "model_anchor_summary": "",
        "paper_method_spec": {},
    }


def normalize_dict(raw_info: dict[str, Any]) -> dict[str, Any]:
    dataset_spec = dict(raw_info.get("dataset", {}))
    preprocessing_spec = dict(raw_info.get("preprocessing", {}))
    feature_spec = dict(raw_info.get("feature", {}))
    model_spec = raw_info.get("model", {})
    if isinstance(model_spec, str):
        model_spec = {"name": model_spec}
    else:
        model_spec = dict(model_spec)

    hyperparameter_spec = raw_info.get("hyperparameters", raw_info.get("hyperparameter", {}))
    if isinstance(hyperparameter_spec, list):
        hyperparameter_spec = {f"param_{index}": value for index, value in enumerate(hyperparameter_spec)}
    elif not isinstance(hyperparameter_spec, dict):
        hyperparameter_spec = {}

    training_spec = dict(raw_info.get("training", raw_info.get("training_strategy", {})))
    metric_spec = raw_info.get("metrics", raw_info.get("metric", []))
    if not isinstance(metric_spec, list):
        metric_spec = [str(metric_spec)]

    dataset_spec.setdefault("smiles_column", raw_info.get("smiles_column"))
    dataset_spec.setdefault("target_column", raw_info.get("target_column"))
    dataset_spec.setdefault("file_path", raw_info.get("file_path"))
    dataset_spec.setdefault("sheet_name", raw_info.get("sheet_name"))
    dataset_spec.setdefault("columns", raw_info.get("columns", dataset_spec.get("columns", [])))

    target_property = raw_info.get("target_property") or raw_info.get("property") or "boiling_point"
    paper_markdown = str(raw_info.get("paper_markdown") or raw_info.get("notes") or "")
    model_anchor_summary = str(raw_info.get("model_anchor_summary") or raw_info.get("retriever_summary") or "")
    paper_method_spec = raw_info.get("paper_method_spec", {})
    if not isinstance(paper_method_spec, dict):
        paper_method_spec = {}
    if not paper_markdown:
        paper_markdown = json.dumps(raw_info, ensure_ascii=False)

    return {
        "dataset": dataset_spec,
        "target_property": target_property,
        "preprocessing": preprocessing_spec,
        "feature": feature_spec,
        "model": model_spec,
        "hyperparameters": hyperparameter_spec,
        "training": training_spec,
        "metrics": metric_spec,
        "paper_markdown": paper_markdown,
        "model_anchor_summary": model_anchor_summary,
        "paper_method_spec": paper_method_spec,
    }
