from __future__ import annotations

import json
import re
from typing import Any

from .schemas import IDENTIFIER_COLUMN_PATTERNS

VALID_MODEL_NAMES = {"random_forest", "ridge", "lasso", "elasticnet", "svr", "knn", "mlp"}
VALID_FEATURE_METHODS = {"descriptor", "morgan", "combined"}


def default_hyperparameters(model_name: str, provided_hyperparameters: dict[str, Any], assumptions: list[str]) -> dict[str, Any]:
    hyperparameters = dict(provided_hyperparameters)
    original_keys = set(hyperparameters)
    if model_name == "random_forest":
        hyperparameters.setdefault("n_estimators", 300)
        hyperparameters.setdefault("random_state", 42)
        hyperparameters.setdefault("n_jobs", -1)
    elif model_name == "ridge":
        hyperparameters.setdefault("alpha", 1.0)
        hyperparameters.setdefault("random_state", 42)
    elif model_name == "lasso":
        hyperparameters.setdefault("alpha", 0.001)
        hyperparameters.setdefault("random_state", 42)
    elif model_name == "elasticnet":
        hyperparameters.setdefault("alpha", 0.001)
        hyperparameters.setdefault("l1_ratio", 0.5)
        hyperparameters.setdefault("random_state", 42)
    elif model_name == "svr":
        hyperparameters.setdefault("kernel", "rbf")
        hyperparameters.setdefault("C", 10.0)
        hyperparameters.setdefault("epsilon", 0.1)
    elif model_name == "knn":
        hyperparameters.setdefault("n_neighbors", 5)
        hyperparameters.setdefault("weights", "distance")
    elif model_name == "mlp":
        hyperparameters.setdefault("hidden_layer_sizes", [256, 128])
        hyperparameters.setdefault("max_iter", 500)
        hyperparameters.setdefault("random_state", 42)

    added_keys = [key for key in hyperparameters if key not in original_keys]
    if added_keys:
        assumptions.append(f"모델 세부값이 부족하여 {model_name}에 대한 기본 하이퍼파라미터를 보완했습니다: {', '.join(added_keys)}")
    return hyperparameters


def infer_model_name(model_spec: dict[str, Any], context_text: str) -> str:
    search_text = json.dumps(model_spec, ensure_ascii=False).lower() + " " + context_text.lower()
    if "randomforestregressor" in search_text or "randomforest" in search_text or "random forest" in search_text:
        return "random_forest"
    if "ridge" in search_text:
        return "ridge"
    if re.search(r"\blasso\b", search_text):
        return "lasso"
    if "elasticnet" in search_text or "elastic net" in search_text:
        return "elasticnet"
    if "svr" in search_text or "support vector" in search_text:
        return "svr"
    if "kneighborsregressor" in search_text or "knn" in search_text or "k-nearest" in search_text:
        return "knn"
    if "mlpregressor" in search_text or "mlp" in search_text or "neural network" in search_text:
        return "mlp"
    return "random_forest"


def infer_feature_method(feature_spec: dict[str, Any], context_text: str) -> str:
    search_text = json.dumps(feature_spec, ensure_ascii=False).lower() + " " + context_text.lower()
    has_morgan = "morgan" in search_text or "fingerprint" in search_text
    has_descriptor = "descriptor" in search_text or "rdkit" in search_text
    if has_morgan and has_descriptor:
        return "combined"
    if has_morgan:
        return "morgan"
    return "descriptor"


def infer_excluded_feature_columns(columns: list[str], smiles_column: str, target_column: str) -> list[str]:
    excluded_columns: list[str] = []
    normalized_target = target_column.lower().strip()
    normalized_smiles = smiles_column.lower().strip()
    for column_name in columns:
        normalized_name = column_name.lower().strip()
        if normalized_name in {normalized_smiles, normalized_target}:
            excluded_columns.append(column_name)
            continue
        if normalized_name in IDENTIFIER_COLUMN_PATTERNS:
            excluded_columns.append(column_name)
            continue
        if any(pattern in normalized_name for pattern in ["compound", "molecule", "identifier", "inchikey", "inchi", "cas"]):
            excluded_columns.append(column_name)
    return excluded_columns


def fill_dataset_details(dataset: dict[str, Any], assumptions: list[str]) -> dict[str, Any]:
    dataset_spec = dict(dataset)
    if not dataset_spec.get("smiles_column"):
        dataset_spec["smiles_column"] = "smiles"
        assumptions.append("SMILES 컬럼 정보가 없어 기본값 'smiles'를 사용했습니다.")
    if not dataset_spec.get("target_column"):
        dataset_spec["target_column"] = "boiling_point"
        assumptions.append("타겟 컬럼 정보가 없어 기본값 'boiling_point'를 사용했습니다.")
    if not dataset_spec.get("file_path"):
        dataset_spec["file_path"] = "data.csv"
        assumptions.append("데이터 파일 경로 정보가 없어 기본값 'data.csv'를 사용했습니다.")
    if "sheet_name" not in dataset_spec:
        dataset_spec["sheet_name"] = None

    columns = dataset_spec.get("columns") or []
    excluded_columns = infer_excluded_feature_columns(columns, dataset_spec["smiles_column"], dataset_spec["target_column"])
    dataset_spec["excluded_feature_columns"] = excluded_columns
    dataset_spec["allowed_feature_columns"] = [column for column in columns if column not in excluded_columns]
    return dataset_spec


def fill_preprocessing_details(preprocessing: dict[str, Any]) -> dict[str, Any]:
    preprocessing_spec = dict(preprocessing)
    preprocessing_spec.setdefault("invalid_smiles", "drop")
    preprocessing_spec.setdefault("missing_target", "drop")
    preprocessing_spec.setdefault("missing_features", "median_impute")
    preprocessing_spec.setdefault("duplicates", "drop")
    return preprocessing_spec


def fill_feature_details(feature: dict[str, Any], context_text: str, assumptions: list[str]) -> dict[str, Any]:
    feature_spec = dict(feature)
    explicit_method = feature_spec.get("method") if feature_spec.get("method") in VALID_FEATURE_METHODS else None
    feature_method = explicit_method or infer_feature_method(feature_spec, context_text)
    feature_spec["method"] = feature_method

    if feature_method == "morgan":
        missing_params = []
        if "radius" not in feature_spec:
            feature_spec["radius"] = 2
            missing_params.append("radius")
        if "n_bits" not in feature_spec:
            feature_spec["n_bits"] = 2048
            missing_params.append("n_bits")
        if not explicit_method or missing_params:
            assumptions.append("feature 정보가 불완전하여 Morgan fingerprint 기본값을 보완했습니다.")
        return feature_spec

    if feature_method == "combined":
        missing_params = []
        if "radius" not in feature_spec:
            feature_spec["radius"] = 2
            missing_params.append("radius")
        if "n_bits" not in feature_spec:
            feature_spec["n_bits"] = 2048
            missing_params.append("n_bits")
        if "use_rdkit_descriptors" not in feature_spec:
            feature_spec["use_rdkit_descriptors"] = True
            missing_params.append("use_rdkit_descriptors")
        if not explicit_method or missing_params:
            assumptions.append("feature 정보가 불완전하여 Morgan fingerprint + RDKit descriptor 구성을 보완했습니다.")
        return feature_spec

    feature_spec["method"] = "descriptor"
    if "use_rdkit_descriptors" not in feature_spec:
        feature_spec["use_rdkit_descriptors"] = True
    if not explicit_method:
        assumptions.append("feature 정보가 불완전하거나 기본 설정이 필요하여 RDKit descriptor를 기본 feature로 사용합니다.")
    return feature_spec


def fill_model_details(model: dict[str, Any], context_text: str) -> dict[str, Any]:
    model_spec = dict(model)
    explicit_name = model_spec.get("name")
    if explicit_name in VALID_MODEL_NAMES:
        return model_spec
    model_spec["name"] = infer_model_name(model_spec, context_text)
    return model_spec


def fill_training_details(training: dict[str, Any], model_name: str, assumptions: list[str]) -> dict[str, Any]:
    training_spec = dict(training)
    if "test_size" not in training_spec:
        training_spec["test_size"] = 0.2
    if "random_state" not in training_spec:
        training_spec["random_state"] = 42
    if "scaling" not in training_spec:
        training_spec["scaling"] = model_name in {"ridge", "lasso", "elasticnet", "svr", "knn", "mlp"}
    if "split_strategy" not in training_spec:
        training_spec["split_strategy"] = "train_test_split"
        assumptions.append("데이터 분할 방식이 없어 train_test_split(test_size=0.2, random_state=42)를 사용합니다.")
    return training_spec
