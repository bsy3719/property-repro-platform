from __future__ import annotations

from typing import Any

from src.utils import internal_model_name
from src.utils.chemistry_features import merge_unique, normalize_exact_feature_terms

SUPPORTED_INTERNAL_MODELS = {"random_forest", "ridge", "lasso", "elasticnet", "svr", "knn", "mlp"}
EXCLUDED_EXACT_FEATURES = {"cmpdname", "iso smiles", "hydrocarbon", "alcohol", "amine"}
SUPPORTED_EXACT_SMILES_FEATURES = {
    "mw",
    "mf",
    "polararea",
    "heavycnt",
    "hbondacc",
    "C number",
    "N number",
    "O number",
    "side chain number",
}


def fill_dataset_details(dataset: dict[str, Any], assumptions: list[str]) -> dict[str, Any]:
    dataset_spec = dict(dataset or {})
    if not dataset_spec.get("smiles_column"):
        dataset_spec["smiles_column"] = "smiles"
        assumptions.append("SMILES 컬럼 정보가 없어 기본값 'smiles'를 사용했습니다.")
    if not dataset_spec.get("target_column"):
        dataset_spec["target_column"] = "boiling_point"
        assumptions.append("타겟 컬럼 정보가 없어 기본값 'boiling_point'를 사용했습니다.")
    if not dataset_spec.get("file_path"):
        dataset_spec["file_path"] = "data.csv"
        assumptions.append("데이터 파일 경로 정보가 없어 기본값 'data.csv'를 사용했습니다.")
    dataset_spec["sheet_name"] = dataset_spec.get("sheet_name")
    dataset_spec["input_columns"] = [dataset_spec["smiles_column"], dataset_spec["target_column"]]
    return dataset_spec


def fill_preprocessing_details(preprocessing: dict[str, Any]) -> dict[str, Any]:
    preprocessing_spec = dict(preprocessing or {})
    preprocessing_spec.setdefault("invalid_smiles", "drop")
    preprocessing_spec.setdefault("missing_target", "drop")
    preprocessing_spec.setdefault("missing_features", "median_impute")
    preprocessing_spec.setdefault("duplicates", "drop")
    return preprocessing_spec


def fill_feature_details(feature: dict[str, Any], assumptions: list[str]) -> dict[str, Any]:
    feature_spec = dict(feature or {})
    exact_smiles_features = [
        feature_name
        for feature_name in merge_unique(
            normalize_exact_feature_terms(feature_spec.get("retained_input_features")),
            normalize_exact_feature_terms(feature_spec.get("derived_feature_names")),
        )
        if feature_name not in EXCLUDED_EXACT_FEATURES and feature_name in SUPPORTED_EXACT_SMILES_FEATURES
    ]
    feature_spec["exact_smiles_features"] = exact_smiles_features
    feature_spec["feature_source"] = "smiles_only"

    if not exact_smiles_features and not feature_spec.get("descriptor_names") and not feature_spec.get("count_feature_names"):  # fingerprint removed
        feature_spec["exact_smiles_features"] = ["mw", "polararea", "heavycnt", "hbondacc"]
        assumptions.append(
            "논문 feature 세부정보가 충분하지 않아 SMILES 기반 기본 feature(mw, polararea, heavycnt, hbondacc)를 사용했습니다."
        )
    return feature_spec


def fill_model_details(model: dict[str, Any], assumptions: list[str]) -> dict[str, Any]:
    model_spec = dict(model or {})
    model_name = str(model_spec.get("name", "")).strip()
    internal_name = internal_model_name(model_name) if model_name else ""
    if internal_name in SUPPORTED_INTERNAL_MODELS:
        model_spec["name"] = internal_name
        return model_spec
    model_spec["name"] = "random_forest"
    assumptions.append("논문 모델 정보가 불충분하여 RandomForestRegressor를 사용했습니다.")
    return model_spec


def default_hyperparameters(model_name: str, provided_hyperparameters: dict[str, Any], assumptions: list[str]) -> dict[str, Any]:
    hyperparameters = dict(provided_hyperparameters or {})
    original_keys = set(hyperparameters)
    if model_name == "random_forest":
        hyperparameters.setdefault("n_estimators", 300)
        hyperparameters.setdefault("random_state", 42)
        hyperparameters.setdefault("n_jobs", -1)
    elif model_name == "ridge":
        hyperparameters.setdefault("alpha", 1.0)
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
        assumptions.append("논문에 없는 하이퍼파라미터는 안전한 기본값으로 보완했습니다: " + ", ".join(added_keys))
    return hyperparameters


def fill_training_details(training: dict[str, Any], model_name: str, assumptions: list[str]) -> dict[str, Any]:
    training_spec = dict(training or {})
    if training_spec.get("split_strategy") in {None, "", "Not found"}:
        training_spec["split_strategy"] = "train_test_split"
        assumptions.append("데이터 분할 방식이 명시되지 않아 train_test_split을 사용했습니다.")
    training_spec.setdefault("test_size", 0.2)
    training_spec.setdefault("random_state", 42)
    training_spec.setdefault("scaling", model_name in {"ridge", "lasso", "elasticnet", "svr", "knn", "mlp"})
    return training_spec
