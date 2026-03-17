from __future__ import annotations

from pprint import pformat

from .resources.rdkit_descriptor_reference import COMMON_CHEMISTRY_DESCRIPTORS

_MODEL_CLASS_MAP: dict[str, str] = {
    "random_forest": "RandomForestRegressor",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elasticnet": "ElasticNet",
    "svr": "SVR",
    "knn": "KNeighborsRegressor",
    "mlp": "MLPRegressor",
}


def build_safety_net_code(
    code_spec: dict,
    successful_features: list[str],
    assumptions: list[str],
) -> str:
    clean_features = [f for f in (successful_features or []) if isinstance(f, str) and f.strip()]
    supplemented: list[str] = []
    if len(clean_features) < 3:
        existing = set(clean_features)
        for desc in COMMON_CHEMISTRY_DESCRIPTORS:
            if desc not in existing:
                supplemented.append(desc)
            if len(clean_features) + len(supplemented) >= 3:
                break
    final_features = (clean_features + supplemented)[:15]
    if not final_features:
        final_features = list(COMMON_CHEMISTRY_DESCRIPTORS[:5])

    model_spec = dict(code_spec.get("model", {}) or {})
    model_internal = str(model_spec.get("name", "random_forest") or "random_forest")
    model_class = _MODEL_CLASS_MAP.get(model_internal, "RandomForestRegressor")
    hyperparams: dict = dict(model_spec.get("hyperparameters", {}) or {})
    if not hyperparams:
        if model_internal in {"random_forest", ""}:
            hyperparams = {"n_estimators": 100, "random_state": 42}
        elif model_internal in {"ridge", "lasso", "elasticnet", "mlp"}:
            hyperparams = {"random_state": 42}

    training_spec = dict(code_spec.get("training", {}) or {})
    test_size: float = float(training_spec.get("test_size") or 0.2)
    random_state: int = int(training_spec.get("random_state") or 42)

    dataset_spec = dict(code_spec.get("dataset", {}) or {})
    smiles_col = str(dataset_spec.get("smiles_column") or "smiles")
    target_col = str(dataset_spec.get("target_column") or "boiling_point")

    safety_assumptions = list(assumptions or [])
    safety_assumptions.append("최대 재시도 도달로 safety net 모드 실행")
    if clean_features and supplemented:
        safety_assumptions.append(f"사용된 feature: {clean_features} + 보충: {supplemented}")
    elif supplemented:
        safety_assumptions.append(f"사용된 feature: 없음, 보충: {supplemented}")
    else:
        safety_assumptions.append(f"사용된 feature: {clean_features}")

    safety_spec = {
        "dataset": {"smiles_column": smiles_col, "target_column": target_col},
        "feature_pipeline": {"descriptor_names": final_features},
        "model": {"name": model_internal, "hyperparameters": hyperparams},
        "training": {"test_size": test_size, "random_state": random_state},
    }

    spec_literal = pformat(safety_spec, sort_dicts=False, width=100)
    assumptions_literal = pformat(safety_assumptions, width=100)
    features_literal = pformat(final_features, width=100)
    hyperparams_literal = pformat(hyperparams, sort_dicts=False, width=80)

    return f"""import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SPEC = {spec_literal}
ASSUMPTIONS = {assumptions_literal}
DESCRIPTOR_NAMES = {features_literal}


def load_data(file_path: str, sheet_name: str | None = None):
    path = Path(file_path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet_name)


def build_feature_matrix(smiles_list):
    descriptor_map = {{name: fn for name, fn in Descriptors.descList}}
    selected = [name for name in DESCRIPTOR_NAMES if name in descriptor_map]
    rows = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(str(smi)) if smi and not (isinstance(smi, float) and np.isnan(float(smi))) else None
        except Exception:
            mol = None
        row = []
        for name in selected:
            try:
                value = float(descriptor_map[name](mol)) if mol is not None else float("nan")
                row.append(value if np.isfinite(value) else float("nan"))
            except Exception:
                row.append(float("nan"))
        rows.append(row)
    return np.array(rows, dtype=float), selected


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=SPEC["training"]["test_size"],
        random_state=SPEC["training"]["random_state"],
    )
    model = {model_class}(**{hyperparams_literal})
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    return {{
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "MSE": mse,
        "R2": float(r2_score(y_test, y_pred)),
    }}


def main():
    parser = argparse.ArgumentParser(description="Safety-net regression (max retries reached)")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sheet-name", default=None)
    parser.add_argument("--smiles-col", default=SPEC["dataset"]["smiles_column"])
    parser.add_argument("--target-col", default=SPEC["dataset"]["target_column"])
    args = parser.parse_args()

    df = load_data(args.data_path, args.sheet_name)
    df = df.dropna(subset=[args.smiles_col, args.target_col]).copy()
    df[args.target_col] = pd.to_numeric(df[args.target_col], errors="coerce")
    df = df.dropna(subset=[args.target_col]).reset_index(drop=True)

    X_raw, feature_names = build_feature_matrix(df[args.smiles_col].tolist())
    y = df[args.target_col].to_numpy(dtype=float)
    X = SimpleImputer(strategy="median").fit_transform(X_raw)

    model, X_test, y_test = train_model(X, y)
    metrics = evaluate_model(model, X_test, y_test)

    print(json.dumps({{
        "metrics": metrics,
        "assumptions": ASSUMPTIONS,
        "spec": SPEC,
        "feature_summary": {{"feature_names": feature_names, "n_features": len(feature_names)}},
    }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
"""
