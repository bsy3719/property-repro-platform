from __future__ import annotations

from pprint import pformat


def build_fallback_code(code_spec: dict, assumptions: list[str]) -> str:
    spec_literal = pformat(code_spec, sort_dicts=False, width=100)
    assumptions_literal = pformat(assumptions, width=100)
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


def parse_args():
    parser = argparse.ArgumentParser(description=\"Boiling point regression reproduction script\")
    parser.add_argument(\"--data-path\", default=SPEC[\"dataset\"][\"file_path\"])
    parser.add_argument(\"--sheet-name\", default=SPEC[\"dataset\"].get(\"sheet_name\"))
    parser.add_argument(\"--smiles-col\", default=SPEC[\"dataset\"][\"smiles_column\"])
    parser.add_argument(\"--target-col\", default=SPEC[\"dataset\"][\"target_column\"])
    return parser.parse_args()


def load_dataframe(data_path: str, sheet_name: str | None):
    path = Path(data_path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet_name)


def mol_from_smiles(smiles: str):
    if pd.isna(smiles):
        return None
    return Chem.MolFromSmiles(str(smiles))


def descriptor_features(mol):
    values = []
    for _, func in Descriptors.descList:
        try:
            value = func(mol)
        except Exception:
            value = np.nan
        if value is None:
            value = np.nan
        values.append(value)
    return np.asarray(values, dtype=float)


def smiles_to_features(smiles_series: pd.Series):
    features = []
    valid_index = []
    for idx, smiles in smiles_series.items():
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        feat = descriptor_features(mol)
        features.append(feat)
        valid_index.append(idx)

    if not features:
        raise ValueError(\"No valid SMILES were found after RDKit parsing.\")

    return np.vstack(features), valid_index


def build_model(model_name: str, hyperparameters: dict):
    if model_name == \"random_forest\":
        return RandomForestRegressor(**hyperparameters)
    if model_name == \"ridge\":
        return Ridge(**hyperparameters)
    if model_name == \"lasso\":
        return Lasso(**hyperparameters)
    if model_name == \"elasticnet\":
        return ElasticNet(**hyperparameters)
    if model_name == \"svr\":
        return SVR(**hyperparameters)
    if model_name == \"knn\":
        return KNeighborsRegressor(**hyperparameters)
    if model_name == \"mlp\":
        return MLPRegressor(**hyperparameters)
    return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)


def main():
    args = parse_args()
    df = load_dataframe(args.data_path, args.sheet_name)
    df = df.drop_duplicates(subset=[args.smiles_col, args.target_col])
    df = df.dropna(subset=[args.smiles_col, args.target_col]).copy()
    df[args.target_col] = pd.to_numeric(df[args.target_col], errors=\"coerce\")
    df = df.dropna(subset=[args.target_col]).reset_index(drop=True)

    X_raw, valid_index = smiles_to_features(df[args.smiles_col])
    df = df.loc[valid_index].reset_index(drop=True)
    y = df[args.target_col].to_numpy(dtype=float)

    imputer = SimpleImputer(strategy=\"median\")
    X = imputer.fit_transform(X_raw)

    if SPEC[\"preprocessing_pipeline\"][\"scaling\"]:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=SPEC[\"training\"][\"test_size\"],
        random_state=SPEC[\"training\"][\"random_state\"],
    )

    model = build_model(SPEC[\"model\"][\"name\"], SPEC[\"model\"][\"hyperparameters\"])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, predictions)

    result = {{
        \"assumptions\": ASSUMPTIONS,
        \"spec\": SPEC,
        \"metrics\": {{
            \"MAE\": float(mae),
            \"RMSE\": float(rmse),
            \"MSE\": float(mse),
            \"R2\": float(r2),
        }},
        \"n_train\": int(len(y_train)),
        \"n_test\": int(len(y_test)),
    }}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == \"__main__\":
    main()
"""
