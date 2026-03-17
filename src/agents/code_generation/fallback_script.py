from __future__ import annotations

from pprint import pformat

COUNT_FEATURE_SNIPPETS = {
    "AtomCount": '                row.append(float(mol.GetNumAtoms()) if mol is not None else float("nan"))',
    "BondCount": '                row.append(float(mol.GetNumBonds()) if mol is not None else float("nan"))',
    "HeavyAtomCount": '                row.append(float(Descriptors.HeavyAtomCount(mol)) if mol is not None else float("nan"))',
    "NumHeteroatoms": '                row.append(float(Descriptors.NumHeteroatoms(mol)) if mol is not None else float("nan"))',
    "RingCount": '                row.append(float(Descriptors.RingCount(mol)) if mol is not None else float("nan"))',
    "NumRotatableBonds": '                row.append(float(Descriptors.NumRotatableBonds(mol)) if mol is not None else float("nan"))',
    "NumAliphaticRings": '                row.append(float(Descriptors.NumAliphaticRings(mol)) if mol is not None else float("nan"))',
    "NumAromaticRings": '                row.append(float(Descriptors.NumAromaticRings(mol)) if mol is not None else float("nan"))',
    "NumSaturatedRings": '                row.append(float(Descriptors.NumSaturatedRings(mol)) if mol is not None else float("nan"))',
    "NumHAcceptors": '                row.append(float(Descriptors.NumHAcceptors(mol)) if mol is not None else float("nan"))',
    "NumHDonors": '                row.append(float(Descriptors.NumHDonors(mol)) if mol is not None else float("nan"))',
    "NHOHCount": '                row.append(float(Descriptors.NHOHCount(mol)) if mol is not None else float("nan"))',
    "NOCount": '                row.append(float(Descriptors.NOCount(mol)) if mol is not None else float("nan"))',
}
# fingerprint removed


def build_fallback_code(code_spec: dict, assumptions: list[str]) -> str:
    spec_literal = pformat(code_spec, sort_dicts=False, width=100)
    assumptions_literal = pformat(assumptions, width=100)
    feature_pipeline = code_spec.get("feature_pipeline", {}) or {}

    exact_feature_body = _build_exact_feature_function_body(list(feature_pipeline.get("exact_smiles_features") or []))
    descriptor_body = _build_descriptor_function_body(list(feature_pipeline.get("descriptor_names") or []))
    count_body = _build_count_function_body(list(feature_pipeline.get("count_feature_names") or []))
    # fingerprint removed

    return f"""import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
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
    parser = argparse.ArgumentParser(description="SMILES-only boiling point regression reproduction")
    parser.add_argument("--data-path", default=SPEC["dataset"]["file_path"])
    parser.add_argument("--sheet-name", default=SPEC["dataset"].get("sheet_name"))
    parser.add_argument("--smiles-col", default=SPEC["dataset"]["smiles_column"])
    parser.add_argument("--target-col", default=SPEC["dataset"]["target_column"])
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


def safe_float(value):
    try:
        numeric_value = float(value)
    except Exception:
        return np.nan
    return numeric_value if np.isfinite(numeric_value) else np.nan


def estimate_side_chain_number(mol):
    if mol is None:
        return np.nan
    return float(sum(max(atom.GetDegree() - 2, 0) for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1))


def build_exact_feature_matrix(mols, exact_smiles_features):
{exact_feature_body}


def build_descriptor_matrix(mols, descriptor_names):
{descriptor_body}


def build_count_feature_matrix(mols, count_feature_names):
{count_body}


# fingerprint removed


def assemble_feature_matrix(df, mols):
    pipeline = SPEC["feature_pipeline"]
    exact_matrix, exact_names = build_exact_feature_matrix(mols, list(pipeline.get("exact_smiles_features") or []))
    descriptor_matrix, descriptor_names = build_descriptor_matrix(mols, list(pipeline.get("descriptor_names") or []))
    count_matrix, count_names = build_count_feature_matrix(mols, list(pipeline.get("count_feature_names") or []))
    # fingerprint removed

    feature_blocks = [block for block in [exact_matrix, descriptor_matrix, count_matrix] if block.shape[1] > 0]
    if not feature_blocks:
        raise ValueError("구성된 SMILES 기반 feature가 없습니다.")

    X = feature_blocks[0] if len(feature_blocks) == 1 else np.hstack(feature_blocks)
    return X, {{
        "feature_source": "smiles_only",
        "exact_smiles_features": exact_names,
        "descriptor_names": descriptor_names,
        "count_feature_names": count_names,
        # fingerprint removed
        "class_label_names": list(pipeline.get("class_label_names") or []),
        "unresolved_feature_terms": list(pipeline.get("unresolved_feature_terms") or []),
    }}


def build_model(model_name: str, hyperparameters: dict):
    if model_name == "random_forest":
        return RandomForestRegressor(**hyperparameters)
    if model_name == "ridge":
        return Ridge(**hyperparameters)
    if model_name == "lasso":
        return Lasso(**hyperparameters)
    if model_name == "elasticnet":
        return ElasticNet(**hyperparameters)
    if model_name == "svr":
        return SVR(**hyperparameters)
    if model_name == "knn":
        return KNeighborsRegressor(**hyperparameters)
    if model_name == "mlp":
        return MLPRegressor(**hyperparameters)
    return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=SPEC["training"]["test_size"],
        random_state=SPEC["training"]["random_state"],
    )
    model = build_model(SPEC["model"]["name"], SPEC["model"]["hyperparameters"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    result = {{
        "y_test": [float(value) for value in y_test.tolist()[:200]],
        "y_pred": [float(value) for value in y_pred.tolist()[:200]],
        "metrics": {{
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "RMSE": rmse,
            "MSE": float(mse),
            "R2": float(r2_score(y_test, y_pred)),
        }},
    }}
    return result


def main():
    args = parse_args()
    df = load_dataframe(args.data_path, args.sheet_name)
    df = df.drop_duplicates(subset=[args.smiles_col, args.target_col])
    df = df.dropna(subset=[args.smiles_col, args.target_col]).copy()
    df[args.target_col] = pd.to_numeric(df[args.target_col], errors="coerce")
    df = df.dropna(subset=[args.target_col]).reset_index(drop=True)

    mols = []
    valid_index = []
    for index, smiles in df[args.smiles_col].items():
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        mols.append(mol)
        valid_index.append(index)

    if not mols:
        raise ValueError("유효한 SMILES가 없습니다.")

    df = df.loc[valid_index].reset_index(drop=True)
    X_raw, feature_summary = assemble_feature_matrix(df, mols)
    y = df[args.target_col].to_numpy(dtype=float)

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw)

    if SPEC["preprocessing_pipeline"].get("scaling"):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    evaluation = train_and_evaluate(X, y)
    runtime_assumptions = list(ASSUMPTIONS)
    if "mf" in set(feature_summary.get("exact_smiles_features") or []):
        runtime_assumptions.append("분자식(mf)은 RDKit 계산 후 one-hot 형태로 모델 입력에 반영했습니다.")
    if "side chain number" in set(feature_summary.get("exact_smiles_features") or []):
        runtime_assumptions.append("side chain number는 논문 정의가 부족해 branching heuristic으로 계산했습니다.")

    result = {{
        "assumptions": runtime_assumptions,
        "spec": SPEC,
        "feature_summary": feature_summary,
        "y_test": evaluation["y_test"],
        "y_pred": evaluation["y_pred"],
        "metrics": evaluation["metrics"],
    }}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
"""


def _build_exact_feature_function_body(exact_smiles_features: list[str]) -> str:
    if not exact_smiles_features:
        return """    return np.zeros((len(mols), 0), dtype=float), []"""

    branch_map = {
        "mw": '                row.append(safe_float(Descriptors.MolWt(mol) if mol is not None else np.nan))',
        "polararea": '                row.append(safe_float(Descriptors.TPSA(mol) if mol is not None else np.nan))',
        "heavycnt": '                row.append(safe_float(Descriptors.HeavyAtomCount(mol) if mol is not None else np.nan))',
        "hbondacc": '                row.append(safe_float(Descriptors.NumHAcceptors(mol) if mol is not None else np.nan))',
        "C number": '                row.append(float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)) if mol is not None else np.nan)',
        "N number": '                row.append(float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)) if mol is not None else np.nan)',
        "O number": '                row.append(float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)) if mol is not None else np.nan)',
        "side chain number": '                row.append(estimate_side_chain_number(mol))',
        "mf": '                row.extend([1.0 if formula_value == label else 0.0 for label in formula_labels])',
    }
    branch_lines: list[str] = []
    for index, feature_name in enumerate(exact_smiles_features):
        prefix = "if" if index == 0 else "elif"
        branch_lines.append(f'            {prefix} feature_name == "{feature_name}":\n{branch_map.get(feature_name, "                row.append(np.nan)")}')
    branches = "\n".join(branch_lines)
    formula_setup = """    formula_values = ["" for _ in mols]
    formula_labels = []"""
    if "mf" in exact_smiles_features:
        formula_setup = """    formula_values = []
    for mol in mols:
        if mol is None:
            formula_values.append("")
            continue
        try:
            formula_values.append(str(rdMolDescriptors.CalcMolFormula(mol) or ""))
        except Exception:
            formula_values.append("")
    formula_labels = sorted(set([value for value in formula_values if value]))"""
    return f"""    exact_feature_names = list(exact_smiles_features or [])
    if not exact_feature_names:
        return np.zeros((len(mols), 0), dtype=float), []

{formula_setup}

    resolved_names = []
    for feature_name in exact_feature_names:
        if feature_name == "mf":
            resolved_names.extend(["mf::" + label for label in formula_labels])
        else:
            resolved_names.append(feature_name)

    if not resolved_names:
        return np.zeros((len(mols), 0), dtype=float), []

    rows = []
    for mol, formula_value in zip(mols, formula_values):
        row = []
        for feature_name in exact_feature_names:
{branches}
            else:
                row.append(np.nan)
        rows.append(row)
    return np.asarray(rows, dtype=float), resolved_names"""


def _build_descriptor_function_body(descriptor_names: list[str]) -> str:
    if not descriptor_names:
        return """    return np.zeros((len(mols), 0), dtype=float), []"""
    return """    descriptor_map = {name: func for name, func in Descriptors.descList}
    selected_names = [name for name in list(descriptor_names or []) if name in descriptor_map]
    if not selected_names:
        return np.zeros((len(mols), 0), dtype=float), []

    rows = []
    for mol in mols:
        row = []
        for name in selected_names:
            try:
                row.append(safe_float(descriptor_map[name](mol)))
            except Exception:
                row.append(np.nan)
        rows.append(row)
    return np.asarray(rows, dtype=float), selected_names"""


def _build_count_function_body(count_feature_names: list[str]) -> str:
    if not count_feature_names:
        return """    return np.zeros((len(mols), 0), dtype=float), []"""

    rows_logic: list[str] = []
    for index, feature_name in enumerate(count_feature_names):
        prefix = "if" if index == 0 else "elif"
        snippet = COUNT_FEATURE_SNIPPETS.get(feature_name, '                row.append(float("nan"))')
        rows_logic.append(f'            {prefix} feature_name == "{feature_name}":\n{snippet}')
    branches = "\n".join(rows_logic)
    return f"""    selected_names = list(count_feature_names or [])
    if not selected_names:
        return np.zeros((len(mols), 0), dtype=float), []

    rows = []
    for mol in mols:
        row = []
        for feature_name in selected_names:
{branches}
            else:
                row.append(float("nan"))
        rows.append(row)
    return np.asarray(rows, dtype=float), selected_names"""


# fingerprint removed
