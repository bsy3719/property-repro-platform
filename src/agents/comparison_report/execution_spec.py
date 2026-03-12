from __future__ import annotations

import ast
import json
from typing import Any

from .paper_spec import format_morgan_descriptor_text, format_morgan_text
from .schemas import MODEL_CLASS_NAMES


def parse_execution_spec(final_output: dict[str, Any], generated_code: str, generated_code_path: str) -> dict[str, Any]:
    execution_result = final_output.get("execution_result", {})
    spec_from_code = extract_spec_literal(generated_code)
    return {
        "status": execution_result.get("status", "unknown"),
        "returncode": execution_result.get("returncode"),
        "preprocessing": execution_preprocessing_text(generated_code, spec_from_code),
        "feature": execution_feature_text(generated_code, spec_from_code),
        "model": execution_model_text(generated_code, spec_from_code),
        "hyperparameters": execution_hyperparameter_text(spec_from_code),
        "training": execution_training_text(generated_code, spec_from_code),
        "metrics": execution_result.get("metrics", {}) if isinstance(execution_result, dict) else {},
        "generated_code_path": generated_code_path,
    }


def extract_spec_literal(generated_code: str) -> dict[str, Any]:
    if not generated_code:
        return {}
    spec_index = generated_code.find("SPEC")
    if spec_index < 0:
        return {}
    start = generated_code.find("{", spec_index)
    if start < 0:
        return {}

    depth = 0
    end = -1
    for index in range(start, len(generated_code)):
        char = generated_code[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end < 0:
        return {}

    candidate = generated_code[start:end]
    try:
        parsed = ast.literal_eval(candidate)
    except Exception:
        try:
            parsed = json.loads(candidate)
        except Exception:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def execution_preprocessing_text(code: str, spec: dict[str, Any]) -> str:
    findings: list[str] = []
    pipeline = spec.get("preprocessing_pipeline", {})
    if "drop_duplicates" in code or pipeline.get("duplicates"):
        findings.append(f"duplicates={pipeline.get('duplicates', 'used')}")
    if "dropna" in code or pipeline.get("missing_target"):
        findings.append(f"missing_target={pipeline.get('missing_target', 'handled')}")
    if "SimpleImputer" in code or pipeline.get("missing_features"):
        findings.append(f"missing_features={pipeline.get('missing_features', 'handled')}")
    if "StandardScaler" in code or pipeline.get("scaling"):
        findings.append(f"scaling={pipeline.get('scaling', False)}")
    if pipeline.get("invalid_smiles"):
        findings.append(f"invalid_smiles={pipeline.get('invalid_smiles')}")
    return ", ".join(findings) if findings else "Unknown from generated code"


def execution_feature_text(code: str, spec: dict[str, Any]) -> str:
    pipeline = spec.get("feature_pipeline", {})
    method = pipeline.get("method")
    if method == "combined":
        return format_morgan_descriptor_text(json.dumps(pipeline, ensure_ascii=False))
    if method == "morgan":
        return format_morgan_text(json.dumps(pipeline, ensure_ascii=False))
    if method == "descriptor":
        return "RDKit descriptors"
    if "Descriptors.descList" in code and "GetMorganFingerprintAsBitVect" in code:
        return "Morgan fingerprint + RDKit descriptors"
    if "Descriptors.descList" in code:
        return "RDKit descriptors"
    if "GetMorganFingerprintAsBitVect" in code:
        return "Morgan fingerprint"
    return "Unknown from generated code"


def execution_model_text(code: str, spec: dict[str, Any]) -> str:
    model_name = spec.get("model", {}).get("name")
    if isinstance(model_name, str) and model_name:
        return normalize_model_name(model_name)
    for class_name in MODEL_CLASS_NAMES:
        if class_name in code:
            return class_name
    return "Unknown from generated code"


def execution_hyperparameter_text(spec: dict[str, Any]) -> str:
    hyperparameters = spec.get("model", {}).get("hyperparameters", {})
    if not hyperparameters:
        return "Unknown from generated code"
    return ", ".join([f"{key}={value}" for key, value in hyperparameters.items()])


def execution_training_text(code: str, spec: dict[str, Any]) -> str:
    training = spec.get("training", {})
    if "train_test_split" not in code and not training:
        return "Unknown from generated code"
    return ", ".join([
        f"split={training.get('split_strategy', 'train_test_split')}",
        f"test_size={training.get('test_size', 'unknown')}",
        f"random_state={training.get('random_state', 'unknown')}",
    ])


def normalize_model_name(model_name: str) -> str:
    lowered = model_name.lower()
    if lowered == "random_forest":
        return "RandomForestRegressor"
    if lowered == "knn":
        return "KNeighborsRegressor"
    if lowered == "elasticnet":
        return "ElasticNet"
    if lowered == "svr":
        return "SVR"
    if lowered == "mlp":
        return "MLPRegressor"
    if lowered == "ridge":
        return "Ridge"
    if lowered == "lasso":
        return "Lasso"
    return model_name
