from __future__ import annotations

import ast
import re
from typing import Any


def run_validation(code: str, code_spec: dict[str, Any] | None = None) -> dict[str, object]:
    generated_spec = extract_spec_literal(code)
    expected_feature_method = str((code_spec or {}).get("feature_pipeline", {}).get("method", "descriptor"))

    checks = {
        "rdkit": "rdkit" in code.lower(),
        "sklearn": "sklearn" in code.lower(),
        "mae": "mean_absolute_error" in code,
        "mse": "mean_squared_error" in code,
        "r2": "r2_score" in code,
        "rmse": "rmse" in code.lower(),
        "feature_logic": has_expected_feature_logic(code, expected_feature_method),
        "syntax": is_valid_python_syntax(code),
        "markdown_fence": "```" not in code,
        "spec_literal": bool(generated_spec),
    }

    if code_spec:
        checks["model_alignment"] = is_model_aligned(code_spec, generated_spec)
        checks["feature_alignment"] = is_feature_aligned(code_spec, generated_spec)
        checks["hyperparameter_alignment"] = are_hyperparameters_aligned(code_spec, generated_spec)
        checks["training_alignment"] = is_training_aligned(code_spec, generated_spec)

    missing_requirements = [name for name, passed in checks.items() if not passed]
    return {
        "is_valid": len(missing_requirements) == 0,
        "missing_requirements": missing_requirements,
        "checks": checks,
    }


def is_valid_python_syntax(code: str) -> bool:
    if not code.strip():
        return False
    try:
        compile(code, "<generated_code>", "exec")
        return True
    except SyntaxError:
        return False


def extract_spec_literal(code: str) -> dict[str, Any]:
    if not code:
        return {}
    match = re.search(r"\bSPEC\s*=", code)
    if not match:
        return {}
    start = code.find("{", match.end())
    if start < 0:
        return {}

    depth = 0
    end = -1
    for index in range(start, len(code)):
        char = code[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end < 0:
        return {}

    candidate = code[start:end]
    try:
        parsed = ast.literal_eval(candidate)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def has_expected_feature_logic(code: str, expected_method: str) -> bool:
    has_descriptor = "Descriptors.descList" in code or "descriptor" in code.lower()
    has_morgan = "GetMorganFingerprintAsBitVect" in code or "morgan" in code.lower()
    if expected_method == "combined":
        return has_descriptor and has_morgan
    if expected_method == "morgan":
        return has_morgan
    return has_descriptor


def is_model_aligned(code_spec: dict[str, Any], generated_spec: dict[str, Any]) -> bool:
    expected_model = str(code_spec.get("model", {}).get("name", "")).strip()
    generated_model = str(generated_spec.get("model", {}).get("name", "")).strip()
    if not expected_model:
        return True
    return expected_model == generated_model


def is_feature_aligned(code_spec: dict[str, Any], generated_spec: dict[str, Any]) -> bool:
    expected_method = str(code_spec.get("feature_pipeline", {}).get("method", "")).strip()
    generated_method = str(generated_spec.get("feature_pipeline", {}).get("method", "")).strip()
    if not expected_method:
        return True
    return expected_method == generated_method


def are_hyperparameters_aligned(code_spec: dict[str, Any], generated_spec: dict[str, Any]) -> bool:
    expected_hyperparameters = code_spec.get("model", {}).get("hyperparameters", {}) or {}
    if not expected_hyperparameters:
        return True
    generated_hyperparameters = generated_spec.get("model", {}).get("hyperparameters", {}) or {}
    if not isinstance(generated_hyperparameters, dict):
        return False
    return set(expected_hyperparameters).issubset(set(generated_hyperparameters))


def is_training_aligned(code_spec: dict[str, Any], generated_spec: dict[str, Any]) -> bool:
    expected_training = code_spec.get("training", {}) or {}
    if not expected_training:
        return True
    generated_training = generated_spec.get("training", {}) or {}
    if not isinstance(generated_training, dict):
        return False

    for key in ["split_strategy", "random_state", "test_size"]:
        expected_value = expected_training.get(key)
        if expected_value in {None, "", "Not found"}:
            continue
        generated_value = generated_training.get(key)
        if key == "test_size":
            try:
                if float(expected_value) != float(generated_value):
                    return False
            except (TypeError, ValueError):
                return False
            continue
        if str(expected_value) != str(generated_value):
            return False
    return True
