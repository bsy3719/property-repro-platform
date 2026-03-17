from __future__ import annotations

import json
import re
from typing import Any

from src.agents.code_generation.validation import extract_spec_literal
from src.utils import sanitize_python_code

# fingerprint removed
COUNT_BRANCH_PATTERNS = [
    'feature_name == "AtomCount"',
    'feature_name == "BondCount"',
    "Descriptors.HeavyAtomCount",
    "Descriptors.NumHeteroatoms",
    "Descriptors.RingCount",
    "Descriptors.NumRotatableBonds",
    "Descriptors.NumAliphaticRings",
    "Descriptors.NumAromaticRings",
    "Descriptors.NumSaturatedRings",
    "Descriptors.NumHAcceptors",
    "Descriptors.NumHDonors",
    "Descriptors.NHOHCount",
    "Descriptors.NOCount",
    "mol.GetNumAtoms()",
    "mol.GetNumBonds()",
]
COUNT_BRANCH_PATTERN_BY_NAME = {
    "AtomCount": ['feature_name == "AtomCount"', "mol.GetNumAtoms()"],
    "BondCount": ['feature_name == "BondCount"', "mol.GetNumBonds()"],
    "HeavyAtomCount": ["Descriptors.HeavyAtomCount"],
    "NumHeteroatoms": ["Descriptors.NumHeteroatoms"],
    "RingCount": ['feature_name == "RingCount"', "Descriptors.RingCount"],
    "NumRotatableBonds": ["Descriptors.NumRotatableBonds"],
    "NumAliphaticRings": ["Descriptors.NumAliphaticRings"],
    "NumAromaticRings": ["Descriptors.NumAromaticRings"],
    "NumSaturatedRings": ["Descriptors.NumSaturatedRings"],
    "NumHAcceptors": ["Descriptors.NumHAcceptors"],
    "NumHDonors": ["Descriptors.NumHDonors"],
    "NHOHCount": ["Descriptors.NHOHCount"],
    "NOCount": ["Descriptors.NOCount"],
}
TABULAR_ENGINEERING_PATTERNS = [
    "select_dtypes",
    "get_dummies",
    "OneHotEncoder",
    "LabelEncoder",
    "pd.factorize",
    "factorize(",
]
SMILES_FEATURE_PATTERN_BY_NAME = {
    "mw": ["Descriptors.MolWt"],
    "mf": ["CalcMolFormula", "mf::"],
    "polararea": ["Descriptors.TPSA"],
    "heavycnt": ["Descriptors.HeavyAtomCount"],
    "hbondacc": ["Descriptors.NumHAcceptors"],
    "C number": ["GetAtomicNum() == 6"],
    "N number": ["GetAtomicNum() == 7"],
    "O number": ["GetAtomicNum() == 8"],
    "side chain number": ["max(atom.GetDegree() - 2, 0)"],
}


def validate_code_contract(code: str, code_spec: dict[str, Any]) -> dict[str, Any]:
    sanitized_code = sanitize_python_code(code)
    generated_spec = extract_spec_literal(sanitized_code)
    feature_pipeline = (code_spec or {}).get("feature_pipeline", {}) or {}
    preprocessing_pipeline = (code_spec or {}).get("preprocessing_pipeline", {}) or {}
    model_spec = (code_spec or {}).get("model", {}) or {}
    training_spec = (code_spec or {}).get("training", {}) or {}

    checks: dict[str, bool] = {}
    issues: list[dict[str, Any]] = []

    checks["spec_literal"] = bool(generated_spec)
    missing_required_columns = list(feature_pipeline.get("missing_required_feature_columns") or [])
    checks["required_feature_columns_resolved"] = not missing_required_columns
    if missing_required_columns:
        issues.append(
            _issue(
                category="feature",
                rule_id="missing_required_feature_columns",
                message="논문에 명시된 필수 feature를 사용 가능한 입력으로 해결하지 못했습니다.",
                expected=feature_pipeline.get("required_model_columns", []),
                actual=missing_required_columns,
                evidence="feature_pipeline.missing_required_feature_columns is non-empty",
            )
        )

    if not generated_spec:
        issues.append(
            _issue(
                category="spec",
                rule_id="missing_spec_literal",
                message="생성 코드에서 SPEC 리터럴을 파싱하지 못했습니다.",
                expected="Python dict literal assigned to SPEC",
                actual="SPEC missing or unparsable",
            )
        )
    else:
        checks["feature_pipeline_match"] = generated_spec.get("feature_pipeline", {}) == feature_pipeline
        if not checks["feature_pipeline_match"]:
            issues.append(
                _issue(
                    category="feature",
                    rule_id="feature_spec_mismatch",
                    message="SPEC.feature_pipeline이 code_spec.feature_pipeline과 일치하지 않습니다.",
                    expected=feature_pipeline,
                    actual=generated_spec.get("feature_pipeline", {}),
                    evidence=_diff_summary("feature_pipeline", feature_pipeline, generated_spec.get("feature_pipeline", {})),
                )
            )

        checks["preprocessing_pipeline_match"] = generated_spec.get("preprocessing_pipeline", {}) == preprocessing_pipeline
        if not checks["preprocessing_pipeline_match"]:
            issues.append(
                _issue(
                    category="preprocessing",
                    rule_id="preprocessing_spec_mismatch",
                    message="SPEC.preprocessing_pipeline이 code_spec.preprocessing_pipeline과 일치하지 않습니다.",
                    expected=preprocessing_pipeline,
                    actual=generated_spec.get("preprocessing_pipeline", {}),
                    evidence=_diff_summary(
                        "preprocessing_pipeline",
                        preprocessing_pipeline,
                        generated_spec.get("preprocessing_pipeline", {}),
                    ),
                )
            )

        checks["model_spec_match"] = generated_spec.get("model", {}) == model_spec
        if not checks["model_spec_match"]:
            issues.append(
                _issue(
                    category="model",
                    rule_id="model_spec_mismatch",
                    message="SPEC.model이 code_spec.model과 일치하지 않습니다.",
                    expected=model_spec,
                    actual=generated_spec.get("model", {}),
                    evidence=_diff_summary("model", model_spec, generated_spec.get("model", {})),
                )
            )

        checks["training_spec_match"] = generated_spec.get("training", {}) == training_spec
        if not checks["training_spec_match"]:
            issues.append(
                _issue(
                    category="training",
                    rule_id="training_spec_mismatch",
                    message="SPEC.training이 code_spec.training과 일치하지 않습니다.",
                    expected=training_spec,
                    actual=generated_spec.get("training", {}),
                    evidence=_diff_summary("training", training_spec, generated_spec.get("training", {})),
                )
            )

    descriptor_body = _function_body(sanitized_code, "build_descriptor_matrix")
    count_body = _function_body(sanitized_code, "build_count_feature_matrix")
    # fingerprint removed
    smiles_body = _function_body(sanitized_code, "build_smiles_feature_matrix")
    tabular_body = _function_body(sanitized_code, "build_tabular_feature_matrix")
    assemble_body = _function_body(sanitized_code, "assemble_feature_matrix")

    # fingerprint removed

    descriptor_issues = _validate_descriptor_logic(descriptor_body, feature_pipeline)
    checks["descriptor_logic_match"] = not descriptor_issues
    issues.extend(descriptor_issues)

    count_issues = _validate_count_logic(count_body, feature_pipeline)
    checks["count_logic_match"] = not count_issues
    issues.extend(count_issues)

    smiles_issues = _validate_smiles_feature_logic(smiles_body, assemble_body, feature_pipeline)
    checks["smiles_feature_logic_match"] = not smiles_issues
    issues.extend(smiles_issues)

    tabular_issues = _validate_tabular_logic(tabular_body, feature_pipeline)
    checks["tabular_logic_match"] = not tabular_issues
    issues.extend(tabular_issues)

    class_label_issues = _validate_class_label_handling(smiles_body + "\n" + tabular_body + "\n" + assemble_body, feature_pipeline)
    checks["class_label_handling_match"] = not class_label_issues
    issues.extend(class_label_issues)

    preprocessing_issues = _validate_required_preprocessing_logic(sanitized_code, preprocessing_pipeline)
    checks["preprocessing_logic_match"] = not preprocessing_issues
    issues.extend(preprocessing_issues)

    model_issues = _validate_required_model_logic(sanitized_code, model_spec)
    checks["model_logic_match"] = not model_issues
    issues.extend(model_issues)

    training_issues = _validate_required_training_logic(sanitized_code, training_spec)
    checks["training_logic_match"] = not training_issues
    issues.extend(training_issues)

    return {
        "is_valid": not any(issue.get("severity") == "error" for issue in issues),
        "issues": issues,
        "checks": checks,
        "generated_spec": generated_spec,
    }
# fingerprint removed


def _validate_descriptor_logic(function_body: str, feature_pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    descriptor_names = list(feature_pipeline.get("descriptor_names") or [])
    use_rdkit_descriptors = bool(feature_pipeline.get("use_rdkit_descriptors"))
    forbidden_patterns: list[str] = []

    if not descriptor_names and not use_rdkit_descriptors:
        forbidden_patterns.extend(["Descriptors.descList", "selected_names = list(descriptor_map)", "use_all_descriptors"])
    elif descriptor_names:
        forbidden_patterns.extend(["selected_names = list(descriptor_map)", "use_all_descriptors=True"])

    issues: list[dict[str, Any]] = []
    for pattern in forbidden_patterns:
        if pattern in function_body:
            issues.append(
                _issue(
                    category="feature",
                    rule_id="unexpected_descriptor_logic",
                    message="허용되지 않은 descriptor fallback 또는 추가 descriptor 로직이 코드에 포함되어 있습니다.",
                    expected={
                        "descriptor_names": descriptor_names,
                        "use_rdkit_descriptors": use_rdkit_descriptors,
                    },
                    actual=pattern,
                    evidence=_snippet(function_body, pattern),
                )
            )
    return issues


def _validate_count_logic(function_body: str, feature_pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    expected_count_names = list(feature_pipeline.get("count_feature_names") or [])
    if expected_count_names:
        issues: list[dict[str, Any]] = []
        for feature_name, patterns in COUNT_BRANCH_PATTERN_BY_NAME.items():
            if feature_name in expected_count_names:
                continue
            for pattern in patterns:
                if pattern in function_body:
                    issues.append(
                        _issue(
                            category="feature",
                            rule_id="unexpected_count_feature_logic",
                            message="evidence-first contract에 없는 count feature 계산 분기가 코드에 포함되어 있습니다.",
                            expected=expected_count_names,
                            actual=feature_name,
                            evidence=_snippet(function_body, pattern),
                        )
                    )
                    break
        return issues

    issues: list[dict[str, Any]] = []
    for pattern in COUNT_BRANCH_PATTERNS:
        if pattern in function_body:
            issues.append(
                _issue(
                    category="feature",
                    rule_id="unexpected_count_feature_logic",
                    message="count feature가 스펙에 없는데 코드에 count feature 계산 분기가 포함되어 있습니다.",
                    expected=[],
                    actual=pattern,
                    evidence=_snippet(function_body, pattern),
                )
            )
    return issues


def _validate_smiles_feature_logic(
    function_body: str,
    assemble_body: str,
    feature_pipeline: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_smiles_feature_names = list(feature_pipeline.get("smiles_feature_names") or [])
    issues: list[dict[str, Any]] = []

    if expected_smiles_feature_names:
        if not function_body:
            issues.append(
                _issue(
                    category="feature",
                    rule_id="missing_smiles_feature_helper",
                    message="SMILES 기반 exact feature helper가 코드에 없습니다.",
                    expected=expected_smiles_feature_names,
                    actual="build_smiles_feature_matrix missing",
                )
            )
            return issues
        if "build_smiles_feature_matrix" not in assemble_body:
            issues.append(
                _issue(
                    category="feature",
                    rule_id="missing_smiles_feature_assembly",
                    message="assemble_feature_matrix()가 SMILES 기반 exact feature helper를 사용하지 않습니다.",
                    expected="build_smiles_feature_matrix(...)",
                    actual="missing",
                )
            )
        for feature_name, patterns in SMILES_FEATURE_PATTERN_BY_NAME.items():
            if feature_name in expected_smiles_feature_names:
                missing_patterns = [pattern for pattern in patterns if pattern not in function_body]
                if missing_patterns:
                    issues.append(
                        _issue(
                            category="feature",
                            rule_id="missing_expected_smiles_feature_logic",
                            message="필수 SMILES 기반 feature 생성 로직이 코드에 없습니다.",
                            expected=feature_name,
                            actual=missing_patterns,
                            evidence=function_body.strip()[:400],
                        )
                    )
            else:
                for pattern in patterns:
                    if pattern in function_body:
                        issues.append(
                            _issue(
                                category="feature",
                                rule_id="unexpected_smiles_feature_logic",
                                message="evidence-first contract에 없는 SMILES 기반 feature 생성 분기가 코드에 포함되어 있습니다.",
                                expected=expected_smiles_feature_names,
                                actual=feature_name,
                                evidence=_snippet(function_body, pattern),
                            )
                        )
                        break
        return issues

    for feature_name, patterns in SMILES_FEATURE_PATTERN_BY_NAME.items():
        for pattern in patterns:
            if pattern in function_body:
                issues.append(
                    _issue(
                        category="feature",
                        rule_id="unexpected_smiles_feature_logic",
                        message="SMILES 기반 exact feature가 스펙에 없는데 코드에 포함되어 있습니다.",
                        expected=[],
                        actual=feature_name,
                        evidence=_snippet(function_body, pattern),
                    )
                )
                break
    return issues


def _validate_tabular_logic(function_body: str, feature_pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    expected_columns = list(feature_pipeline.get("allowed_feature_columns") or [])
    if expected_columns and "allowed_feature_columns" not in function_body:
        issues.append(
            _issue(
                category="feature",
                rule_id="missing_allowed_feature_columns_policy",
                message="tabular feature 생성 로직이 allowed_feature_columns 정책을 직접 사용하지 않습니다.",
                expected=expected_columns,
                actual="allowed_feature_columns not referenced",
            )
        )

    for pattern in TABULAR_ENGINEERING_PATTERNS:
        if pattern in function_body:
            issues.append(
                _issue(
                    category="feature",
                    rule_id="unexpected_tabular_feature_engineering",
                    message="허용되지 않은 추가 tabular feature engineering 로직이 감지되었습니다.",
                    expected=expected_columns,
                    actual=pattern,
                    evidence=_snippet(function_body, pattern),
                )
            )
    return issues


def _validate_class_label_handling(function_body: str, feature_pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    allowed_columns = {str(value).strip().lower() for value in feature_pipeline.get("allowed_feature_columns") or []}
    for class_label_name in feature_pipeline.get("class_label_names") or []:
        label = str(class_label_name).strip().lower()
        if not label or label in allowed_columns:
            continue
        if f'"{class_label_name}"' in function_body or f"'{class_label_name}'" in function_body:
            issues.append(
                _issue(
                    category="feature",
                    rule_id="class_label_used_as_model_feature",
                    message="class_label_names가 모델 feature로 사용되고 있습니다.",
                    expected="metadata only",
                    actual=class_label_name,
                    evidence=_snippet(function_body, class_label_name),
                )
            )
    return issues


def _validate_required_preprocessing_logic(code: str, preprocessing_pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    if preprocessing_pipeline.get("duplicates") == "drop" and "drop_duplicates" not in code:
        issues.append(
            _issue(
                category="preprocessing",
                rule_id="missing_duplicate_drop_logic",
                message="duplicates=drop이 요구되지만 drop_duplicates 로직이 없습니다.",
                expected="drop_duplicates",
                actual="missing",
            )
        )

    if preprocessing_pipeline.get("missing_target") == "drop" and "dropna" not in code:
        issues.append(
            _issue(
                category="preprocessing",
                rule_id="missing_target_drop_logic",
                message="missing_target=drop이 요구되지만 dropna 로직이 없습니다.",
                expected="dropna on target column",
                actual="missing",
            )
        )

    if preprocessing_pipeline.get("missing_features") == "median_impute":
        if "SimpleImputer" not in code or 'strategy="median"' not in code:
            issues.append(
                _issue(
                    category="preprocessing",
                    rule_id="missing_median_imputer_logic",
                    message="missing_features=median_impute가 요구되지만 median imputer 로직이 없습니다.",
                    expected='SimpleImputer(strategy="median")',
                    actual="missing or different",
                )
            )

    if preprocessing_pipeline.get("scaling") and "StandardScaler" not in code:
        issues.append(
            _issue(
                category="preprocessing",
                rule_id="missing_scaling_logic",
                message="scaling=True인데 StandardScaler 로직이 없습니다.",
                expected="StandardScaler",
                actual="missing",
            )
        )

    return issues


def _validate_required_model_logic(code: str, model_spec: dict[str, Any]) -> list[dict[str, Any]]:
    model_name = str(model_spec.get("name", "")).strip()
    expected_token_map = {
        "random_forest": "RandomForestRegressor",
        "ridge": "Ridge",
        "lasso": "Lasso",
        "elasticnet": "ElasticNet",
        "svr": "SVR",
        "knn": "KNeighborsRegressor",
        "mlp": "MLPRegressor",
    }
    expected_token = expected_token_map.get(model_name)
    if not expected_token or expected_token in code:
        return []
    return [
        _issue(
            category="model",
            rule_id="missing_expected_model_logic",
            message="선택된 모델을 구성하는 클래스 로직이 코드에 없습니다.",
            expected=expected_token,
            actual=model_name or "missing",
        )
    ]


def _validate_required_training_logic(code: str, training_spec: dict[str, Any]) -> list[dict[str, Any]]:
    split_strategy = training_spec.get("split_strategy")
    if split_strategy == "train_test_split" and "train_test_split" not in code:
        return [
            _issue(
                category="training",
                rule_id="missing_train_test_split_logic",
                message="split_strategy=train_test_split인데 train_test_split 로직이 없습니다.",
                expected="train_test_split",
                actual="missing",
            )
        ]
    return []


def _function_body(code: str, function_name: str) -> str:
    match = re.search(
        rf"(?ms)^def {re.escape(function_name)}\([^)]*\):\n(?P<body>.*?)(?=^def |\Z)",
        code,
    )
    return match.group("body") if match else ""


def _issue(
    *,
    category: str,
    rule_id: str,
    message: str,
    expected: Any,
    actual: Any,
    evidence: str = "",
    severity: str = "error",
) -> dict[str, Any]:
    return {
        "category": category,
        "severity": severity,
        "rule_id": rule_id,
        "message": message,
        "evidence": evidence,
        "expected": expected,
        "actual": actual,
    }


def _snippet(text: str, pattern: str, radius: int = 160) -> str:
    index = text.find(pattern)
    if index < 0:
        return ""
    start = max(0, index - radius)
    end = min(len(text), index + len(pattern) + radius)
    return text[start:end].strip()


def _diff_summary(section_name: str, expected: Any, actual: Any) -> str:
    return (
        f"{section_name} expected={json.dumps(expected, ensure_ascii=False, sort_keys=True)} "
        f"actual={json.dumps(actual, ensure_ascii=False, sort_keys=True)}"
    )
