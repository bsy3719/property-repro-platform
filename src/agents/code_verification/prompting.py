from __future__ import annotations

import json


def build_repair_prompt(
    code_spec: dict,
    generated_code: str,
    issues: list[dict],
    assumptions: list[str],
    validation_feedback: str,
) -> str:
    return (
        "You are a code verification and repair agent inside a LangGraph workflow.\n"
        "Your job is to rewrite the full Python regression script so it exactly matches the provided code_spec.\n"
        "Return raw Python source code only.\n"
        "Do not use markdown fences.\n\n"
        "Hard rules:\n"
        "1) code_spec is the final implementation contract.\n"
        "2) code_spec.feature_pipeline is already evidence-first and must be treated as the only allowed feature contract.\n"
        "3) Remove any feature generation that is not explicitly allowed by code_spec.\n"
        "4) Do not re-introduce descriptor, count, fingerprint, or tabular features that are not backed by the evidence-first feature contract.\n"
        "5) If smiles_feature_names is non-empty, implement those exact paper-backed features from the SMILES column with RDKit instead of requiring uploaded tabular columns for them.\n"
        "6) If dataset_required_columns remain unresolved and missing_required_feature_columns is non-empty, keep the code in a failed state instead of inventing substitute features.\n"
        "7) class_label_names are metadata only and must not be added to the model feature matrix unless they are also present in allowed_feature_columns.\n"
        "8) If fingerprint_family is null, the script must not contain Morgan, MACCS, atom-pair, topological torsion, or RDKit fingerprint API calls.\n"
        "9) If descriptor_names is empty and use_rdkit_descriptors is false, do not include full Descriptors.descList fallback logic.\n"
        "10) If count_feature_names is empty, keep build_count_feature_matrix() as a no-op without extra count feature branches.\n"
        "11) Keep the canonical scaffold helper names and function order.\n"
        "12) Keep SPEC synchronized with the implemented code.\n"
        "13) Do not invent new features, preprocessing, model settings, or split settings.\n"
        "14) If a helper is not needed, keep it as a minimal no-op instead of adding forbidden logic.\n"
        "15) Preserve JSON output with metrics, assumptions, y_test, and y_pred.\n"
        "16) Every item in ASSUMPTIONS must be written in Korean.\n\n"
        f"Structured code spec:\n{json.dumps(code_spec, ensure_ascii=False, indent=2)}\n\n"
        f"Existing assumptions (must remain Korean):\n{json.dumps(assumptions, ensure_ascii=False, indent=2)}\n\n"
        f"Verification issues to fix:\n{json.dumps(issues, ensure_ascii=False, indent=2)}\n\n"
        f"Previous validation/debug feedback:\n{validation_feedback or 'None'}\n\n"
        f"Current code:\n{generated_code}"
    )
