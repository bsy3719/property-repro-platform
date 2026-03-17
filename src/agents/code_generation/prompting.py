from __future__ import annotations

import json


def build_generation_prompt(
    code_spec: dict,
    assumptions: list[str],
    error_feedback: str = "",
    feature_resolution: dict | None = None,
) -> str:
    resolution_section = ""
    if feature_resolution:
        resolved = feature_resolution.get("resolved", {})
        excluded = feature_resolution.get("excluded", [])
        llm_resolved = feature_resolution.get("llm_resolved", [])
        if resolved or excluded:
            lines = ["Feature name resolution (paper name → RDKit descriptor):"]
            for original, rdkit_name in resolved.items():
                tag = " [LLM]" if original in llm_resolved else ""
                lines.append(f"  {original!r} → {rdkit_name!r}{tag}")
            if excluded:
                lines.append(f"  Excluded (identifiers/labels): {excluded}")
            resolution_section = "\n".join(lines) + "\n\n"

    return (
        "You are a Python code generation agent for chemistry regression reproduction.\n"
        "Generate one executable Python script.\n"
        "Return raw Python code only.\n"
        "Do not use markdown fences.\n\n"
        "Hard requirements:\n"
        "1) The uploaded dataset contains only two columns: smiles and boiling_point.\n"
        "2) Never look for additional feature columns in the uploaded dataset.\n"
        "3) Every feature must be generated from the SMILES column with RDKit.\n"
        "4) The target column is boiling_point.\n"
        "5) The task is regression.\n"
        "6) Always compute MAE, RMSE, MSE, and R2.\n"
        "7) Use the provided paper_method_spec as the main implementation contract.\n"
        "8) Implement exact_smiles_features, descriptor_names, and count_feature_names only when they are present in code_spec.feature_pipeline.\n"
        "9) If the paper does not fully specify a detail, use a minimal safe default and record it in ASSUMPTIONS in Korean.\n"
        "10) Include these helpers: parse_args, load_dataframe, mol_from_smiles, build_descriptor_matrix, build_count_feature_matrix, assemble_feature_matrix, build_model, train_and_evaluate, main.\n"
        "11) You may add helper functions for exact SMILES-derived features when needed.\n"
        "12) train_and_evaluate() must return {'y_test': ..., 'y_pred': ..., 'metrics': ...}.\n"
        "13) Print one final JSON object.\n\n"
        f"{resolution_section}"
        f"Structured code spec:\n{json.dumps(code_spec, ensure_ascii=False, indent=2)}\n\n"
        f"Current assumptions (must remain Korean):\n{json.dumps(assumptions, ensure_ascii=False, indent=2)}\n\n"
        f"Error feedback from the previous run:\n{error_feedback or 'None'}\n"
    )
