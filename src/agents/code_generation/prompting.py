from __future__ import annotations

import json


def build_generation_prompt(code_spec: dict, assumptions: list[str], validation_feedback: str) -> str:
    paper_markdown = str(code_spec.get("paper_markdown", ""))
    model_anchor_summary = str(code_spec.get("model_anchor_summary", ""))
    paper_method_spec = code_spec.get("paper_method_spec", {})
    prompt_context = {
        key: value
        for key, value in code_spec.items()
        if key not in {"paper_markdown", "model_anchor_summary"}
    }
    return (
        "You are a code generation agent inside a LangGraph workflow.\n"
        "Generate one executable Python script for a chemical regression task.\n"
        "Your primary job is to faithfully implement the paper methodology before filling gaps with defaults.\n\n"
        "Reasoning protocol:\n"
        "1) Treat paper_method_spec as the authoritative structured specification extracted in section 6.\n"
        "2) Use paper_markdown as the background source for surrounding implementation details and missing context only.\n"
        "3) Use model_anchor_summary as a human-readable cross-check, not as the main source of truth.\n"
        "4) If paper_method_spec and paper_markdown conflict, keep the selected model/settings from paper_method_spec.\n"
        "5) Only use defaults when the paper_method_spec and paper_markdown both leave a detail unclear.\n"
        "6) Do not invent a different model than the one supported by the paper evidence.\n\n"
        "Implementation requirements:\n"
        "1) Return raw Python source code only.\n"
        "2) Do not wrap the answer in markdown fences like ```python or ```.\n"
        "3) Do not add explanations, bullets, comments outside the Python file, or any prose before/after the script.\n"
        "4) The goal is to reproduce the machine learning model described in the paper and obtain the regression metrics reported by the paper.\n"
        "5) Use RDKit for SMILES feature handling.\n"
        "6) Implement the feature method required by paper_method_spec. If the paper leaves it incomplete, default to RDKit descriptors.\n"
        "7) Write explicit helper functions for RDKit parsing and feature generation.\n"
        "8) For descriptor features, compute descriptor values safely, convert invalid values to NaN, and impute missing values before modeling.\n"
        "9) Use sklearn for modeling.\n"
        "10) Include MAE, RMSE, MSE, and R2 in the final evaluation.\n"
        "11) Handle invalid SMILES, missing values, duplicates, and scaling when needed.\n"
        "12) Support CSV and Excel input.\n"
        "13) Do not use SMILES, compound name, molecule name, identifier-like columns, or the target column as additional tabular model features.\n"
        "14) Use the SMILES column only for RDKit featurization.\n"
        "15) Respect excluded_feature_columns in the spec.\n"
        "16) The script must be directly executable from CLI and print a JSON result with metrics.\n"
        "17) Include a top-level SPEC = {...} Python dictionary literal that exactly matches the implemented dataset, feature_pipeline, preprocessing_pipeline, model, training, and metrics.\n\n"
        f"Paper method spec (authoritative):\n{json.dumps(paper_method_spec, ensure_ascii=False, indent=2)}\n\n"
        f"Paper markdown (background source):\n{paper_markdown}\n\n"
        f"Model anchor summary (cross-check only):\n{model_anchor_summary}\n\n"
        f"Structured code spec JSON:\n{json.dumps(prompt_context, ensure_ascii=False, indent=2)}\n\n"
        f"Assumptions so far:\n{json.dumps(assumptions, ensure_ascii=False, indent=2)}\n\n"
        f"Validation feedback from previous attempt:\n{validation_feedback or 'None'}\n"
    )
