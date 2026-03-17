"""
code_gen_prompt.py

LLM 기반 코드 생성을 위한 시스템 프롬프트 및 유저 프롬프트 빌더.
"""

from __future__ import annotations

import json
from pprint import pformat

from ..resources.rdkit_descriptor_reference import (
    ATOM_BOND_COUNT_REFERENCE,
    COMMON_CHEMISTRY_DESCRIPTORS,
    FEATURE_FALLBACK_MAP,
    RDKIT_DESCRIPTOR_LIST,
)

# ---------------------------------------------------------------------------
# 정적 시스템 프롬프트 구성 요소
# ---------------------------------------------------------------------------

_ROLE = """\
You are a Python code generation agent for chemistry property regression.

Your task:
- Receive a paper_method_spec (extracted from a chemistry ML paper) and code_spec
- Generate one complete, executable Python script that reproduces the ML experiment
- The script trains a regression model on a SMILES-based molecular dataset

Hard constraints:
- Use ONLY the resolved features explicitly listed in feature_resolution or code_spec["feature_pipeline"]["feature_resolution"]
- Final model features may include resolved RDKit descriptors and resolved atom/bond count features — no additions
- Every feature must be computed from SMILES strings using RDKit — never read feature columns from the dataset
- The dataset has exactly two columns: a SMILES column and a numeric target column
- Fingerprint-based features (morgan, maccs, atom_pair, topological_torsion, rdkit FP) are FORBIDDEN
- Do NOT generate or use build_fingerprint_matrix()
- Do NOT use unresolved raw paper feature names directly in the final feature matrix
- If a paper feature was resolved from names like mw, mf, polararea, side chain number, C/N/O number, O number, use ONLY the resolved descriptor/count feature names in the final feature matrix

Excluded from features:
- Identifier columns: smiles, canonical_smiles, iso-smiles, id, mol_id, compound_name, cmpdname, cas, inchi
- Class label columns: hydrocarbon, alcohol, amine, and similar categorical labels
- Target column: boiling_point, activity, or any prediction target\
"""

_STRUCTURE = """\
Required script structure (functions must appear in this order):
1. imports
2. SPEC = { ... }       ← full code_spec as a Python dict literal
3. ASSUMPTIONS = { ... } ← Python dict recording feature mapping decisions and default usage
4. load_data(file_path, sheet_name=None)
5. build_feature_matrix(smiles_list)
   - Each feature computation wrapped in an individual try/except
   - Returns (numpy_array, list_of_feature_names)
6. train_model(X, y)
7. evaluate_model(model, X_test, y_test) → dict with MAE, RMSE, MSE, R2
8. main()
   - Parses --data-path, --smiles-col, --target-col CLI args
   - Calls load_data → build_feature_matrix → impute NaN → train_model → evaluate_model
   - Prints ONE JSON object to stdout with keys: metrics, assumptions, spec, feature_summary

Output format (printed JSON):
{
  "metrics":         { "MAE": float, "RMSE": float, "MSE": float, "R2": float },
  "assumptions":     { ... },
  "spec":            { ... },
  "feature_summary": { "feature_names": [...], "n_features": int }
}\
"""

_FEATURE_RULES = """\
Feature processing rules:
1. Build the final feature set from resolved descriptor names and resolved atom/bond count names only
2. Compute descriptor features using rdkit.Chem.Descriptors with exact resolved names
3. Compute count features using the atom/bond count reference with exact resolved names
4. You may combine descriptor and count blocks with np.hstack when both are present
5. If a feature computation fails for a molecule, fill with np.nan (imputed later by SimpleImputer)
6. Do NOT fall back to full RDKit descriptor sets or any fingerprint family\
"""


def _build_descriptor_reference() -> str:
    lines = ["Available RDKit descriptors by category (use exact names from rdkit.Chem.Descriptors):"]
    for category, names in RDKIT_DESCRIPTOR_LIST.items():
        lines.append(f"\n  [{category}]")
        lines.append("  " + ", ".join(names))
    return "\n".join(lines)


def _build_count_reference() -> str:
    lines = ["Atom/bond count implementation reference (mol = RDKit Mol object):"]
    for group, snippets in ATOM_BOND_COUNT_REFERENCE.items():
        lines.append(f"\n  [{group}]")
        for key, snippet in snippets.items():
            lines.append(f"  {key}: {snippet}")
    return "\n".join(lines)


def _build_fallback_map() -> str:
    lines = ["Feature name → RDKit descriptor fallback map:"]
    lines.append("  (When a paper feature name has no exact RDKit match, use the best candidate below)")
    for name, candidates in FEATURE_FALLBACK_MAP.items():
        lines.append(f"  '{name}' → {candidates}")
    return "\n".join(lines)


def _build_common_descriptors() -> str:
    desc = ", ".join(COMMON_CHEMISTRY_DESCRIPTORS)
    return (
        "Most commonly used descriptors in property/toxicity/bioactivity prediction papers:\n"
        f"  {desc}"
    )


_FEW_SHOT = """\
--- Few-shot example A: resolved descriptor-only features ---
feature_resolution["resolved"] = {
  "mw": "MolWt",
  "polararea": "TPSA",
  "lipophilicity": "MolLogP",
}

ASSUMPTIONS = {
  "feature_mapping": {
    "mw": "MolWt",
    "polararea": "TPSA",
    "lipophilicity": "MolLogP",
  },
  "count_feature_mapping": {},
  "excluded_features": [],
  "notes": ["resolved된 descriptor만 사용"],
}

def build_feature_matrix(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import numpy as np
    descriptor_map = {name: fn for name, fn in Descriptors.descList}
    selected = ["MolWt", "TPSA", "MolLogP"]
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi)) if smi else None
        row = []
        for name in selected:
            try:
                row.append(float(descriptor_map[name](mol)) if mol else float("nan"))
            except Exception:
                row.append(float("nan"))
        rows.append(row)
    X = np.array(rows, dtype=float)
    return X, selected

--- Few-shot example B: fallback-resolved paper terms ---
feature_resolution["resolved"] = {
  "side chain number": "BertzCT",
  "mf": "HeavyAtomCount",
  "ring": "RingCount",
}

def build_feature_matrix(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import numpy as np
    descriptor_map = {name: fn for name, fn in Descriptors.descList}
    selected = ["BertzCT", "HeavyAtomCount", "RingCount"]
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi)) if smi else None
        row = []
        for name in selected:
            try:
                row.append(float(descriptor_map[name](mol)) if mol else float("nan"))
            except Exception:
                row.append(float("nan"))
        rows.append(row)
    return np.array(rows, dtype=float), selected\

--- Few-shot example C: atom/bond count features ---
feature_resolution["resolved"] = {}
feature_resolution["resolved_counts"] = {
  "C/N/O number": ["C_count", "N_count", "O_count"],
  "O number": ["O_count"],
}

def build_feature_matrix(smiles_list):
    from rdkit import Chem
    import numpy as np
    selected = ["C_count", "N_count", "O_count"]
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi)) if smi else None
        row = []
        try:
            row.append(float(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)) if mol else float("nan"))
        except Exception:
            row.append(float("nan"))
        try:
            row.append(float(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)) if mol else float("nan"))
        except Exception:
            row.append(float("nan"))
        try:
            row.append(float(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)) if mol else float("nan"))
        except Exception:
            row.append(float("nan"))
        rows.append(row)
    return np.array(rows, dtype=float), selected\
"""

# ---------------------------------------------------------------------------
# 시스템 프롬프트 (정적, 한 번 빌드)
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    """정적 시스템 프롬프트 — 역할/규칙/레퍼런스 데이터 포함."""
    sections = [
        _ROLE,
        "",
        _STRUCTURE,
        "",
        _FEATURE_RULES,
        "",
        _build_descriptor_reference(),
        "",
        _build_count_reference(),
        "",
        _build_fallback_map(),
        "",
        _build_common_descriptors(),
        "",
        _FEW_SHOT,
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# 유저 프롬프트 (요청마다 동적 생성)
# ---------------------------------------------------------------------------

def build_user_prompt(
    code_spec: dict,
    assumptions: list[str],
    feature_resolution: dict | None = None,
    error_feedback: str = "",
) -> str:
    """동적 유저 프롬프트 — code_spec, assumptions, feature 해석 결과 포함."""
    parts: list[str] = []

    # feature resolution 주입 (resolve_features 단계 결과)
    if feature_resolution:
        resolved = feature_resolution.get("resolved", {})
        resolved_counts = feature_resolution.get("resolved_counts", {})
        excluded = feature_resolution.get("excluded", [])
        llm_resolved = feature_resolution.get("llm_resolved", [])
        if resolved or resolved_counts or excluded:
            lines = ["Feature name resolution result (paper name → RDKit descriptor name):"]
            for original, rdkit_name in resolved.items():
                tag = " [LLM-selected]" if original in llm_resolved else ""
                lines.append(f"  {original!r} → {rdkit_name!r}{tag}")
            for original, count_names in resolved_counts.items():
                lines.append(f"  {original!r} → {count_names!r} [count-features]")
            if excluded:
                lines.append(f"  Excluded (identifiers/labels, do NOT use): {excluded}")
            parts.append("\n".join(lines))
            parts.append("")

    # error feedback (debug loop에서 재호출 시)
    if error_feedback:
        parts.append(f"Error from previous execution:\n{error_feedback}")
        parts.append("")

    # code_spec (JSON)
    parts.append("code_spec (implement exactly this contract):")
    parts.append(json.dumps(code_spec, ensure_ascii=False, indent=2))
    parts.append("")

    # assumptions
    parts.append("Current assumptions (reproduce these in ASSUMPTIONS dict, in Korean):")
    parts.append(json.dumps(_build_assumptions_payload(assumptions, feature_resolution), ensure_ascii=False, indent=2))
    parts.append("")

    parts.append(
        "Generate the complete Python script now. "
        "Return raw Python code only — no markdown fences, no explanation."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 통합 프롬프트 (system + user 결합)
# ---------------------------------------------------------------------------

def build_full_prompt(
    code_spec: dict,
    assumptions: list[str],
    feature_resolution: dict | None = None,
    error_feedback: str = "",
) -> str:
    """system + user를 하나의 문자열로 결합 (OpenAI Responses API input 파라미터용)."""
    system = build_system_prompt()
    user = build_user_prompt(code_spec, assumptions, feature_resolution, error_feedback)
    return f"{system}\n\n{'=' * 60}\n\n{user}"


def _build_assumptions_payload(
    assumptions: list[str],
    feature_resolution: dict | None,
) -> dict:
    resolution = feature_resolution or {}
    return {
        "feature_mapping": dict(resolution.get("resolved", {})),
        "count_feature_mapping": dict(resolution.get("resolved_counts", {})),
        "excluded_features": list(resolution.get("excluded", [])),
        "llm_resolved_features": list(resolution.get("llm_resolved", [])),
        "notes": list(assumptions or []),
    }
