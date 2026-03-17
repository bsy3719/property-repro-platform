from __future__ import annotations

from typing import Any, TypedDict

IDENTIFIER_COLUMN_PATTERNS = {
    "smiles",
    "canonical_smiles",
    "compound",
    "compound_name",
    "compound name",
    "molecule",
    "molecule_name",
    "molecule name",
    "name",
    "inchi",
    "inchikey",
    "id",
    "cas",
    "cas_no",
    "cas number",
}


class CodeGenerationState(TypedDict, total=False):
    raw_paper_info: dict[str, Any] | str
    normalized_spec: dict[str, Any]
    assumptions: list[str]
    code_spec: dict[str, Any]
    feature_resolution: dict[str, Any]
    generated_code: str
    review_result: dict[str, Any]
    validation_result: dict[str, Any]
    validation_feedback: str
    retry_count: int
    final_output: dict[str, Any]
    error: str
