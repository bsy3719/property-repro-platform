"""
validation.py — 생성 코드 검증 2단계 파이프라인.

Step 1  llm_review_code()  : LLM이 6가지 항목을 검토 → {"issues": [...], "fixed_code": "..."}
Step 2  ast_syntax_check() : ast.parse()로 구문 오류만 확인

run_validation()는 code_loop_agent.py 하위 호환용 경량 래퍼로 유지.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from openai import OpenAI

from src.utils import extract_json_object, run_text_response, sanitize_python_code

# ---------------------------------------------------------------------------
# Step 1 — LLM 코드 리뷰
# ---------------------------------------------------------------------------

_REVIEW_CHECKLIST = """\
Review the Python script below and check for ALL of the following issues:

1. FINGERPRINT code: any use of morgan, maccs, atom_pair, topological_torsion,
   rdkit fingerprint, GetMorganFingerprintAsBitVect, GetHashedAtomPairFingerprintAsBitVect,
   MACCSkeys, build_fingerprint_matrix, DataStructs, AllChem fingerprint calls.

2. DATASET COLUMN LEAK: code that reads feature values directly from dataframe columns
   (e.g. df["mw"], df[col] used as feature input) instead of computing them from SMILES.

3. TARGET COLUMN IN FEATURES: target column (boiling_point, activity, bp, etc.)
   included in the feature matrix X.

4. IDENTIFIER COLUMNS IN FEATURES: smiles, canonical_smiles, iso_smiles, iso-smiles,
   name, id, mol_id, compound_name, cmpdname, cas, inchi used as features.

5. MISSING SPEC / ASSUMPTIONS: the script does not define SPEC = {...} and
   ASSUMPTIONS = [...] as Python dict/list literals near the top of the file.

6. MISSING TRY/EXCEPT PER FEATURE: build_feature_matrix() (or equivalent feature
   assembly function) does not wrap each individual feature computation in its own
   try/except block.

Return ONLY a JSON object with this exact schema:
{
  "issues": ["short description of issue 1", ...],
  "fixed_code": "... complete corrected Python script ..."
}

Rules:
- If NO issues are found, return {"issues": [], "fixed_code": ""}
- "fixed_code" must be the COMPLETE corrected script, not a diff or partial snippet
- If "fixed_code" is an empty string the original code will be used unchanged
- Do not add markdown fences around fixed_code

Code to review:
"""


def llm_review_code(client: OpenAI, model_name: str, code: str) -> dict[str, Any]:
    """
    Step 1: LLM이 6가지 항목을 검토하고 수정 코드를 반환한다.

    반환값:
      {
        "issues":     ["issue description", ...],  # 빈 리스트면 문제 없음
        "fixed_code": "...",                        # 빈 문자열이면 원본 유지
        "had_issues": bool,
      }
    """
    prompt = _REVIEW_CHECKLIST + "\n```\n" + code + "\n```"
    try:
        raw = run_text_response(client, model_name, prompt, "LLM 코드 리뷰")
        parsed = extract_json_object(raw)
    except Exception:
        # LLM 호출 실패 → 검토 없이 통과
        return {"issues": [], "fixed_code": "", "had_issues": False}

    issues: list[str] = parsed.get("issues") if isinstance(parsed.get("issues"), list) else []
    raw_fixed: str = parsed.get("fixed_code", "") or ""
    fixed_code = sanitize_python_code(raw_fixed) if raw_fixed.strip() else ""

    return {
        "issues": issues,
        "fixed_code": fixed_code,
        "had_issues": bool(issues),
    }


# ---------------------------------------------------------------------------
# Step 2 — AST 구문 검사
# ---------------------------------------------------------------------------

def ast_syntax_check(code: str) -> dict[str, Any]:
    """
    Step 2: ast.parse()로 구문 오류만 확인한다.

    반환값:
      {
        "ok":      bool,
        "error":   str | None,   # SyntaxError 메시지
        "line":    int | None,   # 오류 발생 줄 번호
        "col":     int | None,   # 오류 발생 컬럼
        "context": str | None,   # 해당 줄 소스 코드
      }
    """
    if not code or not code.strip():
        return {
            "ok": False,
            "error": "코드가 비어 있습니다.",
            "line": None,
            "col": None,
            "context": None,
        }
    try:
        ast.parse(code, mode="exec")
        return {"ok": True, "error": None, "line": None, "col": None, "context": None}
    except SyntaxError as exc:
        lines = code.splitlines()
        lineno = exc.lineno or 0
        context = lines[lineno - 1] if 0 < lineno <= len(lines) else None
        return {
            "ok": False,
            "error": str(exc.msg),
            "line": exc.lineno,
            "col": exc.offset,
            "context": context,
        }


# ---------------------------------------------------------------------------
# 하위 호환 래퍼 (code_loop_agent.py에서 사용)
# ---------------------------------------------------------------------------

_EXPECTED_FUNCTIONS = [
    "load_data",
    "build_feature_matrix",
    "train_model",
    "evaluate_model",
    "main",
]


def run_validation(code: str, code_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    경량 검증: AST 구문 + 필수 함수 존재 + SPEC/ASSUMPTIONS 리터럴 확인.
    LLM 리뷰 없이 즉시 실행 가능 (code_loop_agent의 동기 경로에서 사용).
    """
    syntax = ast_syntax_check(code)
    has_spec = bool(re.search(r"\bSPEC\s*=\s*\{", code or ""))
    has_assumptions = "ASSUMPTIONS" in (code or "")
    has_functions = all(f"def {fn}(" in (code or "") for fn in _EXPECTED_FUNCTIONS)

    issues = []
    if not syntax["ok"]:
        issues.append(f"구문 오류 (line {syntax['line']}): {syntax['error']}")
    if not has_spec:
        issues.append("SPEC 딕셔너리 리터럴 없음")
    if not has_assumptions:
        issues.append("ASSUMPTIONS 리스트 없음")
    if not has_functions:
        missing = [fn for fn in _EXPECTED_FUNCTIONS if f"def {fn}(" not in (code or "")]
        issues.append(f"필수 함수 없음: {missing}")

    return {
        "is_valid": not issues,
        "missing_requirements": issues,
        "checks": {
            "syntax": syntax["ok"],
            "spec_literal": has_spec,
            "assumptions_literal": has_assumptions,
            "required_functions": has_functions,
        },
    }
