from __future__ import annotations

from typing import Any, TypedDict


class VerificationIssue(TypedDict, total=False):
    category: str
    severity: str
    rule_id: str
    message: str
    evidence: str
    expected: Any
    actual: Any


class CodeVerificationState(TypedDict, total=False):
    generated_code: str
    verified_code: str
    code_spec: dict[str, Any]
    assumptions: list[str]
    validation_feedback: str
    code_path: str
    verification_result: dict[str, Any]
    verification_report_path: str
    verification_error: str
    fix_applied: bool
    error: str
