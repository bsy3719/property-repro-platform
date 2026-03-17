from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.utils import create_openai_client, run_text_response, sanitize_python_code

from .prompting import build_repair_prompt
from .schemas import CodeVerificationState
from .validation import validate_code_contract

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GENERATED_CODE_DIR = PROJECT_ROOT / "artifacts" / "generated_code"
GENERATED_CODE_DIR.mkdir(parents=True, exist_ok=True)


class CodeVerificationAgent:
    def __init__(self, model_name: str = "gpt-5.2", max_repair_attempts: int = 1) -> None:
        self.client, self.model_name = create_openai_client(model_name)
        self.max_repair_attempts = max_repair_attempts

    def invoke(self, state: CodeVerificationState) -> CodeVerificationState:
        original_code = sanitize_python_code(state.get("generated_code", ""))
        code_spec = state.get("code_spec", {}) or {}
        assumptions = list(state.get("assumptions", []))
        validation_feedback = str(state.get("validation_feedback", "") or "")
        code_path = state.get("code_path", "")

        initial_validation = validate_code_contract(original_code, code_spec)
        detected_issues = list(initial_validation.get("issues", []))
        final_code = original_code
        final_validation = initial_validation
        repair_attempts = 0
        fix_applied = False

        if not initial_validation.get("is_valid") and original_code:
            final_code, final_validation, repair_attempts, fix_applied = self._repair_once(
                original_code=original_code,
                code_spec=code_spec,
                assumptions=assumptions,
                validation_feedback=validation_feedback,
                issues=detected_issues,
            )

        final_status = "passed"
        if not final_validation.get("is_valid"):
            final_status = "failed"
        elif fix_applied:
            final_status = "fixed"

        report_path = ""
        if fix_applied or detected_issues or not final_validation.get("is_valid"):
            report_path = self._save_report(
                original_code=original_code,
                verified_code=final_code,
                issues=detected_issues or list(final_validation.get("issues", [])),
                fix_applied=fix_applied,
                repair_attempts=repair_attempts,
                code_spec=code_spec,
                code_path=code_path,
                status=final_status,
            )

        verification_result = {
            "is_valid": bool(final_validation.get("is_valid")),
            "issues": detected_issues or list(final_validation.get("issues", [])),
            "fix_applied": fix_applied,
            "original_code_changed": final_code != original_code,
            "checks": final_validation.get("checks", {}),
            "repair_attempts": repair_attempts,
            "report_path": report_path,
        }
        verification_error = ""
        if not verification_result["is_valid"]:
            verification_error = self._issues_to_message(verification_result["issues"])

        return {
            "verified_code": final_code,
            "verification_result": verification_result,
            "verification_report_path": report_path,
            "verification_error": verification_error,
            "fix_applied": fix_applied,
            "error": verification_error,
        }

    def _repair_once(
        self,
        *,
        original_code: str,
        code_spec: dict,
        assumptions: list[str],
        validation_feedback: str,
        issues: list[dict],
    ) -> tuple[str, dict, int, bool]:
        prompt = build_repair_prompt(
            code_spec=code_spec,
            generated_code=original_code,
            issues=issues,
            assumptions=assumptions,
            validation_feedback=validation_feedback,
        )
        try:
            repaired_code = run_text_response(self.client, self.model_name, prompt, "코드 검증 및 수정")
            repaired_code = sanitize_python_code(repaired_code)
        except RuntimeError:
            return original_code, validate_code_contract(original_code, code_spec), 1, False

        if not repaired_code:
            repaired_code = original_code
        final_validation = validate_code_contract(repaired_code, code_spec)
        return repaired_code, final_validation, 1, repaired_code != original_code

    def _save_report(
        self,
        *,
        original_code: str,
        verified_code: str,
        issues: list[dict],
        fix_applied: bool,
        repair_attempts: int,
        code_spec: dict,
        code_path: str,
        status: str,
    ) -> str:
        report_path = self._report_path_for(code_path)
        report_payload = {
            "original_code": original_code,
            "verified_code": verified_code,
            "issues": issues,
            "fix_applied": fix_applied,
            "repair_attempts": repair_attempts,
            "code_spec_snapshot": code_spec,
            "timestamp": datetime.now().isoformat(),
            "status": status,
        }
        report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(report_path)

    def _report_path_for(self, code_path: str) -> Path:
        if code_path:
            path = Path(code_path)
        else:
            path = GENERATED_CODE_DIR / "generated_regression_latest.py"
        return path.with_name(f"{path.stem}.verification.json")

    def _issues_to_message(self, issues: list[dict]) -> str:
        if not issues:
            return "코드 검증에 실패했습니다."
        return "; ".join(f"[{issue.get('rule_id')}] {issue.get('message')}" for issue in issues)
