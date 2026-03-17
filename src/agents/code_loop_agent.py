from __future__ import annotations

import re
from collections import Counter
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.code_execution_agent import CodeExecutionAgent
from src.agents.code_generation_agent import CodeGenerationAgent
from src.agents.code_generation.prompting import build_generation_prompt
from src.agents.code_generation.safety_net import build_safety_net_code
from src.agents.code_generation.validation import run_validation
from src.utils import create_openai_client, run_text_response, sanitize_python_code


class CodeLoopState(TypedDict, total=False):
    raw_paper_info: dict[str, Any] | str
    generated_code: str
    code_path: str
    data_path: str
    sheet_name: str | None
    smiles_column: str
    target_column: str
    python_executable: str
    code_spec: dict[str, Any]
    assumptions: list[str]
    validation_feedback: str
    generation_result: dict[str, Any]
    verification_result: dict[str, Any]
    verification_report_path: str
    execution_result: dict[str, Any]
    iteration: int
    max_iterations: int
    error_history: list[dict[str, Any]]
    repeated_error_count: int
    stop_reason: str
    successful_features: list[str]
    latest_error: str
    latest_stdout: str
    latest_stderr: str
    final_output: dict[str, Any]
    error: str


class CodeGenerationRunDebugAgent:
    """Generate -> execute -> regenerate from execution error."""

    def __init__(self, model_name: str = "gpt-5.2") -> None:
        self.client, self.model_name = create_openai_client(model_name)
        self.generator = CodeGenerationAgent(model_name=model_name)
        self.executor = CodeExecutionAgent()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(CodeLoopState)
        graph.add_node("initialize", self.initialize)
        graph.add_node("generate_code", self.generate_code)
        graph.add_node("run_code", self.run_code)
        graph.add_node("record_failure", self.record_failure)
        graph.add_node("debug_code", self.debug_code)
        graph.add_node("apply_safety_net", self.apply_safety_net)
        graph.add_node("finalize", self.finalize)

        graph.set_entry_point("initialize")
        graph.add_edge("initialize", "generate_code")
        graph.add_conditional_edges("generate_code", self._route_after_generate, {"run": "run_code", "failed": "record_failure"})
        graph.add_conditional_edges("run_code", self._route_after_run, {"success": "finalize", "failed": "record_failure"})
        graph.add_conditional_edges("record_failure", self._route_after_failure, {"stop": "apply_safety_net", "debug": "debug_code"})
        graph.add_edge("debug_code", "run_code")
        graph.add_edge("apply_safety_net", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def invoke(self, state: CodeLoopState) -> CodeLoopState:
        initial_state = {
            "iteration": 0,
            "max_iterations": int(state.get("max_iterations", 4)),
            "error_history": list(state.get("error_history", [])),
            **state,
        }
        return self.graph.invoke(initial_state)

    def initialize(self, state: CodeLoopState) -> CodeLoopState:
        return {
            "iteration": int(state.get("iteration", 0)),
            "max_iterations": max(3, min(int(state.get("max_iterations", 4)), 5)),
            "error_history": list(state.get("error_history", [])),
            "successful_features": list(state.get("successful_features", [])),
            "latest_error": str(state.get("latest_error", "")),
            "latest_stdout": str(state.get("latest_stdout", "")),
            "latest_stderr": str(state.get("latest_stderr", "")),
            "assumptions": list(state.get("assumptions", [])),
        }

    def generate_code(self, state: CodeLoopState) -> CodeLoopState:
        generation_result = self.generator.invoke({"raw_paper_info": state.get("raw_paper_info", {})})
        generated_code = generation_result.get("generated_code") or generation_result.get("final_output", {}).get("generated_code", "")
        final_generation_output = generation_result.get("final_output", {})
        validation_result = final_generation_output.get("validation_result", {})
        verification_result = self._validation_to_verification(validation_result)
        return {
            "generated_code": generated_code,
            "generation_result": generation_result,
            "code_spec": final_generation_output.get("code_spec", {}),
            "assumptions": final_generation_output.get("assumptions", []),
            "validation_feedback": generation_result.get("validation_feedback", ""),
            "verification_result": verification_result,
            "verification_report_path": "",
            "iteration": int(state.get("iteration", 0)) + 1,
        }

    def run_code(self, state: CodeLoopState) -> CodeLoopState:
        execution_state = self.executor.invoke(
            {
                "generated_code": state.get("generated_code", ""),
                "code_path": state.get("code_path"),
                "data_path": state.get("data_path", ""),
                "sheet_name": state.get("sheet_name"),
                "smiles_column": state.get("smiles_column", "smiles"),
                "target_column": state.get("target_column", "boiling_point"),
                "python_executable": state.get("python_executable"),
            }
        )
        final_execution = execution_state.get("execution_result", {}) or execution_state.get("final_output", {}).get("execution_result", {})
        error = execution_state.get("error") or execution_state.get("final_output", {}).get("error") or ""
        successful_features = list(state.get("successful_features", []))
        if not error and final_execution.get("status") == "success" and final_execution.get("returncode") == 0:
            new_features = self._extract_feature_names(final_execution)
            if new_features:
                existing = set(successful_features)
                successful_features = successful_features + [f for f in new_features if f not in existing]
        return {
            "execution_result": final_execution,
            "code_path": execution_state.get("code_path") or state.get("code_path"),
            "error": error,
            "latest_error": error or self._extract_error_message({"execution_result": final_execution}),
            "latest_stdout": final_execution.get("stdout", ""),
            "latest_stderr": final_execution.get("stderr", ""),
            "successful_features": successful_features,
        }

    def record_failure(self, state: CodeLoopState) -> CodeLoopState:
        error_message = self._extract_error_message(state)
        error_signature = self._build_error_signature(error_message)
        error_history = list(state.get("error_history", []))
        error_history.append(
            {
                "iteration": int(state.get("iteration", 0)),
                "signature": error_signature,
                "message": error_message,
                "stderr": state.get("latest_stderr", ""),
                "stdout": state.get("latest_stdout", ""),
            }
        )
        repeated_error_count = Counter(entry.get("signature", "") for entry in error_history).get(error_signature, 0)
        return {
            "error_history": error_history,
            "repeated_error_count": repeated_error_count,
            "error": error_message,
            "latest_error": error_message,
            "stop_reason": self._build_stop_reason(repeated_error_count, int(state.get("iteration", 0)), int(state.get("max_iterations", 4))),
        }

    def debug_code(self, state: CodeLoopState) -> CodeLoopState:
        code_spec = state.get("code_spec", {})
        assumptions = list(state.get("assumptions", []))
        current_code = state.get("generated_code", "")
        prompt = build_generation_prompt(
            code_spec,
            assumptions,
            error_feedback=self._extract_error_message(state),
        ) + f"\n\nCurrent code to fix:\n{current_code}\n"
        try:
            fixed_code = sanitize_python_code(run_text_response(self.client, self.model_name, prompt, "실행 에러 기반 코드 재생성"))
        except RuntimeError:
            fixed_code = ""
        if not fixed_code:
            fixed_code = current_code
        validation_result = run_validation(fixed_code, code_spec)
        if not validation_result.get("is_valid"):
            fixed_code = current_code
            validation_result = run_validation(fixed_code, code_spec)
        return {
            "generated_code": fixed_code,
            "verification_result": self._validation_to_verification(validation_result),
            "verification_report_path": "",
            "iteration": int(state.get("iteration", 0)) + 1,
            "error": "",
        }

    def apply_safety_net(self, state: CodeLoopState) -> CodeLoopState:
        """최대 재시도 도달 시 safety net 코드를 생성하고 실행한다."""
        code_spec = state.get("code_spec", {})
        successful_features = list(state.get("successful_features", []))
        assumptions = list(state.get("assumptions", []))

        safety_code = build_safety_net_code(code_spec, successful_features, assumptions)

        execution_state = self.executor.invoke({
            "generated_code": safety_code,
            "data_path": state.get("data_path", ""),
            "sheet_name": state.get("sheet_name"),
            "smiles_column": state.get("smiles_column", "smiles"),
            "target_column": state.get("target_column", "boiling_point"),
            "python_executable": state.get("python_executable"),
        })
        final_execution = execution_state.get("execution_result", {}) or execution_state.get("final_output", {}).get("execution_result", {})
        error = execution_state.get("error") or execution_state.get("final_output", {}).get("error") or ""
        return {
            "generated_code": safety_code,
            "execution_result": final_execution,
            "code_path": execution_state.get("code_path") or state.get("code_path"),
            "error": error,
            "latest_error": error,
            "latest_stdout": final_execution.get("stdout", ""),
            "latest_stderr": final_execution.get("stderr", ""),
        }

    def finalize(self, state: CodeLoopState) -> CodeLoopState:
        verification_result = state.get("verification_result", {})
        execution_result = state.get("execution_result", {})
        if execution_result.get("status") == "success" and execution_result.get("returncode") == 0:
            verification_status = "passed"
        elif verification_result and not verification_result.get("is_valid", True):
            verification_status = "failed"
        else:
            verification_status = "basic"
        return {
            "final_output": {
                "generated_code": state.get("generated_code", ""),
                "code_path": state.get("code_path"),
                "generation_result": state.get("generation_result", {}),
                "verification_result": verification_result,
                "verification_status": verification_status,
                "verification_issue_count": len(verification_result.get("issues", [])),
                "verification_report_path": "",
                "execution_result": execution_result,
                "iteration": state.get("iteration", 0),
                "max_iterations": state.get("max_iterations", 4),
                "repeated_error_count": state.get("repeated_error_count", 0),
                "stop_reason": state.get("stop_reason", ""),
                "error": state.get("error", ""),
                "latest_error": state.get("latest_error", ""),
                "latest_stdout": state.get("latest_stdout", ""),
                "latest_stderr": state.get("latest_stderr", ""),
                "error_history": state.get("error_history", []),
                "assumptions": state.get("assumptions", []),
                "successful_features": state.get("successful_features", []),
                "safety_net_applied": "safety net 모드" in str(state.get("generated_code", "")),
            }
        }

    def _route_after_generate(self, state: CodeLoopState) -> str:
        verification_result = state.get("verification_result", {})
        return "run" if verification_result.get("is_valid", False) else "failed"

    def _route_after_run(self, state: CodeLoopState) -> str:
        execution_result = state.get("execution_result", {})
        if state.get("error"):
            return "failed"
        if execution_result.get("status") == "success" and execution_result.get("returncode") == 0:
            return "success"
        return "failed"

    def _route_after_failure(self, state: CodeLoopState) -> str:
        return "stop" if state.get("stop_reason") else "debug"

    def _extract_error_message(self, state: CodeLoopState) -> str:
        if state.get("error"):
            return str(state.get("error"))
        execution_result = state.get("execution_result", {})
        if execution_result.get("stderr", "").strip():
            return execution_result["stderr"].strip()
        if execution_result.get("stdout", "").strip():
            return execution_result["stdout"].strip()
        verification_result = state.get("verification_result", {})
        if verification_result and not verification_result.get("is_valid", True):
            return "; ".join(issue.get("message", "") for issue in verification_result.get("issues", []))
        return "Unknown execution error"

    def _build_error_signature(self, message: str) -> str:
        lines: list[str] = []
        for line in [line.strip() for line in message.splitlines() if line.strip()]:
            normalized = re.sub(r'File ".*?"', 'File "<path>"', line)
            normalized = re.sub(r"line \d+", "line <n>", normalized)
            lines.append(normalized)
        return "\n".join(lines[-12:]) if lines else "unknown_error"

    def _extract_feature_names(self, execution_result: dict) -> list[str]:
        """실행 결과의 JSON 출력에서 성공적으로 계산된 feature 이름 목록을 추출한다."""
        parsed = execution_result.get("parsed_output", {})
        if not isinstance(parsed, dict):
            return []
        feature_summary = parsed.get("feature_summary", {})
        if not isinstance(feature_summary, dict):
            return []
        # 새 scaffold: feature_summary.feature_names
        names = feature_summary.get("feature_names")
        if isinstance(names, list) and names:
            return [str(n) for n in names if n]
        # 구 fallback_script scaffold
        old: list[str] = (
            list(feature_summary.get("descriptor_names") or [])
            + list(feature_summary.get("exact_smiles_features") or [])
            + list(feature_summary.get("count_feature_names") or [])
        )
        return [str(n) for n in old if n]

    def _build_stop_reason(self, repeated_error_count: int, iteration: int, max_iterations: int) -> str:
        if repeated_error_count >= 2:
            return "같은 실행 에러가 2회 이상 반복되어 재생성 루프를 중단했습니다."
        if iteration >= max_iterations:
            return "최대 반복 횟수에 도달하여 재생성 루프를 중단했습니다."
        return ""

    def _validation_to_verification(self, validation_result: dict[str, Any]) -> dict[str, Any]:
        issues = [
            {
                "category": "spec",
                "severity": "error",
                "rule_id": requirement,
                "message": requirement,
                "evidence": "",
                "expected": None,
                "actual": None,
            }
            for requirement in validation_result.get("missing_requirements", [])
        ]
        return {
            "is_valid": bool(validation_result.get("is_valid")),
            "issues": issues,
            "fix_applied": False,
            "original_code_changed": False,
            "checks": validation_result.get("checks", {}),
            "repair_attempts": 0,
            "report_path": "",
        }
