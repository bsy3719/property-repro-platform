from __future__ import annotations

import re
from collections import Counter
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.code_execution_agent import CodeExecutionAgent
from src.agents.code_generation_agent import CodeGenerationAgent
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
    generation_result: dict[str, Any]
    execution_result: dict[str, Any]
    iteration: int
    max_iterations: int
    error_history: list[dict[str, Any]]
    repeated_error_count: int
    stop_reason: str
    latest_error: str
    latest_stdout: str
    latest_stderr: str
    final_output: dict[str, Any]
    error: str


class CodeGenerationRunDebugAgent:
    """LangGraph loop agent for generate -> run -> debug retries."""

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
        graph.add_node("finalize", self.finalize)

        graph.set_entry_point("initialize")
        graph.add_conditional_edges("initialize", self._route_after_initialize, {"generate": "generate_code", "run": "run_code"})
        graph.add_edge("generate_code", "run_code")
        graph.add_conditional_edges("run_code", self._route_after_run, {"success": "finalize", "failed": "record_failure"})
        graph.add_conditional_edges("record_failure", self._route_after_failure, {"stop": "finalize", "debug": "debug_code"})
        graph.add_edge("debug_code", "run_code")
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
            "latest_error": str(state.get("latest_error", "")),
            "latest_stdout": str(state.get("latest_stdout", "")),
            "latest_stderr": str(state.get("latest_stderr", "")),
        }

    def generate_code(self, state: CodeLoopState) -> CodeLoopState:
        generation_result = self.generator.invoke({"raw_paper_info": state.get("raw_paper_info", {})})
        generated_code = generation_result.get("generated_code") or generation_result.get("final_output", {}).get("generated_code", "")
        return {
            "generated_code": generated_code,
            "generation_result": generation_result,
            "iteration": int(state.get("iteration", 0)) + 1,
        }

    def run_code(self, state: CodeLoopState) -> CodeLoopState:
        execution_result = self.executor.invoke({
            "generated_code": state.get("generated_code", ""),
            "code_path": state.get("code_path"),
            "data_path": state.get("data_path", ""),
            "sheet_name": state.get("sheet_name"),
            "smiles_column": state.get("smiles_column", "smiles"),
            "target_column": state.get("target_column", "boiling_point"),
            "python_executable": state.get("python_executable"),
        })
        final_execution = execution_result.get("execution_result", {}) or execution_result.get("final_output", {}).get("execution_result", {})
        code_path = execution_result.get("code_path") or execution_result.get("final_output", {}).get("code_path") or state.get("code_path")
        error = execution_result.get("error") or execution_result.get("final_output", {}).get("error") or ""
        return {
            "execution_result": final_execution,
            "code_path": code_path,
            "error": error,
            "latest_stdout": final_execution.get("stdout", ""),
            "latest_stderr": final_execution.get("stderr", ""),
            "latest_error": error or self._extract_error_message({"execution_result": final_execution}),
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
        stop_reason = self._build_stop_reason(repeated_error_count, int(state.get("iteration", 0)), int(state.get("max_iterations", 4)))
        return {
            "error_history": error_history,
            "repeated_error_count": repeated_error_count,
            "error": error_message,
            "latest_error": error_message,
            "stop_reason": stop_reason,
            "execution_result": state.get("execution_result", {}),
        }

    def debug_code(self, state: CodeLoopState) -> CodeLoopState:
        prompt = self._build_debug_prompt(state)
        try:
            fixed_code = run_text_response(self.client, self.model_name, prompt, "코드 디버그")
            fixed_code = sanitize_python_code(fixed_code)
        except RuntimeError:
            fixed_code = state.get("generated_code", "")
        if not fixed_code:
            fixed_code = state.get("generated_code", "")
        return {"generated_code": fixed_code, "iteration": int(state.get("iteration", 0)) + 1}

    def finalize(self, state: CodeLoopState) -> CodeLoopState:
        return {
            "final_output": {
                "generated_code": state.get("generated_code", ""),
                "code_path": state.get("code_path"),
                "generation_result": state.get("generation_result", {}),
                "execution_result": state.get("execution_result", {}),
                "iteration": state.get("iteration", 0),
                "max_iterations": state.get("max_iterations", 4),
                "repeated_error_count": state.get("repeated_error_count", 0),
                "stop_reason": state.get("stop_reason", ""),
                "error": state.get("error", ""),
                "latest_error": state.get("latest_error", ""),
                "latest_stdout": state.get("latest_stdout", ""),
                "latest_stderr": state.get("latest_stderr", ""),
                "error_history": state.get("error_history", []),
            }
        }

    def _route_after_initialize(self, state: CodeLoopState) -> str:
        return "run" if state.get("generated_code") else "generate"

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
        return "Unknown execution error"

    def _build_error_signature(self, message: str) -> str:
        normalized_lines: list[str] = []
        for line in [line.strip() for line in message.splitlines() if line.strip()]:
            normalized_line = re.sub(r'File ".*?"', 'File "<path>"', line)
            normalized_line = re.sub(r'line \d+', 'line <n>', normalized_line)
            normalized_line = re.sub(r'0x[0-9a-fA-F]+', '0x<addr>', normalized_line)
            normalized_lines.append(normalized_line)
        if not normalized_lines:
            return "unknown_error"
        return "\n".join(normalized_lines[-12:])

    def _build_stop_reason(self, repeated_error_count: int, iteration: int, max_iterations: int) -> str:
        if repeated_error_count >= 2:
            return "같은 에러가 2회 이상 반복되어 디버그 루프를 중단했습니다."
        if iteration >= max_iterations:
            return "최대 반복 횟수에 도달하여 디버그 루프를 중단했습니다."
        return ""

    def _build_debug_prompt(self, state: CodeLoopState) -> str:
        recent_errors = state.get("error_history", [])[-2:]
        return (
            "You are a Python debugging agent inside a LangGraph workflow.\n"
            "Fix the provided Python regression script so it runs successfully.\n"
            "Return raw Python source code only.\n"
            "Do not use markdown fences.\n"
            "Use paper_method_spec as the authoritative structured paper specification.\n"
            "Use paper_markdown as background context and model_anchor_summary as a cross-check only.\n"
            "Preserve the intended regression workflow.\n"
            "Use RDKit and sklearn.\n"
            "Prefer RDKit descriptors only when the paper specification leaves the feature details incomplete.\n"
            "Keep MAE, RMSE, MSE, and R2 metrics.\n"
            "Use the execution error history below as direct feedback for the fix.\n\n"
            f"Paper info:\n{state.get('raw_paper_info', {})}\n\n"
            f"Current code:\n{state.get('generated_code', '')}\n\n"
            f"Latest execution error:\n{self._extract_error_message(state)}\n\n"
            f"Recent error history:\n{recent_errors}\n"
        )
