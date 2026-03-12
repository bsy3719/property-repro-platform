from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.utils import sanitize_python_code

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATED_CODE_DIR = PROJECT_ROOT / "artifacts" / "generated_code"
GENERATED_CODE_DIR.mkdir(parents=True, exist_ok=True)


class CodeExecutionState(TypedDict, total=False):
    generated_code: str
    code_path: str
    data_path: str
    sheet_name: str | None
    smiles_column: str
    target_column: str
    python_executable: str
    command: list[str]
    execution_result: dict[str, Any]
    final_output: dict[str, Any]
    error: str


class CodeExecutionAgent:
    """LangGraph agent that saves and executes generated regression code."""

    def __init__(self) -> None:
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(CodeExecutionState)
        graph.add_node("prepare_script", self.prepare_script)
        graph.add_node("build_command", self.build_command)
        graph.add_node("execute_script", self.execute_script)
        graph.add_node("finalize_output", self.finalize_output)
        graph.set_entry_point("prepare_script")
        graph.add_edge("prepare_script", "build_command")
        graph.add_edge("build_command", "execute_script")
        graph.add_edge("execute_script", "finalize_output")
        graph.add_edge("finalize_output", END)
        return graph.compile()

    def invoke(self, state: CodeExecutionState) -> CodeExecutionState:
        return self.graph.invoke(state)

    def prepare_script(self, state: CodeExecutionState) -> CodeExecutionState:
        generated_code = sanitize_python_code(state.get("generated_code", ""))
        code_path = state.get("code_path")
        if code_path and generated_code:
            script_path = Path(code_path)
            script_path.write_text(generated_code, encoding="utf-8")
        elif code_path:
            script_path = Path(code_path)
        else:
            if not generated_code:
                return {"error": "실행할 generated_code가 없습니다."}
            script_path = GENERATED_CODE_DIR / "generated_regression_latest.py"
            script_path.write_text(generated_code, encoding="utf-8")

        if not script_path.exists():
            return {"error": f"실행 스크립트를 찾을 수 없습니다: {script_path}"}
        return {"code_path": str(script_path), "generated_code": generated_code}

    def build_command(self, state: CodeExecutionState) -> CodeExecutionState:
        if state.get("error"):
            return state
        command = [
            state.get("python_executable") or sys.executable,
            state.get("code_path", ""),
            "--data-path",
            state.get("data_path", ""),
            "--smiles-col",
            state.get("smiles_column", "smiles"),
            "--target-col",
            state.get("target_column", "boiling_point"),
        ]
        sheet_name = state.get("sheet_name")
        if sheet_name:
            command.extend(["--sheet-name", str(sheet_name)])
        return {"python_executable": command[0], "command": command}

    def execute_script(self, state: CodeExecutionState) -> CodeExecutionState:
        if state.get("error"):
            return state
        command = state.get("command", [])
        if not command:
            return {"error": "실행 command를 만들지 못했습니다."}

        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=1800,
            )
        except Exception as exc:
            return {"error": f"코드 실행 중 예외가 발생했습니다: {exc}"}

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        parsed_output = self._parse_json_output(stdout)
        execution_result = {
            "status": "success" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "parsed_output": parsed_output,
            "metrics": parsed_output.get("metrics", {}) if isinstance(parsed_output, dict) else {},
            "command": command,
        }
        return {"execution_result": execution_result}

    def finalize_output(self, state: CodeExecutionState) -> CodeExecutionState:
        return {
            "final_output": {
                "code_path": state.get("code_path"),
                "execution_result": state.get("execution_result", {}),
                "error": state.get("error"),
            }
        }

    def _parse_json_output(self, stdout: str) -> dict[str, Any]:
        text = stdout.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start_index = text.find("{")
            end_index = text.rfind("}")
            if start_index == -1 or end_index == -1 or end_index <= start_index:
                return {"raw_output": text}
            candidate = text[start_index : end_index + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return {"raw_output": text}
