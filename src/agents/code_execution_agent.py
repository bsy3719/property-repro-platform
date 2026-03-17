from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.utils import sanitize_python_code
from src.utils.runtime_env import resolve_project_python_executable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATED_CODE_DIR = PROJECT_ROOT / "artifacts" / "generated_code"
GENERATED_CODE_DIR.mkdir(parents=True, exist_ok=True)
METRIC_ALIASES = {
    "mae": "MAE",
    "rmse": "RMSE",
    "mse": "MSE",
    "r2": "R2",
    "r^2": "R2",
    "r²": "R2",
}


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
        python_executable = resolve_project_python_executable(state.get("python_executable"))
        command = [
            python_executable,
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
            return self._normalize_execution_payload(json.loads(text))
        except json.JSONDecodeError:
            start_index = text.find("{")
            end_index = text.rfind("}")
            if start_index == -1 or end_index == -1 or end_index <= start_index:
                return self._normalize_execution_payload({"raw_output": text})
            candidate = text[start_index : end_index + 1]
            try:
                return self._normalize_execution_payload(json.loads(candidate))
            except json.JSONDecodeError:
                return self._normalize_execution_payload({"raw_output": text})

    def _normalize_execution_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}

        normalized = dict(payload)
        normalized["metrics"] = self._extract_metrics(normalized)
        normalized["y_test"] = self._coerce_float_list(normalized.get("y_test"))
        normalized["y_pred"] = self._coerce_float_list(normalized.get("y_pred"))
        assumptions = normalized.get("assumptions", [])
        normalized["assumptions"] = assumptions if isinstance(assumptions, list) else []
        return normalized

    def _extract_metrics(self, payload: dict[str, Any]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for source in [
            payload,
            payload.get("metrics"),
            payload.get("metric"),
            payload.get("results", {}),
            payload.get("results", {}).get("metrics") if isinstance(payload.get("results"), dict) else {},
            payload.get("parsed_output", {}),
        ]:
            metrics.update(self._coerce_metric_map(source))

        if metrics:
            return metrics

        raw_output = str(payload.get("raw_output", ""))
        return self._extract_metrics_from_text(raw_output)

    def _coerce_metric_map(self, source: Any) -> dict[str, float]:
        if not isinstance(source, dict):
            return {}

        metrics: dict[str, float] = {}
        for raw_key, raw_value in source.items():
            canonical_key = METRIC_ALIASES.get(str(raw_key).strip().lower())
            if not canonical_key:
                continue
            numeric_value = self._to_float(raw_value)
            if numeric_value is not None:
                metrics[canonical_key] = numeric_value
        return metrics

    def _extract_metrics_from_text(self, text: str) -> dict[str, float]:
        metrics: dict[str, float] = {}
        patterns = {
            "MAE": r"\bMAE\b\s*[:=]?\s*(-?\d+(?:\.\d+)?)",
            "RMSE": r"\bRMSE\b\s*[:=]?\s*(-?\d+(?:\.\d+)?)",
            "MSE": r"\bMSE\b\s*[:=]?\s*(-?\d+(?:\.\d+)?)",
            "R2": r"(?:\bR\^?2\b|R²)\s*[:=]?\s*(-?\d+(?:\.\d+)?)",
        }
        for metric_name, pattern in patterns.items():
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                numeric_value = self._to_float(match.group(1))
                if numeric_value is not None:
                    metrics[metric_name] = numeric_value
        return metrics

    def _coerce_float_list(self, value: Any) -> list[float]:
        if not isinstance(value, list):
            return []
        result: list[float] = []
        for item in value:
            numeric_value = self._to_float(item)
            if numeric_value is not None:
                result.append(numeric_value)
        return result

    def _to_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
