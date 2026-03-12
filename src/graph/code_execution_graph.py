from __future__ import annotations

from typing import Any

from src.agents.code_execution_agent import CodeExecutionAgent, CodeExecutionState


def build_code_execution_agent() -> CodeExecutionAgent:
    return CodeExecutionAgent()


def run_code_execution(payload: dict[str, Any]) -> CodeExecutionState:
    agent = build_code_execution_agent()
    return agent.invoke(payload)
