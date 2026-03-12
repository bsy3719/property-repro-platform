from __future__ import annotations

from typing import Any

from src.agents.code_loop_agent import CodeGenerationRunDebugAgent, CodeLoopState


def build_code_loop_agent(model_name: str = "gpt-5.2") -> CodeGenerationRunDebugAgent:
    return CodeGenerationRunDebugAgent(model_name=model_name)


def run_code_loop(payload: dict[str, Any]) -> CodeLoopState:
    agent = build_code_loop_agent()
    return agent.invoke(payload)
