from __future__ import annotations

from typing import Any

from src.agents.code_generation_agent import CodeGenerationAgent, CodeGenerationState


def build_code_generation_agent(model_name: str = "gpt-5.2") -> CodeGenerationAgent:
    return CodeGenerationAgent(model_name=model_name)


def run_code_generation(raw_paper_info: dict[str, Any] | str) -> CodeGenerationState:
    agent = build_code_generation_agent()
    return agent.invoke({"raw_paper_info": raw_paper_info})
