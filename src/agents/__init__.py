from __future__ import annotations

__all__ = ["CodeGenerationAgent", "CodeGenerationState"]


def __getattr__(name: str):
    if name in {"CodeGenerationAgent", "CodeGenerationState"}:
        from src.agents.code_generation_agent import CodeGenerationAgent, CodeGenerationState

        exports = {
            "CodeGenerationAgent": CodeGenerationAgent,
            "CodeGenerationState": CodeGenerationState,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
