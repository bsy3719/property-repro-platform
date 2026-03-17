from __future__ import annotations

__all__ = ["CodeGenerationAgent", "CodeGenerationState"]


def __getattr__(name: str):
    if name in {"CodeGenerationAgent", "CodeGenerationState"}:
        from .agent import CodeGenerationAgent
        from .schemas import CodeGenerationState

        exports = {
            "CodeGenerationAgent": CodeGenerationAgent,
            "CodeGenerationState": CodeGenerationState,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
