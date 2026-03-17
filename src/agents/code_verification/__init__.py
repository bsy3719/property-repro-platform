from __future__ import annotations

__all__ = ["CodeVerificationAgent", "CodeVerificationState", "validate_code_contract"]


def __getattr__(name: str):
    if name in {"CodeVerificationAgent", "CodeVerificationState"}:
        from .agent import CodeVerificationAgent
        from .schemas import CodeVerificationState

        exports = {
            "CodeVerificationAgent": CodeVerificationAgent,
            "CodeVerificationState": CodeVerificationState,
        }
        return exports[name]
    if name == "validate_code_contract":
        from .validation import validate_code_contract

        return validate_code_contract
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
