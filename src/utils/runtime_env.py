from __future__ import annotations

import os
import sys
from pathlib import Path


def _python_binary_for_prefix(prefix: Path) -> Path:
    if os.name == "nt":
        return prefix / "python.exe"
    return prefix / "bin" / "python"


def get_preferred_python_executable(preferred_env_name: str = "paper2property") -> str:
    current_python = Path(sys.executable).resolve()
    if f"/envs/{preferred_env_name}/" in str(current_python):
        return str(current_python)

    conda_prefix = os.getenv("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix).resolve()
        candidate = _python_binary_for_prefix(prefix)
        if prefix.name == preferred_env_name and candidate.exists():
            return str(candidate)

    for root in [Path.home() / "miniconda3" / "envs", Path.home() / "anaconda3" / "envs"]:
        candidate = _python_binary_for_prefix(root / preferred_env_name)
        if candidate.exists():
            return str(candidate)

    return str(current_python)


def resolve_project_python_executable(requested: str | None = None, preferred_env_name: str = "paper2property") -> str:
    preferred = get_preferred_python_executable(preferred_env_name=preferred_env_name)
    if requested:
        requested_path = Path(requested).expanduser()
        if requested_path.exists() and f"/envs/{preferred_env_name}/" in str(requested_path.resolve()):
            return str(requested_path.resolve())
    return preferred
