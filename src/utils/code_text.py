from __future__ import annotations

import re


def sanitize_python_code(code: str) -> str:
    cleaned_code = code.strip()
    fenced_match = re.search(r"```(?:python)?\s*(.*?)```", cleaned_code, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        cleaned_code = fenced_match.group(1).strip()
    cleaned_code = re.sub(r"^```(?:python)?\s*", "", cleaned_code, flags=re.IGNORECASE)
    cleaned_code = re.sub(r"\s*```$", "", cleaned_code, flags=re.IGNORECASE)
    return cleaned_code.strip()
