from __future__ import annotations

import re
from typing import Any

from src.utils.chemistry_features import analyze_feature_text

from .schemas import MODEL_ALIASES, MODEL_CLASS_NAMES


def parse_paper_spec(markdown: str) -> dict[str, Any]:
    rows = parse_markdown_table(markdown)
    row_map = {str(row.get("Topic", "")).strip().lower(): row for row in rows}
    combined_text = "\n".join([
        markdown,
        row_text(row_map.get("feature", {})),
        row_text(row_map.get("training", {})),
        row_text(row_map.get("hyperparameter", {})),
    ])
    return {
        "preprocessing": paper_preprocessing_text(combined_text),
        "feature": paper_feature_text(row_map.get("feature", {})),
        "model": paper_model_text(row_map.get("model", {})),
        "hyperparameters": paper_hyperparameter_text(row_map.get("hyperparameter", {})),
        "training": paper_training_text(row_map.get("training", {})),
        "metrics": extract_metrics(row_text(row_map.get("metrics", {}))),
    }


def parse_markdown_table(markdown: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in markdown.splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return []
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def row_text(row: dict[str, Any]) -> str:
    if not row:
        return ""
    values = [str(row.get("What Paper Did", "")).strip(), str(row.get("Key Values or Settings", "")).strip()]
    return " ; ".join([value for value in values if value])


def paper_preprocessing_text(text: str) -> str:
    lowered = text.lower()
    findings: list[str] = []
    if "invalid smiles" in lowered or "drop invalid" in lowered:
        findings.append("invalid_smiles=drop")
    if "duplicate" in lowered or "drop_duplicates" in lowered:
        findings.append("duplicates=drop")
    if "missing" in lowered or "imput" in lowered:
        findings.append("missing_features=handled")
    if "scale" in lowered or "scaler" in lowered or "normaliz" in lowered or "standard" in lowered:
        findings.append("scaling=True")
    return ", ".join(findings) if findings else "Not found"


def paper_feature_text(row: dict[str, Any]) -> str:
    text = row_text(row)
    analysis = analyze_feature_text(text)
    has_descriptor = bool(analysis["descriptor_names"] or analysis["count_feature_names"] or analysis["has_descriptor_signal"])
    has_morgan = bool(analysis["fingerprint_family"])
    if has_descriptor and has_morgan:
        return format_morgan_descriptor_text(text)
    if has_morgan:
        return format_morgan_text(text)
    if has_descriptor:
        return "RDKit descriptors"
    return "Not found"


def paper_model_text(row: dict[str, Any]) -> str:
    text = row_text(row)
    if not text:
        return "Not found"
    lowered = text.lower()
    for class_name in MODEL_CLASS_NAMES:
        if class_name.lower() in lowered:
            return class_name
    for alias, canonical_name in MODEL_ALIASES.items():
        if alias in lowered:
            return canonical_name
    return "Not found"


def paper_hyperparameter_text(row: dict[str, Any]) -> str:
    text = row_text(row)
    if not text:
        return "Not found"
    matches = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,;]+)", text)
    if not matches:
        return str(row.get("Key Values or Settings", "")).strip() or "Not found"
    ordered_items: list[str] = []
    seen_items: set[str] = set()
    for key, value in matches:
        item = f"{key}={value.strip()}"
        if item not in seen_items:
            seen_items.add(item)
            ordered_items.append(item)
    return ", ".join(ordered_items)


def paper_training_text(row: dict[str, Any]) -> str:
    text = row_text(row)
    if not text:
        return "Not found"
    lowered = text.lower()
    items: list[str] = []
    if "train_test_split" in lowered or "split" in lowered:
        items.append("split=train_test_split")
    test_size = extract_numeric_setting(text, ["test_size", "test size"])
    random_state = extract_numeric_setting(text, ["random_state", "random state", "seed"])
    if test_size is not None:
        items.append(f"test_size={test_size}")
    if random_state is not None:
        items.append(f"random_state={random_state}")
    return ", ".join(items) if items else "Not found"


def extract_metrics(text: str) -> dict[str, float]:
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
            try:
                metrics[metric_name] = float(match.group(1))
            except ValueError:
                continue
    return metrics


def extract_param(text: str, key: str) -> str | None:
    pattern = rf"{re.escape(key)}\s*[:=]?\s*([A-Za-z0-9_.-]+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def extract_numeric_setting(text: str, keys: list[str]) -> str | None:
    for key in keys:
        pattern = rf"{re.escape(key)}\s*[:=]?\s*(-?\d+(?:\.\d+)?)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def format_morgan_text(text: str) -> str:
    radius = extract_param(text, "radius")
    n_bits = extract_param(text, "n_bits") or extract_param(text, "nbits")
    extras = []
    if radius:
        extras.append(f"radius={radius}")
    if n_bits:
        extras.append(f"n_bits={n_bits}")
    return f"Morgan fingerprint ({', '.join(extras)})" if extras else "Morgan fingerprint"


def format_morgan_descriptor_text(text: str) -> str:
    radius = extract_param(text, "radius")
    n_bits = extract_param(text, "n_bits") or extract_param(text, "nbits")
    extras = []
    if radius:
        extras.append(f"radius={radius}")
    if n_bits:
        extras.append(f"n_bits={n_bits}")
    suffix = f" ({', '.join(extras)})" if extras else ""
    return f"Morgan fingerprint + RDKit descriptors{suffix}"
