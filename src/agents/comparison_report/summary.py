from __future__ import annotations

from typing import Any

from src.utils import normalize_paper_method_spec, paper_method_spec_to_comparison_spec

from .execution_spec import parse_execution_spec

METRIC_ORDER = ["MAE", "RMSE", "MSE", "R2"]


def build_reproduction_summary(
    paper_method_spec: dict[str, Any],
    execution_final_output: dict[str, Any],
    generated_code: str = "",
    generated_code_path: str = "",
) -> dict[str, Any]:
    normalized_paper_spec = normalize_paper_method_spec(paper_method_spec or {})
    paper_spec = paper_method_spec_to_comparison_spec(normalized_paper_spec)
    execution_spec = parse_execution_spec(execution_final_output, generated_code, generated_code_path)
    metric_rows = _build_metric_rows(paper_spec.get("metrics", {}), execution_spec.get("metrics", {}))

    headline, status = _build_headline(metric_rows, execution_spec)
    paragraphs = [_build_metric_overview(metric_rows)]

    methodology_line = _build_methodology_line(normalized_paper_spec, execution_spec)
    if methodology_line:
        paragraphs.append(methodology_line)

    feature_line = _build_feature_line(paper_spec.get("feature"), execution_spec.get("feature"))
    if feature_line:
        paragraphs.append(feature_line)

    return {
        "status": status,
        "headline": headline,
        "paragraphs": [paragraph for paragraph in paragraphs if paragraph],
        "metric_rows": metric_rows,
        "paper_metrics": paper_spec.get("metrics", {}),
        "reproduced_metrics": execution_spec.get("metrics", {}),
        "paper_feature": paper_spec.get("feature", "Not found"),
        "reproduced_feature": execution_spec.get("feature", "Unknown from generated code"),
        "paper_training": paper_spec.get("training", "Not found"),
        "reproduced_training": execution_spec.get("training", "Unknown from generated code"),
        "paper_hyperparameters": paper_spec.get("hyperparameters", "Not found"),
        "reproduced_hyperparameters": execution_spec.get("hyperparameters", "Unknown from generated code"),
    }


def _build_metric_rows(paper_metrics: dict[str, Any], reproduced_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name in METRIC_ORDER:
        paper_value = _to_float(paper_metrics.get(metric_name))
        reproduced_value = _to_float(reproduced_metrics.get(metric_name))
        if paper_value is None and reproduced_value is None:
            continue

        diff = None
        relative_error_pct = None
        signed_relative_error_pct = None
        severity = "unknown"
        if paper_value is not None and reproduced_value is not None:
            diff = reproduced_value - paper_value
            denominator = abs(paper_value) if paper_value not in {0.0, -0.0} else 1.0
            relative_error_pct = abs(diff) / denominator * 100.0
            signed_relative_error_pct = diff / denominator * 100.0
            severity = _metric_severity(relative_error_pct)

        rows.append(
            {
                "metric": metric_name,
                "paper": paper_value,
                "reproduced": reproduced_value,
                "diff": diff,
                "relative_error_pct": relative_error_pct,
                "signed_relative_error_pct": signed_relative_error_pct,
                "severity": severity,
            }
        )
    return rows


def _build_headline(metric_rows: list[dict[str, Any]], execution_spec: dict[str, Any]) -> tuple[str, str]:
    if execution_spec.get("status") != "success":
        return "실행이 완료되지 않아 재현값을 안정적으로 비교할 수 없습니다.", "failed"

    comparable_rows = [row for row in metric_rows if row.get("relative_error_pct") is not None]
    if not comparable_rows:
        return "재현 코드는 실행되었지만 논문 보고값과 직접 비교할 수 있는 지표가 충분하지 않습니다.", "limited"

    good_count = sum(1 for row in comparable_rows if row["severity"] == "good")
    acceptable_count = sum(1 for row in comparable_rows if row["severity"] == "acceptable")

    if good_count == len(comparable_rows):
        return "전반적으로 성공적인 재현으로 평가됩니다.", "good"
    if good_count + acceptable_count == len(comparable_rows):
        return "전반적으로 양호한 재현이지만 일부 지표에서 눈에 띄는 편차가 있습니다.", "partial"
    return "재현은 일부 성과가 있지만 논문 보고값과의 차이가 비교적 큽니다.", "poor"


def _build_metric_overview(metric_rows: list[dict[str, Any]]) -> str:
    comparable_rows = [row for row in metric_rows if row.get("relative_error_pct") is not None]
    if not comparable_rows:
        return ""

    primary = _select_primary_metric(comparable_rows)
    major_gap = max(comparable_rows, key=lambda row: row.get("relative_error_pct", -1.0))

    sentences = [
        f"{primary['metric']} 차이는 {_format_signed_percent(primary['signed_relative_error_pct'])}로 {_tolerance_phrase(primary['severity'])}."
    ]
    if major_gap["metric"] != primary["metric"] and (major_gap.get("relative_error_pct") or 0.0) >= 8.0:
        sentences.append(
            f"{major_gap['metric']}의 경우 {_format_signed_percent(major_gap['signed_relative_error_pct'])} 편차가 발생했습니다."
        )
    return " ".join(sentences)


def _build_methodology_line(paper_spec: dict[str, Any], execution_spec: dict[str, Any]) -> str:
    notes: list[str] = []

    paper_training = paper_spec.get("training", {})
    missing_split_fields = [
        field_name
        for field_name in ["test_size", "random_state"]
        if paper_training.get(field_name) in {None, "", "Not found"}
    ]
    if missing_split_fields:
        notes.append(
            f"논문이 {', '.join(missing_split_fields)}를 명시하지 않아 재현 코드에서는 기본 분할 설정을 사용했습니다."
        )

    paper_hyperparameters = paper_spec.get("hyperparameters", {}).get("values", {})
    if not paper_hyperparameters:
        notes.append("논문에 세부 하이퍼파라미터가 충분히 보고되지 않아 기본값 또는 안전한 보완값을 사용했을 가능성이 있습니다.")

    execution_training_text = str(execution_spec.get("training", ""))
    if "unknown" in execution_training_text.lower():
        notes.append("학습 설정 일부가 코드 결과에서 완전히 추출되지 않아 비교 해석에 제한이 있습니다.")

    return " ".join(notes[:2])


def _build_feature_line(paper_feature: str, reproduced_feature: str) -> str:
    paper_summary = _summarize_feature_text(paper_feature)
    reproduced_summary = _summarize_feature_text(reproduced_feature)
    normalized_paper = _normalize_label(paper_summary)
    normalized_reproduced = _normalize_label(reproduced_summary)
    if not normalized_paper or normalized_paper == "not found":
        return ""
    if not normalized_reproduced or normalized_reproduced == "unknown from generated code":
        return f"Feature 조합은 논문에서 {paper_summary}로 보고되었지만, 현재 실행 결과만으로는 구현 feature를 완전히 추적하지 못했습니다."
    if normalized_paper == normalized_reproduced:
        return f"Feature 조합 ({paper_summary})의 재현은 보고된 내용과 일치합니다."
    return f"Feature 구성은 논문={paper_summary}, 재현={reproduced_summary}로 일부 차이가 있습니다."


def _summarize_feature_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Not found"
    if text == "Unknown from generated code":
        return text

    lowered = text.lower()
    parts: list[str] = []

    if "tabular/rdkit chemistry features" in lowered:
        parts.append("tabular/RDKit chemistry features")
    elif "rdkit descriptors" in lowered:
        parts.append("RDKit descriptors")
    elif "tabular chemistry features" in lowered:
        parts.append("tabular chemistry features")

    if "count" in lowered:
        parts.append("atom/bond count features")

    feature_count = _extract_feature_count(text)
    if feature_count is not None:
        parts.append(f"feature_count={feature_count}")

    class_labels = _extract_list_count(text, "class_labels")
    if class_labels:
        parts.append(f"class labels {class_labels}개")

    if not parts:
        compact = text.split("{", 1)[0].strip()
        compact = compact.split(";", 1)[0].strip()
        return compact or "Not found"

    return " + ".join(_dedupe_preserve_order(parts))


def _extract_feature_count(text: str) -> int | None:
    import re

    match = re.search(r"feature_count\s*=\s*(\d+)", text, re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_list_count(text: str, key: str) -> int:
    import re

    match = re.search(rf"{re.escape(key)}=\[([^\]]*)\]", text, re.IGNORECASE)
    if not match:
        return 0
    items = [item.strip() for item in match.group(1).split(",") if item.strip() and item.strip() != "..."]
    return len(items)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _select_primary_metric(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    for metric_name in ["MAE", "RMSE", "R2", "MSE"]:
        for row in metric_rows:
            if row["metric"] == metric_name:
                return row
    return metric_rows[0]


def _metric_severity(relative_error_pct: float) -> str:
    if relative_error_pct <= 5.0:
        return "good"
    if relative_error_pct <= 20.0:
        return "acceptable"
    return "poor"


def _tolerance_phrase(severity: str) -> str:
    if severity == "good":
        return "허용 범위 내에 있습니다"
    if severity == "acceptable":
        return "대체로 수용 가능한 수준입니다"
    return "유의미한 차이가 있습니다"


def _format_signed_percent(value: float | None) -> str:
    if value is None:
        return "비교 불가"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().lower()


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
