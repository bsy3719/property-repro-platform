from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

REPORTS_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_comparison_table(paper_spec: dict[str, Any], execution_spec: dict[str, Any]) -> str:
    rows = [
        ["Item", "Paper", "Reproduced", "Difference / Notes"],
        build_text_row("Preprocessing", paper_spec, execution_spec),
        build_text_row("Feature", paper_spec, execution_spec),
        build_text_row("Model", paper_spec, execution_spec),
        build_text_row("Hyperparameters", paper_spec, execution_spec),
        build_text_row("Training", paper_spec, execution_spec),
    ]
    rows.extend(build_metric_rows(paper_spec.get("metrics", {}), execution_spec.get("metrics", {})))
    return markdown_table(rows)


def compose_report(
    execution_spec: dict[str, Any],
    comparison_table: str,
    analysis_markdown: str,
    summary_headline: str,
    summary_paragraphs: list[str],
    execution_final_output: dict[str, Any],
    generated_code_path: str,
) -> str:
    summary_lines = ""
    if summary_headline or summary_paragraphs:
        summary_lines = "## 재현 분석 요약\n"
        if summary_headline:
            summary_lines += f"{summary_headline}\n"
        if summary_paragraphs:
            summary_lines += "\n".join([f"- {paragraph}" for paragraph in summary_paragraphs if str(paragraph).strip()]) + "\n\n"

    return (
        "# 논문 대비 재현 결과 비교 보고서\n\n"
        "## 실행 요약\n"
        f"- 실행 상태: {execution_spec.get('status', 'unknown')}\n"
        f"- 반환 코드: {execution_spec.get('returncode', 'unknown')}\n"
        f"- 반복 횟수: {execution_final_output.get('iteration', 'unknown')} / {execution_final_output.get('max_iterations', 'unknown')}\n"
        f"- 중단 사유: {execution_final_output.get('stop_reason', '') or '없음'}\n"
        f"- 생성 코드 경로: {generated_code_path or 'Not available'}\n\n"
        f"{summary_lines}"
        "## 비교 표\n\n"
        f"{comparison_table}\n\n"
        f"{analysis_markdown}\n"
    )


def save_report(report_markdown: str) -> str:
    report_path = REPORTS_DIR / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.write_text(report_markdown, encoding='utf-8')
    return str(report_path)


def fallback_analysis(paper_spec: dict[str, Any], execution_spec: dict[str, Any], warning: str = "") -> str:
    lines = [
        "## Overall Assessment",
        "논문 요약 정보와 최종 재현 실행 결과를 기준으로 비교했습니다.",
    ]
    if warning:
        lines.append(f"- {warning}")
    lines.extend([
        "",
        "## Methodology Differences",
        f"- 전처리 비교: 논문={paper_spec.get('preprocessing', 'Not found')} / 재현={execution_spec.get('preprocessing', 'Unknown from generated code')}",
        f"- feature 비교: 논문={paper_spec.get('feature', 'Not found')} / 재현={execution_spec.get('feature', 'Unknown from generated code')}",
        f"- 모델 비교: 논문={paper_spec.get('model', 'Not found')} / 재현={execution_spec.get('model', 'Unknown from generated code')}",
        f"- 하이퍼파라미터 비교: 논문={paper_spec.get('hyperparameters', 'Not found')} / 재현={execution_spec.get('hyperparameters', 'Unknown from generated code')}",
        f"- 학습 비교: 논문={paper_spec.get('training', 'Not found')} / 재현={execution_spec.get('training', 'Unknown from generated code')}",
        "",
        "## Metric Differences",
        f"- 논문 metric: {paper_spec.get('metrics', {})}",
        f"- 재현 metric: {execution_spec.get('metrics', {})}",
        "",
        "## Likely Causes",
        "- 논문 metric 또는 방법론 정보가 일부 누락되었을 수 있습니다.",
        "- 전처리, feature 생성, split/random_state, invalid SMILES 처리 차이가 metric 차이에 영향을 줄 수 있습니다.",
        "- 논문 세부 설정이 없는 경우 재현 코드는 기본값을 사용했을 가능성이 있습니다.",
    ])
    return "\n".join(lines)


def build_text_row(item: str, paper_spec: dict[str, Any], execution_spec: dict[str, Any]) -> list[str]:
    key = item.lower() if item != "Hyperparameters" else "hyperparameters"
    if item == "Preprocessing":
        key = "preprocessing"
    elif item == "Feature":
        key = "feature"
    elif item == "Model":
        key = "model"
    elif item == "Training":
        key = "training"
    paper_value = paper_spec.get(key, "Not found")
    execution_value = execution_spec.get(key, "Unknown from generated code")
    return [item, paper_value, execution_value, text_difference_note(paper_value, execution_value)]


def build_metric_rows(paper_metrics: dict[str, Any], execution_metrics: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for metric_name in ["MAE", "RMSE", "MSE", "R2"]:
        paper_value = paper_metrics.get(metric_name)
        execution_value = execution_metrics.get(metric_name)
        rows.append([
            metric_name,
            format_metric_value(paper_value),
            format_metric_value(execution_value),
            metric_difference(paper_value, execution_value),
        ])
    return rows


def text_difference_note(paper: str, reproduced: str) -> str:
    if not paper or paper == "Not found":
        return "논문 정보 부족"
    if not reproduced or reproduced == "Unknown from generated code":
        return "재현 코드에서 자동 추출 실패"
    if paper.strip().lower() == reproduced.strip().lower():
        return "일치"
    return "차이 있음"


def metric_difference(paper_val: Any, reproduced_val: Any) -> str:
    if paper_val is None:
        return "Not available"
    if reproduced_val is None:
        return "재현 metric 없음"
    try:
        return f"{float(reproduced_val) - float(paper_val):+.4f}"
    except (TypeError, ValueError):
        return "계산 불가"


def format_metric_value(value: Any) -> str:
    if value is None:
        return "Not available"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(rows: list[list[str]]) -> str:
    safe_rows = [[str(cell).replace("|", "/") for cell in row] for row in rows]
    header = "| " + " | ".join(safe_rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(safe_rows[0])) + " |"
    body = ["| " + " | ".join(row) + " |" for row in safe_rows[1:]]
    return "\n".join([header, separator, *body])
