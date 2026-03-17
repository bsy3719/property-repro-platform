from __future__ import annotations

import hashlib
import json
import sys
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.column_selection_agent import ColumnSelectionAgent
from src.agents.comparison_report.summary import build_reproduction_summary
from src.graph.code_loop_graph import run_code_loop
from src.graph.comparison_report_graph import run_comparison_report
from src.graph.paper_parsing_graph import run_paper_parsing
from src.graph.rag_graph import run_rag_pipeline
from src.utils import has_meaningful_paper_method_spec, normalize_paper_method_spec, paper_method_spec_to_comparison_spec
from src.utils.runtime_env import resolve_project_python_executable

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MARKDOWN_CACHE_DIR = ARTIFACTS_DIR / "markdown_cache"
GENERATED_CODE_DIR = ARTIFACTS_DIR / "generated_code"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
for directory in [RAW_DATA_DIR, ARTIFACTS_DIR, MARKDOWN_CACHE_DIR, GENERATED_CODE_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def format_run_timestamp(dt: datetime | None = None) -> str:
    return (dt or datetime.now()).strftime("%y%m%d_%H%M%S")


def save_run_artifact(directory: Path, filename: str, content: str) -> str:
    path = directory / filename
    path.write_text(content, encoding="utf-8")
    return str(path)


def build_reproduction_report_markdown(
    session: dict[str, Any],
    reproduction_summary: dict[str, Any],
    final_output: dict[str, Any],
    generated_code_path: str,
    final_code_path: str,
    run_timestamp: str,
) -> str:
    comparison_report = final_output.get("comparison_report", {}) if isinstance(final_output, dict) else {}
    report_markdown = str(comparison_report.get("report_markdown", "")).strip() if isinstance(comparison_report, dict) else ""
    if report_markdown:
        return report_markdown if report_markdown.endswith("\n") else report_markdown + "\n"

    execution_result = final_output.get("execution_result", {}) if isinstance(final_output, dict) else {}
    paper_method_spec = normalize_paper_method_spec(session.get("paper_method_spec", {}))
    paper_spec = paper_method_spec_to_comparison_spec(paper_method_spec)
    generation_output = final_output.get("generation_result", {}).get("final_output", {}) if isinstance(final_output, dict) else {}
    code_spec = generation_output.get("code_spec", {}) if isinstance(generation_output, dict) else {}
    implementation_feature = code_spec.get("feature_pipeline", {}) if isinstance(code_spec, dict) else {}
    implementation_preprocessing = code_spec.get("preprocessing_pipeline", {}) if isinstance(code_spec, dict) else {}
    implementation_model = code_spec.get("model", {}) if isinstance(code_spec, dict) else {}
    implementation_training = code_spec.get("training", {}) if isinstance(code_spec, dict) else {}
    assumptions = _collect_report_assumptions(final_output)
    success_line = _build_reproduction_success_line(reproduction_summary, execution_result)
    lines = [
        "# 재현 분석 요약 보고서",
        "",
        f"- 생성 시각: {run_timestamp}",
        "",
        "## 논문 정보",
        f"- 논문 파일: {session.get('pdf_name', '알 수 없음')}",
        f"- 대상 물성: boiling point",
        f"- 논문 마크다운 경로: {session.get('paper_markdown_path', '없음') or '없음'}",
        f"- 모델 선택 근거: {_display_text(paper_method_spec.get('selection_basis', {}).get('summary'))}",
        "",
        "## 데이터 정보",
        f"- 데이터 파일: {session.get('data_name', '알 수 없음')}",
        f"- 시트 이름: {session.get('data_sheet_name', '없음') or '없음'}",
        f"- 행/열 수: {session.get('num_rows', 'unknown')} rows / {session.get('num_columns', 'unknown')} columns",
        f"- SMILES 컬럼: {session.get('smiles_column', '없음') or '없음'}",
        f"- 타깃 컬럼: {session.get('target_column', '없음') or '없음'}",
        "",
        "## Feature 구성",
        f"- 논문 기준 feature: {_display_text(paper_spec.get('feature'))}",
        f"- 구현 feature pipeline: {_format_feature_pipeline(implementation_feature)}",
        f"- 전처리: {_format_preprocessing_pipeline(paper_method_spec.get('preprocessing', {}), implementation_preprocessing)}",
        "",
        "## 모델 정보",
        f"- 논문 기준 모델: {_display_text(paper_spec.get('model'))}",
        f"- 구현 모델: {_format_model_spec(implementation_model)}",
        f"- 하이퍼파라미터: {_format_hyperparameter_spec(paper_method_spec.get('hyperparameters', {}), implementation_model)}",
        "",
        "## 학습 설정",
        f"- 논문 기준 학습 설정: {_display_text(paper_spec.get('training'))}",
        f"- 구현 학습 설정: {_format_training_spec(implementation_training)}",
        "",
        "## 재현 성공 여부",
        f"- 판정: {success_line}",
        f"- 실행 상태: {execution_result.get('status', 'unknown')}",
        "",
        "## 실행 정보",
        f"- 실행 상태: {execution_result.get('status', 'unknown')}",
        f"- 반환 코드: {execution_result.get('returncode', 'unknown')}",
        f"- 반복 횟수: {final_output.get('iteration', 'unknown')} / {final_output.get('max_iterations', 'unknown')}",
        f"- 생성 코드 경로: {generated_code_path or '없음'}",
        f"- 최종 코드 경로: {final_code_path or '없음'}",
        f"- 검증 상태: {final_output.get('verification_status', 'unknown') or 'unknown'}",
        f"- 검증 이슈 수: {final_output.get('verification_issue_count', 0)}",
    ]
    verification_report_path = final_output.get("verification_report_path", "")
    if verification_report_path:
        lines.append(f"- 검증 리포트 경로: {verification_report_path}")

    headline = str(reproduction_summary.get("headline", "")).strip()
    paragraphs = [str(paragraph).strip() for paragraph in reproduction_summary.get("paragraphs", []) if str(paragraph).strip()]
    if headline or paragraphs:
        lines.extend(["", "## 재현 분석 요약"])
        if headline:
            lines.append(headline)
        lines.extend([f"- {paragraph}" for paragraph in paragraphs])

    paper_metrics = reproduction_summary.get("paper_metrics", {}) or {}
    reproduced_metrics = reproduction_summary.get("reproduced_metrics", {}) or {}
    metric_rows = reproduction_summary.get("metric_rows", []) or []
    if paper_metrics or reproduced_metrics or metric_rows:
        lines.extend([
            "",
            "## 지표 비교",
            "| 지표 | 논문 보고값 | 재현값 | 상대 오차(%) |",
            "| --- | --- | --- | --- |",
        ])
        for row in metric_rows:
            lines.append(
                "| {metric} | {paper} | {reproduced} | {relative} |".format(
                    metric=row.get("metric", "—"),
                    paper=_format_report_metric_value(row.get("paper")),
                    reproduced=_format_report_metric_value(row.get("reproduced")),
                    relative=_format_report_percent(row.get("signed_relative_error_pct")),
                )
            )

    if assumptions:
        lines.extend(["", "## 가정사항"])
        lines.extend([f"- {assumption}" for assumption in assumptions])
    else:
        lines.extend(["", "## 가정사항", "- 별도로 기록된 가정사항이 없습니다."])

    return "\n".join(lines) + "\n"


def persist_run_outputs(
    session: dict[str, Any],
    final_output: dict[str, Any],
    reproduction_summary: dict[str, Any],
    run_timestamp: str,
) -> dict[str, str]:
    generation_result = final_output.get("generation_result", {}) if isinstance(final_output, dict) else {}
    initial_generated_code = (
        generation_result.get("generated_code")
        or generation_result.get("final_output", {}).get("generated_code", "")
        or ""
    )
    final_code = final_output.get("generated_code", "") or ""

    generated_code_path = save_run_artifact(
        GENERATED_CODE_DIR,
        f"generated_code_{run_timestamp}.py",
        initial_generated_code or final_code,
    )
    final_code_path = save_run_artifact(
        GENERATED_CODE_DIR,
        f"final_code_{run_timestamp}.py",
        final_code or initial_generated_code,
    )
    report_markdown = build_reproduction_report_markdown(
        session=session,
        reproduction_summary=reproduction_summary,
        final_output=final_output,
        generated_code_path=generated_code_path,
        final_code_path=final_code_path,
        run_timestamp=run_timestamp,
    )
    reproduction_report_path = save_run_artifact(
        REPORTS_DIR,
        f"reproduction_report_{run_timestamp}.md",
        report_markdown,
    )
    comparison_report = final_output.get("comparison_report", {}) if isinstance(final_output, dict) else {}
    comparison_report_path = ""
    if isinstance(comparison_report, dict):
        comparison_report_path = str(comparison_report.get("report_path", "") or "")

    session["saved_artifacts"] = {
        "generated_code_path": generated_code_path,
        "final_code_path": final_code_path,
        "reproduction_report_path": reproduction_report_path,
        "comparison_report_path": comparison_report_path,
    }
    return session["saved_artifacts"]


def _format_report_metric_value(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_report_percent(value: Any) -> str:
    if value is None:
        return "—"
    try:
        signed_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    sign = "+" if signed_value > 0 else ""
    return f"{sign}{signed_value:.1f}"


def _display_text(value: Any, default: str = "Not found") -> str:
    text = str(value or "").strip()
    return text if text else default


def _format_feature_pipeline(feature_pipeline: dict[str, Any]) -> str:
    if not isinstance(feature_pipeline, dict) or not feature_pipeline:
        return "Not found"
    parts: list[str] = []
    for key in [
        "feature_source",
        "method",
        "exact_smiles_features",
        "fingerprint_family",
        "radius",
        "n_bits",
        "descriptor_names",
        "count_feature_names",
        "retained_input_features",
        "derived_feature_names",
        "class_label_names",
    ]:
        value = feature_pipeline.get(key)
        if value is None or value == "" or value == [] or value == {}:
            continue
        parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "Not found"


def _format_preprocessing_pipeline(paper_preprocessing: dict[str, Any], implementation_preprocessing: dict[str, Any]) -> str:
    implementation_parts: list[str] = []
    if isinstance(implementation_preprocessing, dict):
        for key in ["invalid_smiles", "missing_target", "missing_features", "duplicates", "scaling"]:
            value = implementation_preprocessing.get(key)
            if value is None or value == "" or value == [] or value == {}:
                continue
            implementation_parts.append(f"{key}={value}")
    implementation_label = ", ".join(implementation_parts) if implementation_parts else "Not found"
    paper_label_parts: list[str] = []
    if isinstance(paper_preprocessing, dict):
        for key in ["invalid_smiles", "missing_target", "missing_features", "duplicates", "scaling"]:
            value = paper_preprocessing.get(key)
            if value in {None, "", "Not found"}:
                continue
            paper_label_parts.append(f"{key}={value}")
    paper_label = ", ".join(paper_label_parts) if paper_label_parts else "Not found"
    return f"논문={paper_label} / 구현={implementation_label}"


def _format_model_spec(model_spec: dict[str, Any]) -> str:
    if not isinstance(model_spec, dict) or not model_spec:
        return "Not found"
    name = _display_text(model_spec.get("name"), default="")
    hyperparameters = model_spec.get("hyperparameters", {})
    if isinstance(hyperparameters, dict) and hyperparameters:
        return f"{name} ({', '.join(f'{key}={value}' for key, value in hyperparameters.items())})"
    return name or "Not found"


def _format_hyperparameter_spec(paper_hyperparameters: dict[str, Any], implementation_model: dict[str, Any]) -> str:
    paper_values = paper_hyperparameters.get("values", {}) if isinstance(paper_hyperparameters, dict) else {}
    impl_values = implementation_model.get("hyperparameters", {}) if isinstance(implementation_model, dict) else {}
    paper_label = ", ".join(f"{key}={value}" for key, value in paper_values.items()) if paper_values else "Not found"
    impl_label = ", ".join(f"{key}={value}" for key, value in impl_values.items()) if impl_values else "Not found"
    return f"논문={paper_label} / 구현={impl_label}"


def _format_training_spec(training_spec: dict[str, Any]) -> str:
    if not isinstance(training_spec, dict) or not training_spec:
        return "Not found"
    parts = []
    for key in ["split_strategy", "test_size", "random_state"]:
        value = training_spec.get(key)
        if value is None or value == "" or value == [] or value == {}:
            continue
        parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "Not found"


def _collect_report_assumptions(final_output: dict[str, Any]) -> list[str]:
    assumptions: list[str] = []
    generation_output = final_output.get("generation_result", {}).get("final_output", {}) if isinstance(final_output, dict) else {}
    for source in [
        generation_output.get("assumptions", []),
        final_output.get("assumptions", []),
        final_output.get("execution_result", {}).get("parsed_output", {}).get("assumptions", []),
    ]:
        if not isinstance(source, list):
            continue
        for item in source:
            text = str(item).strip()
            if text and text not in assumptions:
                assumptions.append(text)
    return assumptions


def _build_reproduction_success_line(reproduction_summary: dict[str, Any], execution_result: dict[str, Any]) -> str:
    if execution_result.get("status") != "success":
        return "실패"
    status = str(reproduction_summary.get("status", "")).strip().lower()
    mapping = {
        "good": "성공",
        "partial": "부분 성공",
        "limited": "제한적 성공",
        "poor": "낮은 재현도",
        "failed": "실패",
    }
    return mapping.get(status, "판정 불가")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_uploaded_bytes(filename: str, data: bytes, suffix: str) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = filename.replace(" ", "_")
    file_hash = sha256_bytes(data)
    suffix_to_add = suffix if not safe_name.lower().endswith(suffix) else ""
    destination = RAW_DATA_DIR / f"{timestamp}_{safe_name}{suffix_to_add}"
    destination.write_bytes(data)
    return destination, file_hash


def load_tabular_bytes(filename: str, data: bytes, sheet_name: str | None = None) -> tuple[pd.DataFrame, str | None, str, list[str]]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(BytesIO(data)), None, suffix, []
    if suffix in {".xlsx", ".xls"}:
        excel_file = pd.ExcelFile(BytesIO(data))
        sheets = excel_file.sheet_names
        selected_sheet = sheet_name or (sheets[0] if sheets else None)
        dataframe = pd.read_excel(BytesIO(data), sheet_name=selected_sheet)
        return dataframe, selected_sheet, suffix, sheets
    raise ValueError("지원하지 않는 파일 형식입니다.")


def load_tabular_path(data_path: str, sheet_name: str | None = None) -> tuple[pd.DataFrame, str | None, str, list[str]]:
    path = Path(data_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path), None, suffix, []
    if suffix in {".xlsx", ".xls"}:
        excel_file = pd.ExcelFile(path)
        sheets = excel_file.sheet_names
        selected_sheet = sheet_name or (sheets[0] if sheets else None)
        dataframe = pd.read_excel(path, sheet_name=selected_sheet)
        return dataframe, selected_sheet, suffix, sheets
    raise ValueError("지원하지 않는 파일 형식입니다.")


def default_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered_columns = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered_columns:
            return lowered_columns[candidate.lower()]
    return columns[0] if columns else None


def build_column_detection_profile(dataframe: pd.DataFrame, selected_sheet: str | None = None) -> dict[str, Any]:
    column_summaries: list[dict[str, Any]] = []
    for column in dataframe.columns.tolist():
        series = dataframe[column]
        non_null = series.dropna()
        sample_values = [str(value)[:120] for value in non_null.head(5).tolist()]
        column_summaries.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "non_null_count": int(non_null.shape[0]),
                "null_count": int(series.isna().sum()),
                "unique_count": int(non_null.nunique(dropna=True)),
                "sample_values": sample_values,
                "numeric_ratio": float(pd.to_numeric(series, errors="coerce").notna().mean()),
            }
        )
    return {
        "sheet_name": selected_sheet,
        "num_rows": int(len(dataframe)),
        "num_columns": int(len(dataframe.columns)),
        "columns": [str(column) for column in dataframe.columns.tolist()],
        "column_summaries": column_summaries,
    }


def detect_columns_with_llm(dataframe: pd.DataFrame, file_hash: str, selected_sheet: str | None = None) -> dict[str, Any]:
    profile = build_column_detection_profile(dataframe, selected_sheet=selected_sheet)
    try:
        agent = ColumnSelectionAgent(model_name="gpt-5.2")
        result = agent.invoke(dataset_profile=profile, target_property="boiling point")
        suggestion = result.get("suggestion", {}) if isinstance(result, dict) else {}
        return {
            "smiles_column": suggestion.get("smiles_column"),
            "target_column": suggestion.get("target_column"),
            "smiles_confidence": suggestion.get("smiles_confidence", "low"),
            "target_confidence": suggestion.get("target_confidence", "low"),
            "reasoning": suggestion.get("reasoning", {}) if isinstance(suggestion.get("reasoning"), dict) else {},
            "error": result.get("error") if isinstance(result, dict) else None,
            "file_hash": file_hash,
        }
    except Exception as exc:
        return {
            "smiles_column": None,
            "target_column": None,
            "smiles_confidence": "low",
            "target_confidence": "low",
            "reasoning": {},
            "error": str(exc),
            "file_hash": file_hash,
        }


def dataframe_preview(dataframe: pd.DataFrame, max_rows: int = 8) -> list[dict[str, Any]]:
    preview_df = dataframe.head(max_rows).copy()
    preview_df = preview_df.where(pd.notna(preview_df), None)
    return preview_df.to_dict(orient="records")


def build_session_from_upload(pdf_name: str, pdf_bytes: bytes, data_name: str, data_bytes: bytes, sheet_name: str | None = None) -> dict[str, Any]:
    pdf_path, pdf_hash = save_uploaded_bytes(pdf_name, pdf_bytes, ".pdf")
    data_path, data_hash = save_uploaded_bytes(data_name, data_bytes, Path(data_name).suffix.lower())
    dataframe, selected_sheet, data_suffix, available_sheets = load_tabular_bytes(data_name, data_bytes, sheet_name=sheet_name)
    if dataframe.empty:
        raise ValueError("업로드한 데이터가 비어 있습니다.")

    columns = dataframe.columns.tolist()
    detection = detect_columns_with_llm(dataframe, file_hash=data_hash, selected_sheet=selected_sheet)
    smiles_column = detection.get("smiles_column") or default_column(columns, ["smiles", "SMILES", "canonical_smiles"])
    target_column = detection.get("target_column") or default_column(columns, ["boiling_point", "BoilingPoint", "boiling point", "bp", "BP", "target"])

    return {
        "session_id": uuid.uuid4().hex,
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "pdf_hash": pdf_hash,
        "data_name": data_name,
        "data_path": str(data_path),
        "data_hash": data_hash,
        "data_suffix": data_suffix,
        "data_sheet_name": selected_sheet,
        "available_sheets": available_sheets,
        "data_columns": columns,
        "num_rows": int(len(dataframe)),
        "num_columns": int(len(columns)),
        "column_detection": detection,
        "smiles_column": smiles_column,
        "target_column": target_column,
        "data_preview": dataframe_preview(dataframe),
    }


def cache_markdown_path(pdf_hash: str) -> Path:
    return MARKDOWN_CACHE_DIR / f"{pdf_hash}.md"


def parse_paper_for_session(session: dict[str, Any]) -> dict[str, Any]:
    pdf_hash = session["pdf_hash"]
    cached_path = cache_markdown_path(pdf_hash)
    if cached_path.exists():
        markdown = cached_path.read_text(encoding="utf-8")
        session["paper_markdown"] = markdown
        session["paper_markdown_path"] = str(cached_path)
        session["paper_raw_text_preview"] = session.get("paper_raw_text_preview", "")
        return {
            "cached": True,
            "markdown": markdown,
            "paper_markdown_path": str(cached_path),
        }

    result = run_paper_parsing(Path(session["pdf_path"]))
    if result.get("error"):
        raise RuntimeError(result["error"])
    markdown = result.get("markdown", "")
    raw_text = result.get("raw_text", "")
    cached_path.write_text(markdown, encoding="utf-8")
    session["paper_markdown"] = markdown
    session["paper_markdown_path"] = str(cached_path)
    session["paper_raw_text_preview"] = raw_text[:4000] if raw_text else ""
    return {
        "cached": False,
        "markdown": markdown,
        "paper_markdown_path": str(cached_path),
        "paper_raw_text_preview": session["paper_raw_text_preview"],
    }


def run_rag_for_session(session: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    markdown = session.get("paper_markdown", "")
    if not markdown:
        raise ValueError("먼저 논문 파싱을 실행해 주세요.")
    source_id = f"pdf_{session['pdf_hash'][:16]}"
    rag_result = run_rag_pipeline(markdown=markdown, source_id=source_id, top_k=top_k)
    if rag_result.get("error"):
        raise RuntimeError(rag_result["error"])
    paper_method_spec = rag_result.get("paper_method_spec", {})
    if not has_meaningful_paper_method_spec(paper_method_spec):
        raise RuntimeError("리트리버 결과에서 사용할 수 있는 structured spec을 만들지 못했습니다.")

    session["paper_method_spec"] = paper_method_spec
    session["vector_info"] = rag_result.get("vector_info", {})
    session["spec_build_trace"] = rag_result.get("spec_build_trace", {})
    return {
        "paper_method_spec": paper_method_spec,
        "vector_info": session["vector_info"],
        "spec_build_trace": session["spec_build_trace"],
    }


def build_raw_paper_info(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": {
            "file_path": session.get("data_path"),
            "sheet_name": session.get("data_sheet_name"),
            "smiles_column": session.get("smiles_column"),
            "target_column": session.get("target_column"),
            "columns": session.get("data_columns", []),
        },
        "target_property": "boiling_point",
        "paper_markdown": session.get("paper_markdown", ""),
        "paper_method_spec": session.get("paper_method_spec", {}),
        "metrics": ["MAE", "RMSE", "MSE", "R2"],
    }


def run_generation_for_session(session: dict[str, Any]) -> dict[str, Any]:
    if not session.get("paper_method_spec"):
        raise ValueError("먼저 RAG 추출을 실행해 주세요.")

    run_timestamp = format_run_timestamp()
    payload = {
        "raw_paper_info": build_raw_paper_info(session),
        "generated_code": session.get("generated_code", ""),
        "code_path": session.get("generated_code_path"),
        "data_path": session.get("data_path"),
        "sheet_name": session.get("data_sheet_name"),
        "smiles_column": session.get("smiles_column"),
        "target_column": session.get("target_column"),
        "python_executable": resolve_project_python_executable(sys.executable),
        "max_iterations": 4,
    }
    loop_result = run_code_loop(payload)
    session["loop_result"] = loop_result
    final_output = loop_result.get("final_output", {}) if isinstance(loop_result, dict) else {}
    session["generated_code"] = final_output.get("generated_code", "")
    session["generated_code_path"] = final_output.get("generated_code_path", "") or final_output.get("code_path", "")
    comparison_report_output: dict[str, Any] = {}
    try:
        comparison_report_result = run_comparison_report(
            {
                "paper_method_spec": session.get("paper_method_spec", {}),
                "execution_final_output": final_output,
                "generated_code": session.get("generated_code", ""),
                "generated_code_path": session.get("generated_code_path", ""),
                "report_context": {
                    "pdf_name": session.get("pdf_name", ""),
                    "data_name": session.get("data_name", ""),
                    "target_property": "boiling_point",
                },
            }
        )
        comparison_report_output = (
            comparison_report_result.get("final_output", {})
            if isinstance(comparison_report_result, dict)
            else {}
        )
    except Exception:
        comparison_report_output = {}

    reproduction_summary = {
        "headline": str(comparison_report_output.get("summary_headline", "")).strip(),
        "paragraphs": [
            str(paragraph).strip()
            for paragraph in comparison_report_output.get("summary_paragraphs", [])
            if str(paragraph).strip()
        ],
        "status": str(comparison_report_output.get("summary_status", "")).strip().lower(),
    }
    if not reproduction_summary["headline"] or not reproduction_summary["paragraphs"]:
        reproduction_summary = build_reproduction_summary(
            paper_method_spec=session.get("paper_method_spec", {}),
            execution_final_output=final_output,
            generated_code=session.get("generated_code", ""),
            generated_code_path=session.get("generated_code_path", ""),
        )
    final_output["reproduction_summary"] = reproduction_summary
    final_output["comparison_report"] = {
        "report_markdown": comparison_report_output.get("report_markdown", ""),
        "report_path": comparison_report_output.get("report_path", ""),
        "comparison_table_markdown": comparison_report_output.get("comparison_table_markdown", ""),
        "analysis_markdown": comparison_report_output.get("analysis_markdown", ""),
        "summary_headline": reproduction_summary.get("headline", ""),
        "summary_paragraphs": reproduction_summary.get("paragraphs", []),
    }
    session["reproduction_summary"] = reproduction_summary
    session["comparison_report"] = final_output["comparison_report"]
    saved_artifacts = persist_run_outputs(
        session=session,
        final_output=final_output,
        reproduction_summary=reproduction_summary,
        run_timestamp=run_timestamp,
    )
    final_output["saved_artifacts"] = saved_artifacts
    final_output["run_timestamp"] = run_timestamp
    return loop_result
