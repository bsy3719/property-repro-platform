from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.code_loop_graph import run_code_loop
from src.graph.comparison_report_graph import run_comparison_report
from src.graph.paper_parsing_graph import run_paper_parsing
from src.graph.rag_graph import run_rag_pipeline
from src.utils import has_meaningful_paper_method_spec

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MARKDOWN_CACHE_DIR = ARTIFACTS_DIR / "markdown_cache"
GENERATED_CODE_DIR = ARTIFACTS_DIR / "generated_code"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

for directory in [RAW_DATA_DIR, ARTIFACTS_DIR, MARKDOWN_CACHE_DIR, GENERATED_CODE_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def default_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered_columns = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered_columns:
            return lowered_columns[candidate.lower()]
    return columns[0] if columns else None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reset_execution_state() -> None:
    for key in [
        "auto_loop_signature",
        "loop_result",
        "generated_code",
        "generated_code_path",
        "comparison_signature",
        "comparison_result",
    ]:
        st.session_state.pop(key, None)


def reset_retrieval_state() -> None:
    for key in ["retriever_summary_markdown", "paper_method_spec", "vector_info"]:
        st.session_state.pop(key, None)
    reset_execution_state()


def reset_document_state() -> None:
    for key in ["paper_markdown", "paper_markdown_path", "paper_raw_text_preview"]:
        st.session_state.pop(key, None)
    reset_retrieval_state()


def save_upload(file_obj, suffix: str) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = file_obj.name.replace(" ", "_")
    file_bytes = file_obj.getvalue()
    file_hash = sha256_bytes(file_bytes)
    suffix_to_add = suffix if not safe_name.lower().endswith(suffix) else ""
    destination = RAW_DATA_DIR / f"{timestamp}_{safe_name}{suffix_to_add}"
    destination.write_bytes(file_bytes)
    return destination, file_hash


def save_markdown(markdown: str) -> Path:
    output_path = ARTIFACTS_DIR / f"paper_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def save_generated_code(code: str) -> Path:
    output_path = GENERATED_CODE_DIR / f"generated_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    output_path.write_text(code, encoding="utf-8")
    return output_path


def cache_markdown_path(pdf_hash: str) -> Path:
    return MARKDOWN_CACHE_DIR / f"{pdf_hash}.md"


def load_tabular_data(uploaded_file) -> tuple[pd.DataFrame, str | None, str]:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file), None, suffix
    if suffix in {".xlsx", ".xls"}:
        file_bytes = uploaded_file.getvalue()
        excel_file = pd.ExcelFile(BytesIO(file_bytes))
        sheet_name = st.selectbox("Excel 시트 선택", options=excel_file.sheet_names, key="excel_sheet_name")
        dataframe = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)
        return dataframe, sheet_name, suffix
    raise ValueError("지원하지 않는 파일 형식입니다.")


def build_raw_paper_info() -> dict:
    return {
        "dataset": {
            "file_path": st.session_state.get("saved_data_path"),
            "sheet_name": st.session_state.get("data_sheet_name"),
            "smiles_column": st.session_state.get("smiles_column"),
            "target_column": st.session_state.get("target_column"),
            "columns": st.session_state.get("data_columns", []),
        },
        "target_property": "boiling_point",
        "paper_markdown": st.session_state.get("paper_markdown", ""),
        "model_anchor_summary": st.session_state.get("retriever_summary_markdown", ""),
        "paper_method_spec": st.session_state.get("paper_method_spec", {}),
        "metrics": ["MAE", "RMSE", "MSE", "R2"],
    }


def build_auto_loop_signature() -> str | None:
    paper_markdown = st.session_state.get("paper_markdown")
    paper_method_spec = st.session_state.get("paper_method_spec", {})
    data_path = st.session_state.get("saved_data_path")
    smiles_column = st.session_state.get("smiles_column")
    target_column = st.session_state.get("target_column")
    if not all([paper_markdown, data_path, smiles_column, target_column]):
        return None
    if not has_meaningful_paper_method_spec(paper_method_spec):
        return None

    payload = {
        "paper_markdown": paper_markdown,
        "model_anchor_summary": st.session_state.get("retriever_summary_markdown", ""),
        "paper_method_spec": paper_method_spec,
        "data_path": data_path,
        "sheet_name": st.session_state.get("data_sheet_name"),
        "smiles_column": smiles_column,
        "target_column": target_column,
        "columns": st.session_state.get("data_columns", []),
    }
    return sha256_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))


def build_comparison_signature() -> str | None:
    paper_method_spec = st.session_state.get("paper_method_spec", {})
    loop_result = st.session_state.get("loop_result", {})
    final_output = loop_result.get("final_output", {}) if isinstance(loop_result, dict) else {}
    if not has_meaningful_paper_method_spec(paper_method_spec) or not final_output:
        return None

    payload = {
        "paper_method_spec": paper_method_spec,
        "paper_summary_markdown": st.session_state.get("retriever_summary_markdown", ""),
        "execution_final_output": final_output,
        "generated_code": st.session_state.get("generated_code", ""),
        "generated_code_path": st.session_state.get("generated_code_path", ""),
    }
    return sha256_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))


def run_pdf_extraction() -> None:
    saved_pdf_path = st.session_state.get("saved_pdf_path")
    pdf_hash = st.session_state.get("pdf_hash")
    if not saved_pdf_path:
        st.error("먼저 '입력 확정'을 눌러 PDF/데이터 파일을 저장해 주세요.")
        st.stop()
    if not pdf_hash:
        st.error("PDF 해시가 없습니다. 다시 입력 확정을 진행해 주세요.")
        st.stop()

    cached_path = cache_markdown_path(pdf_hash)
    if cached_path.exists():
        markdown = cached_path.read_text(encoding="utf-8")
        output_path = save_markdown(markdown)
        reset_retrieval_state()
        st.session_state["paper_markdown"] = markdown
        st.session_state["paper_markdown_path"] = str(output_path)
        st.success("기존 마크다운 캐시를 재사용했습니다.")
        st.write(f"- 캐시 파일: {cached_path}")
        st.write(f"- 저장된 마크다운: {output_path}")
        return

    with st.spinner("PDF 텍스트 추출 및 논문 마크다운 정리 중..."):
        result = run_paper_parsing(Path(saved_pdf_path))
    if result.get("error"):
        st.error(result["error"])
        return

    markdown = result.get("markdown", "")
    raw_text = result.get("raw_text", "")
    output_path = save_markdown(markdown)
    cached_path.write_text(markdown, encoding="utf-8")
    reset_retrieval_state()
    st.session_state["paper_markdown"] = markdown
    st.session_state["paper_markdown_path"] = str(output_path)
    st.session_state["paper_raw_text_preview"] = raw_text[:4000] if raw_text else ""
    st.success("PDF 내용 추출이 완료되었습니다.")
    st.write(f"- 추출 텍스트 길이: {len(raw_text):,} chars")
    st.write(f"- 저장된 마크다운: {output_path}")
    st.write(f"- 캐시 파일 생성: {cached_path}")


def run_retriever(top_k: int) -> None:
    markdown = st.session_state.get("paper_markdown")
    pdf_hash = st.session_state.get("pdf_hash")
    if not markdown:
        st.error("먼저 'PDF 추출 실행'을 완료해 주세요.")
        st.stop()
    if not pdf_hash:
        st.error("PDF 해시가 없습니다. 다시 입력 확정을 진행해 주세요.")
        st.stop()

    source_id = f"pdf_{pdf_hash[:16]}"
    with st.spinner("벡터DB 구성 및 멀티 리트리버 검색 중..."):
        rag_result = run_rag_pipeline(markdown=markdown, source_id=source_id, top_k=top_k)
    if rag_result.get("error"):
        st.error(rag_result["error"])
        return

    paper_method_spec = rag_result.get("paper_method_spec", {})
    if not has_meaningful_paper_method_spec(paper_method_spec):
        st.error("리트리버 결과에서 사용할 수 있는 structured spec을 만들지 못했습니다.")
        return

    st.session_state["retriever_summary_markdown"] = rag_result.get("summary_markdown", "")
    st.session_state["paper_method_spec"] = paper_method_spec
    st.session_state["vector_info"] = rag_result.get("vector_info", {})
    reset_execution_state()
    st.success("멀티 리트리버 실행 완료")


def run_generation_and_execution() -> None:
    current_signature = build_auto_loop_signature()
    if current_signature is None:
        st.error("데이터 입력, PDF 추출, section 6 structured spec 생성이 모두 필요합니다.")
        return

    payload = {
        "raw_paper_info": build_raw_paper_info(),
        "generated_code": st.session_state.get("generated_code", ""),
        "code_path": st.session_state.get("generated_code_path"),
        "data_path": st.session_state.get("saved_data_path"),
        "sheet_name": st.session_state.get("data_sheet_name"),
        "smiles_column": st.session_state.get("smiles_column"),
        "target_column": st.session_state.get("target_column"),
        "python_executable": sys.executable,
        "max_iterations": 4,
    }

    st.session_state.pop("comparison_signature", None)
    st.session_state.pop("comparison_result", None)
    with st.spinner("코드 생성 및 실행 중..."):
        loop_result = run_code_loop(payload)

    final_output = loop_result.get("final_output", {})
    generated_code = final_output.get("generated_code", "")
    if generated_code:
        code_path = save_generated_code(generated_code)
        st.session_state["generated_code"] = generated_code
        st.session_state["generated_code_path"] = str(code_path)

    st.session_state["loop_result"] = loop_result
    st.session_state["auto_loop_signature"] = current_signature


def build_generation_input_status() -> dict[str, object]:
    paper_markdown = st.session_state.get("paper_markdown", "")
    retriever_summary = st.session_state.get("retriever_summary_markdown", "")
    paper_method_spec = st.session_state.get("paper_method_spec", {})
    selected_model = ""
    if isinstance(paper_method_spec, dict):
        selected_model = str(paper_method_spec.get("model", {}).get("name", ""))
    return {
        "has_paper_markdown": bool(paper_markdown.strip()),
        "paper_markdown_chars": len(paper_markdown),
        "has_retriever_summary": bool(retriever_summary.strip()),
        "retriever_summary_chars": len(retriever_summary),
        "has_paper_method_spec": has_meaningful_paper_method_spec(paper_method_spec),
        "selected_model": selected_model,
        "saved_data_path": st.session_state.get("saved_data_path", ""),
        "smiles_column": st.session_state.get("smiles_column", ""),
        "target_column": st.session_state.get("target_column", ""),
    }


def run_comparison_report_action() -> None:
    comparison_signature = build_comparison_signature()
    if not comparison_signature:
        st.error("section 6 structured spec과 코드 실행 결과가 있어야 비교 보고서를 생성할 수 있습니다.")
        return

    if st.session_state.get("comparison_signature") == comparison_signature and st.session_state.get("comparison_result"):
        return

    loop_result = st.session_state.get("loop_result", {})
    final_output = loop_result.get("final_output", {}) if isinstance(loop_result, dict) else {}
    payload = {
        "paper_method_spec": st.session_state.get("paper_method_spec", {}),
        "paper_summary_markdown": st.session_state.get("retriever_summary_markdown", ""),
        "execution_final_output": final_output,
        "generated_code": st.session_state.get("generated_code", ""),
        "generated_code_path": st.session_state.get("generated_code_path", ""),
        "report_context": {
            "target_property": "boiling_point",
            "pdf_hash": st.session_state.get("pdf_hash", ""),
            "data_path": st.session_state.get("saved_data_path", ""),
            "paper_markdown_path": st.session_state.get("paper_markdown_path", ""),
        },
    }

    with st.spinner("논문 대비 재현 결과 비교 보고서 생성 중..."):
        comparison_result = run_comparison_report(payload)
    st.session_state["comparison_result"] = comparison_result
    st.session_state["comparison_signature"] = comparison_signature
