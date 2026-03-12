from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from workflow import (
    build_auto_loop_signature,
    build_generation_input_status,
    default_column,
    load_tabular_data,
    reset_document_state,
    run_comparison_report_action,
    run_generation_and_execution,
    run_pdf_extraction,
    run_retriever,
    save_upload,
)


def render_input_section() -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("1) 논문 PDF")
        pdf_file = st.file_uploader("논문 파일 업로드", type=["pdf"], key="pdf_uploader")
    with right:
        st.subheader("2) 데이터 파일")
        data_file = st.file_uploader("CSV 또는 Excel 파일 업로드", type=["csv", "xlsx", "xls"], key="data_uploader")

    if data_file is None:
        st.info("CSV 또는 Excel 업로드 후 컬럼 지정 섹션이 활성화됩니다.")
        return

    try:
        dataframe, selected_sheet, data_suffix = load_tabular_data(data_file)
    except Exception as exc:
        st.error(f"데이터 파일을 읽는 중 오류가 발생했습니다: {exc}")
        st.stop()

    if dataframe.empty:
        st.warning("업로드한 데이터가 비어 있습니다.")
        st.stop()

    st.subheader("3) 컬럼 지정")
    if selected_sheet is not None:
        st.caption(f"선택된 시트: {selected_sheet}")
    st.dataframe(pd.DataFrame({"columns": dataframe.columns.tolist()}), use_container_width=True, hide_index=True)

    columns = dataframe.columns.tolist()
    default_smiles = default_column(columns, ["smiles", "SMILES", "canonical_smiles"])
    default_target = default_column(columns, ["boiling_point", "BoilingPoint", "boiling point", "bp", "BP", "target"])
    smiles_column = st.selectbox("SMILES 컬럼", options=columns, index=columns.index(default_smiles) if default_smiles in columns else 0, key="smiles_col")
    target_column = st.selectbox("Boiling Point(타겟) 컬럼", options=columns, index=columns.index(default_target) if default_target in columns else 0, key="target_col")

    if smiles_column == target_column:
        st.warning("SMILES 컬럼과 Boiling Point 컬럼은 서로 달라야 합니다.")

    st.subheader("4) 미리보기")
    preview_columns = [smiles_column, target_column] if smiles_column != target_column else [smiles_column]
    st.dataframe(dataframe[preview_columns].head(10), use_container_width=True)

    if not st.button("입력 확정", type="primary", use_container_width=True):
        return
    if pdf_file is None:
        st.error("먼저 논문 PDF를 업로드해 주세요.")
        st.stop()
    if smiles_column == target_column:
        st.error("SMILES/Boiling Point 컬럼이 동일합니다. 다시 선택해 주세요.")
        st.stop()

    saved_pdf, pdf_hash = save_upload(pdf_file, ".pdf")
    saved_data, _ = save_upload(data_file, data_suffix)
    st.session_state["saved_pdf_path"] = str(saved_pdf)
    st.session_state["saved_data_path"] = str(saved_data)
    st.session_state["data_sheet_name"] = selected_sheet
    st.session_state["smiles_column"] = smiles_column
    st.session_state["target_column"] = target_column
    st.session_state["pdf_hash"] = pdf_hash
    st.session_state["data_columns"] = columns
    reset_document_state()

    st.success("입력 설정이 저장되었습니다.")
    st.json({
        "pdf_path": str(saved_pdf),
        "data_path": str(saved_data),
        "data_sheet_name": selected_sheet,
        "pdf_hash": pdf_hash,
        "smiles_column": smiles_column,
        "target_column": target_column,
        "data_columns": columns,
        "num_rows": int(len(dataframe)),
        "num_columns": int(len(dataframe.columns)),
    })


def render_pdf_section() -> None:
    st.divider()
    st.subheader("5) PDF 내용 추출 및 마크다운 정리")
    saved_pdf_path = st.session_state.get("saved_pdf_path")
    if saved_pdf_path:
        st.caption(f"저장된 PDF: {saved_pdf_path}")

    if st.button("PDF 추출 실행", use_container_width=True):
        run_pdf_extraction()

    markdown = st.session_state.get("paper_markdown")
    if not markdown:
        return

    raw_preview = st.session_state.get("paper_raw_text_preview", "")
    if raw_preview:
        with st.expander("원본 추출 텍스트(앞 4000자)", expanded=False):
            st.text(raw_preview)
    st.markdown("### 정리된 마크다운 결과")
    st.markdown(markdown)
    paper_markdown_path = st.session_state.get("paper_markdown_path")
    st.download_button(
        "마크다운 다운로드",
        data=markdown.encode("utf-8"),
        file_name=Path(paper_markdown_path).name if paper_markdown_path else "paper_summary.md",
        mime="text/markdown",
        use_container_width=True,
    )


def render_retriever_section() -> None:
    st.divider()
    st.subheader("6) 벡터 DB 생성 및 리트리버 결과 정리(보고서용)")
    top_k = st.slider("각 질문별 Top-K", min_value=3, max_value=10, value=5)
    if st.button("벡터DB 생성 + 멀티 리트리버 실행", use_container_width=True):
        run_retriever(top_k)

    vector_info = st.session_state.get("vector_info")
    if vector_info:
        st.write({
            "source_id": vector_info.get("source_id"),
            "num_chunks": vector_info.get("num_chunks"),
            "status": "재사용됨(캐시)" if vector_info.get("cached", False) else "신규 생성",
        })

    summary_markdown = st.session_state.get("retriever_summary_markdown")
    if summary_markdown:
        st.markdown("### 리트리버 정리 결과")
        st.markdown(summary_markdown)

    paper_method_spec = st.session_state.get("paper_method_spec")
    if paper_method_spec:
        with st.expander("structured spec (코드 생성/비교용)", expanded=False):
            st.json(paper_method_spec)


def render_execution_section() -> None:
    st.divider()
    st.subheader("7) 코드 생성 및 실행")

    if not st.session_state.get("paper_method_spec"):
        st.info("6) 단계에서 structured spec을 생성한 뒤 실행할 수 있습니다.")
        return
    if build_auto_loop_signature() is None:
        st.info("데이터 입력, PDF 추출, section 6 structured spec 생성이 모두 필요합니다.")
        return

    input_status = build_generation_input_status()
    st.caption("코드 생성 입력 상태")
    st.json(input_status)

    if st.button("코드 생성 및 실행 시작", type="primary", use_container_width=True):
        run_generation_and_execution()

    loop_result = st.session_state.get("loop_result")
    if not loop_result:
        return

    final_output = loop_result.get("final_output", {})
    execution_info = final_output.get("execution_result", {})
    st.write({
        "iteration": final_output.get("iteration"),
        "max_iterations": final_output.get("max_iterations"),
        "repeated_error_count": final_output.get("repeated_error_count"),
        "stop_reason": final_output.get("stop_reason"),
        "status": execution_info.get("status"),
        "returncode": execution_info.get("returncode"),
        "metrics": execution_info.get("metrics", {}),
    })

    generated_code = final_output.get("generated_code") or st.session_state.get("generated_code", "")
    generated_code_path = st.session_state.get("generated_code_path")
    if generated_code:
        if generated_code_path:
            st.caption(f"저장 경로: {generated_code_path}")
        st.markdown("### 생성된 Python 코드")
        st.code(generated_code, language="python")
        st.download_button(
            "생성 코드 다운로드",
            data=generated_code.encode("utf-8"),
            file_name=Path(generated_code_path).name if generated_code_path else "generated_regression.py",
            mime="text/x-python",
            use_container_width=True,
        )

    if execution_info:
        with st.expander("표준 출력", expanded=False):
            st.text(execution_info.get("stdout", ""))
        with st.expander("표준 에러", expanded=False):
            st.text(execution_info.get("stderr", ""))
        if final_output.get("error_history"):
            with st.expander("디버그 루프 에러 이력", expanded=False):
                st.json(final_output.get("error_history"))


def render_comparison_section() -> None:
    st.divider()
    st.subheader("8) 논문 대비 재현 결과 비교 보고서")

    if not st.session_state.get("paper_method_spec"):
        st.info("비교 보고서를 생성하려면 먼저 6) 단계의 structured spec이 필요합니다.")
        return
    if not st.session_state.get("loop_result"):
        st.info("비교 보고서를 생성하려면 7) 코드 생성 및 실행을 먼저 수행해야 합니다.")
        return

    if st.button("비교 보고서 생성", use_container_width=True):
        run_comparison_report_action()

    comparison_result = st.session_state.get("comparison_result")
    if not comparison_result:
        st.info("버튼을 눌러 비교 보고서를 생성합니다.")
        return

    final_output = comparison_result.get("final_output", {})
    comparison_table = final_output.get("comparison_table_markdown", "")
    analysis_markdown = final_output.get("analysis_markdown", "")
    report_path = final_output.get("report_path", "")

    if comparison_table:
        st.markdown("### 비교 표")
        st.markdown(comparison_table)
    if analysis_markdown:
        st.markdown("### 차이 분석")
        st.markdown(analysis_markdown)
    if report_path:
        st.caption(f"보고서 저장 경로: {report_path}")
