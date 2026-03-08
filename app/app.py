from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.paper_parsing_graph import run_paper_parsing
from src.graph.rag_graph import run_rag_pipeline

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MARKDOWN_CACHE_DIR = ARTIFACTS_DIR / "markdown_cache"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MARKDOWN_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _default_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return columns[0] if columns else None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _save_upload(file_obj, suffix: str) -> tuple[Path, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = file_obj.name.replace(" ", "_")
    data = file_obj.getvalue()
    file_hash = _sha256_bytes(data)
    dest = RAW_DATA_DIR / f"{ts}_{safe_name}{suffix if not safe_name.lower().endswith(suffix) else ''}"
    dest.write_bytes(data)
    return dest, file_hash


def _save_markdown(markdown: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = ARTIFACTS_DIR / f"paper_summary_{ts}.md"
    out_path.write_text(markdown, encoding="utf-8")
    return out_path


def _cache_markdown_path(pdf_hash: str) -> Path:
    return MARKDOWN_CACHE_DIR / f"{pdf_hash}.md"


st.set_page_config(page_title="Paper2Property - Input", layout="wide")
st.title("Paper2Property: 입력 데이터 설정")
st.caption("논문 PDF와 데이터 CSV를 업로드하고 smiles/LFL 컬럼을 지정합니다.")

left, right = st.columns(2)

with left:
    st.subheader("1) 논문 PDF")
    pdf_file = st.file_uploader("논문 파일 업로드", type=["pdf"], key="pdf_uploader")

with right:
    st.subheader("2) 데이터 CSV")
    csv_file = st.file_uploader("CSV 파일 업로드", type=["csv"], key="csv_uploader")

if csv_file is not None:
    try:
        df = pd.read_csv(csv_file)
    except Exception as exc:
        st.error(f"CSV를 읽는 중 오류가 발생했습니다: {exc}")
        st.stop()

    if df.empty:
        st.warning("CSV가 비어 있습니다.")
        st.stop()

    st.subheader("3) 컬럼 지정")
    st.write("CSV 컬럼 목록")
    st.dataframe(pd.DataFrame({"columns": df.columns.tolist()}), use_container_width=True, hide_index=True)

    columns = df.columns.tolist()
    default_smiles = _default_column(columns, ["smiles", "SMILES", "canonical_smiles"])
    default_target = _default_column(columns, ["LFL", "lfl", "target"])

    smiles_col = st.selectbox(
        "SMILES 컬럼",
        options=columns,
        index=columns.index(default_smiles) if default_smiles in columns else 0,
        key="smiles_col",
    )

    lfl_col = st.selectbox(
        "LFL(타겟) 컬럼",
        options=columns,
        index=columns.index(default_target) if default_target in columns else 0,
        key="lfl_col",
    )

    if smiles_col == lfl_col:
        st.warning("SMILES 컬럼과 LFL 컬럼은 서로 달라야 합니다.")

    st.subheader("4) 미리보기")
    preview_cols = [smiles_col, lfl_col] if smiles_col != lfl_col else [smiles_col]
    st.dataframe(df[preview_cols].head(10), use_container_width=True)

    if st.button("입력 확정", type="primary", use_container_width=True):
        if pdf_file is None:
            st.error("먼저 논문 PDF를 업로드해 주세요.")
            st.stop()
        if smiles_col == lfl_col:
            st.error("SMILES/LFL 컬럼이 동일합니다. 다시 선택해 주세요.")
            st.stop()

        saved_pdf, pdf_hash = _save_upload(pdf_file, ".pdf")
        saved_csv, _ = _save_upload(csv_file, ".csv")
        st.session_state["saved_pdf_path"] = str(saved_pdf)
        st.session_state["saved_csv_path"] = str(saved_csv)
        st.session_state["smiles_column"] = smiles_col
        st.session_state["target_column"] = lfl_col
        st.session_state["pdf_hash"] = pdf_hash

        st.success("입력 설정이 저장되었습니다.")
        st.json(
            {
                "pdf_path": str(saved_pdf),
                "csv_path": str(saved_csv),
                "pdf_hash": pdf_hash,
                "smiles_column": smiles_col,
                "target_column": lfl_col,
                "num_rows": int(len(df)),
                "num_columns": int(len(df.columns)),
            }
        )
else:
    st.info("CSV 업로드 후 컬럼 지정 섹션이 활성화됩니다.")

st.divider()
st.subheader("5) PDF 내용 추출 및 마크다운 정리")
st.caption("고정 모델: gpt-5-mini")

saved_pdf_path = st.session_state.get("saved_pdf_path")
pdf_hash = st.session_state.get("pdf_hash")
if saved_pdf_path:
    st.caption(f"저장된 PDF: {saved_pdf_path}")

if st.button("PDF 추출 실행", use_container_width=True):
    if not saved_pdf_path:
        st.error("먼저 '입력 확정'을 눌러 PDF/CSV를 저장해 주세요.")
        st.stop()
    if not pdf_hash:
        st.error("PDF 해시가 없습니다. 다시 입력 확정을 진행해 주세요.")
        st.stop()

    cache_path = _cache_markdown_path(pdf_hash)

    if cache_path.exists():
        markdown = cache_path.read_text(encoding="utf-8")
        out_path = _save_markdown(markdown)
        st.session_state["paper_markdown"] = markdown
        st.session_state["paper_markdown_path"] = str(out_path)

        st.success("기존 마크다운 캐시를 재사용했습니다.")
        st.write(f"- 캐시 파일: {cache_path}")
        st.write(f"- 저장된 마크다운: {out_path}")

        st.markdown("### 정리된 마크다운 결과")
        st.markdown(markdown)

        st.download_button(
            "마크다운 다운로드",
            data=markdown.encode("utf-8"),
            file_name=out_path.name,
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        with st.spinner("PDF 텍스트 추출 및 논문 마크다운 정리 중..."):
            result = run_paper_parsing(Path(saved_pdf_path))

        if result.get("error"):
            st.error(result["error"])
        else:
            raw_text = result.get("raw_text", "")
            markdown = result.get("markdown", "")
            out_path = _save_markdown(markdown)
            cache_path.write_text(markdown, encoding="utf-8")

            st.session_state["paper_markdown"] = markdown
            st.session_state["paper_markdown_path"] = str(out_path)

            st.success("PDF 내용 추출이 완료되었습니다.")
            st.write(f"- 추출 텍스트 길이: {len(raw_text):,} chars")
            st.write(f"- 저장된 마크다운: {out_path}")
            st.write(f"- 캐시 파일 생성: {cache_path}")

            with st.expander("원본 추출 텍스트(앞 4000자)", expanded=False):
                st.text(raw_text[:4000] if raw_text else "")

            st.markdown("### 정리된 마크다운 결과")
            st.markdown(markdown)

            st.download_button(
                "마크다운 다운로드",
                data=markdown.encode("utf-8"),
                file_name=out_path.name,
                mime="text/markdown",
                use_container_width=True,
            )

st.divider()
st.subheader("6) 벡터 DB 생성 및 리트리버 결과 정리(보고서용)")
st.caption("임베딩 모델: text-embedding-3-large")
st.caption("고정 질문: model/feature/hyperparameter/training/metrics")

top_k = st.slider("각 질문별 Top-K", min_value=3, max_value=10, value=5)

if st.button("벡터DB 생성 + 멀티 리트리버 실행", use_container_width=True):
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
        rag_result = run_rag_pipeline(
            markdown=markdown,
            source_id=source_id,
            top_k=top_k,
        )

    if rag_result.get("error"):
        st.error(rag_result["error"])
    else:
        vector_info = rag_result.get("vector_info", {})
        summary_markdown = rag_result.get("summary_markdown", "")

        cache_flag = vector_info.get("cached", False)
        cache_text = "재사용됨(캐시)" if cache_flag else "신규 생성"

        st.success("멀티 리트리버 실행 완료")
        st.info("코드 생성은 기존 마크다운 원문을 사용하고, 아래 정리 내용은 최종 비교 보고서 작성에 사용하세요.")
        st.write(
            {
                "source_id": vector_info.get("source_id"),
                "num_chunks": vector_info.get("num_chunks"),
                "embedding_dim": vector_info.get("embedding_dim"),
                "status": cache_text,
            }
        )

        if summary_markdown:
            st.markdown("### 리트리버 정리 결과(최종 보고서용)")
            st.markdown(summary_markdown)
        else:
            st.warning("리트리버 정리 결과가 비어 있습니다.")


