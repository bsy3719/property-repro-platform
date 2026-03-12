from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[0]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sections import (
    render_comparison_section,
    render_execution_section,
    render_input_section,
    render_pdf_section,
    render_retriever_section,
)


st.set_page_config(page_title="Paper2Property - Input", layout="wide")
st.title("Paper2Property: 입력 데이터 설정")

render_input_section()
render_pdf_section()
render_retriever_section()
render_execution_section()
render_comparison_section()
