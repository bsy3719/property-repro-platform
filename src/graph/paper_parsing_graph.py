from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader
from langgraph.graph import END, StateGraph

from src.agents.paper_parsing_agent import PaperParsingAgent


class PaperParseState(TypedDict, total=False):
    pdf_path: str
    raw_text: str
    markdown: str
    error: str


def _extract_pdf_text(state: PaperParseState) -> PaperParseState:
    pdf_path = state["pdf_path"]
    reader = PdfReader(pdf_path)

    pages: list[str] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n\n[Page {idx}]\n{text}")

    raw_text = "\n".join(pages).strip()
    if not raw_text:
        return {"error": "PDF에서 텍스트를 추출하지 못했습니다.", "raw_text": ""}

    return {"raw_text": raw_text}


def _run_paper_parsing_agent(state: PaperParseState) -> PaperParseState:
    if state.get("error"):
        return state

    agent = PaperParsingAgent(model_name="gpt-5-mini")
    result = agent.invoke(state.get("raw_text", ""))
    if result.get("error"):
        return {"error": result["error"]}

    return {"markdown": result.get("markdown", "")}


def build_graph():
    graph = StateGraph(PaperParseState)
    graph.add_node("extract_pdf_text", _extract_pdf_text)
    graph.add_node("run_paper_parsing_agent", _run_paper_parsing_agent)

    graph.set_entry_point("extract_pdf_text")
    graph.add_edge("extract_pdf_text", "run_paper_parsing_agent")
    graph.add_edge("run_paper_parsing_agent", END)

    return graph.compile()


PAPER_PARSING_GRAPH = build_graph()


def run_paper_parsing(pdf_path: Path) -> PaperParseState:
    result = PAPER_PARSING_GRAPH.invoke({"pdf_path": str(pdf_path)})
    return result
