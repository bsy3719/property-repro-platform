from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.utils import create_openai_client, run_text_response


class PaperAgentState(TypedDict, total=False):
    raw_text: str
    markdown: str
    error: str


class PaperParsingAgent:
    """LangGraph agent that converts boiling-point paper text to markdown while preserving source structure."""

    def __init__(self, model_name: str = "gpt-5.2") -> None:
        self.client, self.model_name = create_openai_client(model_name)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(PaperAgentState)
        graph.add_node("paper_to_markdown", self._paper_to_markdown)
        graph.set_entry_point("paper_to_markdown")
        graph.add_edge("paper_to_markdown", END)
        return graph.compile()

    @staticmethod
    def _strip_non_essential_sections(text: str) -> str:
        pattern = re.compile(r"(?im)^\s*(references|bibliography|acknowledg(?:e)?ments?|appendix)\b")
        match = pattern.search(text)
        if not match:
            return text
        return text[: match.start()].rstrip()

    def _paper_to_markdown(self, state: PaperAgentState) -> PaperAgentState:
        raw_text = state.get("raw_text", "")
        if not raw_text.strip():
            return {"error": "입력된 논문 텍스트가 비어 있습니다."}

        filtered_text = self._strip_non_essential_sections(raw_text)
        prompt = self._build_prompt(filtered_text[:140000])
        try:
            markdown = run_text_response(self.client, self.model_name, prompt, "논문 마크다운 변환")
        except RuntimeError as exc:
            return {"error": str(exc)}
        return {"markdown": markdown}

    def invoke(self, raw_text: str) -> PaperAgentState:
        return self.graph.invoke({"raw_text": raw_text})

    def _build_prompt(self, trimmed_text: str) -> str:
        return (
            "You are a scientific paper formatting agent for boiling point prediction papers.\n"
            "Goal: Convert extracted paper text into Markdown while preserving the original paper structure and wording.\n\n"
            "Rules:\n"
            "1) Keep the original section order and hierarchy as much as possible.\n"
            "2) Do NOT summarize, paraphrase, or reinterpret the content.\n"
            "3) Convert headings/lists/tables into markdown format only.\n"
            "4) Keep key numbers, equations, and experimental details exactly as written when possible.\n"
            "5) Exclude non-essential sections for model building: References, Bibliography, Acknowledgements, Appendix.\n"
            "6) Return markdown only.\n\n"
            "Paper text:\n"
            f"{trimmed_text}"
        )
