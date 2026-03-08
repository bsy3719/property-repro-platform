from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import END, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"


class PaperAgentState(TypedDict, total=False):
    raw_text: str
    markdown: str
    error: str


class PaperParsingAgent:
    """LangGraph agent that converts paper text to markdown while preserving source structure."""

    def __init__(self, model_name: str = "gpt-5-mini") -> None:
        load_dotenv(DOTENV_PATH)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
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

        filtered = self._strip_non_essential_sections(raw_text)
        trimmed = filtered[:140000]

        prompt = (
            "You are a scientific paper formatting agent.\n"
            "Goal: Convert extracted paper text into Markdown while preserving the original paper structure and wording.\n\n"
            "Rules:\n"
            "1) Keep the original section order and hierarchy as much as possible.\n"
            "2) Do NOT summarize, paraphrase, or reinterpret the content.\n"
            "3) Convert headings/lists/tables into markdown format only.\n"
            "4) Keep key numbers, equations, and experimental details exactly as written when possible.\n"
            "5) Exclude non-essential sections for model building: References, Bibliography, Acknowledgements, Appendix.\n"
            "6) Return markdown only.\n\n"
            "Paper text:\n"
            f"{trimmed}"
        )

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
        )
        markdown = (response.output_text or "").strip()
        if not markdown:
            return {"error": "LLM이 마크다운 결과를 반환하지 않았습니다."}

        return {"markdown": markdown}

    def invoke(self, raw_text: str) -> PaperAgentState:
        return self.graph.invoke({"raw_text": raw_text})

