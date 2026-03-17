from __future__ import annotations

import re
from collections import Counter
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
    def _normalize_whitespace(text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\u00a0", " ").replace("\u200b", "")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r" ?\n ?", "\n", normalized)
        return normalized.strip()

    @staticmethod
    def _split_pages(text: str) -> list[str]:
        normalized = PaperParsingAgent._normalize_whitespace(text)
        if not normalized:
            return []
        matches = list(re.finditer(r"(?m)^\[Page \d+\]\s*$", normalized))
        if not matches:
            return [normalized]

        pages: list[str] = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
            chunk = normalized[start:end].strip()
            if chunk:
                pages.append(chunk)
        return pages

    @staticmethod
    def _is_probable_heading(line: str) -> bool:
        text = line.strip()
        if not text:
            return False
        if re.match(r"^\d+(\.\d+)*\s+[A-Z]", text):
            return True
        if text.isupper() and 3 <= len(text.split()) <= 14:
            return True
        if re.match(r"^(abstract|introduction|results?|discussion|conclusions?|materials? and methods?|methods?|experimental|references)\b", text, re.IGNORECASE):
            return True
        words = text.split()
        return len(words) <= 14 and text == text.title() and text[-1].isalnum()

    @staticmethod
    def _remove_repeated_page_artifacts(text: str) -> str:
        pages = PaperParsingAgent._split_pages(text)
        if len(pages) < 2:
            return text

        candidate_counter: Counter[str] = Counter()
        page_lines: list[list[str]] = []
        for page in pages:
            lines = [line.strip() for line in page.splitlines() if line.strip()]
            page_lines.append(lines)
            edge_lines = lines[:3] + lines[-3:]
            seen_on_page: set[str] = set()
            for line in edge_lines:
                canonical = re.sub(r"\s+", " ", line).strip()
                if len(canonical) > 120 or canonical.startswith("[Page "):
                    continue
                if re.fullmatch(r"\d+", canonical):
                    continue
                if canonical in seen_on_page:
                    continue
                seen_on_page.add(canonical)
                candidate_counter[canonical] += 1

        threshold = max(2, len(pages) // 2)
        repeated = {line for line, count in candidate_counter.items() if count >= threshold}
        if not repeated:
            return text

        cleaned_pages: list[str] = []
        for lines in page_lines:
            kept: list[str] = []
            for line in lines:
                canonical = re.sub(r"\s+", " ", line).strip()
                if canonical in repeated and not canonical.startswith("[Page "):
                    continue
                kept.append(line)
            cleaned_pages.append("\n".join(kept).strip())
        return "\n\n".join([page for page in cleaned_pages if page]).strip()

    @staticmethod
    def _join_broken_lines(text: str) -> str:
        lines = [line.rstrip() for line in text.splitlines()]
        merged: list[str] = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                if merged and merged[-1] != "":
                    merged.append("")
                continue

            if re.fullmatch(r"\[Page \d+\]", line):
                if merged and merged[-1] != "":
                    merged.append("")
                merged.append(line)
                continue

            if re.fullmatch(r"\d+", line):
                continue

            if merged:
                prev = merged[-1]
                if prev and not re.fullmatch(r"\[Page \d+\]", prev):
                    prev_heading = PaperParsingAgent._is_probable_heading(prev)
                    current_heading = PaperParsingAgent._is_probable_heading(line)
                    current_list = bool(re.match(r"^[-*•]\s+", line) or re.match(r"^\(?[a-z0-9]+\)\s+", line, re.IGNORECASE))
                    if not prev_heading and not current_heading and not current_list:
                        if prev.endswith("-") and re.search(r"[A-Za-z]-$", prev):
                            merged[-1] = prev[:-1] + line
                        else:
                            merged[-1] = prev + " " + line
                        continue

            merged.append(line)

        compact = "\n".join(merged)
        compact = re.sub(r"\n{3,}", "\n\n", compact)
        return compact.strip()

    @staticmethod
    def _strip_non_essential_sections(text: str) -> str:
        pattern = re.compile(
            r"(?im)^\s*(references|bibliography|acknowledg(?:e)?ments?|appendix|supplementary materials?|supporting information|author contributions?|conflicts? of interest|funding)\b"
        )
        match = pattern.search(text)
        if not match:
            return text
        return text[: match.start()].rstrip()

    @staticmethod
    def _prepare_text(raw_text: str) -> str:
        prepared = PaperParsingAgent._normalize_whitespace(raw_text)
        prepared = PaperParsingAgent._remove_repeated_page_artifacts(prepared)
        prepared = PaperParsingAgent._join_broken_lines(prepared)
        prepared = PaperParsingAgent._strip_non_essential_sections(prepared)
        return prepared.strip()

    @staticmethod
    def _rule_based_markdown(text: str) -> str:
        output: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                if output and output[-1] != "":
                    output.append("")
                continue
            if re.fullmatch(r"\[Page \d+\]", stripped):
                output.append(f"<!-- {stripped} -->")
                output.append("")
                continue
            if re.match(r"^[-*•]\s+", stripped):
                output.append("- " + re.sub(r"^[-*•]\s+", "", stripped))
                continue
            if re.match(r"^\d+(\.\d+)*\s+", stripped) or PaperParsingAgent._is_probable_heading(stripped):
                output.append(f"## {stripped}")
                output.append("")
                continue
            output.append(stripped)
        return "\n".join(output).strip()

    def _paper_to_markdown(self, state: PaperAgentState) -> PaperAgentState:
        raw_text = state.get("raw_text", "")
        if not raw_text.strip():
            return {"error": "입력된 논문 텍스트가 비어 있습니다."}

        prepared_text = self._prepare_text(raw_text)
        prompt = self._build_prompt(prepared_text[:160000])
        try:
            markdown = run_text_response(self.client, self.model_name, prompt, "논문 마크다운 변환")
        except RuntimeError as exc:
            fallback = self._rule_based_markdown(prepared_text)
            if fallback:
                return {"markdown": fallback}
            return {"error": str(exc)}
        return {"markdown": markdown}

    def invoke(self, raw_text: str) -> PaperAgentState:
        return self.graph.invoke({"raw_text": raw_text})

    def _build_prompt(self, trimmed_text: str) -> str:
        return (
            "You are a scientific paper formatting agent for boiling point prediction papers.\n"
            "Goal: Convert extracted paper text into Markdown while preserving the original paper structure and wording as faithfully as possible.\n\n"
            "Rules:\n"
            "1) Keep the original section order, hierarchy, and wording as much as possible.\n"
            "2) Do NOT summarize, paraphrase, rewrite, or reinterpret the content.\n"
            "3) Preserve explicit headings, subheadings, lists, tables, captions, equations, and page order in markdown form.\n"
            "4) Repair only obvious PDF extraction artifacts such as broken line wraps or hyphenated word breaks.\n"
            "5) Keep key numbers, chemical names, model settings, and experimental details exactly as written when possible.\n"
            "6) Remove only non-paper-content or non-essential back matter such as References, Bibliography, Acknowledgements, Appendix, Supporting Information, Funding, and Conflict of Interest sections.\n"
            "7) If page markers like [Page N] appear, preserve them as HTML comments such as <!-- [Page N] -->.\n"
            "8) Return markdown only.\n\n"
            "Paper text:\n"
            f"{trimmed_text}"
        )
