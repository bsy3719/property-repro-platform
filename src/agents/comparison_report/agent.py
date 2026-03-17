from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.utils import (
    has_meaningful_paper_method_spec,
    normalize_paper_method_spec,
    paper_method_spec_to_comparison_spec,
    create_openai_client,
    run_text_response,
)

from .formatting import build_comparison_table, compose_report, fallback_analysis, save_report
from .parsing import parse_execution_spec, parse_paper_spec
from .schemas import ComparisonReportState


class ComparisonReportAgent:
    def __init__(self, model_name: str = "gpt-5.2") -> None:
        self.client, self.model_name = create_openai_client(model_name)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ComparisonReportState)
        graph.add_node("parse_paper_summary", self.parse_paper_summary)
        graph.add_node("parse_execution_result", self.parse_execution_result)
        graph.add_node("build_comparison_rows", self.build_comparison_rows)
        graph.add_node("analyze_differences", self.analyze_differences)
        graph.add_node("build_ui_summary", self.build_ui_summary)
        graph.add_node("compose_report", self.compose_report)
        graph.add_node("save_report", self.save_report)
        graph.add_node("finalize_output", self.finalize_output)

        graph.set_entry_point("parse_paper_summary")
        graph.add_edge("parse_paper_summary", "parse_execution_result")
        graph.add_edge("parse_execution_result", "build_comparison_rows")
        graph.add_edge("build_comparison_rows", "analyze_differences")
        graph.add_edge("analyze_differences", "build_ui_summary")
        graph.add_edge("build_ui_summary", "compose_report")
        graph.add_edge("compose_report", "save_report")
        graph.add_edge("save_report", "finalize_output")
        graph.add_edge("finalize_output", END)
        return graph.compile()

    def invoke(self, payload: dict[str, object]) -> ComparisonReportState:
        return self.graph.invoke(payload)

    def parse_paper_summary(self, state: ComparisonReportState) -> ComparisonReportState:
        paper_method_spec = normalize_paper_method_spec(state.get("paper_method_spec", {}))
        if has_meaningful_paper_method_spec(paper_method_spec):
            return {
                "paper_method_spec": paper_method_spec,
                "paper_spec": paper_method_spec_to_comparison_spec(paper_method_spec),
            }

        markdown = state.get("paper_summary_markdown", "")
        if not markdown.strip():
            return {"error": "논문 요약 markdown이 없습니다."}
        return {"paper_spec": parse_paper_spec(markdown), "paper_method_spec": paper_method_spec}

    def parse_execution_result(self, state: ComparisonReportState) -> ComparisonReportState:
        final_output = state.get("execution_final_output", {})
        if not final_output:
            return {"error": "실행 결과 final_output이 없습니다."}
        generated_code = state.get("generated_code") or final_output.get("generated_code", "")
        generated_code_path = state.get("generated_code_path") or final_output.get("code_path", "")
        return {"execution_spec": parse_execution_spec(final_output, generated_code, generated_code_path)}

    def build_comparison_rows(self, state: ComparisonReportState) -> ComparisonReportState:
        comparison_table = build_comparison_table(state.get("paper_spec", {}), state.get("execution_spec", {}))
        return {"comparison_table_markdown": comparison_table}

    def analyze_differences(self, state: ComparisonReportState) -> ComparisonReportState:
        paper_spec = state.get("paper_spec", {})
        execution_spec = state.get("execution_spec", {})
        comparison_table = state.get("comparison_table_markdown", "")
        report_context = state.get("report_context", {})
        prompt = (
            "You are writing a reproducibility comparison report in Korean.\n"
            "Compare the paper methodology and the reproduced result.\n"
            "Use only the provided structured inputs.\n"
            "Do not exaggerate. Do not claim certainty without evidence.\n"
            "If some paper details are missing, explicitly state that the reproduced code used defaults.\n"
            "Return markdown only with these exact sections:\n"
            "## Overall Assessment\n"
            "## Methodology Differences\n"
            "## Metric Differences\n"
            "## Likely Causes\n\n"
            f"Paper spec:\n{paper_spec}\n\n"
            f"Paper method spec:\n{state.get('paper_method_spec', {})}\n\n"
            f"Execution spec:\n{execution_spec}\n\n"
            f"Comparison table:\n{comparison_table}\n\n"
            f"Report context:\n{report_context}"
        )
        try:
            analysis_markdown = run_text_response(self.client, self.model_name, prompt, "비교 보고서 분석")
            warning = ""
        except RuntimeError as exc:
            analysis_markdown = ""
            warning = str(exc)

        if not analysis_markdown:
            analysis_markdown = fallback_analysis(paper_spec, execution_spec, warning)
        return {"analysis_markdown": analysis_markdown}

    def build_ui_summary(self, state: ComparisonReportState) -> ComparisonReportState:
        paper_spec = state.get("paper_spec", {})
        execution_spec = state.get("execution_spec", {})
        comparison_table = state.get("comparison_table_markdown", "")
        prompt = (
            "You are writing a short reproducibility summary for a Korean UI.\n"
            "Use only the provided structured inputs.\n"
            "Return JSON only with this schema:\n"
            '{\n'
            '  "headline": "한 문장 요약",\n'
            '  "paragraphs": ["문단1", "문단2", "문단3"]\n'
            '}\n'
            "Rules:\n"
            "- Korean only\n"
            "- headline must be exactly one sentence\n"
            "- paragraphs must contain 2 to 4 natural-language sentences\n"
            "- No markdown fences\n"
            "- No code, no JSON literals in the text, no Python-like syntax\n"
            "- Do not use SPEC =, ASSUMPTIONS =, def, import, dict/list literal text\n"
            "- Focus on reproduction quality, major metric gaps, methodology differences, and likely causes\n\n"
            f"Paper spec:\n{paper_spec}\n\n"
            f"Paper method spec:\n{state.get('paper_method_spec', {})}\n\n"
            f"Execution spec:\n{execution_spec}\n\n"
            f"Comparison table:\n{comparison_table}\n\n"
            f"Analysis markdown:\n{state.get('analysis_markdown', '')}"
        )
        try:
            raw = run_text_response(self.client, self.model_name, prompt, "비교 보고서 UI 요약")
            summary = self._parse_summary_payload(raw)
        except RuntimeError:
            summary = {}

        headline = str(summary.get("headline", "")).strip()
        paragraphs = [
            self._sanitize_summary_line(paragraph)
            for paragraph in summary.get("paragraphs", [])
            if self._sanitize_summary_line(paragraph)
        ]
        if not headline or not paragraphs:
            fallback = self._fallback_ui_summary(state)
            headline = fallback["headline"]
            paragraphs = fallback["paragraphs"]

        return {
            "summary_status": self._infer_summary_status(state),
            "summary_headline": self._sanitize_summary_line(headline),
            "summary_paragraphs": paragraphs,
        }

    def compose_report(self, state: ComparisonReportState) -> ComparisonReportState:
        execution_spec = state.get("execution_spec", {})
        report_markdown = compose_report(
            execution_spec,
            state.get("comparison_table_markdown", ""),
            state.get("analysis_markdown", ""),
            state.get("summary_headline", ""),
            state.get("summary_paragraphs", []),
            state.get("execution_final_output", {}),
            execution_spec.get("generated_code_path") or state.get("generated_code_path", ""),
        )
        return {"report_markdown": report_markdown}

    def save_report(self, state: ComparisonReportState) -> ComparisonReportState:
        report_markdown = state.get("report_markdown", "")
        if not report_markdown:
            return {"error": "저장할 보고서가 없습니다."}
        return {"report_path": save_report(report_markdown)}

    def finalize_output(self, state: ComparisonReportState) -> ComparisonReportState:
        return {
            "final_output": {
                "paper_method_spec": state.get("paper_method_spec", {}),
                "paper_spec": state.get("paper_spec", {}),
                "execution_spec": state.get("execution_spec", {}),
                "comparison_table_markdown": state.get("comparison_table_markdown", ""),
                "analysis_markdown": state.get("analysis_markdown", ""),
                "summary_status": state.get("summary_status", ""),
                "summary_headline": state.get("summary_headline", ""),
                "summary_paragraphs": state.get("summary_paragraphs", []),
                "report_markdown": state.get("report_markdown", ""),
                "report_path": state.get("report_path", ""),
            }
        }

    def _parse_summary_payload(self, raw: str) -> dict[str, object]:
        import json

        text = str(raw or "").strip()
        if not text:
            return {}
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start : end + 1] if start >= 0 and end > start else text
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _sanitize_summary_line(self, value: object) -> str:
        import re

        text = str(value or "").strip()
        if not text:
            return ""
        code_like_patterns = [
            r"^spec\s*=",
            r"^assumptions\s*=",
            r"^def\s+\w+\s*\(",
            r"^class\s+\w+",
            r"^from\s+\S+\s+import\s+",
            r"^import\s+\S+",
        ]
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in code_like_patterns):
            return ""
        if "```" in text:
            return ""
        if (("{" in text and "}" in text) or ("[" in text and "]" in text)) and len(text) > 80:
            return ""
        return text

    def _fallback_ui_summary(self, state: ComparisonReportState) -> dict[str, object]:
        execution_spec = state.get("execution_spec", {})
        paper_spec = state.get("paper_spec", {})
        metrics = execution_spec.get("metrics", {}) if isinstance(execution_spec, dict) else {}
        paper_metrics = paper_spec.get("metrics", {}) if isinstance(paper_spec, dict) else {}

        headline = (
            "전반적으로 재현 결과를 비교할 수 있습니다."
            if execution_spec.get("status") == "success"
            else "실행이 완료되지 않아 재현 차이를 제한적으로만 해석할 수 있습니다."
        )
        paragraphs: list[str] = []
        if paper_metrics and metrics:
            available = [name for name in ["MAE", "RMSE", "MSE", "R2"] if name in paper_metrics and name in metrics]
            if available:
                paragraphs.append(
                    f"논문 보고 지표와 재현 지표를 비교해 {', '.join(available)} 기준의 차이를 확인했습니다."
                )
        paragraphs.append("논문에 보고되지 않은 세부 설정은 재현 과정에서 기본값이나 보완 설정을 사용했을 수 있습니다.")
        paragraphs.append("세부 차이는 전처리, feature 구성, 데이터 분할, 하이퍼파라미터 차이에서 발생했을 가능성이 있습니다.")
        return {"headline": headline, "paragraphs": paragraphs[:3]}

    def _infer_summary_status(self, state: ComparisonReportState) -> str:
        paper_spec = state.get("paper_spec", {})
        execution_spec = state.get("execution_spec", {})
        if execution_spec.get("status") != "success":
            return "failed"

        paper_metrics = paper_spec.get("metrics", {}) if isinstance(paper_spec, dict) else {}
        execution_metrics = execution_spec.get("metrics", {}) if isinstance(execution_spec, dict) else {}
        comparable_errors: list[float] = []
        for metric_name in ["MAE", "RMSE", "MSE", "R2"]:
            try:
                paper_value = float(paper_metrics.get(metric_name))
                execution_value = float(execution_metrics.get(metric_name))
            except (TypeError, ValueError):
                continue
            denominator = abs(paper_value) if paper_value not in {0.0, -0.0} else 1.0
            comparable_errors.append(abs(execution_value - paper_value) / denominator * 100.0)

        if not comparable_errors:
            return "limited"
        if max(comparable_errors) <= 5.0:
            return "good"
        if max(comparable_errors) <= 20.0:
            return "partial"
        if min(comparable_errors) <= 20.0:
            return "partial"
        return "poor"
