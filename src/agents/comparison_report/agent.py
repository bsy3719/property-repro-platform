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
        graph.add_node("compose_report", self.compose_report)
        graph.add_node("save_report", self.save_report)
        graph.add_node("finalize_output", self.finalize_output)

        graph.set_entry_point("parse_paper_summary")
        graph.add_edge("parse_paper_summary", "parse_execution_result")
        graph.add_edge("parse_execution_result", "build_comparison_rows")
        graph.add_edge("build_comparison_rows", "analyze_differences")
        graph.add_edge("analyze_differences", "compose_report")
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

    def compose_report(self, state: ComparisonReportState) -> ComparisonReportState:
        execution_spec = state.get("execution_spec", {})
        report_markdown = compose_report(
            execution_spec,
            state.get("comparison_table_markdown", ""),
            state.get("analysis_markdown", ""),
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
                "report_markdown": state.get("report_markdown", ""),
                "report_path": state.get("report_path", ""),
            }
        }
