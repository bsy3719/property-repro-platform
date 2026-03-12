from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.utils import create_openai_client, run_text_response, sanitize_python_code

from .fallback_script import build_fallback_code
from .normalization import build_code_spec, fill_missing_details, parse_paper_info
from .prompting import build_generation_prompt
from .schemas import CodeGenerationState
from .validation import run_validation


class CodeGenerationAgent:
    """LangGraph agent that generates reproducible sklearn regression code."""

    def __init__(self, model_name: str = "gpt-5.2", max_retries: int = 2) -> None:
        self.client, self.model_name = create_openai_client(model_name)
        self.max_retries = max_retries
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(CodeGenerationState)
        graph.add_node("parse_paper_info", self.parse_paper_info)
        graph.add_node("fill_missing_details", self.fill_missing_details)
        graph.add_node("build_code_spec", self.build_code_spec)
        graph.add_node("generate_python_code", self.generate_python_code)
        graph.add_node("validate_code_requirements", self.validate_code_requirements)
        graph.add_node("finalize_output", self.finalize_output)

        graph.set_entry_point("parse_paper_info")
        graph.add_edge("parse_paper_info", "fill_missing_details")
        graph.add_edge("fill_missing_details", "build_code_spec")
        graph.add_edge("build_code_spec", "generate_python_code")
        graph.add_edge("generate_python_code", "validate_code_requirements")
        graph.add_conditional_edges(
            "validate_code_requirements",
            self._route_after_validation,
            {"retry": "generate_python_code", "finalize": "finalize_output"},
        )
        graph.add_edge("finalize_output", END)
        return graph.compile()

    def invoke(self, state: CodeGenerationState) -> CodeGenerationState:
        initial_state: CodeGenerationState = {"retry_count": 0, "assumptions": [], **state}
        return self.graph.invoke(initial_state)

    def parse_paper_info(self, state: CodeGenerationState) -> CodeGenerationState:
        normalized_spec, assumptions = parse_paper_info(state.get("raw_paper_info", {}))
        return {"normalized_spec": normalized_spec, "assumptions": assumptions}

    def fill_missing_details(self, state: CodeGenerationState) -> CodeGenerationState:
        normalized_spec, assumptions = fill_missing_details(
            state.get("normalized_spec", {}),
            list(state.get("assumptions", [])),
        )
        return {"normalized_spec": normalized_spec, "assumptions": assumptions}

    def build_code_spec(self, state: CodeGenerationState) -> CodeGenerationState:
        return {"code_spec": build_code_spec(state.get("normalized_spec", {}))}

    def generate_python_code(self, state: CodeGenerationState) -> CodeGenerationState:
        assumptions = list(state.get("assumptions", []))
        code_spec = state.get("code_spec", {})
        prompt = build_generation_prompt(code_spec, assumptions, state.get("validation_feedback", ""))

        try:
            generated_code = run_text_response(self.client, self.model_name, prompt, "코드 생성")
            generated_code = sanitize_python_code(generated_code)
        except RuntimeError as exc:
            assumptions.append(str(exc))
            generated_code = build_fallback_code(code_spec, assumptions)

        if not generated_code:
            generated_code = build_fallback_code(code_spec, assumptions)

        return {
            "generated_code": generated_code,
            "assumptions": assumptions,
            "final_output": {
                "code_generation_input_status": {
                    "has_paper_markdown": bool(str(code_spec.get("paper_markdown", "")).strip()),
                    "paper_markdown_chars": len(str(code_spec.get("paper_markdown", ""))),
                    "has_model_anchor_summary": bool(str(code_spec.get("model_anchor_summary", "")).strip()),
                    "model_anchor_summary_chars": len(str(code_spec.get("model_anchor_summary", ""))),
                    "has_paper_method_spec": bool(code_spec.get("paper_method_spec", {})),
                    "selected_model": str(code_spec.get("paper_method_spec", {}).get("model", {}).get("name", "")),
                }
            },
        }

    def validate_code_requirements(self, state: CodeGenerationState) -> CodeGenerationState:
        generated_code = sanitize_python_code(state.get("generated_code", ""))
        validation_result = run_validation(generated_code, state.get("code_spec", {}))
        retry_count = int(state.get("retry_count", 0))
        validation_feedback = ""
        if not validation_result["is_valid"]:
            missing_checks = ", ".join(validation_result["missing_requirements"])
            validation_feedback = f"The generated code is missing or failing these checks: {missing_checks}"
            retry_count += 1

        return {
            "generated_code": generated_code,
            "validation_result": validation_result,
            "validation_feedback": validation_feedback,
            "retry_count": retry_count,
        }

    def finalize_output(self, state: CodeGenerationState) -> CodeGenerationState:
        existing_final_output = state.get("final_output", {})
        return {
            "final_output": {
                **existing_final_output,
                "normalized_spec": state.get("normalized_spec", {}),
                "assumptions": state.get("assumptions", []),
                "code_spec": state.get("code_spec", {}),
                "generated_code": state.get("generated_code", ""),
                "validation_result": state.get("validation_result", {}),
            }
        }

    def _route_after_validation(self, state: CodeGenerationState) -> str:
        validation_result = state.get("validation_result", {})
        retry_count = int(state.get("retry_count", 0))
        if validation_result.get("is_valid"):
            return "finalize"
        if retry_count <= self.max_retries:
            return "retry"
        return "finalize"
