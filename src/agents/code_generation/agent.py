from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.utils import create_openai_client, run_text_response, sanitize_python_code

from .feature_resolver import FeatureResolver
from .normalization import build_code_spec, fill_missing_details, parse_paper_info
from .prompts.code_gen_prompt import build_full_prompt
from .schemas import CodeGenerationState
from .validation import ast_syntax_check, llm_review_code


class CodeGenerationAgent:
    """SMILES-only code generation agent driven by paper_method_spec."""

    def __init__(self, model_name: str = "gpt-5.2", max_retries: int = 0) -> None:
        self.max_retries = max_retries
        self.client, self.model_name = create_openai_client(model_name)
        self.feature_resolver = FeatureResolver(model_name=model_name)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(CodeGenerationState)
        graph.add_node("parse_paper_info", self.parse_paper_info)
        graph.add_node("fill_missing_details", self.fill_missing_details)
        graph.add_node("build_code_spec", self.build_code_spec)
        graph.add_node("resolve_features", self.resolve_features)
        graph.add_node("generate_python_code", self.generate_python_code)
        graph.add_node("review_code", self.review_code)
        graph.add_node("check_syntax", self.check_syntax)
        graph.add_node("finalize_output", self.finalize_output)

        graph.set_entry_point("parse_paper_info")
        graph.add_edge("parse_paper_info", "fill_missing_details")
        graph.add_edge("fill_missing_details", "build_code_spec")
        graph.add_edge("build_code_spec", "resolve_features")
        graph.add_edge("resolve_features", "generate_python_code")
        graph.add_edge("generate_python_code", "review_code")
        graph.add_edge("review_code", "check_syntax")
        graph.add_edge("check_syntax", "finalize_output")
        graph.add_edge("finalize_output", END)
        return graph.compile()

    def invoke(self, state: CodeGenerationState) -> CodeGenerationState:
        return self.graph.invoke({"retry_count": 0, "assumptions": [], **state})

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

    def resolve_features(self, state: CodeGenerationState) -> CodeGenerationState:
        code_spec = dict(state.get("code_spec", {}))
        assumptions = list(state.get("assumptions", []))
        feature_pipeline = dict(code_spec.get("feature_pipeline", {}))
        paper_feature_spec = dict(code_spec.get("paper_method_spec", {}).get("feature", {}) or {})
        dataset_spec = dict(code_spec.get("dataset", {}) or {})

        # 해석 대상: retained / derived / descriptor / count / exact feature 전체
        descriptor_names: list[str] = list(feature_pipeline.get("descriptor_names") or [])
        exact_smiles_features: list[str] = list(feature_pipeline.get("exact_smiles_features") or [])
        count_feature_names: list[str] = list(feature_pipeline.get("count_feature_names") or [])
        retained_input_features: list[str] = list(paper_feature_spec.get("retained_input_features") or [])
        derived_feature_names: list[str] = list(paper_feature_spec.get("derived_feature_names") or [])
        class_label_names: list[str] = list(
            feature_pipeline.get("class_label_names")
            or paper_feature_spec.get("class_label_names")
            or []
        )

        resolver_inputs: list[str] = []
        for names in [
            retained_input_features,
            derived_feature_names,
            exact_smiles_features,
            descriptor_names,
            count_feature_names,
            class_label_names,
        ]:
            for name in names:
                normalized_name = str(name).strip()
                if normalized_name and normalized_name not in resolver_inputs:
                    resolver_inputs.append(normalized_name)

        resolution = self.feature_resolver.resolve(
            resolver_inputs,
            class_label_names=class_label_names,
            target_columns=[str(dataset_spec.get("target_column", "")).strip()],
        )

        # resolved 이름으로 descriptor_names / count_feature_names 갱신 (중복 제거, 순서 유지)
        resolved_map: dict[str, str] = resolution.get("resolved", {})
        resolved_count_map: dict[str, list[str]] = resolution.get("resolved_counts", {})
        excluded_set: set[str] = set(resolution.get("excluded", []))
        updated_descriptors: list[str] = []
        seen: set[str] = set()
        for name in descriptor_names:
            if name in excluded_set:
                continue
            rdkit_name = resolved_map.get(name, name)
            if rdkit_name not in seen:
                updated_descriptors.append(rdkit_name)
                seen.add(rdkit_name)

        updated_exact_features = [
            name
            for name in exact_smiles_features
            if name not in excluded_set and name not in resolved_count_map
        ]
        updated_count_features: list[str] = []
        seen_counts: set[str] = set()

        for name in count_feature_names:
            if name in excluded_set:
                continue
            for resolved_name in resolved_count_map.get(name, [name]):
                if resolved_name not in seen_counts:
                    updated_count_features.append(resolved_name)
                    seen_counts.add(resolved_name)

        for name in resolver_inputs:
            if name in excluded_set or name not in resolved_count_map:
                continue
            for resolved_name in resolved_count_map[name]:
                if resolved_name not in seen_counts:
                    updated_count_features.append(resolved_name)
                    seen_counts.add(resolved_name)

        feature_pipeline["descriptor_names"] = updated_descriptors
        feature_pipeline["exact_smiles_features"] = updated_exact_features
        feature_pipeline["count_feature_names"] = updated_count_features
        feature_pipeline["feature_resolution"] = resolution
        feature_pipeline["resolver_input_features"] = resolver_inputs
        code_spec["feature_pipeline"] = feature_pipeline

        assumptions.extend(resolution.get("assumptions", []))
        return {
            "code_spec": code_spec,
            "feature_resolution": resolution,
            "assumptions": assumptions,
        }

    def generate_python_code(self, state: CodeGenerationState) -> CodeGenerationState:
        assumptions = list(state.get("assumptions", []))
        code_spec = state.get("code_spec", {})
        feature_resolution = state.get("feature_resolution")
        prompt = build_full_prompt(code_spec, assumptions, feature_resolution)
        raw = run_text_response(self.client, self.model_name, prompt, "코드 생성")
        generated_code = sanitize_python_code(raw)
        return {
            "generated_code": generated_code,
            "assumptions": assumptions,
        }

    def review_code(self, state: CodeGenerationState) -> CodeGenerationState:
        """Step 1: LLM이 6가지 항목 검토 → 문제 있으면 fixed_code로 교체."""
        code = state.get("generated_code", "")
        review = llm_review_code(self.client, self.model_name, code)
        if review["had_issues"] and review["fixed_code"]:
            code = review["fixed_code"]
        return {
            "generated_code": code,
            "review_result": review,
        }

    def check_syntax(self, state: CodeGenerationState) -> CodeGenerationState:
        """Step 2: ast.parse()로 구문 오류만 확인."""
        code = state.get("generated_code", "")
        syntax = ast_syntax_check(code)
        review = state.get("review_result", {})
        is_valid = syntax["ok"]
        validation_feedback = ""
        if not is_valid:
            validation_feedback = (
                f"구문 오류 (line {syntax['line']}, col {syntax['col']}): {syntax['error']}"
            )
            if syntax.get("context"):
                validation_feedback += f"\n  {syntax['context']}"
        return {
            "validation_result": {
                "is_valid": is_valid,
                "syntax": syntax,
                "review": {
                    "issues": review.get("issues", []),
                    "had_issues": review.get("had_issues", False),
                },
            },
            "validation_feedback": validation_feedback,
        }

    def finalize_output(self, state: CodeGenerationState) -> CodeGenerationState:
        validation_result = state.get("validation_result", {})
        return {
            "generated_code": state.get("generated_code", ""),
            "validation_feedback": state.get("validation_feedback", ""),
            "final_output": {
                "normalized_spec": state.get("normalized_spec", {}),
                "assumptions": state.get("assumptions", []),
                "code_spec": state.get("code_spec", {}),
                "generated_code": state.get("generated_code", ""),
                "validation_result": validation_result,
            },
        }
