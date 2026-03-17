from __future__ import annotations

__all__ = [
    "RDKIT_DESCRIPTOR_SNAPSHOT",
    "analyze_feature_text",
    "build_evidence_first_feature_contract",
    "augment_feature_payload",
    "merge_unique",
    "match_dataset_feature_columns",
    "normalize_count_feature_names",
    "normalize_descriptor_names",
    "normalize_exact_feature_terms",
    "normalize_fingerprint_family",
    "normalize_string_list",
    "sanitize_python_code",
    "create_openai_client",
    "run_text_response",
    "build_paper_method_summary_markdown",
    "canonical_model_name",
    "extract_json_object",
    "has_meaningful_paper_method_spec",
    "internal_model_name",
    "normalize_feature_method",
    "normalize_paper_method_spec",
    "paper_method_spec_to_comparison_spec",
]


def __getattr__(name: str):
    if name in {
        "RDKIT_DESCRIPTOR_SNAPSHOT",
        "analyze_feature_text",
        "build_evidence_first_feature_contract",
        "augment_feature_payload",
        "merge_unique",
        "match_dataset_feature_columns",
        "normalize_count_feature_names",
        "normalize_descriptor_names",
        "normalize_exact_feature_terms",
        "normalize_fingerprint_family",
        "normalize_string_list",
    }:
        from .chemistry_features import (
            RDKIT_DESCRIPTOR_SNAPSHOT,
            analyze_feature_text,
            build_evidence_first_feature_contract,
            augment_feature_payload,
            merge_unique,
            match_dataset_feature_columns,
            normalize_count_feature_names,
            normalize_descriptor_names,
            normalize_exact_feature_terms,
            normalize_fingerprint_family,
            normalize_string_list,
        )

        exports = {
            "RDKIT_DESCRIPTOR_SNAPSHOT": RDKIT_DESCRIPTOR_SNAPSHOT,
            "analyze_feature_text": analyze_feature_text,
            "build_evidence_first_feature_contract": build_evidence_first_feature_contract,
            "augment_feature_payload": augment_feature_payload,
            "merge_unique": merge_unique,
            "match_dataset_feature_columns": match_dataset_feature_columns,
            "normalize_count_feature_names": normalize_count_feature_names,
            "normalize_descriptor_names": normalize_descriptor_names,
            "normalize_exact_feature_terms": normalize_exact_feature_terms,
            "normalize_fingerprint_family": normalize_fingerprint_family,
            "normalize_string_list": normalize_string_list,
        }
        return exports[name]

    if name == "sanitize_python_code":
        from .code_text import sanitize_python_code

        return sanitize_python_code

    if name in {"create_openai_client", "run_text_response"}:
        from .openai_client import create_openai_client, run_text_response

        exports = {
            "create_openai_client": create_openai_client,
            "run_text_response": run_text_response,
        }
        return exports[name]

    if name in {
        "build_paper_method_summary_markdown",
        "canonical_model_name",
        "extract_json_object",
        "has_meaningful_paper_method_spec",
        "internal_model_name",
        "normalize_feature_method",
        "normalize_paper_method_spec",
        "paper_method_spec_to_comparison_spec",
    }:
        from .paper_method_spec import (
            build_paper_method_summary_markdown,
            canonical_model_name,
            extract_json_object,
            has_meaningful_paper_method_spec,
            internal_model_name,
            normalize_feature_method,
            normalize_paper_method_spec,
            paper_method_spec_to_comparison_spec,
        )

        exports = {
            "build_paper_method_summary_markdown": build_paper_method_summary_markdown,
            "canonical_model_name": canonical_model_name,
            "extract_json_object": extract_json_object,
            "has_meaningful_paper_method_spec": has_meaningful_paper_method_spec,
            "internal_model_name": internal_model_name,
            "normalize_feature_method": normalize_feature_method,
            "normalize_paper_method_spec": normalize_paper_method_spec,
            "paper_method_spec_to_comparison_spec": paper_method_spec_to_comparison_spec,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
