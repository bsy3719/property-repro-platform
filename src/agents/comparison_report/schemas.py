from __future__ import annotations

from typing import Any, TypedDict

MODEL_CLASS_NAMES = [
    "RandomForestRegressor",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "SVR",
    "KNeighborsRegressor",
    "MLPRegressor",
]
MODEL_ALIASES = {
    "random forest": "RandomForestRegressor",
    "randomforest": "RandomForestRegressor",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elastic net": "ElasticNet",
    "elasticnet": "ElasticNet",
    "svr": "SVR",
    "support vector regression": "SVR",
    "support vector regressor": "SVR",
    "k-nearest neighbors": "KNeighborsRegressor",
    "kneighbors": "KNeighborsRegressor",
    "mlp": "MLPRegressor",
    "multi-layer perceptron": "MLPRegressor",
}


class ComparisonReportState(TypedDict, total=False):
    paper_summary_markdown: str
    paper_method_spec: dict[str, Any]
    execution_final_output: dict[str, Any]
    generated_code: str
    generated_code_path: str
    report_context: dict[str, Any]
    paper_spec: dict[str, Any]
    execution_spec: dict[str, Any]
    comparison_table_markdown: str
    analysis_markdown: str
    summary_status: str
    summary_headline: str
    summary_paragraphs: list[str]
    report_markdown: str
    report_path: str
    final_output: dict[str, Any]
    error: str
