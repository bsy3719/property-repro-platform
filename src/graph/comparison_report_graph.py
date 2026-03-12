from __future__ import annotations

from typing import Any

from src.agents.comparison_report_agent import ComparisonReportAgent, ComparisonReportState


def build_comparison_report_agent(model_name: str = "gpt-5.2") -> ComparisonReportAgent:
    return ComparisonReportAgent(model_name=model_name)


def run_comparison_report(payload: dict[str, Any]) -> ComparisonReportState:
    agent = build_comparison_report_agent()
    return agent.invoke(payload)
