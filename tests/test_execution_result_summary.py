from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from app.backend_core import format_run_timestamp, persist_run_outputs, run_generation_for_session
from src.agents.code_execution_agent import CodeExecutionAgent
from src.agents.comparison_report import ComparisonReportAgent


class ExecutionResultSummaryTests(unittest.TestCase):
    def test_code_execution_agent_normalizes_top_level_metric_keys(self) -> None:
        agent = CodeExecutionAgent()
        parsed = agent._parse_json_output(
            """
            {
              "mae": 10.5,
              "rmse": 14.2,
              "mse": 201.64,
              "r2": 0.81,
              "y_test": [1, 2, "x"],
              "y_pred": [1.1, 2.2, 3.3]
            }
            """
        )

        self.assertEqual(parsed["metrics"]["MAE"], 10.5)
        self.assertEqual(parsed["metrics"]["RMSE"], 14.2)
        self.assertEqual(parsed["metrics"]["MSE"], 201.64)
        self.assertEqual(parsed["metrics"]["R2"], 0.81)
        self.assertEqual(parsed["y_test"], [1.0, 2.0])
        self.assertEqual(parsed["y_pred"], [1.1, 2.2, 3.3])

    @patch("src.agents.comparison_report.agent.run_text_response")
    @patch("src.agents.comparison_report.agent.create_openai_client")
    def test_comparison_report_agent_builds_structured_ui_summary(
        self,
        mock_client_factory,
        mock_run_text_response,
    ) -> None:
        mock_client_factory.return_value = (object(), "gpt-5.2")
        mock_run_text_response.side_effect = [
            "## Overall Assessment\n요약 분석\n\n## Methodology Differences\n차이\n\n## Metric Differences\n차이\n\n## Likely Causes\n원인",
            """
            {
              "headline": "전반적으로 양호한 재현으로 평가됩니다.",
              "paragraphs": [
                "MAE와 RMSE는 논문 보고값과 비교 가능한 수준으로 확인되었습니다.",
                "일부 차이는 데이터 분할과 세부 하이퍼파라미터 미보고에서 비롯되었을 가능성이 있습니다.",
                "Feature 구성 차이가 재현 편차에 영향을 주었을 수 있습니다."
              ]
            }
            """,
        ]

        agent = ComparisonReportAgent()
        result = agent.invoke(
            {
                "paper_method_spec": {
                    "feature": {"method": "descriptor", "descriptor_names": ["MolWt", "TPSA"]},
                    "model": {"name": "RandomForestRegressor"},
                    "metrics": {"reported": {"MAE": 10.0, "RMSE": 20.0, "MSE": 400.0, "R2": 0.8}},
                },
                "execution_final_output": {
                    "execution_result": {
                        "status": "success",
                        "returncode": 0,
                        "metrics": {"MAE": 10.4, "RMSE": 22.0, "MSE": 484.0, "R2": 0.78},
                    }
                },
                "generated_code": """
SPEC = {
    "feature_pipeline": {
        "method": "descriptor",
        "descriptor_names": ["MolWt", "TPSA"],
        "count_feature_names": []
    },
    "model": {"name": "RandomForestRegressor", "hyperparameters": {}},
    "training": {"split_strategy": "train_test_split", "test_size": 0.2, "random_state": 42}
}
""",
                "generated_code_path": "artifacts/generated_code/generated_regression_latest.py",
            }
        )

        final_output = result["final_output"]
        self.assertEqual(final_output["summary_status"], "partial")
        self.assertEqual(final_output["summary_headline"], "전반적으로 양호한 재현으로 평가됩니다.")
        self.assertEqual(len(final_output["summary_paragraphs"]), 3)
        self.assertFalse(any("SPEC =" in paragraph for paragraph in final_output["summary_paragraphs"]))
        self.assertTrue(final_output["report_markdown"])

    def test_run_outputs_are_saved_with_agent_report_markdown(self) -> None:
        session = {
            "pdf_name": "paper.pdf",
            "paper_markdown_path": "artifacts/markdown_cache/paper.md",
            "data_name": "data.csv",
            "data_sheet_name": None,
            "num_rows": 123,
            "num_columns": 8,
            "smiles_column": "smiles",
            "target_column": "boiling_point",
            "paper_method_spec": {
                "selection_basis": {"summary": "보일링 포인트 예측 성능이 가장 좋은 모델을 선택함"},
                "preprocessing": {"missing_target": "drop", "invalid_smiles": "drop"},
                "feature": {
                    "method": "descriptor",
                    "descriptor_names": ["MolWt", "TPSA"],
                },
                "model": {"name": "RandomForestRegressor"},
                "training": {"split_strategy": "train_test_split", "test_size": 0.2},
                "hyperparameters": {"values": {"n_estimators": 100}},
            },
        }
        final_output = {
            "generated_code": "print('final')\n",
            "generation_result": {
                "generated_code": "print('generated')\n",
                "final_output": {
                    "assumptions": ["train/test split 방식이 명시되지 않아 80:20 분할을 사용함"],
                },
            },
            "execution_result": {
                "status": "success",
                "returncode": 0,
                "parsed_output": {
                    "assumptions": ["일부 파생 feature 컬럼을 입력 데이터에서 직접 찾지 못했습니다."]
                },
            },
            "comparison_report": {
                "report_markdown": "# 논문 대비 재현 결과 비교 보고서\n\n## 재현 분석 요약\n전반적으로 양호한 재현입니다.\n\n- 요약 문단 1\n- 요약 문단 2\n",
                "report_path": "artifacts/reports/comparison_report_existing.md",
            },
            "iteration": 1,
            "max_iterations": 4,
            "verification_status": "passed",
            "verification_issue_count": 0,
        }
        reproduction_summary = {
            "headline": "전반적으로 양호한 재현입니다.",
            "paragraphs": ["요약 문단 1", "요약 문단 2"],
            "status": "partial",
        }
        run_timestamp = format_run_timestamp(datetime(2024, 3, 15, 14, 22, 10))

        with tempfile.TemporaryDirectory() as tmp_dir:
            generated_dir = Path(tmp_dir) / "generated_code"
            reports_dir = Path(tmp_dir) / "reports"
            generated_dir.mkdir()
            reports_dir.mkdir()

            with patch("app.backend_core.GENERATED_CODE_DIR", generated_dir), patch("app.backend_core.REPORTS_DIR", reports_dir):
                saved = persist_run_outputs(
                    session=session,
                    final_output=final_output,
                    reproduction_summary=reproduction_summary,
                    run_timestamp=run_timestamp,
                )

            self.assertTrue(saved["generated_code_path"].endswith("generated_code_240315_142210.py"))
            self.assertTrue(saved["final_code_path"].endswith("final_code_240315_142210.py"))
            self.assertTrue(saved["reproduction_report_path"].endswith("reproduction_report_240315_142210.md"))
            self.assertEqual(saved["comparison_report_path"], "artifacts/reports/comparison_report_existing.md")
            report_text = Path(saved["reproduction_report_path"]).read_text(encoding="utf-8")
            self.assertIn("## 재현 분석 요약", report_text)
            self.assertIn("요약 문단 1", report_text)
            self.assertNotIn("## 논문 정보", report_text)

    @patch("app.backend_core.persist_run_outputs")
    @patch("app.backend_core.run_comparison_report")
    @patch("app.backend_core.run_code_loop")
    def test_run_generation_for_session_uses_report_agent_summary(
        self,
        mock_run_code_loop,
        mock_run_comparison_report,
        mock_persist_run_outputs,
    ) -> None:
        mock_run_code_loop.return_value = {
            "final_output": {
                "generated_code": "print('ok')\n",
                "generated_code_path": "artifacts/generated_code/final.py",
                "execution_result": {"status": "success", "returncode": 0},
            }
        }
        mock_run_comparison_report.return_value = {
            "final_output": {
                "summary_status": "partial",
                "summary_headline": "전반적으로 양호한 재현으로 평가됩니다.",
                "summary_paragraphs": [
                    "주요 지표는 비교 가능한 수준으로 확인되었습니다.",
                    "일부 차이는 세부 설정 미보고에서 비롯되었을 수 있습니다.",
                ],
                "report_markdown": "# 비교 보고서\n\n## 재현 분석 요약\n전반적으로 양호한 재현으로 평가됩니다.\n",
                "report_path": "artifacts/reports/comparison_report_1.md",
                "comparison_table_markdown": "| Item | Paper | Reproduced |",
                "analysis_markdown": "## Overall Assessment\n분석",
            }
        }
        mock_persist_run_outputs.return_value = {
            "generated_code_path": "artifacts/generated_code/generated_code_1.py",
            "final_code_path": "artifacts/generated_code/final_code_1.py",
            "reproduction_report_path": "artifacts/reports/reproduction_report_1.md",
            "comparison_report_path": "artifacts/reports/comparison_report_1.md",
        }

        session = {
            "paper_method_spec": {"model": {"name": "RandomForestRegressor"}},
            "data_path": "data/raw/data.xlsx",
            "data_sheet_name": "Sheet1",
            "smiles_column": "smiles",
            "target_column": "boiling_point",
            "data_columns": ["smiles", "boiling_point"],
            "pdf_name": "paper.pdf",
            "data_name": "data.xlsx",
        }

        result = run_generation_for_session(session)

        final_output = result["final_output"]
        self.assertEqual(final_output["reproduction_summary"]["headline"], "전반적으로 양호한 재현으로 평가됩니다.")
        self.assertEqual(final_output["reproduction_summary"]["status"], "partial")
        self.assertEqual(
            final_output["reproduction_summary"]["paragraphs"],
            [
                "주요 지표는 비교 가능한 수준으로 확인되었습니다.",
                "일부 차이는 세부 설정 미보고에서 비롯되었을 수 있습니다.",
            ],
        )
        self.assertEqual(final_output["comparison_report"]["report_path"], "artifacts/reports/comparison_report_1.md")
        self.assertEqual(final_output["saved_artifacts"]["comparison_report_path"], "artifacts/reports/comparison_report_1.md")


if __name__ == "__main__":
    unittest.main()
