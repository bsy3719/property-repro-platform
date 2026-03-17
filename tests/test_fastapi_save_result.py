from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.fastapi_server import RESULTS_DIR, SESSIONS, app


class FastApiSaveResultTests(unittest.TestCase):
    def setUp(self) -> None:
        SESSIONS.clear()
        self.client = TestClient(app)
        SESSIONS["session-1"] = {
            "session_id": "session-1",
            "pdf_hash": "abc123456789",
            "data_hash": "def987654321",
            "pdf_name": "paper.pdf",
            "data_name": "data.csv",
            "paper_method_spec": {
                "model": {"name": "RandomForestRegressor", "summary": "Random Forest"},
                "feature": {"method": "descriptor", "use_rdkit_descriptors": True},
                "metrics": {"reported": {"MAE": 1.2, "RMSE": 2.3, "MSE": 5.29, "R2": 0.81}},
            },
        }

    def tearDown(self) -> None:
        SESSIONS.clear()

    def test_save_result_writes_sample_shaped_json(self) -> None:
        payload = {
            "session_id": "session-1",
            "final_output": {
                "execution_result": {
                    "metrics": {"MAE": 1.4, "RMSE": 2.5, "MSE": 6.25, "R2": 0.79},
                    "parsed_output": {
                        "assumptions": ["train/test split: 80/20", "random_state=42"],
                    },
                },
                "reproduction_summary": {
                    "status": "partial",
                    "headline": "부분 재현",
                    "paragraphs": ["문단 1", "문단 2"],
                    "paper_metrics": {"MAE": 1.2, "RMSE": 2.3, "MSE": 5.29, "R2": 0.81},
                    "reproduced_metrics": {"MAE": 1.4, "RMSE": 2.5, "MSE": 6.25, "R2": 0.79},
                },
                "comparison_report": {
                    "report_markdown": "## 재현 비교\n| 지표 | 논문 | 재현 |",
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("app.fastapi_server.RESULTS_DIR", Path(tmp_dir)):
                response = self.client.post("/api/save-result", json=payload)

                self.assertEqual(response.status_code, 200)
                body = response.json()
                saved_path = Path(body["path"])
                self.assertTrue(saved_path.exists())

                record = json.loads(saved_path.read_text(encoding="utf-8"))
                self.assertEqual(record["result_id"], body["result_id"])
                self.assertEqual(record["pdf_name"], "paper.pdf")
                self.assertEqual(record["data_name"], "data.csv")
                self.assertEqual(record["model_name"], "RandomForestRegressor")
                self.assertEqual(record["feature_method"], "descriptor")
                self.assertEqual(record["paper_metrics"]["MAE"], 1.2)
                self.assertEqual(record["reproduced_metrics"]["RMSE"], 2.5)
                self.assertEqual(record["reproduction_status"], "partial")
                self.assertEqual(record["reproduction_headline"], "부분 재현")
                self.assertEqual(record["reproduction_paragraphs"], ["문단 1", "문단 2"])
                self.assertEqual(record["assumptions"], ["train/test split: 80/20", "random_state=42"])
                self.assertIn("comparison_report_markdown", record)

    def test_save_result_generates_unique_result_ids(self) -> None:
        payload = {
            "session_id": "session-1",
            "final_output": {
                "execution_result": {"metrics": {"MAE": 1.0, "RMSE": 2.0, "MSE": 4.0, "R2": 0.8}},
                "reproduction_summary": {"status": "good"},
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("app.fastapi_server.RESULTS_DIR", Path(tmp_dir)):
                first = self.client.post("/api/save-result", json=payload)
                second = self.client.post("/api/save-result", json=payload)

                self.assertEqual(first.status_code, 200)
                self.assertEqual(second.status_code, 200)
                self.assertNotEqual(first.json()["result_id"], second.json()["result_id"])
                self.assertEqual(len(list(Path(tmp_dir).glob("*.json"))), 2)


if __name__ == "__main__":
    unittest.main()
