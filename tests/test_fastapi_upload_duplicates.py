from __future__ import annotations

import hashlib
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.fastapi_server import SESSIONS, app


class FastApiUploadDuplicateTests(unittest.TestCase):
    def setUp(self) -> None:
        SESSIONS.clear()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        SESSIONS.clear()

    @patch("app.fastapi_server.build_session_from_upload")
    def test_duplicate_upload_reuses_existing_session(self, mock_build_session_from_upload) -> None:
        pdf_bytes = b"%PDF-1.4 test pdf"
        data_bytes = b"smiles,boiling_point\nCC,100\n"
        mock_build_session_from_upload.return_value = {
            "session_id": "session-1",
            "pdf_name": "paper.pdf",
            "pdf_path": "data/raw/paper.pdf",
            "pdf_hash": hashlib.sha256(pdf_bytes).hexdigest(),
            "data_name": "data.csv",
            "data_path": "data/raw/data.csv",
            "data_hash": hashlib.sha256(data_bytes).hexdigest(),
            "data_sheet_name": None,
            "data_columns": ["smiles", "boiling_point"],
            "column_detection": {},
        }

        files = {
            "pdf_file": ("paper.pdf", pdf_bytes, "application/pdf"),
            "data_file": ("data.csv", data_bytes, "text/csv"),
        }

        first = self.client.post("/api/upload", files=files)
        second = self.client.post("/api/upload", files=files)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertFalse(first.json()["duplicate_upload"])
        self.assertTrue(second.json()["duplicate_upload"])
        self.assertEqual(first.json()["session"]["session_id"], "session-1")
        self.assertEqual(second.json()["session"]["session_id"], "session-1")
        self.assertEqual(mock_build_session_from_upload.call_count, 1)

    @patch("app.fastapi_server.build_session_from_upload")
    def test_same_files_with_different_sheet_do_not_reuse_session(self, mock_build_session_from_upload) -> None:
        pdf_bytes = b"%PDF-1.4 test pdf"
        data_bytes = b"excel-binary"
        mock_build_session_from_upload.side_effect = [
            {
                "session_id": "session-1",
                "pdf_name": "paper.pdf",
                "pdf_path": "data/raw/paper.pdf",
                "pdf_hash": hashlib.sha256(pdf_bytes).hexdigest(),
                "data_name": "data.xlsx",
                "data_path": "data/raw/data.xlsx",
                "data_hash": hashlib.sha256(data_bytes).hexdigest(),
                "data_sheet_name": "Sheet1",
                "data_columns": ["smiles", "boiling_point"],
                "column_detection": {},
            },
            {
                "session_id": "session-2",
                "pdf_name": "paper.pdf",
                "pdf_path": "data/raw/paper.pdf",
                "pdf_hash": hashlib.sha256(pdf_bytes).hexdigest(),
                "data_name": "data.xlsx",
                "data_path": "data/raw/data.xlsx",
                "data_hash": hashlib.sha256(data_bytes).hexdigest(),
                "data_sheet_name": "Sheet2",
                "data_columns": ["smiles", "boiling_point"],
                "column_detection": {},
            },
        ]

        files = {
            "pdf_file": ("paper.pdf", pdf_bytes, "application/pdf"),
            "data_file": ("data.xlsx", data_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        }

        first = self.client.post("/api/upload", files=files, data={"sheet_name": "Sheet1"})
        second = self.client.post("/api/upload", files=files, data={"sheet_name": "Sheet2"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertFalse(first.json()["duplicate_upload"])
        self.assertFalse(second.json()["duplicate_upload"])
        self.assertEqual(first.json()["session"]["session_id"], "session-1")
        self.assertEqual(second.json()["session"]["session_id"], "session-2")
        self.assertEqual(mock_build_session_from_upload.call_count, 2)


if __name__ == "__main__":
    unittest.main()
