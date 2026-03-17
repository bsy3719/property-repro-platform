from __future__ import annotations

import unittest

from src.utils.spec_builder import (
    assemble_paper_method_spec,
    build_selected_model_terms,
    filter_rows_for_selected_model,
    validate_paper_method_spec_contract,
)


class RagSpecPipelineTests(unittest.TestCase):
    def test_filter_rows_for_selected_model_prefers_matching_rows(self) -> None:
        rows = [
            {"chunk_id": 1, "text": "The final Random Forest regressor used MolWt and TPSA."},
            {"chunk_id": 2, "text": "SVR with ECFP4 was also evaluated."},
        ]

        filtered = filter_rows_for_selected_model(
            rows,
            model_name="RandomForestRegressor",
            selected_model_terms=build_selected_model_terms("RandomForestRegressor", ["random forest"]),
            topic="feature",
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["chunk_id"], 1)

    def test_filter_rows_for_selected_model_keeps_model_less_dataset_feature_rows(self) -> None:
        rows = [
            {"chunk_id": 10, "text": "The final Random Forest regressor predicted boiling point for the unknown dataset."},
            {
                "chunk_id": 11,
                "text": (
                    "The data cleaning process leaves only necessary characteristics "
                    "(cmpdname, mw, mf, polararea, heavycnt, hbondacc, iso-smiles)."
                ),
            },
            {"chunk_id": 12, "text": "SVR with ECFP4 was also evaluated for boiling point regression."},
        ]

        filtered = filter_rows_for_selected_model(
            rows,
            model_name="RandomForestRegressor",
            selected_model_terms=build_selected_model_terms("RandomForestRegressor", ["random forest"]),
            topic="feature",
        )

        self.assertEqual([row["chunk_id"] for row in filtered], [10, 11])

    def test_assemble_spec_normalizes_feature_details(self) -> None:
        selection_result = {
            "selection_basis": {
                "summary": "Random forest was the final best model.",
                "key_values": "final best model",
                "evidence_chunks": ["chunk-0"],
                "evidence_snippet": "Random forest achieved the best performance.",
            },
            "model": {
                "summary": "Random forest regressor",
                "key_values": "Random forest",
                "name": "RandomForestRegressor",
                "evidence_chunks": ["chunk-0"],
                "evidence_snippet": "Random forest achieved the best performance.",
            },
        }
        feature_result = {
            "feature": {
                "summary": "The final model used 12 characteristics as input features.",
                "key_values": "12 characteristics",
                "raw_feature_mentions": ["12 characteristics"],
                "raw_tool_mentions": [],
                "evidence_chunks": ["chunk-1", "chunk-2"],
                "evidence_snippet": "The dataset contains 12 characteristics without boiling point.",
            }
        }
        method_result = {
            "preprocessing": {
                "summary": "Invalid SMILES and duplicates were dropped.",
                "key_values": "invalid_smiles=drop, duplicates=drop",
                "invalid_smiles": "drop",
                "missing_target": "drop",
                "missing_features": "median_impute",
                "duplicates": "drop",
                "scaling": None,
                "evidence_chunks": ["chunk-2"],
                "evidence_snippet": "Invalid SMILES and duplicates were removed.",
            },
            "hyperparameters": {
                "summary": "n_estimators=300",
                "key_values": "n_estimators=300",
                "values": {"n_estimators": 300},
                "evidence_chunks": ["chunk-3"],
                "evidence_snippet": "The random forest used 300 trees.",
            },
            "training": {
                "summary": "train_test_split with test_size=0.2",
                "key_values": "split=train_test_split, test_size=0.2, random_state=42",
                "split_strategy": "train_test_split",
                "test_size": 0.2,
                "random_state": 42,
                "evidence_chunks": ["chunk-4"],
                "evidence_snippet": "A test size of 0.2 and random_state 42 were used.",
            },
            "metrics": {
                "summary": "MAE and RMSE were reported.",
                "key_values": "MAE=12.0, RMSE=15.0",
                "reported": {"MAE": 12.0, "RMSE": 15.0},
                "evidence_chunks": ["chunk-5"],
                "evidence_snippet": "MAE was 12.0 and RMSE was 15.0.",
            },
        }
        filtered_by_topic = {
            "feature": [{"chunk_id": 1, "text": "The unknown dataset has 12 characteristics without boiling point."}],
            "preprocessing": [{"chunk_id": 2, "text": "Invalid SMILES and duplicates were removed."}],
            "hyperparameter": [{"chunk_id": 3, "text": "The random forest used 300 trees."}],
            "training": [{"chunk_id": 4, "text": "test_size=0.2 and random_state=42"}],
            "metrics": [{"chunk_id": 5, "text": "MAE 12.0 RMSE 15.0"}],
            "model": [{"chunk_id": 0, "text": "Random forest achieved the best performance."}],
        }
        feature_rows = [
            {"chunk_id": 1, "text": "The unknown dataset has 12 characteristics without boiling point."},
            {
                "chunk_id": 2,
                "text": (
                    "The data cleaning process leaves only necessary characteristics "
                    "(cmpdname, mw, mf, polararea, heavycnt, hbondacc, iso-smiles). "
                    "Through the data transformation process, 4 characteristics "
                    "(C number, N number, O number and side chain number) were added. "
                    "Labeling compounds as hydrocarbons, alcohols, and amines was performed."
                ),
            },
        ]

        spec = assemble_paper_method_spec(
            selection_result=selection_result,
            feature_result=feature_result,
            method_result=method_result,
            filtered_by_topic=filtered_by_topic,
            feature_rows=feature_rows,
        )

        self.assertEqual(spec["model"]["name"], "RandomForestRegressor")
        self.assertEqual(spec["feature"]["method"], "descriptor")
        self.assertEqual(
            spec["feature"]["retained_input_features"],
            ["cmpdname", "mw", "mf", "polararea", "heavycnt", "hbondacc", "iso smiles"],
        )
        self.assertEqual(
            spec["feature"]["derived_feature_names"],
            ["C number", "N number", "O number", "side chain number"],
        )
        self.assertEqual(spec["feature"]["class_label_names"], ["hydrocarbon", "alcohol", "amine"])
        self.assertEqual(spec["feature"]["dataset_feature_count"], 12)

    def test_validate_spec_contract_warns_for_unresolved_features(self) -> None:
        spec = {
            "model": {"name": "RandomForestRegressor"},
            "feature": {
                "method": "descriptor",
                "descriptor_names": ["MolWt"],
                "unresolved_feature_terms": ["Mordred"],
            },
        }

        validation = validate_paper_method_spec_contract(
            spec=spec,
            selected_model_name="RandomForestRegressor",
            selected_model_terms=build_selected_model_terms("RandomForestRegressor"),
            filtered_by_topic={"feature": [{"chunk_id": 1, "text": "Mordred and MolWt were mentioned."}]},
        )

        self.assertTrue(validation["is_valid"])
        self.assertTrue(any("unresolved terms" in warning for warning in validation["warnings"]))


if __name__ == "__main__":
    unittest.main()
