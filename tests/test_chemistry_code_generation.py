from __future__ import annotations

import unittest
from unittest.mock import patch

from src.agents.code_generation.agent import CodeGenerationAgent
from src.agents.code_generation.normalization import build_code_spec, fill_missing_details
from src.agents.code_generation.safety_net import build_safety_net_code
from src.utils.paper_method_spec import normalize_paper_method_spec


class ChemistryCodeGenerationTests(unittest.TestCase):
    def test_fill_missing_details_builds_smiles_only_feature_plan(self) -> None:
        normalized_spec = {
            "target_property": "boiling_point",
            "dataset": {
                "file_path": "data.csv",
                "sheet_name": None,
                "smiles_column": "smiles",
                "target_column": "boiling_point",
            },
            "paper_method_spec": normalize_paper_method_spec(
                {
                    "feature": {
                        "retained_input_features": ["mw", "mf", "polararea", "heavycnt", "hbondacc", "iso smiles"],
                        "derived_feature_names": ["C number", "N number", "O number", "side chain number"],
                        "class_label_names": ["hydrocarbon", "alcohol", "amine"],
                    },
                    "model": {"name": "RandomForestRegressor"},
                }
            ),
            "feature": {},
            "preprocessing": {},
            "model": {},
            "hyperparameters": {},
            "training": {},
            "metrics": ["MAE", "RMSE", "MSE", "R2"],
        }

        filled_spec, assumptions = fill_missing_details(normalized_spec, assumptions=[])
        code_spec = build_code_spec(filled_spec)

        self.assertEqual(code_spec["dataset"]["input_columns"], ["smiles", "boiling_point"])
        self.assertEqual(
            code_spec["feature_pipeline"]["exact_smiles_features"],
            ["mw", "mf", "polararea", "heavycnt", "hbondacc", "C number", "N number", "O number", "side chain number"],
        )
        self.assertEqual(code_spec["feature_pipeline"]["feature_source"], "smiles_only")
        self.assertFalse(any("컬럼" in assumption and "feature" in assumption for assumption in assumptions))

    @patch("src.agents.code_generation.agent.llm_review_code")
    @patch("src.agents.code_generation.agent.run_text_response")
    @patch("src.agents.code_generation.agent.create_openai_client")
    def test_generation_agent_resolves_full_feature_inputs(
        self,
        mock_client_factory,
        mock_run_text_response,
        mock_llm_review_code,
    ) -> None:
        mock_client_factory.return_value = (object(), "gpt-5.2")
        mock_run_text_response.return_value = "\n".join(
            [
                "SPEC = {}",
                "ASSUMPTIONS = {}",
                "def load_data(file_path: str): return file_path",
                "def build_feature_matrix(smiles_list): return [], []",
                "def train_model(X, y): return None",
                "def evaluate_model(model, X_test, y_test): return {}",
                "def main(): return None",
            ]
        )
        mock_llm_review_code.return_value = {"issues": [], "fixed_code": "", "had_issues": False}

        agent = CodeGenerationAgent()
        result = agent.invoke(
            {
                "raw_paper_info": {
                    "dataset": {"file_path": "data.csv", "smiles_column": "smiles", "target_column": "boiling_point"},
                    "paper_method_spec": {
                        "feature": {
                            "retained_input_features": ["cmpdname", "mw", "mf", "polararea", "iso smiles"],
                            "derived_feature_names": ["side chain number"],
                            "class_label_names": ["hydrocarbon", "alcohol"],
                        },
                        "model": {"name": "RandomForestRegressor"},
                    },
                }
            }
        )

        final_output = result["final_output"]
        feature_pipeline = final_output["code_spec"]["feature_pipeline"]
        resolution = feature_pipeline["feature_resolution"]

        self.assertIn("mw", feature_pipeline["resolver_input_features"])
        self.assertIn("mf", feature_pipeline["resolver_input_features"])
        self.assertIn("side chain number", feature_pipeline["resolver_input_features"])
        self.assertIn("hydrocarbon", feature_pipeline["resolver_input_features"])
        self.assertEqual(resolution["resolved"]["mw"], "MolWt")
        self.assertEqual(resolution["resolved"]["polararea"], "TPSA")
        self.assertIn("cmpdname", resolution["excluded"])
        self.assertIn("hydrocarbon", resolution["excluded"])

    @patch("src.agents.code_generation.agent.llm_review_code")
    @patch("src.agents.code_generation.agent.run_text_response")
    @patch("src.agents.code_generation.agent.create_openai_client")
    def test_generation_agent_resolves_atom_count_features(
        self,
        mock_client_factory,
        mock_run_text_response,
        mock_llm_review_code,
    ) -> None:
        mock_client_factory.return_value = (object(), "gpt-5.2")
        mock_run_text_response.return_value = "\n".join(
            [
                "SPEC = {}",
                "ASSUMPTIONS = {}",
                "def load_data(file_path: str): return file_path",
                "def build_feature_matrix(smiles_list): return [], []",
                "def train_model(X, y): return None",
                "def evaluate_model(model, X_test, y_test): return {}",
                "def main(): return None",
            ]
        )
        mock_llm_review_code.return_value = {"issues": [], "fixed_code": "", "had_issues": False}

        agent = CodeGenerationAgent()
        result = agent.invoke(
            {
                "raw_paper_info": {
                    "dataset": {"file_path": "data.csv", "smiles_column": "smiles", "target_column": "boiling_point"},
                    "paper_method_spec": {
                        "feature": {
                            "retained_input_features": ["mw"],
                            "derived_feature_names": ["C/N/O number", "O number"],
                        },
                        "model": {"name": "RandomForestRegressor"},
                    },
                }
            }
        )

        feature_pipeline = result["final_output"]["code_spec"]["feature_pipeline"]
        resolution = feature_pipeline["feature_resolution"]
        self.assertEqual(resolution["resolved_counts"]["C/N/O number"], ["C_count", "N_count", "O_count"])
        self.assertEqual(resolution["resolved_counts"]["O number"], ["O_count"])
        self.assertNotIn("C/N/O number", feature_pipeline["exact_smiles_features"])
        self.assertNotIn("O number", feature_pipeline["exact_smiles_features"])
        self.assertIn("C_count", feature_pipeline["count_feature_names"])
        self.assertIn("N_count", feature_pipeline["count_feature_names"])
        self.assertIn("O_count", feature_pipeline["count_feature_names"])

    def test_safety_net_code_uses_new_module_contract(self) -> None:
        code_spec = {
            "dataset": {
                "smiles_column": "smiles",
                "target_column": "boiling_point",
            },
            "feature_pipeline": {
                "descriptor_names": ["MolWt"],
            },
            "model": {"name": "random_forest", "hyperparameters": {"n_estimators": 10, "random_state": 42}},
            "training": {"test_size": 0.2, "random_state": 42},
        }

        code = build_safety_net_code(code_spec, ["MolWt"], assumptions=["기본 가정"])

        self.assertIn("ASSUMPTIONS =", code)
        self.assertIn("DESCRIPTOR_NAMES =", code)
        self.assertIn("최대 재시도 도달로 safety net 모드 실행", code)
        self.assertNotIn("fingerprint_family", code)


if __name__ == "__main__":
    unittest.main()
