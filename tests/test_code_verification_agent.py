from __future__ import annotations

import unittest
from unittest.mock import patch

from src.agents.code_generation.safety_net import build_safety_net_code
from src.agents.code_generation.validation import run_validation
from src.agents.code_loop_agent import CodeGenerationRunDebugAgent


def build_smiles_only_code_spec() -> dict:
    return {
        "task_type": "regression",
        "target_property": "boiling_point",
        "paper_method_spec": {},
        "dataset": {
            "file_path": "data.csv",
            "sheet_name": None,
            "smiles_column": "smiles",
            "target_column": "boiling_point",
            "input_columns": ["smiles", "boiling_point"],
        },
        "feature_pipeline": {
            "feature_source": "smiles_only",
            "input_column": "smiles",
            "method": "descriptor",
            "exact_smiles_features": [],
            "descriptor_names": ["MolWt"],
            "count_feature_names": [],
            "class_label_names": [],
            "feature_terms": [],
            "unresolved_feature_terms": [],
        },
        "preprocessing_pipeline": {
            "invalid_smiles": "drop",
            "missing_target": "drop",
            "missing_features": "median_impute",
            "duplicates": "drop",
            "scaling": False,
        },
        "model": {
            "name": "random_forest",
            "hyperparameters": {"n_estimators": 10, "random_state": 42},
        },
        "training": {
            "split_strategy": "train_test_split",
            "test_size": 0.2,
            "random_state": 42,
        },
        "metrics": ["MAE", "RMSE", "MSE", "R2"],
    }


class CodeLoopTests(unittest.TestCase):
    def test_safety_net_code_passes_basic_validation(self) -> None:
        code_spec = build_smiles_only_code_spec()
        code = build_safety_net_code(code_spec, ["MolWt"], assumptions=[])

        validation = run_validation(code, code_spec)

        self.assertTrue(validation["is_valid"], validation)

    @patch("src.agents.code_loop_agent.run_text_response", return_value="")
    @patch("src.agents.code_loop_agent.CodeExecutionAgent")
    @patch("src.agents.code_loop_agent.CodeGenerationAgent")
    @patch("src.agents.code_loop_agent.create_openai_client")
    def test_loop_retries_without_deterministic_fallback(
        self,
        mock_client_factory,
        mock_generator_cls,
        mock_executor_cls,
        _mock_run_text_response,
    ) -> None:
        code_spec = build_smiles_only_code_spec()
        generated_code = "\n".join(
            [
                "SPEC = {}",
                "ASSUMPTIONS = {}",
                "def load_data(file_path): return file_path",
                "def build_feature_matrix(smiles_list): return [], []",
                "def train_model(X, y): return None",
                "def evaluate_model(model, X_test, y_test): return {}",
                "def main(): return None",
            ]
        )
        mock_client_factory.return_value = (object(), "gpt-5.2")

        generator = mock_generator_cls.return_value
        generator.invoke.return_value = {
            "generated_code": generated_code,
            "validation_feedback": "",
            "final_output": {
                "generated_code": generated_code,
                "code_spec": code_spec,
                "assumptions": [],
                "validation_result": {"is_valid": True, "missing_requirements": [], "checks": {}},
            },
        }

        executor = mock_executor_cls.return_value
        executor.invoke.side_effect = [
            {
                "execution_result": {
                    "status": "failed",
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "NameError: x is not defined",
                }
            },
            {
                "execution_result": {
                    "status": "success",
                    "returncode": 0,
                    "stdout": "{}",
                    "stderr": "",
                }
            },
        ]

        agent = CodeGenerationRunDebugAgent()
        result = agent.invoke({"raw_paper_info": {}, "max_iterations": 4})
        final_output = result["final_output"]

        self.assertEqual(final_output["execution_result"]["status"], "success")
        self.assertGreaterEqual(final_output["iteration"], 2)
        self.assertEqual(final_output["verification_status"], "passed")
        self.assertNotIn("build_exact_feature_matrix", final_output["generated_code"])


if __name__ == "__main__":
    unittest.main()
