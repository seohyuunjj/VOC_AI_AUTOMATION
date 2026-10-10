import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import voc_pipeline as vp


def row(review_id, text, labels, score=3):
    return {
        "review_id": review_id,
        "app": "Test App",
        "review_text": text,
        "normalized_text": vp.normalize_text(text),
        "content_hash": vp.content_hash(text),
        "score": str(score),
        "thumbs_up_count": "0",
        "gold_labels": "|".join(labels),
    }


class PipelineTests(unittest.TestCase):
    def test_normalization_and_hash_are_stable(self):
        self.assertEqual(vp.normalize_text("  Login\n  fails  "), "Login fails")
        self.assertEqual(vp.content_hash("Login  fails"), vp.content_hash("Login fails"))

    def test_rule_problem(self):
        prediction = vp.rule_classify(row("1", "The app crashed and does not work", ["problem_report"], 1))
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.labels, ["problem_report"])

    def test_rule_feature(self):
        prediction = vp.rule_classify(row("2", "Please add an export option", ["feature_request"], 4))
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.labels, ["feature_request"])

    def test_ambiguous_review_routes_to_ai(self):
        prediction = vp.rule_classify(
            row(
                "3",
                "After the update the watch sometimes loses data and I would appreciate better controls",
                ["problem_report", "feature_request"],
                2,
            )
        )
        self.assertIsNone(prediction)

    def test_oracle_hybrid_and_metrics(self):
        rows = [
            row("1", "The app crashed and does not work", ["problem_report"], 1),
            row("2", "Please add an export option", ["feature_request"], 4),
            row("3", "Great app", ["irrelevant"], 5),
        ]
        predictions, ai_count = vp.classify_rows(rows, "hybrid", vp.OracleProvider({}))
        self.assertEqual(ai_count, 0)
        metrics = vp.multilabel_metrics(rows, predictions)
        self.assertEqual(metrics["exact_match"], 1.0)

    def test_database_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.db"
            connection = sqlite3.connect(path)
            vp.init_db(connection)
            tables = {
                item[0]
                for item in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            connection.close()
            self.assertTrue({"reviews", "analyses", "runs"}.issubset(tables))

    def test_gemini_model_resolution(self):
        config = {
            "model": "gpt-test",
            "gemini_model": "gemini-test",
        }
        self.assertEqual(vp.resolved_model("gemini", config, None), "gemini-test")
        self.assertEqual(vp.resolved_model("gemini", config, "override"), "override")

    def test_structured_output_schema_requires_all_fields(self):
        schema = vp.OpenAIProvider.schema()
        item = schema["properties"]["classifications"]["items"]
        self.assertEqual(set(item["properties"]), set(item["required"]))
        self.assertFalse(item["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
