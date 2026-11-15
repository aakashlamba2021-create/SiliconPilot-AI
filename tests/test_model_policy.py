"""Tests for model trust/provenance metadata and backward-compatible registry APIs."""
import json
import tempfile
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_policy import ModelTrustPolicy


def _valid_entry(**overrides):
    entry = {
        "trust_label": "restricted",
        "data_origin": "synthetic",
        "training_data": "fixture dataset",
        "evaluation_limitations": ["synthetic labels"],
        "permitted_uses": ["metadata_review"],
    }
    entry.update(overrides)
    return entry


class ModelPolicyTests(unittest.TestCase):
  def test_valid_policy_loading_and_status_resolution(self):
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temp_dir:
      path = Path(temp_dir) / "policy.json"
      path.write_text(json.dumps({"schema_version": 1, "models": {"fixture": _valid_entry()}}), encoding="utf-8")

      status = ModelTrustPolicy(path).get_status("fixture")

      self.assertEqual(status["trust_label"], "restricted")
      self.assertEqual(status["data_origin"], "synthetic")
      self.assertFalse(status["approved"])
      self.assertEqual(status["evaluation_limitations"], ["synthetic labels"])

  def test_missing_policy_and_missing_model_fail_closed(self):
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temp_dir:
      missing = ModelTrustPolicy(Path(temp_dir) / "absent.json").get_status("fixture")
      self.assertEqual(missing["trust_label"], "unapproved")
      self.assertFalse(missing["approved"])

      path = Path(temp_dir) / "policy.json"
      path.write_text('{"schema_version":1,"models":{}}', encoding="utf-8")
      absent_entry = ModelTrustPolicy(path).get_status("fixture")
      self.assertEqual(absent_entry["trust_label"], "unapproved")
      self.assertFalse(absent_entry["approved"])

  def test_invalid_policy_document_and_entries_fail_closed(self):
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temp_dir:
      path = Path(temp_dir) / "policy.json"
      path.write_text("{invalid", encoding="utf-8")
      self.assertEqual(ModelTrustPolicy(path).get_status("fixture")["trust_label"], "unapproved")

      invalid_entries = [
        _valid_entry(trust_label="trusted-ish"),
        _valid_entry(trust_label=[]),
        _valid_entry(data_origin="lab_magic"),
        _valid_entry(data_origin=[]),
        _valid_entry(evaluation_limitations="not a list"),
        {"trust_label": "approved"},
      ]
      for index, entry in enumerate(invalid_entries):
        path.write_text(json.dumps({"schema_version": 1, "models": {"fixture": entry}}), encoding="utf-8")
        status = ModelTrustPolicy(path).get_status("fixture")
        self.assertEqual(status["trust_label"], "unapproved", index)
        self.assertFalse(status["approved"], index)

  def test_registry_status_is_separate_and_load_api_remains_compatible(self):
    from src.models.model_registry import ModelRegistry

    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temp_dir:
      root = Path(temp_dir)
      registry_dir = root / "registry"
      registry_dir.mkdir()
      policy = root / "policy.json"
      policy.write_text(json.dumps({"schema_version": 1, "models": {
          "fixture": _valid_entry(trust_label="approved", permitted_uses=["test"]),
          "quarantined": _valid_entry(trust_label="quarantined", permitted_uses=[]),
          "exploratory": _valid_entry(trust_label="exploratory", permitted_uses=["explore"]),
      }}), encoding="utf-8")
      registry = ModelRegistry(registry_dir=registry_dir)
      registry._trust_policy = ModelTrustPolicy(policy)
      registry.save_model("fixture", model_obj={"kind": "fixture"}, preprocessor=None)
      registry.save_model("quarantined", model_obj={"kind": "blocked"}, preprocessor=None)
      registry.save_model("exploratory", model_obj={"kind": "explore"}, preprocessor=None)
      bundle = registry.load_model("fixture")

      self.assertIsInstance(bundle, dict)
      self.assertEqual(set(bundle), {
        "model", "preprocessor", "feature_names", "target_names", "metrics",
        "dataset_name", "description", "saved_at",
      })
      self.assertEqual(bundle["model"], {"kind": "fixture"})
      self.assertEqual(registry.get_model_status("fixture")["trust_label"], "approved")
      self.assertIsNone(registry.load_model("quarantined"))
      self.assertIsNone(registry.load_model("exploratory"))
      self.assertIsNone(registry.load_model_for_permitted_use("exploratory", "unknown"))
      exploratory_bundle = registry.load_model_for_permitted_use("exploratory", "explore")
      self.assertEqual(exploratory_bundle["model"], {"kind": "explore"})


if __name__ == "__main__":
    unittest.main()
