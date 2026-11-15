"""Read-only model provenance and trust metadata resolution.

This module describes model trust. It deliberately does not enforce inference or
optimization restrictions; those are applied by a later integration stage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.settings import PROJECT_ROOT

DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "model_trust_policy.json"
VALID_TRUST_LABELS = {"unapproved", "restricted", "quarantined", "exploratory", "approved"}
VALID_DATA_ORIGINS = {"synthetic", "benchmark", "simulated", "measured", "unknown"}


class ModelTrustPolicy:
    """Load policy metadata and resolve a fail-closed status for model names."""

    def __init__(self, policy_path: str | Path | None = None) -> None:
        self.policy_path = Path(policy_path or DEFAULT_POLICY_PATH)
        self._models: dict[str, Any] = {}
        self._load_error: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            with self.policy_path.open("r", encoding="utf-8") as stream:
                document = json.load(stream)
            if (
                not isinstance(document, dict)
                or document.get("schema_version") != 1
                or not isinstance(document.get("models"), dict)
            ):
                raise ValueError("policy must use schema_version 1 and contain a 'models' object")
            self._models = document["models"]
        except FileNotFoundError:
            self._load_error = f"Policy file is missing: {self.policy_path}"
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            self._load_error = f"Policy file is invalid: {exc}"

    def get_status(self, model_name: str) -> dict[str, Any]:
        """Return validated metadata; missing or invalid entries are unapproved."""
        default = {
            "model_name": model_name,
            "trust_label": "unapproved",
            "data_origin": "unknown",
            "training_data": "Unknown",
            "evaluation_limitations": ["No valid model trust policy entry is available."],
            "permitted_uses": [],
            "approved": False,
        }
        if self._load_error:
            default["reason"] = self._load_error
            return default

        entry = self._models.get(model_name)
        if not isinstance(entry, dict):
            default["reason"] = "Model has no policy entry."
            return default

        required = ("trust_label", "data_origin", "training_data", "evaluation_limitations", "permitted_uses")
        if any(key not in entry for key in required):
            default["reason"] = "Policy entry is missing required fields."
            return default
        if not isinstance(entry["trust_label"], str) or entry["trust_label"] not in VALID_TRUST_LABELS:
            default["reason"] = "Policy entry has an invalid trust_label."
            return default
        if not isinstance(entry["data_origin"], str) or entry["data_origin"] not in VALID_DATA_ORIGINS:
            default["reason"] = "Policy entry has an invalid data_origin."
            return default
        if not isinstance(entry["training_data"], str) or not entry["training_data"].strip():
            default["reason"] = "Policy entry has invalid training_data."
            return default
        if not isinstance(entry["evaluation_limitations"], list) or not all(
            isinstance(item, str) and item.strip() for item in entry["evaluation_limitations"]
        ):
            default["reason"] = "Policy entry has invalid evaluation_limitations."
            return default
        if not isinstance(entry["permitted_uses"], list) or not all(
            isinstance(item, str) for item in entry["permitted_uses"]
        ):
            default["reason"] = "Policy entry has invalid permitted_uses."
            return default

        return {
            "model_name": model_name,
            **entry,
            "approved": entry["trust_label"] == "approved",
            "reason": entry.get("reason", "Status resolved from model trust policy."),
        }
