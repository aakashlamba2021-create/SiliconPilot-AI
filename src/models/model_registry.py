"""
Model Registry — SiliconPilot AI
Handles saving, loading, versioning, and metadata tracking of trained models and preprocessors.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from config.settings import MODEL_DIR
from src.models.model_policy import ModelTrustPolicy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Manages persistence and versioning of trained model artifacts and evaluation metadata.
    """

    def __init__(self, registry_dir: Path | None = None) -> None:
        self.registry_dir = Path(registry_dir or MODEL_DIR)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_dir / "models_manifest.json"
        self._manifest = self._load_manifest()
        self._trust_policy = ModelTrustPolicy()

    def _load_manifest(self) -> Dict[str, Any]:
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load manifest ({e}); starting fresh.")
        return {}

    def _save_manifest(self) -> None:
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2)

    def save_model(
        self,
        model_name: str,
        model_obj: Any,
        preprocessor: Any = None,
        feature_names: list[str] | None = None,
        target_names: list[str] | None = None,
        metrics: Dict[str, Any] | None = None,
        dataset_name: str = "Unknown",
        description: str = "",
    ) -> Path:
        """
        Serializes the model, preprocessor, and saves metadata into the registry.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_tag = f"{model_name}_{timestamp}"
        artifact_path = self.registry_dir / f"{version_tag}.joblib"

        bundle = {
            "model": model_obj,
            "preprocessor": preprocessor,
            "feature_names": feature_names or [],
            "target_names": target_names or [],
            "metrics": metrics or {},
            "dataset_name": dataset_name,
            "description": description,
            "saved_at": timestamp,
        }

        joblib.dump(bundle, artifact_path)
        logger.info(f"Model bundle saved to {artifact_path}")

        # Update manifest
        self._manifest[model_name] = {
            "latest_artifact": str(artifact_path.name),
            "version_tag": version_tag,
            "saved_at": timestamp,
            "dataset_name": dataset_name,
            "metrics": metrics or {},
            "feature_names": feature_names or [],
            "target_names": target_names or [],
            "description": description,
        }
        self._save_manifest()
        return artifact_path

    def load_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Loads only approved model bundles. Signature and bundle format remain stable.
        """
        status = self.get_model_status(model_name)
        if status.get("trust_label") != "approved":
            logger.warning("Model '%s' is not approved for default loading (trust=%s).", model_name, status.get("trust_label"))
            return None
        return self._load_registered_bundle(model_name)

    def load_model_for_permitted_use(self, model_name: str, use_case: str) -> Optional[Dict[str, Any]]:
        """Load an exploratory model only for a use explicitly listed by policy."""
        status = self.get_model_status(model_name)
        if status.get("trust_label") not in {"exploratory", "approved"}:
            logger.warning("Model '%s' is blocked for use '%s' (trust=%s).", model_name, use_case, status.get("trust_label"))
            return None
        if use_case not in status.get("permitted_uses", []):
            logger.warning("Model '%s' is not permitted for use '%s'.", model_name, use_case)
            return None
        return self._load_registered_bundle(model_name)

    def _load_registered_bundle(self, model_name: str) -> Optional[Dict[str, Any]]:
        if model_name not in self._manifest:
            logger.warning(f"Model '{model_name}' not found in registry.")
            return None

        entry = self._manifest[model_name]
        artifact_name = entry["latest_artifact"]
        artifact_path = self.registry_dir / artifact_name

        if not artifact_path.exists():
            logger.error(f"Artifact file {artifact_path} does not exist.")
            return None

        logger.info(f"Loading model bundle '{model_name}' from {artifact_path}")
        return joblib.load(artifact_path)

    def list_models(self) -> Dict[str, Any]:
        """Returns all registered models and their summary metrics."""
        return self._manifest

    def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Return provenance and trust metadata separately from model bundles.

        This read-only API preserves load_model()'s existing return type, keys,
        and behavior. Status is informational in this stage and does not block
        loading, inference, or optimization.
        """
        return self._trust_policy.get_status(model_name)

