"""
Layout Quality & Verification Classifier — SiliconPilot AI
Classifies whether a candidate layout design will pass DRC/LVS physical verification.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier

from config.settings import RANDOM_SEED
from src.data.preprocessing import DataPreprocessor
from src.evaluation.model_evaluator import ModelEvaluator
from src.models.model_registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QualityClassifier:
    """
    Binary / multi-class classifier predicting physical verification outcomes (DRC Clean / LVS pass).
    """

    def __init__(self, algorithm: str = "rf", n_estimators: int = 100) -> None:
        self.algorithm = algorithm
        self.n_estimators = n_estimators
        self.model = self._build_model()
        self.preprocessor = DataPreprocessor(scaler_type="robust")
        self.feature_names: list[str] = []
        self.metrics_: Dict[str, float] = {}

    def _build_model(self) -> Any:
        if self.algorithm == "rf":
            return RandomForestClassifier(n_estimators=self.n_estimators, random_state=RANDOM_SEED, n_jobs=-1)
        elif self.algorithm == "et":
            return ExtraTreesClassifier(n_estimators=self.n_estimators, random_state=RANDOM_SEED, n_jobs=-1)
        elif self.algorithm == "gb":
            return GradientBoostingClassifier(n_estimators=self.n_estimators, random_state=RANDOM_SEED)
        return RandomForestClassifier(n_estimators=self.n_estimators, random_state=RANDOM_SEED, n_jobs=-1)

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        y_test: pd.Series | np.ndarray,
    ) -> Dict[str, float]:
        """Trains classification model and evaluates accuracy, precision, recall, and F1."""
        self.feature_names = list(X_train.columns)

        logger.info(f"Training QualityClassifier ({self.algorithm}) on {len(X_train)} samples...")
        X_tr_scaled = self.preprocessor.fit_transform(X_train)
        X_te_scaled = self.preprocessor.transform(X_test)

        self.model.fit(X_tr_scaled, y_train)

        y_pred = self.model.predict(X_te_scaled)
        metrics = ModelEvaluator.evaluate_classification(y_test, y_pred)
        self.metrics_ = metrics

        logger.info(f"Classification evaluation complete. Test Accuracy: {metrics['Accuracy']*100:.2f}%, F1: {metrics['F1']:.4f}")
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts binary verification outcome."""
        X_scaled = self.preprocessor.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns verification pass probabilities."""
        X_scaled = self.preprocessor.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, registry: ModelRegistry, model_name: str = "layout_quality_classifier") -> str:
        """Saves classification bundle into registry."""
        path = registry.save_model(
            model_name=model_name,
            model_obj=self.model,
            preprocessor=self.preprocessor,
            feature_names=self.feature_names,
            target_names=["drc_clean_probability"],
            metrics=self.metrics_,
            dataset_name="ICLR-ADAPT / OSIRIS Verification",
            description=f"Layout DRC/LVS verification classifier using {self.algorithm.upper()}",
        )
        return str(path)

