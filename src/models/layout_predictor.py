"""
Physical Layout Predictor — SiliconPilot AI
Predicts layout area, interconnect routing lengths (signal/power), and cut counts from pre-layout actions/geometry.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from config.settings import RANDOM_SEED
from src.data.preprocessing import DataPreprocessor
from src.evaluation.model_evaluator import ModelEvaluator
from src.models.model_registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LayoutPredictor:
    """
    Multi-target physical layout geometry predictor.
    Targets typically: Aeval (µm²), Leq_signal (µm), Leq_power (µm), Ncut_intermetal
    """

    def __init__(self, algorithm: str = "rf", n_estimators: int = 100) -> None:
        self.algorithm = algorithm
        self.n_estimators = n_estimators
        self.model = self._build_model()
        self.preprocessor = DataPreprocessor(scaler_type="robust")
        self.target_names: list[str] = []
        self.feature_names: list[str] = []
        self.metrics_: Dict[str, Any] = {}

    def _build_model(self) -> Any:
        if self.algorithm == "rf":
            base = RandomForestRegressor(n_estimators=self.n_estimators, random_state=RANDOM_SEED, n_jobs=-1)
        elif self.algorithm == "et":
            base = ExtraTreesRegressor(n_estimators=self.n_estimators, random_state=RANDOM_SEED, n_jobs=-1)
        elif self.algorithm == "gb":
            base = GradientBoostingRegressor(n_estimators=self.n_estimators, random_state=RANDOM_SEED)
            return MultiOutputRegressor(base)
        else:
            base = RandomForestRegressor(n_estimators=self.n_estimators, random_state=RANDOM_SEED, n_jobs=-1)
        return base

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Fits the predictor and calculates evaluation metrics on the reserved test set."""
        self.feature_names = list(X_train.columns)
        self.target_names = list(y_train.columns) if isinstance(y_train, pd.DataFrame) else ["target"]

        logger.info(f"Training LayoutPredictor ({self.algorithm}) on {len(X_train)} samples with {len(self.feature_names)} features...")
        X_tr_scaled = self.preprocessor.fit_transform(X_train)
        X_te_scaled = self.preprocessor.transform(X_test)

        self.model.fit(X_tr_scaled, y_train)

        # Predict on test set
        y_pred = self.model.predict(X_te_scaled)
        metrics = ModelEvaluator.evaluate_regression(y_test, y_pred, target_names=self.target_names)
        self.metrics_ = metrics

        logger.info(f"Evaluation complete. Mean Test R²: {metrics['overall'].get('mean_R2', metrics['overall'].get('R2')):.4f}")
        return metrics

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generates predictions for new unseen layout actions or parameters."""
        X_scaled = self.preprocessor.transform(X)
        preds = self.model.predict(X_scaled)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return pd.DataFrame(preds, columns=self.target_names, index=X.index)

    def save(self, registry: ModelRegistry, model_name: str = "layout_geometry_predictor") -> str:
        """Saves artifact into the model registry."""
        path = registry.save_model(
            model_name=model_name,
            model_obj=self.model,
            preprocessor=self.preprocessor,
            feature_names=self.feature_names,
            target_names=self.target_names,
            metrics=self.metrics_,
            dataset_name="ICLR-ADAPT / OSIRIS Layout",
            description=f"Physical layout geometry predictor using {self.algorithm.upper()}",
        )
        return str(path)

