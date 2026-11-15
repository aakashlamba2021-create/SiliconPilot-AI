"""
Electrical Performance Predictor — SiliconPilot AI
Predicts post-layout electrical QoS (Gain, GBW, Phase Margin, Slew Rate, CMRR, Power)
using safe pre-layout and placement features (zero data leakage).
"""
from __future__ import annotations

from typing import Any, Dict

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


class PerformancePredictor:
    """
    Predicts analog electrical QoS metrics from transistor sizing and layout placement.
    """

    def __init__(self, algorithm: str = "et", n_estimators: int = 120) -> None:
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
            base = ExtraTreesRegressor(n_estimators=self.n_estimators, random_state=RANDOM_SEED, n_jobs=-1)
        return base

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Trains performance regression model and evaluates on test split."""
        self.feature_names = list(X_train.columns)
        self.target_names = list(y_train.columns) if isinstance(y_train, pd.DataFrame) else ["QoS_target"]

        logger.info(f"Training PerformancePredictor ({self.algorithm}) on {len(X_train)} samples with targets: {self.target_names}...")
        X_tr_scaled = self.preprocessor.fit_transform(X_train)
        X_te_scaled = self.preprocessor.transform(X_test)

        self.model.fit(X_tr_scaled, y_train)

        # Test evaluation
        y_pred = self.model.predict(X_te_scaled)
        metrics = ModelEvaluator.evaluate_regression(y_test, y_pred, target_names=self.target_names)
        self.metrics_ = metrics

        logger.info(f"Performance evaluation complete. Test Mean R²: {metrics['overall'].get('mean_R2', metrics['overall'].get('R2')):.4f}")
        return metrics

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predicts electrical performance for new candidate designs."""
        X_scaled = self.preprocessor.transform(X)
        preds = self.model.predict(X_scaled)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return pd.DataFrame(preds, columns=self.target_names, index=X.index)

    def save(self, registry: ModelRegistry, model_name: str = "electrical_performance_predictor") -> str:
        """Saves model bundle to registry."""
        path = registry.save_model(
            model_name=model_name,
            model_obj=self.model,
            preprocessor=self.preprocessor,
            feature_names=self.feature_names,
            target_names=self.target_names,
            metrics=self.metrics_,
            dataset_name="OSIRIS Analog Layouts",
            description=f"Parasitic-aware electrical QoS predictor using {self.algorithm.upper()}",
        )
        return str(path)

