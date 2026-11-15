"""
Model Evaluation Suite — SiliconPilot AI
Calculates engineering metrics: MAE, RMSE, R², NMAE, Accuracy, Precision, Recall, F1-score.
Correctly handles distinct physical units for each analog IC target.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_UNITS: Dict[str, str] = {
    "dc_gain": "dB",
    "gain_bandwidth_product": "Hz",
    "phase_margin": "°",
    "slew_rate": "V/µs",
    "cmrr": "dB",
    "power_consumption": "W",
    "area": "µm²",
    "Aeval": "µm²",
    "Leq_signal": "µm",
    "Leq_power": "µm",
    "Ncut_intermetal": "cuts",
    "pex_score_low": "score",
    "pex_score_mid": "score",
    "pex_score_high": "score",
}


class ModelEvaluator:
    """
    Evaluates regression and classification models with standard metrics,
    tracking physical units independently and computing dimensionless Normalized MAE (NMAE).
    """

    @staticmethod
    def evaluate_regression(
        y_true: np.ndarray | pd.DataFrame | pd.Series,
        y_pred: np.ndarray | pd.DataFrame | pd.Series,
        target_names: list[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Calculates MAE, RMSE, R2, and NMAE for single or multi-target regression.
        Units are tracked per target to avoid physically invalid unit averaging.
        """
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)

        metrics: Dict[str, Any] = {}

        if y_t.ndim == 1 or (y_t.ndim == 2 and y_t.shape[1] == 1):
            y_t = y_t.ravel()
            y_p = y_p.ravel()
            mae = float(mean_absolute_error(y_t, y_p))
            rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
            r2 = float(r2_score(y_t, y_p))
            range_val = float(np.ptp(y_t)) if len(y_t) > 1 and np.ptp(y_t) > 0 else 1.0
            nmae = float(mae / range_val)
            name = target_names[0] if target_names else "target"
            unit = TARGET_UNITS.get(name, "")

            metrics["per_target"] = {
                name: {
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "NMAE": nmae,
                    "unit": unit,
                }
            }
            metrics["overall"] = {
                "R2": r2,
                "mean_R2": r2,
                "NMAE": nmae,
                "mean_NMAE": nmae,
                "MAE": mae,
            }
        else:
            n_targets = y_t.shape[1]
            names = target_names or [f"target_{i}" for i in range(n_targets)]
            per_target = {}
            maes, rmses, r2s, nmaes = [], [], [], []

            for i, name in enumerate(names):
                yt_i = y_t[:, i]
                yp_i = y_p[:, i]
                mae = float(mean_absolute_error(yt_i, yp_i))
                rmse = float(np.sqrt(mean_squared_error(yt_i, yp_i)))
                r2 = float(r2_score(yt_i, yp_i))
                range_i = float(np.ptp(yt_i)) if len(yt_i) > 1 and np.ptp(yt_i) > 0 else 1.0
                nmae = float(mae / range_i)
                unit = TARGET_UNITS.get(name, "")

                per_target[name] = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "NMAE": nmae,
                    "unit": unit,
                }
                maes.append(mae)
                rmses.append(rmse)
                r2s.append(r2)
                nmaes.append(nmae)

            metrics["per_target"] = per_target
            metrics["overall"] = {
                "mean_R2": float(np.mean(r2s)),
                "mean_NMAE": float(np.mean(nmaes)),
                # Legacy compatibility field (note: cross-unit arithmetic mean)
                "mean_MAE": float(np.mean(maes)),
                "mean_RMSE": float(np.mean(rmses)),
            }

        return metrics

    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
    ) -> Dict[str, float]:
        """
        Calculates Accuracy, Precision, Recall, and F1 score for binary classification.
        """
        y_t = np.asarray(y_true).ravel()
        y_p = np.asarray(y_pred).ravel()

        return {
            "Accuracy": float(accuracy_score(y_t, y_p)),
            "Precision": float(precision_score(y_t, y_p, zero_division=0)),
            "Recall": float(recall_score(y_t, y_p, zero_division=0)),
            "F1": float(f1_score(y_t, y_p, zero_division=0)),
        }

    @staticmethod
    def format_metrics_table(metrics: Dict[str, Any]) -> pd.DataFrame:
        """Converts regression or classification metric dictionary to a display DataFrame."""
        if "per_target" in metrics:
            return pd.DataFrame.from_dict(metrics["per_target"], orient="index")
        elif "overall" in metrics:
            return pd.DataFrame([metrics["overall"]])
        else:
            return pd.DataFrame([metrics])
