"""
Analog IC Feature Engineering — SiliconPilot AI
Extracts domain-specific physical layout and electrical proxy features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnalogFeatureEngineer:
    """
    Computes domain-specific engineered features for analog IC layout analysis.
    """

    @staticmethod
    def engineer_osiris_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives layout physics features from transistor sizing and component placements.
        """
        feats = df.copy()

        # 1. Transistor aspect ratio (W / L)
        if "transistor_width_W" in feats.columns and "transistor_length_L" in feats.columns:
            feats["aspect_ratio_W_L"] = feats["transistor_width_W"] / (feats["transistor_length_L"] + 1e-6)
            feats["log_aspect_ratio"] = np.log1p(feats["aspect_ratio_W_L"])

        # 2. Total active transistor gate area: W * L * fingers * multiplicity
        if all(c in feats.columns for c in ["transistor_width_W", "transistor_length_L", "finger_count", "multiplicity"]):
            feats["total_active_gate_area"] = (
                feats["transistor_width_W"]
                * feats["transistor_length_L"]
                * feats["finger_count"]
                * feats["multiplicity"]
            )
            feats["log_active_area"] = np.log1p(feats["total_active_gate_area"])

        # 3. Placement perturbation magnitude: sqrt(delta_x^2 + delta_y^2)
        if "delta_x" in feats.columns and "delta_y" in feats.columns:
            feats["perturbation_displacement"] = np.sqrt(feats["delta_x"] ** 2 + feats["delta_y"] ** 2)
            feats["is_high_perturbation"] = (feats["perturbation_displacement"] > 3.0).astype(float)

        # 4. Placement position distance from origin: sqrt(x^2 + y^2)
        if "x_position" in feats.columns and "y_position" in feats.columns:
            feats["radial_distance"] = np.sqrt(feats["x_position"] ** 2 + feats["y_position"] ** 2)

        # 5. Finger density proxy
        if "finger_count" in feats.columns and "transistor_width_W" in feats.columns:
            feats["width_per_finger"] = feats["transistor_width_W"] / (feats["finger_count"] + 1e-6)

        return feats

    @staticmethod
    def engineer_iclr_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives physical geometry interaction and density features from ICLR-ADAPT metrics.
        """
        feats = df.copy()

        # Signal-to-Power metal ratio
        if "Leq_signal" in feats.columns and "Leq_power" in feats.columns:
            feats["signal_to_power_metal_ratio"] = feats["Leq_signal"] / (feats["Leq_power"] + 1e-6)

        # Metal routing density proxy: total metal / area
        if all(c in feats.columns for c in ["Leq_signal", "Leq_power", "Aeval"]):
            feats["total_metal_length"] = feats["Leq_signal"] + feats["Leq_power"]
            feats["metal_density_proxy"] = feats["total_metal_length"] / (feats["Aeval"] + 1e-6)

        # Cut density: Ncut / area
        if "Ncut_intermetal" in feats.columns and "Aeval" in feats.columns:
            feats["cut_density_per_area"] = feats["Ncut_intermetal"] / (feats["Aeval"] + 1e-6)

        return feats

