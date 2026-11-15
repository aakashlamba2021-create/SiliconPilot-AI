"""
Data Preprocessing Pipeline — SiliconPilot AI
Provides robust preprocessing, missing value imputation, scaling, and leakage-free train/test splitting.
"""
from __future__ import annotations

from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from config.settings import RANDOM_SEED, TEST_SIZE
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Handles data transformations, missing value handling, and leakage-safe dataset splitting.
    """

    def __init__(self, scaler_type: str = "robust") -> None:
        self.scaler_type = scaler_type
        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.fitted = False
        self.feature_names_: list[str] = []
        self.medians_: Dict[str, float] = {}

    def clean_and_impute(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Replace infs with NaNs and impute with column median.
        If is_training=True, calculates and stores column medians.
        If is_training=False, reuses training medians to prevent test-distribution leakage.
        """
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        
        for col in num_cols:
            if is_training:
                median_val = float(df[col].median()) if not df[col].dropna().empty else 0.0
                self.medians_[col] = median_val if not np.isnan(median_val) else 0.0
            
            fill_val = self.medians_.get(col, 0.0)
            if df[col].isna().any():
                df[col] = df[col].fillna(fill_val)
                
        return df

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        groups: pd.Series | None = None,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_SEED,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
        """
        Splits data into train and test sets.
        If 'groups' is supplied, uses GroupShuffleSplit to prevent group-level data leakage
        (e.g., keeping entire circuit topologies or instances strictly in either train or test).
        """
        if groups is not None and groups.nunique() > 1:
            logger.info(f"Using GroupShuffleSplit on {groups.nunique()} unique groups to prevent data leakage.")
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = (
                y.iloc[train_idx] if isinstance(y, (pd.DataFrame, pd.Series)) else y[train_idx],
                y.iloc[test_idx] if isinstance(y, (pd.DataFrame, pd.Series)) else y[test_idx],
            )
        else:
            logger.info("Using standard train_test_split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        logger.info(f"Split completed: Train={len(X_train)} samples, Test={len(X_test)} samples")
        return X_train, X_test, y_train, y_test

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit scaler on training set and transform."""
        self.feature_names_ = list(X_train.columns)
        X_clean = self.clean_and_impute(X_train, is_training=True)
        scaled = self.scaler.fit_transform(X_clean)
        self.fitted = True
        return scaled

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform test or unseen data using fitted scaler and training medians."""
        if not self.fitted:
            raise RuntimeError("DataPreprocessor must be fitted before transform.")
        X_clean = self.clean_and_impute(X, is_training=False)
        return self.scaler.transform(X_clean)
