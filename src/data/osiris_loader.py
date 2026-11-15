"""
OSIRIS Dataset Loader & Adapter — SiliconPilot AI
Official Hugging Face repo: hardware-fab/osiris
Handles acquiring, caching, and preprocessing analog layout variants with parasitic/simulation metrics.
Supports both official variants.parquet (89,108 layouts) and detailed sample simulation QoS metrics.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, OSIRIS_DIR, RAW_DIR, SAMPLE_DIR, SAMPLE_MODE
from src.utils.logger import get_logger

logger = get_logger(__name__)

OSIRIS_PARQUET_URL = "https://huggingface.co/datasets/hardware-fab/osiris/resolve/main/viewer/variants.parquet"


class OsirisLoader:
    """
    Acquires, validates, and prepares the OSIRIS analog IC layout dataset.
    Features:
      - 89,108 variants across 5 circuit families:
        (ota_ff, ota_lpf, miller_ota, ahuja_ota, five_transistors_ota)
      - Transistor sizing and placement perturbation indices
      - Parasitic extraction (PEX) scores across evaluation intervals
      - Physical layout area (µm²)
      - Electrical performance QoS metrics (in sample / simulated variants).
    """

    def __init__(self, raw_dir: Path | None = None, sample_mode: bool = False) -> None:
        self.raw_dir = Path(raw_dir or (RAW_DIR / "osiris"))
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_file = self.raw_dir / "variants.parquet"
        self.sample_mode = sample_mode
        self._df: pd.DataFrame | None = None

    def download_if_missing(self, force: bool = False) -> Path:
        """Download viewer/variants.parquet (~3.7 MB) from Hugging Face if not cached."""
        if self.parquet_file.exists() and not force:
            logger.info(f"OSIRIS variants.parquet already cached at {self.parquet_file} ({self.parquet_file.stat().st_size:,} bytes)")
            return self.parquet_file

        logger.info(f"Downloading OSIRIS variants.parquet from {OSIRIS_PARQUET_URL}...")
        req = urllib.request.Request(OSIRIS_PARQUET_URL, headers={"User-Agent": "SiliconPilot-AI/1.0"})
        temp_file = self.parquet_file.with_suffix(".tmp")
        try:
            with urllib.request.urlopen(req, timeout=40) as resp, open(temp_file, "wb") as out_f:
                total_bytes = 0
                while True:                                                                                                                                                                                                                                             
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    total_bytes += len(chunk)

            temp_file.replace(self.parquet_file)
            logger.info(f"Successfully downloaded {total_bytes:,} bytes to {self.parquet_file}")
            return self.parquet_file
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            logger.error(f"Failed downloading OSIRIS dataset: {e}")
            raise

    def load(self, source: str = "auto", nrows: int | None = None) -> pd.DataFrame:
        """
        Load OSIRIS dataset.
        source:
          - 'parquet': Official 89,108 variant layout archive
          - 'sample': Physics-correlated 500-sample dataset with full transistor sizing + QoS metrics
          - 'auto': Use sample if sample_mode is True, else parquet
        """
        sample_csv = Path(__file__).resolve().parents[2] / "data" / "sample" / "osiris_sample.csv"
        
        use_sample = (source == "sample") or (source == "auto" and self.sample_mode)
        
        if use_sample and sample_csv.exists():
            logger.info(f"Loading local OSIRIS sample data from {sample_csv}")
            df = pd.read_csv(sample_csv, nrows=nrows)
            self._df = df
            return df

        if not self.parquet_file.exists():
            try:
                self.download_if_missing()
            except Exception as e:
                logger.warning(f"Live OSIRIS download failed ({e}). Falling back to sample dataset...")
                if sample_csv.exists():
                    df = pd.read_csv(sample_csv, nrows=nrows)
                    self._df = df
                    return df
                raise

        logger.info(f"Loading official OSIRIS variants from {self.parquet_file}...")
        df = pd.read_parquet(self.parquet_file)
        if nrows:
            df = df.head(nrows)

        # Parse variant index and unpack pex_score
        df = self._parse_parquet_features(df)
        self._df = df
        logger.info(f"Loaded {len(df)} OSIRIS layout observations with {len(df.columns)} columns")
        return df

    def _parse_parquet_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract structured features from the official OSIRIS parquet."""
        df = df.copy()
        
        # Extract numeric variant index: 'variant_12' -> 12
        if "variant" in df.columns:
            df["variant_idx"] = df["variant"].astype(str).str.extract(r"(\d+)").astype(float).fillna(0)

        # Unpack pex_score ndarray into scalar columns
        if "pex_score" in df.columns:
            try:
                def extract_pex(val):
                    if isinstance(val, (list, np.ndarray)) and len(val) >= 3:
                        s0 = float(val[0][2]) if len(val[0]) >= 3 else 0.0
                        s1 = float(val[1][2]) if len(val[1]) >= 3 else 0.0
                        s2 = float(val[2][2]) if len(val[2]) >= 3 else 0.0
                        return s0, s1, s2
                    return 0.0, 0.0, 0.0

                scores = df["pex_score"].apply(extract_pex)
                df["pex_score_low"] = [s[0] for s in scores]
                df["pex_score_mid"] = [s[1] for s in scores]
                df["pex_score_high"] = [s[2] for s in scores]
            except Exception as e:
                logger.warning(f"Could not unpack pex_score: {e}")

        return df

    def get_features_and_targets(
        self,
        task: str = "performance",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Extract clean features (X) and targets (y), strictly avoiding post-verification data leakage.
        """
        if self._df is None:
            self.load()
        df = self._df.copy()

        # Check whether we are using the official parquet or the sample dataframe
        is_official_parquet = "circuit_family" in df.columns

        if is_official_parquet:
            # Official OSIRIS parquet tasks
            # Features: circuit_family one-hot + variant_idx
            family_dummies = pd.get_dummies(df["circuit_family"], prefix="fam", drop_first=True, dtype=float)
            X = pd.concat([df[["variant_idx"]], family_dummies], axis=1)
            groups = df["circuit_instance"] if "circuit_instance" in df.columns else df["circuit_family"]

            if task in ["area", "layout", "performance"]:
                # Target: layout area and parasitic extraction scores
                targets = ["area", "pex_score_low", "pex_score_mid", "pex_score_high"]
                avail_targets = [c for c in targets if c in df.columns]
                valid = df[avail_targets].notna().all(axis=1)
                X = X.loc[valid]
                y = df.loc[valid, avail_targets]
                groups = groups[valid]
            elif task == "quality":
                # Layouts with low parasitic degradation are clean candidates
                threshold = df["pex_score_high"].median()
                y = (df["pex_score_high"] < threshold).astype(int)
            else:
                y = df[["area"]]
            return X, y, groups

        else:
            # Physics-correlated sample format with full QoS metrics
            feature_cols = [
                "transistor_width_W",
                "transistor_length_L",
                "finger_count",
                "multiplicity",
                "finger_arrangement",
                "x_position",
                "y_position",
                "orientation",
                "delta_x",
                "delta_y",
                "perturbation_id",
            ]
            available_features = [c for c in feature_cols if c in df.columns]

            if "topology" in df.columns:
                top_dummies = pd.get_dummies(df["topology"], prefix="top", drop_first=True, dtype=float)
                X = pd.concat([df[available_features], top_dummies], axis=1)
            else:
                X = df[available_features].copy()

            groups = df["topology"] if "topology" in df.columns else df.index

            if task == "performance":
                targets = ["dc_gain", "gain_bandwidth_product", "phase_margin", "slew_rate", "cmrr"]
                target_cols = [c for c in targets if c in df.columns]
                if "drc_clean" in df.columns:
                    valid_mask = df["drc_clean"].astype(bool)
                    X = X.loc[valid_mask]
                    y = df.loc[valid_mask, target_cols]
                    groups = groups[valid_mask]
                else:
                    y = df[target_cols]
            elif task in ["area", "area_power"]:
                targets = ["area", "power_consumption"]
                target_cols = [c for c in targets if c in df.columns]
                y = df[target_cols]
            elif task == "quality":
                if "drc_clean" in df.columns:
                    y = df["drc_clean"].astype(int)
                else:
                    y = pd.Series(np.ones(len(df), dtype=int))
            else:
                raise ValueError(f"Unknown task: {task}")

            return X, y, groups
