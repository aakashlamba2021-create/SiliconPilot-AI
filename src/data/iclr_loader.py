"""
ICLR-ADAPT Dataset Loader & Adapter — SiliconPilot AI
Official repository: https://github.com/xin-xi-xiao/ICLR-ADAPT
Handles acquiring, caching, parsing 35D vectors, and structuring ICLR-ADAPT observations.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, ICLR_ADAPT_DIR, RAW_DIR, SAMPLE_MODE
from src.utils.logger import get_logger

logger = get_logger(__name__)

ICLR_INDEX_URL = "https://raw.githubusercontent.com/xin-xi-xiao/ICLR-ADAPT/main/index.csv"


class ICLRAdaptLoader:
    """
    Acquires, validates, and prepares the ICLR-ADAPT dataset.
    Features:
      - 35D layout action vectors (normalized35, requested35, executed35)
      - Physical layout geometry metrics (Aeval, Leq_signal, Leq_power, Ncut_intermetal)
      - Quality & verification labels (G, C, L, DRC_total, OPEN, SHORT, quality_tier)
      - Circuit case identifiers for leakage-free group-based train/test splitting.
    """

    def __init__(self, raw_dir: Path | None = None, sample_mode: bool = False) -> None:
        self.raw_dir = Path(raw_dir or (RAW_DIR / "iclr_adapt"))
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.raw_dir / "index.csv"
        self.sample_mode = sample_mode
        self._df: pd.DataFrame | None = None

    def download_if_missing(self, force: bool = False) -> Path:
        """Download index.csv (~9 MB) if not already present."""
        if self.index_file.exists() and not force:
            logger.info(f"ICLR-ADAPT index.csv already exists at {self.index_file} ({self.index_file.stat().st_size:,} bytes)")
            return self.index_file

        logger.info(f"Downloading ICLR-ADAPT index.csv from {ICLR_INDEX_URL}...")
        req = urllib.request.Request(ICLR_INDEX_URL, headers={"User-Agent": "SiliconPilot-AI/1.0"})
        
        temp_file = self.index_file.with_suffix(".tmp")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(temp_file, "wb") as out_f:
                total_bytes = 0
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    total_bytes += len(chunk)
            
            temp_file.replace(self.index_file)
            logger.info(f"Successfully downloaded {total_bytes:,} bytes to {self.index_file}")
            return self.index_file
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            logger.error(f"Failed to download ICLR-ADAPT index.csv: {e}")
            raise

    def load(self, nrows: int | None = None) -> pd.DataFrame:
        """Load and parse the index.csv into a clean, feature-expanded DataFrame."""
        if not self.index_file.exists():
            try:
                self.download_if_missing()
            except Exception as e:
                logger.warning(f"Could not download live index.csv ({e}). Checking for local sample...")
                return self._generate_fallback_sample()

        logger.info(f"Loading ICLR-ADAPT from {self.index_file}...")
        df = pd.read_csv(self.index_file, nrows=nrows if not self.sample_mode else 500)
        
        # Convert string representations of JSON/arrays into parsed arrays
        df = self._parse_structured_columns(df)
        self._df = df
        logger.info(f"Loaded {len(df)} ICLR-ADAPT observations with {len(df.columns)} columns")
        return df

    def _parse_structured_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse vector columns (normalized35, requested35) into individual numerical columns."""
        df = df.copy()
        
        # Boolean conversions
        for col in ["G", "C", "L", "input_included"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])

        # Numeric conversions
        for col in ["Aeval", "Leq_signal", "Leq_power", "Ncut_intermetal", "DRC_total", "OPEN", "SHORT", "elapsed_s"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Parse normalized35 (35-dimensional continuous action array)
        if "normalized35" in df.columns:
            try:
                parsed_35 = df["normalized35"].apply(lambda v: json.loads(v) if isinstance(v, str) else v)
                arr_35 = np.array(parsed_35.tolist(), dtype=np.float32)
                for i in range(arr_35.shape[1]):
                    df[f"action_norm_{i:02d}"] = arr_35[:, i]
            except Exception as e:
                logger.warning(f"Could not unpack normalized35: {e}")

        return df

    def get_features_and_targets(
        self,
        target_type: str = "geometry",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Extract clean features (X), targets (y), and circuit grouping identifiers (groups)
        to prevent data leakage across circuit families.
        
        target_type:
          - 'geometry': Continuous physical layout metrics (Aeval, Leq_signal, Leq_power, Ncut_intermetal)
          - 'quality': Binary/categorical verification labels (L, DRC_total == 0)
        """
        if self._df is None:
            self.load()
        df = self._df.copy()

        # Input features: 35D action dimensions + one-hot PDK/circuit if needed
        action_cols = [c for c in df.columns if c.startswith("action_norm_")]
        if not action_cols:
            # Fallback if action_norm wasn't unpacked
            feature_cols = ["sobol_block_index"]
        else:
            feature_cols = action_cols

        # Group identifier (crucial for zero-leakage GroupKFold / GroupShuffleSplit)
        groups = df["circuit"] if "circuit" in df.columns else df.index

        if target_type == "geometry":
            targets = ["Aeval", "Leq_signal", "Leq_power", "Ncut_intermetal"]
            target_cols = [c for c in targets if c in df.columns]
            valid_mask = df[target_cols].notna().all(axis=1)
            X = df.loc[valid_mask, feature_cols]
            y = df.loc[valid_mask, target_cols]
            groups = groups[valid_mask]
        elif target_type == "quality":
            # Target is whether layout is signoff clean ('L' label in ICLR-ADAPT)
            y = df["L"].astype(int)
            X = df[feature_cols]
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        return X, y, groups

    def _generate_fallback_sample(self) -> pd.DataFrame:
        """Generates realistic synthetic ICLR-ADAPT format data if offline."""
        logger.info("Generating realistic local ICLR-ADAPT sample...")
        rng = np.random.default_rng(42)
        n = 300
        circuits = [f"OTA_{i}" for i in range(10)]
        pdks = ["14nm", "65nm", "SKY130"]
        
        circ_choice = rng.choice(circuits, size=n)
        pdk_choice = rng.choice(pdks, size=n)
        actions = rng.uniform(0.0, 1.0, size=(n, 35))
        
        # Realistic correlations: area grows with sum of first 5 action coordinates
        aeval = 100.0 + actions[:, :5].sum(axis=1) * 60.0 + rng.normal(0, 5, n)
        leq_sig = aeval * 2.1 + rng.normal(0, 10, n)
        leq_pwr = aeval * 5.4 + rng.normal(0, 20, n)
        ncuts = (aeval * 4.5 + rng.normal(0, 30, n)).astype(int)
        drc_clean = rng.uniform(0, 1, n) > 0.3
        
        data = {
            "case_id": [f"case_{i:04d}" for i in range(n)],
            "circuit": circ_choice,
            "pdk": pdk_choice,
            "G": True,
            "C": drc_clean,
            "L": drc_clean & (rng.uniform(0, 1, n) > 0.2),
            "DRC_total": np.where(drc_clean, 0, rng.integers(1, 15, n)),
            "Aeval": aeval.round(3),
            "Leq_signal": leq_sig.round(3),
            "Leq_power": leq_pwr.round(3),
            "Ncut_intermetal": ncuts,
            "normalized35": [json.dumps(row.tolist()) for row in actions],
        }
        df = pd.DataFrame(data)
        for i in range(35):
            df[f"action_norm_{i:02d}"] = actions[:, i]
        self._df = df
        return df

