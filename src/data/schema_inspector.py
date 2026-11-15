"""
Schema Inspector — SiliconPilot AI
Automatically discovers schema, checks data quality, and reports on any DataFrame.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Column classification helpers
# ---------------------------------------------------------------------------

def _infer_column_role(col: str, dtype: Any) -> str:
    """Heuristically classify a column as 'id', 'categorical', 'numeric', 'boolean', 'target', 'unknown'."""
    col_lower = col.lower()
    if any(k in col_lower for k in ("_id", "id_", "index", "name", "template", "netlist")):
        return "id"
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if isinstance(dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        return "categorical"
    if pd.api.types.is_integer_dtype(dtype):
        # Small int — may be categorical
        return "ordinal_int"
    if pd.api.types.is_float_dtype(dtype):
        return "numeric"
    return "unknown"


# ---------------------------------------------------------------------------
# Main inspector
# ---------------------------------------------------------------------------

class SchemaInspector:
    """
    Analyses a pandas DataFrame and produces a schema + quality report.

    Usage
    -----
    inspector = SchemaInspector(df, name="OSIRIS")
    report = inspector.run()
    inspector.save_report("reports/osiris_schema.json")
    inspector.print_summary()
    """

    def __init__(self, df: pd.DataFrame, name: str = "Dataset") -> None:
        self.df = df
        self.name = name
        self._report: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute all checks and return the full report dict."""
        logger.info(f"Running schema inspection on '{self.name}' ({len(self.df)} rows × {len(self.df.columns)} cols)")

        self._report = {
            "dataset_name": self.name,
            "shape": {"rows": len(self.df), "columns": len(self.df.columns)},
            "columns": self._column_report(),
            "missing_values": self._missing_report(),
            "duplicates": self._duplicate_report(),
            "statistics": self._statistics_report(),
            "feature_target_candidates": self._feature_target_candidates(),
            "data_quality_score": None,  # filled after all checks
        }
        self._report["data_quality_score"] = self._compute_quality_score()
        logger.info(f"Schema inspection complete. Quality score: {self._report['data_quality_score']:.1f}/100")
        return self._report

    def save_report(self, path: str | Path) -> None:
        """Save JSON report to disk."""
        if not self._report:
            self.run()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._report, f, indent=2, default=str)
        logger.info(f"Schema report saved to {path}")

    def print_summary(self) -> None:
        """Print a human-readable summary to the console."""
        if not self._report:
            self.run()
        r = self._report
        print(f"\n{'='*60}")
        print(f"  Schema Report: {r['dataset_name']}")
        print(f"{'='*60}")
        print(f"  Rows:    {r['shape']['rows']:,}")
        print(f"  Columns: {r['shape']['columns']}")
        print(f"  Quality: {r['data_quality_score']:.1f}/100")
        print(f"\n  {'Column':<30} {'Type':<12} {'Role':<14} {'Missing%':<10} {'Unique'}")
        print(f"  {'-'*78}")
        for col, info in r["columns"].items():
            print(
                f"  {col:<30} {info['dtype']:<12} {info['role']:<14} "
                f"{info['missing_pct']:<10.1f} {info['n_unique']}"
            )
        mv = r["missing_values"]
        print(f"\n  Columns with missing values: {mv['n_cols_with_missing']}")
        print(f"  Total missing cells: {mv['total_missing']:,} ({mv['total_missing_pct']:.2f}%)")
        dup = r["duplicates"]
        print(f"  Duplicate rows: {dup['n_duplicates']:,} ({dup['duplicate_pct']:.2f}%)")
        cands = r["feature_target_candidates"]
        print(f"\n  Suggested features:  {cands['suggested_features']}")
        print(f"  Suggested targets:   {cands['suggested_targets']}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _column_report(self) -> dict[str, dict]:
        result = {}
        for col in self.df.columns:
            series = self.df[col]
            dtype = series.dtype
            n_missing = int(series.isna().sum())
            n_unique = int(series.nunique(dropna=True))
            role = _infer_column_role(col, dtype)
            result[col] = {
                "dtype": str(dtype),
                "role": role,
                "missing": n_missing,
                "missing_pct": round(n_missing / len(self.df) * 100, 2) if len(self.df) else 0.0,
                "n_unique": n_unique,
                "sample_values": series.dropna().head(3).tolist(),
            }
        return result

    def _missing_report(self) -> dict:
        total_cells = self.df.size
        total_missing = int(self.df.isna().sum().sum())
        cols_with_missing = {
            col: int(self.df[col].isna().sum())
            for col in self.df.columns
            if self.df[col].isna().any()
        }
        return {
            "total_missing": total_missing,
            "total_missing_pct": round(total_missing / total_cells * 100, 4) if total_cells else 0.0,
            "n_cols_with_missing": len(cols_with_missing),
            "cols_with_missing": cols_with_missing,
        }

    def _duplicate_report(self) -> dict:
        n_dup = int(self.df.duplicated().sum())
        return {
            "n_duplicates": n_dup,
            "duplicate_pct": round(n_dup / len(self.df) * 100, 2) if len(self.df) else 0.0,
        }

    def _statistics_report(self) -> dict:
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {}
        desc = numeric_df.describe().round(4)
        return desc.to_dict()

    def _feature_target_candidates(self) -> dict:
        """
        Heuristic: columns with 'gain', 'power', 'area', 'margin', 'cmrr',
        'slew', 'bandwidth', 'settling' are likely targets.
        Boolean columns 'drc', 'lvs' are likely classification targets.
        """
        target_keywords = {
            "dc_gain", "gain_bandwidth", "gbw", "phase_margin", "power",
            "area", "slew_rate", "cmrr", "settling", "drc_clean", "lvs_verified",
            "parasitic", "performance",
        }
        suggested_targets = []
        suggested_features = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in target_keywords):
                suggested_targets.append(col)
            else:
                role = _infer_column_role(col, self.df[col].dtype)
                if role not in ("id",):
                    suggested_features.append(col)
        return {
            "suggested_features": suggested_features,
            "suggested_targets": suggested_targets,
        }

    def _compute_quality_score(self) -> float:
        """Score 0–100. Deductions for: missing values, duplicates, all-constant columns."""
        score = 100.0
        mv = self._report["missing_values"]
        score -= min(30.0, mv["total_missing_pct"] * 3)  # up to -30 for missing

        dup = self._report["duplicates"]
        score -= min(20.0, dup["duplicate_pct"] * 2)  # up to -20 for duplicates

        # Constant columns (zero variance)
        numeric_df = self.df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            n_constant = int((numeric_df.std() == 0).sum())
            score -= min(10.0, n_constant * 2)  # up to -10 for constant cols

        return max(0.0, round(score, 1))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def inspect_dataframe(
    df: pd.DataFrame,
    name: str = "Dataset",
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """One-shot inspection. Returns report dict and optionally saves JSON."""
    inspector = SchemaInspector(df, name=name)
    report = inspector.run()
    inspector.print_summary()
    if save_path:
        inspector.save_report(save_path)
    return report

