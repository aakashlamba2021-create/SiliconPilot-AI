"""
SiliconPilot AI - Simulation Validator Module
Compares ML model predictions against actual ngspice SPICE simulation results,
computing absolute error, relative error (%), and agreement metrics across electrical figures of merit.

Phase 4.1 Repairs:
- Handles unavailable (None) SPICE metrics gracefully without fabricating errors.
- Never computes relative errors against missing or unmeasured quantities.
- Explicitly flags non-converged or incomplete simulations.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from src.simulation.simulation_metrics import ExtractedMetrics
from src.utils.logger import get_logger

logger = get_logger("src.simulation.simulation_validator")


@dataclass
class MetricComparison:
    """Rigorous comparison record for an individual electrical performance figure."""
    metric_name: str
    unit: str
    ml_predicted: Optional[float]
    spice_simulated: Optional[float]
    absolute_error: Optional[float]
    relative_error_pct: Optional[float]
    agreement_grade: str
    ml_data_source: str
    spice_data_source: str


@dataclass
class ValidationReport:
    """Full electrical validation report for a design candidate."""
    candidate_id: str
    topology: str
    comparisons: List[MetricComparison]
    mean_relative_error_pct: Optional[float]
    simulation_status: str
    validation_verdict: str
    engineering_summary: str

    def to_dataframe(self) -> pd.DataFrame:
        """Converts comparison records to a formatted pandas DataFrame."""
        rows = []
        for c in self.comparisons:
            ml_str = f"{c.ml_predicted:.2f} {c.unit}" if c.ml_predicted is not None else "N/A"
            sp_str = f"{c.spice_simulated:.2f} {c.unit}" if c.spice_simulated is not None else "Unavailable"
            abs_str = f"{c.absolute_error:.2f} {c.unit}" if c.absolute_error is not None else "N/A"
            rel_str = f"{c.relative_error_pct:.1f}%" if c.relative_error_pct is not None else "N/A"

            rows.append({
                "Metric": c.metric_name,
                "Unit": c.unit,
                "ML Prediction": ml_str,
                "SPICE Result": sp_str,
                "Abs Error (|Δ|)": abs_str,
                "Rel Error (%)": rel_str,
                "Agreement": c.agreement_grade,
                "ML Source": c.ml_data_source,
                "SPICE Source": c.spice_data_source,
            })
        return pd.DataFrame(rows)


class SimulationValidator:
    """
    Validation engine strictly distinguishing ML approximations from SPICE simulation truths.
    Computes delta and error distributions with proper physical dimensional units.
    """

    def __init__(self):
        pass

    def validate_candidate(
        self,
        ml_preds: Dict[str, Any],
        spice_metrics: ExtractedMetrics,
        candidate_id: str = "CAND_0",
        topology: str = "miller_2stage",
    ) -> ValidationReport:
        """
        Validates ML predictions against actual ngspice simulation metrics.

        Args:
            ml_preds: Dictionary of ML model predictions (e.g. dc_gain, gain_bandwidth_product, etc.)
            spice_metrics: ExtractedMetrics instance from ngspice simulation.
            candidate_id: Identifier for candidate layout.
            topology: Circuit topology name.

        Returns:
            ValidationReport with metric comparisons, error analysis, and verdict.
        """
        comparisons: List[MetricComparison] = []
        rel_errors: List[float] = []

        # 1. DC Gain (dB)
        ml_gain = float(ml_preds.get("dc_gain", 0.0)) if "dc_gain" in ml_preds else None
        sp_gain = spice_metrics.dc_gain_db
        comp_gain = self._build_comparison(
            metric_name="DC Gain",
            unit="dB",
            ml_val=ml_gain,
            sp_val=sp_gain,
            ml_source="ML ExtraTrees / RF Proxy Model",
            sp_source="ngspice Level-1 / BSIM Small-Signal AC",
        )
        comparisons.append(comp_gain)
        if comp_gain.relative_error_pct is not None:
            rel_errors.append(comp_gain.relative_error_pct)

        # 2. Gain-Bandwidth Product (GBW / UGB)
        ml_gbw_mhz = None
        if "gain_bandwidth_product" in ml_preds:
            ml_gbw_raw = float(ml_preds["gain_bandwidth_product"])
            ml_gbw_mhz = ml_gbw_raw / 1e6 if ml_gbw_raw > 1000 else ml_gbw_raw
        sp_gbw_mhz = (spice_metrics.gain_bandwidth_product_hz / 1e6) if spice_metrics.gain_bandwidth_product_hz is not None else None
        comp_gbw = self._build_comparison(
            metric_name="Unity-Gain Bandwidth (GBW)",
            unit="MHz",
            ml_val=ml_gbw_mhz,
            sp_val=sp_gbw_mhz,
            ml_source="ML ExtraTrees / RF Proxy Model",
            sp_source="ngspice AC 0-dB Crossing Vector Interpolation",
        )
        comparisons.append(comp_gbw)
        if comp_gbw.relative_error_pct is not None:
            rel_errors.append(comp_gbw.relative_error_pct)

        # 3. Phase Margin (deg)
        ml_pm = float(ml_preds.get("phase_margin", 0.0)) if "phase_margin" in ml_preds else None
        sp_pm = spice_metrics.phase_margin_deg
        comp_pm = self._build_comparison(
            metric_name="Phase Margin",
            unit="deg",
            ml_val=ml_pm,
            sp_val=sp_pm,
            ml_source="ML ExtraTrees / RF Proxy Model",
            sp_source="ngspice Open-Loop Phase at Unity Frequency",
        )
        comparisons.append(comp_pm)
        if comp_pm.relative_error_pct is not None:
            rel_errors.append(comp_pm.relative_error_pct)

        # 4. Slew Rate (V/µs)
        ml_sr = float(ml_preds.get("slew_rate", 0.0)) if "slew_rate" in ml_preds else None
        sp_sr = spice_metrics.slew_rate_v_per_us
        comp_sr = self._build_comparison(
            metric_name="Slew Rate",
            unit="V/µs",
            ml_val=ml_sr,
            sp_val=sp_sr,
            ml_source="ML ExtraTrees / RF Proxy Model",
            sp_source="ngspice Large-Signal Step Transient max(dV/dt)",
        )
        comparisons.append(comp_sr)
        if comp_sr.relative_error_pct is not None:
            rel_errors.append(comp_sr.relative_error_pct)

        # 5. Power Consumption (mW)
        ml_power_mw = None
        if "power_consumption" in ml_preds:
            ml_power_raw = float(ml_preds["power_consumption"])
            ml_power_mw = ml_power_raw * 1000.0 if ml_power_raw < 0.1 else ml_power_raw
        sp_power_mw = spice_metrics.power_consumption_mw
        comp_power = self._build_comparison(
            metric_name="Power Consumption",
            unit="mW",
            ml_val=ml_power_mw,
            sp_val=sp_power_mw,
            ml_source="ML ExtraTrees / RF Proxy Model",
            sp_source="ngspice DC Operating Point (VDD * I_delivered)",
        )
        comparisons.append(comp_power)
        if comp_power.relative_error_pct is not None:
            rel_errors.append(comp_power.relative_error_pct)

        mean_rel_err = float(np.mean(rel_errors)) if len(rel_errors) > 0 else None

        # Verdict
        if spice_metrics.convergence_status != "Converged":
            verdict = f"REJECTED ({spice_metrics.convergence_status})"
            summary = f"Candidate {candidate_id} failed physical electrical validation ({spice_metrics.convergence_status})."
        elif mean_rel_err is None:
            verdict = "INCOMPLETE (No Valid Metrics)"
            summary = f"Candidate {candidate_id} simulation completed but yielded no measurable metrics."
        elif mean_rel_err <= 15.0:
            verdict = "VALIDATED (High Agreement)"
            summary = f"Candidate {candidate_id} passed SPICE validation with high ML-to-physics fidelity (mean error {mean_rel_err:.1f}%)."
        elif mean_rel_err <= 35.0:
            verdict = "VALIDATED (Acceptable Alignment)"
            summary = f"Candidate {candidate_id} verified by SPICE simulation with acceptable agreement (mean error {mean_rel_err:.1f}%)."
        else:
            verdict = "FLAGGED (High Divergence)"
            summary = f"Candidate {candidate_id} shows significant deviation between ML approximation and SPICE physics (mean error {mean_rel_err:.1f}%)."

        return ValidationReport(
            candidate_id=candidate_id,
            topology=topology,
            comparisons=comparisons,
            mean_relative_error_pct=float(np.round(mean_rel_err, 1)) if mean_rel_err is not None else None,
            simulation_status=spice_metrics.convergence_status,
            validation_verdict=verdict,
            engineering_summary=summary,
        )

    def _build_comparison(
        self,
        metric_name: str,
        unit: str,
        ml_val: Optional[float],
        sp_val: Optional[float],
        ml_source: str,
        sp_source: str,
    ) -> MetricComparison:
        """Constructs an individual MetricComparison with agreement grading, safely handling None."""
        if sp_val is None or ml_val is None or not np.isfinite(sp_val) or not np.isfinite(ml_val):
            return MetricComparison(
                metric_name=metric_name,
                unit=unit,
                ml_predicted=float(np.round(ml_val, 2)) if ml_val is not None and np.isfinite(ml_val) else None,
                spice_simulated=float(np.round(sp_val, 2)) if sp_val is not None and np.isfinite(sp_val) else None,
                absolute_error=None,
                relative_error_pct=None,
                agreement_grade="Unavailable (Not Measured)",
                ml_data_source=ml_source,
                spice_data_source="ngspice (Unavailable/Failed)" if sp_val is None else sp_source,
            )

        abs_err = abs(sp_val - ml_val)
        denom = abs(sp_val) if abs(sp_val) > 1e-6 else 1.0
        rel_err_pct = (abs_err / denom) * 100.0

        if rel_err_pct <= 10.0:
            grade = "Excellent (≤10%)"
        elif rel_err_pct <= 25.0:
            grade = "Good (10–25%)"
        elif rel_err_pct <= 50.0:
            grade = "Moderate (25–50%)"
        else:
            grade = "Divergent (>50%)"

        return MetricComparison(
            metric_name=metric_name,
            unit=unit,
            ml_predicted=float(np.round(ml_val, 2)),
            spice_simulated=float(np.round(sp_val, 2)),
            absolute_error=float(np.round(abs_err, 2)),
            relative_error_pct=float(np.round(rel_err_pct, 1)),
            agreement_grade=grade,
            ml_data_source=ml_source,
            spice_data_source=sp_source,
        )
