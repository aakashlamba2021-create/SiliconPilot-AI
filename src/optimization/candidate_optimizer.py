"""
Advanced Candidate Layout Optimizer & Multi-Objective Engine — SiliconPilot AI
Provides:
  - Controlled perturbation within empirical training bounds
  - Out-of-Distribution (OOD) parameter rejection and constraint filtering
  - Multi-objective evaluation with tree-ensemble prediction uncertainty (sigma)
  - True N-dimensional Pareto frontier non-dominated sorting
  - Detailed optimization explainability and baseline-vs-candidate delta auditing
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.models.model_policy import ModelTrustPolicy

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Empirical Parameter Bounds (Derived from Training Benchmark Distributions)
# ---------------------------------------------------------------------------
SUPPORTED_PARAMETER_BOUNDS: Dict[str, Dict[str, Any]] = {
    "transistor_width_W": {"min": 0.5, "max": 50.0, "unit": "µm", "type": "float"},
    "transistor_length_L": {"min": 0.18, "max": 2.0, "unit": "µm", "type": "float"},
    "finger_count": {"min": 1, "max": 16, "unit": "count", "type": "int"},
    "multiplicity": {"min": 1, "max": 8, "unit": "multiplier", "type": "int"},
    "finger_arrangement": {"min": 1, "max": 4, "unit": "style", "type": "int"},
    "x_position": {"min": 0.0, "max": 100.0, "unit": "µm", "type": "float"},
    "y_position": {"min": 0.0, "max": 100.0, "unit": "µm", "type": "float"},
    "orientation": {"allowed": [0, 90, 180, 270], "unit": "degrees", "type": "categorical"},
    "delta_x": {"min": -5.0, "max": 5.0, "unit": "µm", "type": "float"},
    "delta_y": {"min": -5.0, "max": 5.0, "unit": "µm", "type": "float"},
    "aspect_ratio_W_L": {"min": 0.25, "max": 280.0, "unit": "ratio", "type": "float"},
}

ELECTRICAL_METRIC_COLUMNS = (
    "dc_gain",
    "gain_bandwidth_product",
    "phase_margin",
    "slew_rate",
)


def _uniform_evidence_origin(
    df: pd.DataFrame,
    column: str,
    allowed_origins: set[str],
) -> Optional[str]:
    """Return one explicit, uniform evidence origin; fail closed otherwise."""
    if column not in df.columns or df.empty:
        return None
    values = df[column]
    if values.isna().any():
        return None
    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            return None
        origin = value.strip().lower()
        if origin not in allowed_origins:
            return None
        normalized.add(origin)
    return next(iter(normalized)) if len(normalized) == 1 else None


class CandidateOptimizer:
    """
    Intelligent analog IC layout candidate generator, multi-objective Pareto optimizer,
    and explainability engine.
    """

    def __init__(
        self,
        min_phase_margin: float = 50.0,  # degrees
        min_dc_gain: float = 50.0,       # dB
        min_gbw: float = 2.0e6,          # Hz (2 MHz)
        min_slew_rate: float = 2.0,      # V/µs
        max_area: float = 5000.0,        # µm²
        min_drc_clean_prob: float = 0.70,  # Retained for call compatibility; synthetic DRC model is not used.
    ) -> None:
        self.min_phase_margin = min_phase_margin
        self.min_dc_gain = min_dc_gain
        self.min_gbw = min_gbw
        self.min_slew_rate = min_slew_rate
        self.max_area = max_area
        self.min_drc_clean_prob = min_drc_clean_prob

    # ------------------------------------------------------------------
    # 1. Parameter Validation & OOD Detection
    # ------------------------------------------------------------------

    @staticmethod
    def validate_parameter_bounds(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates whether design parameters stay strictly within empirical training domain.
        Returns (is_valid, violation_reasons).
        """
        violations = []
        for param, bounds in SUPPORTED_PARAMETER_BOUNDS.items():
            if param not in params:
                continue
            val = params[param]

            if "allowed" in bounds:
                if val not in bounds["allowed"]:
                    violations.append(f"{param}={val} not in allowed options {bounds['allowed']}")
            else:
                p_min, p_max = bounds["min"], bounds["max"]
                if val < p_min or val > p_max:
                    violations.append(f"{param}={val:.3f} outside supported range [{p_min}, {p_max}] {bounds.get('unit', '')}")

        # Check aspect ratio
        if "transistor_width_W" in params and "transistor_length_L" in params:
            wl = params["transistor_width_W"] / max(1e-6, params["transistor_length_L"])
            wl_bounds = SUPPORTED_PARAMETER_BOUNDS["aspect_ratio_W_L"]
            if wl < wl_bounds["min"] or wl > wl_bounds["max"]:
                violations.append(f"Aspect ratio W/L={wl:.1f} outside valid range [{wl_bounds['min']}, {wl_bounds['max']}]")

        return len(violations) == 0, violations

    # ------------------------------------------------------------------
    # 2. Controlled Candidate Generation with Uncertainty
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        baseline_params: Dict[str, Any],
        perf_bundle: Dict[str, Any],
        qual_bundle: Optional[Dict[str, Any]] = None,
        num_candidates: int = 36,
        perturbation_scale: float = 1.0,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generates design variations centered around baseline parameters via controlled Gaussian/discrete steps.
        Evaluates candidates with the exploratory electrical model and computes ensemble spread.
        The quarantined DRC classifier is intentionally ignored.
        Rejects out-of-distribution (OOD) parameters.
        """
        rng = np.random.default_rng(random_seed)
        perf_status = ModelTrustPolicy().get_status("osiris_electrical_performance_predictor")
        if perf_status.get("trust_label") not in {"exploratory", "approved"} or "exploratory_candidate_generation" not in perf_status.get("permitted_uses", []):
            raise RuntimeError("Electrical model is not approved for exploratory candidate generation.")
        rows: List[Dict[str, Any]] = []

        base_w = float(baseline_params.get("transistor_width_W", 25.0))
        base_l = float(baseline_params.get("transistor_length_L", 0.5))
        base_top = str(baseline_params.get("topology", "miller_2stage"))
        base_fc = int(baseline_params.get("finger_count", 4))
        base_m = int(baseline_params.get("multiplicity", 2))
        base_x = float(baseline_params.get("x_position", 50.0))
        base_y = float(baseline_params.get("y_position", 50.0))
        base_orient = int(baseline_params.get("orientation", 0))

        # Always include baseline as candidate 0
        baseline_dict = {
            "candidate_id": "cand_baseline",
            "is_baseline": True,
            "topology": base_top,
            "transistor_width_W": base_w,
            "transistor_length_L": base_l,
            "finger_count": base_fc,
            "multiplicity": base_m,
            "finger_arrangement": int(baseline_params.get("finger_arrangement", 1)),
            "x_position": base_x,
            "y_position": base_y,
            "orientation": base_orient,
            "delta_x": 0.0,
            "delta_y": 0.0,
            "perturbation_id": 0,
            "top_miller_2stage": 1.0 if base_top == "miller_2stage" else 0.0,
            "top_telescopic": 1.0 if base_top == "telescopic" else 0.0,
        }
        rows.append(baseline_dict)

        # Generate variations
        finger_options = [1, 2, 4, 6, 8, 12, 16]
        multiplicity_options = [1, 2, 4, 6, 8]
        orientation_options = [0, 90, 180, 270]

        attempts = 0
        while len(rows) < num_candidates and attempts < num_candidates * 5:
            attempts += 1
            # Controlled perturbation around baseline
            w_factor = float(rng.normal(1.0, 0.15 * perturbation_scale))
            l_factor = float(rng.normal(1.0, 0.10 * perturbation_scale))

            cand_w = np.clip(round(base_w * w_factor, 2), 0.5, 50.0)
            cand_l = np.clip(round(base_l * l_factor, 3), 0.18, 2.0)

            # Sample fingering near baseline
            fc_idx = finger_options.index(base_fc) if base_fc in finger_options else 2
            fc_step = int(rng.choice([-1, 0, 1]))
            new_fc_idx = np.clip(fc_idx + fc_step, 0, len(finger_options) - 1)
            cand_fc = finger_options[new_fc_idx]

            m_idx = multiplicity_options.index(base_m) if base_m in multiplicity_options else 1
            m_step = int(rng.choice([-1, 0, 1]))
            new_m_idx = np.clip(m_idx + m_step, 0, len(multiplicity_options) - 1)
            cand_m = multiplicity_options[new_m_idx]

            # Perturbation offsets (spatial displacement)
            dx = round(float(rng.normal(0.0, 1.5 * perturbation_scale)), 3)
            dy = round(float(rng.normal(0.0, 1.5 * perturbation_scale)), 3)
            dx = np.clip(dx, -5.0, 5.0)
            dy = np.clip(dy, -5.0, 5.0)

            cand_x = np.clip(round(base_x + dx, 2), 0.0, 100.0)
            cand_y = np.clip(round(base_y + dy, 2), 0.0, 100.0)

            orient = base_orient if rng.uniform() > 0.3 else int(rng.choice(orientation_options))

            candidate_dict = {
                "candidate_id": f"cand_{len(rows):02d}",
                "is_baseline": False,
                "topology": base_top,
                "transistor_width_W": cand_w,
                "transistor_length_L": cand_l,
                "finger_count": cand_fc,
                "multiplicity": cand_m,
                "finger_arrangement": int(rng.choice([1, 2, 3, 4])),
                "x_position": cand_x,
                "y_position": cand_y,
                "orientation": orient,
                "delta_x": dx,
                "delta_y": dy,
                "perturbation_id": len(rows) % 10,
                "top_miller_2stage": 1.0 if base_top == "miller_2stage" else 0.0,
                "top_telescopic": 1.0 if base_top == "telescopic" else 0.0,
            }

            # Check bounds
            is_valid, _ = self.validate_parameter_bounds(candidate_dict)
            if is_valid:
                rows.append(candidate_dict)

        df = pd.DataFrame(rows)

        # ------------------------------------------------------------------
        # Model-Based Inference with Ensemble Uncertainty
        # ------------------------------------------------------------------
        expected_features = perf_bundle["feature_names"]
        target_names = perf_bundle["target_names"]

        feat_df = df.copy()
        for col in expected_features:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        X_input = feat_df[expected_features]

        # 1. Electrical Performance Prediction + Tree Uncertainty
        preprocessor = perf_bundle["preprocessor"]
        perf_model = perf_bundle["model"]
        X_scaled = preprocessor.transform(X_input)

        if hasattr(perf_model, "estimators_"):
            # Tree ensemble variance estimation
            all_tree_preds = np.array([tree.predict(X_scaled) for tree in perf_model.estimators_])
            mean_preds = all_tree_preds.mean(axis=0)
            std_preds = all_tree_preds.std(axis=0)
        else:
            mean_preds = perf_model.predict(X_scaled)
            std_preds = np.zeros_like(mean_preds)

        for idx, t_name in enumerate(target_names):
            df[t_name] = mean_preds[:, idx].round(2)
            df[f"{t_name}_uncertainty_sigma"] = std_preds[:, idx].round(3)
        df["electrical_result_type"] = "Synthetic-data-trained exploratory estimate; not SPICE simulated or measured"
        df["electrical_evidence_origin"] = "synthetic"

        # 2. Geometric Active Area Calculation
        df["area"] = (
            df["transistor_width_W"]
            * df["transistor_length_L"]
            * df["finger_count"]
            * df["multiplicity"]
            * 2.5
        ).round(1)
        df["area_evidence_origin"] = "parameter_proxy"
        df["area_result_type"] = "Formula-based parameter proxy; not extracted physical area"

        # Verification is not inferred. Generated candidates have no DRC evidence.
        df["drc_status"] = "Not verified"

        # Perturbation displacement
        df["perturbation_displacement"] = np.sqrt(df["delta_x"] ** 2 + df["delta_y"] ** 2).round(3)

        return df

    # ------------------------------------------------------------------
    # 3. True Non-Dominated Sorting (Pareto Frontier)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_pareto_frontiers(
        df: pd.DataFrame,
        objectives: Dict[str, str],  # col_name: 'maximize' or 'minimize'
    ) -> pd.DataFrame:
        """
        Computes non-dominated Pareto ranking across arbitrary N objectives.
        Pareto Rank 1: Non-dominated solutions (true Pareto frontier).
        Pareto Rank 2: Non-dominated among remaining solutions, etc.
        """
        res = df.copy()
        valid_objs = {k: v for k, v in objectives.items() if k in res.columns}
        if not valid_objs:
            res["pareto_rank"] = 1
            res["is_pareto_optimal"] = True
            return res

        # Convert to minimization space
        M = np.zeros((len(res), len(valid_objs)))
        for idx, (col, direction) in enumerate(valid_objs.items()):
            vals = res[col].to_numpy(dtype=float)
            M[:, idx] = -vals if direction.lower() == "maximize" else vals

        n_samples = len(res)
        ranks = np.zeros(n_samples, dtype=int)
        remaining = list(range(n_samples))
        current_rank = 1

        while remaining:
            frontier = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    # j dominates i if j <= i everywhere and j < i in at least one objective
                    if np.all(M[j] <= M[i]) and np.any(M[j] < M[i]):
                        dominated = True
                        break
                if not dominated:
                    frontier.append(i)

            if not frontier:
                # Break cycle if any remaining are identical
                for r in remaining:
                    ranks[r] = current_rank
                break

            for f in frontier:
                ranks[f] = current_rank
                remaining.remove(f)
            current_rank += 1

        res["pareto_rank"] = ranks
        res["is_pareto_optimal"] = res["pareto_rank"] == 1
        return res

    # ------------------------------------------------------------------
    # 4. Multi-Objective Composite Scoring & Hard Constraints
    # ------------------------------------------------------------------

    def evaluate_and_rank_candidates(
        self,
        df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
        custom_constraints: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Applies hard constraint filters, computes Pareto frontiers, and scores candidates
        using normalized multi-objective weights.
        """
        ranked = df.copy()

        electrical_origin = _uniform_evidence_origin(
            ranked, "electrical_evidence_origin", {"simulated", "measured"}
        )
        area_origin = _uniform_evidence_origin(
            ranked, "area_evidence_origin", {"simulated", "measured", "parameter_proxy"}
        )
        use_electrical_metrics = (
            electrical_origin is not None
            and any(column in ranked.columns for column in ELECTRICAL_METRIC_COLUMNS)
        )
        use_area = area_origin is not None and "area" in ranked.columns
        ranking_components = []
        if use_electrical_metrics:
            ranking_components.append(f"electrical metrics ({electrical_origin})")
        if use_area:
            ranking_components.append(f"area ({area_origin})")
        if "perturbation_displacement" in ranked.columns:
            ranking_components.append("parameter-derived displacement")
        ranking_basis = " + ".join(ranking_components) if ranking_components else "no approved ranking evidence"
        ranked["ranking_basis"] = ranking_basis
        ranked["electrical_metrics_used_for_ranking"] = use_electrical_metrics

        # Update constraints if provided
        min_gain = custom_constraints.get("min_dc_gain", self.min_dc_gain) if custom_constraints else self.min_dc_gain
        min_pm = custom_constraints.get("min_phase_margin", self.min_phase_margin) if custom_constraints else self.min_phase_margin
        min_gbw = custom_constraints.get("min_gbw", self.min_gbw) if custom_constraints else self.min_gbw
        max_area = custom_constraints.get("max_area", self.max_area) if custom_constraints else self.max_area

        # 1. Constraint Auditing
        violations_list = []
        is_valid_list = []
        for _, row in ranked.iterrows():
            v = []
            evaluation_notes = []
            if use_electrical_metrics and "dc_gain" in row and row["dc_gain"] < min_gain:
                v.append(f"DC Gain {row['dc_gain']:.1f} dB < {min_gain} dB")
            if use_electrical_metrics and "phase_margin" in row and row["phase_margin"] < min_pm:
                v.append(f"Phase Margin {row['phase_margin']:.1f}° < {min_pm}° (Stability)")
            if use_electrical_metrics and "gain_bandwidth_product" in row and row["gain_bandwidth_product"] < min_gbw:
                v.append(f"GBW {row['gain_bandwidth_product']/1e6:.1f} MHz < {min_gbw/1e6:.1f} MHz")
            if use_area and row["area"] > max_area:
                v.append(f"Area {row['area']:.1f} µm² > {max_area} µm²")

            if not use_electrical_metrics:
                evaluation_notes.append("Electrical constraints not evaluated: no uniform measured or simulated evidence")
            if not use_area:
                evaluation_notes.append("Area constraint not evaluated: no uniform measured, simulated, or parameter-proxy evidence")

            status_parts = v + evaluation_notes
            violations_list.append("; ".join(status_parts) if status_parts else "Satisfied")
            is_valid_list.append(len(v) == 0)

        ranked["constraint_status"] = violations_list
        ranked["is_valid_candidate"] = is_valid_list
        ranked["is_clean_candidate"] = is_valid_list  # Backwards compatibility alias
        ranked["violation_status"] = violations_list   # Backwards compatibility alias

        # 2. Pareto Non-Dominated Sorting
        objectives_dir = {}
        if use_electrical_metrics:
            objectives_dir.update({
                column: "maximize"
                for column in ELECTRICAL_METRIC_COLUMNS
                if column in ranked.columns
            })
        if use_area:
            objectives_dir["area"] = "minimize"
        if "perturbation_displacement" in ranked.columns:
            objectives_dir["perturbation_displacement"] = "minimize"
        ranked = self.compute_pareto_frontiers(ranked, objectives=objectives_dir)

        # 3. Normalized Weighted Score
        default_weights = {
            "dc_gain": 0.25,
            "gain_bandwidth_product": 0.20,
            "phase_margin": 0.15,
            "slew_rate": 0.10,
            "area": -0.20,
            "perturbation_displacement": -0.10,
        }
        w = weights or default_weights
        # Ignore legacy classifier columns/weights even when supplied by callers.
        w = {name: value for name, value in w.items() if name not in {"drc_clean", "drc_clean_prob"}}
        allowed_weights = set(objectives_dir)
        w = {name: value for name, value in w.items() if name in allowed_weights}

        score = np.zeros(len(ranked))
        for col, weight in w.items():
            if col in ranked.columns:
                c_min, c_max = ranked[col].min(), ranked[col].max()
                if c_max > c_min:
                    norm = (ranked[col] - c_min) / (c_max - c_min)
                else:
                    norm = 0.5
                score += weight * norm

        # Penalty for constraint violations & Pareto rank bonus
        pareto_bonus = np.where(ranked["is_pareto_optimal"], 0.25, 0.0)
        violation_penalty = np.where(ranked["is_valid_candidate"], 0.0, -1.5)
        score = score + pareto_bonus + violation_penalty

        ranked["siliconpilot_score"] = (score * 100).round(2)
        ranked = ranked.sort_values(by="siliconpilot_score", ascending=False).reset_index(drop=True)
        return ranked

    def rank_candidates(
        self,
        candidates_df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Backwards compatibility wrapper for evaluate_and_rank_candidates."""
        return self.evaluate_and_rank_candidates(candidates_df, weights=weights)

    # ------------------------------------------------------------------
    # 5. Optimization Explainability & Delta Audit
    # ------------------------------------------------------------------

    @staticmethod
    def generate_detailed_explanation(
        recommended: pd.Series,
        baseline: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Constructs an explainability report breaking down parameter shifts,
        predicted physical/electrical metrics with confidence intervals,
        and Pareto trade-off rationale.
        """
        report: Dict[str, Any] = {
            "candidate_id": recommended.get("candidate_id", str(recommended.name)),
            "siliconpilot_score": recommended.get("siliconpilot_score", 0.0),
            "pareto_rank": int(recommended.get("pareto_rank", 1)),
            "is_pareto_optimal": bool(recommended.get("is_pareto_optimal", False)),
            "constraint_status": recommended.get("constraint_status", "Satisfied"),
            "drc_status": "Not verified",
            "parameter_shifts": {},
            "predicted_metrics": {},
            "tradeoff_rationale": [],
        }

        # Parameter shifts vs baseline
        params_to_track = ["transistor_width_W", "transistor_length_L", "finger_count", "multiplicity", "delta_x", "delta_y"]
        for p in params_to_track:
            if p in recommended:
                val = float(recommended[p]) if isinstance(recommended[p], (int, float, np.integer, np.floating)) else recommended[p]
                b_val = None
                delta_val = 0.0
                if baseline is not None and p in baseline:
                    b_val = float(baseline[p]) if isinstance(baseline[p], (int, float, np.integer, np.floating)) else baseline[p]
                    if isinstance(val, (int, float)) and isinstance(b_val, (int, float)):
                        delta_val = float(val - b_val)
                shift_str = f"{val}" if b_val is None else f"{b_val} → {val} (Δ: {delta_val:+.3f})"
                report["parameter_shifts"][p] = {
                    "original": b_val,
                    "proposed": val,
                    "delta": delta_val,
                    "summary": shift_str,
                }

        # Predicted metrics with model uncertainty
        metrics_to_track = [
            ("dc_gain", "dB"),
            ("gain_bandwidth_product", "Hz"),
            ("phase_margin", "°"),
            ("slew_rate", "V/µs"),
            ("area", "µm²"),
        ]
        for m, unit in metrics_to_track:
            if m in recommended:
                val = float(recommended[m])
                sigma = float(recommended.get(f"{m}_uncertainty_sigma", 0.0))
                b_val = float(baseline[m]) if baseline is not None and m in baseline else None
                pct_change = ((val - b_val) / (abs(b_val) + 1e-6) * 100) if b_val is not None else None

                report["predicted_metrics"][m] = {
                    "value": val,
                    "uncertainty_sigma": sigma,
                    "unit": unit,
                    "baseline_value": b_val,
                    "improvement_pct": pct_change,
                }

        # Rationale construction
        gain_val = recommended.get("dc_gain", 0.0)
        pm_val = recommended.get("phase_margin", 0.0)
        area_val = recommended.get("area", 0.0)

        report["tradeoff_rationale"].append(
            f"**Pareto Status:** Rank {report['pareto_rank']} "
            f"({'Non-Dominated Optimal Solution' if report['is_pareto_optimal'] else 'Sub-optimal candidate dominated by frontier'})."
        )
        if baseline is not None:
            if "area" in baseline and "area" in recommended:
                area_delta = ((area_val - baseline["area"]) / baseline["area"]) * 100
                report["tradeoff_rationale"].append(
                    f"**Area Efficiency:** {area_delta:+.1f}% physical footprint shift ({area_val:.1f} vs {baseline['area']:.1f} µm²)."
                )
            if "dc_gain" in baseline and "dc_gain" in recommended:
                gain_delta = gain_val - baseline["dc_gain"]
                report["tradeoff_rationale"].append(
                    f"**Small-Signal Gain Benefit:** {gain_delta:+.2f} dB differential gain benefit over baseline."
                )

        report["tradeoff_rationale"].append(
            f"**Stability vs Speed Trade-off:** Phase margin is {pm_val:.1f}° with predicted GBW of "
            f"{recommended.get('gain_bandwidth_product', 0)/1e6:.2f} MHz."
        )
        report["tradeoff_rationale"].append("**DRC Status:** Not verified; no actual DRC execution result is available.")

        return report

    # ------------------------------------------------------------------
    # 6. SPICE-in-the-Loop Simulation Validation (Phase 4)
    # ------------------------------------------------------------------

    def run_simulation_validation(
        self,
        ranked_candidates: pd.DataFrame,
        top_k: int = 3,
        topology: str = "miller_2stage",
        pex_parasitics: Optional[Dict[str, float]] = None,
        record_in_db: bool = True,
        min_phase_margin: float = 40.0,
        filter_pareto_front: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes SPICE-in-the-loop validation for the top-k Pareto-optimal candidates.
        Translates candidate sizing parameters to actual SPICE netlists, simulates via ngspice,
        parses frequency and time-domain waveforms, validates ML estimates against physics,
        and declares a simulation-verified winner.

        Args:
            ranked_candidates: DataFrame of ranked candidates from rank_candidates_pareto.
            top_k: Number of leading candidates to simulate (default 3).
            topology: Circuit topology ('miller_2stage', 'five_transistors_ota', etc.).
            pex_parasitics: Optional dictionary of layout parasitics for netlist injection.
            record_in_db: Whether to record simulation validation runs in FeedbackDatabase.
            min_phase_margin: Stability threshold in degrees (default 40.0°).
            filter_pareto_front: Whether to filter strictly for non-dominated (rank 1) candidates first.

        Returns:
            Dictionary containing:
            - evaluated_candidates: List of summary dicts
            - validation_reports: Dict of ValidationReport objects
            - spice_metrics: Dict of ExtractedMetrics objects
            - winner_id: str
            - winner_justification: str
            - comparison_dataframe: pd.DataFrame
        """
        from src.simulation.ngspice_runner import NGSpiceRunner
        from src.simulation.netlist_manager import NetlistManager
        from src.simulation.simulation_parser import SimulationParser
        from src.simulation.simulation_metrics import SimulationMetricsExtractor
        from src.simulation.simulation_validator import SimulationValidator
        from src.simulation.feedback_db import FeedbackDatabase

        runner = NGSpiceRunner()
        net_mgr = NetlistManager()
        parser = SimulationParser()
        extractor = SimulationMetricsExtractor()
        validator = SimulationValidator()
        db = FeedbackDatabase() if record_in_db else None

        if not runner.is_available():
            logger.warning("SPICE backend not available. Returning unsimulated candidate summary.")
            return {
                "success": False,
                "error": "ngspice backend not available",
                "evaluated_candidates": [],
                "winner_id": None,
                "winner_justification": "ngspice simulation engine unavailable.",
            }

        # Filter strictly for Pareto Rank 1 (non-dominated) candidates before simulating
        if filter_pareto_front:
            if "pareto_rank" in ranked_candidates.columns:
                rank1_cands = ranked_candidates[ranked_candidates["pareto_rank"] == 1]
            elif "is_pareto_optimal" in ranked_candidates.columns:
                rank1_cands = ranked_candidates[ranked_candidates["is_pareto_optimal"] == True]
            else:
                # Fail closed: missing Pareto metadata when strict Pareto filtering is requested
                logger.warning("Strict Pareto filtering failed closed: candidates lack 'pareto_rank' or 'is_pareto_optimal'.")
                return {
                    "success": False,
                    "status": "Failed: Candidates lack Pareto metadata ('pareto_rank' or 'is_pareto_optimal' required)",
                    "comparison_table": [],
                    "evaluated_candidates": [],
                    "winner_id": None,
                    "winner_justification": "Pareto metadata missing; strict Pareto filtering rejected candidate set.",
                }

            if len(rank1_cands) == 0:
                # Fail closed: an empty rank-1 front must remain empty! Never fall back to rank 2!
                logger.warning("Strict Pareto filtering found 0 rank-1 candidates; failing closed.")
                return {
                    "success": False,
                    "status": "Failed: No rank-1 Pareto optimal candidates found",
                    "comparison_table": [],
                    "evaluated_candidates": [],
                    "winner_id": None,
                    "winner_justification": "No non-dominated (rank 1) Pareto candidates available for simulation.",
                }

            cands_to_evaluate = rank1_cands
        else:
            cands_to_evaluate = ranked_candidates

        # Apply deterministic ordering: composite_score descending, then index ascending
        if "composite_score" in cands_to_evaluate.columns:
            ordered_cands = cands_to_evaluate.sort_values(by="composite_score", ascending=False)
        elif "siliconpilot_score" in cands_to_evaluate.columns:
            ordered_cands = cands_to_evaluate.sort_values(by="siliconpilot_score", ascending=False)
        else:
            ordered_cands = cands_to_evaluate

        top_slice = ordered_cands.head(top_k)
        evaluated = []
        reports = {}
        spice_metrics_dict = {}
        comparison_rows = []

        for idx, row in top_slice.iterrows():
            cand_id = f"Candidate_{idx}" if "candidate_id" not in row else str(row["candidate_id"])
            cand_dict = row.to_dict()

            # 1. Generate AC Netlist & Run
            ac_netlist = net_mgr.generate_netlist(
                topology=topology,
                params=cand_dict,
                mode="ac",
                pex_parasitics=pex_parasitics,
            )
            res_ac = runner.run_netlist(ac_netlist, run_ac=True, run_tran=False, run_op=True)
            ac_data = parser.parse_ac_file(res_ac.ac_file)

            # 2. Generate TRAN Netlist & Run
            tran_netlist = net_mgr.generate_netlist(
                topology=topology,
                params=cand_dict,
                mode="tran",
                pex_parasitics=pex_parasitics,
            )
            res_tran = runner.run_netlist(tran_netlist, run_ac=False, run_tran=True, run_op=False)
            tran_data = parser.parse_tran_file(res_tran.tran_file)

            # 3. Extract Standard Physical Metrics (requires both AC and TRAN results)
            m = extractor.extract(
                sim_result=res_ac,
                ac_data=ac_data,
                tran_data=tran_data,
                vdd=float(cand_dict.get("vdd", 1.8)),
                sim_result_tran=res_tran,
                supply_source="VDD",
                supply_node="vdd",
            )
            spice_metrics_dict[cand_id] = m

            # 4. Validate ML Predictions vs SPICE Simulation Truth
            report = validator.validate_candidate(
                ml_preds=cand_dict,
                spice_metrics=m,
                candidate_id=cand_id,
                topology=topology,
            )
            reports[cand_id] = report

            # 5. Record to Feedback Database if requested (strictly require Converged status and valid error)
            if db is not None:
                try:
                    db.record_simulation(
                        circuit_id=cand_id,
                        topology=topology,
                        parameters=cand_dict,
                        ml_predictions=cand_dict,
                        spice_results=m.to_dict(),
                        errors={"mean_relative_error_pct": report.mean_relative_error_pct},
                        simulation_status=m.convergence_status,
                        verified_for_training=(
                            m.convergence_status == "Converged"
                            and report.mean_relative_error_pct is not None
                            and report.mean_relative_error_pct < 25.0
                            and m.phase_margin_deg is not None
                            and m.phase_margin_deg >= min_phase_margin
                        ),
                    )
                except Exception as e:
                    logger.debug(f"Failed to record simulation to feedback DB: {e}")

            summary_item = {
                "candidate_id": cand_id,
                "pareto_rank": cand_dict.get("pareto_rank", 1),
                "is_pareto_optimal": cand_dict.get("is_pareto_optimal", True),
                "composite_score": cand_dict.get("composite_score", 0.0),
                "ml_dc_gain": float(cand_dict.get("dc_gain", 0.0)),
                "spice_dc_gain": m.dc_gain_db,
                "ml_gbw_mhz": float(cand_dict.get("gain_bandwidth_product", 0.0)) / 1e6 if float(cand_dict.get("gain_bandwidth_product", 0.0)) > 1000 else float(cand_dict.get("gain_bandwidth_product", 0.0)),
                "spice_gbw_mhz": (m.gain_bandwidth_product_hz / 1e6) if m.gain_bandwidth_product_hz is not None else None,
                "ml_phase_margin": float(cand_dict.get("phase_margin", 0.0)),
                "spice_phase_margin": m.phase_margin_deg,
                "ml_slew_rate": float(cand_dict.get("slew_rate", 0.0)),
                "spice_slew_rate": m.slew_rate_v_per_us,
                "spice_power_mw": m.power_consumption_mw,
                "mean_rel_error_pct": report.mean_relative_error_pct,
                "simulation_status": m.convergence_status,
                "verdict": report.validation_verdict,
            }
            evaluated.append(summary_item)

            comparison_rows.append({
                "Candidate": cand_id,
                "Status": m.convergence_status,
                "SPICE Gain (dB)": f"{m.dc_gain_db:.1f}" if m.dc_gain_db is not None else "Unavailable",
                "ML Gain (dB)": f"{summary_item['ml_dc_gain']:.1f}",
                "SPICE GBW (MHz)": f"{m.gain_bandwidth_product_hz/1e6:.1f}" if m.gain_bandwidth_product_hz is not None else "Unavailable",
                "SPICE PM (°)": f"{m.phase_margin_deg:.1f}" if m.phase_margin_deg is not None else "Unavailable",
                "SPICE SR (V/µs)": f"{m.slew_rate_v_per_us:.1f}" if m.slew_rate_v_per_us is not None else "Unavailable",
                "SPICE Power (mW)": f"{m.power_consumption_mw:.3f}" if m.power_consumption_mw is not None else "Unavailable",
                "Mean Error (%)": f"{report.mean_relative_error_pct:.1f}%" if report.mean_relative_error_pct is not None else "N/A",
                "Validation Verdict": report.validation_verdict,
            })

        # 6. Select Simulation-Validated Winner
        # Filter strictly for converged, stable candidates with measured PM >= min_phase_margin
        converged_cands = [c for c in evaluated if c["simulation_status"] == "Converged"]
        stable_cands = [
            c for c in converged_cands
            if c["spice_phase_margin"] is not None
            and c["spice_phase_margin"] >= min_phase_margin
            and c["spice_dc_gain"] is not None
        ]

        if stable_cands:
            # Sort by SPICE DC Gain descending, then GBW descending
            stable_cands.sort(key=lambda x: (x["spice_dc_gain"], x["spice_gbw_mhz"] or 0.0), reverse=True)
            winner = stable_cands[0]
            winner_id = winner["candidate_id"]
            justification = (
                f"{winner_id} is selected as the Simulation-Validated Winner. "
                f"It achieved {winner['spice_dc_gain']:.1f} dB SPICE gain with robust stability "
                f"(Phase Margin {winner['spice_phase_margin']:.1f}°, exceeding the {min_phase_margin:.0f}° threshold) "
                f"and {winner['spice_gbw_mhz']:.1f} MHz unity-gain bandwidth."
            )
        else:
            winner_id = None
            justification = (
                f"No candidate achieved unconditional physical stability (Phase Margin ≥ {min_phase_margin:.0f}° with "
                "converged SPICE metrics). Stability gating eliminated all candidates."
            )

        return {
            "success": True,
            "evaluated_candidates": evaluated,
            "validation_reports": reports,
            "spice_metrics": spice_metrics_dict,
            "winner_id": winner_id,
            "winner_justification": justification,
            "comparison_dataframe": pd.DataFrame(comparison_rows),
        }
