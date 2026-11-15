"""
Phase 3 Advanced Optimizer & Visualization Test Suite — SiliconPilot AI
Tests:
  1. Parameter bounds validation & OOD detection
  2. Model-based candidate generation & uncertainty calculation
  3. N-dimensional Pareto frontier non-dominated sorting correctness
  4. Multi-objective objective directionality (max vs min)
  5. Constraint violation detection and penalty scoring
  6. Explainability report generation and delta auditing
  7. Layout visualizer Plotly figures rendering
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.candidate_optimizer import CandidateOptimizer, SUPPORTED_PARAMETER_BOUNDS
from src.models.model_registry import ModelRegistry
from src.models.model_policy import ModelTrustPolicy
from dashboard.components.layout_visualizer import (
    render_transistor_layout,
    render_layout_comparison,
    render_pareto_tradeoff_chart,
)


def test_parameter_bounds_and_ood():
    print("Testing parameter bounds validation...")
    valid_params = {
        "transistor_width_W": 20.0,
        "transistor_length_L": 0.5,
        "finger_count": 4,
        "multiplicity": 2,
        "orientation": 0,
        "delta_x": 0.5,
        "delta_y": -0.5,
    }
    is_valid, violations = CandidateOptimizer.validate_parameter_bounds(valid_params)
    assert is_valid, f"Expected valid, got violations: {violations}"

    # Invalid width (exceeds max 50 µm)
    invalid_params = valid_params.copy()
    invalid_params["transistor_width_W"] = 75.0
    is_valid, violations = CandidateOptimizer.validate_parameter_bounds(invalid_params)
    assert not is_valid, "Expected out-of-bounds width to be rejected"
    assert any("transistor_width_W" in v for v in violations)

    # Invalid orientation
    invalid_orient = valid_params.copy()
    invalid_orient["orientation"] = 45  # Not in [0, 90, 180, 270]
    is_valid, violations = CandidateOptimizer.validate_parameter_bounds(invalid_orient)
    assert not is_valid, "Expected non-orthogonal orientation to be rejected"

    print("✅ test_parameter_bounds_and_ood passed")


def test_pareto_dominance_correctness():
    print("Testing Pareto non-dominated sorting algorithm...")
    # Candidate A: Area = 100 (better), Gain = 70 (better) -> Strictly dominates B
    # Candidate B: Area = 200 (worse), Gain = 50 (worse)
    # Candidate C: Area = 300 (worse), Gain = 80 (better) -> Non-dominated trade-off with A
    df = pd.DataFrame([
        {"candidate_id": "cand_A", "area": 100.0, "dc_gain": 70.0},
        {"candidate_id": "cand_B", "area": 200.0, "dc_gain": 50.0},
        {"candidate_id": "cand_C", "area": 300.0, "dc_gain": 80.0},
    ])
    objs = {"area": "minimize", "dc_gain": "maximize"}
    res = CandidateOptimizer.compute_pareto_frontiers(df, objectives=objs)

    # Candidate A and C must be Pareto Optimal (Rank 1)
    # Candidate B must be dominated (Rank 2)
    pareto_set = set(res[res["is_pareto_optimal"]]["candidate_id"])
    assert pareto_set == {"cand_A", "cand_C"}, f"Expected {{cand_A, cand_C}}, got {pareto_set}"
    assert res.loc[res["candidate_id"] == "cand_B", "pareto_rank"].iloc[0] == 2
    print("✅ test_pareto_dominance_correctness passed")


def test_candidate_generation_and_uncertainty():
    print("Testing candidate generation with live models & uncertainty...")
    reg = ModelRegistry()
    perf = reg.load_model_for_permitted_use(
        "osiris_electrical_performance_predictor", "exploratory_candidate_generation"
    )

    opt = CandidateOptimizer()
    base = {
        "transistor_width_W": 25.0,
        "transistor_length_L": 0.5,
        "finger_count": 4,
        "multiplicity": 2,
        "topology": "miller_2stage",
    }
    cands = opt.generate_candidates(base, perf, num_candidates=16, random_seed=42)

    assert len(cands) == 16
    assert cands.iloc[0]["is_baseline"] == True
    # Verify uncertainty columns exist
    assert "dc_gain_uncertainty_sigma" in cands.columns
    assert "phase_margin_uncertainty_sigma" in cands.columns
    assert "drc_status" in cands.columns
    assert (cands["drc_status"] == "Not verified").all()
    assert "drc_clean_prob" not in cands.columns
    assert cands["electrical_result_type"].str.contains("not SPICE simulated or measured").all()
    assert (cands["electrical_evidence_origin"] == "synthetic").all()
    assert (cands["area_evidence_origin"] == "parameter_proxy").all()
    # Check that predictions are non-trivial
    assert (cands["dc_gain"] > 0).all()
    assert (cands["area"] > 0).all()
    print("✅ test_candidate_generation_and_uncertainty passed")


def test_model_trust_provenance_policy():
    policy = ModelTrustPolicy()
    assert policy.get_status("layout_verification_classifier")["trust_label"] == "quarantined"
    assert policy.get_status("osiris_electrical_performance_predictor")["trust_label"] == "exploratory"
    assert policy.get_status("osiris_electrical_performance_predictor")["data_origin"] == "synthetic"
    assert policy.get_status("iclr_layout_geometry_predictor")["trust_label"] == "restricted"
    assert policy.get_status("osiris_area_pex_predictor")["data_origin"] == "benchmark"


def test_drc_columns_do_not_affect_ranking():
    opt = CandidateOptimizer()
    base = pd.DataFrame([
        {"candidate_id": "a", "dc_gain": 60.0, "phase_margin": 60.0, "area": 1000.0,
         "drc_clean_prob": 0.0, "drc_clean": False},
        {"candidate_id": "b", "dc_gain": 55.0, "phase_margin": 55.0, "area": 1200.0,
         "drc_clean_prob": 1.0, "drc_clean": True},
    ])
    no_drc = base.drop(columns=["drc_clean_prob", "drc_clean"])
    ranked_with = opt.evaluate_and_rank_candidates(base, weights={"dc_gain": 1.0, "drc_clean_prob": 100.0})
    ranked_without = opt.evaluate_and_rank_candidates(no_drc, weights={"dc_gain": 1.0})
    assert ranked_with["candidate_id"].tolist() == ranked_without["candidate_id"].tolist()
    assert ranked_with["siliconpilot_score"].tolist() == ranked_without["siliconpilot_score"].tolist()


def test_constraint_filtering_and_ranking():
    print("Testing multi-objective ranking and constraint penalties...")
    opt = CandidateOptimizer(min_dc_gain=60.0, min_phase_margin=50.0)
    df = pd.DataFrame([
        {"candidate_id": "c1", "dc_gain": 65.0, "phase_margin": 60.0, "area": 1000.0, "area_evidence_origin": "parameter_proxy", "electrical_evidence_origin": "simulated", "drc_clean_prob": 0.9, "perturbation_displacement": 0.5},
        {"candidate_id": "c2", "dc_gain": 45.0, "phase_margin": 60.0, "area": 800.0, "area_evidence_origin": "parameter_proxy", "electrical_evidence_origin": "simulated", "drc_clean_prob": 0.9, "perturbation_displacement": 0.5}, # fails gain
        {"candidate_id": "c3", "dc_gain": 70.0, "phase_margin": 35.0, "area": 1200.0, "area_evidence_origin": "parameter_proxy", "electrical_evidence_origin": "simulated", "drc_clean_prob": 0.9, "perturbation_displacement": 0.5}, # fails PM
    ])
    ranked = opt.evaluate_and_rank_candidates(df)

    # c1 must be valid, c2 and c3 invalid
    assert ranked.loc[ranked["candidate_id"] == "c1", "is_valid_candidate"].iloc[0] == True
    assert ranked.loc[ranked["candidate_id"] == "c2", "is_valid_candidate"].iloc[0] == False
    assert ranked.loc[ranked["candidate_id"] == "c3", "is_valid_candidate"].iloc[0] == False

    # Valid candidate c1 must rank highest
    assert ranked.iloc[0]["candidate_id"] == "c1"
    print("✅ test_constraint_filtering_and_ranking passed")


def test_synthetic_electrical_estimates_do_not_affect_candidate_ranking():
    opt = CandidateOptimizer(min_dc_gain=60.0)
    base = pd.DataFrame([
        {
            "candidate_id": "small_area",
            "area": 100.0,
            "area_evidence_origin": "parameter_proxy",
            "perturbation_displacement": 0.5,
            "dc_gain": 10.0,
            "phase_margin": 10.0,
            "gain_bandwidth_product": 1.0,
            "electrical_evidence_origin": "synthetic",
        },
        {
            "candidate_id": "large_area",
            "area": 200.0,
            "area_evidence_origin": "parameter_proxy",
            "perturbation_displacement": 0.1,
            "dc_gain": 90.0,
            "phase_margin": 90.0,
            "gain_bandwidth_product": 100.0,
            "electrical_evidence_origin": "synthetic",
        },
    ])
    changed_estimates = base.copy()
    changed_estimates[["dc_gain", "phase_margin", "gain_bandwidth_product"]] = [
        [1000.0, 1000.0, 1000.0],
        [0.0, 0.0, 0.0],
    ]
    kwargs = {
        "weights": {"dc_gain": 1.0, "area": -1.0, "perturbation_displacement": 0.0},
        "custom_constraints": {"min_dc_gain": 100.0, "max_area": 500.0},
    }
    ranked_original = opt.evaluate_and_rank_candidates(base, **kwargs)
    ranked_changed = opt.evaluate_and_rank_candidates(changed_estimates, **kwargs)

    assert ranked_original["candidate_id"].tolist() == ranked_changed["candidate_id"].tolist()
    assert ranked_original["siliconpilot_score"].tolist() == ranked_changed["siliconpilot_score"].tolist()
    assert ranked_original["pareto_rank"].tolist() == ranked_changed["pareto_rank"].tolist()
    assert not ranked_original["electrical_metrics_used_for_ranking"].any()
    assert ranked_original["is_valid_candidate"].all()
    assert ranked_original["ranking_basis"].str.contains("area (parameter_proxy)", regex=False).all()
    assert ranked_original["constraint_status"].str.contains("Electrical constraints not evaluated").all()


def test_missing_or_unapproved_provenance_excludes_area_and_electrical_metrics():
    opt = CandidateOptimizer()
    df = pd.DataFrame([
        {"candidate_id": "a", "area": 10.0, "area_evidence_origin": "benchmark",
         "dc_gain": 10.0, "electrical_evidence_origin": "unknown", "perturbation_displacement": 0.5},
        {"candidate_id": "b", "area": 1000.0, "area_evidence_origin": "benchmark",
         "dc_gain": 1000.0, "electrical_evidence_origin": "synthetic", "perturbation_displacement": 0.5},
    ])
    ranked = opt.evaluate_and_rank_candidates(df, weights={"area": -1.0, "dc_gain": 1.0})

    assert ranked["ranking_basis"].eq("parameter-derived displacement").all()
    assert not ranked["electrical_metrics_used_for_ranking"].any()
    assert ranked["pareto_rank"].eq(1).all()
    assert ranked["siliconpilot_score"].eq(ranked["siliconpilot_score"].iloc[0]).all()


def test_measured_or_simulated_evidence_can_be_used_explicitly():
    opt = CandidateOptimizer(min_dc_gain=50.0)
    df = pd.DataFrame([
        {"candidate_id": "simulated_high", "dc_gain": 80.0, "electrical_evidence_origin": "simulated",
         "area": 100.0, "area_evidence_origin": "parameter_proxy"},
        {"candidate_id": "simulated_low", "dc_gain": 20.0, "electrical_evidence_origin": "simulated",
         "area": 100.0, "area_evidence_origin": "parameter_proxy"},
    ])
    ranked = opt.evaluate_and_rank_candidates(df, weights={"dc_gain": 1.0})

    assert ranked.iloc[0]["candidate_id"] == "simulated_high"
    assert ranked["electrical_metrics_used_for_ranking"].all()
    assert ranked["ranking_basis"].str.contains("electrical metrics (simulated)", regex=False).all()


def test_optimization_explainability():
    print("Testing optimization explainability report generator...")
    baseline = pd.Series({
        "candidate_id": "cand_baseline",
        "transistor_width_W": 20.0,
        "transistor_length_L": 0.5,
        "finger_count": 4,
        "area": 2000.0,
        "dc_gain": 58.0,
        "phase_margin": 55.0,
    })
    recommended = pd.Series({
        "candidate_id": "cand_opt_01",
        "transistor_width_W": 28.0,
        "transistor_length_L": 0.48,
        "finger_count": 8,
        "area": 1500.0,
        "dc_gain": 64.5,
        "dc_gain_uncertainty_sigma": 2.1,
        "phase_margin": 60.0,
        "phase_margin_uncertainty_sigma": 3.4,
        "siliconpilot_score": 78.5,
        "pareto_rank": 1,
        "is_pareto_optimal": True,
        "constraint_status": "Satisfied",
    })

    rep = CandidateOptimizer.generate_detailed_explanation(recommended, baseline)
    assert rep["candidate_id"] == "cand_opt_01"
    assert rep["is_pareto_optimal"] == True
    assert "transistor_width_W" in rep["parameter_shifts"]
    assert rep["parameter_shifts"]["transistor_width_W"]["delta"] == 8.0
    assert "dc_gain" in rep["predicted_metrics"]
    assert rep["predicted_metrics"]["dc_gain"]["uncertainty_sigma"] == 2.1
    assert len(rep["tradeoff_rationale"]) >= 3
    print("✅ test_optimization_explainability passed")


def test_layout_visualizers():
    print("Testing 2D layout geometry visualizers...")
    cand = {
        "transistor_width_W": 30.0,
        "transistor_length_L": 0.45,
        "finger_count": 6,
        "multiplicity": 2,
        "orientation": 0,
        "x_position": 50.0,
        "y_position": 50.0,
        "delta_x": 1.2,
        "delta_y": -0.8,
        "area": 1200.0,
    }
    fig1 = render_transistor_layout(cand, title="Test Layout")
    assert fig1 is not None
    assert len(fig1.layout.shapes) >= 4  # BBox, Well, Diffusion, Fingers

    fig2 = render_layout_comparison(cand, cand)
    assert fig2 is not None

    df_test = pd.DataFrame([
        {"candidate_id": "c1", "area": 100, "dc_gain": 60, "is_pareto_optimal": True},
        {"candidate_id": "c2", "area": 200, "dc_gain": 50, "is_pareto_optimal": False},
    ])
    fig3 = render_pareto_tradeoff_chart(df_test, "area", "dc_gain")
    assert fig3 is not None
    print("✅ test_layout_visualizers passed")


if __name__ == "__main__":
    print("\n--- Running Phase 3 Advanced Optimizer Test Suite ---")
    test_parameter_bounds_and_ood()
    test_pareto_dominance_correctness()
    test_candidate_generation_and_uncertainty()
    test_model_trust_provenance_policy()
    test_drc_columns_do_not_affect_ranking()
    test_constraint_filtering_and_ranking()
    test_optimization_explainability()
    test_layout_visualizers()
    print("\n🎉 ALL PHASE 3 ADVANCED OPTIMIZER TESTS PASSED!\n")

