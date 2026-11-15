"""
Page 5: Candidate Comparison & Advanced Multi-Objective Optimizer — SiliconPilot AI
"""
import json
import streamlit as st
import pandas as pd
import numpy as np

from dashboard.components.header import render_header
from dashboard.components.layout_visualizer import (
    render_transistor_layout,
    render_layout_comparison,
    render_pareto_tradeoff_chart,
)
from src.optimization.candidate_optimizer import CandidateOptimizer, SUPPORTED_PARAMETER_BOUNDS
from src.data.osiris_loader import OsirisLoader
from src.models.model_registry import ModelRegistry


@st.cache_data
def load_candidate_pool():
    loader = OsirisLoader()
    df = loader.load(source="sample")
    df["candidate_id"] = [f"sample_{i:03d}" for i in range(len(df))]
    df = df.head(40).copy()
    df["area"] = (
        df["transistor_width_W"]
        * df["transistor_length_L"]
        * df["finger_count"]
        * df["multiplicity"]
        * 2.5
    ).round(1)
    df["area_evidence_origin"] = "parameter_proxy"
    df["area_result_type"] = "Formula-based parameter proxy; not extracted physical area"
    df["electrical_evidence_origin"] = "synthetic"
    df["electrical_result_type"] = "Synthetic sample data; not simulated or measured"
    return df


@st.cache_resource
def load_models_for_optimizer():
    reg = ModelRegistry()
    perf = reg.load_model_for_permitted_use(
        "osiris_electrical_performance_predictor", "exploratory_candidate_generation"
    )
    status = reg.get_model_status("osiris_electrical_performance_predictor")
    return perf, status


def render():
    render_header(
        "Layout Candidate Comparator & Pareto Optimizer",
        "Multi-Objective Optimization, 2D Floorplan Visualization & Design Explainability",
    )

    perf_bundle, perf_status = load_models_for_optimizer()
    st.warning(
        f"Exploratory candidate ranking only. Electrical model: {perf_status['trust_label']} / "
        f"{perf_status['data_origin']}. Synthetic or unverified electrical values are excluded "
        "from constraints, Pareto membership, and candidate scores."
    )

    # Sidebar / Tab configuration for generation source
    st.subheader("🎯 1. Candidate Generation & Baseline Configuration")
    
    gen_col1, gen_col2 = st.columns([1, 1])
    with gen_col1:
        source_mode = st.radio(
            "Evaluation Source Mode:",
            [
                "Controlled parameter perturbations (show exploratory estimates)",
                "Generated sample-data pool (not benchmark parquet)",
            ],
            index=0,
        )
    
    with gen_col2:
        num_variants = st.slider("Number of Candidates to Generate", min_value=12, max_value=60, value=30, step=6)

    optimizer = CandidateOptimizer()

    if "Controlled parameter perturbations" in source_mode:
        if perf_bundle is None:
            st.error("Electrical model unavailable. No fallback values are supplied.")
            return
        with st.expander("Electrical model provenance and evaluation limitations"):
            st.write(perf_status["training_data"])
            for limitation in perf_status["evaluation_limitations"]:
                st.markdown(f"- {limitation}")

        with st.expander("🛠️ Baseline Sizing & Perturbation Controls", expanded=True):
            bc1, bc2, bc3, bc4 = st.columns(4)
            with bc1:
                base_top = st.selectbox("Baseline Topology", ["miller_2stage", "folded_cascode", "telescopic"], index=0)
                base_w = st.slider("Baseline Width W (µm)", 1.0, 50.0, 25.0, step=0.5)
            with bc2:
                base_l = st.slider("Baseline Length L (µm)", 0.18, 1.80, 0.50, step=0.02)
                base_fc = st.selectbox("Baseline Fingers", [1, 2, 4, 8, 12, 16], index=2)
            with bc3:
                base_m = st.selectbox("Baseline Multiplicity (M)", [1, 2, 4, 8], index=1)
                base_orient = st.selectbox("Baseline Orientation", [0, 90, 180, 270], index=0)
            with bc4:
                perturb_scale = st.slider("Perturbation Scale (Radius)", 0.2, 2.0, 1.0, step=0.2)
                seed_val = st.number_input("Random Generator Seed", min_value=1, max_value=99999, value=42)

            baseline_params = {
                "topology": base_top,
                "transistor_width_W": base_w,
                "transistor_length_L": base_l,
                "finger_count": base_fc,
                "multiplicity": base_m,
                "orientation": base_orient,
                "x_position": 50.0,
                "y_position": 50.0,
            }

            candidates = optimizer.generate_candidates(
                baseline_params=baseline_params,
                perf_bundle=perf_bundle,
                num_candidates=num_variants,
                perturbation_scale=perturb_scale,
                random_seed=seed_val,
            )
            st.info(
                f"Generated {len(candidates)} candidates by parameter perturbation. "
                f"{perf_bundle['model'].__class__.__name__} estimates are display-only and "
                "do not affect candidate ranking."
            )

    else:
        candidates = load_candidate_pool()
        candidates = candidates.drop(columns=["drc_clean", "lvs_verified"], errors="ignore")
        candidates["drc_status"] = "Not verified"
        if "perturbation_displacement" not in candidates.columns:
            candidates["perturbation_displacement"] = np.sqrt(candidates.get("delta_x", 0)**2 + candidates.get("delta_y", 0)**2).round(2)

    # -----------------------------------------------------------------------
    # Multi-Objective Weights & Hard Constraints Panel
    # -----------------------------------------------------------------------
    st.subheader("⚖️ 2. Multi-Objective Priorities & Design Constraints")
    
    with st.expander("⚙️ Configure Objective Weights & Hard Constraints", expanded=False):
        st.markdown("**Parameter-proxy trade-offs (electrical estimates are excluded from ranking):**")
        w_c1, w_c2 = st.columns(2)
        with w_c1:
            w_area = st.slider("Minimize formula-based area proxy", -0.50, 0.0, -0.20, step=0.05)
        with w_c2:
            w_displacement = st.slider("Minimize parameter displacement", -0.50, 0.0, -0.10, step=0.05)

        st.markdown("**Exploratory area-proxy threshold (not a physical signoff limit):**")
        max_area = st.number_input("Maximum formula-based area proxy (µm²)", value=4500.0, step=250.0)

        custom_weights = {
            "area": w_area,
            "perturbation_displacement": w_displacement,
        }
        custom_constraints = {
            "max_area": max_area,
        }

    # Execute Multi-Objective Ranking & Pareto Sorting
    ranked_df = optimizer.evaluate_and_rank_candidates(
        candidates,
        weights=custom_weights if 'custom_weights' in locals() else None,
        custom_constraints=custom_constraints if 'custom_constraints' in locals() else None,
    )

    clean_count = int(ranked_df["is_valid_candidate"].sum())
    pareto_count = int(ranked_df["is_pareto_optimal"].sum())
    best_cand = ranked_df.iloc[0]
    st.caption("Constraint and Pareto results are exploratory rankings of estimates and entered parameters. They are not layout recommendations, DRC/LVS results, or physical signoff.")
    st.caption(
        f"Ranking basis: {ranked_df['ranking_basis'].iloc[0]}. Synthetic electrical estimates remain visible "
        "for context but are excluded from constraints, Pareto sorting, and scores."
    )
    baseline_cand = ranked_df[ranked_df["is_baseline"]].iloc[0] if "is_baseline" in ranked_df.columns and ranked_df["is_baseline"].any() else ranked_df.iloc[-1]

    # Metrics Summary Bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Evaluated Candidates", f"{len(ranked_df)}")
    m2.metric("Pareto Optimal (Rank 1)", f"{pareto_count} Candidates", "Non-Dominated")
    m3.metric("Within exploratory thresholds", f"{clean_count} / {len(ranked_df)}", f"{(clean_count/len(ranked_df))*100:.0f}% of estimates")
    m4.metric("Top Exploratory Score", f"{best_cand['siliconpilot_score']}", f"Rank {best_cand.get('pareto_rank', 1)}")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Interactive Visualizations & Tabs
    # -----------------------------------------------------------------------
    tab_pareto, tab_layout, tab_explain, tab_table = st.tabs([
        "📈 Pareto Trade-Off Landscape",
        "📐 2D Layout Geometry Visualizer",
        "💡 Optimization Explainability Report",
        "📋 Complete Candidates Table",
    ])

    with tab_pareto:
        pc1, pc2 = st.columns([1, 1])
        with pc1:
            x_axis = st.selectbox("X-Axis Parameter Proxy", ["area", "perturbation_displacement"], index=0)
        with pc2:
            y_axis = st.selectbox("Y-Axis Parameter Proxy", ["area", "perturbation_displacement"], index=1)

        st.caption("Pareto membership is computed from these parameter proxies, not from predicted electrical metrics.")
        st.plotly_chart(render_pareto_tradeoff_chart(ranked_df, x_col=x_axis, y_col=y_axis), width="stretch")

    with tab_layout:
        st.markdown("### 📐 Parameter-based Geometry Preview")
        st.caption("Schematic preview from entered parameters; this is not a physical layout or extracted geometry.")
        
        vl_col1, vl_col2 = st.columns(2)
        with vl_col1:
            st.markdown(f"#### 🏷️ Top-ranked exploratory candidate: `{best_cand['candidate_id']}`")
            fig_rec = render_transistor_layout(best_cand, title=f"Parameter-based preview ({best_cand['candidate_id']})")
            st.plotly_chart(fig_rec, width="stretch")

        with vl_col2:
            st.markdown(f"#### 🏷️ Baseline parameters: `{baseline_cand['candidate_id']}`")
            fig_base = render_transistor_layout(baseline_cand, title=f"Baseline Layout ({baseline_cand['candidate_id']})")
            st.plotly_chart(fig_base, width="stretch")

        st.markdown("#### 🔄 Floorplan Area & Displacement Comparison")
        fig_comp = render_layout_comparison(baseline_cand, best_cand)
        st.plotly_chart(fig_comp, width="stretch")

    with tab_explain:
        st.markdown("### 🔍 Optimization Explainability & Delta Audit")
        explanation = optimizer.generate_detailed_explanation(best_cand, baseline_cand)

        e_col1, e_col2 = st.columns([1, 1])
        with e_col1:
            st.markdown("#### 🔄 Parameter Shifts vs Baseline")
            param_shift_rows = []
            for p, pdata in explanation["parameter_shifts"].items():
                param_shift_rows.append({
                    "Parameter": p,
                    "Baseline": pdata["original"],
                    "Proposed": pdata["proposed"],
                    "Delta": f"{pdata['delta']:+.3f}",
                })
            st.dataframe(pd.DataFrame(param_shift_rows), width="stretch")

            st.markdown("#### 🛡️ Constraint & Pareto Status")
            st.info(f"**Exploratory threshold check:** {explanation['constraint_status']}")
            st.info(f"**Pareto position:** Rank {explanation['pareto_rank']} ({'non-dominated within the selected estimated objectives' if explanation['is_pareto_optimal'] else 'dominated within the selected estimated objectives'})")

        with e_col2:
            st.markdown("#### ⚡ Exploratory electrical estimates (display-only)")
            st.caption("Synthetic electrical values are not simulated or measured and are excluded from ranking and constraints.")
            metric_rows = []
            for m, mdata in explanation["predicted_metrics"].items():
                sigma_str = f"± {mdata['uncertainty_sigma']:.2f}" if mdata['uncertainty_sigma'] > 0 else "N/A"
                val_str = f"{mdata['value']:.2f} {mdata['unit']}"
                delta_str = f"{mdata['improvement_pct']:+.1f}%" if mdata['improvement_pct'] is not None else "Baseline"
                metric_rows.append({
                    "Metric": m.replace("_", " ").title(),
                    "Exploratory estimate": val_str,
                    "Ensemble spread (not calibrated)": sigma_str,
                    "Evidence origin": best_cand.get("electrical_evidence_origin", "unknown"),
                    "Used in ranking": bool(best_cand.get("electrical_metrics_used_for_ranking", False)),
                    "Estimate vs baseline": delta_str,
                })
            st.dataframe(pd.DataFrame(metric_rows), width="stretch")

        st.markdown("#### 📜 Exploratory trade-off summary")
        for r_line in explanation["tradeoff_rationale"]:
            st.markdown(f"- {r_line}")

        # Download Report JSON
        st.download_button(
            label="📥 Download Detailed Optimization Report (JSON)",
            data=json.dumps({
                "evidence_basis": "Synthetic-data-trained exploratory estimates and entered parameters; not simulated or measured.",
                "drc_status": "Not verified",
                "ranking_basis": best_cand.get("ranking_basis", "unknown"),
                "electrical_metrics_used_for_ranking": bool(best_cand.get("electrical_metrics_used_for_ranking", False)),
                "analysis": explanation,
            }, indent=2, default=str),
            file_name=f"siliconpilot_optimization_{best_cand['candidate_id']}.json",
            mime="application/json",
        )

    with tab_table:
        st.markdown("### 📋 Complete Ranked Candidates Matrix")
        display_cols = [
            "candidate_id",
            "siliconpilot_score",
            "pareto_rank",
            "is_pareto_optimal",
            "is_valid_candidate",
            "dc_gain",
            "gain_bandwidth_product",
            "phase_margin",
            "slew_rate",
            "area",
            "drc_status",
            "electrical_evidence_origin",
            "electrical_result_type",
            "area_evidence_origin",
            "area_result_type",
            "ranking_basis",
            "electrical_metrics_used_for_ranking",
            "perturbation_displacement",
        ]
        avail = [c for c in display_cols if c in ranked_df.columns]
        st.dataframe(ranked_df[avail], width="stretch")
