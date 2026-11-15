"""
Page 8: Simulation & Electrical Validation Studio — SiliconPilot AI
SPICE-in-the-Loop physical electrical simulation, multi-candidate verification,
and feedback learning database interface powered by ngspice.
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.components.header import render_header
from src.simulation.ngspice_runner import get_shared_ngspice_runner
from src.simulation.netlist_manager import NetlistManager
from src.simulation.simulation_parser import SimulationParser
from src.simulation.simulation_metrics import SimulationMetricsExtractor
from src.simulation.simulation_validator import SimulationValidator
from src.simulation.feedback_db import FeedbackDatabase
from src.optimization.candidate_optimizer import CandidateOptimizer
from src.data.osiris_loader import OsirisLoader
from src.models.model_registry import ModelRegistry


@st.cache_resource
def get_simulation_engine():
    runner = get_shared_ngspice_runner()
    net_mgr = NetlistManager()
    parser = SimulationParser()
    extractor = SimulationMetricsExtractor()
    validator = SimulationValidator()
    feedback_db = FeedbackDatabase()
    optimizer = CandidateOptimizer()
    return runner, net_mgr, parser, extractor, validator, feedback_db, optimizer


@st.cache_resource
def get_ml_models():
    reg = ModelRegistry()
    perf = reg.load_model_for_permitted_use(
        "osiris_electrical_performance_predictor", "exploratory_candidate_generation"
    )
    status = reg.get_model_status("osiris_electrical_performance_predictor")
    return perf, status


def plot_bode_diagram(ac_data, title="SPICE Open-Loop Frequency Response (Bode Plot)"):
    """Creates a dual-panel interactive Plotly Bode plot (Magnitude dB & Phase)."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Magnitude Response |A_v|(f) [dB]", "Phase Response ∠A_v(f) [Degrees]")
    )

    freqs = ac_data.frequencies
    gains = ac_data.gain_db
    phases = ac_data.phase_deg

    # Panel 1: Gain (dB)
    fig.add_trace(
        go.Scatter(
            x=freqs, y=gains,
            mode="lines",
            name="Gain (dB)",
            line=dict(color="#38BDF8", width=2.5),
            hovertemplate="Frequency: %{x:.2e} Hz<br>Gain: %{y:.2f} dB<extra></extra>",
        ),
        row=1, col=1
    )
    # 0 dB Unity Gain Line
    fig.add_hline(y=0.0, line_dash="dash", line_color="#EF4444", line_width=1.5, annotation_text="0 dB (Unity Gain)", row=1, col=1)

    # Panel 2: Phase (degrees)
    fig.add_trace(
        go.Scatter(
            x=freqs, y=phases,
            mode="lines",
            name="Phase (deg)",
            line=dict(color="#F59E0B", width=2.5),
            hovertemplate="Frequency: %{x:.2e} Hz<br>Phase: %{y:.1f}°<extra></extra>",
        ),
        row=2, col=1
    )

    fig.update_xaxes(type="log", title_text="Frequency [Hz]", row=2, col=1)
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_yaxes(title_text="Gain [dB]", row=1, col=1)
    fig.update_yaxes(title_text="Phase [°]", row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        height=520,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_transient_step(tran_data, title="SPICE Closed-Loop Step Response & Slew Rate"):
    """Creates an interactive Plotly transient step response chart."""
    fig = go.Figure()

    t_ns = tran_data.time_s * 1e9  # Convert to nanoseconds

    fig.add_trace(
        go.Scatter(
            x=t_ns, y=tran_data.vin_v,
            mode="lines",
            name="Input V_in(t)",
            line=dict(color="#94A3B8", width=1.5, dash="dot"),
            hovertemplate="Time: %{x:.2f} ns<br>V_in: %{y:.3f} V<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_ns, y=tran_data.vout_v,
            mode="lines",
            name="Output V_out(t)",
            line=dict(color="#22C55E", width=2.5),
            hovertemplate="Time: %{x:.2f} ns<br>V_out: %{y:.3f} V<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        xaxis_title="Time [ns]",
        yaxis_title="Voltage [V]",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render():
    render_header(
        "SPICE Electrical Simulation & Validation Studio",
        "ngspice benchmark simulation; not silicon measurement, DRC/LVS signoff, or extracted parasitics",
    )

    runner, net_mgr, parser, extractor, validator, feedback_db, optimizer = get_simulation_engine()
    perf_bundle, perf_status = get_ml_models()
    st.caption("The built-in device models are empirical 180 nm models, not a foundry-qualified PDK. Entered Rout/Cout are assumptions, not extracted parasitics.")
    st.info("DRC status: Not verified. ngspice electrical simulation does not perform DRC or LVS.")
    with st.expander("Electrical model provenance and limitations"):
        st.warning(f"Trust: {perf_status['trust_label']} · Origin: {perf_status['data_origin']}. Estimates are not SPICE validated.")
        st.write(perf_status["training_data"])
        for limitation in perf_status["evaluation_limitations"]:
            st.markdown(f"- {limitation}")

    # Backend status badge
    backend_status = runner.get_backend_status()
    if backend_status["status"] == "healthy" and backend_status["available_for_extraction"]:
        st.success(f"Active simulation backend: {runner.get_backend_info()} (DLL loaded and session healthy)")
    elif backend_status["status"] == "cli_limited":
        st.warning(f"ngspice CLI detected at `{backend_status['path']}`, but this studio requires the shared DLL for vector extraction.")
        st.caption(backend_status["diagnostic"])
    else:
        st.error(f"ngspice backend is not ready: {backend_status['status']}")
        st.caption(backend_status["diagnostic"] or "No diagnostic details are available.")

    tabs = st.tabs([
        "🔬 1. Interactive Circuit Simulation",
        "🎯 2. SPICE-in-the-Loop Top-3 Validation",
        "📊 3. Feedback Learning Database",
        "🧬 4. Physics vs ML Diagnostics (Module 7)",
    ])

    # ------------------------------------------------------------------
    # TAB 1: Single-Circuit Simulation & Waveform Analyzer
    # ------------------------------------------------------------------
    with tabs[0]:
        st.markdown("### 🎛️ Parametric Circuit Configuration & Real-Time SPICE Execution")
        st.caption("Configure transistor sizing, differential pair geometry, and load conditions to execute native ngspice physics simulation.")

        col_top, col_bias = st.columns([1, 1])
        with col_top:
            topology = st.selectbox(
                "Circuit Topology:",
                ["miller_2stage", "five_transistors_ota", "folded_cascode", "telescopic"],
                format_func=lambda x: {
                    "miller_2stage": "Two-Stage Miller Compensated OTA (miller_2stage)",
                    "five_transistors_ota": "Classic 5-Transistor Single-Stage OTA (five_transistors_ota)",
                    "folded_cascode": "Folded Cascode Operational Amplifier (folded_cascode)",
                    "telescopic": "Telescopic Cascode Operational Amplifier (telescopic)",
                }.get(x, x),
            )

        with col_bias:
            bias_current = st.slider("Bias Current I_bias [µA]:", min_value=5.0, max_value=100.0, value=20.0, step=5.0)

        st.markdown("#### Sizing & Load Parameters")
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        with p_c1:
            w_val = st.slider("Transistor Width W [µm]:", 5.0, 100.0, 40.0, 2.5)
        with p_c2:
            l_val = st.slider("Channel Length L [µm]:", 0.18, 2.0, 0.5, 0.05)
        with p_c3:
            m_val = st.slider("Multiplicity M:", 1, 8, 2, 1)
        with p_c4:
            cload_val = st.slider("Load Cap C_L [pF]:", 0.5, 10.0, 1.5, 0.5)

        with st.expander("🛠️ Advanced Settings & Parasitics (PEX Injection)"):
            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                c_pex = st.number_input("Output Parasitic Cap C_pex [pF]:", min_value=0.0, max_value=2.0, value=0.0, step=0.05)
            with exp_c2:
                r_pex = st.number_input("Output Parasitic Res R_pex [Ω]:", min_value=0.0, max_value=500.0, value=0.0, step=10.0)

        pex_dict = {}
        if c_pex > 0:
            pex_dict["c_pex_out"] = c_pex
        if r_pex > 0:
            pex_dict["r_pex_out"] = r_pex

        params = {
            "transistor_width_W": w_val,
            "transistor_length_L": l_val,
            "finger_count": 8,
            "multiplicity": m_val,
            "ibias_ua": bias_current,
            "cload_pf": cload_val,
            "vdd": 1.8,
        }

        if st.button("🚀 Run SPICE Simulation", type="primary", width="stretch"):
            with st.spinner("Synthesizing SPICE netlists and executing ngspice AC & Transient solvers..."):
                # AC Simulation
                ac_net = net_mgr.generate_netlist(topology, params, mode="ac", pex_parasitics=pex_dict if pex_dict else None)
                res_ac = runner.run_netlist(ac_net, run_ac=True, run_tran=False, run_op=True, supply_source="VDD")
                ac_data = parser.parse_ac_file(res_ac.ac_file)

                # Transient Simulation
                tran_net = net_mgr.generate_netlist(topology, params, mode="tran", pex_parasitics=pex_dict if pex_dict else None)
                res_tran = runner.run_netlist(tran_net, run_ac=False, run_tran=True, run_op=False, supply_source="VDD")
                tran_data = parser.parse_tran_file(res_tran.tran_file)

                # Metric Extraction
                metrics = extractor.extract(res_ac, ac_data, tran_data, vdd=1.8, sim_result_tran=res_tran, supply_source="VDD")

            total_time = res_ac.execution_time_sec + res_tran.execution_time_sec
            if metrics.convergence_status == "Converged":
                st.success(f"Simulation completed successfully in {total_time:.3f} seconds.")
            else:
                st.warning(f"Simulation completed with status: {metrics.convergence_status} (in {total_time:.3f}s)")

            # Metrics Cards with safe handling for unavailable/None metrics
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("DC Gain", f"{metrics.dc_gain_db:.1f} dB" if metrics.dc_gain_db is not None else "Unavailable")
            m2.metric("Unity-Gain BW", f"{metrics.gain_bandwidth_product_hz/1e6:.1f} MHz" if metrics.gain_bandwidth_product_hz is not None else "Unavailable")
            m3.metric("Phase Margin", f"{metrics.phase_margin_deg:.1f}°" if metrics.phase_margin_deg is not None else "Unavailable")
            m4.metric("Slew Rate", f"{metrics.slew_rate_v_per_us:.1f} V/µs" if metrics.slew_rate_v_per_us is not None else "Unavailable")
            m5.metric("Power Dissipation", f"{metrics.power_consumption_mw:.3f} mW" if metrics.power_consumption_mw is not None else "Unavailable")
            m6.metric("Settling Time", f"{metrics.settling_time_ns:.1f} ns" if metrics.settling_time_ns is not None else "Unavailable")

            # Waveform Visualizations
            bode_col, tran_col = st.columns([3, 2])
            with bode_col:
                if ac_data.is_valid:
                    st.plotly_chart(plot_bode_diagram(ac_data), width="stretch")
            with tran_col:
                if tran_data.is_valid:
                    st.plotly_chart(plot_transient_step(tran_data), width="stretch")

            with st.expander("📄 View Generated SPICE Netlist & Engine Console Output"):
                st.code(ac_net, language="spice")

    # ------------------------------------------------------------------
    # TAB 2: SPICE-in-the-Loop Top-3 Candidate Validation
    # ------------------------------------------------------------------
    with tabs[1]:
        st.markdown("### 🏆 Closed-Loop Pareto Candidate Validation (Top-3)")
        st.caption(
            "Workflow: generate exploratory candidates → rank estimates → simulate selected candidates with ngspice. "
            "This benchmark simulation is not silicon measurement, DRC/LVS, or foundry signoff."
        )

        val_col1, val_col2 = st.columns([1, 1])
        with val_col1:
            val_topology = st.selectbox(
                "Target Circuit Family:",
                ["miller_2stage", "five_transistors_ota", "folded_cascode", "telescopic"],
                key="val_top_select"
            )
        with val_col2:
            num_cands_to_rank = st.slider("Initial Pareto Candidate Pool Size:", 12, 40, 24, 4)

        if st.button("⚡ Execute SPICE-in-the-Loop Optimization", type="primary", width="stretch", disabled=perf_bundle is None):
            with st.spinner("Generating candidates, predicting ML proxies, and solving SPICE physics..."):
                # Baseline
                baseline = {
                    "transistor_width_W": 30.0,
                    "transistor_length_L": 0.5,
                    "finger_count": 8,
                    "multiplicity": 2,
                    "finger_arrangement": 1,
                    "x_position": 50.0,
                    "y_position": 50.0,
                    "orientation": 0,
                    "delta_x": 0.0,
                    "delta_y": 0.0,
                    "area": 4200.0,
                    "dc_gain": 52.0,
                    "gain_bandwidth_product": 35.0e6,
                    "phase_margin": 55.0,
                    "slew_rate": 30.0,
                    "power_consumption": 0.00015,
                }

                # 1. Generate & Predict
                cands_df = optimizer.generate_candidates(
                    baseline_params=baseline,
                    num_candidates=num_cands_to_rank,
                    perturbation_scale=0.20,
                    perf_bundle=perf_bundle,
                )

                # 2. Pareto Ranking
                ranked_df = optimizer.rank_candidates(cands_df)

                # 3. SPICE-in-the-loop on Top 3
                sim_val_results = optimizer.run_simulation_validation(
                    ranked_candidates=ranked_df,
                    top_k=3,
                    topology=val_topology,
                    record_in_db=True,
                )

            st.session_state["last_sim_validation"] = sim_val_results

        if "last_sim_validation" in st.session_state:
            res = st.session_state["last_sim_validation"]

            st.markdown("---")
            # Winner Banner
            winner_id = res["winner_id"]
            if winner_id:
                st.markdown(
                    f"""
                    <div style="background-color: #064E3B; border: 1px solid #059669; padding: 18px; border-radius: 10px; margin-bottom: 20px;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span style="font-size: 28px;">🏅</span>
                            <div>
                                <h3 style="margin: 0; color: #34D399; font-size: 1.3rem;">Top Candidate for This ngspice Simulation: {winner_id}</h3>
                                <p style="margin: 4px 0 0 0; color: #D1FAE5; font-size: 0.95rem;">{res['winner_justification']}</p>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.warning(f"⚠️ No candidates satisfied verification requirements: {res.get('winner_justification', 'None converged')}")

            # Comparison Table
            st.markdown("#### 📋 SPICE vs ML Predictions — Top 3 Candidates")
            st.dataframe(res["comparison_dataframe"], width="stretch")

            # Detailed Candidate Deep-Dive
            st.markdown("#### 🔍 Individual Candidate Validation Reports")
            for cand_summary in res["evaluated_candidates"]:
                c_id = cand_summary["candidate_id"]
                err_str = f"{cand_summary['mean_rel_error_pct']:.1f}%" if cand_summary.get('mean_rel_error_pct') is not None else "N/A"
                with st.expander(f"📌 {c_id} Detailed Breakdown (Verdict: {cand_summary['verdict']}, Mean Error: {err_str})"):
                    report = res["validation_reports"][c_id]
                    st.dataframe(report.to_dataframe(), width="stretch")
                    st.info(f"**Engineering Summary:** {report.engineering_summary}")

    # ------------------------------------------------------------------
    # TAB 3: Feedback Learning Database
    # ------------------------------------------------------------------
    with tabs[2]:
        st.markdown("### 🗄️ Simulation Results & Feedback Learning Archive")
        st.caption("Central SQLite database storing verified SPICE simulation observations for controlled ML retraining.")

        history_df = feedback_db.get_all_records()

        col_count, col_exp = st.columns([3, 1])
        with col_count:
            st.metric("Total Simulated Records Logged", len(history_df))
        with col_exp:
            csv_data = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Database CSV",
                data=csv_data,
                file_name="siliconpilot_spice_simulation_records.csv",
                mime="text/csv",
                width="stretch",
            )

        if len(history_df) > 0:
            st.dataframe(
                history_df[["record_id", "circuit_id", "topology", "timestamp", "simulation_status", "mean_relative_error", "verified_for_training"]],
                width="stretch",
            )
        else:
            st.info("No simulation runs recorded yet. Execute simulations in Tab 1 or Tab 2 to populate the archive.")

    # ------------------------------------------------------------------
    # TAB 4: Physics vs ML Diagnostics (Module 7)
    # ------------------------------------------------------------------
    with tabs[3]:
        st.markdown("### 🔬 Module 7: Physics vs ML Empirical Diagnostics")
        st.caption("Rigorous scientific investigation into model prediction limitations, non-linear pole-zero doublets, and feature engineering.")

        st.markdown(
            r"""
            #### 1. Why Phase Margin R² (0.4503) and CMRR R² (0.2003) Diverge
            - **Non-Linear Doublet Migration:** Phase margin is dictated by high-frequency non-dominant poles ($p_2 = g_{m2}/C_L$) and right-half-plane zeros ($z_1 = g_{m6}/C_c$). As transistor dimensions shift, pole-zero doublets migrate rapidly across decades of frequency. Tree ensembles with tabular geometric features perform step-wise approximations that smooth out sharp multi-pole phase transitions.
            - **Common-Mode Rejection (CMRR):** CMRR depends on infinitesimal transistor threshold voltage and transconductance mismatches ($\Delta V_{th}, \Delta \beta$) between the differential pair halves. Sizing parameters alone without Monte Carlo mismatch parameters lack the physical degrees of freedom needed to predict CMRR with high R².
            - **Engineering Takeaway:** **ML models serve as efficient initial filters (pruning 90% of poor candidates), but physical SPICE simulation-in-the-loop is strictly necessary to guarantee electrical stability and true gain.**

            #### 2. Strict Scientific Integrity Rules
            - **ML Estimates** are labeled as *Statistical Tree-Ensemble Approximations*.
            - **SPICE Simulations** are labeled as *ngspice Physical Small-Signal & Transient Vector Truths*.
            - **Physical Layout Area** is a purely 2D floorplan geometric metric and is never claimed to be calculated by SPICE.
            - **PEX Parasitics** are injected into simulation netlists only when explicit parasitic RC networks are present.
            """
        )
