"""
Page 4: Layout Prediction Studio — SiliconPilot AI
"""
import streamlit as st
import pandas as pd
import numpy as np
from dashboard.components.header import render_header
from dashboard.components.layout_visualizer import render_transistor_layout
from src.models.model_registry import ModelRegistry


@st.cache_resource
def load_prediction_bundle():
    reg = ModelRegistry()
    perf_bundle = reg.load_model_for_permitted_use(
        "osiris_electrical_performance_predictor", "exploratory_candidate_generation"
    )
    perf_status = reg.get_model_status("osiris_electrical_performance_predictor")
    return perf_bundle, perf_status


def render():
    render_header(
        "Layout Prediction Studio",
        "Exploratory electrical estimates from synthetic training data; no physical verification is performed",
    )

    perf_bundle, perf_status = load_prediction_bundle()

    if perf_bundle is None:
        st.error("Electrical model is unavailable. No fallback prediction is provided.")
        return

    st.markdown("### 🎛️ Transistor Sizing & Placement Design Space")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Device Dimensions**")
        width = st.slider("Transistor Width W (µm)", min_value=0.5, max_value=50.0, value=25.0, step=0.5)
        length = st.slider("Transistor Length L (µm)", min_value=0.18, max_value=2.0, value=0.50, step=0.02)
        topology = st.selectbox("Circuit Topology", ["miller_2stage", "folded_cascode", "telescopic"])

    with col2:
        st.markdown("**Fingering & Multiplicity**")
        fingers = st.slider("Finger Count", min_value=1, max_value=16, value=4, step=1)
        multiplicity = st.slider("Multiplicity (M)", min_value=1, max_value=8, value=2, step=1)
        arrangement = st.selectbox("Finger Arrangement Style", [1, 2, 3, 4], index=0)

    with col3:
        st.markdown("**Placement & Perturbation**")
        x_pos = st.slider("Placement X (µm)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        y_pos = st.slider("Placement Y (µm)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        delta_x = st.slider("Perturbation ΔX (µm)", min_value=-5.0, max_value=5.0, value=0.5, step=0.1)
        delta_y = st.slider("Perturbation ΔY (µm)", min_value=-5.0, max_value=5.0, value=-0.2, step=0.1)

    orientation = st.selectbox("Transistor Orientation", [0, 90, 180, 270], index=0)

    # Compute live approximate area
    approx_area = round(width * length * fingers * multiplicity * 2.5, 1)

    # Assemble candidate dictionary for layout geometry preview
    candidate_preview = {
        "transistor_width_W": width,
        "transistor_length_L": length,
        "finger_count": fingers,
        "multiplicity": multiplicity,
        "finger_arrangement": arrangement,
        "x_position": x_pos,
        "y_position": y_pos,
        "orientation": orientation,
        "delta_x": delta_x,
        "delta_y": delta_y,
        "area": approx_area,
    }

    # Assemble input DataFrame matching trained feature names
    input_dict = {
        "transistor_width_W": [width],
        "transistor_length_L": [length],
        "finger_count": [fingers],
        "multiplicity": [multiplicity],
        "finger_arrangement": [arrangement],
        "x_position": [x_pos],
        "y_position": [y_pos],
        "orientation": [orientation],
        "delta_x": [delta_x],
        "delta_y": [delta_y],
        "perturbation_id": [0],
        "top_miller_2stage": [1.0 if topology == "miller_2stage" else 0.0],
        "top_telescopic": [1.0 if topology == "telescopic" else 0.0],
    }
    input_df = pd.DataFrame(input_dict)

    # Ensure all expected columns exist
    expected_cols = perf_bundle["feature_names"]
    for c in expected_cols:
        if c not in input_df.columns:
            input_df[c] = 0.0
    input_df = input_df[expected_cols]

    st.markdown("---")

    col_pred, col_geom = st.columns([1, 1])

    with col_pred:
        st.subheader("Exploratory electrical model estimates (not simulated or measured)")
        st.warning(f"Trust: {perf_status['trust_label']} · Origin: {perf_status['data_origin']}. {perf_status['reason']}")
        st.caption("± values show tree-to-tree ensemble spread. They are not calibrated prediction uncertainty or confidence intervals.")
        with st.expander("Training provenance and evaluation limitations"):
            st.write(perf_status["training_data"])
            for limitation in perf_status["evaluation_limitations"]:
                st.markdown(f"- {limitation}")
        preprocessor = perf_bundle["preprocessor"]
        model = perf_bundle["model"]
        target_names = perf_bundle["target_names"]

        # Run inference with ensemble uncertainty
        X_scaled = preprocessor.transform(input_df)

        if hasattr(model, "estimators_"):
            all_tree_preds = np.array([tree.predict(X_scaled) for tree in model.estimators_])
            preds = all_tree_preds.mean(axis=0)[0]
            sigmas = all_tree_preds.std(axis=0)[0]
        else:
            preds = model.predict(X_scaled)[0]
            sigmas = np.zeros_like(preds)

        pred_dict = dict(zip(target_names, preds))
        sigma_dict = dict(zip(target_names, sigmas))

        # Metrics display with confidence interval
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.metric(
                "DC Gain",
                f"{pred_dict['dc_gain']:.2f} dB" if "dc_gain" in pred_dict else "Unavailable",
                f"± {sigma_dict['dc_gain']:.2f} dB" if "dc_gain" in sigma_dict else "Uncertainty unavailable",
            )
            st.metric(
                "Phase Margin",
                f"{pred_dict['phase_margin']:.1f}°" if "phase_margin" in pred_dict else "Unavailable",
                f"± {sigma_dict['phase_margin']:.1f}°" if "phase_margin" in sigma_dict else "Uncertainty unavailable",
            )
            st.metric(
                "CMRR",
                f"{pred_dict['cmrr']:.1f} dB" if "cmrr" in pred_dict else "Unavailable",
                f"± {sigma_dict['cmrr']:.1f} dB" if "cmrr" in sigma_dict else "Uncertainty unavailable",
            )

        with m_c2:
            st.metric(
                "Bandwidth (GBW)",
                f"{pred_dict['gain_bandwidth_product']/1e6:.2f} MHz" if "gain_bandwidth_product" in pred_dict else "Unavailable",
                f"± {sigma_dict['gain_bandwidth_product']/1e6:.2f} MHz" if "gain_bandwidth_product" in sigma_dict else "Uncertainty unavailable",
            )
            st.metric(
                "Slew Rate",
                f"{pred_dict['slew_rate']:.2f} V/µs" if "slew_rate" in pred_dict else "Unavailable",
                f"± {sigma_dict['slew_rate']:.2f} V/µs" if "slew_rate" in sigma_dict else "Uncertainty unavailable",
            )
            st.metric(
                "Illustrative formula-based area proxy",
                f"{approx_area:.1f} µm²",
                f"Aspect: {width/length:.1f}",
            )

        st.markdown("### Physical verification")
        st.info("DRC status: **Not verified.** No actual DRC execution result is available. The synthetic-label DRC prediction model is quarantined.")

    with col_geom:
        st.subheader("Parameter-based 2D schematic preview")
        st.caption("This drawing is generated from entered parameters; it is not a physical layout or geometry-model prediction.")
        fig_layout = render_transistor_layout(
            candidate_preview,
            title=f"Floorplan Geometry (W={width}µ, L={length}µ, Nf={fingers})",
            canvas_size=460,
        )
        st.plotly_chart(fig_layout, width="stretch")
