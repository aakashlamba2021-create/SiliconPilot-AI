"""
Page 3: Model Training & Evaluation — SiliconPilot AI
"""
import streamlit as st
import pandas as pd
from dashboard.components.header import render_header
from dashboard.components.plots import plot_metric_bars
from src.models.model_registry import ModelRegistry


def render():
    render_header("Model Training & Evaluation", "Model metrics with training-data provenance and trust limitations")

    registry = ModelRegistry()
    models = registry.list_models()

    if not models:
        st.warning("No model metadata is currently registered.")
        return

    st.subheader("📋 Registered Models & Validation Performance")
    
    # Display model cards
    for model_name, info in models.items():
        status = registry.get_model_status(model_name)
        with st.expander(f"📦 Model: {model_name}", expanded=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"**Dataset Source:** `{info.get('dataset_name', 'N/A')}`")
                st.markdown(f"**Description:** {info.get('description', '')}")
                st.markdown(f"**Saved Version:** `{info.get('version_tag', '')}`")
                st.markdown(f"**Trust:** `{status['trust_label']}` · **Data origin:** `{status['data_origin']}`")
                st.markdown(f"**Training data:** {status['training_data']}")
                st.markdown(f"**Use status:** {status['reason']}")
                for limitation in status["evaluation_limitations"]:
                    st.markdown(f"- {limitation}")
                st.markdown(f"**Features ({len(info.get('feature_names', []))}):** `{', '.join(info.get('feature_names', [])[:8])}...`")
            
            with col2:
                metrics = info.get("metrics", {})
                if "per_target" in metrics:
                    st.markdown("**Saved manifest test metrics (historical; not rerun on this page):**")
                    table_df = pd.DataFrame.from_dict(metrics["per_target"], orient="index")
                    st.dataframe(table_df.round(4), width="stretch")
                    st.plotly_chart(plot_metric_bars(metrics["per_target"], title=f"{model_name} saved test R²"), width="stretch")
                elif "Accuracy" in metrics:
                    st.markdown("**Saved synthetic-label test metrics (not DRC verification):**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", f"{metrics['Accuracy']*100:.2f}%")
                    c2.metric("Precision", f"{metrics['Precision']*100:.2f}%")
                    c3.metric("Recall", f"{metrics['Recall']*100:.2f}%")
                    c4.metric("F1 Score", f"{metrics['F1']:.4f}")

    st.markdown("---")
    st.subheader("🔍 Important Engineering Findings on Generalization")
    st.info(
        """
        - Electrical QoS targets are synthetic equation-derived labels. These metrics measure agreement with held-out synthetic rows, not ngspice or silicon.
        - The DRC classifier is quarantined: its labels are synthetic and its saved test predictions were all DRC-clean. No DRC verification was performed.
        - The geometry model has negative R² on all saved test targets and is not approved for engineering ranking.
        - OSIRIS area/PEX estimates are benchmark-derived estimates, not measured area or parasitic extraction.
        """
    )

