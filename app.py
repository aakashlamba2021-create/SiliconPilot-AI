"""SiliconPilot AI multi-page engineering workbench."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="SiliconPilot AI",
    page_icon=":material/memory:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.pages import (
    ai_assistant,
    candidate_comparison,
    dataset_explorer,
    layout_prediction,
    model_training,
    overview,
    simulation_studio,
    system_info,
)

with st.sidebar:
    st.markdown("### SiliconPilot AI")
    st.caption("Analog IC engineering workbench")

page = st.navigation(
    {
        "Assistant": [
            st.Page(
                ai_assistant.render,
                title="AI Engineering Assistant",
                url_path="assistant",
                icon=":material/psychology_alt:",
                help="Conversational Analog IC copilot with multi-image layout inspection, DRC/LVS guidance, and EDA artifact import.",
                default=True,
            ),
        ],
        "Workbench": [
            st.Page(
                overview.render,
                title="Project Overview",
                url_path="overview",
                icon=":material/dashboard:",
                help="Operational status, available datasets, model trust, and direct access to engineering workflows.",
            ),
            st.Page(
                dataset_explorer.render,
                title="Dataset Explorer",
                url_path="datasets",
                icon=":material/database:",
                help="Inspect the ICLR-ADAPT and OSIRIS benchmark datasets, their statistics, and schema quality.",
            ),
            st.Page(
                layout_prediction.render,
                title="Layout Prediction",
                url_path="prediction",
                icon=":material/tune:",
                help="Explore parameter-based geometry previews and clearly labelled exploratory model estimates.",
            ),
            st.Page(
                candidate_comparison.render,
                title="Candidate Comparison",
                url_path="candidates",
                icon=":material/compare_arrows:",
                help="Generate and compare exploratory design candidates with their assumptions and validation gaps.",
            ),
            st.Page(
                model_training.render,
                title="Model Evaluation & Trust",
                url_path="models",
                icon=":material/monitoring:",
                help="Review saved model evaluations together with data origin, trust status, and limitations.",
            ),
        ],
        "Verification & Diagnostics": [
            st.Page(
                simulation_studio.render,
                title="Electrical Simulation",
                url_path="simulation",
                icon=":material/electrical_services:",
                help="Run explicit ngspice AC and transient analyses using the available empirical device models.",
            ),
            st.Page(
                system_info.render,
                title="System Health & Diagnostics",
                url_path="system",
                icon=":material/health_and_safety:",
                help="Check runtime health, live Gemini API and ngspice connectivity probes, model trust categories, and diagnostic telemetry.",
            ),
        ],
    },
    position="sidebar",
)

page.run()
