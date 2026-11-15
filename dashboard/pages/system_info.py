"""System configuration and real backend status for this Streamlit process."""
from __future__ import annotations

import platform
import sys

import pandas as pd
import streamlit as st

from config.settings import DATA_DIR, RAW_DIR
from dashboard.components.header import render_header
from src.assistant.vision import resolve_configured_provider_configs
from src.models.model_registry import ModelRegistry
from src.simulation.ngspice_runner import get_shared_ngspice_runner


def _secrets() -> dict:
    try:
        section = st.secrets.get("vision_assistant", {})
        return {"vision_assistant": dict(section)} if hasattr(section, "items") else {}
    except Exception:
        return {}


def render() -> None:
    render_header("System information", "Runtime, data availability, provider configuration, and backend diagnostics")

    cols = st.columns(3)
    cols[0].metric("Python", platform.python_version())
    cols[1].metric("Operating system", f"{platform.system()} {platform.release()}")
    cols[2].metric("Process architecture", platform.machine())
    st.caption(f"Interpreter: `{sys.executable}`")

    runner = get_shared_ngspice_runner()
    if runner.is_available() and runner.is_session_healthy():
        st.success(f"Simulation backend ready in this process: {runner.get_backend_info()}")
    else:
        st.error(f"Simulation backend unavailable or unhealthy: {runner.get_backend_info()}")
        diagnostic = getattr(runner, "get_backend_status", lambda: {})().get("diagnostic")
        if diagnostic:
            st.code(diagnostic)

    providers = resolve_configured_provider_configs(secrets=_secrets())
    st.subheader("Vision provider configuration")
    provider_rows = []
    for name, title in (("gemini", "Google Gemini"), ("openai", "OpenAI")):
        config = providers.get(name)
        provider_rows.append({
            "Provider": title,
            "Configuration": "Model and credential present" if config else "Not configured",
            "Model": config.model if config else "Unavailable",
            "Request verification": "Not yet verified" if config else "Unavailable",
        })
    st.dataframe(pd.DataFrame(provider_rows), width="stretch", hide_index=True)
    st.caption("Credentials are never displayed. A configured provider is not API-validated until a request succeeds.")

    registry = ModelRegistry()
    model_rows = []
    for name, info in registry.list_models().items():
        status = registry.get_model_status(name)
        model_rows.append({
            "Model": name,
            "Trust": status.get("trust_label", "unapproved"),
            "Data origin": status.get("data_origin", "unknown"),
            "Training data": status.get("training_data", "Unknown"),
            "Evaluation limitations": "; ".join(status.get("evaluation_limitations", [])),
        })
    st.subheader("Model registry and provenance")
    if model_rows:
        st.dataframe(pd.DataFrame(model_rows), width="stretch", hide_index=True)
    else:
        st.info("No model artifacts are registered.")
    st.caption("Saved evaluation reports are historical records; their presence does not make restricted or exploratory models trusted decision signals.")

    storage_rows = []
    for name, path in (
        ("ICLR-ADAPT index", RAW_DIR / "iclr_adapt" / "index.csv"),
        ("OSIRIS variants archive", RAW_DIR / "osiris" / "variants.parquet"),
        ("OSIRIS generated sample", DATA_DIR / "sample" / "osiris_sample.csv"),
    ):
        storage_rows.append({
            "Dataset asset": name,
            "Availability": "Cached locally" if path.is_file() else "Not cached",
            "Bytes": path.stat().st_size if path.is_file() else None,
        })
    st.subheader("Dataset cache")
    st.dataframe(pd.DataFrame(storage_rows), width="stretch", hide_index=True)
    st.info(
        "The ngspice built-in templates use empirical 180 nm device models, not a foundry-qualified PDK. "
        "Synthetic electrical estimates, benchmark-derived PEX estimates, and visual AI assessments remain distinct from actual simulation and physical verification."
    )
