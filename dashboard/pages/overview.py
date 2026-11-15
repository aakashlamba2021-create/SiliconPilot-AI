"""Project overview with live local capability and trust status."""
from __future__ import annotations

import sqlite3
from collections import Counter

import pandas as pd
import streamlit as st

from config.settings import DATA_DIR, RAW_DIR
from dashboard.components.header import render_header
from src.assistant.vision import resolve_configured_provider_configs
from src.models.model_registry import ModelRegistry
from src.simulation.ngspice_runner import get_shared_ngspice_runner


def _provider_secrets() -> dict:
    try:
        section = st.secrets.get("vision_assistant", {})
        return {"vision_assistant": dict(section)} if hasattr(section, "items") else {}
    except Exception:
        return {}


@st.cache_data(ttl=60)
def _simulation_record_count() -> int | None:
    path = DATA_DIR / "simulation_results.db"
    if not path.is_file():
        return None
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=1)
        try:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='simulation_history'"
            ).fetchone()
            if not table:
                return 0
            return int(connection.execute("SELECT COUNT(*) FROM simulation_history").fetchone()[0])
        finally:
            connection.close()
    except sqlite3.Error:
        return None


def render() -> None:
    render_header(
        "Engineering workspace",
        "Analog IC layout exploration with explicit data provenance and verification boundaries",
    )

    registry = ModelRegistry()
    models = registry.list_models()
    trust_counts = Counter(registry.get_model_status(name).get("trust_label", "unapproved") for name in models)
    providers = resolve_configured_provider_configs(secrets=_provider_secrets())
    runner = get_shared_ngspice_runner()

    dataset_assets = {
        "ICLR-ADAPT benchmark index": RAW_DIR / "iclr_adapt" / "index.csv",
        "OSIRIS benchmark archive": RAW_DIR / "osiris" / "variants.parquet",
        "OSIRIS generated sample": DATA_DIR / "sample" / "osiris_sample.csv",
    }
    available_assets = sum(path.is_file() for path in dataset_assets.values())
    columns = st.columns(4)
    columns[0].metric("Local data assets", f"{available_assets} / {len(dataset_assets)}")
    columns[1].metric(
        "Registered models",
        str(len(models)),
        f"{trust_counts.get('exploratory', 0)} exploratory · {trust_counts.get('restricted', 0)} restricted · {trust_counts.get('quarantined', 0)} quarantined",
    )
    columns[2].metric("Vision providers configured", f"{len(providers)} / 2")
    backend_ready = runner.is_available() and runner.is_session_healthy()
    columns[3].metric("ngspice backend", "Ready" if backend_ready else "Unavailable")

    st.subheader("Operational status")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Data availability")
        data_rows = [
            {
                "Asset": label,
                "Status": "Cached locally" if path.is_file() else "Not cached",
                "Size": f"{path.stat().st_size / (1024 * 1024):.1f} MB" if path.is_file() else "Unavailable",
            }
            for label, path in dataset_assets.items()
        ]
        st.dataframe(pd.DataFrame(data_rows), width="stretch", hide_index=True)
        st.caption("The OSIRIS sample is generated data; benchmark records and simulation records have separate provenance.")

    with right:
        st.markdown("#### Model trust")
        trust_rows = [
            {"Trust status": label.title(), "Registered models": count}
            for label, count in sorted(trust_counts.items())
        ]
        if trust_rows:
            st.dataframe(pd.DataFrame(trust_rows), width="stretch", hide_index=True)
        else:
            st.info("No model metadata is currently registered.")
        st.caption("Model presence is not model approval. Check the evaluation page for provenance and limitations.")

    status_cols = st.columns(2)
    with status_cols[0]:
        if backend_ready:
            st.success(f"ngspice loaded in this process: {runner.get_backend_info()}")
        else:
            st.warning(f"ngspice is not ready in this process: {runner.get_backend_info()}")
            diagnostic = getattr(runner, "get_backend_status", lambda: {})().get("diagnostic")
            if diagnostic:
                st.caption(diagnostic)
        st.caption("A loaded backend does not mean a simulation has run. Simulation results appear only after an explicit run.")

    with status_cols[1]:
        for provider_id, label in (("gemini", "Gemini"), ("openai", "OpenAI")):
            config = providers.get(provider_id)
            st.write(f"**{label}:** configured · `{config.model}`" if config else f"**{label}:** not configured")
        st.caption("Configuration means a model ID and credential are present. API access is unverified until a request succeeds.")

    record_count = _simulation_record_count()
    st.subheader("Engineering capabilities")
    capability_cols = st.columns(4)
    capability_cols[0].write("**Image analysis**\n\nSession-only uploads; provider calls require explicit consent.")
    capability_cols[1].write("**EDA evidence**\n\nRead-only report, netlist, DEF/LEF, and GDSII/OASIS metadata inspection.")
    capability_cols[2].write("**Electrical simulation**\n\nAC and transient analyses across four empirical circuit templates.")
    capability_cols[3].write(
        "**Recorded simulation activity**\n\n" + (f"{record_count} stored records." if record_count is not None else "Unavailable.")
    )

    st.info(
        "Evidence boundaries: the DRC classifier is quarantined; the geometry model is restricted; the electrical predictor is trained on synthetic equations; "
        "OSIRIS PEX outputs are benchmark-derived estimates. None of these are actual DRC/LVS results, extracted parasitics, or measured silicon."
    )
