"""Dataset explorer with source and evidence limitations kept explicit."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.components.header import render_header
from dashboard.components.plots import plot_correlation_heatmap
from src.data.iclr_loader import ICLRAdaptLoader
from src.data.osiris_loader import OsirisLoader
from src.data.schema_inspector import SchemaInspector

logger = logging.getLogger(__name__)


@st.cache_data(ttl=900)
def load_iclr_data() -> pd.DataFrame:
    return ICLRAdaptLoader().load()


@st.cache_data(ttl=900)
def load_osiris_sample() -> pd.DataFrame:
    return OsirisLoader().load(source="sample")


@st.cache_data(ttl=900)
def load_osiris_parquet_meta() -> pd.DataFrame:
    return OsirisLoader().load(source="parquet", nrows=1000)


def _load_or_report(loader, dataset_name: str) -> pd.DataFrame | None:
    try:
        with st.spinner(f"Loading {dataset_name}…"):
            frame = loader()
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            st.warning(f"{dataset_name} is unavailable or contains no rows.")
            return None
        return frame
    except Exception as exc:
        logger.exception("Could not load dataset %s", dataset_name)
        st.error(f"Could not load {dataset_name}. Check its local cache and network access, then retry.")
        st.caption(f"Load error: {type(exc).__name__}")
        return None


def _numeric(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    subset = frame[columns] if columns is not None else frame
    return subset.select_dtypes(include=[np.number]).apply(pd.to_numeric, errors="coerce")


def render() -> None:
    render_header("Dataset explorer", "Inspect benchmark data and generated samples with their provenance visible")

    dataset_choice = st.selectbox(
        "Dataset",
        [
            "ICLR-ADAPT benchmark",
            "OSIRIS generated electrical sample",
            "OSIRIS benchmark variants archive",
        ],
        key="dataset_explorer_choice",
    )

    if dataset_choice == "ICLR-ADAPT benchmark":
        df = _load_or_report(load_iclr_data, "ICLR-ADAPT")
        if df is None:
            return
        st.caption(
            "Origin: ICLR-ADAPT benchmark. These are recorded benchmark observations and labels; "
            "SiliconPilot has not run DRC/LVS on a user's layout."
        )
        numeric_l = pd.to_numeric(df["L"], errors="coerce") if "L" in df else pd.Series(dtype=float)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows loaded", f"{len(df):,}")
        c2.metric("Circuits", f"{df['circuit'].nunique():,}" if "circuit" in df else "Unavailable")
        c3.metric("PDKs", f"{df['pdk'].nunique():,}" if "pdk" in df else "Unavailable")
        c4.metric(
            "Recorded `L` label",
            f"{int(numeric_l.sum()):,} / {int(numeric_l.notna().sum()):,}" if numeric_l.notna().any() else "Unavailable",
        )

        tab_data, tab_corr, tab_schema = st.tabs(["Data table", "Feature correlations", "Schema and quality"])
        with tab_data:
            preferred = ["circuit", "pdk", "Aeval", "Leq_signal", "Leq_power", "Ncut_intermetal", "G", "C", "L", "DRC_total"]
            visible = [column for column in preferred if column in df.columns]
            st.dataframe(df[visible].head(100), width="stretch", hide_index=True)
        with tab_corr:
            candidates = ["Aeval", "Leq_signal", "Leq_power", "Ncut_intermetal", "DRC_total", "elapsed_s"]
            numeric = _numeric(df, [column for column in candidates if column in df.columns])
            st.plotly_chart(plot_correlation_heatmap(numeric, title="ICLR-ADAPT benchmark correlations"), width="stretch")
        with tab_schema:
            report = SchemaInspector(df, name="ICLR-ADAPT").run()
            st.metric("Schema quality score", f"{report['data_quality_score']:.1f} / 100")
            st.json(report)

    elif dataset_choice == "OSIRIS generated electrical sample":
        df = _load_or_report(load_osiris_sample, "OSIRIS generated sample")
        if df is None:
            return
        st.warning(
            "This is locally generated OSIRIS-like sample data. Electrical targets are produced by synthetic equations; "
            "`drc_clean` is a synthetic label and is not DRC verification."
        )
        numeric_drc = pd.to_numeric(df["drc_clean"], errors="coerce") if "drc_clean" in df else pd.Series(dtype=float)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Generated rows", f"{len(df):,}")
        c2.metric("Mean synthetic gain", f"{pd.to_numeric(df['dc_gain'], errors='coerce').mean():.1f} dB" if "dc_gain" in df else "Unavailable")
        c3.metric("Mean synthetic GBW", f"{pd.to_numeric(df['gain_bandwidth_product'], errors='coerce').mean()/1e6:.1f} MHz" if "gain_bandwidth_product" in df else "Unavailable")
        c4.metric(
            "Synthetic `drc_clean=True` share",
            f"{numeric_drc.mean()*100:.1f}%" if numeric_drc.notna().any() else "Unavailable",
        )

        tab_data, tab_corr, tab_stats = st.tabs(["Data table", "Feature correlations", "Statistics"])
        with tab_data:
            st.dataframe(df.head(100), width="stretch", hide_index=True)
        with tab_corr:
            st.plotly_chart(plot_correlation_heatmap(_numeric(df), title="Generated sample correlations"), width="stretch")
        with tab_stats:
            numeric = _numeric(df)
            if numeric.empty:
                st.info("No numeric columns are available for summary statistics.")
            else:
                st.dataframe(numeric.describe().round(4).T, width="stretch")

    else:
        df = _load_or_report(load_osiris_parquet_meta, "OSIRIS benchmark archive preview")
        if df is None:
            return
        st.caption(
            "Origin: OSIRIS benchmark Parquet archive. This preview is limited to the rows loaded locally; "
            "benchmark PEX targets are not extraction results from a user's layout."
        )
        c1, c2 = st.columns(2)
        c1.metric("Preview rows loaded", f"{len(df):,}")
        topology_column = next((column for column in ("topology", "circuit", "circuit_family") if column in df), None)
        c2.metric("Families in preview", f"{df[topology_column].nunique():,}" if topology_column else "Unavailable")
        st.dataframe(df.head(100), width="stretch", hide_index=True)
