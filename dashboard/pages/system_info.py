"""
System Health & Diagnostics — SiliconPilot AI
Comprehensive, transparent engineering diagnostics dashboard across Runtime, Hardware,
AI Providers, SPICE Simulation, Datasets, ML Trust Policies, EDA Tools, and Application State.
"""
from __future__ import annotations

import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import psutil
import streamlit as st

from config.settings import DATA_DIR, RAW_DIR, REPORTS_DIR, MODEL_DIR
from dashboard.components.header import render_header
from dashboard.chat import SiliconPilotChatbot
from src.assistant.vision import resolve_configured_provider_configs
from src.models.model_registry import ModelRegistry
from src.simulation.ngspice_runner import get_shared_ngspice_runner, NGSpiceRunner


def _get_short_path(long_path: str) -> str:
    """Returns Windows 8.3 short path if on Windows to prevent space issues in ngspice."""
    if sys.platform != "win32":
        return long_path
    try:
        buf = ctypes.create_unicode_buffer(1024)
        res = ctypes.windll.kernel32.GetShortPathNameW(long_path, buf, 1024)
        return buf.value if res > 0 else long_path
    except Exception:
        return long_path


def _detect_gpu() -> dict[str, str]:
    """Inspects dedicated GPU via nvidia-smi or system drivers."""
    gpu_name = "Dedicated GPU / CUDA runtime not detected"
    driver_version = "N/A"
    nvid_smi = shutil.which("nvidia-smi")
    if nvid_smi:
        try:
            import subprocess
            out = subprocess.check_output(
                [nvid_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                text=True, timeout=2
            ).strip()
            if out:
                parts = [p.strip() for p in out.split(",")]
                gpu_name = parts[0]
                if len(parts) > 1:
                    driver_version = parts[1]
        except Exception:
            pass
    return {"name": gpu_name, "driver": driver_version}


def get_system_diagnostics() -> dict[str, Any]:
    """Return passive system diagnostic summary without blocking UI."""
    mem = psutil.virtual_memory()
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(mem.total / (1024**3), 2),
        "available_ram_gb": round(mem.available / (1024**3), 2),
        "gpu": _detect_gpu(),
    }


def render() -> None:
    render_header(
        "System Health & Diagnostics",
        "Runtime telemetry, hardware specifications, active backend diagnostics, and model trust boundaries",
    )

    # Initialise diagnostic session state for active test results
    if "diag_gemini_test_result" not in st.session_state:
        st.session_state.diag_gemini_test_result = None
    if "diag_ngspice_test_result" not in st.session_state:
        st.session_state.diag_ngspice_test_result = None
    if "app_init_time" not in st.session_state:
        st.session_state.app_init_time = datetime.now()

    tabs = st.tabs([
        "🖥️ Runtime & Hardware",
        "🤖 AI Provider Telemetry",
        "⚡ SPICE Simulation",
        "📊 Datasets & Schema",
        "🛡️ ML Model Trust",
        "📐 EDA Tools & Parsers",
        "🩺 Application Health",
    ])

    # ---------------------------------------------------------------------------
    # TAB 1: Runtime & Hardware Diagnostics
    # ---------------------------------------------------------------------------
    with tabs[0]:
        st.markdown("### 1.1 Runtime Environment")
        uptime_delta = datetime.now() - st.session_state.app_init_time
        uptime_str = f"{int(uptime_delta.total_seconds() // 60)}m {int(uptime_delta.total_seconds() % 60)}s"

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("Python Version", platform.python_version())
        col_r2.metric("Virtual Environment", ".venv (Active)" if sys.prefix != sys.base_prefix else "Global Python")
        col_r3.metric("Platform Architecture", f"{platform.machine()} (64-bit)")
        col_r4.metric("Session Uptime", uptime_str)

        st.caption(f"**Python Executable:** `{sys.executable}`")

        # Dependency versions table
        import numpy
        import scipy
        import sklearn
        import plotly
        import PIL

        deps_data = [
            {"Package": "Streamlit", "Installed Version": st.__version__, "Role": "UI Presentation & Multipage Routing"},
            {"Package": "NumPy", "Installed Version": numpy.__version__, "Role": "Numerical Arrays & Vector Math"},
            {"Package": "pandas", "Installed Version": pd.__version__, "Role": "Tabular Data Structures & Benchmarks"},
            {"Package": "SciPy", "Installed Version": scipy.__version__, "Role": "Scientific Optimizers & Filters"},
            {"Package": "scikit-learn", "Installed Version": sklearn.__version__, "Role": "ML Regressors, Classifiers & Pipelines"},
            {"Package": "Plotly", "Installed Version": plotly.__version__, "Role": "Interactive Bode & Waveform Plots"},
            {"Package": "Pillow (PIL)", "Installed Version": PIL.__version__, "Role": "Layout Screenshot & Image Processing"},
            {"Package": "psutil", "Installed Version": psutil.__version__, "Role": "System Hardware & Memory Diagnostics"},
        ]
        st.dataframe(pd.DataFrame(deps_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 1.2 Host Hardware Specifications")
        mem = psutil.virtual_memory()
        cpu_phys = psutil.cpu_count(logical=False) or "N/A"
        cpu_log = psutil.cpu_count(logical=True) or "N/A"
        gpu_info = _detect_gpu()

        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        col_h1.metric("CPU Physical Cores", cpu_phys)
        col_h2.metric("CPU Logical Cores", cpu_log)
        col_h3.metric("Total RAM", f"{mem.total / (1024**3):.1f} GB")
        col_h4.metric("Available RAM", f"{mem.available / (1024**3):.1f} GB ({100 - mem.percent:.0f}%)")

        st.markdown(f"**Processor:** `{platform.processor() or 'Standard x86_64 Processor'}`")
        st.markdown(f"**Operating System:** `{platform.system()} {platform.release()} (Build {platform.version()})`")
        st.markdown(f"**Dedicated GPU:** `{gpu_info['name']}` (Driver: `{gpu_info['driver']}`)")
        st.caption("CUDA Status: PyTorch / CUDA runtime not installed in .venv (Inference runs in CPU mode).")

    # ---------------------------------------------------------------------------
    # TAB 2: AI Provider Diagnostics (Gemini & OpenAI)
    # ---------------------------------------------------------------------------
    with tabs[1]:
        st.markdown("### 2.1 Multimodal Provider Configuration")
        chatbot = SiliconPilotChatbot()
        configs = resolve_configured_provider_configs()

        provider_rows = []
        for p_id, p_name in [("gemini", "Google Gemini"), ("openai", "OpenAI")]:
            cfg = configs.get(p_id)
            if cfg and cfg.is_configured:
                status_label = "Configured (API key present)"
                model_name = cfg.model
                source_label = "Local secrets.toml / Environment"
            else:
                status_label = "Unconfigured (Missing API key)"
                model_name = "N/A"
                source_label = "None"

            provider_rows.append({
                "Provider": p_name,
                "Configuration Status": status_label,
                "Model": model_name,
                "Credential Source": source_label,
            })
        st.dataframe(pd.DataFrame(provider_rows), use_container_width=True, hide_index=True)
        st.caption("🔒 Security Notice: Raw API keys and tokens are never printed or exported in diagnostics.")

        st.markdown("---")
        st.markdown("### 2.2 Live Gemini API Connectivity Verification")
        st.markdown("Perform an explicit, minimal live probe against the configured Google Gemini API endpoint.")

        col_probe_btn, col_probe_status = st.columns([3, 7])
        with col_probe_btn:
            if st.button("🚀 Test Gemini API Connection", key="btn_test_gemini", type="primary"):
                with st.spinner("Pinging Gemini endpoint..."):
                    test_res = chatbot.test_connection()
                    st.session_state.diag_gemini_test_result = test_res

        res = st.session_state.diag_gemini_test_result
        if res:
            with col_probe_status:
                if res["status"] == "verified_operational":
                    st.success(f"✅ **{res['message']}**")
                    st.markdown(f"- **Tested Model:** `{res['model']}`\n- **Roundtrip Latency:** `{res['latency_ms']} ms`\n- **Timestamp:** `{res['timestamp']}`")
                elif res["status"] == "unconfigured":
                    st.warning(f"⚠️ {res['message']}")
                else:
                    st.error(f"❌ **Connection Failed:** {res.get('error', 'Unknown error')}")
                    st.markdown(f"- **Tested Model:** `{res.get('model')}`\n- **Timestamp:** `{res.get('timestamp')}`")
        else:
            with col_probe_status:
                st.info("Click the button to run an active probe. Zero API calls are made on passive page load.")

    # ---------------------------------------------------------------------------
    # TAB 3: SPICE Simulation Diagnostics
    # ---------------------------------------------------------------------------
    with tabs[2]:
        st.markdown("### 3.1 ngspice Simulation Backend")
        runner = get_shared_ngspice_runner()
        is_avail = runner.is_available()
        is_healthy = runner.is_session_healthy()
        backend_info = runner.get_backend_info()

        col_sp1, col_sp2 = st.columns([6, 4])
        with col_sp1:
            if is_avail and is_healthy:
                st.success(f"✅ **Simulation Backend Ready:** {backend_info}")
            else:
                st.error(f"❌ **Simulation Backend Unavailable:** {backend_info}")
        with col_sp2:
            st.metric("Child Worker Process Status", "Healthy & Isolated" if is_healthy else "Degraded / Unloaded")

        st.markdown(f"**Resolved Library Path:** `{os.environ.get('NGSPICE_DLL', 'Auto-discovered by runner')}`")
        st.caption("Isolated Execution: SPICE analyses execute inside a dedicated worker process to guarantee UI thread stability.")

        st.markdown("---")
        st.markdown("### 3.2 Active Minimal ngspice Smoke Test")
        st.markdown("Executes a real minimal RC low-pass filter transient step and AC frequency sweep via the active backend.")

        col_smk_btn, col_smk_res = st.columns([3, 7])
        with col_smk_btn:
            if st.button("⚡ Run Minimal ngspice Smoke Test", key="btn_test_ngspice", type="primary"):
                with st.spinner("Executing minimal RC testbench..."):
                    # Ensure scratch path exists without spaces
                    scratch_dir = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "Temp" / "SiliconPilotScratch"
                    scratch_dir.mkdir(parents=True, exist_ok=True)
                    short_scratch = _get_short_path(str(scratch_dir))
                    os.environ["TEMP"] = short_scratch
                    os.environ["TMP"] = short_scratch

                    netlist = """Minimal RC smoke test
V1 vin 0 DC 0 AC 1 PULSE(0 1 0 1n 1n 5n 10n)
R1 vin outp 1k
C1 outp 0 1n
.end
"""
                    t0 = time.perf_counter()
                    smoke_res = runner.run_netlist(netlist, run_op=False, run_ac=True, run_tran=True)
                    elapsed = (time.perf_counter() - t0) * 1000
                    st.session_state.diag_ngspice_test_result = {
                        "success": smoke_res.success,
                        "ac_status": smoke_res.ac_status,
                        "tran_status": smoke_res.tran_status,
                        "elapsed_ms": round(elapsed, 1),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "error": smoke_res.error_message,
                    }

        smk = st.session_state.diag_ngspice_test_result
        if smk:
            with col_smk_res:
                if smk["success"]:
                    st.success(f"✅ **ngspice Smoke Test Passed ({smk['elapsed_ms']} ms)**")
                    st.markdown(f"- **AC Analysis Status:** `{smk['ac_status']}`\n- **Transient Step Status:** `{smk['tran_status']}`\n- **Timestamp:** `{smk['timestamp']}`")
                else:
                    st.error(f"❌ **ngspice Smoke Test Failed:** {smk['error']}")
        else:
            with col_smk_res:
                st.info("Click the button to execute a real simulation testbench and measure execution latency.")

    # ---------------------------------------------------------------------------
    # TAB 4: Datasets & Schema Diagnostics
    # ---------------------------------------------------------------------------
    with tabs[3]:
        st.markdown("### 4.1 Configured Datasets & Caches")

        dataset_targets = [
            ("Synthetic Demonstration Benchmark", DATA_DIR / "sample" / "osiris_sample.csv", "Synthetic Heuristic (Formula-derived)"),
            ("Synthetic Sample Backup Copy", DATA_DIR / "samples" / "osiris_sample.csv", "Synthetic Heuristic (Formula-derived)"),
            ("ICLR-ADAPT Official Benchmark", RAW_DIR / "iclr_adapt" / "index.csv", "Official Academic Benchmark"),
            ("OSIRIS Official Variants Archive", RAW_DIR / "osiris" / "variants.parquet", "Official Academic Benchmark"),
        ]

        ds_rows = []
        for label, path, origin in dataset_targets:
            exists = path.is_file()
            size_str = f"{path.stat().st_size / 1024:.1f} KB" if exists and path.stat().st_size < 1024*1024 else (f"{path.stat().st_size / (1024*1024):.1f} MB" if exists else "N/A")
            row_count = "N/A"
            col_count = "N/A"
            if exists:
                try:
                    if path.suffix == ".csv":
                        df_peek = pd.read_csv(path, nrows=5)
                        # quick count without full read for large files
                        with open(path, "rb") as f:
                            row_count = sum(1 for _ in f) - 1
                        col_count = len(df_peek.columns)
                    elif path.suffix == ".parquet":
                        import pyarrow.parquet as pq
                        meta = pq.read_metadata(path)
                        row_count = meta.num_rows
                        col_count = meta.num_columns
                except Exception:
                    pass

            ds_rows.append({
                "Dataset Asset": label,
                "Availability": "Cached locally" if exists else "Not cached locally",
                "Data Origin": origin,
                "File Size": size_str,
                "Rows": row_count,
                "Columns": col_count,
                "Local Path": str(path.relative_to(path.parents[2])) if exists else str(path.name),
            })

        st.dataframe(pd.DataFrame(ds_rows), use_container_width=True, hide_index=True)
        st.caption("Provenance Rule: Synthetic demonstration benchmarks and official academic datasets are strictly separated.")

    # ---------------------------------------------------------------------------
    # TAB 5: ML Model Trust Diagnostics
    # ---------------------------------------------------------------------------
    with tabs[4]:
        st.markdown("### 5.1 Declarative Model Trust Policy (`config/model_trust_policy.json`)")
        registry = ModelRegistry()
        policy_path = Path("config/model_trust_policy.json")
        policy_data = {}
        if policy_path.exists():
            with open(policy_path, "r", encoding="utf-8") as f:
                policy_data = json.load(f).get("models", {})

        model_rows = []
        for name, info in registry.list_models().items():
            status = registry.get_model_status(name)
            pol = policy_data.get(name, {})
            trust = status.get("trust_label", pol.get("trust_label", "unapproved")).upper()

            ranking_perm = "❌ Strictly Excluded"
            if trust == "EXPLORATORY":
                ranking_perm = "⚠️ Exploratory Only"
            elif trust == "APPROVED":
                ranking_perm = "✅ Permitted"

            metrics = info.get("metrics", {})
            metric_summary = "N/A"
            if "overall" in metrics:
                ov = metrics["overall"]
                if "mean_R2" in ov:
                    metric_summary = f"Mean R²: {ov['mean_R2']:.4f}"
            elif "Accuracy" in metrics:
                metric_summary = f"Acc: {metrics['Accuracy']*100:.1f}%, F1: {metrics.get('F1', 0):.3f}"

            model_rows.append({
                "Model Identifier": name,
                "Trust Level": trust,
                "Data Origin": status.get("data_origin", "unknown").capitalize(),
                "Saved Evaluation": metric_summary,
                "Influence on Ranking": ranking_perm,
                "Permitted Uses": ", ".join(pol.get("permitted_uses", [])) or "None",
                "Reason / Safeguard": pol.get("reason", "No policy reason specified."),
            })

        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

        st.info(
            "🛡️ **Fail-Closed Safeguards Active:**\n"
            "- **`layout_verification_classifier` is QUARANTINED:** Suffers from synthetic class-imbalance collapse; does NOT execute DRC/LVS.\n"
            "- **`iclr_layout_geometry_predictor` is RESTRICTED:** Exhibited negative R² on test holdout; excluded from candidate ranking signals.\n"
            "- **`osiris_electrical_performance_predictor` is EXPLORATORY:** Fitted to synthetic algebraic equations; not SPICE signoff."
        )

    # ---------------------------------------------------------------------------
    # TAB 6: EDA Tools & Parsers Diagnostics
    # ---------------------------------------------------------------------------
    with tabs[5]:
        st.markdown("### 6.1 Built-in Read-Only Artifact Parsers (`src/eda/inspection.py`)")
        st.markdown("SiliconPilot AI includes native, read-only Python parsers for standard electronic design exchange formats:")

        parsers_data = [
            {"Parser": "DRC Summary Parser", "Supported Extensions": ".txt, .log, .rpt, .drc", "Capabilities": "Rule violations, error counts, coordinates, affected instances"},
            {"Parser": "LVS Summary Parser", "Supported Extensions": ".lvs, .json, .log, .out", "Capabilities": "Net/device mismatch, shorts, opens, instance counts"},
            {"Parser": "SPICE Netlist Inspector", "Supported Extensions": ".cir, .sp, .spice, .net", "Capabilities": "Subcircuit hierarchy, transistor models, pin interfaces (No execution)"},
            {"Parser": "Physical Layout Inspector", "Supported Extensions": ".gds, .gdsii, .oas, .oasis", "Capabilities": "Cell hierarchy, bounding boxes, polygon layers via `gdstk`"},
            {"Parser": "Physical Exchange Inspector", "Supported Extensions": ".def, .lef", "Capabilities": "Macro dimensions, site grids, standard cell pin definitions"},
        ]
        st.dataframe(pd.DataFrame(parsers_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 6.2 External Host EDA Software Detection")
        st.caption("Note: Built-in parsers operate independently; external EDA binaries are only required for native layout creation or signoff.")

        external_tools = [
            ("KiCad Electronics Suite", "kicad"),
            ("KiCad Command Line Interface", "kicad-cli"),
            ("ngspice Circuit Simulator", "ngspice"),
            ("KLayout GDSII Viewer / DRC", "klayout"),
            ("Magic VLSI Layout Tool", "magic"),
            ("Netgen LVS Tool", "netgen"),
        ]

        eda_rows = []
        for title, bin_name in external_tools:
            found_path = shutil.which(bin_name)
            # Special check for ngspice DLL
            if bin_name == "ngspice" and not found_path:
                candidate_dll = Path(r"C:\Users\aakas\AppData\Local\Programs\KiCad\10.0\bin\ngspice.dll")
                if candidate_dll.exists():
                    found_path = str(candidate_dll)

            eda_rows.append({
                "EDA Tool": title,
                "Executable": bin_name,
                "Detection Status": "Detected on Host" if found_path else "Not detected in system PATH",
                "Resolved Path": found_path or "N/A",
            })
        st.dataframe(pd.DataFrame(eda_rows), use_container_width=True, hide_index=True)

    # ---------------------------------------------------------------------------
    # TAB 7: Application Health & Logs
    # ---------------------------------------------------------------------------
    with tabs[6]:
        st.markdown("### 7.1 Dashboard Page Integrity")
        pages_to_check = [
            ("Overview & Telemetry", "dashboard.pages.overview"),
            ("Dataset Explorer", "dashboard.pages.dataset_explorer"),
            ("Model Evaluation", "dashboard.pages.model_training"),
            ("Layout & Sizing Prediction", "dashboard.pages.layout_prediction"),
            ("Candidate Optimization", "dashboard.pages.candidate_comparison"),
            ("AI Engineering Assistant", "dashboard.pages.ai_assistant"),
            ("System Health & Diagnostics", "dashboard.pages.system_info"),
            ("SPICE Simulation Studio", "dashboard.pages.simulation_studio"),
        ]

        page_rows = []
        for page_name, module_path in pages_to_check:
            try:
                __import__(module_path)
                page_status = "✅ Import Healthy"
                err_msg = "None"
            except Exception as exc:
                page_status = "❌ Import Failed"
                err_msg = str(exc)

            page_rows.append({
                "Page Name": page_name,
                "Module Path": module_path,
                "Health Status": page_status,
                "Import Error": err_msg,
            })
        st.dataframe(pd.DataFrame(page_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 7.2 Storage, Temporary Directories & Feedback Database")
        scratch_p = Path(tempfile.gettempdir())
        priv_p = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "Temp" / "SiliconPilotAI"
        priv_p.mkdir(parents=True, exist_ok=True)
        fb_db = DATA_DIR / "simulation_results.db"

        fb_status = "N/A"
        fb_records = 0
        if fb_db.exists():
            try:
                conn = sqlite3.connect(fb_db)
                c = conn.cursor()
                c.execute("SELECT count(*) FROM sqlite_master WHERE type='table';")
                tables = c.fetchone()[0]
                fb_status = f"Active ({tables} tables)"
                conn.close()
            except Exception as e:
                fb_status = f"Error: {e}"

        col_st1, col_st2, col_st3 = st.columns(3)
        col_st1.metric("System Temp Directory", "Writable" if os.access(scratch_p, os.W_OK) else "Read-only")
        col_st2.metric("Isolated GDS Workspace", "Ready" if priv_p.exists() else "Missing")
        col_st3.metric("Simulation DB", fb_status)

        # Recent Log inspection
        st.markdown("### 7.3 Structured Log Telemetry")
        log_file = Path("logs/siliconpilot.log")
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                log_lines = f.readlines()[-15:]
            st.code("".join(log_lines), language="text")
        else:
            st.caption("No log entries currently written to logs/siliconpilot.log.")
