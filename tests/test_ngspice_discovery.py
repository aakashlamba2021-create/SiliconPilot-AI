from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.simulation.ngspice_runner import DEFAULT_DLL_SEARCH_PATHS, NGSpiceRunner, _pe_architecture


def _kicad10_user_dll() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        pytest.skip("LOCALAPPDATA is unavailable on this machine.")
    return Path(local_app_data) / "Programs" / "KiCad" / "10.0" / "bin" / "ngspice.dll"


def test_kicad10_user_install_is_a_supported_discovery_candidate():
    dll = _kicad10_user_dll()
    assert str(dll).lower() in {str(path).lower() for path in DEFAULT_DLL_SEARCH_PATHS if path}
    if dll.exists():
        assert _pe_architecture(str(dll)) in {"AMD64", None}


def test_real_kicad10_dll_loads_and_reports_actual_backend_state():
    dll = Path(os.environ.get("NGSPICE_DLL", str(_kicad10_user_dll())))
    if not dll.is_file():
        pytest.skip("KiCad 10 ngspice.dll is not installed on this machine.")
    runner = NGSpiceRunner(dll_path=str(dll))
    assert runner.is_available_for_extraction(), runner.get_backend_info()
    status = runner.get_backend_status()
    assert status["status"] == "healthy", status
    assert status["path"] == str(dll)
    assert status["session_healthy"] is True
