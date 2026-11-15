"""
SiliconPilot AI - ngspice Runner Module
Executes electrical simulations using ngspice shared library C API (via ctypes)
or ngspice CLI batch execution.
"""

import os
import sys
import time
import math
import shutil
import ctypes
import tempfile
import uuid
import threading
import multiprocessing as mp
import gc
import platform
import struct
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("src.simulation.ngspice_runner")

# Windows kernel32 helper for safe module loading and unloading
_kernel32 = None
_DLL_DIRECTORY_HANDLES: list[Any] = []
if sys.platform == "win32":
    try:
        _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _kernel32.LoadLibraryW.restype = ctypes.c_void_p
        _kernel32.LoadLibraryW.argtypes = [ctypes.c_wchar_p]
        _kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
        _kernel32.FreeLibrary.restype = ctypes.c_int
    except Exception as e:
        logger.debug(f"Could not initialize kernel32 bindings: {e}")

# Common ngspice library installation paths on Windows / Linux / macOS.
_LOCAL_APP_DATA = os.environ.get("LOCALAPPDATA")
if not _LOCAL_APP_DATA and sys.platform == "win32":
    _LOCAL_APP_DATA = str(Path.home() / "AppData" / "Local")
_KICAD10_USER_DLL = (
    os.path.join(_LOCAL_APP_DATA, "Programs", "KiCad", "10.0", "bin", "ngspice.dll")
    if _LOCAL_APP_DATA else None
)
DEFAULT_DLL_SEARCH_PATHS = [
    os.environ.get("NGSPICE_DLL"),
    _KICAD10_USER_DLL,
    r"C:\Program Files\KiCad\10.0\bin\ngspice.dll",
    r"C:\Program Files\KiCad\9.0\bin\ngspice.dll",
    r"C:\Program Files\KiCad\8.0\bin\ngspice.dll",
    r"C:\Program Files\ngspice\bin\ngspice.dll",
    r"C:\msys64\mingw64\bin\libngspice.dll",
    "/usr/lib/x86_64-linux-gnu/libngspice.so",
    "/usr/local/lib/libngspice.so",
    "/opt/homebrew/lib/libngspice.dylib",
]


def _pe_architecture(path: str) -> str | None:
    """Return a Windows PE machine label when the DLL header is readable."""
    if sys.platform != "win32":
        return None
    try:
        with open(path, "rb") as stream:
            header = stream.read(64)
            if len(header) < 64 or header[:2] != b"MZ":
                return None
            pe_offset = struct.unpack_from("<I", header, 0x3C)[0]
            stream.seek(pe_offset)
            signature_and_machine = stream.read(6)
        if signature_and_machine[:4] != b"PE\0\0" or len(signature_and_machine) < 6:
            return None
        machine = struct.unpack_from("<H", signature_and_machine, 4)[0]
        return {0x8664: "AMD64", 0x14C: "x86", 0xAA64: "ARM64"}.get(machine, f"PE machine 0x{machine:04X}")
    except (OSError, struct.error):
        return None


@dataclass
class SimulationResult:
    """Dataclass holding the outcome of a SPICE simulation run with per-analysis status tracking."""
    success: bool
    ac_file: Optional[str] = None
    tran_file: Optional[str] = None
    op_output: Dict[str, float] = field(default_factory=dict)
    log_messages: List[str] = field(default_factory=list)
    execution_time_sec: float = 0.0
    error_message: Optional[str] = None
    backend_info: str = "ngspice"
    op_status: str = "NOT_REQUESTED"     # "SUCCESS", "FAILED", "NOT_REQUESTED", "ERROR", "UNSUPPORTED"
    ac_status: str = "NOT_REQUESTED"     # "SUCCESS", "FAILED", "NOT_REQUESTED", "ERROR", "UNSUPPORTED"
    tran_status: str = "NOT_REQUESTED"   # "SUCCESS", "FAILED", "NOT_REQUESTED", "ERROR", "UNSUPPORTED"
    analysis_statuses: Dict[str, str] = field(default_factory=dict)
    analysis_errors: Dict[str, str] = field(default_factory=dict)


def _ngspice_worker_loop(conn: Any, dll_path: str) -> None:
    """
    Dedicated worker process loop for process-isolated ngspice simulation.
    Runs in a dedicated child process, loading ngspice.dll into its own clean address space.
    If a simulation crashes or causes an unrecoverable ngspice state, the worker terminates cleanly,
    preventing any memory or static state corruption in the parent process.
    """
    import ctypes
    import os
    import sys
    import time
    import numpy as np

    # Ensure DLL directory is in PATH on Windows for dependent DLLs
    if sys.platform == "win32" and dll_path:
        bin_dir = os.path.dirname(dll_path)
        if bin_dir and bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory") and bin_dir and os.path.exists(bin_dir):
            try:
                os.add_dll_directory(bin_dir)
            except Exception:
                pass

    SendChar = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)
    ExitCb = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_int,
        ctypes.c_void_p,
    )

    logs: List[str] = []

    def on_char(msg: bytes, ident: int, user_data: Any) -> int:
        if msg:
            logs.append(msg.decode("utf-8", errors="ignore"))
        return 0

    def on_exit(status: int, immediate: bool, quitexit: bool, ident: int, user_data: Any) -> int:
        logs.append(
            f"stderr ngspice ControlledExit: status={status}, immediate={immediate}, quitexit={quitexit}, ident={ident}"
        )
        return 0

    cb1 = SendChar(on_char)
    cb2 = ExitCb(on_exit)

    try:
        if sys.platform == "win32":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.LoadLibraryW.restype = ctypes.c_void_p
            kernel32.LoadLibraryW.argtypes = [ctypes.c_wchar_p]
            handle = kernel32.LoadLibraryW(dll_path)
            dll = ctypes.CDLL(dll_path, handle=handle) if handle else ctypes.CDLL(dll_path)
        else:
            dll = ctypes.CDLL(dll_path)

        dll.ngSpice_Init.argtypes = [
            SendChar,
            ctypes.c_void_p,
            ExitCb,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        dll.ngSpice_Init.restype = ctypes.c_int
        dll.ngSpice_Command.argtypes = [ctypes.c_char_p]
        dll.ngSpice_Command.restype = ctypes.c_int
        if hasattr(dll, "ngSpice_Reset"):
            dll.ngSpice_Reset.argtypes = []
            dll.ngSpice_Reset.restype = ctypes.c_int

        r_init = dll.ngSpice_Init(cb1, None, cb2, None, None, None, None)
        if r_init != 0:
            conn.send({"type": "INIT_ERROR", "error": f"ngSpice_Init failed with code {r_init}"})
            return
    except Exception as e:
        conn.send({"type": "INIT_ERROR", "error": str(e)})
        return

    conn.send({"type": "READY", "init_ret": r_init})

    while True:
        try:
            msg = conn.recv()
        except (EOFError, BrokenPipeError):
            break
        if msg == "SHUTDOWN":
            break

        cmd_type = msg.get("type")
        if cmd_type == "RUN_SIM":
            cir_path = msg["cir_path"]
            output_node = msg["output_node"]
            supply_node = msg["supply_node"]
            supply_source = msg["supply_source"]
            ac_out_path = msg.get("ac_out_path")
            tran_out_path = msg.get("tran_out_path")
            run_op = msg.get("run_op", True)
            run_ac = msg.get("run_ac", True)
            run_tran = msg.get("run_tran", True)

            logs.clear()
            norm_cir = cir_path.replace("\\", "/")

            def exec_cmd(c_str: str) -> Tuple[int, List[str]]:
                idx = len(logs)
                r = dll.ngSpice_Command(c_str.encode("utf-8"))
                return r, logs[idx:]

            analysis_statuses = {}
            analysis_errors = {}
            op_status = "NOT_REQUESTED"
            ac_status = "NOT_REQUESTED"
            tran_status = "NOT_REQUESTED"

            # 1. Source netlist
            ret_src, l_src = exec_cmd(f"source {norm_cir}")
            src_err = any("error:" in l.lower() or "cannot find" in l.lower() for l in l_src)
            unrec = any("cannot recover and awaits to be reset or detached" in l.lower() for l in l_src)

            if ret_src != 0 or src_err or unrec:
                err_msg = next(
                    (l.strip() for l in l_src if "error:" in l.lower() or "cannot find" in l.lower()),
                    f"Failed to source netlist (exit code {ret_src})",
                )
                exec_cmd("remcirc")
                exec_cmd("destroy all")
                if run_op:
                    op_status = "FAILED"
                    analysis_errors["op"] = err_msg
                if run_ac:
                    ac_status = "FAILED"
                    analysis_errors["ac"] = err_msg
                if run_tran:
                    tran_status = "FAILED"
                    analysis_errors["tran"] = err_msg
                analysis_statuses.update({"op": op_status, "ac": ac_status, "tran": tran_status})
                conn.send({
                    "success": False,
                    "error_message": err_msg,
                    "op_status": op_status,
                    "ac_status": ac_status,
                    "tran_status": tran_status,
                    "analysis_statuses": analysis_statuses,
                    "analysis_errors": analysis_errors,
                    "op_output": {},
                    "ac_file": None,
                    "tran_file": None,
                    "logs": list(logs),
                    "must_restart": True,
                })
                continue

            # 2. Run solver
            ret_run, l_run = exec_cmd("run")
            run_abort = any(
                "run simulation(s) aborted" in l.lower()
                or "timestep too small" in l.lower()
                or "singular matrix" in l.lower()
                or "error:" in l.lower()
                for l in l_run
            )
            unrec = unrec or any(
                "cannot recover and awaits to be reset or detached" in l.lower() for l in l_run
            )

            if ret_run != 0 or run_abort or unrec:
                err_msg = next(
                    (l.strip() for l in l_run if "error:" in l.lower() or "aborted" in l.lower() or "singular" in l.lower()),
                    f"SPICE solver execution failed (exit code {ret_run})",
                )
                exec_cmd("remcirc")
                exec_cmd("destroy all")
                if run_op:
                    op_status = "FAILED"
                    analysis_errors["op"] = err_msg
                if run_ac:
                    ac_status = "FAILED"
                    analysis_errors["ac"] = err_msg
                if run_tran:
                    tran_status = "FAILED"
                    analysis_errors["tran"] = err_msg
                analysis_statuses.update({"op": op_status, "ac": ac_status, "tran": tran_status})
                conn.send({
                    "success": False,
                    "error_message": f"SPICE Convergence / Model Error: {err_msg}",
                    "op_status": op_status,
                    "ac_status": ac_status,
                    "tran_status": tran_status,
                    "analysis_statuses": analysis_statuses,
                    "analysis_errors": analysis_errors,
                    "op_output": {},
                    "ac_file": None,
                    "tran_file": None,
                    "logs": list(logs),
                    "must_restart": True,
                })
                continue

            # 3. Extract AC
            ac_file = None
            if run_ac and ac_out_path:
                ret_pl, l_pl = exec_cmd("setplot ac1")
                plot_err = any("error:" in l.lower() or "cannot find" in l.lower() for l in l_pl)
                if ret_pl != 0 or plot_err:
                    ac_status = "FAILED"
                    analysis_errors["ac"] = f"Failed to switch to ac1 plot (exit code {ret_pl})"
                else:
                    norm_ac = ac_out_path.replace("\\", "/")
                    t_wr_start = time.time()
                    ret_wr, l_wr = exec_cmd(f"wrdata {norm_ac} vdb({output_node}) vp({output_node})")
                    wr_err = any("error:" in l.lower() for l in l_wr)
                    if (
                        ret_wr != 0
                        or wr_err
                        or not os.path.exists(ac_out_path)
                        or os.path.getsize(ac_out_path) == 0
                        or os.path.getmtime(ac_out_path) < (t_wr_start - 1.0)
                    ):
                        ac_status = "FAILED"
                        analysis_errors["ac"] = "AC output vector file missing, empty, stale, or write failed"
                    else:
                        ac_status = "SUCCESS"
                        ac_file = ac_out_path

            # 4. Extract TRAN
            tran_file = None
            if run_tran and tran_out_path:
                ret_pl, l_pl = exec_cmd("setplot tran1")
                plot_err = any("error:" in l.lower() or "cannot find" in l.lower() for l in l_pl)
                if ret_pl != 0 or plot_err:
                    tran_status = "FAILED"
                    analysis_errors["tran"] = f"Failed to switch to tran1 plot (exit code {ret_pl})"
                else:
                    norm_tran = tran_out_path.replace("\\", "/")
                    t_wr_start = time.time()
                    ret_wr, l_wr = exec_cmd(f"wrdata {norm_tran} v(vin) v({output_node})")
                    wr_err = any("error:" in l.lower() for l in l_wr)
                    if (
                        ret_wr != 0
                        or wr_err
                        or not os.path.exists(tran_out_path)
                        or os.path.getsize(tran_out_path) == 0
                        or os.path.getmtime(tran_out_path) < (t_wr_start - 1.0)
                    ):
                        tran_status = "FAILED"
                        analysis_errors["tran"] = "Transient output vector file missing, empty, stale, or write failed"
                    else:
                        tran_status = "SUCCESS"
                        tran_file = tran_out_path

            # 5. Extract OP
            op_dict = {}
            if run_op:
                ret_pl, l_pl = exec_cmd("setplot op1")
                plot_err = any("error:" in l.lower() or "cannot find" in l.lower() for l in l_pl)
                if ret_pl != 0 or plot_err:
                    ret_pl2, l_pl2 = exec_cmd("setplot op2")
                    plot_err2 = any("error:" in l.lower() or "cannot find" in l.lower() for l in l_pl2)
                    if ret_pl2 != 0 or plot_err2:
                        op_status = "FAILED"
                        analysis_errors["op"] = f"Failed to switch to op plot (exit code {ret_pl})"
                    else:
                        plot_err = False
                if not plot_err:
                    ret_pr, l_pr = exec_cmd(f"print v({supply_node}) i({supply_source}) v({output_node})")
                    pr_err = any("error:" in l.lower() for l in l_pr)
                    # Parse OP
                    for line in l_pr:
                        line_str = line.strip()
                        if "=" in line_str:
                            parts = line_str.split("=")
                            if len(parts) == 2:
                                k = parts[0].strip().lower()
                                if " " in k:
                                    k = k.split()[-1]
                                try:
                                    v = float(parts[1].strip().split()[0])
                                    if np.isfinite(v):
                                        op_dict[k] = v
                                except Exception:
                                    pass
                    curr_key = f"i({supply_source.lower()})"
                    if ret_pr != 0 or pr_err or curr_key not in op_dict:
                        op_status = "FAILED"
                        analysis_errors["op"] = f"Operating point query missing supply current for source '{supply_source}'"
                    else:
                        op_status = "SUCCESS"

            # Clean memory plots and circuits
            exec_cmd("remcirc")
            exec_cmd("destroy all")

            req_checks = []
            if run_ac:
                req_checks.append(ac_status == "SUCCESS")
            if run_tran:
                req_checks.append(tran_status == "SUCCESS")
            if run_op:
                req_checks.append(op_status == "SUCCESS")

            overall_success = len(req_checks) > 0 and all(req_checks)

            analysis_statuses["ac"] = ac_status
            analysis_statuses["tran"] = tran_status
            analysis_statuses["op"] = op_status

            err_summary = None
            if not overall_success and analysis_errors:
                err_summary = "; ".join(f"{k}: {v}" for k, v in analysis_errors.items())

            conn.send({
                "success": overall_success,
                "error_message": err_summary,
                "op_status": op_status,
                "ac_status": ac_status,
                "tran_status": tran_status,
                "analysis_statuses": analysis_statuses,
                "analysis_errors": analysis_errors,
                "op_output": op_dict,
                "ac_file": ac_file,
                "tran_file": tran_file,
                "logs": list(logs),
                "must_restart": not overall_success,
            })


class _NGSpiceProcessContext:
    """
    Process-wide singleton context for the underlying ngspice shared library.
    Since ngspice.dll relies on global C static memory, multiple NGSpiceRunner instances
    in the same Python process must share the single library handle, synchronize access
    via a re-entrant lock, and route callback logs to whichever runner instance is currently executing.
    """
    lock: threading.RLock = threading.RLock()
    dll: Optional[ctypes.CDLL] = None
    dll_handle: Optional[int] = None
    backend_path: Optional[str] = None
    cb_char_ref: Any = None
    cb_exit_ref: Any = None
    active_log_buffer: Optional[List[str]] = None
    session_healthy: bool = True
    is_initialized: bool = False
    active_calls: int = 0


class NGSpiceRunner:
    """
    Direct interface to ngspice simulation engine.
    Supports in-memory ctypes integration with ngspice.dll as well as CLI subprocess fallback.
    Includes session health monitoring, explicit cleanup commands, and dynamic library detachment recovery.
    """

    def __init__(self, dll_path: Optional[str] = None, cli_path: Optional[str] = None):
        self.dll_path = dll_path or os.environ.get("NGSPICE_DLL")
        self.cli_path = cli_path or os.environ.get("NGSPICE_CLI")
        self._backend: Optional[str] = None  # "dll" or "cli" or None
        self._backend_diagnostic: str | None = None
        self._backend_attempts: list[dict[str, str]] = []
        self._last_dll_load_error: str | None = None
        self._log_buffer: List[str] = []
        self._worker_process: Optional[mp.Process] = None
        self._worker_conn: Optional[Any] = None
        self._worker_lock: threading.Lock = threading.Lock()
        self._init_backend()

    def _init_backend(self) -> None:
        """Locates and initializes the ngspice backend."""
        with _NGSpiceProcessContext.lock:
            # 1. Check if process context is already loaded
            if _NGSpiceProcessContext.is_initialized and _NGSpiceProcessContext.dll:
                self.dll_path = _NGSpiceProcessContext.backend_path
                self._backend = "dll"
                return

            # 2. Try specified or default DLL paths
            search_paths = [self.dll_path] if self.dll_path else []
            search_paths.extend(DEFAULT_DLL_SEARCH_PATHS)
            seen_paths: set[str] = set()
            for path in search_paths:
                if not path:
                    continue
                normalized = os.path.normcase(os.path.abspath(path))
                if normalized in seen_paths:
                    continue
                seen_paths.add(normalized)
                if not os.path.isfile(path):
                    if path == self.dll_path or path == _KICAD10_USER_DLL:
                        self._backend_attempts.append({"path": path, "result": "file not found"})
                    continue
                actual_architecture = _pe_architecture(path)
                process_bits = struct.calcsize("P") * 8
                expected_architecture = "AMD64" if process_bits == 64 and platform.machine().lower() in {"amd64", "x86_64"} else None
                if actual_architecture and expected_architecture and actual_architecture != expected_architecture:
                    diagnostic = f"Incompatible DLL architecture: found {actual_architecture}, but this process requires {expected_architecture}."
                    self._backend_attempts.append({"path": path, "result": diagnostic})
                    self._backend_diagnostic = diagnostic
                    continue
                if self._load_dll(path):
                    self.dll_path = path
                    self._backend = "dll"
                    self._backend_diagnostic = None
                    logger.info(f"Initialized ngspice DLL backend: {path}")
                    return
                diagnostic = self._last_dll_load_error or "DLL initialization failed; required ngspice exports or callbacks were unavailable."
                self._backend_attempts.append({"path": path, "result": diagnostic})
                self._backend_diagnostic = diagnostic

            # 3. Try CLI in PATH
            cli = self.cli_path or shutil.which("ngspice") or shutil.which("ngspice.exe")
            if cli:
                self.cli_path = cli
                self._backend = "cli"
                self._backend_diagnostic = "The CLI backend is available; vector extraction features require the shared DLL backend."
                _NGSpiceProcessContext.session_healthy = True
                logger.info(f"Initialized ngspice CLI backend: {cli}")
                return

            if not self._backend_diagnostic:
                self._backend_diagnostic = "No usable ngspice DLL or CLI was found in the configured or supported locations."
            logger.warning("ngspice backend is unavailable: %s", self._backend_diagnostic)

    def _load_dll(self, path: str) -> bool:
        """Loads and binds ctypes functions for ngspice shared library with serialized handle tracking."""
        with _NGSpiceProcessContext.lock:
            try:
                self._last_dll_load_error = None
                handle = None
                if sys.platform == "win32" and _kernel32:
                    bin_dir = os.path.dirname(path)
                    if hasattr(os, "add_dll_directory") and bin_dir and os.path.isdir(bin_dir):
                        try:
                            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(bin_dir))
                        except OSError as exc:
                            logger.debug("Could not add ngspice DLL directory %s: %s", bin_dir, exc)
                    handle = _kernel32.LoadLibraryW(path)
                    if not handle:
                        err = ctypes.get_last_error()
                        try:
                            detail = ctypes.FormatError(err).strip()
                        except Exception:
                            detail = ""
                        self._last_dll_load_error = f"Windows could not load the ngspice DLL (WinError {err}{': ' + detail if detail else ''}). Check dependent DLL availability and architecture."
                        logger.debug("LoadLibraryW failed on %s with Win32 error %s", path, err)
                        return False
                    _NGSpiceProcessContext.dll_handle = handle
                    dll = ctypes.CDLL(path, handle=handle)
                else:
                    dll = ctypes.CDLL(path)
                    _NGSpiceProcessContext.dll_handle = None

                if hasattr(dll, "ngSpice_Init") and hasattr(dll, "ngSpice_Command"):
                    _NGSpiceProcessContext.dll = dll
                    _NGSpiceProcessContext.backend_path = path
                    ok = self._setup_ctypes(dll)
                    if not ok and handle and sys.platform == "win32" and _kernel32:
                        try:
                            _kernel32.FreeLibrary(handle)
                        except Exception:
                            pass
                        _NGSpiceProcessContext.dll_handle = None
                        _NGSpiceProcessContext.dll = None
                    if not ok:
                        self._last_dll_load_error = "The DLL loaded, but ngSpice_Init callback initialization failed."
                    return ok
                else:
                    missing = [name for name in ("ngSpice_Init", "ngSpice_Command") if not hasattr(dll, name)]
                    self._last_dll_load_error = "The DLL is missing required exports: " + ", ".join(missing)
                    if handle and sys.platform == "win32" and _kernel32:
                        try:
                            _kernel32.FreeLibrary(handle)
                        except Exception:
                            pass
                        _NGSpiceProcessContext.dll_handle = None
                    return False
            except Exception as e:
                self._last_dll_load_error = f"DLL load or initialization raised {type(e).__name__}: {e}"
                logger.debug(f"Could not load ngspice DLL at {path}: {e}")
                return False

    def _setup_ctypes(self, dll: ctypes.CDLL) -> bool:
        """Sets up function prototypes and callbacks matching exact native sharedspice.h ABI."""
        SendChar = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)
        # Exact 5-argument ControlledExit ABI: (int status, bool immediate, bool quitexit, int ident, void* userData)
        ExitCb = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.c_int,
            ctypes.c_void_p,
        )

        dll.ngSpice_Init.argtypes = [
            SendChar,
            ctypes.c_void_p,
            ExitCb,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        dll.ngSpice_Init.restype = ctypes.c_int

        dll.ngSpice_Command.argtypes = [ctypes.c_char_p]
        dll.ngSpice_Command.restype = ctypes.c_int

        if hasattr(dll, "ngSpice_Reset"):
            dll.ngSpice_Reset.argtypes = []
            dll.ngSpice_Reset.restype = ctypes.c_int

        def on_char(msg: bytes, ident: int, user_data: Any) -> int:
            with _NGSpiceProcessContext.lock:
                if msg and _NGSpiceProcessContext.active_log_buffer is not None:
                    s = msg.decode("utf-8", errors="ignore")
                    _NGSpiceProcessContext.active_log_buffer.append(s)
            return 0

        def on_exit(status: int, immediate: bool, quitexit: bool, ident: int, user_data: Any) -> int:
            with _NGSpiceProcessContext.lock:
                _NGSpiceProcessContext.session_healthy = False
                if _NGSpiceProcessContext.active_log_buffer is not None:
                    _NGSpiceProcessContext.active_log_buffer.append(
                        f"stderr ngspice ControlledExit: status={status}, immediate={immediate}, quitexit={quitexit}, ident={ident}"
                    )
            return 0

        _NGSpiceProcessContext.cb_char_ref = SendChar(on_char)
        _NGSpiceProcessContext.cb_exit_ref = ExitCb(on_exit)
        ret = dll.ngSpice_Init(
            _NGSpiceProcessContext.cb_char_ref,
            None,
            _NGSpiceProcessContext.cb_exit_ref,
            None,
            None,
            None,
            None,
        )
        if ret != 0:
            logger.error(f"ngSpice_Init failed with return code {ret}")
            _NGSpiceProcessContext.session_healthy = False
            _NGSpiceProcessContext.is_initialized = False
            return False

        _NGSpiceProcessContext.session_healthy = True
        _NGSpiceProcessContext.is_initialized = True
        return True

    def _ensure_worker(self) -> bool:
        """Ensures that the process-isolated ngspice worker is running and responsive."""
        if self._backend != "dll" or not self.dll_path:
            return False
        with self._worker_lock:
            if self._worker_process is not None and self._worker_process.is_alive() and self._worker_conn is not None:
                return True
            self._terminate_worker_locked()
            try:
                parent_conn, child_conn = mp.Pipe()
                p = mp.Process(
                    target=_ngspice_worker_loop,
                    args=(child_conn, self.dll_path),
                    daemon=True,
                )
                p.start()
                if parent_conn.poll(timeout=10.0):
                    init_msg = parent_conn.recv()
                    if isinstance(init_msg, dict) and init_msg.get("type") == "READY":
                        self._worker_process = p
                        self._worker_conn = parent_conn
                        with _NGSpiceProcessContext.lock:
                            _NGSpiceProcessContext.session_healthy = True
                        return True
                    else:
                        logger.error(f"Worker init failed: {init_msg}")
                        p.terminate()
                        p.join(timeout=1.0)
                        return False
                else:
                    logger.error("Worker process timed out during initialization.")
                    p.terminate()
                    p.join(timeout=1.0)
                    return False
            except Exception as e:
                logger.error(f"Could not spawn ngspice worker process: {e}")
                return False

    def _terminate_worker_locked(self) -> None:
        """Terminates the worker process under lock."""
        if self._worker_conn is not None:
            try:
                self._worker_conn.send("SHUTDOWN")
            except Exception:
                pass
            try:
                self._worker_conn.close()
            except Exception:
                pass
            self._worker_conn = None
        if self._worker_process is not None:
            try:
                if self._worker_process.is_alive():
                    self._worker_process.terminate()
                    self._worker_process.join(timeout=1.0)
            except Exception:
                pass
            self._worker_process = None

    def __del__(self) -> None:
        try:
            self._terminate_worker_locked()
        except Exception:
            pass

    def _reload_dll(self) -> bool:
        """
        Safely detaches and reloads the ngspice shared library to recover from fatal engine crashes.
        Terminates the worker process and reloads the in-process DLL.
        """
        with self._worker_lock:
            self._terminate_worker_locked()

        with _NGSpiceProcessContext.lock:
            path = self.dll_path or _NGSpiceProcessContext.backend_path
            if not path:
                return False

            if _NGSpiceProcessContext.active_calls > 0:
                logger.warning("Cannot unload ngspice DLL while native calls are in flight.")
                return False

            logger.info("Safely detaching and reloading ngspice DLL to restore session health...")

            # 1. Unbind ctypes function pointers and clear callback references
            _NGSpiceProcessContext.dll = None
            _NGSpiceProcessContext.cb_char_ref = None
            _NGSpiceProcessContext.cb_exit_ref = None
            _NGSpiceProcessContext.active_log_buffer = None
            _NGSpiceProcessContext.is_initialized = False
            _NGSpiceProcessContext.session_healthy = False

            # 2. Force GC to collect any lingering ctypes wrapper objects
            gc.collect()

            # 3. Detach library handle on Windows if owned
            if sys.platform == "win32" and _kernel32 and _NGSpiceProcessContext.dll_handle:
                try:
                    ret_free = _kernel32.FreeLibrary(_NGSpiceProcessContext.dll_handle)
                    if ret_free == 0:
                        err = ctypes.get_last_error()
                        logger.debug(f"FreeLibrary returned 0 (Win32 error: {err})")
                except Exception as e:
                    logger.debug(f"FreeLibrary exception: {e}")
                _NGSpiceProcessContext.dll_handle = None

            # 4. Reload clean library instance
            ok = self._load_dll(path)
            if ok:
                self._backend = "dll"
                self.dll_path = path
                logger.info("Successfully reloaded pristine ngspice shared library instance.")
            else:
                logger.error("Failed to reload ngspice shared library.")
            return ok

    @property
    def _session_healthy(self) -> bool:
        with _NGSpiceProcessContext.lock:
            return _NGSpiceProcessContext.session_healthy

    @_session_healthy.setter
    def _session_healthy(self, value: bool) -> None:
        with _NGSpiceProcessContext.lock:
            _NGSpiceProcessContext.session_healthy = value

    def is_available(self) -> bool:
        """Returns True if a working ngspice backend is loaded."""
        with _NGSpiceProcessContext.lock:
            return self._backend is not None

    def is_available_for_extraction(self) -> bool:
        """
        Returns True only if the active backend satisfies the full vector extraction contract.
        Currently, only the in-memory shared library DLL satisfies this contract.
        """
        with _NGSpiceProcessContext.lock:
            return self._backend == "dll"

    def is_session_healthy(self) -> bool:
        """Returns True if the current simulation session has not encountered an unrecoverable failure."""
        with _NGSpiceProcessContext.lock:
            return _NGSpiceProcessContext.session_healthy and (self._backend is not None)

    def reset_session(self) -> bool:
        """
        Deterministically resets ngspice simulator state to recover from errors.
        Terminates the isolated worker process and prepares a pristine worker instance.
        """
        with self._worker_lock:
            self._terminate_worker_locked()
        with _NGSpiceProcessContext.lock:
            _NGSpiceProcessContext.session_healthy = True
            _NGSpiceProcessContext.is_initialized = True
            dll = _NGSpiceProcessContext.dll
            if self._backend == "dll" and dll and hasattr(dll, "ngSpice_Reset") and hasattr(dll, "ngSpice_Init"):
                try:
                    dll.ngSpice_Reset()
                    dll.ngSpice_Init(
                        _NGSpiceProcessContext.cb_char_ref,
                        None,
                        _NGSpiceProcessContext.cb_exit_ref,
                        None,
                        None,
                        None,
                        None,
                    )
                except Exception:
                    pass
        return True

    def get_backend_info(self) -> str:
        """Returns human-readable info about the active ngspice backend."""
        with _NGSpiceProcessContext.lock:
            if self._backend == "dll":
                status_str = "Healthy" if self.is_session_healthy() else "Awaiting Reset/Reload"
                return f"ngspice Shared Library DLL ({self.dll_path}) [{status_str}]"
            elif self._backend == "cli":
                return f"ngspice CLI Executable ({self.cli_path}) [Extraction Unsupported: DLL required]"
            diagnostic = self._backend_diagnostic or "No ngspice backend is configured."
            return f"ngspice Not Available [{diagnostic}]"

    def get_backend_status(self) -> dict[str, Any]:
        """Return the actual backend state and the reason it is unavailable or limited."""
        with _NGSpiceProcessContext.lock:
            if self._backend == "dll":
                healthy = self.is_session_healthy()
                status = "healthy" if healthy else "unhealthy"
                diagnostic = None if healthy else "The loaded ngspice session has reported an unrecoverable failure; reset or reload it before another run."
                path = self.dll_path
            elif self._backend == "cli":
                status = "cli_limited"
                diagnostic = "The CLI backend is available, but DLL-only vector extraction is unsupported."
                path = self.cli_path
            else:
                load_failures = [item for item in self._backend_attempts if item.get("result") != "file not found"]
                status = "load_failed" if load_failures else "missing_dll"
                diagnostic = self._backend_diagnostic or "No ngspice DLL was found in supported locations."
                path = None
            return {
                "status": status,
                "backend": self._backend,
                "path": path,
                "available_for_extraction": self.is_available_for_extraction(),
                "session_healthy": self.is_session_healthy(),
                "diagnostic": diagnostic,
                "attempts": [dict(item) for item in self._backend_attempts],
            }

    def _exec_cmd(self, cmd_str: str) -> Tuple[int, List[str]]:
        """Executes a single ngspice command with serialized thread access and return code validation."""
        with _NGSpiceProcessContext.lock:
            dll = _NGSpiceProcessContext.dll
            if self._backend != "dll" or not dll:
                return -1, ["ngspice DLL not available"]
            _NGSpiceProcessContext.active_log_buffer = self._log_buffer
            idx_start = len(self._log_buffer)
            _NGSpiceProcessContext.active_calls += 1
            try:
                ret = dll.ngSpice_Command(cmd_str.encode("utf-8"))
            finally:
                _NGSpiceProcessContext.active_calls = max(0, _NGSpiceProcessContext.active_calls - 1)
            new_logs = self._log_buffer[idx_start:]
            # Check if this command triggered fatal unrecoverable state
            if any("cannot recover and awaits to be reset or detached" in l.lower() for l in new_logs):
                _NGSpiceProcessContext.session_healthy = False
            return ret, new_logs

    def run_netlist(
        self,
        netlist_content: str,
        output_node: str = "outp",
        supply_node: str = "vdd",
        supply_source: str = "VDD",
        run_ac: bool = True,
        run_tran: bool = True,
        run_op: bool = True,
        ac_sweep: str = "dec 20 1k 100Meg",
        tran_step_stop: str = "0.1n 50n",
    ) -> SimulationResult:
        """
        Executes a complete SPICE simulation run on the provided netlist string with per-analysis validation.

        Args:
            netlist_content: Full SPICE netlist string including .end.
            output_node: Signal output node for AC/Transient measurement.
            supply_node: Positive power supply node name (e.g. 'vdd').
            supply_source: Voltage source name (e.g. 'VDD').
            run_ac: Whether to execute AC frequency sweep.
            run_tran: Whether to execute transient time-domain simulation.
            run_op: Whether to execute DC operating point analysis.
            ac_sweep: Parameter string for AC sweep (e.g. 'dec 20 1k 100Meg').
            tran_step_stop: Step and stop time for transient (e.g. '0.1n 50n').

        Returns:
            SimulationResult with per-analysis statuses, error messages, and vector file paths.
        """
        if not self.is_available():
            return SimulationResult(
                success=False,
                error_message="ngspice simulation backend is not available.",
                backend_info="None",
                op_status="FAILED" if run_op else "NOT_REQUESTED",
                ac_status="FAILED" if run_ac else "NOT_REQUESTED",
                tran_status="FAILED" if run_tran else "NOT_REQUESTED",
                analysis_statuses={"backend": "NOT_AVAILABLE"},
            )

        start_time = time.perf_counter()
        self._log_buffer.clear()

        # Temporary files for netlist and output vectors with UUID to guarantee no collisions
        temp_dir = tempfile.gettempdir()
        cir_fd, cir_path = tempfile.mkstemp(suffix=".cir", prefix="sp_sim_")
        os.close(cir_fd)

        unique_id = uuid.uuid4().hex
        ac_out_path = os.path.join(temp_dir, f"sp_ac_{os.getpid()}_{unique_id}.txt")
        tran_out_path = os.path.join(temp_dir, f"sp_tran_{os.getpid()}_{unique_id}.txt")

        # Ensure target files do not exist prior to simulation run
        for p in [ac_out_path, tran_out_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

        try:
            # Prepare netlist
            netlist_lines = netlist_content.strip().splitlines()
            cleaned_lines = []
            for line in netlist_lines:
                sline = line.strip().lower()
                if sline == ".end":
                    continue
                if run_ac and sline.startswith(".ac"):
                    continue
                if run_tran and sline.startswith(".tran"):
                    continue
                if run_op and sline.startswith(".op"):
                    continue
                cleaned_lines.append(line)

            if run_op:
                cleaned_lines.append(".op")
            if run_ac:
                cleaned_lines.append(f".ac {ac_sweep}")
            if run_tran:
                cleaned_lines.append(f".tran {tran_step_stop}")
            cleaned_lines.append(".end\n")

            full_netlist = "\n".join(cleaned_lines)
            with open(cir_path, "w", encoding="utf-8") as f:
                f.write(full_netlist)

            # Execution via DLL backend
            if self._backend == "dll":
                return self._run_via_dll(
                    cir_path=cir_path,
                    output_node=output_node,
                    supply_node=supply_node,
                    supply_source=supply_source,
                    ac_out_path=ac_out_path if run_ac else None,
                    tran_out_path=tran_out_path if run_tran else None,
                    run_op=run_op,
                    run_ac=run_ac,
                    run_tran=run_tran,
                    start_time=start_time,
                )
            else:
                return self._run_via_cli(
                    cir_path=cir_path,
                    start_time=start_time,
                )

        except Exception as e:
            exec_time = time.perf_counter() - start_time
            logger.error(f"Simulation execution failed: {e}", exc_info=True)
            return SimulationResult(
                success=False,
                error_message=f"Simulation error: {str(e)}",
                execution_time_sec=exec_time,
                log_messages=list(self._log_buffer),
                backend_info=self.get_backend_info(),
                op_status="FAILED" if run_op else "NOT_REQUESTED",
                ac_status="FAILED" if run_ac else "NOT_REQUESTED",
                tran_status="FAILED" if run_tran else "NOT_REQUESTED",
                analysis_statuses={"exception": str(e)},
            )
        finally:
            if os.path.exists(cir_path):
                try:
                    os.remove(cir_path)
                except Exception:
                    pass

    def _run_via_dll(
        self,
        cir_path: str,
        output_node: str,
        supply_node: str,
        supply_source: str,
        ac_out_path: Optional[str],
        tran_out_path: Optional[str],
        run_op: bool,
        run_ac: bool,
        run_tran: bool,
        start_time: float,
    ) -> SimulationResult:
        """Executes simulation commands via isolated worker process with automatic recovery."""
        if not self._ensure_worker():
            return SimulationResult(
                success=False,
                error_message="Could not start ngspice worker process.",
                execution_time_sec=time.perf_counter() - start_time,
                backend_info=self.get_backend_info(),
                op_status="FAILED" if run_op else "NOT_REQUESTED",
                ac_status="FAILED" if run_ac else "NOT_REQUESTED",
                tran_status="FAILED" if run_tran else "NOT_REQUESTED",
                analysis_statuses={"worker": "SPAWN_FAILED"},
            )

        payload = {
            "type": "RUN_SIM",
            "cir_path": cir_path,
            "output_node": output_node,
            "supply_node": supply_node,
            "supply_source": supply_source,
            "ac_out_path": ac_out_path,
            "tran_out_path": tran_out_path,
            "run_op": run_op,
            "run_ac": run_ac,
            "run_tran": run_tran,
        }

        try:
            with self._worker_lock:
                if self._worker_conn is None:
                    raise BrokenPipeError("Worker connection is not available.")
                self._worker_conn.send(payload)
                if self._worker_conn.poll(timeout=30.0):
                    res_data = self._worker_conn.recv()
                else:
                    self._terminate_worker_locked()
                    with _NGSpiceProcessContext.lock:
                        _NGSpiceProcessContext.session_healthy = False
                    return SimulationResult(
                        success=False,
                        error_message="Simulation timed out in ngspice worker process (exceeded 30s)",
                        execution_time_sec=time.perf_counter() - start_time,
                        backend_info=self.get_backend_info(),
                        op_status="FAILED" if run_op else "NOT_REQUESTED",
                        ac_status="FAILED" if run_ac else "NOT_REQUESTED",
                        tran_status="FAILED" if run_tran else "NOT_REQUESTED",
                        analysis_statuses={"worker": "TIMEOUT"},
                    )

            # If worker requested restart (e.g. fatal netlist error), recycle worker process immediately
            if res_data.get("must_restart", False):
                with self._worker_lock:
                    self._terminate_worker_locked()
                with _NGSpiceProcessContext.lock:
                    _NGSpiceProcessContext.session_healthy = True

            self._log_buffer.extend(res_data.get("logs", []))
            exec_time = time.perf_counter() - start_time

            return SimulationResult(
                success=res_data.get("success", False),
                ac_file=res_data.get("ac_file"),
                tran_file=res_data.get("tran_file"),
                op_output=res_data.get("op_output", {}),
                log_messages=list(self._log_buffer),
                execution_time_sec=exec_time,
                error_message=res_data.get("error_message"),
                backend_info=self.get_backend_info(),
                op_status=res_data.get("op_status", "FAILED" if run_op else "NOT_REQUESTED"),
                ac_status=res_data.get("ac_status", "FAILED" if run_ac else "NOT_REQUESTED"),
                tran_status=res_data.get("tran_status", "FAILED" if run_tran else "NOT_REQUESTED"),
                analysis_statuses=res_data.get("analysis_statuses", {}),
                analysis_errors=res_data.get("analysis_errors", {}),
            )
        except (EOFError, BrokenPipeError, ConnectionResetError) as e:
            logger.error(f"Worker connection broken (process terminated/crashed): {e}")
            with self._worker_lock:
                self._terminate_worker_locked()
            with _NGSpiceProcessContext.lock:
                _NGSpiceProcessContext.session_healthy = True
            return SimulationResult(
                success=False,
                error_message=f"ngspice worker process terminated unexpectedly: {e}",
                execution_time_sec=time.perf_counter() - start_time,
                backend_info=self.get_backend_info(),
                op_status="FAILED" if run_op else "NOT_REQUESTED",
                ac_status="FAILED" if run_ac else "NOT_REQUESTED",
                tran_status="FAILED" if run_tran else "NOT_REQUESTED",
                analysis_statuses={"worker": "CRASHED"},
            )

    def _run_via_cli(self, cir_path: str, start_time: float) -> SimulationResult:
        """
        CLI backend fallback.
        Explicitly reports UNSUPPORTED for metric extraction because it does not satisfy
        the required in-memory vector/plot output contract.
        """
        exec_time = time.perf_counter() - start_time
        err_msg = (
            "ngspice CLI backend does not implement the vector output contract "
            "required for physical metric extraction. In-memory DLL backend is required."
        )
        return SimulationResult(
            success=False,
            error_message=err_msg,
            execution_time_sec=exec_time,
            backend_info=self.get_backend_info(),
            op_status="UNSUPPORTED",
            ac_status="UNSUPPORTED",
            tran_status="UNSUPPORTED",
            analysis_statuses={"op": "UNSUPPORTED", "ac": "UNSUPPORTED", "tran": "UNSUPPORTED"},
            analysis_errors={"cli": err_msg},
        )

    def _parse_op_from_logs(
        self, logs: List[str], supply_node: str, supply_source: str, output_node: str
    ) -> Dict[str, float]:
        """Parses node voltages and supply current from printed OP logs."""
        op_data: Dict[str, float] = {}
        for line in logs:
            line_str = line.strip()
            # Example: "v(vdd) = 1.800000e+00", "i(vdd) = -1.02636e-04"
            if "=" in line_str:
                parts = line_str.split("=")
                if len(parts) == 2:
                    k = parts[0].strip().lower()
                    # Strip leading stdout/stderr tags if present
                    if " " in k:
                        k = k.split()[-1]
                    try:
                        v = float(parts[1].strip())
                        if np.isfinite(v):
                            op_data[k] = v
                    except ValueError:
                        pass
        return op_data


@lru_cache(maxsize=1)
def get_shared_ngspice_runner() -> NGSpiceRunner:
    """Return one backend instance for all Streamlit pages in this process."""
    return NGSpiceRunner()
