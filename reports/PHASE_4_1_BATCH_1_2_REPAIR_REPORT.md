# SiliconPilot AI — Phase 4.1 Batch 1.2 Repair Report
## Critical Simulation Reliability, Native ABI Conformance, Thread Safety & Evidence Validation

**Date:** 12 November 2026  
**Auditor Reference:** Independent Codex Review & Sign-Off (`reports/CODEX_BATCH_1_FINAL_SIGNOFF.md`)  
**Scope:** Phase 4.1, Batch 1.2 Critical Reliability Repair  
**Execution Environment:** Windows 10/11 x64, Python 3.11.9 (`.venv`), ngspice-46 shared library (`C:\Program Files\KiCad\9.0\bin\ngspice.dll`)  
**Status:** All 6 Codex-identified test failures resolved; 43/43 repository tests passing repeatedly across fresh processes.

---

## 1. Executive Summary

Following the independent Codex audit of Batch 1, which returned **NOT APPROVED** due to 6 reproduced failures (33 passed, 6 failed in two consecutive independent runs) and high-severity architectural defects, a comprehensive engineering overhaul was performed on the simulation runner, database boundary, candidate optimizer, and parser subsystems.

Key architectural improvements in Batch 1.2:
1. **Native ABI Conformance:** Corrected the `ControlledExit` callback signature from 4 parameters to the exact 5-parameter ABI declared in ngspice's `sharedspice.h`.
2. **Synchronized Process Context & Safe Lifecycle:** Enforced thread synchronization across all native ctypes calls using `threading.RLock()`, added in-flight call tracking (`active_calls`), explicit handle ownership, Win32 `FreeLibrary` / `LoadLibraryW` return-code verification, and deterministic recovery via `ngSpice_Reset()` + `ngSpice_Init()` prior to any module detachment.
3. **Database Trust Boundary Hardening:** Upgraded `_validate_simulation_evidence` in `FeedbackDatabase` to enforce simulator identity (`ngspice`), version strings, netlist/circuit provenance, non-empty physical sizing parameters, complete analysis statuses, finite and physically bounded metric values, and contradictory-provenance rejection.
4. **Strict Pareto Filtering:** Enforced fail-closed behavior when `filter_pareto_front=True`. If candidates lack Pareto metadata or if the rank-1 front is empty, the selector fails closed with an empty selection and `winner_id=None`, never falling back to rank-2 candidates or the original candidate set.
5. **AC Metric Validity Safeguards:** Enforced that missing unity-gain crossings and missing phase vectors return `None` (unavailable) rather than synthetic numeric values (0° phase or sweep endpoint GBW). Missing stability evidence fails closed during candidate winner selection.

---

## 2. Root Cause Analysis of the Six Test Failures

All six test failures reported by Codex stemmed from a single cascading failure in native shared-library lifecycle management:

| Failing Test | Reported Symptom | Root Cause | Engineering Correction |
|---|---|---|---|
| `test_11_real_ngspice_power_measurement_parity` | `res.success` was False | The ngspice DLL entered an unrecoverable state after earlier syntax error tests (`test_01`). The previous reload mechanism called `FreeLibrary` while Python ctypes objects still held internal references, preventing clean re-initialization. | Replaced uncontrolled module reload with synchronized in-process reset (`ngSpice_Reset()` + `ngSpice_Init()`) and safe serialized dynamic reload. |
| `test_16_dll_session_health_and_automatic_recovery` | Recovery / live follow-up assertion failed | In `reset_session()`, the check `if ret != 0` mistakenly treated return code `0` (success) from `ngSpice_Reset()` as failure, causing unnecessary and corrupting module reload cycles. | Corrected return-code handling (`ret == 0` is success) and restored clean re-initialization. |
| `test_03_simulation_execution_ac_and_tran` | Live simulation execution failed | Cascading consequence of the unrecovered ngspice session from `test_16`. Sourcing netlists on a corrupted session returned non-zero. | Session health is maintained and deterministically reset between simulation runs. |
| `test_04_simulation_parsing` | AC parsed output was invalid | Due to solver failure in `test_03`, output vector files were either unwritten or contained partial error text. | Runner now verifies fresh file modification time (`mtime >= start_time - 1.0`), non-empty existence, and valid parsed points. |
| `test_05_metric_extraction` | `stderr Error: ngspice.dll cannot recover and awaits to be reset or detached` | Shared library session locked up because ngspice's internal state machine was never reset after previous test failures. | Handled via in-process `ngSpice_Reset()` + callback re-registration. |
| `test_08_simulation_in_the_loop_candidate_selection` | `winner_id` was `None` | Because live SPICE simulations returned `None` due to library lock-up, candidate phase margins were `None`, causing all candidates to be excluded by the 40° stability gate. | Clean live simulations produce valid measured PM (40.25° for Cand_B), correctly selecting Cand_B as winner. |

---

## 3. Native Callback ABI Correction (`ControlledExit`)

### A. The Defect
ngspice's public shared-library interface header (`sharedspice.h`, line 1130) defines:
```c
typedef int (ControlledExit)(int exit_status, bool immediate, bool quit, int id, void *user_data);
```
The previous implementation declared:
```python
ExitCb = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool, ctypes.c_void_p)
```
This signature omitted the fourth argument (`int id`), causing Python ctypes to interpret `id` as `user_data` and leave the native stack/registers misaligned during controlled exits.

### B. The Correction
Corrected `ExitCb` in `src/simulation/ngspice_runner.py` to match the exact 5-parameter ABI:
```python
# Exact 5-argument ControlledExit ABI: (int status, bool immediate, bool quitexit, int ident, void* userData)
ExitCb = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_void_p,
)

def on_exit(status: int, immediate: bool, quitexit: bool, ident: int, user_data: Any) -> int:
    with _NGSpiceProcessContext.lock:
        _NGSpiceProcessContext.session_healthy = False
        if _NGSpiceProcessContext.active_log_buffer is not None:
            _NGSpiceProcessContext.active_log_buffer.append(
                f"stderr ngspice ControlledExit: status={status}, immediate={immediate}, quitexit={quitexit}, ident={ident}"
            )
    return 0
```
A dedicated regression test (`test_17_controlled_exit_callback_5arg_abi`) was added to verify invocation with all 5 arguments without stack corruption or ctypes errors.

---

## 4. DLL Lifecycle Architecture, Thread Safety & Recovery

### A. Thread-Safe Process Context
Because ngspice is a native C library relying on process-global static state, concurrent native calls from multiple threads can corrupt memory or overwrite log buffers.
- Added `_NGSpiceProcessContext.lock = threading.RLock()`.
- Every entry point (`_load_dll`, `_reload_dll`, `_exec_cmd`, `reset_session`, `is_session_healthy`) acquires the lock.
- Added `_NGSpiceProcessContext.active_calls: int = 0` to track in-flight native calls. DLL detachment is strictly forbidden while `active_calls > 0`.

### B. Safe Handle Ownership & Dynamic Detach
- When loading on Windows, `LoadLibraryW` obtains the module handle.
- Before `FreeLibrary` is invoked:
  1. All ctypes function pointer references (`dll`, `cb_char_ref`, `cb_exit_ref`, `active_log_buffer`) are set to `None`.
  2. `gc.collect()` is invoked to finalize lingering ctypes wrapper objects.
  3. `_kernel32.FreeLibrary(handle)` is called and its return code is validated against 0 (with `ctypes.get_last_error()` logging on error).
  4. Pristine module instance is reloaded via `LoadLibraryW` and function prototypes are rebound.

### C. Deterministic Recovery Design
In-process soft reset via `dll.ngSpice_Reset()` clears circuits, plots, and simulation structures while preserving the module mapping. Calling `dll.ngSpice_Init(...)` immediately after re-registers callbacks and restores ngspice to ready status in $<1\text{ ms}$.
Dynamic module detachment is retained as a second-tier fallback if and only if in-process reset fails.

---

## 5. Feedback Database Evidence Validation Rules

In `src/simulation/feedback_db.py`, `_validate_simulation_evidence` was hardened against caller forgery:

1. **Status:** `simulation_status == "Converged"` (case-sensitive).
2. **Provenance:** Non-empty `circuit_id` and `topology` strings.
3. **Parameters:** Non-empty dictionary with finite numeric values.
4. **Simulator Identity:** If provided, must not be empty, None, or `"unknown"`.
5. **Simulator Version:** If provided, must be a non-empty string.
6. **Physical Metric Bounds:**
   - Power dissipation must be strictly positive ($> 0\text{ mW}$) and reasonable ($< 10\text{ W}$).
   - DC Gain must be physically plausible ($-100\text{ dB} \le \text{gain} \le +250\text{ dB}$).
   - Non-finite metrics (NaN, $+ \infty$, $- \infty$) are rejected.
7. **Analysis Statuses:** No analysis may have status `"FAILED"`, `"ERROR"`, or `"UNSUPPORTED"`. At least one analysis must be `"SUCCESS"`.
8. **Contradictory Provenance:** If a metric value is claimed, its provenance must not be `"unavailable"`.
9. **Promotion Security:** `mark_as_verified` loads stored JSON fields from SQLite and runs the complete evidence validator before setting `verified_for_training = 1`.

---

## 6. Strict Pareto Filtering Changes

In `src/optimization/candidate_optimizer.py`, `run_simulation_validation` now fails closed when strict Pareto filtering (`filter_pareto_front=True`) is requested:

- **Missing Metadata:** If the candidate DataFrame lacks both `"pareto_rank"` and `"is_pareto_optimal"`, the selector fails closed, returning `success=False`, `evaluated_candidates=[]`, and `winner_id=None`.
- **Empty Rank-1 Front:** If no candidate has `pareto_rank == 1` or `is_pareto_optimal == True`, the selector fails closed with `evaluated_candidates=[]` and `winner_id=None`. It **never** falls back to rank 2 or the unranked candidate set.
- **Deterministic Slicing:** When rank-1 candidates exist, they are sorted deterministically by `composite_score` descending and capped at `top_k`. Dominated candidates are strictly excluded from SPICE simulation.

---

## 7. AC Metric Validity Safeguards

In `src/simulation/simulation_parser.py` and `src/simulation/simulation_metrics.py`:

- **Missing Phase:** When ngspice vector outputs lack the phase column (`parts < 4`), `ACSimulationData.has_phase` is set to `False` and `phase_deg` is left empty. Missing phase remains `None`, never converted to `0.0°`.
- **Missing Unity Crossing:** If the AC gain response never drops $\le 0\text{ dB}$ within the sweep range (or if DC gain $\le 0\text{ dB}$), both `gain_bandwidth_product_hz` and `phase_margin_deg` return `None`. Sweep endpoints are never used to invent synthetic GBW or PM.
- **Closed Stability Gating:** In `CandidateOptimizer`, candidate qualification strictly requires `spice_phase_margin is not None` and `spice_phase_margin >= min_phase_margin`. A candidate with `None` phase margin cannot pass the stability gate.

*(Full complex differential transfer function extraction and phase unwrapping remain scoped for Batch 2).*

---

## 8. Files Modified & Backups Created

### Backups Created
Preserved in [backups/phase4_1_batch1_2_pre_repair/](file:///c:/Users/SHREYA/OneDrive/Desktop/samsung%20project/SiliconPilot-AI/backups/phase4_1_batch1_2_pre_repair):
- `candidate_optimizer.py`
- `feedback_db.py`
- `ngspice_runner.py`
- `simulation_metrics.py`
- `simulation_parser.py`
- `test_phase4_1_batch1.py`
- `test_phase4_simulation.py`

### Files Modified
- [src/simulation/ngspice_runner.py](file:///c:/Users/SHREYA/OneDrive/Desktop/samsung%20project/SiliconPilot-AI/src/simulation/ngspice_runner.py): Corrected `ControlledExit` 5-argument callback, thread lock `RLock`, in-flight call tracking, soft reset via `ngSpice_Reset()`, and output file mtime freshness verification.
- [src/simulation/feedback_db.py](file:///c:/Users/SHREYA/OneDrive/Desktop/samsung%20project/SiliconPilot-AI/src/simulation/feedback_db.py): Strengthened evidence validation with circuit/topology provenance, simulator identity/version checks, unphysical metric rejection, and contradictory-provenance rejection.
- [src/optimization/candidate_optimizer.py](file:///c:/Users/SHREYA/OneDrive/Desktop/samsung%20project/SiliconPilot-AI/src/optimization/candidate_optimizer.py): Implemented fail-closed strict Pareto filtering on missing metadata or empty rank-1 front.
- [src/simulation/simulation_parser.py](file:///c:/Users/SHREYA/OneDrive/Desktop/samsung%20project/SiliconPilot-AI/src/simulation/simulation_parser.py): Added `has_phase` tracking and stopped defaulting missing phase to `0.0°`.
- [src/simulation/simulation_metrics.py](file:///c:/Users/SHREYA/OneDrive/Desktop/samsung%20project/SiliconPilot-AI/src/simulation/simulation_metrics.py): Guarded GBW and PM extraction against missing crossings and missing phase vectors.
- [tests/test_phase4_1_batch1.py](file:///c:/Users/SHREYA/OneDrive/Desktop/samsung%20project/SiliconPilot-AI/tests/test_phase4_1_batch1.py): Expanded test suite to 20 tests (added tests 17–20 covering callback ABI, strict Pareto fail-closed behaviors, and AC metric safeguards).

---

## 9. Independent Fresh-Process Test Results

Executed via PowerShell using the project `.venv` and KiCad 9 ngspice DLL:
```powershell
$env:NGSPICE_DLL = 'C:\Program Files\KiCad\9.0\bin\ngspice.dll'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PATH = 'C:\Program Files\KiCad\9.0\bin;' + $env:PATH
Set-Location 'c:\Users\SHREYA\OneDrive\Desktop\samsung project\SiliconPilot-AI'
& '.\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests -v
```

### Process 1 (Fresh Process Run)
- **Total Tests Collected:** 43
- **Passed:** **43**
- **Failed:** **0**
- **Skipped:** **0**
- **Execution Time:** **4.28 seconds**
- **Exit Code:** 0

### Process 2 (Independent Fresh Process Repeatability Run)
- **Total Tests Collected:** 43
- **Passed:** **43**
- **Failed:** **0**
- **Skipped:** **0**
- **Execution Time:** **4.53 seconds**
- **Exit Code:** 0

### Test Distribution (43 Total)
- `tests/test_phase4_1_batch1.py`: 20 tests (all passed)
- `tests/test_phase4_simulation.py`: 10 tests (all passed)
- `tests/test_phase3_optimizer.py`: 6 tests (all passed)
- `tests/test_pipeline.py`: 7 tests (all passed)

---

## 10. Engineering Limitations & Known Risks

1. **Schematic-Level Simulation:** All simulations use Level-1 180nm CMOS models ($V_{DD} = 1.8\text{ V}$). These provide functional comparative evaluation and topological sizing guidance. They do **NOT** represent post-layout parasitic extraction, foundry signoff, or physical DRC/LVS.
2. **Phase Margin Scope:** The current phase margin calculation provides a benchmark heuristic. Complex differential transfer function evaluation, continuous phase unwrapping, and multi-crossing ambiguity resolution remain scheduled for **Batch 2**.
3. **Transient Slew Rate Scope:** Transient metrics currently calculate two-point $dV/dt$. Windowed slew rate (20%–80% or 10%–90%) and settling time tolerance bands remain scheduled for **Batch 3**.
4. **Sizing Parameter Mapping:** Netlist generation parameterizes core bias and differential pair dimensions. Multi-stage transistor sizing mapping remains scheduled for **Batch 3**.

