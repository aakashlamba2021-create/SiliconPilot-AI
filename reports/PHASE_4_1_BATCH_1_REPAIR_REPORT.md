# SiliconPilot AI — Phase 4.1 Batch 1 Repair Report
## Simulation Reliability, Lifecycle Recovery & Trust Boundary Hardening

**Date:** 12 November 2026  
**Auditor Reference:** Independent Codex Review (`reports/CODEX_BATCH_1_INDEPENDENT_REVIEW.md`)  
**Workspace:** `SiliconPilot-AI`  
**Execution Environment:** Windows 10/11, Python 3.11.9 (`.venv`), ngspice DLL (KiCad 9.0.7 build, 64-bit)  
**Status:** All 5 Codex-identified test failures resolved; 39/39 repository tests passing repeatedly across fresh processes.

---

## 1. Executive Summary

Following the independent Codex audit of Phase 4.1 Batch 1, which revealed 5 test failures (29 passed, 5 failed in 9.47s) and multiple architectural defects, a comprehensive engineering repair was performed across the simulation, optimization, and feedback curation subsystems.

### Core Defects Identified & Resolved
1. **ngspice Shared-Library Lifecycle State Corruption:** The ngspice C shared library (`ngspice.dll`) corrupted its internal memory state following deliberately failed netlist commands, outputting `"ngspice.dll cannot recover and awaits to be reset or detached"`. Additionally, multiple `NGSpiceRunner` instances repeatedly calling `ngSpice_Init` overwrote the process-wide `SendChar` callback, breaking log capturing for earlier instances.
   - **Resolution:** Introduced a thread-safe singleton `_NGSpiceProcessContext` managing the single OS-level DLL instance. Registered the `ControlledExit` callback to prevent abnormal termination. Implemented dynamic DLL detachment and reloading via Win32 `kernel32.FreeLibrary` and `LoadLibraryW` when ngspice reaches an unrecoverable state, enabling automated reset and continuous execution across tests.
2. **Database Trust Boundary Vulnerability:** The feedback database previously accepted caller-supplied status strings (`simulation_status="Converged"`) and booleans (`verified_for_training=True`), allowing empty, unphysical, or forged records into the ML training corpus.
   - **Resolution:** Implemented an evidence validation engine (`_validate_simulation_evidence`) inside `FeedbackDatabase`. It enforces that the status is `"Converged"`, parameters are present, no requested analysis failed, and `spice_results` contains valid finite electrical measurements (`dc_gain_db`, `power_consumption_mw`, etc.). Caller booleans are automatically overridden if physical evidence is missing. Manual promotion via `mark_as_verified` performs the same evidence validation against stored JSON before promoting records.
3. **Dominated Candidate Simulation (Missing Pareto Rank-1 Filter):** `CandidateOptimizer.run_simulation_validation` took the top $k$ rows of the candidate DataFrame without filtering for Pareto optimality, causing dominated (rank $\ge 2$) candidates to be sent to SPICE if they scored high on composite score.
   - **Resolution:** Enforced `filter_pareto_front=True` before slicing `top_k`. Candidates are filtered strictly for `pareto_rank == 1` (or `is_pareto_optimal == True`), deterministically sorted by `composite_score`, and only then capped at `top_k`.
4. **Metric Completeness & False Convergence:** If analysis data was omitted while the runner reported success, extractor fell through to `"Converged"` with all metrics `None`.
   - **Resolution:** In `SimulationMetricsExtractor.extract`, an explicit requested-analysis contract is enforced. If an analysis was requested (or data provided), its status must be `"SUCCESS"` and contain $\ge 2$ parsed points. If no physical metrics can be extracted, overall status is explicitly marked `"Incomplete: No physical electrical metrics extracted"`, never `"Converged"`.
5. **Power Measurement Voltage Provenance:** Supply voltage was assumed or defaulted without distinguishing whether voltage came from an actual OP simulation vector or a configured supply parameter.
   - **Resolution:** Added `voltage_provenance` tracking (`spice_op_simulation` / `measured_op` when measured from `v(vdd)`, or `calculated_simulated_current_x_configured_vdd` / `configured_supply_parameter` when computed using the configured supply parameter).
6. **Command Reliability & Stale File Prevention:** Initialization, cleanup (`destroy all`, `remcirc`), and output generation were hardened. Output vector files now use `uuid.uuid4().hex` to prevent millisecond timestamp collisions and are verified for non-empty existence and fresh mtimes.

---

## 2. Independent Test Suite Execution Results

The full test suite was executed in two consecutive clean Python processes with no bytecode caching (`-p no:cacheprovider`), mimicking Codex's exact execution environment:

```powershell
$env:NGSPICE_DLL = 'C:\Program Files\KiCad\9.0\bin\ngspice.dll'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PATH = 'C:\Program Files\KiCad\9.0\bin;' + $env:PATH
& '.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests -v
```

### Execution Run 1 (Fresh Process)
- **Status:** **39 passed in 4.41s** (0 failed, 0 warnings, exit code 0)
- **Breakdown:**
  - `tests/test_phase3_optimizer.py`: 6/6 passed
  - `tests/test_phase4_1_batch1.py`: 16/16 passed
  - `tests/test_phase4_simulation.py`: 10/10 passed
  - `tests/test_pipeline.py`: 7/7 passed

### Execution Run 2 (Fresh Process Repeatability Verification)
- **Status:** **39 passed in 4.88s** (0 failed, 0 warnings, exit code 0)
- **Breakdown:** 39/39 passed with identical zero-failure behavior.

### Resolution of Specific Codex Failures
| Failing Test Reported by Codex | Root Cause | Engineering Fix Applied | Verification Result |
|---|---|---|---|
| `test_11_real_ngspice_power_measurement_parity` | DLL session corrupted by earlier test execution | Dynamic DLL lifecycle recovery in `_NGSpiceProcessContext` | **PASSED** |
| `test_03_simulation_execution_ac_and_tran` | ngspice DLL locked in unrecoverable state | Process context singleton + `FreeLibrary`/`LoadLibraryW` reload | **PASSED** |
| `test_04_simulation_parsing` | Stale/corrupted AC simulation vectors from failed state | Output vector UUID isolation + clean analysis commands | **PASSED** |
| `test_05_metric_extraction` | `stderr: ngspice.dll cannot recover and awaits to be reset` | `is_session_healthy()` verification and auto-recovery | **PASSED** |
| `test_08_simulation_in_the_loop_candidate_selection` | Dominated candidate simulated; winner ID mismatch | Pareto rank-1 pre-filtering (`cand_c` rank 2 filtered out) | **PASSED** |

---

## 3. Detailed Component Modifications

### A. ngspice Shared Library Runner (`src/simulation/ngspice_runner.py`)
- **Process Singleton Context (`_NGSpiceProcessContext`):** Because ngspice is a native C library loaded into the process address space with global static buffers, multiple `ngSpice_Init` calls or multiple `NGSpiceRunner` instances corrupt state. The context ensures one shared DLL instance, routes callbacks dynamically to the active runner, and records initialization return codes.
- **Controlled Exit Handling:** Added `ControlledExit` callback registration (`ctypes.CFUNCTYPE(c_int, c_int, c_bool, c_bool, c_int, c_void_p)`) so fatal ngspice exits trigger managed Python recovery instead of abruptly aborting the process.
- **Dynamic Detach & Reload (`_reload_dll`):** When ngspice outputs `"cannot recover and awaits to be reset or detached"`, soft resets fail. Win32 `kernel32.FreeLibrary` detaches the corrupt DLL and `kernel32.LoadLibraryW` reloads a pristine instance.
- **Command Checking & File Hygiene:** Return codes for `source`, `run`, `remcirc`, and `destroy all` are validated. Temporary `.txt` vector files use UUID suffixes and are checked against stale modification times before parsing.

### B. Feedback Database Trust Boundary (`src/simulation/feedback_db.py`)
- **Evidence-Based Ingestion:** Added `_validate_simulation_evidence()`. Ingestion requires:
  1. `simulation_status == "Converged"`.
  2. `parameters` dictionary is non-empty.
  3. `spice_results` contains at least one finite physical electrical metric (`dc_gain_db`, `power_consumption_mw`, `phase_margin_deg`, `unity_gain_bandwidth_hz`, `slew_rate_v_per_us`).
  4. None of the requested analysis statuses in provenance are `"FAILED"`.
- **Caller Override Prevention:** In `record_simulation()`, caller-provided `verified_for_training=True` is forcibly downgraded to `False` (0) if evidence validation fails.
- **Promotion Security:** `mark_as_verified()` reads existing JSON evidence from disk/db and runs full validation before setting `verified_for_training=1`.
- **Timestamp Collision Resolution:** Added UUID suffix (`_uuid.uuid4().hex[:8]`) to `record_id` to prevent SQLite unique constraint violations during rapid consecutive test inserts.

### C. Candidate Optimizer (`src/optimization/candidate_optimizer.py`)
- **Pareto Rank-1 Filtering:** In `run_simulation_validation()`, implemented `filter_pareto_front: bool = True`. Only candidates where `pareto_rank == 1` or `is_pareto_optimal == True` are eligible for SPICE validation.
- **Deterministic Slicing:** Pareto candidates are sorted deterministically by `composite_score` descending before capping at `top_k`.
- **Phase Margin Gating:**
  - Training eligibility in `record_simulation` requires `m.phase_margin_deg >= min_phase_margin` (default 40.0°).
  - Candidate winner selection strictly requires `m.phase_margin_deg >= min_phase_margin` and non-null gain. Dominated or unstable candidates cannot become winners.

### D. Simulation Metrics Extractor (`src/simulation/simulation_metrics.py`)
- **Completeness Enforcement:** `extract()` verifies that if an analysis was requested (or data supplied), its status is `"SUCCESS"` and $\ge 2$ data points were parsed. If no valid physical metrics could be derived, status is set to `"Incomplete: No physical electrical metrics extracted"`, preventing false `"Converged"` status.
- **Power Voltage Provenance:** Added `voltage_provenance` to `ExtractedMetrics`. Power results explicitly indicate whether voltage was directly measured from `v(vdd)` (`spice_op_simulation` / `measured_op`) or supplied via the netlist configured supply parameter (`calculated_simulated_current_x_configured_vdd` / `configured_supply_parameter`). Assumed current fallbacks remain completely removed.

### E. Test Suite Expansion (`tests/test_phase4_1_batch1.py`)
Added 5 comprehensive end-to-end and adversarial test cases (tests 12–16):
- `test_12_feedback_db_boundary_forged_converged_rejection`: Adversarially tests forged converged payloads with empty results, non-finite values, failed analysis statuses, and empty parameters, asserting that neither direct insertion nor manual promotion bypasses validation.
- `test_13_pareto_rank1_filtering_before_simulation`: Verifies with a mocked runner that interleaved rank-1 and rank-2 candidates are filtered such that rank-2 dominated candidates are never submitted to SPICE.
- `test_14_metric_completeness_and_omitted_data_rejection`: Asserts that runs with omitted analysis data or non-success statuses are rejected from `Converged`.
- `test_15_power_voltage_provenance_distinction`: Validates that power extraction properly assigns `voltage_provenance` when `v(vdd)` is absent versus when it is present.
- `test_16_dll_session_health_and_automatic_recovery`: Executes a live deliberately malformed netlist, verifies session health degradation, triggers automated DLL reset/reload, and executes live AC and transient simulations successfully.

---

## 4. Pre-Repair Backups

Before code edits were performed, verbatim copies of all touched source and test files were preserved in:
`backups/phase4_1_batch1_pre_repair/`
- `candidate_optimizer.py`
- `feedback_db.py`
- `ngspice_runner.py`
- `simulation_metrics.py`
- `simulation_parser.py`
- `simulation_validator.py`
- `test_phase4_1_batch1.py`
- `test_phase4_simulation.py`

---

## 5. Scope & Engineering Limitations Disclosure

In accordance with strict analog IC engineering standards:
1. **Schematic-Level Simulation:** All simulations currently performed use schematic netlists with generic Level-1 180nm CMOS models ($V_{DD} = 1.8\text{ V}$). These provide functional topological verification and relative comparative guidance. They do **NOT** represent foundry-calibrated silicon sign-off or post-layout parasitic extraction.
2. **Phase Margin Scope:** The current phase margin calculation provides a benchmark heuristic. It is preserved for Batch 1 contract stability. Full complex differential transfer function calculation, phase unwrapping, and bracketed unity-gain crossing verification remain scoped for **Batch 2**.
3. **Transient Slew Rate Scope:** Transient metrics currently calculate two-point $dV/dt$. Full windowed slew rate (20%–80% or 10%–90%) and settling time tolerance bands remain scoped for **Batch 3**.
4. **Sizing Parameter Mapping:** Netlist generation currently parameterizes core bias and differential pair widths. Complete multi-stage transistor sizing mapping remains scoped for **Batch 3**.

---

## 6. Conclusion & Readiness Assessment

Phase 4.1 Batch 1 repairs have successfully addressed all findings raised in the independent Codex review. The ngspice shared library session is resilient against fatal netlist errors, the feedback database strictly defends its trust boundary against unverified records, candidate selection strictly respects the Pareto front, and the entire test suite passes repeatedly and deterministically in 4.4–4.9 seconds across clean processes.

Batch 1 is now stable, robust, and verified. SiliconPilot AI is fully prepared for Phase 4.1 Batch 2 (AC Analysis & Phase Margin Extraction Repair).

