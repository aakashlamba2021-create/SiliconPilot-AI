# Phase 4.1 Batch 1 — Final Independent Sign-off Review

**Review date:** 2026-11-12  
**Review type:** Independent, read-only code and test audit.  
**Verdict:** **NOT APPROVED**

No source, test, database, model, or existing report files were intentionally edited. The requested final review report is the sole intended project-file addition. The test suite itself wrote temporary test data and application logs as part of execution; its Batch 1 DB fixture uses a per-process file under the OS temp directory and removes it at teardown.

## Test execution

I executed the full suite twice in separate fresh processes from the project root, using the requested venv interpreter and KiCad ngspice DLL. Each process was started with `NGSPICE_DLL=C:\Program Files\KiCad\9.0\bin\ngspice.dll`, the KiCad `bin` folder added to that process's PATH, `PYTHONDONTWRITEBYTECODE=1`, and pytest cache disabled.

**Run 1:** Pytest collected 39 tests; **33 passed, 6 failed in 5.06 s**.  
**Run 2:** Pytest collected 39 tests; **33 passed, 6 failed in 5.12 s**.

The same six tests failed in both runs:

1. `tests/test_phase4_1_batch1.py::TestPhase41Batch1::test_11_real_ngspice_power_measurement_parity` — `res.success` was false.
2. `tests/test_phase4_1_batch1.py::TestPhase41Batch1::test_16_dll_session_health_and_automatic_recovery` — recovery/live follow-up assertion failed.
3. `tests/test_phase4_simulation.py::TestPhase4Simulation::test_03_simulation_execution_ac_and_tran` — live simulation execution failed.
4. `tests/test_phase4_simulation.py::TestPhase4Simulation::test_04_simulation_parsing` — AC parsed output was invalid.
5. `tests/test_phase4_simulation.py::TestPhase4Simulation::test_05_metric_extraction` — metrics status reported `Failed: stderr Error: ngspice.dll cannot recover and awaits to be reset or detached`.
6. `tests/test_phase4_simulation.py::TestPhase4Simulation::test_08_simulation_in_the_loop_candidate_selection` — `winner_id` was `None`.

Thus the reported two fresh-process 39/39 passes were not reproduced. Reproducible DLL recovery is still a release-blocking issue for Batch 2.

## Changed-file inspection and diff availability

There is no `.git` directory in the project or its parent, so `git status`/`git diff` cannot provide a repository diff. The repair report says pre-repair copies were saved. I compared those backups to current files by SHA-256. Files that differ from the reported backups are:

- `src/optimization/candidate_optimizer.py`
- `src/simulation/feedback_db.py`
- `src/simulation/ngspice_runner.py`
- `src/simulation/simulation_metrics.py`
- `tests/test_phase4_1_batch1.py`
- `tests/test_phase4_simulation.py`

`src/simulation/simulation_parser.py` and `src/simulation/simulation_validator.py` match their listed backups byte-for-byte. I inspected the current changed files and the new Batch 1 tests directly.

## Findings

### CRITICAL — The live ngspice session is not reliably recovered; both fresh-process suite runs reproduce six failures

**Evidence:** The exact test results above. The tests emit repeated “Detaching and reloading ngspice DLL” messages, yet subsequent real simulations still fail and logs include ngspice's “cannot recover and awaits to be reset or detached” error. The newly added `test_16_dll_session_health_and_automatic_recovery` itself fails. Therefore the repair's core lifecycle guarantee is not met on the stated KiCad 9 DLL and Python 3.11.9 environment.

**Impact:** Simulation results are unreliable after malformed input; the engine can remain unusable across the same process, invalidating the claimed repeated-run/recovery capability and the Batch 2 foundation.

### HIGH — `ControlledExit` callback has the wrong ABI signature

**Source:** `src/simulation/ngspice_runner.py:155-156,182-189`.

**Evidence:** The implementation defines `ExitCb = CFUNCTYPE(c_int, c_int, c_bool, c_bool, c_void_p)` and `on_exit(status, immediate, quitexit, user_data)`: four arguments. ngspice's shared API declares `ControlledExit(int, bool, bool, int, void*)`: five arguments, including the library identification integer before `userData`. The callback therefore interprets `ident` as `user_data` and drops the actual fifth argument. The API documentation also states that `ControlledExit` is required. The callback's return type and `CFUNCTYPE` C calling convention are otherwise appropriate for this C callback interface.

**Impact:** The callback pointer does not match the native function type on the fatal-exit path. On this 64-bit build the register calling convention may tolerate an ignored trailing argument, but this is still an ABI defect in the exact callback used for recovery; it cannot be signed off as memory-safe across supported Windows builds. The callback should match all five native arguments exactly and be tested through a controlled exit.

**Reference:** [ngspice shared API header](https://github.com/ngspice/ngspice/blob/master/src/include/ngspice/sharedspice.h#L1125-L1139), [ngspice manual callback/API definitions](https://ngspice.sourceforge.io/docs/ngspice-45-manual.pdf), [ctypes calling conventions and library handles](https://docs.python.org/3/library/ctypes.html#loading-shared-libraries).

### HIGH — Process-wide singleton and DLL reload are not synchronized; concurrent use can race unload against active calls

**Source:** `src/simulation/ngspice_runner.py:66-80,131-147,176-190,209-231,278-290`.

**Evidence:** `_NGSpiceProcessContext` is a set of unsynchronized class attributes. There is no lock around initialization, `_exec_cmd`, callback-buffer routing, health transitions, or `_reload_dll`. `ctypes.CDLL` calls release the Python GIL while executing native functions; the Python ctypes documentation explicitly documents this. Thus two threads can call the shared library concurrently, overwrite `active_log_buffer`, or one thread can invoke `FreeLibrary` while another is executing a function or callback in that module.

**Impact:** The current manual unload strategy can create a dangling function pointer/callback or unload code while executing, which can crash the process or corrupt memory. The serial test suite does not rule this out. This is the material memory-safety risk; I do not find evidence of a deterministic double-`FreeLibrary` in the single-threaded path described below.

### MEDIUM — Serial `CDLL`/Win32 handle ownership is plausibly balanced, but unload safety depends on exclusive access

**Source:** `src/simulation/ngspice_runner.py:131-147,209-231`.

**Evidence:** On Windows `_load_dll` obtains one module reference with `LoadLibraryW`, then wraps that existing handle with `ctypes.CDLL(path, handle=handle)`. CPython's `CDLL.__init__` sets `_handle` to the supplied value without another `LoadLibrary`, and `CDLL` has no Python-level destructor that calls `FreeLibrary`. `_reload_dll` explicitly calls `FreeLibrary` once for the explicit load reference, clears the shared wrapper, then calls `LoadLibraryW` again. So in the ordinary single-threaded lifecycle I see no automatic second unload by `CDLL`; this does not itself demonstrate double unloading.

**Remaining risk:** `FreeLibrary`'s return value is ignored and the handle is discarded even if the call fails. The code also does not prove there are no outside function-pointer references/in-flight calls before unload. Without synchronization and ownership assertions, the more serious concurrent dangling-reference risk remains. `_load_dll` error paths can also retain a loaded handle when symbol setup/init fails.

### HIGH — Database still accepts forged records as verified if they contain one finite metric and caller-supplied success evidence

**Source:** `src/simulation/feedback_db.py:69-126,128-192,204-262`.

**Evidence:** `_validate_simulation_evidence` checks for exactly `Converged`, nonempty parameters, and at least one finite primary metric. Analysis statuses are optional; if present, the function rejects `FAILED` and requires only one `SUCCESS`, not all requested analyses. It does not require `metric_provenance`, simulator identity/version, a netlist hash, a complete analysis contract, or evidence that data came from ngspice. The test's “genuine complete record” at `tests/test_phase4_1_batch1.py:319-332` is itself a hand-constructed dictionary with two numeric values and caller-supplied `SUCCESS` statuses, yet it is accepted as “genuine.”

**Impact:** A forged `Converged` payload with a single finite value and fabricated or omitted status metadata can still be marked verified for training. This is a trust-boundary failure despite rejection of empty/NaN/explicitly failed examples.

### HIGH — Pareto filtering is rank 1 only on the normal path, but the selector fails open when rank 1 is absent

**Source:** `src/optimization/candidate_optimizer.py:584-605`.

**Evidence:** When rank-1 rows exist, the code filters before `.head(top_k)`, which correctly excludes dominated rows. But if `pareto_rank` exists and no row has rank 1, it falls back to rows with the minimum rank (which may be rank 2 or higher). If only `is_pareto_optimal` exists and no row is true, it falls back to the entire original DataFrame. If neither field exists, it also simulates all rows. The public `filter_pareto_front=False` option can bypass filtering by explicit caller choice.

**Impact:** The report's absolute claim that only rank-1 candidates are eligible is too strong. Invalid/missing front metadata can still send dominated/unverified rows to SPICE. Fail closed with an empty selection/error when rank 1 is required but unavailable, or explicitly document a non-Pareto mode.

### MEDIUM — Voltage provenance now distinguishes a configured supply from a measured OP voltage

**Source:** `src/simulation/simulation_metrics.py:208-223,331-344,363-374`.

**Evidence:** When `v(vdd)` is absent, power can use `vdd_override`, but returned provenance is `calculated_simulated_current_x_configured_vdd`, and `voltage_provenance` becomes `configured_supply_parameter`. When the OP voltage exists and is finite/positive, it labels it `measured_op`. This corrects the earlier ambiguous provenance. The calculation still uses configured voltage by default via `extract(vdd=1.8)` where no OP vector exists; consumers must preserve the provenance field and must not label all `power_consumption_mw` as fully OP-measured.

**Assessment:** No assumed-current or fixed-power fallback is apparent in the current extraction code. Supply current is obtained from `i(vdd)`, uses `i_delivered = -i_source`, and units are `W`, `mW=W*1000`, `µA=A*1e6`. The Batch 1 live power test fails, so that sign/measurement path is not independently verified by this run.

### MEDIUM — UUID file names greatly reduce stale-file reuse, but the claimed mtime freshness check is absent

**Source:** `src/simulation/ngspice_runner.py:335-350,523-553`.

**Evidence:** Output names include PID plus `uuid.uuid4().hex`, and any pre-existing target is removed before the run. This makes accidental stale collision extremely unlikely. However, the repair report says output files are checked for fresh mtimes; there is no mtime check in the runner. If removal fails, the exception is swallowed; runner success then relies on nonempty path existence and does not parse/validate the just-written file before marking that analysis successful.

**Impact:** UUID names are a strong practical improvement, not a proof of freshness/data validity. Use exclusive file creation or verify creation time and validate parsed vectors before marking analysis `SUCCESS`.

### MEDIUM — `Converged` means requested analyses returned parsable point counts, not that all expected physical metrics were valid

**Source:** `src/simulation/simulation_metrics.py:89-152,194-198`; `src/simulation/simulation_parser.py:76-89,92-147`.

**Evidence:** The extractor now rejects omitted/failed requested analyses and prevents the all-metrics-`None` case from remaining `Converged`. This is useful. But it does not define a required metric set per requested-analysis contract. The AC parser accepts a row with only frequency and gain and sets missing phase to numeric `0.0` (`simulation_parser.py:79-88`); it then treats multirow finite data as valid. The unchanged Batch 0 AC extractor can produce a numeric PM and endpoint GBW without a bracketed 0 dB crossing (`simulation_metrics.py` `_extract_ac_metrics`). Thus required phase/PM or a genuine unity crossing may be absent while metrics/status look populated. A nonempty AC/TRAN analysis is not sufficient evidence that every downstream physical metric is valid.

**Impact:** Missing phase or a non-crossing response may become plausible numeric metrics rather than unavailable values; the candidate PM gate can consume this value. The planned Batch 2 correction is still required before PM≥40° can represent a reliable stability gate.

### HIGH — Batch 2 phase-margin correction remains outstanding

**Source:** `src/simulation/simulation_metrics.py`, `_extract_ac_metrics` (unchanged Batch 0 algorithm); confirmed by `reports/PHASE_4_1_BATCH_1_REPAIR_REPORT.md` §5.2.

**Evidence:** The Phase 4.1 report explicitly scopes complex differential transfer, phase unwrapping, and bracketed unity crossing to Batch 2. Current code still uses output phase interpolation without unwrapping and reports the sweep endpoint when no unity crossing exists. The optimizer now uses `m.phase_margin_deg >= min_phase_margin` (`candidate_optimizer.py:676-682,721-728`), but that gate is applied to the uncorrected PM extraction.

**Impact:** Stability decisions and training eligibility based on PM are provisional and must not be treated as engineering sign-off. This is a required Batch 2 correction, not a Batch 1 completion claim.

## Claim-by-claim disposition

| Gemini claim | Independent result |
|---|---|
| 39/39 passed twice | **Not reproduced.** Both independent runs: 33 passed, 6 failed; exact failures above. |
| Five previous failures fixed | **False for this environment/run.** The earlier live simulation/power/candidate-selection failures recur; new recovery test also fails. |
| Five new adversarial/recovery tests added | **True.** Tests 12–16 are present. They improve coverage, but do not prove native ABI correctness or concurrent lifecycle safety; test 16 fails live. |
| Singleton lifecycle, manual unload/reload is safe | **Not established.** Serial explicit handle ownership appears balanced, but callback ABI is wrong and no lock prevents concurrent use/unload. |
| Fresh UUID files and mtime verification | **Partly true.** UUID paths and pre-run deletion are present; no freshness mtime check exists. |
| Forged/incomplete DB records rejected | **Partly true.** Empty, nonfinite, and explicit failed-status examples are rejected; omitted/fabricated provenance and partial-success records can pass. |
| Rank-1 Pareto filtering before simulation | **True only when rank-1 metadata is present.** Fail-open fallback remains when the front is empty or unavailable. |
| Supply-voltage provenance fixed | **True.** Configured supply voltage is explicitly labeled as configured; preserve this distinction in consumers. |
| Missing required metrics cannot be Converged | **Partly true.** Omitted analyses and all-None metrics are rejected, but absent AC phase is parsed as 0° and PM extraction can invent endpoint metrics. |

## Verdict and unresolved high/critical items

**NOT APPROVED.** Do not begin Batch 2 on this implementation as a stable foundation yet.

Unresolved **CRITICAL/HIGH** items requiring correction before approval:

1. **CRITICAL:** Both independent suite runs reproduce six failures, including the new recovery test and five live ngspice-dependent failures. Fix and rerun twice until both runs are clean.
2. **HIGH:** Correct the `ControlledExit` callback to the exact five-argument native ABI and add a controlled-exit test.
3. **HIGH:** Serialize all access to the process singleton and prevent DLL unload/reload while any native call/callback is in flight. Add explicit handle/ref ownership and check Win32 return codes.
4. **HIGH:** Make feedback DB training eligibility require genuine, complete, provenance-backed simulator evidence; optional/omitted analysis statuses and one fabricated metric must not qualify.
5. **HIGH:** Make Pareto filtering fail closed when rank-1 metadata/front is absent or empty; no rank>1 fallback when strict filtering is requested.
6. **HIGH:** Complete Batch 2 AC transfer/phase-unwrapping/bracketed-crossing work before interpreting the new 40° gate as stability evidence.

No electrical simulation here establishes physical layout verification, DRC/LVS, foundry accuracy, or silicon correlation.
