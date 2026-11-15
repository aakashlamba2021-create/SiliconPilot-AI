# SiliconPilot AI — Batch 1.2 Final Independent Audit

**Audit date:** 12 November 2026  
**Scope:** Read-only review of current source, Batch 1.2 report, test changes and two independent test executions.  
**Verdict: NOT APPROVED**

## Executive result

The reported 43/43 result was not reproducible. I ran the requested pytest command twice in separate `.venv` Python processes against KiCad 9's actual ngspice DLL. Both runs collected **39 tests**, with **33 passed, 6 failed, 0 skipped**. The same six tests failed in both runs. The native reset symbol is exported by the loaded DLL, and the five-argument `ControlledExit` callback shape matches ngspice's public ABI, but the core recovery path demonstrably does not recover from the malformed-netlist scenario. The implementation is therefore not ready to proceed as a reliable simulation-assisted engineering prototype.

## Test execution

For each run, the following commands were executed in a new PowerShell process (the second invocation was a distinct fresh Python process):

```powershell
$env:NGSPICE_DLL = 'C:\Program Files\KiCad\9.0\bin\ngspice.dll'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PATH = 'C:\Program Files\KiCad\9.0\bin;' + $env:PATH
& '.\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests -v
```

| Run | Collected | Passed | Failed | Skipped | Pytest duration |
|---|---:|---:|---:|---:|---:|
| 1 | 39 | 33 | 6 | 0 | 5.06 s |
| 2 | 39 | 33 | 6 | 0 | 5.12 s |

Both runs failed the same tests:

- `tests/test_phase4_1_batch1.py::TestPhase41Batch1::test_11_real_ngspice_power_measurement_parity`
- `tests/test_phase4_1_batch1.py::TestPhase41Batch1::test_16_dll_session_health_and_automatic_recovery`
- `tests/test_phase4_simulation.py::TestPhase4Simulation::test_03_simulation_execution_ac_and_tran`
- `tests/test_phase4_simulation.py::TestPhase4Simulation::test_04_simulation_parsing`
- `tests/test_phase4_simulation.py::TestPhase4Simulation::test_05_metric_extraction`
- `tests/test_phase4_simulation.py::TestPhase4Simulation::test_08_simulation_in_the_loop_candidate_selection`

The failure output included `ngspice.dll cannot recover and awaits to be reset or detached`, failed live simulation success assertions, and a missing validated winner. No test was skipped. These are live integration tests: the suite's setup constructs the real `NGSpiceRunner`, and the requested DLL path was supplied. The test named `test_11_real_ngspice_power_measurement_parity` is explicitly a live parity check. Results above are my executions, not Gemini's report.

## Gemini claims checked

| Claim | Independent finding |
|---|---|
| 43 tests passed twice | **Contradicted.** Two requested-command runs each collected 39 and had the same six failures. |
| All six prior failures fixed | **False.** The six named in the previous sign-off are the six reproduced failures in both runs. |
| Five-argument `ControlledExit` | **Verified at the Python declaration level.** `CFUNCTYPE(c_int, c_int, c_bool, c_bool, c_int, c_void_p)` and the Python callback both have five arguments. This matches ngspice sharedspice.h's `ControlledExit(int, bool, bool, int, void *)` callback type. The callback is retained in the process context. This validates signature shape, not all platform ABI behavior. |
| `ngSpice_Reset()` exists in loaded DLL | **Verified.** In a separate process loading the exact KiCad path, `hasattr` returned `True` for `ngSpice_Reset`, `ngSpice_Init`, and `ngSpice_Command`. Existence does not prove reset/re-init recovers a session; the live recovery test fails. |
| RLock, active call tracking, reset recovery | **Partially verified; reliability claim contradicted.** The source uses a process-global `RLock`, increments/decrements `active_calls` around `ngSpice_Command`, and tries Reset then Init before manual reload. Live recovery still fails. See findings below. |
| Feedback validation strengthened | **Partially verified, not a secure evidence boundary.** It rejects several malformed inputs but does not require simulator identity/version or analysis statuses, does not actually require a non-empty circuit/topology at this internal method boundary, and accepts non-numeric parameter values because conversion exceptions are ignored. |
| Pareto filtering fails closed | **Verified with a qualification.** `run_simulation_validation` selects rank 1 before slicing and simulating, and returns empty on absent metadata or empty rank 1. If both `pareto_rank` and `is_pareto_optimal` exist, only `pareto_rank` is consulted; an inconsistent rank-2 row marked `is_pareto_optimal=True` is excluded, but a rank-1 row that is otherwise dominated due to forged/stale metadata will still be simulated. The selector trusts upstream rank metadata rather than recomputing dominance. |
| Invalid AC cannot pass PM gate | **Partially verified.** Missing crossing and missing phase return `None` in the helper and the winner list requires non-None PM. However, the reported AC test checks only helper outputs; it does not exercise winner selection with missing PM. Full loop is indirectly constrained by the metrics extractor, though current simulation failures prevent a live end-to-end pass. |

ABI source reference: [ngspice `sharedspice.h`](https://github.com/ngspice/ngspice/blob/master/src/include/ngspice/sharedspice.h) (`ControlledExit` typedef and `ngSpice_Init` declaration). The project report identifies its target library as ngspice-46; the DLL exports the expected reset entry point as independently checked above.

## Findings

### HIGH — Native session recovery does not recover in the tested malformed-netlist path

**Evidence:** `src/simulation/ngspice_runner.py`, `reset_session`, `_reload_dll`, `_exec_cmd`, and `_run_via_dll`. The DLL exports `ngSpice_Reset`, and `reset_session()` invokes it and re-registers callbacks. Yet `test_16_dll_session_health_and_automatic_recovery` fails, followed by the real power test and legacy live simulation tests. Captured errors state that the DLL cannot recover and awaits reset/detach. This directly contradicts the claim that the six failures were resolved.

The context lock is held around each command and around reset/reload; `active_calls` is incremented while the same lock is held and decremented in `finally`. This serializes entry through this wrapper, but it is not independent proof that ngspice has no asynchronous native callback or worker activity at unload time. `_reload_dll` clears callback references and the CDLL object before `FreeLibrary`; safety therefore relies on the lock and native call returning synchronously. A callback arriving from a separate native thread could contend on the Python lock while the calling native operation waits for it. No concurrency stress test covers this boundary.

`_reload_dll` also sets `dll_handle = None` even if `FreeLibrary` returns 0 or raises, losing ownership tracking while proceeding to reload. That is a resource/ownership defect on the failure path. The ordinary `ctypes.CDLL(path, handle=handle)` wrapper shares the explicitly loaded Windows handle; dropping the Python wrapper is not by itself proof that callbacks or other native references have ceased.

**Required before next phase:** Reproduce and fix recovery on the actual DLL; preserve ownership on failed unload; prove reset/re-init and fallback reload in fresh-process live tests, including repeated runs and a malformed-netlist recovery sequence. Add a concurrency/unload test or explicitly constrain and enforce single-thread use including callback completion.

### HIGH — Verified feedback eligibility can be forged without simulator identity or analysis evidence

**Evidence:** `src/simulation/feedback_db.py::_validate_simulation_evidence` and `mark_as_verified`. Simulator identity and version are checked only when present. `analysis_statuses` is rejected only when a non-empty dictionary is present; absent statuses pass. The validator docstring says `circuit_id` and `topology` must be non-empty, but checks only invalid non-None values, so `None` passes at this internal method boundary. In the sizing loop, failed float conversion is caught with `pass`, so values such as `{"W": "not-a-number"}` can pass. A caller can therefore insert a `Converged` record with a finite metric, plausible strings, parameters and no simulator/analysis provenance, set the caller flag, and be eligible for training.

The method verifies schema-like claims, not provenance authenticity: a caller-supplied `simulator_identity="ngspice"`, version `"46"`, and `SUCCESS` status are not tied to a runner-generated evidence token, netlist hash or immutable simulation result. The tests' “valid evidence” is constructed as dictionaries, not cryptographically or structurally bound to a live run. The new “missing simulator identity” case tests an explicit empty string, not omitted identity, and does not test absent analysis status or nonnumeric parameter values.

**Required before next phase:** Require identity/version, circuit/netlist provenance and all required analysis statuses; reject every nonnumeric sizing parameter; ensure only runner-produced successful results can be promoted (e.g. persist verifiable netlist/simulation evidence and validate all requested analysis outputs).

### HIGH — Full live simulation pipeline remains unusable after errors

**Evidence:** Repeated suite failures in `test_03`, `test_04`, `test_05`, and `test_08`. The `run_netlist` tests report `SimulationResult.success=False`; parsing consequently lacks valid AC vectors, metric extraction surfaces the unrecovered-state diagnostic, and optimizer has no stable simulated candidate to select. Since all repeat in a fresh Python process and under the actual DLL, this is not merely stale state from a prior pytest process.

### MEDIUM — Reset test and four new tests are not sufficient proof of claimed reliability

`test_17_controlled_exit_callback_5arg_abi` invokes the Python callback object directly with five arguments. This verifies Python callable shape, not a native DLL-originated callback using the sharedspice ABI. `test_18` and `test_19` check fail-closed returns but do not instrument the runner to prove zero native invocations; implementation inspection does show the early return precedes the loop. `test_20` uses synthetic arrays and checks missing-crossing/missing-phase helper results; it does not test an actual parsed file with the phase column absent, nor candidate selection with absent PM. These are useful focused unit tests but not end-to-end validation.

The rank-1 simulation test calls the real optimizer path but does not spy on the runner; it infers simulated IDs from returned `evaluated_candidates`. The loop's source slice confirms the filtering is before simulation. It does not independently establish that rank metadata is correct/non-forgeable.

### MEDIUM — Power and operating-point validation still needs tighter edge handling

**Evidence:** `src/simulation/simulation_metrics.py::_extract_power_metrics`. The source uses the expected SPICE sign convention (`-i(VDD)`) and converts W to mW by multiplying by 1000. The `test_15_power_voltage_provenance_distinction` checks measured node voltage versus configured override and expected power. But negative delivered current (source absorbing power) is converted into negative power without being rejected here, and the feedback validator rejects nonpositive power only if subsequently promoted. `test_11_real_ngspice_power_measurement_parity` is the actual live validation and failed. Therefore the unit/sign implementation is inspectable, but live parity is not established by this audit.

### LOW — Gemini's test total is inconsistent with the current checkout

The current collection is 39 tests, not 43. `tests/test_phase4_1_batch1.py` contains tests numbered 01–20; four new tests are numbered 17–20. The Batch 1.2 report claims 20 tests in that file and 43 repository tests. The independent pytest collection is authoritative for this checkout. Backup comparisons show additions in the two test files; no tests were skipped in execution. There is no Git repository metadata available in this workspace to produce a canonical `git diff`, so file-level conclusions are based on source and the supplied pre-repair backups where present.

## Six prior failures and four newly added tests

**Original six failures:** Not fixed in practice: all six reappeared in both independent runs. The callback signature change is present and correct; the recovery correction is not effective on the malformed-netlist path, and downstream live tests remain broken.

**Four new tests (17–20):** They cover callback invocation shape, missing Pareto metadata, empty rank-1 filtering, and helper-level AC no-crossing/no-phase behavior. They do not weaken a test by skipping it; no skips occurred. The Pareto and AC tests use synthetic data. Test 17 calls a callback from Python rather than triggering it from native code. These tests support local branch behavior but do not establish native lifecycle reliability or authentic evidence provenance.

## Overall assessment

The current implementation is **not safe for the next development phase** as a simulation-assisted analog IC engineering prototype. The callback signature and Pareto prefilter changes are real, and missing phase/crossing handling is improved. However, the actual KiCad ngspice session fails recovery and causes six recurring integration failures, while the feedback database still permits verification of caller-constructed evidence missing mandatory simulator and analysis provenance. Batch 2 should wait until the HIGH findings are corrected and independently verified. This verdict does not make any claim about physical layout verification or foundry accuracy.
