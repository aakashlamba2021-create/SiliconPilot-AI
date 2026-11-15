# Final Independent Audit — SiliconPilot AI Batch 1.2

**Audit date:** 12 November 2026  
**Scope:** Current source, Gemini's final reliability report, test integrity, and one fresh full-suite run.  
**Verdict: NOT APPROVED**

## Summary

The process-isolated worker is genuinely implemented and the configured suite now collects 43 tests. However, the requested run failed the same six live simulation/recovery tests that failed in the prior audit. The worker design has useful isolation properties, but resource cleanup and error handling are incomplete. Feedback validation still permits caller-constructed records with omitted simulator and analysis provenance to be marked verified. Phase and crossing safeguards are improved, though the phase-margin calculation and power extraction still have engineering limitations. Batch 2 should not start until the HIGH findings are corrected and independently verified.

## Final test run

I ran the requested command once using the project `.venv`, with the KiCad 9 DLL configured in the environment. Pytest used the real Windows Python 3.11.9 interpreter and collected **43 items**:

- **Passed:** 37
- **Failed:** 6
- **Skipped:** 0
- **Errors:** 0
- **Pytest duration:** 21.82 seconds

The six failures were:

1. `tests/test_phase4_1_batch1.py::TestPhase41Batch1::test_11_real_ngspice_power_measurement_parity`
2. `tests/test_phase4_1_batch1.py::TestPhase41Batch1::test_16_dll_session_health_and_automatic_recovery`
3. `tests/test_phase4_simulation.py::TestPhase4Simulation::test_03_simulation_execution_ac_and_tran`
4. `tests/test_phase4_simulation.py::TestPhase4Simulation::test_04_simulation_parsing`
5. `tests/test_phase4_simulation.py::TestPhase4Simulation::test_05_metric_extraction`
6. `tests/test_phase4_simulation.py::TestPhase4Simulation::test_08_simulation_in_the_loop_candidate_selection`

The recovery test's subsequent valid simulation returned `ngspice.dll cannot recover and awaits to be reset or detached`. AC execution and parsing failed downstream, metric extraction was marked failed, the live power test returned unsuccessful, and the optimizer selected no winner. This run does not verify Gemini's 43/43 result.

## Assessment of Gemini's claims

### Worker process isolation — implemented, not yet safe/reliable enough

`src/simulation/ngspice_runner.py` defines top-level `_ngspice_worker_loop`, starts it with `multiprocessing.Process`, and passes requests/results over `multiprocessing.Pipe`. The worker loads `self.dll_path` itself, which is the KiCad DLL path configured in the parent. The parent serializes requests per runner with `_worker_lock`; `poll(timeout=30.0)` handles a stalled response, while EOF/broken-pipe exceptions are converted to a failed result and trigger worker termination. Fatal source/run paths return `must_restart=True`; the parent terminates the worker and can spawn a new one on the next call. This is a real OS-process boundary, not a mock.

But the live malformed-netlist recovery test fails, so successful recovery is disproven by this run. Additional lifecycle concerns in `_ensure_worker`, `_terminate_worker_locked`, and `_ngspice_worker_loop`:

- The parent does not close its local `child_conn` endpoint after `p.start()`. Initialization failure/timeout paths also leave pipe endpoints and the `Process` object without explicit close/reap cleanup.
- `_terminate_worker_locked` waits one second after `terminate()` and then drops the process reference without verifying that the process exited or escalating/closing its handle. This does not establish that every failed worker is reaped before replacement.
- Job processing has no broad per-request exception handler in the child. A Python exception can kill the child; the parent detects this eventually through EOF/broken pipe, but only after the IPC failure path.
- A 30-second poll detects a missing reply but there is no independent worker health check or request identifier. The per-instance lock prevents request interleaving on one runner, but separate runner instances have separate worker/pipe state.
- The parent still loads and initializes another ngspice DLL in `_init_backend` / `_load_dll`, although simulations are routed to the worker. Parent `reset_session()` also calls reset/re-init on that parent copy. This is redundant state and muddies which engine health the exposed `is_session_healthy()` represents.

**Finding — HIGH:** Process isolation is the right direction, but current recovery fails under the actual suite and teardown does not prove complete worker/pipe cleanup. It is not ready to be treated as a reliable isolation boundary.

### Native callback

Both the worker and parent declare `ControlledExit` with five arguments: `int, bool, bool, int, void*`, returning `int`, using `ctypes.CFUNCTYPE`. That matches the public ngspice shared library header's callback typedef and callback parameter in `ngSpice_Init`: `ControlledExit(int, bool, bool, int, void*)`. In the worker, local `cb1` and `cb2` references remain in scope for the full worker loop; in the parent, callback references are retained on `_NGSpiceProcessContext`.

The regression test calls the Python callback object directly with five arguments. It does not induce a callback from the actual native DLL, so it tests Python-callable shape but not native invocation/calling-convention integration. The type/signature is consistent with the public API; live callback ABI behavior remains unproven.

### Feedback database evidence — forged or incomplete records can still be promoted

The current `src/simulation/feedback_db.py::_validate_simulation_evidence` does not match Gemini's report:

- `circuit_id` and `topology` are checked only if non-`None`; omitted values pass this validator.
- Simulator identity and version are optional. They are validated only if supplied; arbitrary non-empty identity is not restricted to `ngspice`.
- Analysis statuses are optional. If absent or empty, the validator does not reject the record.
- Parameter float conversion failures are swallowed with `pass`, so a value like `{"W": "not-a-number"}` is not rejected by that loop. It also does not enforce positive sizing values.
- There is no required netlist hash or trusted runner-produced provenance binding. A caller can construct a plausible metrics dictionary and status strings itself.
- Voltage provenance consistency and general metric bounds described in Gemini's report are not present in this validator. The visible physical checks principally bound positive power and DC gain; other fields are finite-checked but not comprehensively range-validated.

`record_simulation` does prevent its boolean from bypassing the validator, and `mark_as_verified` revalidates the stored JSON. However, because required evidence fields are optional, a caller can supply a finite in-range metric, a non-empty parameter dictionary containing nonnumeric content, and `Converged` status without simulator identity/version or analysis statuses, and request verification. The record is not proven genuine simply because it satisfies those caller-controlled fields.

**Finding — HIGH:** Mandatory evidence/provenance enforcement claimed by Gemini is not implemented. Verified training records can still be forged or lack required provenance.

### Pareto filtering — early filter exists, rank metadata is trusted

`CandidateOptimizer.run_simulation_validation` filters for `pareto_rank == 1` (or, only if that column is absent, `is_pareto_optimal == True`) before `head(top_k)` and before entering the simulation loop. Missing metadata and an empty rank-1 subset return early with no candidates. This verifies the basic rank-1 prefilter and fail-closed branches.

The selector does not recompute dominance. If a dominated candidate is labeled `pareto_rank == 1`, it will be simulated. Where both metadata columns exist, `pareto_rank` takes precedence, even if the boolean conflicts. Where only the boolean exists, it is trusted directly. Therefore dominated candidates cannot bypass *correct* rank metadata, but inconsistent or forged rank metadata can bypass the intended strict Pareto contract.

**Finding — MEDIUM:** Recompute the front from objective values or validate rank metadata before simulation; do not treat caller-supplied rank as proof of nondominance.

### AC metrics and stability gating

`simulation_parser.py` sets `has_phase=False` and leaves `phase_deg` empty when the parsed rows lack a phase column. `_extract_ac_metrics` returns `None` for GBW and PM when it does not find a bracketed first 0 dB crossing; it returns `None` PM when phase is absent. Winner qualification in `candidate_optimizer.py` requires `spice_phase_margin is not None` and at least the configured threshold, so `None` cannot pass the 40-degree gate.

Limitations remain. The PM calculation linearly interpolates the supplied phase values and computes `max(0, 180 - abs(phase_dc - phase_at_unity))`; it does not unwrap phase or compute a complex differential transfer function. Wrapped, non-monotonic, or otherwise invalid phase vectors may still yield a plausible numeric PM and pass the stability gate. The new test only checks helper outputs using synthetic arrays; it does not validate a real missing-phase file or exercise a missing-PM candidate through the selector.

**Finding — MEDIUM:** Missing phase/crossing fail closed, but a numeric phase margin is not independently validated as an engineering-valid measurement. The report itself acknowledges full complex transfer/phase handling is deferred.

### Power measurement

The extractor uses the right nominal formula, `P = VDD * (-I(VDD))`, and multiplies watts by 1000 for mW. It uses an operating-point voltage when present and labels configured-voltage multiplication as calculated provenance otherwise. No assumed current fallback is visible in `_extract_power_metrics`.

The implementation does **not** reject `i_source >= 0`; a source absorbing current produces a negative “power consumption” value and a negative delivered current rather than unavailable/failed measurement. Also, `test_11_real_ngspice_power_measurement_parity` failed in the live run, so the claimed 0.1847 mW parity was not reproduced. The displayed 0.1847 mW claim in Gemini's report is not independent evidence.

**Finding — MEDIUM:** Reject wrong-sign/negative delivered power as invalid and establish parity with a successful live run before asserting measured power.

## Test integrity and baseline

The supplied `backups/phase4_1_batch1_2_pre_repair` contains the earlier tests. It has 16 tests in `test_phase4_1_batch1.py`, 10 in `test_phase4_simulation.py`, 6 in `test_phase3_optimizer.py`, and 7 in `test_pipeline.py`, totaling the prior **39-test baseline**. The current suite retains those 39 test cases and adds four methods (`test_17`–`test_20`) in `test_phase4_1_batch1.py`, yielding 43 collected tests.

Comparison of the pre-repair test backup with current `test_phase4_1_batch1.py` shows the original valid-evidence fixture was updated to include simulator identity/version, and four new test methods were added. The changed fixture strengthens its data rather than weakening an assertion. The four additions cover callback invocation shape, missing Pareto metadata, empty rank-1 front, and AC crossing/phase helper behavior. They are useful unit/regression checks but do not prove the claimed native callback invocation, worker failure recovery, stale handle/pipe cleanup, authentic database provenance, or end-to-end missing-PM gating. No test was skipped in the execution requested here; the two skip branches are conditional on ngspice absence, and the DLL was available so they did not skip.

The six prior failures are **not fixed**: all six failed again in this one fresh-process execution.

## Remaining limitations and demo readiness

- **HIGH:** Worker recovery fails in the live test suite and worker teardown does not prove all process/pipe handles are closed.
- **HIGH:** The feedback database does not require simulator identity/version or analysis statuses and accepts nonnumeric parameter values in the sizing check; caller-supplied evidence remains forgeable.
- **MEDIUM:** Pareto filtering trusts rank metadata and can simulate dominated candidates mislabeled rank 1.
- **MEDIUM:** AC phase-margin extraction can return plausible values from phase data without phase-unwrapping/transfer validity checks.
- **MEDIUM:** Power extractor accepts a non-delivering source current; live parity test fails.

The app may be shown as a UI/software demo if clearly labeled as a prototype, but it is **not demo-ready as a credible simulation-assisted engineering tool** because the live simulation/recovery path is not reliable in this audit run. No physical layout verification or foundry accuracy is established here.

## Overall assessment

**NOT APPROVED.** The isolated worker architecture exists, and several branch-level safeguards are present, but the observed tests refute the reported clean suite and recovery claims. The HIGH findings must be corrected and independently verified before Batch 2.
