# Phase 4.1 Batch 1 — Independent Read-Only Review

**Review date:** 2026-11-12  
**Scope:** Batch 1 source changes, added tests, and independent execution of the repository test suite. No source, test, database, model, or existing report files were edited.  
**Git limitation:** The project directory and its parent do not contain a `.git` directory. `git status` and `git diff` therefore fail with “not a git repository”; I could not independently establish the modified-file list or compare Batch 1 against a tracked baseline. Findings below are based on direct inspection of the current files.

## Executive result

Batch 1 contains meaningful improvements: the 80 µA current fallback is gone, result fields can be `None`, the runner tracks requested analysis statuses and checks many command return codes, and the optimizer no longer returns an unstable fallback winner. However, some claims are only partially satisfied. Exact command failures are not all handled; status extraction can still overstate completion if inputs are omitted; database “verified” remains caller-controlled for converged status; optimizer training eligibility does not apply the 40° phase-margin gate; Pareto rows are still not filtered before simulation; and the Batch 1 tests largely assert mocks rather than the end-to-end trust boundary. The AC phase-margin algorithm remains the Batch 0 implementation and has not received the planned Batch 2 correction.

I independently ran the suite using `.venv\Scripts\python.exe` and the KiCad 9 DLL path. Pytest collected 34 tests. **29 passed and 5 failed in 9.47 seconds**. This does not reproduce Gemini's claimed 34/34 in 4.81 seconds.

## Severity-ranked findings

### HIGH — The independently executed suite is not green; live ngspice becomes unrecoverable during the run

**Evidence:** Command run from the project root:

```powershell
$env:NGSPICE_DLL='C:\Program Files\KiCad\9.0\bin\ngspice.dll'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PATH='C:\Program Files\KiCad\9.0\bin;' + $env:PATH
& '.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests -v
```

Pytest reported `collected 34 items`, `29 passed, 5 failed in 9.47s`. Failures:

- `tests/test_phase4_1_batch1.py::TestPhase41Batch1::test_11_real_ngspice_power_measurement_parity` — `res.success` was false.
- `tests/test_phase4_simulation.py::TestPhase4Simulation::test_03_simulation_execution_ac_and_tran` — failed in live simulation execution.
- `tests/test_phase4_simulation.py::TestPhase4Simulation::test_04_simulation_parsing` — AC parser data was invalid.
- `tests/test_phase4_simulation.py::TestPhase4Simulation::test_05_metric_extraction` — status was `Failed: stderr Error: ngspice.dll cannot recover and awaits to be reset or detached`, not `Converged`.
- `tests/test_phase4_simulation.py::TestPhase4Simulation::test_08_simulation_in_the_loop_candidate_selection` — `winner_id` was `None`.

The exact ngspice “cannot recover” diagnostic indicates the shared-library session was left unusable by the time later live tests ran. The present run does not establish whether the malformed-netlist test, cleanup sequence, or another earlier command is the triggering action, but repeated live analyses in one process need an isolated lifecycle/recovery test. The 4.81-second claim is not reproduced by this run.

### HIGH — Database verification is still not proof of a complete, genuine SPICE simulation

**Evidence:** `src/simulation/feedback_db.py:89-90` sets eligibility to `(simulation_status == "Converged") and bool(verified_for_training)`. This is just a status string plus a caller-supplied boolean. It does not validate simulator provenance, completed `ac/op/tran` statuses, parsed finite metric coverage, netlist/effective-parameter identity, or that `spice_results` contains measured values. `mark_as_verified` at lines 148-170 checks only `simulation_status == "Converged"` before promotion. The optimizer supplies `verified_for_training` based on status and model error, not proof of complete analyses (`src/optimization/candidate_optimizer.py:105-120`).

**Defect:** A direct caller can submit fabricated `spice_results`, status `Converged`, and `verified_for_training=True`; the DB accepts it. A converged AC/OP run with absent transient metrics can also meet the optimizer's status/error criterion if the validator has enough other metrics to calculate an error.

**Required correction:** Eligibility must be enforced at the database boundary from structured run evidence. Require simulator/source identity, complete requested-analysis statuses, valid finite measured fields and netlist/effective-parameter provenance. Keep “simulation completed” separate from “ML agreement.” Manual verification must not promote a record missing that evidence.

### HIGH — The new DB test proves rejection of one status string, not the database trust boundary

**Evidence:** `tests/test_phase4_1_batch1.py::test_10_prevention_of_incomplete_results_in_feedback_db` inserts a record marked `Incomplete: AC invalid`, then checks the flag is zero and retrieval excludes it. It never attempts a forged `Converged` status with empty `spice_results` and caller `verified_for_training=True`, nor a converged record with only some required metrics or fabricated provenance. Its manual-promotion case repeats the same non-converged status test.

**Defect:** The test passes while the database accepts arbitrary “Converged” payloads as training verified. This is a material false sense of safety because source still trusts the caller for all evidence beyond the status string.

**Required correction:** Add adversarial tests for forged converged payloads, incomplete analysis evidence, absent/invalid metrics, missing provenance, and attempts to manually promote each. The only eligible result should be a complete record backed by a successful run artifact.

### HIGH — Pareto-front filtering is still absent from the simulation selector

**Evidence:** `src/optimization/candidate_optimizer.py:54` remains `top_slice = ranked_candidates.head(top_k)`. It does not filter `pareto_rank == 1` or `is_pareto_optimal`. Thus the row with rank 2 can be sent to ngspice if it appears in the first `top_k` rows. The new optimizer logic changes winner gating, not candidate selection.

**Defect:** “Top-k Pareto candidates” is still implemented as first-k rows, which can include dominated candidates.

**Required correction:** Select from a validated rank-1 set first, apply deterministic ordering, then cap at `top_k`. Add a test with interleaved rank-1/rank-2 rows and spy on the runner to verify exactly which candidates are simulated.

### HIGH — Optimizer stability gate applies only to winner selection, not feedback eligibility; it is not the planned configured hard constraint

**Evidence:** Winner candidates in `src/optimization/candidate_optimizer.py:158-184` require `simulation_status == "Converged"`, non-None measured phase margin `>= 40.0`, and non-None gain; otherwise `winner_id=None`. That is a genuine improvement over the earlier unstable fallback. But database eligibility at lines 116-120 has no PM requirement, so a stable-gate failure can still be marked verified for training if mean ML error is below 25%. The threshold is hard-coded, and PM itself is calculated by the unchanged extractor algorithm.

**Defect:** The phrase “optimizer eligibility now requires converged results and PM >= 40 degrees” is true only for `winner_id`, not for training-database eligibility. Also, `convergence_status` is not an independent indication that the reported PM is mathematically reliable.

**Required correction:** Separate `winner_eligible`, `simulation_record_valid`, and `training_eligible` policies. Make PM threshold explicit/configured where intended. Do not use the current phase-margin output as a trusted stability result until Batch 2 corrects the transfer-phase and crossing method.

### HIGH — Phase-margin extraction is unchanged and must not yet be treated as a reliable stability measurement

**Evidence:** `src/simulation/simulation_metrics.py:180-230` retains the previous `_extract_ac_metrics` implementation: it uses the absolute difference between first-point and crossover output phases, does no phase unwrap, and returns the last sweep frequency as GBW when there is no 0 dB crossing. It may create a plausible numeric PM without a bracketed unity crossing. The Batch 1 diff cannot be obtained due to absent Git metadata, but current code itself confirms no planned Batch 2 correction is present.

**Required correction:** Batch 2 should compute the complex differential transfer function, unwrap phase, interpolate only a bracketed 0 dB crossing, and mark PM/GBW unavailable or ambiguous at sweep limits/multiple crossings. Until then, PM≥40° is only a gate on the current algorithm's value, not a validated engineering stability guarantee.

### MEDIUM — Not every ngspice command failure is checked; cleanup and initialization return codes are ignored

**Evidence:** `src/simulation/ngspice_runner.py:103-123` calls `ngSpice_Init` but ignores its return value. `_run_via_dll` checks return codes for source, run, setplot, wrdata, and OP print (`:295-419`), but cleanup `destroy all` at lines 337 and 422 is called directly and its return code/logs are ignored. Error-text detection is a fixed list of phrases and is not a general command failure mechanism. `wrdata` success is based on nonempty file presence, not successful parsing/expected row count at runner level. Files are named using millisecond timestamps and are not removed before writing, so a same-process same-millisecond name collision could let a stale nonempty file satisfy the existence test if a write fails without a detectable message.

**Defect:** Most analysis commands now have stronger checks, but it is inaccurate to say every ngspice command failure is detected. DLL initialization/recovery and cleanup failures are unreported; output-file existence alone is weaker than validated output data.

**Required correction:** Check and record the init result. Route cleanup through the same command checker and surface cleanup failure as session health/recovery state. Create unique output paths (or unlink before use), verify the freshly written file parses and contains valid vectors, and add injected command-return/log tests for every command stage.

### MEDIUM — Partial analyses are mostly rejected by the runner, but metric convergence can still be overstated for omitted data/statuses

**Evidence:** `_run_via_dll` computes `overall_success` from all requested analysis statuses (`src/simulation/ngspice_runner.py:424-458`), so a requested AC/TRAN/OP extraction marked `FAILED` prevents success. This is a meaningful improvement. However, `SimulationMetricsExtractor.extract` (`src/simulation/simulation_metrics.py:89-114`) only declares incomplete if a non-None data object is invalid. If both `ac_data` and `tran_data` are omitted and the run result says success, it falls through to `Converged` while all electrical metrics remain `None`. The `ac_ok`/`tran_ok` predicates reject only status `FAILED`, not all statuses other than `SUCCESS` (`:89-100`).

**Defect:** Completeness is inferred partly from optional argument presence rather than the requested-analysis contract. A caller can provide successful result + missing parsed data and receive `Converged`.

**Required correction:** Pass an explicit requested-analysis/required-metric set into extraction; require the corresponding `SUCCESS` status and valid parsed vectors. Any missing required vectors/status should make overall status `Incomplete`, never `Converged`. Test success results with missing data, `NOT_REQUESTED`, `UNSUPPORTED`, `ERROR`, and mismatched file/status combinations.

### MEDIUM — Power fallback is removed and current sign/units are internally consistent, with one voltage-provenance qualification

**Evidence:** `src/simulation/simulation_metrics.py:263-317` has no assumed-current substitution. It looks for exact `i(vdd)` and a legacy alternate key, returns `None` if current is missing/nonfinite, computes `i_delivered=-i_source`, then `P_W=V*i_delivered`, `mW=W*1000`, and `µA=A*1e6`. `src/simulation/ngspice_runner.py:408-419` queries `i(VDD)` and verifies that parsed `i(vdd)` exists. Parser values are lowercased. The hand-authored cases in test 08 check -150 µA → +0.27 mW and +50 µA → -0.09 mW.

**Qualification:** The extractor uses `vdd_override` when `v(vdd)` is absent (`simulation_metrics.py:284-291`), and `extract()` defaults this override to 1.8 V. Thus the result uses actual simulated source current multiplied by the configured/default supply voltage when the OP voltage vector is missing; provenance says `spice_op_simulation` even though voltage came from the override. No assumed current or fabricated power fallback remains in this extraction function, but the voltage provenance is not always wholly OP-measured.

**Required correction:** Keep no-current behavior as implemented. Either require measured `v(vdd)` or record `voltage_source=netlist_parameter` separately and label power as calculated from simulated current × applied supply voltage. The live parity test intended to verify this did not pass in the suite (`test_11`).

### MEDIUM — `None` metrics are supported, but `Converged` can coexist with unavailable metrics

**Evidence:** `ExtractedMetrics` fields are Optional and initialized to `None` (`simulation_metrics.py:25-40`); the AC/TRAN paths leave metrics `None` when not extractable. This removes silent zero values from the output dataclass. Yet status can still be `Converged` when data arguments are omitted (above). Candidate summaries map missing ML fields to `0.0` (`candidate_optimizer.py:130,132,134,136`), which is an ML-summary default and must not be confused with SPICE metrics. In the validator, missing SPICE values are handled as unavailable, but `mean_relative_error_pct` can still be computed from whichever subset remains, and the optimizer may treat that as eligible.

**Defect:** Individual missing SPICE quantities are visible as `None`, but the global status and training decision do not require a defined complete metric set.

**Required correction:** Make completeness policy explicit by metric/analysis and distinguish `Converged` from `MetricsComplete`/`Validated`. Do not allow a subset-based mean error to imply full validation unless the required comparison set is stated and satisfied.

### MEDIUM — Batch 1 tests 2–10 mostly exercise mocked values or helper behavior, not the whole engineering path

**Evidence:** `tests/test_phase4_1_batch1.py`:

- Test 01 is a real malformed-netlist ngspice invocation (but it does not cover every command return code or ensure the DLL can recover for the next run).
- Tests 02–03 construct `SimulationResult` and parser results manually; they do not inject actual runner output failures through the optimizer.
- Tests 04–05 create synthetic text files/dictionaries and exercise parser/power helper logic.
- Tests 06–08 use hard-coded OP dictionaries and check arithmetic/availability; these are useful unit checks but not measured ngspice parity.
- Test 09 directly calls the CLI private method and asserts the unsupported status, rather than validating backend-selection behavior end-to-end.
- Test 10 checks only a non-Converged DB status, not a forged Converged record or full evidence policy.
- Test 11 is the one live ngspice power test, and it **failed** in the independent run.

**Defect:** The suite proves some local behaviors but does not substantiate all claims about commands, session stability, successful complete simulation, or safe feedback curation.

**Required correction:** Add command-adapter failure injection for source/run/setplot/wrdata/print/init/destroy, end-to-end partial-run and DB gating tests, and successful standalone plus repeated live ngspice tests. Keep synthetic unit tests, but label them as such and do not count them as physical validation.

### MEDIUM — Phase-margin result and candidate records must remain labeled as algorithmic SPICE extraction, not physical stability or layout verification

**Evidence:** Phase 4 report uses broad “simulation-verified” and stability claims. Current runner uses benchmark Level-1 models and the old PM calculation. None of the Batch 1 changes perform physical layout DRC/LVS or establish foundry model accuracy. Candidate summaries include ML prediction columns alongside SPICE columns; UI/database consumers must preserve those source distinctions.

**Required correction:** Until Batch 2 and topology/bias validation, call PM a provisional AC-derived metric under the benchmark model. Do not claim foundry stability signoff or physical layout verification.

## Claim-by-claim disposition

| Gemini claim | Independent disposition |
|---|---|
| 34/34 tests passed in 4.81 s | **Not reproduced.** This run: 29 passed, 5 failed in 9.47 s. |
| Simulation command validation strengthened | **Partly true.** Major analysis commands are checked; init and cleanup outcomes are ignored and logs use substring heuristics. |
| Per-analysis status tracking introduced | **True.** `SimulationResult` now has OP/AC/TRAN statuses and status/error dictionaries. Completeness still has extractor/API gaps. |
| 80 µA fallback removed | **True in relevant power extractor.** Missing current returns unavailable. |
| Supply current uses `i(vdd)` with corrected polarity | **True in source.** Query maps `VDD` to lowercased `i(vdd)`; delivery is `-i_source`. Live verification test failed. |
| Feedback DB verification strengthened | **Partly true.** Non-Converged status cannot be promoted, but arbitrary Converged payloads and caller verification flags remain trusted. |
| Optimizer eligibility requires converged and PM ≥40° | **Partly true / misleading.** This gates winner selection only; DB training eligibility omits PM. PM extraction itself is unchanged. |

## Batch 2 readiness

**Batch 1 is not genuinely ready for Batch 2 sign-off yet.** Batch 2 work may proceed as a development branch, but do not treat Batch 1 as a reliable simulation foundation or use its PM gate as engineering stability evidence. Corrections required first:

1. Resolve the five failing tests and demonstrate repeatable ngspice DLL operation after a deliberately failed netlist.
2. Make extractor completeness depend on explicit required analyses and `SUCCESS` statuses; missing data must not yield `Converged`.
3. Enforce DB eligibility using complete, provenance-backed SPICE evidence rather than caller-controlled status/boolean and partial mean error.
4. Add a rank-1 Pareto filter before top-k simulation and separate winner qualification from training eligibility.
5. Clarify applied-supply-voltage provenance in the power result; keep the current fallback removed.
6. Make initialization, analysis, output extraction, and cleanup command failures observable; verify freshly written data, not mere file existence.
7. Preserve the planned Batch 2 scope: correct differential transfer, phase unwrapping, bracketed unity crossing and unavailable status at sweep limits before treating PM≥40° as meaningful.
8. Add adversarial/end-to-end proof tests; the current new tests are predominantly mocks and do not prove the database trust boundary or stable DLL lifecycle.

No result in this review constitutes physical layout verification.
