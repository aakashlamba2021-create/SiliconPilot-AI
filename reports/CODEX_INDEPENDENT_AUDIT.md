# SiliconPilot AI — Independent Phase 4 Code Audit

**Review date:** 2026-11-12  
**Scope:** Independent review of the Phase 2–4 reports and implementation under `SiliconPilot-AI/`.  
**Method:** Source inspection and attempted independent test execution. No existing project files were modified. The requested 23 tests could not be executed in this environment: neither `python` nor `py` nor `ngspice` is available on PATH. Therefore this audit makes no claim that the suite passed.

## Overall assessment

**Not yet a credible simulation-assisted analog IC engineering prototype for making engineering decisions.** It is a useful early demonstration scaffold: it has recognizable netlist templates, an ngspice DLL binding, vector parsing, and an optimization flow. However, important reported outputs are not established as valid measurements. The CLI backend cannot return parsed waveform outputs, the power path substitutes a nominal current, the winner selection can accept failed simulation data, top-three selection is not constrained to the Pareto front, and the database's training-eligibility check relies on a status that can be declared converged despite missing analyses. Circuit operating points/topologies and AC/transient testbench semantics also need engineering validation. The benchmark Level-1 models are explicitly non-foundry models; the report does correctly disclose this, although its “calibrated” wording is unsupported by provenance in the repository.

## Findings

### HIGH — Simulation success/convergence is not tied to valid results from all required analyses

**Evidence:** `src/simulation/ngspice_runner.py`, `_run_via_dll` returns success if *any* AC file, transient file, or OP value exists; the candidate flow then extracts metrics from the AC run only. `src/simulation/simulation_metrics.py:65-108` returns `Converged` whenever `sim_result.success` is true, even if AC data is absent/invalid or transient data is absent/invalid. In `src/optimization/candidate_optimizer.py` the AC and transient runs are separate, but only `res_ac` is passed to the extractor; `res_tran.success` is ignored. Zero/default metrics can therefore coexist with a `Converged` label.

**Impact:** Failed or incomplete runs can enter the candidate comparison and may be treated as successful physical observations. A parser failure or missing analysis is not equivalent to convergence of the full requested simulation.

### HIGH — Feedback database admits records based on a weak status and a vacuous/unstable error gate

**Evidence:** `src/optimization/candidate_optimizer.py` sets `verified_for_training` when `m.convergence_status == "Converged"` and `report.mean_relative_error_pct < 25.0`. The prior finding shows convergence status does not validate both runs or data. The `SimulationMetricsExtractor` can generate default zero metrics for absent vectors. `src/simulation/feedback_db.py:65-112` trusts caller-provided `simulation_status` and `verified_for_training`, while `mark_as_verified` at line 145 can set any record verified without checking provenance/status. Retrieval filters on those two flags only.

**Impact:** The database does not itself guarantee that a “verified” item is genuinely simulated, complete, converged, or suitable for training. The report's guarantee is stronger than the implementation.

### HIGH — Power can be a hard-coded estimate and the requested supply-current vector name does not match the netlist source

**Evidence:** `src/simulation/netlist_manager.py:193-198` names the supply source `VDD`; `src/simulation/ngspice_runner.py` builds `i(v{supply_node})`, i.e. `i(vvdd)` for default `supply_node="vdd"`. The requested device current should correspond to the actual voltage source `VDD` (`i(vdd)` in ngspice naming). If no current is parsed, `src/simulation/simulation_metrics.py:197-211` silently substitutes `80e-6 A` and reports `vdd * 80 µA` as measured power. It takes absolute current, which is appropriate for positive dissipation under ngspice's voltage-source sign convention, but only after reading the correct source current.

**Impact:** Displayed power/current may be invented fallback values rather than a DC operating-point result. A successful voltage-only OP parse still causes this fallback.

### HIGH — Phase-margin extraction is not robust to phase conventions/wrapping and sweep-edge cases

**Evidence:** `src/simulation/simulation_metrics.py:134-163` interpolates a 0 dB crossing in log-frequency and uses `180 - abs(phase_dc - phase_fu)`, but parser phase is taken from `vp(outp)` in `src/simulation/ngspice_runner.py` then heuristically converted in `src/simulation/simulation_parser.py:94-99`. This is output phase, not explicitly the phase of the transfer function `V(outp)/(VIN-VIP)`; in the differential testbench the input difference is 1 V, but this is implicit. No phase unwrapping is performed before linear interpolation. If no crossing occurs, GBW is set to a sweep endpoint and a plausible PM may be fabricated. If crossing occurs at first sample, it returns PM=0. No crossing-validity flag is exposed.

**Impact:** GBW and PM can be numerically plausible while representing sweep limits or a phase-wrapped interpolation rather than unity-gain stability.

### HIGH — Transient “slew rate” and settling time are not validated engineering measurements

**Evidence:** `src/simulation/simulation_metrics.py:166-195` takes the largest adjacent-sample absolute derivative across the entire trace, including input edges, startup and numerical artifacts; it does not estimate a sustained output slew interval or distinguish rising/falling slew. Settling is measured against the *last sample* as final value and uses the final 2% magnitude (or fixed 10 mV), rather than a defined step amplitude/final steady-state estimate. It reports the time after the last out-of-band sample, without validating monotonic time, noise, multiple steps, or that the trace remains settled to its end. The tests' threshold-only checks do not validate accuracy.

**Impact:** SR may be a single-step numerical spike; settling can be reported despite an inappropriate reference or incomplete/non-settling response.

### HIGH — CLI fallback is not a working equivalent backend

**Evidence:** `src/simulation/ngspice_runner.py`, `_run_via_cli` invokes batch ngspice with `-r rawfile.raw` but returns no `ac_file`, `tran_file`, or `op_output`; the rest of the pipeline parses `wrdata` text files that the CLI path does not create. The fixed `rawfile.raw` path is also relative to the process working directory. `test_01` asserts a backend exists and later integration tests require output files, so CLI-only installations will not satisfy the advertised fallback contract.

**Impact:** Backend discovery may report the CLI as available while the actual simulation validation flow cannot extract measurements.

### HIGH — DLL result handling does not check command return codes for analyses or result extraction

**Evidence:** `_run_via_dll` checks the `source` return code, but ignores return codes from `run`, `setplot`, `wrdata`, `print`, and `destroy all`. Error detection scans only selected text fragments in the shared callback logs. `success` is inferred from output file existence/nonempty size or parsed OP entries, not required analyses or vector validity.

**Impact:** DLL command failures, missing plot names, or partial analysis execution may not be reported as failed simulations. Persistent callback reference `_cb_char_ref` is correctly held on the runner instance; the `SendChar` signature and `ngSpice_Init` argument layout are plausible, but callback/lifetime safety and repeat-run stability cannot be certified from source inspection alone. There is no independent executed stress/repeated-run test here.

### HIGH — “Top three Pareto-optimal candidates” is implemented as the first three rows

**Evidence:** `src/optimization/candidate_optimizer.py` sets `top_slice = ranked_candidates.head(top_k)` and does not filter `is_pareto_optimal` or `pareto_rank == 1` before simulation. Winner selection then sorts converged candidates on gain/GBW, not the requested hard constraints or a joint objective. The fallback branch expressly chooses a converged candidate even if PM < 40° and still returns it as `winner_id`.

**Impact:** Non-Pareto or unstable candidates can be simulated/selected. Stability gating is advisory rather than a hard gate for declaring a winner.

### HIGH — The transient testbench does not clearly test a closed-loop amplifier step response

**Evidence:** `src/simulation/netlist_manager.py:178-184` for Miller transient drives `vin` with a pulse and connects `RFB outp vip 10` (10 Ω), while the other input remains the PULSE source. Five-transistor/folded/telescopic templates similarly use a 10 Ω feedback resistor (`:256-260`, `:320-324`, `:390-394`). This is not a conventional defined unity-gain feedback testbench for a differential op amp, and 10 Ω is an unusually strong feedback path. There is no explicit measurement of output swing, step amplitude, common-mode/bias validity, or overload recovery.

**Impact:** Transient SR and settling figures do not establish expected closed-loop amplifier behavior.

### MEDIUM — No independent-candidate model-versus-SPICE validation study

**Evidence:** `tests/test_phase4_simulation.py:test_07_prediction_vs_simulation_validation` compares hand-constructed mock metrics and mock predictions. `test_08_simulation_in_the_loop_candidate_selection` uses three manually specified candidates and checks only success/count/winner presence, without verifying prediction errors, candidate selection, or that supplied parameters materially change the netlist/results. Phase 4 report's validation narrative reports illustrative agreement but the checked-in artifacts inspected here do not establish a held-out candidate protocol, sample count, or independent seed/split.

**Impact:** The reported validation does not demonstrate generalization beyond selected examples; no supported claim of model-versus-SPICE accuracy across independent candidates follows from these tests.

### MEDIUM — Candidate parameters do not all map to the simulated circuit; clipping/defaults are silent

**Evidence:** `src/simulation/netlist_manager.py:124-152` maps only width, length, multiplicity, finger count (returned but not used in the device instances), bias, load, compensation, and supply. The four templates do not use `finger_count`, `finger_arrangement`, layout displacements, or most optimizer dimensions. Width and length are clamped at `:131-135`; other values such as bias/load/capacitance are not similarly range-checked. Device widths incorporate multiplicity (`w_in = w*m`) but not finger count. Therefore candidates differing only in ignored fields produce the same netlist; out-of-range requested W/L candidates can be simulated at silently altered values while stored parameters retain originals.

**Impact:** Candidate identity/ML parameter records can disagree with actual simulated devices. This weakens claims that candidate optimization is validated at the proposed geometry.

### MEDIUM — Circuit template biasing and topology validity are not demonstrated by structural tests

**Evidence:** `src/simulation/netlist_manager.py:154-443` contains four templates, but `test_02_netlist_generation_all_topologies` only checks for `.ac`, `.tran`, `.end`, and the substring `NMOS_180`; it does not assert topology-specific device connectivity or OP regions. The Miller template comment claims a stage sizing relationship but `MP6`/`MN7` dimensions are fixed from load/tail, and biasing depends on a simple Level-1 mirror. Folded-cascode and telescopic templates use fixed bias voltage sources with no parameterized bias generation or headroom/current verification. Topology labels are therefore not proof of electrically valid bias points.

**Impact:** The templates are useful illustrative netlists, but need OP-point and transistor-region review before design-level use.

### MEDIUM — Device model provenance is absent; “calibrated” is not substantiated

**Evidence:** `src/simulation/netlist_manager.py:12-45` defines hand-entered Level=1 parameters and calls them “Benchmark Empirical” and “Predictive Level-1 / BSIM Calibrated.” There is no citation, source file, extraction procedure, calibration dataset, or validation against a named process. The Phase 4 report appropriately disclaims proprietary foundry accuracy at `reports/PHASE_4_REPORT.md` §3.2, but its repeated “calibrated standard” phrasing has no repository evidence.

**Impact:** The explicit no-foundry-accuracy disclosure is good; nevertheless, “calibrated” should be treated as unverified and this model must not be represented as a foundry PDK or silicon-correlated model.

### MEDIUM — AC gain measurement uses output voltage rather than explicit differential transfer function

**Evidence:** DLL extraction writes `vdb(outp)` and `vp(outp)` only. AC source conventions are opposite signs at inputs in the templates, making the differential input nominally 1 V in common test setups. The extractor assumes that convention without reading source amplitude, polarity, or measuring `V(vin)-V(vip)`.

**Impact:** Current templates' gain may numerically equal differential gain under their source setup, but this is brittle and can be wrong for modified testbenches or common-mode excitation. No independent analytical AC-network check is in tests.

### LOW — Test assertions are mostly smoke tests and contain synthetic/false-positive conditions

**Evidence:** `tests/test_phase4_simulation.py`: test 02 checks substrings, not valid topology or operating point. Test 05 checks broad lower bounds on metrics rather than an independently calculated expected result. Test 07 uses mocks and a single favorable example. Test 08 checks only that a winner exists, even if it is unstable. Test 09 marks arbitrary hand-authored values as converged/verified and then confirms retrieval, rather than verifying trust-boundary enforcement. Test 01 hard-requires local ngspice instead of conditionally testing backend-independent behavior. None of these were executed in this audit because Python is unavailable.

**Impact:** Passing these tests (as claimed by the Phase 4 report) would not establish metric correctness, model validation, complete convergence, safe curation, or stability gating.

### LOW — Phase reports overstate verified guarantees relative to implementation

**Evidence:** `reports/PHASE_4_REPORT.md` describes “guarantees stability-verified circuit layouts,” “top three Pareto-optimal,” and “never fed synthetic data pollution”; source behavior above does not support those absolute statements. It also claims 23/23 tests pass, but this review could not independently execute them.

**Impact:** Update technical claims to distinguish implemented demonstration flow from verified engineering guarantees.

## Test execution record

Attempted commands from `SiliconPilot-AI/`:

- `python -m unittest discover -s tests -v` — not run; PowerShell reported `python` is not recognized.
- `py -m unittest discover -s tests -v` — not run; PowerShell reported `py` is not recognized.
- `where.exe python`, `where.exe py`, `where.exe ngspice` — no executable found for any command.

The test suite consists of the files `tests/test_pipeline.py`, `tests/test_phase3_optimizer.py`, and `tests/test_phase4_simulation.py`. The report's 7+6+10 count is not an independent execution result from this audit.

## Recommendations before relying on results

1. Make simulation status per-analysis and require validated AC, transient, and OP data (where requested); record ngspice return codes and analysis diagnostics.
2. Fix the voltage-source current query and remove the nominal-current fallback from any field labeled measured.
3. Define a documented, topology-appropriate differential/open-loop AC bench and closed-loop transient bench; validate operating point/headroom and specify step amplitude, settling band, and SR measurement window.
4. Unwrap transfer-function phase and reject unity crossings outside the sweep; compare extraction to analytical synthetic vectors with known crossings.
5. Filter the simulated set explicitly to Pareto rank 1, and never name an unstable candidate a stability-verified winner.
6. Store effective netlist parameters, including clipped values, or reject candidates outside range; wire every claimed optimization parameter to the circuit or label it layout-only.
7. Enforce database verification inside the database using evidence fields and immutable provenance; remove unrestricted manual verification for training eligibility.
8. Add independent candidate holdout comparisons and stress/repeated-run tests; execute all tests in an environment with a known ngspice version before asserting a pass.
9. Cite model provenance or remove “calibrated” claims; retain the explicit foundry-accuracy disclaimer.
