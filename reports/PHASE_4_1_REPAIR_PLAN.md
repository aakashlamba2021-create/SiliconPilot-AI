# SiliconPilot AI — Phase 4.1 Simulation Engine Reliability Repair Plan

**Prepared:** 2026-11-12  
**Status:** Review and proposed plan only. No source files have been modified and no tests have been run in this phase.  
**Scope:** Runtime result validity, measured-vs-predicted labeling, electrical metric extraction, candidate gating, feedback curation, and sizing-to-netlist traceability. This plan makes no claim of physical layout verification.

## Context and boundary

The Phase 4 report describes a simulation-in-the-loop flow. The independent audit identified gaps in the path from ngspice output to claimed measurements and database eligibility. The project uses benchmark Level-1 transistor models; outputs would be results of those models and the specified testbench, not foundry-correlated silicon measurements. ML predictions, analytical estimates, and SPICE-derived values must remain separate fields and be labeled by source. Passing electrical simulation does not establish layout DRC/LVS, parasitic extraction, or physical layout verification.

The user reports that the actual Windows environment is now configured and all 23 current tests passed in 6.34 seconds. That result is user-provided context; this planning turn did not repeat the tests.

## Repair priorities

### 1. Simulation success/failure validation

**Source:** `src/simulation/ngspice_runner.py`, `NGSpiceRunner.run_netlist`, `_run_via_dll`, `_run_via_cli`; `src/simulation/simulation_parser.py`, `parse_ac_file`, `parse_tran_file`; `src/simulation/simulation_metrics.py`, `SimulationMetricsExtractor.extract`; `src/optimization/candidate_optimizer.py`, `CandidateOptimizer.run_simulation_validation`.

**Defect:** DLL execution checks only the `source` command return code. It ignores return codes for `run`, `setplot`, `wrdata`, and `print`. DLL success is true when any one of an AC file, transient file, or parsed OP value exists. The metrics extractor sets status to `Converged` whenever the passed `SimulationResult.success` is true, even when required waveform data is missing/invalid. Candidate validation runs AC and transient independently, passes only `res_ac` into extraction, and does not require `res_tran.success`. The CLI backend returns no parsed AC/transient file paths or OP dictionary, so it is discovered as available but is not functionally equivalent for the existing parser pipeline.

**Correct engineering behavior:** Report per-analysis outcomes (OP, AC, transient) separately. A requested analysis must have a successful ngspice command, a selected valid plot, expected vectors, finite/nonempty values, and a valid parser result. Overall success should mean all analyses required for the requested metric set succeeded. Do not call incomplete/missing measurements converged. CLI mode must either implement the same output contract or report unsupported/unavailable for this flow.

**Proposed fix:** Extend the result type with structured per-analysis statuses and diagnostic messages. Check all native command results and output vectors. Parse/validate raw outputs before marking an analysis successful. Pass the AC run, transient run, and OP run results distinctly into metric extraction; require each for the requested metrics. Make missing metrics explicitly unavailable (`None` plus status/reason) rather than silently substituting zero. Implement a CLI route that emits the same vector files/OP fields, or disable it for metric validation until that contract exists. Preserve callback references and make repeat-run cleanup/status explicit.

**Independent proof test:** Use a fake/stub runner or injectable command adapter to force each command/analysis to fail in turn and assert that overall status is failed/incomplete, no candidate is labeled converged, and no record is eligible for training. Test valid AC with absent transient and vice versa. Test malformed, empty, NaN/Inf, and missing-vector outputs. Run DLL repeated simulations with unique output paths and check returned statuses/files. For CLI, assert parity of AC/transient parsed vectors against DLL on the same deterministic netlist, or assert it is rejected as unsupported.

### 2. Remove assumed-current fallback; label measured versus non-measured values

**Source:** `src/simulation/simulation_metrics.py`, `SimulationMetricsExtractor._extract_power_metrics`; `src/simulation/ngspice_runner.py`, `_run_via_dll` OP query and `_parse_op_from_logs`; `src/simulation/netlist_manager.py`, voltage-source declarations; `src/optimization/candidate_optimizer.py`, result summaries and database payload.

**Defect:** `_extract_power_metrics` substitutes 80 µA when no supply current is found, then reports `vdd * current` in fields named power consumption. The default OP query constructs `i(vvdd)` from supply node `vdd`, while templates name the supply source `VDD`. An absent/unparsed current is therefore converted into an apparent measured power. The absolute-current convention is sensible for positive dissipation only when the current belongs to the correct source and the supply polarity is accounted for.

**Correct engineering behavior:** Measured SPICE power requires valid OP values for the actual supply source(s), correct source current sign/polarity, and the actual supply voltage. If these are not available, power/current must be unavailable with a reason; never substitute typical or assumed current into a SPICE result. If an estimate is later desired, put it in a separate clearly labeled estimate field and never use it as measured simulation data.

**Proposed fix:** Query the actual voltage source name(s) used by the generated netlist (e.g. `VDD`→current vector `i(vdd)` in ngspice), make source-name mapping explicit, parse exact expected keys, and compute delivered power using a documented sign convention. Reject missing/nonfinite OP current and voltage as `unavailable`, not a fallback. Include provenance labels such as `spice_op`, `ml_prediction`, or `analytical_estimate` in API/UI/database fields.

**Independent proof test:** Feed OP dictionaries with (a) known positive current under ngspice source convention, (b) the corresponding negative current, (c) wrong source key, (d) missing current, and (e) NaN/Inf. Assert the first two produce the same positive dissipated power under the documented convention and all others produce unavailable power, never 80 µA. An integration test should compare queried current against ngspice's direct OP print for the template source and verify unit conversions W/mW/µA.

### 3. AC gain, unity-gain frequency, and phase margin

**Source:** `src/simulation/ngspice_runner.py`, `_run_via_dll` AC `wrdata`; `src/simulation/simulation_parser.py`, `SimulationParser.parse_ac_file`; `src/simulation/simulation_metrics.py`, `SimulationMetricsExtractor._extract_ac_metrics`.

**Defect:** The runner exports only `vdb(outp)` and `vp(outp)`, while gain extraction assumes output voltage equals transfer gain. That depends on the current differential source amplitude and polarity being exactly one volt and does not compute `Vout/(Vin−Vip)`. Phase is converted using a magnitude heuristic, not unwrapped. PM is calculated as `180 - abs(phase_dc - phase_fu)` without explicit transfer-function phase convention. If the gain never crosses 0 dB, the routine returns a sweep endpoint as GBW and may report a plausible PM; sweep-edge values are not identified as limits. Invalid/ambiguous crossings have no explicit status.

**Correct engineering behavior:** Compute the complex open-loop differential transfer function using the actual differential input. Find a valid unity-gain crossing by interpolation in log-frequency, with bracketing samples and phase unwrapped consistently. Compute PM from the transfer phase at that crossing with a declared sign convention (e.g. `180° + ∠A(jωu)` for phase expressed as negative lag). If there is no bracketed crossing or multiple crossings, report GBW/PM unavailable or ambiguous unless a documented selection policy applies; do not substitute the sweep boundary as a measured crossing. Treat DC gain as the low-frequency transfer value only if the low end is in the flat region.

**Proposed fix:** Export complex transfer components or gain and transfer phase based on the differential input vector. Parser should validate increasing positive frequencies, vector lengths, finite values, and phase units; explicitly unwrap radians/degrees before interpolation. Return metric values with crossing validity and sweep-limit metadata. Expand the AC sweep adaptively or define a sufficient range when crossing is outside bounds.

**Independent proof test:** Construct synthetic complex frequency responses with known DC gain, pole/crossover, and known phase at crossover. Verify gain/crossover/PM against analytical values with tolerances; test phase wrapping across ±180°, radians/degrees, non-unit differential input, no crossing, crossing at the first/last sample, multiple crossings, nonmonotonic frequencies, and NaNs. Compare one simple RC/op-amp test circuit against direct ngspice vectors and independently computed transfer gain/phase.

### 4. Transient slew-rate and settling-time extraction

**Source:** `src/simulation/simulation_metrics.py`, `SimulationMetricsExtractor._extract_transient_metrics`; `src/simulation/simulation_parser.py`, `SimulationParser.parse_tran_file`; `src/simulation/netlist_manager.py`, transient testbench generation in all four templates.

**Defect:** Slew rate is the maximum absolute adjacent-sample derivative over the whole trace, so it can capture a numerical spike, startup artifact, or unsuitable interval. Settling uses the final sample as steady-state reference and an error band based on 2% of that value (or a fixed 10 mV near zero), without establishing step time/amplitude, settling band semantics, sample time monotonicity, or persistence inside the band. Current templates apply pulse inputs and 10 Ω feedback, which does not establish a documented conventional closed-loop step test for each topology.

**Correct engineering behavior:** Define the transient bench first: topology-specific feedback, operating point, input step polarity/amplitude, measurement node, and load. Compute SR only over a declared post-step output ramp window, excluding edge artifacts and using a robust slope fit over a specified output interval; report positive and negative slew separately if applicable. Estimate final steady state from a settled tail window and define settling band relative to step amplitude or final value per explicit spec. Settling time is the first time after the step when output enters the band and remains there through the end. Reject malformed/nonmonotonic/incomplete traces or responses that do not settle within the simulated interval.

**Proposed fix:** Parameterize per-topology transient testbenches, document the feedback configuration, define consistent input step and load, and expose step-time/amplitude/band settings. Validate waveform time ordering and finite values. Use windowed regression for SR and sustained-band detection for settling. Return unavailable/status when the trace lacks a valid response. Keep these outputs labeled as SPICE transient measurements for the benchmark netlist only.

**Independent proof test:** Feed synthetic waveforms with known linear slopes, ringing, noise spikes, multiple steps, incomplete settling, and time jitter. Assert the SR estimate matches the injected slope while ignoring isolated spikes and settling is not reported for an unstable/incomplete trace. Verify exact settling band/time on a known first-order step response. For each topology, inspect/parse the generated feedback connections and run a stable reference step with expected direction, non-clipped range, and settling behavior; test that a deliberately open-loop/invalid bench is rejected.

### 5. Pareto candidate selection and hard stability gate

**Source:** `src/optimization/candidate_optimizer.py`, `CandidateOptimizer.run_simulation_validation`; `src/simulation/simulation_validator.py`, validation verdict calculation.

**Defect:** The simulation flow selects `ranked_candidates.head(top_k)` without filtering for Pareto rank 1. It chooses a winner by gain then GBW among converged candidates. If no candidate meets 40° PM, the fallback still returns the highest-gain converged candidate as `winner_id`. This conflicts with the claim of a stability-gated simulation-verified winner and can include dominated rows.

**Correct engineering behavior:** Define whether top-k means top-ranked rows or Pareto front. For the stated Pareto validation flow, filter to valid rank-1 candidates, apply deterministic ordering, and simulate only up to `top_k`. Winner eligibility must require complete valid SPICE measurements, the configured hard minimum phase margin (not a separate hidden 40° if the requested constraint differs), and all other required hard constraints. If none pass, return `winner_id=None` and an explicit “no qualifying candidate” result. A failed/unstable candidate can be reported but never as a stability-verified winner.

**Proposed fix:** Add explicit `min_phase_margin_deg` and other hard constraints to the validation API; normalize/validate Pareto rank fields; filter first, then simulate. Separate `best_simulated_candidate` from `qualified_winner`, and keep ML constraints/predictions distinct from SPICE-measured gates. Do not silently fall back to an unstable winner.

**Independent proof test:** Use a candidate table with rank-1 and rank-2 entries, unsorted indices, missing/invalid ranks, and a Pareto-front size greater than `top_k`; inject deterministic simulation results including (a) all stable, (b) all below PM threshold, (c) one failed, and (d) one incomplete. Assert only rank-1 rows are simulated, exact top-k are selected deterministically, no unstable/failed candidate is the qualified winner, and no qualified candidates yields `None`.

### 6. Feedback database: genuinely successful, validated simulation records only

**Source:** `src/optimization/candidate_optimizer.py`, `run_simulation_validation` record step; `src/simulation/feedback_db.py`, `FeedbackDatabase.record_simulation`, `get_verified_records_for_retraining`, `mark_as_verified`; `src/simulation/simulation_validator.py`, `validate_candidate`.

**Defect:** The optimizer decides eligibility from the weak `Converged` status and an ML-vs-SPICE mean relative error threshold. That error is not proof of simulation completeness or physical validity. The database accepts caller-supplied status/verified flags and `mark_as_verified` can set eligibility without checking simulation provenance or completeness. A direct API caller can insert arbitrary values as verified.

**Correct engineering behavior:** A training-eligible record must carry provenance that it came from the configured simulator, netlist/topology and effective parameters, simulator/backend/version, completed analysis statuses, validated finite metric payload, and explicit quality/curation criteria. “Simulation succeeded” and “ML prediction agrees” are separate properties: prediction agreement may be useful for model evaluation, but it must not define whether a physical simulation occurred. Failed, partial, synthetic, analytical, and ML-predicted data must never be returned as verified SPICE training observations. Manual curation should not bypass required provenance.

**Proposed fix:** Store structured `data_source`, simulator identity/version, netlist hash/effective parameter hash, requested/completed analyses, status/reason, and metric validity. Derive eligibility at the database boundary from required evidence (and optionally separate human-review state); reject impossible status/flag combinations. Restrict `mark_as_verified` to quality review of records that already satisfy machine-verifiable provenance. Keep ML prediction/error fields separate from SPICE result fields. Retrieval should enforce all conditions, not trust one boolean.

**Independent proof test:** Attempt database insertion with arbitrary mock dictionaries, caller-set `Converged`, failed/partial statuses, absent provenance, missing metrics, nonfinite values, and valid complete ngspice evidence. Assert only the fully valid case is returned as training-eligible. Assert manual verification cannot promote missing/failed/synthetic records. Verify ML prediction values do not populate SPICE measurements and error thresholds do not convert failed simulations into verified records.

### 7. Candidate parameters must affect generated netlists and persisted evidence

**Source:** `src/simulation/netlist_manager.py`, `NetlistManager.generate_netlist`, `_extract_sizes`, and four `_build_*` methods; `src/optimization/candidate_optimizer.py`, `run_simulation_validation` candidate dictionary/persistence.

**Defect:** Candidate parameters are passed as a dictionary, but templates ignore `finger_count`, `finger_arrangement`, layout coordinates/displacements, and other optimizer features. Multiplicity changes selected widths; finger count does not affect any device instance. Width and length are silently clamped, while the original candidate dictionary is what gets stored. Missing parameters use defaults. Thus different proposed candidates can create identical netlists, or the actual netlist can use dimensions different from the stored candidate.

**Correct engineering behavior:** Every parameter advertised as affecting electrical simulation must have a defined mapping into the netlist and an independently verified effect. Layout-only parameters must be explicitly classified as not electrically modeled unless corresponding parasitics/geometry are injected. Reject out-of-range electrical parameters or store and report both requested and effective values; never imply the unclamped candidate was simulated. Persist a canonical netlist and effective-parameter record/hash with results.

**Proposed fix:** Define a per-topology parameter schema: required, optional/defaulted, electrical, layout-only, or unsupported. Validate ranges before generation. Map sizing and multiplicity/finger semantics correctly (or remove unsupported fields from simulated candidate claims). Return a generation artifact containing the netlist text/hash and normalized effective parameters. Record those effective values with every simulation result.

**Independent proof test:** For each supported electrical parameter, vary only that parameter and assert the generated netlist changes in the expected instance/value location; verify outputs can be independently parsed for W/L/geometry. For ignored layout-only parameters, assert they are marked as not represented or their declared parasitic model changes the relevant circuit element. Test boundary/out-of-range inputs reject explicitly or retain both requested and clamped effective values. Ensure persisted parameters/hash match the exact generated netlist.

## Measurement and claim vocabulary

Use these distinct categories consistently in code, reports, UI, and database:

- **SPICE measurement:** a quantity extracted from a successful, complete ngspice analysis of a recorded netlist and effective parameter set. It is a simulation result under the benchmark model/testbench, not a silicon measurement.
- **ML prediction:** a model output, with model/version and uncertainty where available. Never label it SPICE or measured.
- **Analytical estimate:** a calculated value from a stated simplified model. Keep it separate from solver outputs.
- **Assumed/default value:** an input/configuration assumption. It must not be represented as a result. In particular, no assumed current may populate measured power.
- **Physical layout verification:** DRC/LVS/PEX or equivalent evidence from physical-design tools. None of the electrical SPICE checks in this plan constitute physical layout verification.

## Suggested implementation sequence

1. Establish structured per-analysis status and remove silent defaults/fallback measurements.
2. Fix supply-current naming/sign and OP evidence.
3. Repair AC transfer/phase/crossing handling with analytical synthetic-vector tests.
4. Define valid topology-specific transient benches, then repair SR/settling extraction.
5. Enforce Pareto filtering and a hard configured stability gate.
6. Enforce provenance and eligibility at the database boundary.
7. Make candidate-to-netlist mapping/effective parameter evidence explicit.
8. Add independent tests for each behavior, then run the full suite in the configured Python 3.11/ngspice environment and report exact results. Do not claim layout verification.
