# Known Limitations and Validation Record

## Evidence boundaries

- **DRC/LVS:** no DRC/LVS tool is integrated into the application. The synthetic-label classifier is quarantined. Actual verification is unavailable unless performed in an external physical-verification flow and imported as evidence.
- **Electrical model:** the electrical performance targets in the sample are equation-derived with noise. Held-out metrics measure fit to synthetic labels, not ngspice, measurement, or silicon accuracy.
- **Candidate ranking:** synthetic, benchmark-only, missing, mixed, or invalid evidence origins are excluded from electrical constraints, Pareto sorting, and scores. Current exploratory ranking uses a formula-based area proxy and parameter-derived displacement; these are not physical signoff metrics.
- **Geometry model:** saved R² is negative for `Aeval` (-8.5724), `Leq_signal` (-4.5242), `Leq_power` (-1.5312), and `Ncut_intermetal` (-0.3519). It is not used as a trusted ranking signal.
- **Area/PEX model:** trained on OSIRIS benchmark targets. It estimates benchmark quantities; it does not run parasitic extraction on a user layout.
- **Simulation:** bundled empirical Level-1 device models are not foundry PDK models and do not include extraction from an imported layout. Simulation success applies only to the submitted netlist and model assumptions.
- **Image analysis:** screenshots do not establish exact geometry, connectivity, DRC/LVS, parasitics, or actual measured performance.
- **EDA formats:** supported parsers expose a conservative subset of reports/layout metadata. A parsed field does not guarantee full standards compliance or design validation. Uploaded SPICE netlists are never executed by the inspector.
- **Providers:** provider behavior, quota, credentials, account billing, and live endpoint/model availability are not validated by mocked tests. A live request is not claimed unless performed with configured credentials.
- **Benchmark caches:** raw benchmark datasets may be unavailable locally. Missing caches are shown as unavailable; the synthetic sample is not substituted for official benchmark data.

## Current Windows ngspice setup

The validated library path on this laptop is:

`C:\Users\aakas\AppData\Local\Programs\KiCad\10.0\bin\ngspice.dll`

The DLL and its API load under the project Python 3.11 `.venv`. A simulation requires the KiCad `bin` directory on the child process `PATH` for dependent DLL lookup. A project-local scratch directory is used for `TEMP`, `TMP`, `TMPDIR`, and `HOME`; its 8.3 short path is required because the long project path contains spaces and ngspice's internal temporary-file handling failed with it. These settings are process-local.

## Validation record

Results should be updated after a validation run; test success is not inferred from source inspection.

| Check | Status |
|---|---|
| KiCad 10 DLL discovery/load | Passed in this environment; see actual test results from the current run. |
| RC AC + transient vector smoke test | Passed with the project-local short scratch path. |
| Four supported topology AC + transient integration checks | Passed: `14 passed, 4 subtests passed` in the focused ngspice command. |
| Full maintained pytest suite | Passed: 107 passed, 7 subtests passed. |
| Live HTTP health and eight-page browser smoke test | Passed: HTTP health 200 at http://127.0.0.1:8502; all eight navigation routes opened; simulation page reported the KiCad 10 DLL loaded and session healthy. |
| Live Gemini/OpenAI request | Not run: no provider credentials configured. |

See `scratch/logs/` for local launcher logs. Do not include secrets in logs or support reports.
