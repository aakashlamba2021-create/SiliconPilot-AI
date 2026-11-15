# SiliconPilot AI Technical Architecture

## Application entry and navigation

`app.py` initializes the native Streamlit multipage navigation and delegates each route to an existing page module under `dashboard/pages/`. The eight page modules use shared components under `dashboard/components/`; `dashboard/chat.py` provides deterministic text conversation behavior without requiring a provider.

The theme is configured in `.streamlit/config.toml`. Page state, uploaded image bytes, and conversation state are held in Streamlit session state.

## Main modules

| Concern | Implementation |
|---|---|
| Dataset access | `src/data/` loaders and `src/data/schema_inspector.py` |
| Model artifacts and trust | `src/models/model_registry.py`, `src/models/model_policy.py`, `config/model_trust_policy.json`, and `models/saved_models/models_manifest.json` |
| Candidate generation and ranking | `src/optimization/candidate_optimizer.py`, surfaced by `dashboard/pages/candidate_comparison.py` |
| Image analysis | `src/assistant/vision.py` and `src/assistant/providers/` |
| Engineering artifact inspection | `src/eda/inspection.py`, surfaced by the assistant page |
| ngspice and result parsing | `src/simulation/ngspice_runner.py`, `netlist_manager.py`, `simulation_parser.py`, and `simulation_metrics.py` |
| Persistent simulation history | `src/simulation/feedback_db.py` and `data/simulation_results.db` |

## Model trust and ranking boundary

`ModelTrustPolicy` validates the schema and each entry. Missing, malformed, or unrecognized entries resolve to `unapproved`. `ModelRegistry.get_model_status()` is the separate status interface; it preserves the original `load_model()` bundle shape.

`load_model_for_permitted_use()` gates explicitly permitted inference. The DRC classifier is quarantined with no allowed uses. The synthetic electrical model is permitted only for exploratory estimates and metadata review. The negative-R² geometry model is restricted to metadata review. The benchmark area/PEX predictor is available for explicitly labelled exploratory analysis and metadata review. Candidate pages may display synthetic electrical estimates, but the ranker excludes them from constraints, Pareto sorting, and scores unless all candidate rows have one explicit, uniform measured or simulated evidence origin. Current exploratory ranking uses a formula-based area proxy and parameter-derived displacement; the restricted geometry model and area/PEX predictor do not rank candidates.

The trust policy describes known provenance and does not rewrite or replace model files, manifest entries, reports, or datasets.

## Assistant request flow

The text-only path handles common greetings and contextual follow-up locally. Vision providers use explicit provider selection and consent. Single-provider mode sends to only the selected provider. Comparison mode requires explicit consent for both and sends the same prompt/images to both; outputs are presented side by side. There is no automatic fallback.

The response schema contains a summary, visible observations, possible concerns, recommendations, trade-offs, validation steps, confidence and limitations, image references, and provider/model metadata. The report export omits image bytes. Provider usage and cost are displayed only when the response and configured rates provide the necessary evidence.

## Artifact inspection

`src/eda/inspection.py` applies an upload size bound, reads supported text/JSON/CSV/HTML formats conservatively, and returns file hash, filename, import time, parser format and parsed fields. SPICE netlists are parsed as text; the inspection path never submits them to ngspice. GDSII/OASIS metadata is read with `gdstk`; DEF/LEF parsing is a limited text subset. Unknown or absent evidence remains unset.

## Simulation

`NGSpiceRunner` discovers the configured shared library, checks load and API availability, and runs DLL work inside an isolated worker process. The process-local launcher sets `NGSPICE_DLL` and prepends the KiCad `bin` directory so dependent libraries resolve. AC and transient outputs are written to the configured temp path and parsed by the simulation modules. All dashboard pages use one shared runner instance for consistent status.

The current Windows launcher uses the 8.3 short path for the project-local scratch directory because the KiCad DLL’s temporary-file handling fails on this machine when the scratch path contains spaces. `TEMP`, `TMP`, `TMPDIR`, and `HOME` are process-local to the app child. Native GDSII/OASIS parser copies go to `%LOCALAPPDATA%\Temp\SiliconPilotAI`, outside the OneDrive-synced project directory, and are deleted after parsing. No permanent Windows environment setting is changed.

## Data and persistence

The raw benchmark datasets are local optional caches. Their absence is presented as unavailable and is not replaced by fabricated official data. `data/sample/osiris_sample.csv` is synthetic demonstration data. The simulation feedback database is existing persistent user data and is read for status and only changed by explicit simulation/history actions.

## Test organization

`tests/` contains registry/trust, assistant/provider contract, schema, EDA parsing, optimizer, and simulation coverage. Live DLL tests run only when the `NGSPICE_DLL` environment is configured. Mock provider tests validate request and response behavior but do not validate live credentials, provider quota, or billing.
