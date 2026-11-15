# SiliconPilot AI User Guide

## Start

From the project folder, run `./scripts/run_local.ps1` in PowerShell. If it reports that a local Streamlit server already owns one of the app ports, it will not launch a second process. Open the URL it reports; the default is `http://127.0.0.1:8502`.

The launcher configures `NGSPICE_DLL`, its `bin` directory on the child process `PATH`, and a project-local scratch folder for the launched process. It does not alter machine-level variables.

## Choose a page

Use the sidebar sections **Workspace**, **Design analysis**, and **Verification**. The eight pages are Overview, Dataset explorer, Model evaluation, Layout prediction, Candidate comparison, AI engineering assistant, System information, and Electrical simulation.

### Dataset explorer

The page reports data availability and shows the source and target fields present in each loaded table. The included small OSIRIS-like CSV is generated from synthetic equations. Its labels and metrics are for demonstrating the interface, not for verifying a design. Official data is shown only if its local cache is available or the loader obtains it.

### Model evaluation

Each saved model card includes its origin, training data, saved metrics, trust status, and evaluation limitations. Metrics in the saved manifest are historical reported results; a metric is not a fresh evaluation unless the current session explicitly runs one.

### Layout prediction and candidate comparison

The electrical model is synthetic-data-trained and is limited to exploratory estimates. The geometry model is restricted due to negative saved R² across all four targets. The OSIRIS area/PEX model is benchmark-derived and exploratory. Candidate ranking uses a formula-based area proxy and parameter-derived displacement; synthetic electrical estimates are shown for context but excluded from constraints, Pareto sorting, and scores. These rankings are not physical recommendations or signoff results.

No model prediction constitutes an actual DRC or LVS result. The DRC classifier is quarantined.

### AI engineering assistant

Text discussion, greetings, and capability questions work without a vision provider. To analyze a screenshot, add PNG/JPG/JPEG files in the upload area, review the previews, describe the question, select a provider, and explicitly consent before sending. Comparison mode sends the same selected images and prompt to both providers only after confirming each. If a provider is unconfigured or returns an error, the page reports that state; it does not silently switch providers.

The assistant’s visual observations are limited to visible content. It cannot establish connectivity, net names, exact dimensions, DRC/LVS, parasitic values, or measured electrical behavior from a screenshot. Image assets stay in the current Streamlit session; clearing them also clears image-linked assistant history.

The artifact inspector accepts only the formats listed in its uploader. It computes file provenance for the session, parses supported metadata, and does not save uploaded source files. GDSII/OASIS parsing uses a temporary copy under the local Windows temp folder and removes that copy after parsing. SPICE input is inspected statically and is not automatically executed.

### Electrical simulation

Select a supported topology and analysis before running. Review the generated netlist and model assumptions. Simulation metrics are results from the submitted netlist and the bundled empirical device model under ngspice; they are not foundry PDK, extracted-layout, or measured-silicon results. Unavailable analyses remain unavailable.

## Exporting

The assistant can export a text/JSON analysis record. Candidate and simulation pages provide their own report exports. Treat exported values according to their included evidence basis, provenance, and limitations.

## Help and limitations

The Overview and System information pages show current data/cache, model, provider-configuration, database, and ngspice state. See [Known limitations](KNOWN_LIMITATIONS.md) for evidence boundaries and actual validation status.
