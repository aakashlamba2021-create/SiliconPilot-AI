# SiliconPilot AI Demo Guide

## Preflight

1. Open PowerShell in the SiliconPilot-AI project folder.
2. Run `./scripts/run_local.ps1`.
3. Confirm the launcher reports HTTP 200 from `/_stcore/health` and gives the local URL.
4. If it reports an existing listener, use that process’s URL and logs; do not launch another copy.
5. On Overview, check that the data cache state, model trust state, simulation backend, and history count match this machine.

The app is intended for local use at `http://127.0.0.1:8502`.

## Suggested 5-minute route

1. **Overview** — show current local system state. Explain that missing dataset caches stay missing and the sample dataset is synthetic.
2. **Dataset explorer** — open the OSIRIS sample and show the synthetic-label disclosure; inspect a schema tab.
3. **Model evaluation** — show model trust labels and saved metric limits. Point out the quarantined DRC classifier and negative-R² geometry model.
4. **Layout prediction** — adjust one sizing control. Describe its electrical numbers as synthetic-model exploratory estimates, not SPICE results. Show that DRC remains “Not verified.”
5. **AI engineering assistant** — demonstrate greetings/capability questions without credentials. If a provider is configured, use only a non-confidential screenshot with explicit consent. Show the returned provenance and limitations. A provider request is optional and may incur account usage.
6. **Electrical simulation** — verify backend status and run one small AC/transient analysis. Distinguish the ngspice result from the screenshot/model estimates and note the empirical device-model assumptions.
7. **System information** — finish with runtime, backend, data, and provider configuration diagnostics.

## Acceptance checklist

- All eight navigation entries open.
- `/_stcore/health` returns HTTP 200.
- Simulation page reports the configured DLL backend and a minimal AC/transient run produces parsed vectors.
- A screenshot provider response is shown only if the selected provider was explicitly requested and its live call succeeded.
- No view labels a prediction as actual DRC, LVS, extracted parasitics, or measured silicon.

## If startup fails

Review the timestamped files in `scratch/logs/`. Confirm the reported listener and port before retrying. The launcher does not reinstall KiCad/ngspice and does not modify permanent Windows environment variables. The known compatible DLL path is listed in `PROVIDER_SETUP.md` and `KNOWN_LIMITATIONS.md`.
