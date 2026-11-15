# SiliconPilot AI Presentation Script

## Opening

“SiliconPilot AI is a local analog IC engineering workbench. It brings dataset inspection, model provenance, exploratory candidate comparison, screenshot-based discussion, read-only artifact inspection, and ngspice experiments into one Streamlit application.”

## Evidence-first model review

“The model cards keep the data origin next to saved metrics. The electrical predictor is trained on synthetic equations, so its output is exploratory. The geometry predictor has negative saved R² for all four targets and is restricted. The DRC classifier was trained on synthetic labels and is quarantined, so the application reports actual DRC as not verified. The OSIRIS area/PEX model is a benchmark estimate, not a new extraction.”

## Assistant

“Text help works locally. For image analysis, I choose a provider and explicitly consent before any image leaves this machine. The response separates visible observations from recommendations and identifies the provider, limitations, and follow-up verification steps. It cannot infer signoff, exact measurements, or connectivity from a screenshot.”

If no provider is configured, demonstrate the text conversation and explain that live vision analysis is unavailable until credentials and provider access are configured. Do not imply that a mock test is a live API call.

## Simulation

“The simulation page uses the local KiCad 10 ngspice DLL with an empirical demonstration model. These results are actual runs of the submitted circuit under that model, but they are not foundry PDK or extracted-layout signoff.”

## Closing

“The central design principle is evidence provenance: a benchmark, synthetic estimate, simulation, and measured value are different evidence types, and unavailable evidence remains unavailable.”
