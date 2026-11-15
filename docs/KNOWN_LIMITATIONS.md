# SiliconPilot AI — Known Limitations & Evidence Boundaries

An essential principle of **SiliconPilot AI** is technical honesty and engineering transparency. This document details the exact evidence boundaries, model trust classifications, and simulation scope of the current system.

---

## 1. Machine Learning Model Boundaries

The project maintains a declarative trust policy in `config/model_trust_policy.json`, enforced programmatically via `src/models/model_policy.py`.

### 1.1. Layout Verification Classifier (`layout_verification_classifier`)
* **Status:** 🔴 **QUARANTINED** (`permitted_uses: []`)
* **Data Origin:** Synthetic labels from `data/sample/generate_sample.py`.
* **Root Cause for Quarantine:** The classifier achieved 84.76% accuracy, 84.76% precision, and 100% recall. This mathematical profile proves that the model collapsed into predicting `drc_clean = True` for 100% of inputs, capitalizing on an 85% class imbalance in the synthetic generator.
* **Engineering Impact:** The model does not execute DRC or LVS decks (e.g., Calibre, Pegasus, KLayout). Calling its predictions "DRC Clean" presents a severe false-positive hazard and has been strictly quarantined from all inference.

### 1.2. Layout Geometry Predictor (`iclr_layout_geometry_predictor`)
* **Status:** 🟡 **RESTRICTED** (`permitted_uses: ["metadata_review"]`)
* **Data Origin:** ICLR-ADAPT academic benchmark (`index.csv`).
* **Root Cause for Restriction:** Saved evaluation reports demonstrate negative $R^2$ scores across all four target dimensions:
  * Total Area ($A_{eval}$): $R^2 = -8.572$
  * Signal Wirelength ($L_{eq,signal}$): $R^2 = -4.524$
  * Power Wirelength ($L_{eq,power}$): $R^2 = -1.531$
  * Intermetal Cuts ($N_{cut}$): $R^2 = -0.352$
* **Engineering Impact:** An $R^2 < 0$ proves the model performs worse than simply predicting the mean of the training data. It is blocked from steering candidate optimization ranking.

### 1.3. Electrical Performance Predictor (`osiris_electrical_performance_predictor`)
* **Status:** 🔵 **EXPLORATORY** (`permitted_uses: ["exploratory_candidate_generation", "metadata_review"]`)
* **Data Origin:** 500-sample synthetic dataset (`data/sample/osiris_sample.csv`).
* **Limitations:** The model was trained on algebraic NumPy formulas ($W/L$ ratios and linear offsets) rather than physical SPICE sweeps. While useful for rapid exploration and candidate ranking demonstration, its outputs must not be confused with physical silicon or SPICE measurements.

### 1.4. Area & PEX Predictor (`osiris_area_pex_predictor`)
* **Status:** 🔵 **EXPLORATORY** (`permitted_uses: ["exploratory_analysis", "metadata_review"]`)
* **Data Origin:** Official OSIRIS benchmark (`variants.parquet`, 89,108 variants).
* **Limitations:** The model exhibits strong predictive correlation ($R^2 = 0.822$ on area, $R^2 = 0.914$ on mid-PEX score). However, the `pex_score` targets represent relative benchmark scores, not physical parasitic extraction netlists from user layouts.

---

## 2. SPICE Simulation & Semiconductor Physics Boundaries

1. **Empirical Level-1 Device Models:** The bundled 180 nm models (`NMOS_180`, `PMOS_180`) in `NetlistManager` are textbook Level-1 Shichman-Hodges square-law models ($KP, \lambda, \gamma, V_{TO}$). They do not include short-channel effects, subthreshold slope, DIBL, gate leakage, or layout-dependent effects (WPE/LOD) found in BSIM3v3, BSIM4, or foundry PDKs.
2. **Lumped Parasitic Loading:** The "PEX" sliders in the Electrical Simulation Studio inject lumped RC elements into the netlist output nodes. They do not represent physical layout extraction.
3. **Convergence Safeguards:** Extremely wide or narrow transistor aspect ratios ($W/L < 0.1$ or $W/L > 1000$) may encounter DC operating-point non-convergence in ngspice.

---

## 3. Computer Vision & Layout Inspection Boundaries

1. **Optical Resolution Limitations:** A standard 1080p screenshot represents each pixel as tens or hundreds of nanometers. Visual inspection can never establish sub-micron clearance or enclosure compliance.
2. **Layer Ambiguity:** Overlapping translucent metal and via layers in layout screenshots blend optically. Vision models cannot trace netlist connectivity or verify LVS equivalence from an image.
3. **PDK Color Non-Standardization:** Color palettes for metal, poly, and diffusion layers differ across tools and foundries. Vision outputs are strictly advisory.
4. **Evidence Classification:** All vision assistant outputs must be interpreted according to the 4-part Evidence Hierarchy:
   * **[VISIBLE OBSERVATIONS]:** Optically confirmed topological features.
   * **[ENGINEERING INFERENCES]:** Theoretical principles (e.g., gradient cancellation).
   * **[UNVERIFIED RISKS]:** Unverified physical and DRC rules.
   * **[RECOMMENDATIONS]:** Advisory layout heuristics.

---

## 4. EDA File Ingestion Boundaries

* The EDA artifact inspector in `src/eda/inspection.py` provides **read-only static metadata extraction**.
* Uploaded GDSII, OASIS, LEF, DEF, or SPICE netlist files are parsed for cell hierarchy, pin counts, and boundary metadata.
* Uploaded SPICE netlists are **never executed** by the artifact inspector to prevent arbitrary code execution or unvetted simulation loads.
