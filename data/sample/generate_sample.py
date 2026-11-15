"""
generate_sample.py
Generates 500 synthetic OSIRIS-like analog IC layout samples as a CSV.
Uses only numpy and pandas. Targets are physically correlated with inputs.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)
N = 500

OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "osiris_sample.csv"

# ── Input Feature Generation ─────────────────────────────────────────────────
topologies = ["miller_2stage", "folded_cascode", "telescopic"]
topology = RNG.choice(topologies, size=N)

transistor_width_W = RNG.uniform(0.5, 50.0, size=N)       # μm
transistor_length_L = RNG.uniform(0.18, 2.0, size=N)      # μm
finger_count = RNG.integers(1, 17, size=N)                 # 1–16 inclusive
multiplicity = RNG.integers(1, 9, size=N)                  # 1–8 inclusive
finger_arrangement = RNG.integers(1, 5, size=N)            # 1–4 inclusive

x_position = RNG.uniform(0.0, 100.0, size=N)              # μm
y_position = RNG.uniform(0.0, 100.0, size=N)              # μm
orientation = RNG.choice([0, 90, 180, 270], size=N)

delta_x = RNG.normal(0.0, 2.0, size=N)                    # μm
delta_y = RNG.normal(0.0, 2.0, size=N)                    # μm
perturbation_id = RNG.integers(0, 10, size=N)             # 0–9 inclusive

# ── Derived / Physics-Correlated Targets ─────────────────────────────────────

# W/L ratio: key transistor performance driver
wl_ratio = transistor_width_W / transistor_length_L        # dimensionless

# Topology scaling factors (folded cascode and telescopic have higher gain)
topo_gain_factor = np.where(
    topology == "miller_2stage", 1.0,
    np.where(topology == "folded_cascode", 1.2, 1.4)       # telescopic highest
)
topo_gbw_factor = np.where(
    topology == "miller_2stage", 1.0,
    np.where(topology == "folded_cascode", 1.5, 1.3)
)

# dc_gain: 40–80 dB — increases with W/L ratio, topology factor
# Base: scale W/L (range ~0.25–278) to 40–70 dB range, then multiply by topo factor
wl_norm = np.clip((np.log1p(wl_ratio) / np.log1p(278)) * 30, 0, 30)  # 0–30 dB range
dc_gain_base = 40.0 + wl_norm * topo_gain_factor
dc_gain_noise = RNG.normal(0.0, 2.0, size=N)
dc_gain = np.clip(dc_gain_base + dc_gain_noise, 40.0, 80.0)

# gain_bandwidth_product: 1e6–100e6 Hz
# Higher W/L and lower L → better frequency response; more fingers reduce it slightly
gbw_base = (
    1e6 +
    (wl_ratio / 278) * 80e6 * topo_gbw_factor
    - finger_count * 0.5e6
)
gbw_noise = RNG.normal(0.0, 3e6, size=N)
gain_bandwidth_product = np.clip(gbw_base + gbw_noise, 1e6, 100e6)

# phase_margin: 45–90 degrees
# Higher multiplicity and larger fingers tend to load the output → reduced PM
pm_base = 75.0 - (finger_count - 1) * 1.2 - (multiplicity - 1) * 1.5
pm_noise = RNG.normal(0.0, 4.0, size=N)
phase_margin = np.clip(pm_base + pm_noise, 45.0, 90.0)

# power_consumption: 1e-5–1e-3 W — increases with W (wider = more current)
power_base = 1e-5 + (transistor_width_W / 50.0) * 9.9e-4
power_noise = RNG.normal(0.0, 5e-5, size=N)
power_consumption = np.clip(power_base + power_noise, 1e-5, 1e-3)

# area: W * L * finger_count * multiplicity * 2.5 + noise (μm²)
area_base = transistor_width_W * transistor_length_L * finger_count * multiplicity * 2.5
area_noise = RNG.normal(0.0, area_base * 0.05)  # 5% relative noise
area = np.clip(area_base + area_noise, 0.1, None)

# slew_rate: 1–50 V/μs — correlated with GBW / capacitive load
sr_base = 1.0 + (gain_bandwidth_product / 100e6) * 45.0
sr_noise = RNG.normal(0.0, 2.0, size=N)
slew_rate = np.clip(sr_base + sr_noise, 1.0, 50.0)

# cmrr: 60–100 dB — better with matched devices (lower perturbations, more fingers)
perturbation_magnitude = np.sqrt(delta_x**2 + delta_y**2)
cmrr_base = 90.0 - perturbation_magnitude * 3.0 + (finger_count - 1) * 0.5
cmrr_noise = RNG.normal(0.0, 3.0, size=N)
cmrr = np.clip(cmrr_base + cmrr_noise, 60.0, 100.0)

# drc_clean: True for 85% of samples; correlated with delta magnitude
# Large deltas (> 4μm) are more likely to cause DRC violations
drc_prob = np.where(perturbation_magnitude > 4.0, 0.60, 0.90)
drc_rand = RNG.uniform(0.0, 1.0, size=N)
drc_clean = drc_rand < drc_prob                            # bool

# lvs_verified: True where drc_clean=True, 95% of those
lvs_rand = RNG.uniform(0.0, 1.0, size=N)
lvs_verified = drc_clean & (lvs_rand < 0.95)              # bool

# ── Assemble DataFrame ───────────────────────────────────────────────────────
df = pd.DataFrame({
    # Categorical
    "topology": topology,
    # Transistor sizing
    "transistor_width_W": transistor_width_W.round(4),
    "transistor_length_L": transistor_length_L.round(4),
    "finger_count": finger_count,
    "multiplicity": multiplicity,
    "finger_arrangement": finger_arrangement,
    # Placement geometry
    "x_position": x_position.round(4),
    "y_position": y_position.round(4),
    "orientation": orientation,
    "delta_x": delta_x.round(4),
    "delta_y": delta_y.round(4),
    "perturbation_id": perturbation_id,
    # Targets — performance metrics
    "dc_gain": dc_gain.round(4),
    "gain_bandwidth_product": gain_bandwidth_product.round(2),
    "phase_margin": phase_margin.round(4),
    "power_consumption": power_consumption.round(8),
    "area": area.round(4),
    "slew_rate": slew_rate.round(4),
    "cmrr": cmrr.round(4),
    # Targets — verification
    "drc_clean": drc_clean,
    "lvs_verified": lvs_verified,
})

# ── Save ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Generated {len(df)} samples → {OUTPUT_FILE}")
print(f"   Columns ({len(df.columns)}): {list(df.columns)}")
print(f"   DRC clean: {df['drc_clean'].sum()} / {len(df)} ({df['drc_clean'].mean()*100:.1f}%)")
print(f"   LVS verified: {df['lvs_verified'].sum()} / {len(df)} ({df['lvs_verified'].mean()*100:.1f}%)")
print("\nSample statistics:")
print(df[["dc_gain", "gain_bandwidth_product", "phase_margin", "power_consumption",
          "area", "slew_rate", "cmrr"]].describe().round(4).to_string())
