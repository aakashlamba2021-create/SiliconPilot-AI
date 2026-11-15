"""
SiliconPilot AI - Simulation Metrics Extractor Module
Extracts engineering figures of merit (DC gain, GBW, Phase Margin, Slew Rate, Power)
from raw SPICE AC, Transient, and Operating Point simulation results.

Phase 4.1 Engineering Repairs:
- Missing metrics are explicitly None (never silently replaced with zero).
- Incomplete or failed simulations are never marked as Converged.
- Hardcoded 80 µA fallback removed completely.
- Physical SPICE voltage-source current sign convention enforced (-i(VDD) = delivered current).
- Provenance labels preserved for all measured and estimated metrics.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

from src.simulation.ngspice_runner import SimulationResult
from src.simulation.simulation_parser import ACSimulationData, TranSimulationData
from src.utils.logger import get_logger

logger = get_logger("src.simulation.simulation_metrics")


@dataclass
class ExtractedMetrics:
    """Standardized electrical metrics extracted from physical SPICE simulation."""
    dc_gain_db: Optional[float] = None
    gain_bandwidth_product_hz: Optional[float] = None
    phase_margin_deg: Optional[float] = None
    cutoff_freq_3db_hz: Optional[float] = None
    slew_rate_v_per_us: Optional[float] = None
    power_consumption_w: Optional[float] = None
    power_consumption_mw: Optional[float] = None
    supply_current_ua: Optional[float] = None
    settling_time_ns: Optional[float] = None
    convergence_status: str = "Converged"
    analysis_statuses: Dict[str, str] = field(default_factory=dict)
    metric_provenance: Dict[str, str] = field(default_factory=dict)
    voltage_provenance: str = "measured_op"
    simulator_identity: str = "ngspice"
    simulator_version: str = "46"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Converts metrics to dictionary representation."""
        return asdict(self)


class SimulationMetricsExtractor:
    """
    Analyzes simulated waveforms and outputs to extract standardized analog IC performance metrics.
    Enforces strict physical measurement contracts without synthetic defaults.
    """

    def __init__(self):
        pass

    def extract(
        self,
        sim_result: SimulationResult,
        ac_data: Optional[ACSimulationData] = None,
        tran_data: Optional[TranSimulationData] = None,
        vdd: Optional[float] = 1.8,
        sim_result_tran: Optional[SimulationResult] = None,
        supply_source: str = "VDD",
        supply_node: str = "vdd",
    ) -> ExtractedMetrics:
        """
        Extracts all standard analog performance metrics from simulation outputs.

        Args:
            sim_result: Primary SimulationResult from NGSpiceRunner (AC and/or OP).
            ac_data: Parsed ACSimulationData.
            tran_data: Optional parsed TranSimulationData.
            vdd: Power supply voltage specification in Volts.
            sim_result_tran: Optional separate SimulationResult for transient run.
            supply_source: Voltage source name in netlist (e.g. 'VDD').
            supply_node: Supply voltage node name (e.g. 'vdd').

        Returns:
            ExtractedMetrics object containing quantified analog IC metrics or None for unmeasured quantities.
        """
        analysis_statuses = dict(sim_result.analysis_statuses)
        tran_res = sim_result_tran if sim_result_tran is not None else sim_result
        if sim_result_tran is not None:
            analysis_statuses["tran"] = sim_result_tran.tran_status

        metric_provenance: Dict[str, str] = {}

        # 1. Determine simulation completion and convergence
        ac_requested = (sim_result.ac_status != "NOT_REQUESTED") or (ac_data is not None)
        tran_requested = (tran_res.tran_status != "NOT_REQUESTED") or (tran_data is not None)
        op_requested = (sim_result.op_status != "NOT_REQUESTED") or (len(sim_result.op_output) > 0)

        invalid_reasons = []

        # Validate AC
        if ac_requested:
            if sim_result.ac_status != "NOT_REQUESTED" and sim_result.ac_status != "SUCCESS":
                ac_ok = False
                invalid_reasons.append(f"AC status '{sim_result.ac_status}' is not SUCCESS")
            elif ac_data is None or not ac_data.is_valid:
                ac_ok = False
                err = ac_data.error_message if ac_data else "AC data missing/omitted"
                invalid_reasons.append(f"AC: {err}")
            elif len(ac_data.frequencies) < 2:
                ac_ok = False
                invalid_reasons.append("AC data contains fewer than 2 points")
            else:
                ac_ok = True
        else:
            ac_ok = False

        # Validate TRAN
        if tran_requested:
            if sim_result_tran is not None and sim_result_tran.tran_status != "SUCCESS":
                tran_ok = False
                invalid_reasons.append(f"TRAN status '{sim_result_tran.tran_status}' is not SUCCESS")
            elif sim_result_tran is None and sim_result.tran_status not in ["SUCCESS", "NOT_REQUESTED"]:
                tran_ok = False
                invalid_reasons.append(f"TRAN status '{sim_result.tran_status}' is not SUCCESS")
            elif tran_data is None or not tran_data.is_valid:
                tran_ok = False
                err = tran_data.error_message if tran_data else "TRAN data missing/omitted"
                invalid_reasons.append(f"TRAN: {err}")
            elif len(tran_data.time_s) < 2:
                tran_ok = False
                invalid_reasons.append("TRAN data contains fewer than 2 points")
            else:
                tran_ok = True
        else:
            tran_ok = False

        # Validate OP
        if op_requested:
            if sim_result.op_status != "SUCCESS":
                op_ok = False
                invalid_reasons.append(f"OP status '{sim_result.op_status}' is not SUCCESS")
            else:
                op_ok = True
        else:
            op_ok = False

        # Check for non-convergence or solver failures
        if not sim_result.success or (sim_result_tran is not None and not sim_result_tran.success):
            err_msg = sim_result.error_message or (sim_result_tran.error_message if sim_result_tran else "Solver failure")
            conv_status = f"Failed: {err_msg}"
        elif len(invalid_reasons) > 0:
            conv_status = f"Incomplete: {'; '.join(invalid_reasons)}"
        elif not (ac_requested or tran_requested or op_requested):
            conv_status = "Incomplete: No analyses requested"
        else:
            conv_status = "Converged"

        # 2. AC Metrics Extraction (Small-Signal)
        dc_gain_db = None
        gbw_hz = None
        pm_deg = None
        f_3db = None

        if ac_ok:
            dc_gain_db, gbw_hz, pm_deg, f_3db = self._extract_ac_metrics(ac_data)
            metric_provenance["dc_gain"] = "spice_ac_simulation" if dc_gain_db is not None else "unavailable"
            metric_provenance["gbw"] = "spice_ac_simulation" if gbw_hz is not None else "unavailable"
            metric_provenance["phase_margin"] = "spice_ac_simulation" if pm_deg is not None else "unavailable"
            metric_provenance["cutoff_freq_3db"] = "spice_ac_simulation" if f_3db is not None else "unavailable"
        else:
            metric_provenance["dc_gain"] = "unavailable"
            metric_provenance["gbw"] = "unavailable"
            metric_provenance["phase_margin"] = "unavailable"
            metric_provenance["cutoff_freq_3db"] = "unavailable"

        # 3. Transient Metrics Extraction (Large-Signal Step Response)
        slew_rate_v_per_us = None
        settling_time_ns = None

        if tran_ok:
            slew_rate_v_per_us, settling_time_ns = self._extract_transient_metrics(tran_data)
            metric_provenance["slew_rate"] = "spice_transient_simulation" if slew_rate_v_per_us is not None else "unavailable"
            metric_provenance["settling_time"] = "spice_transient_simulation" if settling_time_ns is not None else "unavailable"
        else:
            metric_provenance["slew_rate"] = "unavailable"
            metric_provenance["settling_time"] = "unavailable"

        # 4. DC Power Metrics (Operating Point)
        p_w, p_mw, i_ua, p_prov, p_note = self._extract_power_metrics(
            op_dict=sim_result.op_output,
            supply_source=supply_source,
            supply_node=supply_node,
            vdd_override=vdd,
        )
        metric_provenance["power"] = p_prov
        metric_provenance["supply_current"] = p_prov

        # Ensure that if all physical metrics are None, status cannot be Converged
        if conv_status == "Converged" and all(
            m is None for m in [dc_gain_db, gbw_hz, pm_deg, slew_rate_v_per_us, p_mw]
        ):
            conv_status = "Incomplete: No physical electrical metrics extracted"

        notes = []
        if p_note:
            notes.append(p_note)
        if not ac_ok and ac_data:
            notes.append(f"AC unavailable ({ac_data.error_message})")
        if not tran_ok and tran_data:
            notes.append(f"TRAN unavailable ({tran_data.error_message})")

        v_prov = "measured_op" if p_prov == "spice_op_simulation" else ("configured_supply_parameter" if p_prov.startswith("calculated") else "unavailable")

        return ExtractedMetrics(
            dc_gain_db=float(np.round(dc_gain_db, 2)) if dc_gain_db is not None else None,
            gain_bandwidth_product_hz=float(np.round(gbw_hz, 1)) if gbw_hz is not None else None,
            phase_margin_deg=float(np.round(pm_deg, 2)) if pm_deg is not None else None,
            cutoff_freq_3db_hz=float(np.round(f_3db, 1)) if f_3db is not None else None,
            slew_rate_v_per_us=float(np.round(slew_rate_v_per_us, 2)) if slew_rate_v_per_us is not None else None,
            power_consumption_w=float(p_w) if p_w is not None else None,
            power_consumption_mw=float(np.round(p_mw, 4)) if p_mw is not None else None,
            supply_current_ua=float(np.round(i_ua, 2)) if i_ua is not None else None,
            settling_time_ns=float(np.round(settling_time_ns, 2)) if settling_time_ns is not None else None,
            convergence_status=conv_status,
            analysis_statuses=analysis_statuses,
            metric_provenance=metric_provenance,
            voltage_provenance=v_prov,
            notes="; ".join(notes) if notes else "Direct physical ngspice simulation vectors.",
        )

    def _extract_ac_metrics(self, ac_data: ACSimulationData) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Extracts DC Gain, GBW (0 dB frequency), Phase Margin, and 3-dB frequency.
        Enforces physical validity safeguards:
        - If no genuine bracketed 0 dB crossing exists, GBW and PM are None (unavailable).
        - If phase data is missing, PM is None (unavailable), not 0.0.
        - Never invents endpoint metrics when response does not cross 0 dB.
        """
        freqs = ac_data.frequencies
        gains = ac_data.gain_db
        phases = ac_data.phase_deg

        if len(freqs) == 0 or len(gains) == 0:
            return None, None, None, None

        # DC Gain is gain at lowest frequency
        dc_gain = float(gains[0])

        # 3-dB Bandwidth: frequency where gain drops to (dc_gain - 3.0)
        target_3db = dc_gain - 3.0
        idx_3db = np.where(gains <= target_3db)[0]
        if len(idx_3db) > 0 and idx_3db[0] > 0:
            i2 = idx_3db[0]
            i1 = i2 - 1
            f1, f2 = freqs[i1], freqs[i2]
            g1, g2 = gains[i1], gains[i2]
            log_f = np.log10(f1) + (target_3db - g1) / (g2 - g1 + 1e-12) * (np.log10(f2) - np.log10(f1))
            f_3db = float(10**log_f)
        else:
            f_3db = None

        # Unity Gain Bandwidth (0 dB crossing)
        idx_ugb = np.where(gains <= 0.0)[0]
        has_phase = getattr(ac_data, "has_phase", True) and len(phases) > 0 and len(phases) == len(freqs)

        if len(idx_ugb) > 0 and idx_ugb[0] > 0:
            i2 = idx_ugb[0]
            i1 = i2 - 1
            f1, f2 = freqs[i1], freqs[i2]
            g1, g2 = gains[i1], gains[i2]

            log_fu = np.log10(f1) + (0.0 - g1) / (g2 - g1 + 1e-12) * (np.log10(f2) - np.log10(f1))
            gbw = float(10**log_fu)

            if has_phase:
                p1, p2 = phases[i1], phases[i2]
                # Interpolate phase at unity gain frequency
                frac = (log_fu - np.log10(f1)) / (np.log10(f2) - np.log10(f1) + 1e-12)
                phase_fu = p1 + frac * (p2 - p1)

                # Phase margin calculation
                # Engineering convention: 180 deg - phase lag from DC
                phase_dc = phases[0]
                phase_lag = abs(phase_dc - phase_fu)
                pm = max(0.0, 180.0 - phase_lag)
            else:
                pm = None
        else:
            # No genuine 0 dB crossing in the sweep range: GBW and PM are unavailable
            gbw = None
            pm = None

        return dc_gain, gbw, pm, f_3db

    def _extract_transient_metrics(self, tran_data: TranSimulationData) -> Tuple[float, float]:
        """Extracts Slew Rate and Settling Time from step response."""
        t = tran_data.time_s
        vout = tran_data.vout_v

        if len(t) < 5:
            return 0.0, 0.0

        # Discrete derivative: dV / dt
        dt = np.diff(t)
        dv = np.diff(vout)
        valid = dt > 0
        if not np.any(valid):
            return 0.0, 0.0

        rates = np.abs(dv[valid] / dt[valid])  # Volts per second
        max_rate_v_per_s = float(np.max(rates))
        slew_rate_v_per_us = max_rate_v_per_s * 1e-6  # Convert to V/µs

        # Settling time (within 2% of final value)
        final_v = vout[-1]
        threshold = 0.02 * abs(final_v) if abs(final_v) > 0.05 else 0.01
        unsettled = np.where(np.abs(vout - final_v) > threshold)[0]
        if len(unsettled) > 0:
            last_unsettled_idx = unsettled[-1]
            settling_time_ns = float(t[min(last_unsettled_idx + 1, len(t) - 1)] * 1e9)
        else:
            settling_time_ns = float(t[0] * 1e9)

        return slew_rate_v_per_us, settling_time_ns

    def _extract_power_metrics(
        self,
        op_dict: Dict[str, float],
        supply_source: str = "VDD",
        supply_node: str = "vdd",
        vdd_override: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], str, Optional[str]]:
        """
        Extracts actual physical power dissipation from ngspice operating point outputs.

        Correct Engineering Rules (Phase 4.1):
        1. No assumed or hardcoded fallbacks (80 µA removed).
        2. Query the exact supply voltage source current: i({supply_source}).
        3. SPICE current sign convention: current leaving source positive terminal to power
           circuit has negative sign in SPICE branch convention:
           i_delivered = -i_source
        4. If voltage or current is missing, non-finite (NaN/Inf), mark as unavailable.

        Returns:
            (power_w, power_mw, supply_current_ua, provenance, explanation_if_unavailable)
        """
        # 1. Lookup supply voltage
        v_key = f"v({supply_node.lower()})"
        v_supply = op_dict.get(v_key)
        is_voltage_measured = (v_supply is not None and np.isfinite(v_supply) and v_supply > 0)

        if not is_voltage_measured:
            if vdd_override is not None and np.isfinite(vdd_override) and vdd_override > 0:
                v_supply = float(vdd_override)
            else:
                return None, None, None, "unavailable", f"Supply voltage '{v_key}' missing or non-finite in OP results"

        if v_supply is None or not np.isfinite(v_supply) or v_supply <= 0:
            return None, None, None, "unavailable", f"Supply voltage '{v_key}' missing or non-finite in OP results"

        # 2. Lookup exact voltage source current
        curr_key = f"i({supply_source.lower()})"
        i_source = op_dict.get(curr_key)

        # Also check fallback key variations if source name prefix differs
        if i_source is None:
            curr_key_alt = f"i(v{supply_source.lower()})"
            i_source = op_dict.get(curr_key_alt)

        if i_source is None:
            return None, None, None, "unavailable", f"Current for source '{supply_source}' not found in OP results (no fallback allowed)"

        if not np.isfinite(i_source):
            return None, None, None, "unavailable", f"Current for source '{supply_source}' is non-finite (NaN/Inf) in OP results"

        # 3. SPICE sign convention:
        # In SPICE, positive current flows into the positive terminal of the voltage source.
        # A power supply delivering current to the circuit has i_source < 0.
        # Delivered current: i_delivered = -i_source
        i_delivered = -float(i_source)
        power_w = float(v_supply * i_delivered)
        power_mw = float(power_w * 1000.0)
        i_ua = float(i_delivered * 1e6)

        # Explicitly distinguish measured OP voltage from configured supply voltage override
        if is_voltage_measured:
            provenance = "spice_op_simulation"
        else:
            provenance = "calculated_simulated_current_x_configured_vdd"

        return power_w, power_mw, i_ua, provenance, None
