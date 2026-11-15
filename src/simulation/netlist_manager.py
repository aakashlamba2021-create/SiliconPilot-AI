"""
SiliconPilot AI - Netlist Manager Module
Generates parameterized SPICE netlists for analog IC topologies, mapping candidate
sizing parameters to transistor devices, bias networks, compensation, and parasitic loads.
"""

from typing import Dict, Any, Optional
from src.utils.logger import get_logger

logger = get_logger("src.simulation.netlist_manager")

# Standard 180nm CMOS Model Library (Predictive/Benchmark)
# Strictly calibrated standard models for SPICE simulation.
# NOTE: These are standard benchmark device models, not a proprietary foundry PDK.
STANDARD_180NM_MODELS = """
* ==============================================================================
* SiliconPilot Standard Calibrated 180nm CMOS Model Library
* Type: Benchmark Empirical SPICE Models (Predictive Level-1 / BSIM Calibrated)
* NOTE: Non-proprietary open models for academic & comparative validation.
* ==============================================================================
.model NMOS_180 NMOS (
+ LEVEL=1 
+ VTO=0.45 
+ KP=170U 
+ GAMMA=0.4 
+ LAMBDA=0.05 
+ CJ=0.0008 
+ CJSW=0.0003 
+ PB=0.8
+ CGSO=2.5E-10
+ CGDO=2.5E-10
)

.model PMOS_180 PMOS (
+ LEVEL=1 
+ VTO=-0.45 
+ KP=50U 
+ GAMMA=0.4 
+ LAMBDA=0.08 
+ CJ=0.0008 
+ CJSW=0.0003 
+ PB=0.8
+ CGSO=2.5E-10
+ CGDO=2.5E-10
)
"""


class NetlistManager:
    """
    Manages parameterized analog IC SPICE netlist generation for circuit simulation.
    Maps ML candidate features to physical circuit devices and testbenches.
    """

    SUPPORTED_TOPOLOGIES = [
        "miller_2stage",
        "five_transistors_ota",
        "folded_cascode",
        "telescopic",
    ]

    def __init__(self):
        pass

    def get_supported_topologies(self) -> list:
        """Returns the list of topologies with valid netlist templates."""
        return list(self.SUPPORTED_TOPOLOGIES)

    def generate_netlist(
        self,
        topology: str,
        params: Dict[str, Any],
        mode: str = "ac",
        ac_sweep: str = "dec 20 100 100Meg",
        tran_params: str = "0.1n 40n",
        pex_parasitics: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Generates a complete, ready-to-run SPICE netlist for the given circuit topology and parameters.

        Args:
            topology: Circuit topology name ('miller_2stage', 'five_transistors_ota', etc.)
            params: Dictionary containing sizing parameters:
                    - transistor_width_W: float (microns)
                    - transistor_length_L: float (microns)
                    - finger_count: int
                    - multiplicity: int
                    - ibias_ua: optional float (microamps)
                    - cload_pf: optional float (picofarads)
                    - ccomp_pf: optional float (picofarads, for 2-stage)
                    - vdd: optional float (Volts)
            mode: 'ac' (open-loop frequency sweep), 'tran' (closed-loop step response), or 'both'.
            ac_sweep: SPICE AC sweep arguments.
            tran_params: SPICE TRAN analysis arguments.
            pex_parasitics: Optional dictionary of layout parasitics (e.g. c_pex_out, r_pex_out).

        Returns:
            SPICE netlist string.
        """
        top = topology.lower().strip()
        if top not in self.SUPPORTED_TOPOLOGIES:
            if "miller" in top:
                top = "miller_2stage"
            elif "5t" in top or "five" in top or "ota_ff" in top or "ota_lpf" in top:
                top = "five_transistors_ota"
            elif "folded" in top:
                top = "folded_cascode"
            elif "telescopic" in top:
                top = "telescopic"
            else:
                top = "miller_2stage"

        if top == "miller_2stage":
            return self._build_miller_2stage(params, mode, ac_sweep, tran_params, pex_parasitics)
        elif top == "five_transistors_ota":
            return self._build_five_transistors_ota(params, mode, ac_sweep, tran_params, pex_parasitics)
        elif top == "folded_cascode":
            return self._build_folded_cascode(params, mode, ac_sweep, tran_params, pex_parasitics)
        elif top == "telescopic":
            return self._build_telescopic(params, mode, ac_sweep, tran_params, pex_parasitics)
        else:
            return self._build_miller_2stage(params, mode, ac_sweep, tran_params, pex_parasitics)

    def _extract_sizes(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Safely extracts and standardizes sizing parameters."""
        w = float(params.get("transistor_width_W", 35.0))
        l = float(params.get("transistor_length_L", 0.5))
        m = float(params.get("multiplicity", 2.0))
        nf = float(params.get("finger_count", 4.0))

        # Clamp to realistic analog IC fabrication limits
        w = max(2.0, min(200.0, w))
        l = max(0.18, min(5.0, l))
        m = max(1.0, min(16.0, m))
        nf = max(1.0, min(32.0, nf))

        ibias_ua = float(params.get("ibias_ua", 20.0))
        cload_pf = float(params.get("cload_pf", 1.5))
        ccomp_pf = float(params.get("ccomp_pf", max(0.5, 0.03 * w * m)))
        vdd = float(params.get("vdd", 1.8))

        return {
            "w": w,
            "l": l,
            "m": m,
            "nf": nf,
            "ibias_ua": ibias_ua,
            "cload_pf": cload_pf,
            "ccomp_pf": ccomp_pf,
            "vdd": vdd,
            "vcm": vdd / 2.0,
        }

    def _build_miller_2stage(
        self,
        params: Dict[str, Any],
        mode: str,
        ac_sweep: str,
        tran_params: str,
        pex: Optional[Dict[str, float]],
    ) -> str:
        """Constructs Two-Stage Miller Compensated Operational Amplifier Netlist."""
        s = self._extract_sizes(params)
        w_in = s["w"] * s["m"]
        w_load = max(15.0, s["w"] * 1.2)
        w_tail = 20.0
        l_tail = 1.0
        w_stage2_p = 2.0 * w_load
        w_stage2_n = 2.0 * w_tail

        pex_block = ""
        out_node = "outp"
        if pex:
            c_pex = pex.get("c_pex_out", 0.0)
            if c_pex > 0:
                pex_block += f"CPEX_OUT outp 0 {c_pex:.4e}p\n"

        # Signal sources based on mode
        if mode == "tran":
            sources = f"""* Closed-Loop Step Response Sources
VIN vin vcm pulse(-0.2 0.2 5n 0.5n 0.5n 20n 50n)
RFB outp vip 10
"""
        else:
            sources = """* Open-Loop Differential AC Sources
VIP vip vcm DC 0 AC -0.5
VIN vin vcm DC 0 AC 0.5
"""

        netlist = f"""* SiliconPilot AI - Two-Stage Miller OTA ({mode.upper()})
{STANDARD_180NM_MODELS}

* Power Supplies & Bias
VDD vdd 0 DC {s['vdd']}
VSS vss 0 DC 0.0
VCM vcm 0 DC {s['vcm']}
IBIAS vdd nbias DC {s['ibias_ua']}u

{sources}

* Bias Current Mirror
MNB nbias nbias vss vss NMOS_180 W=10u L={l_tail}u
MNT tail  nbias vss vss NMOS_180 W={w_tail}u L={l_tail}u

* Stage 1: Differential Input Pair (vip=inverting, vin=non-inverting)
MN1 out1 vip tail vss NMOS_180 W={w_in:.2f}u L={s['l']:.2f}u
MN2 out2 vin tail vss NMOS_180 W={w_in:.2f}u L={s['l']:.2f}u

* Stage 1: Active PMOS Current Mirror Load
MP1 out1 out1 vdd vdd PMOS_180 W={w_load:.2f}u L={s['l']:.2f}u
MP2 out2 out1 vdd vdd PMOS_180 W={w_load:.2f}u L={s['l']:.2f}u

* Stage 2: PMOS Common Source Driver & NMOS Current Load
MP6 outp out2 vdd vdd PMOS_180 W={w_stage2_p:.2f}u L={s['l']:.2f}u
MN7 outp nbias vss vss NMOS_180 W={w_stage2_n:.2f}u L={l_tail}u

* Miller Compensation & Load Capacitance
CC out2 outp {s['ccomp_pf']:.3f}p
CL {out_node} 0 {s['cload_pf']:.3f}p

* Layout Parasitics (PEX Injection)
{pex_block}

* Analyses
.op
"""
        if mode in ("ac", "both"):
            netlist += f".ac {ac_sweep}\n"
        if mode in ("tran", "both"):
            netlist += f".tran {tran_params}\n"
        netlist += ".end\n"
        return netlist

    def _build_five_transistors_ota(
        self,
        params: Dict[str, Any],
        mode: str,
        ac_sweep: str,
        tran_params: str,
        pex: Optional[Dict[str, float]],
    ) -> str:
        """Constructs Classic Five-Transistor Single-Stage Differential OTA Netlist."""
        s = self._extract_sizes(params)
        w_in = s["w"] * s["m"]
        w_load = max(15.0, s["w"] * 1.5)
        w_tail = 25.0
        l_tail = 1.0

        pex_block = ""
        out_node = "outp"
        if pex:
            c_pex = pex.get("c_pex_out", 0.0)
            if c_pex > 0:
                pex_block += f"CPEX_OUT outp 0 {c_pex:.4e}p\n"

        if mode == "tran":
            sources = f"""* Closed-Loop Step Response Sources
VIP vip vcm pulse(-0.2 0.2 5n 0.5n 0.5n 20n 50n)
RFB outp vin 10
"""
        else:
            sources = """* Open-Loop Differential AC Sources
VIP vip vcm DC 0 AC 0.5
VIN vin vcm DC 0 AC -0.5
"""

        netlist = f"""* SiliconPilot AI - Five-Transistor OTA ({mode.upper()})
{STANDARD_180NM_MODELS}

* Power Supplies & Bias
VDD vdd 0 DC {s['vdd']}
VSS vss 0 DC 0.0
VCM vcm 0 DC {s['vcm']}
IBIAS vdd nbias DC {s['ibias_ua']}u

{sources}

* Bias Current Mirror & Tail
MNB nbias nbias vss vss NMOS_180 W=10u L={l_tail}u
MNT tail  nbias vss vss NMOS_180 W={w_tail}u L={l_tail}u

* Differential Input Pair
MN1 outn vip tail vss NMOS_180 W={w_in:.2f}u L={s['l']:.2f}u
MN2 outp vin tail vss NMOS_180 W={w_in:.2f}u L={s['l']:.2f}u

* PMOS Current Mirror Load
MP1 outn outn vdd vdd PMOS_180 W={w_load:.2f}u L={s['l']:.2f}u
MP2 outp outn vdd vdd PMOS_180 W={w_load:.2f}u L={s['l']:.2f}u

* Load Capacitance
CL {out_node} 0 {s['cload_pf']:.3f}p

* Layout Parasitics (PEX Injection)
{pex_block}

* Analyses
.op
"""
        if mode in ("ac", "both"):
            netlist += f".ac {ac_sweep}\n"
        if mode in ("tran", "both"):
            netlist += f".tran {tran_params}\n"
        netlist += ".end\n"
        return netlist

    def _build_folded_cascode(
        self,
        params: Dict[str, Any],
        mode: str,
        ac_sweep: str,
        tran_params: str,
        pex: Optional[Dict[str, float]],
    ) -> str:
        """Constructs Folded Cascode OTA Netlist."""
        s = self._extract_sizes(params)
        w_in = s["w"] * s["m"]
        w_cas = s["w"] * 1.2
        w_load = s["w"] * 1.5

        if mode == "tran":
            sources = f"""* Closed-Loop Step Response Sources
VIP vip vcm pulse(-0.2 0.2 5n 0.5n 0.5n 20n 50n)
RFB outp vin 10
"""
        else:
            sources = """* Open-Loop Differential AC Sources
VIP vip vcm DC 0 AC 0.5
VIN vin vcm DC 0 AC -0.5
"""

        netlist = f"""* SiliconPilot AI - Folded Cascode OTA ({mode.upper()})
{STANDARD_180NM_MODELS}

* Supplies
VDD vdd 0 DC {s['vdd']}
VSS vss 0 DC 0.0
VCM vcm 0 DC {s['vcm']}

* Bias Voltages
VBP1 vbp1 0 DC {s['vdd'] - 0.5}
VBP2 vbp2 0 DC {s['vdd'] - 0.9}
VBN1 vbn1 0 DC 0.9
VBN2 vbn2 0 DC 0.6

{sources}

* PMOS Differential Input Pair
MP1 fold1 vip tailp vdd PMOS_180 W={w_in:.2f}u L={s['l']:.2f}u
MP2 fold2 vin tailp vdd PMOS_180 W={w_in:.2f}u L={s['l']:.2f}u
MPT tailp vbp1 vdd vdd PMOS_180 W={w_load*2:.2f}u L={s['l']:.2f}u

* Folding NMOS Current Sources & Cascodes
MNB1 fold1 vbn2 vss vss NMOS_180 W={w_load:.2f}u L={s['l']:.2f}u
MNB2 fold2 vbn2 vss vss NMOS_180 W={w_load:.2f}u L={s['l']:.2f}u
MNC1 outn  vbn1 fold1 vss NMOS_180 W={w_cas:.2f}u L={s['l']:.2f}u
MNC2 outp  vbn1 fold2 vss NMOS_180 W={w_cas:.2f}u L={s['l']:.2f}u

* PMOS Cascode Active Load
MPC1 outn  vbp2 nodep1 vdd PMOS_180 W={w_cas*1.5:.2f}u L={s['l']:.2f}u
MPC2 outp  vbp2 nodep2 vdd PMOS_180 W={w_cas*1.5:.2f}u L={s['l']:.2f}u
MPB1 nodep1 outn vdd vdd PMOS_180 W={w_load*1.5:.2f}u L={s['l']:.2f}u
MPB2 nodep2 outn vdd vdd PMOS_180 W={w_load*1.5:.2f}u L={s['l']:.2f}u

* Load
CL outp 0 {s['cload_pf']:.3f}p

.op
"""
        if mode in ("ac", "both"):
            netlist += f".ac {ac_sweep}\n"
        if mode in ("tran", "both"):
            netlist += f".tran {tran_params}\n"
        netlist += ".end\n"
        return netlist

    def _build_telescopic(
        self,
        params: Dict[str, Any],
        mode: str,
        ac_sweep: str,
        tran_params: str,
        pex: Optional[Dict[str, float]],
    ) -> str:
        """Constructs Telescopic Cascode OTA Netlist."""
        s = self._extract_sizes(params)
        w_in = s["w"] * s["m"]
        w_cas = s["w"] * 1.2
        w_load = s["w"] * 1.5

        if mode == "tran":
            sources = f"""* Closed-Loop Step Response Sources
VIP vip vcm pulse(-0.2 0.2 5n 0.5n 0.5n 20n 50n)
RFB outp vin 10
"""
        else:
            sources = """* Open-Loop Differential AC Sources
VIP vip vcm DC 0 AC 0.5
VIN vin vcm DC 0 AC -0.5
"""

        netlist = f"""* SiliconPilot AI - Telescopic Cascode OTA ({mode.upper()})
{STANDARD_180NM_MODELS}

* Supplies
VDD vdd 0 DC {s['vdd']}
VSS vss 0 DC 0.0
VCM vcm 0 DC 0.7

* Bias Voltages
VBN_TAIL vbn_tail 0 DC 0.5
VBN_CAS  vbn_cas  0 DC 0.9
VBP_CAS  vbp_cas  0 DC {s['vdd'] - 0.7}
VBP_LOAD vbp_load 0 DC {s['vdd'] - 0.4}

{sources}

* Tail
MNT tail vbn_tail vss vss NMOS_180 W={w_load*2:.2f}u L={s['l']:.2f}u

* Input Differential Pair
MN1 d1 vip tail vss NMOS_180 W={w_in:.2f}u L={s['l']:.2f}u
MN2 d2 vin tail vss NMOS_180 W={w_in:.2f}u L={s['l']:.2f}u

* NMOS Cascodes
MNC1 outn vbn_cas d1 vss NMOS_180 W={w_cas:.2f}u L={s['l']:.2f}u
MNC2 outp vbn_cas d2 vss NMOS_180 W={w_cas:.2f}u L={s['l']:.2f}u

* PMOS Active Cascode Mirror Load
MPC1 outn vbp_cas s1 vdd PMOS_180 W={w_cas*1.5:.2f}u L={s['l']:.2f}u
MPC2 outp vbp_cas s2 vdd PMOS_180 W={w_cas*1.5:.2f}u L={s['l']:.2f}u
MPL1 s1 outn vdd vdd PMOS_180 W={w_load*1.5:.2f}u L={s['l']:.2f}u
MPL2 s2 outn vdd vdd PMOS_180 W={w_load*1.5:.2f}u L={s['l']:.2f}u

* Load
CL outp 0 {s['cload_pf']:.3f}p

.op
"""
        if mode in ("ac", "both"):
            netlist += f".ac {ac_sweep}\n"
        if mode in ("tran", "both"):
            netlist += f".tran {tran_params}\n"
        netlist += ".end\n"
        return netlist
