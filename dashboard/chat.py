"""
AI Engineering Assistant Engine — SiliconPilot AI
Provides domain-specific technical explanations for Analog IC Layout,
MOSFET design, DRC/LVS, parasitics, and inspects live trained models and optimization results.
Modular architecture allowing seamless drop-in of external LLM APIs (Gemini, Claude, OpenAI).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.models.model_registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Analog IC Engineering Knowledge Base
# ---------------------------------------------------------------------------

ANALOG_KNOWLEDGE_BASE: Dict[str, Dict[str, str]] = {
    "differential_pair": {
        "title": "Differential Pair Layout & Matching",
        "keywords": ["differential", "pair", "diff pair", "mismatch", "offset", "common centroid"],
        "content": (
            "**Differential Pair Layout Guidelines:**\n"
            "- **Common-Centroid Placement:** Transistors M1 and M2 must be interdigitated (e.g., ABBA or ABBA-BAAB) "
            "to cancel linear process gradients (oxide thickness tox, threshold voltage Vth, doping concentration).\n"
            "- **Dummy Devices:** Place dummy poly strips at the array boundaries to prevent optical proximity effects (OPE) "
            "and etch rate variation from altering the outer transistor edge dimensions.\n"
            "- **Routing Symmetry:** Keep gate, drain, and source interconnects geometrically identical. "
            "Unequal parasitic capacitance (Cp) to ground on either branch converts common-mode noise into differential offset.\n"
            "- **Well Enclosure:** Enclose both devices inside a single deep uniform well to avoid well-proximity effect (WPE) delta."
        ),
    },
    "current_mirror": {
        "title": "Current Mirror Layout & Systematic Error Mitigation",
        "keywords": ["current mirror", "mirror", "cascode", "channel length modulation", "wpe"],
        "content": (
            "**Current Mirror Layout Guidelines:**\n"
            "- **Unit Transistor Sizing:** Always construct mirror ratios using identical unit transistors (e.g. 1:4 ratio = 1 unit vs 4 parallel units). "
            "Never implement ratios by simply drawing a 4x wider transistor due to differing fringe capacitance and edge conduction.\n"
            "- **Orientation Alignment:** All transistors in the current mirror must share the exact same orientation (0° or 180°). "
            "Silicon substrate piezoresistive stress varies significantly between orthogonal crystal axes.\n"
            "- **VDS Matching (Cascode):** High output resistance mirrors (e.g., Wilson or Wide-Swing Cascode) maintain identical Vds "
            "across the reference and output branch, eliminating channel-length modulation errors (early voltage)."
        ),
    },
    "drc_lvs": {
        "title": "Physical Verification (DRC & LVS)",
        "keywords": ["drc", "lvs", "design rule", "verification", "short", "open", "antenna"],
        "content": (
            "**Physical Verification Fundamentals:**\n"
            "- **DRC (Design Rule Checking):** Validates geometric constraints imposed by the foundry process design kit (PDK), "
            "including minimum metal width, minimum spacing, metal density rules, enclosure rules, and antenna diode ratios.\n"
            "- **LVS (Layout Versus Schematic):** Extracts the electrical connectivity and device geometry directly from polygon layers "
            "and performs graph isomorphism comparison against the SPICE schematic to guarantee topological and sizing equivalence.\n"
            "- **Antenna Rule:** Long metal tracks connected solely to transistor gates during manufacturing collect plasma charge, "
            "leading to gate oxide breakdown. Solved by routing to higher metal layers via jumpers or tying reverse-biased antenna diodes."
        ),
    },
    "parasitics": {
        "title": "Parasitic Resistance and Capacitance (PEX / RC Extraction)",
        "keywords": ["parasitic", "pex", "capacitance", "resistance", "rc", "interconnect", "coupling"],
        "content": (
            "**Parasitic Interconnect & Device Effects:**\n"
            "- **Parasitic Capacitance (C_para):** Comprises plate capacitance (area to substrate/lower layers), fringe capacitance (sidewall to adjacent layers), "
            "and lateral coupling capacitance between adjacent wires. Lateral coupling between switching signals and high-impedance analog nodes causes cross-talk.\n"
            "- **Parasitic Resistance (R_para):** Thin lower-metal layers (e.g., M1, M2) exhibit non-negligible sheet resistance (0.1–1.0 Ω/sq). "
            "High DC bias currents across thin metal produce IR drops that starve headroom; high AC signals suffer RC bandwidth degradation.\n"
            "- **Mitigation:** Use top thick metal layers (e.g., M5, M6, TopMetal) for power rails and critical high-speed signal paths."
        ),
    },
    "mosfet_layout": {
        "title": "Transistor Fingering, Multiplicity and Sizing",
        "keywords": ["mosfet", "finger", "fingers", "multiplicity", "gate resistance", "rg", "width"],
        "content": (
            "**MOSFET Fingering & Multiplicity:**\n"
            "- **Fingering:** Splitting a wide transistor into N parallel fingers reduces gate resistance Rg by a factor of N² (or N²/3 for double-ended contact), "
            "significantly improving maximum oscillation frequency (fmax) and noise figure.\n"
            "- **Drain/Source Sharing:** Alternating drain and source diffusions (S-G-D-G-S) halves the total junction capacitance (Cdb and Csb), "
            "substantially boosting circuit bandwidth.\n"
            "- **Multiplicity (M):** Used when devices are separated physically or placed across multiple common-centroid cells."
        ),
    },
    "opamp_architectures": {
        "title": "Operational Amplifier Topologies & Trade-offs",
        "keywords": ["opamp", "ota", "miller", "folded cascode", "telescopic", "gain", "gbw", "phase margin"],
        "content": (
            "**Operational Amplifier Topologies Comparison:**\n"
            "- **Miller Two-Stage:** High DC gain (Av ~ 60–80 dB) and wide output voltage swing. Requires Miller compensation capacitor (Cc) "
            "for pole-splitting stability. Sensitive to capacitive load loading.\n"
            "- **Folded Cascode:** High input common-mode range, superior frequency response (single dominant pole at output), "
            "moderate gain (Av ~ 50–70 dB). Lower output swing than two-stage.\n"
            "- **Telescopic Cascode:** Highest speed and power efficiency, high gain, but restricted voltage swing due to stacked transistors."
        ),
    },
}


def _format_model_registry(registry: ModelRegistry) -> str | None:
    models = registry.list_models()
    if not models:
        return None
    lines = ["### SiliconPilot AI — Trained Model Registry Status and provenance"]
    for name, info in models.items():
        status = registry.get_model_status(name)
        trust = status.get("trust_label", "unapproved")
        lines.extend([
            f"- **Model:** `{name}`",
            f"  - **Trust status:** `{trust}` · **Data origin:** `{status.get('data_origin', 'unknown')}`",
            f"  - **Training data:** {status.get('training_data', 'Unknown')}",
        ])
        if trust == "quarantined":
            lines.append("  - **Engineering use:** quarantined; excluded from recommendations and ranking.")
        elif trust == "restricted":
            lines.append("  - **Engineering use:** restricted; not a trusted ranking signal.")
        elif trust == "exploratory":
            lines.append("  - **Engineering use:** exploratory estimate only; not validation evidence.")
        elif trust != "approved":
            lines.append("  - **Engineering use:** unapproved; do not use as an engineering decision signal.")
        limitations = status.get("evaluation_limitations", [])
        if limitations:
            lines.append("  - **Evaluation limitations:** " + " ".join(limitations))
        metrics = info.get("metrics", {})
        if trust in {"approved", "exploratory"} and "overall" in metrics:
            overall = metrics["overall"]
            value = overall.get("mean_R2", overall.get("R2"))
            if isinstance(value, (int, float)):
                lines.append(f"  - **Saved evaluation R²:** `{value:.4f}` (historical report; not independently reproduced here)")
        elif trust in {"approved", "exploratory"} and isinstance(metrics.get("Accuracy"), (int, float)):
            lines.append(f"  - **Saved evaluation accuracy:** `{metrics['Accuracy'] * 100:.2f}%` (historical report; not independent verification)")
    return "\n".join(lines)


class SiliconPilotChatbot:
    """
    Local domain-knowledge AI engineering assistant with model introspection capabilities.
    """

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def respond(self, query: str, history: list[dict[str, Any]] | None = None) -> str:
        """
        Processes an engineer's question and returns a structured technical response.
        """
        q = query.strip().lower()
        if not q:
            return "Type a question about analog layout, circuits, model provenance, or simulation."

        normalized = re.sub(r"[^a-z0-9' ]+", " ", q).strip()
        normalized = re.sub(r"\s+", " ", normalized)
        if re.fullmatch(r"(hi|hello|hey|hiya|good morning|good afternoon|good evening)( there)?", normalized):
            return "Hey! Welcome to SiliconPilot AI. What are we working on today?"
        if re.fullmatch(r"(how are you|how's it going|hows it going|how are things)( today)?", normalized):
            return "Doing well, thanks! Ready to help with your analog layout work. What are you looking into?"
        if normalized in {"thanks", "thank you", "thx"}:
            return "You're welcome. Let me know what you want to look at next."
        if normalized in {"bye", "goodbye", "see you", "see you later"}:
            return "See you later. I’ll be here when you want to continue the design discussion."
        if any(phrase in normalized for phrase in ("what can you do", "what do you do", "your capabilities", "help me get started")):
            return (
                "I can explain analog layout concepts, inspect uploaded layout screenshots with a configured vision provider, "
                "compare labelled alternatives, review model provenance, inspect supported EDA reports and netlists, and run explicit ngspice analyses. "
                "Visual suggestions are hypotheses; DRC/LVS, connectivity, extracted parasitics, and silicon performance require their own evidence."
            )
        if normalized in {"who are you", "what is siliconpilot", "what is siliconpilot ai"}:
            return (
                "SiliconPilot AI is a local analog IC engineering workbench for dataset exploration, exploratory model analysis, "
                "layout screenshot discussion, read-only artifact inspection, and explicit ngspice simulation."
            )
        if re.search(r"\b(analy[sz]e|inspect|review)\b.*\b(layout|screenshot|image)\b", normalized):
            return "Upload a PNG or JPEG layout screenshot above, then ask a specific question. I can only make visual observations; screenshots do not establish DRC/LVS or electrical behavior."

        if any(w in q for w in ["model", "trained", "accuracy", "r2", "performance", "metric", "registry"]):
            registry_response = _format_model_registry(self.registry)
            if registry_response:
                return registry_response

        # 1. Check for Model Registry / Status inquiries
        if any(w in q for w in ["model", "trained", "accuracy", "r2", "performance", "metric", "registry"]):
            models = self.registry.list_models()
            if models:
                resp = ["### 🤖 SiliconPilot AI — Trained Model Registry Status:\n"]
                for name, info in models.items():
                    resp.append(f"- **Model:** `{name}`")
                    resp.append(f"  - **Type/Dataset:** {info.get('dataset_name', 'N/A')} ({info.get('description', '')})")
                    resp.append(f"  - **Saved Version:** `{info.get('version_tag', '')}`")
                    m = info.get("metrics", {})
                    if "overall" in m:
                        overall = m["overall"]
                        if "mean_R2" in overall or "R2" in overall:
                            r2_val = overall.get("mean_R2", overall.get("R2", 0))
                            resp.append(f"  - **Test R² Score:** `{r2_val:.4f}`")
                    elif "Accuracy" in m:
                        resp.append(f"  - **Test Accuracy:** `{m['Accuracy']*100:.2f}%` (F1: `{m.get('F1', 0):.4f}`)")
                return "\n".join(resp)

        recent_user_text = " ".join(
            str(item.get("content", ""))
            for item in (history or [])[-8:]
            if item.get("role") == "user" and str(item.get("content", "")).strip()
        )
        effective_query = f"{recent_user_text} {q}" if len(normalized.split()) <= 5 else q

        # 2. Match against Analog Knowledge Base
        best_match = None
        best_score = 0
        for topic_key, topic_data in ANALOG_KNOWLEDGE_BASE.items():
            score = sum(1 for kw in topic_data["keywords"] if kw in effective_query)
            if score > best_score:
                best_score = score
                best_match = topic_data

        if best_match and best_score > 0:
            return f"### 📘 {best_match['title']}\n\n{best_match['content']}\n\n*Source: SiliconPilot Engineering Knowledge Engine*"

        if history and any(item.get("image_ids") for item in history if item.get("role") == "user"):
            return (
                "I can see that this session includes an earlier image discussion, but no image was sent with this text-only turn. "
                "Ask about a specific point already covered above, or reselect the image if you want a vision provider to inspect its pixels again. "
                "I will not infer new visual details without a new image-analysis request."
            )

        return (
            "I can help with analog layout, matching, routing, model provenance, imported EDA evidence, and ngspice workflows. "
            "Tell me the circuit or design issue you are working on, and I?ll keep the answer focused."
        )
