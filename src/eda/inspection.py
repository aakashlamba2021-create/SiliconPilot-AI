"""Read-only inspection for EDA reports, SPICE netlists, and layout metadata.

The parsers intentionally extract a bounded subset of common formats. They never
execute uploaded netlists, run verification tools, or alter uploaded artifacts.
"""
from __future__ import annotations

import csv
import hashlib
from html.parser import HTMLParser
import io
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_ARTIFACT_BYTES = 25 * 1024 * 1024
MAX_STRUCTURED_REPORT_BYTES = 10 * 1024 * 1024
MAX_REPORT_TEXT_CHARS = 2_000_000
MAX_NETLIST_LINES = 100_000
MAX_NETLIST_INSTANCES = 20_000
MAX_NATIVE_LAYOUT_TEXT_BYTES = 10 * 1024 * 1024
MAX_GDSTK_CELLS = 2_000
MAX_GDSTK_OBJECTS = 200_000
REPORT_SUFFIXES = {".txt", ".log", ".out", ".rpt", ".drc", ".lvs", ".spef", ".csv", ".json", ".html", ".htm"}
NETLIST_SUFFIXES = {".cir", ".sp", ".spice", ".net", ".txt"}
LAYOUT_SUFFIXES = {".gds", ".gdsii", ".oas", ".oasis", ".def", ".lef"}


class ArtifactInspectionError(ValueError):
    """An artifact is unsupported, malformed, or above the safe input limit."""


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "head"}:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "head"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth and data.strip():
            self.parts.append(data.strip())


def _filename(value: str) -> str:
    result = Path(str(value)).name
    if not result or result in {".", ".."}:
        raise ArtifactInspectionError("The artifact filename is missing or invalid.")
    return result[:255]


def _decode(data: bytes) -> str:
    if not isinstance(data, bytes) or not data:
        raise ArtifactInspectionError("The uploaded artifact is empty or unreadable.")
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ArtifactInspectionError(f"Artifact exceeds the {MAX_ARTIFACT_BYTES // (1024 * 1024)} MB inspection limit.")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _base_provenance(filename: str, data: bytes, format_name: str, kind: str) -> dict[str, Any]:
    return {
        "source_filename": _filename(filename),
        "source_format": format_name,
        "artifact_kind": kind,
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "Imported by the user for read-only inspection; SiliconPilot did not run the source EDA tool.",
    }


def _flatten_json(
    value: Any, prefix: str = "", *, depth: int = 0, budget: list[int] | None = None
) -> list[str]:
    if budget is None:
        budget = [0]
    if depth >= 12:
        return [f"{prefix}: [maximum nesting depth reached]"]
    lines: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if budget[0] >= 20_000:
                lines.append("[JSON preview truncated at 20,000 values]")
                break
            budget[0] += 1
            label = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten_json(child, label, depth=depth + 1, budget=budget))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if budget[0] >= 20_000:
                lines.append("[JSON preview truncated at 20,000 values]")
                break
            budget[0] += 1
            lines.extend(_flatten_json(child, f"{prefix}[{index}]", depth=depth + 1, budget=budget))
    else:
        lines.append(f"{prefix}: {value}")
    return lines


def _report_text(filename: str, data: bytes, suffix: str) -> tuple[str, Any, bool]:
    text = _decode(data)
    parsed: Any = None
    preview_truncated = False
    if suffix == ".json":
        if len(data) > MAX_STRUCTURED_REPORT_BYTES:
            raise ArtifactInspectionError("JSON inspection is limited to 10 MB to bound parser work.")
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ArtifactInspectionError("The JSON report is malformed and could not be inspected.") from exc
        text = "\n".join(_flatten_json(parsed))
        preview_truncated = "[maximum nesting depth reached]" in text or "[JSON preview truncated" in text
    elif suffix == ".csv":
        if len(data) > MAX_STRUCTURED_REPORT_BYTES:
            raise ArtifactInspectionError("CSV inspection is limited to 10 MB to bound parser work.")
        parsed = []
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            parsed.append({key: value for index, (key, value) in enumerate(row.items()) if index < 200})
            if len(parsed) >= 2000:
                preview_truncated = True
                break
        if not parsed:
            raise ArtifactInspectionError("The CSV report has no header or data rows.")
        text = "\n".join(
            f"{key}: {value}" for row in parsed[:2000] for key, value in row.items()
            if value is not None and str(value).strip()
        )
    elif suffix in {".html", ".htm"}:
        parser = _VisibleText()
        parser.feed(text)
        text = "\n".join(parser.parts)
        if not text:
            raise ArtifactInspectionError("The HTML report contains no readable text content.")
    elif not text.strip():
        raise ArtifactInspectionError("The text report is empty.")
    if len(text) > MAX_REPORT_TEXT_CHARS:
        preview_truncated = True
        text = text[:1_500_000] + "\n[Middle of report omitted from bounded preview]\n" + text[-500_000:]
    return text, parsed, preview_truncated


def _first_count(text: str, patterns: tuple[str, ...]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return int(match.group(1))
    return None


def _json_count(document: Any, names: set[str]) -> int | None:
    stack = [(document, 0)]
    visited = 0
    while stack and visited < 20_000:
        value, depth = stack.pop()
        visited += 1
        if depth >= 12:
            continue
        if isinstance(value, dict):
            for index, (key, child) in enumerate(value.items()):
                if index >= 2000:
                    break
                normalized = re.sub(r"[^a-z]", "", str(key).lower())
                if normalized in names:
                    if isinstance(child, bool):
                        continue
                    if isinstance(child, int) and child >= 0:
                        return child
                    if isinstance(child, list):
                        return len(child)
                stack.append((child, depth + 1))
        elif isinstance(value, list):
            stack.extend((child, depth + 1) for child in value[:2000])
    return None


def inspect_eda_report(filename: str, data: bytes) -> dict[str, Any]:
    source = _filename(filename)
    suffix = Path(source).suffix.lower()
    if suffix not in REPORT_SUFFIXES:
        raise ArtifactInspectionError(f"Unsupported report extension '{suffix or '(none)'}'. Use text, CSV, JSON, or HTML.")
    text, parsed, preview_truncated = _report_text(source, data, suffix)
    lower = (source + "\n" + text[:200_000]).lower()
    if "drc" in lower or "design rule" in lower:
        report_type = "DRC"
        explanation = "DRC reports list rule checks from the source tool; extracted text does not validate that tool's rules or deck configuration."
    elif "lvs" in lower or "layout versus schematic" in lower:
        report_type = "LVS"
        explanation = "LVS reports describe source-tool comparison results; this import does not rerun connectivity extraction or comparison."
    elif "pex" in lower or "parasitic extraction" in lower:
        report_type = "PEX"
        explanation = "PEX measurements are imported values from the source report, not extraction performed by SiliconPilot."
    elif "spice" in lower or "ngspice" in lower or "simulation" in lower:
        report_type = "SPICE simulation report"
        explanation = "Simulation values are imported from the source report; the report alone does not prove a fresh simulation ran here."
    else:
        report_type = "Unknown EDA report"
        explanation = "The file was readable, but its report type could not be identified reliably."

    tool = re.search(r"(?:tool|software|generator|product)\s*[:=]\s*([^\r\n;,]+)", text, re.IGNORECASE)
    timestamp = re.search(
        r"(?:timestamp|date|generated\s+on|run\s+date)\s*[:=]\s*([^\r\n;,]+)",
        text, re.IGNORECASE,
    )
    error_patterns = (r"(?:total\s+)?(?:drc\s+)?errors?\s*[:=]\s*(\d+)", r"(\d+)\s+(?:drc\s+)?errors?\b")
    warning_patterns = (r"(?:total\s+)?warnings?\s*[:=]\s*(\d+)", r"(\d+)\s+warnings?\b")
    errors = _json_count(parsed, {"errorcount", "totalerrors", "drcerrors"}) if parsed is not None else None
    warnings = _json_count(parsed, {"warningcount", "totalwarnings"}) if parsed is not None else None
    if errors is None:
        errors = _first_count(text, error_patterns)
    if warnings is None:
        warnings = _first_count(text, warning_patterns)

    rules = sorted(set(re.findall(
        r"(?:rule(?:\s+(?:id|name))?|check(?:\s+(?:id|name))?)\s*[:=]\s*([A-Za-z0-9_.:/-]+)",
        text, flags=re.IGNORECASE,
    )))[:100]
    instances = sorted(set(re.findall(
        r"\b(?:instance|device)\s*[:=]\s*([A-Za-z0-9_./\[\]-]+)", text, flags=re.IGNORECASE,
    )))[:100]
    nets = sorted(set(re.findall(r"\bnet\s*[:=]\s*([A-Za-z0-9_./\[\]-]+)", text, flags=re.IGNORECASE)))[:100]
    coordinates = sorted(set(re.findall(
        r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)|\b(?:x|y)\s*[:=]\s*-?\d+(?:\.\d+)?",
        text, flags=re.IGNORECASE,
    )))[:100]
    metric_pattern = re.compile(
        r"\b(gain|gbw|phase\s*margin|slew\s*rate|power|area|capacitance|resistance|delay|frequency)\b"
        r"\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*([A-Za-zÂµÎ©/]+)?",
        flags=re.IGNORECASE,
    )
    measurements = [
        {"name": match.group(1), "value": float(match.group(2)), "unit": match.group(3)}
        for match in list(metric_pattern.finditer(text))[:100]
    ]
    issue_lines = [
        line.strip()[:500] for line in text.splitlines()
        if re.search(r"error|violation|warning|unmatched|short|open|failed", line, re.IGNORECASE)
    ][:50]
    summary_lines = [line.strip()[:500] for line in text.splitlines() if re.search(r"summary|overall result|status\s*[:=]", line, re.IGNORECASE)][:20]

    return {
        **_base_provenance(source, data, suffix.lstrip("."), "EDA report"),
        "status": "inspected",
        "report_type": report_type,
        "tool_name": tool.group(1).strip() if tool else None,
        "report_timestamp": timestamp.group(1).strip() if timestamp else None,
        "error_count": errors,
        "warning_count": warnings,
        "rule_names": rules,
        "affected_instances": instances,
        "coordinates": coordinates,
        "nets": nets,
        "measurements": measurements,
        "summary_lines": summary_lines,
        "issue_samples": issue_lines,
        "preview_truncated": preview_truncated,
        "interpretation": explanation,
        "source_tool_result": "Imported evidence only; not executed or independently validated by SiliconPilot.",
    }


def inspect_spice_netlist(filename: str, data: bytes) -> dict[str, Any]:
    source = _filename(filename)
    suffix = Path(source).suffix.lower()
    if suffix not in NETLIST_SUFFIXES:
        raise ArtifactInspectionError("Unsupported SPICE netlist extension. Use .cir, .sp, .spice, .net, or .txt.")
    text = _decode(data)
    if "\x00" in text:
        raise ArtifactInspectionError("The file is binary and cannot be inspected as a SPICE text netlist.")
    logical_lines: list[str] = []
    for line_number, raw in enumerate(io.StringIO(text), start=1):
        if line_number > MAX_NETLIST_LINES:
            raise ArtifactInspectionError(f"SPICE inspection is limited to {MAX_NETLIST_LINES:,} lines.")
        if len(raw) > 100_000:
            raise ArtifactInspectionError("A SPICE line exceeds the 100,000-character inspection limit.")
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("+") and logical_lines:
            logical_lines[-1] += " " + line[1:].strip()
        else:
            logical_lines.append(line)

    subcircuits: list[dict[str, Any]] = []
    models: set[str] = set()
    nodes: set[str] = set()
    component_counts: dict[str, int] = {}
    instances: list[dict[str, Any]] = []
    non_executable_directives: list[str] = []
    for line in logical_lines:
        if line.startswith("$") or line.startswith(";"):
            continue
        tokens = line.split()
        if not tokens:
            continue
        head = tokens[0].lower()
        if head == ".subckt" and len(tokens) >= 2:
            if len(subcircuits) >= 1000:
                raise ArtifactInspectionError("SPICE inspection is limited to 1,000 subcircuit definitions.")
            subcircuits.append({"name": tokens[1], "ports": tokens[2:]})
            continue
        if head in {".include", ".lib", ".control", ".shell", ".exec"}:
            non_executable_directives.append(line[:300])
            continue
        if head.startswith("."):
            continue
        kind = tokens[0][0].upper()
        if kind not in {"R", "C", "L", "V", "I", "D", "Q", "M", "J", "X"}:
            continue
        if kind in {"R", "C", "L", "V", "I"} and len(tokens) >= 4:
            connected_nodes = tokens[1:3]
        elif kind == "D" and len(tokens) >= 4:
            connected_nodes = tokens[1:3]
            models.add(tokens[3])
        elif kind in {"Q", "J"} and len(tokens) >= 5:
            connected_nodes = tokens[1:4]
            models.add(tokens[4])
        elif kind == "M" and len(tokens) >= 6:
            connected_nodes = tokens[1:5]
            models.add(tokens[5])
        elif kind == "X" and len(tokens) >= 3:
            non_parameter_tokens = [token for token in tokens[1:] if "=" not in token]
            if len(non_parameter_tokens) < 2:
                continue
            connected_nodes = non_parameter_tokens[:-1]
            models.add(non_parameter_tokens[-1])
        else:
            continue
        if len(instances) >= MAX_NETLIST_INSTANCES:
            raise ArtifactInspectionError(f"SPICE inspection is limited to {MAX_NETLIST_INSTANCES:,} device instances.")
        nodes.update(connected_nodes)
        if len(nodes) > 50_000:
            raise ArtifactInspectionError("SPICE inspection is limited to 50,000 unique node names.")
        component_counts[kind] = component_counts.get(kind, 0) + 1
        instances.append({"name": tokens[0], "device_type": kind, "nodes": connected_nodes[:32]})

    if not instances and not subcircuits:
        raise ArtifactInspectionError("No supported SPICE elements or .SUBCKT definitions were recognized.")
    return {
        **_base_provenance(source, data, suffix.lstrip("."), "SPICE netlist"),
        "status": "inspected",
        "dialect_support": "Common line-oriented SPICE subset; dialect-specific syntax may be omitted.",
        "subcircuits": subcircuits[:100],
        "component_counts_by_prefix": component_counts,
        "component_count": len(instances),
        "instances": instances[:500],
        "unique_node_names": sorted(nodes)[:500],
        "model_references": sorted(models)[:200],
        "directives_not_executed": sorted(set(non_executable_directives))[:100],
        "executed": False,
        "explanation": "This is a static text inspection. It does not resolve included model files, expand subcircuits, or run SPICE.",
    }


def _inspect_gdstk(filename: str, data: bytes, suffix: str) -> dict[str, Any]:
    try:
        import gdstk
    except ImportError as exc:
        raise ArtifactInspectionError("GDSII/OASIS inspection requires gdstk in the project virtual environment.") from exc
    temporary_path: Path | None = None
    cleanup_error = False
    try:
        private_temp = Path(
            os.environ.get("SILICONPILOT_PRIVATE_TEMP") or tempfile.gettempdir()
        ).expanduser()
        private_temp.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=suffix, prefix="siliconpilot-layout-", dir=private_temp, delete=False
        ) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)
        library = gdstk.read_gds(str(temporary_path)) if suffix in {".gds", ".gdsii"} else gdstk.read_oas(str(temporary_path))
    except Exception as exc:
        raise ArtifactInspectionError("The native layout file could not be parsed by gdstk.") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                cleanup_error = True
    if cleanup_error:
        raise ArtifactInspectionError(
            "The temporary native-layout copy could not be removed from the local temp directory."
        )

    if len(library.cells) > MAX_GDSTK_CELLS:
        raise ArtifactInspectionError(
            f"Native layout metadata extraction is limited to {MAX_GDSTK_CELLS:,} cells."
        )

    layer_pairs: set[tuple[int, int]] = set()
    cell_rows = []
    total_counts = {"polygons": 0, "paths": 0, "labels": 0, "references": 0}
    unit_to_um = float(library.unit) * 1e6
    processed_objects = 0
    metadata_truncated = len(library.cells) > 1000
    for cell in library.cells[:1000]:
        bbox = cell.bounding_box()
        bbox_um = None
        if bbox is not None:
            bbox_um = [[round(float(value) * unit_to_um, 6) for value in point] for point in bbox]
        for polygon in cell.polygons:
            if processed_objects >= MAX_GDSTK_OBJECTS:
                metadata_truncated = True
                break
            layer_pairs.add((int(polygon.layer), int(polygon.datatype)))
            processed_objects += 1
        for path_obj in cell.paths:
            if processed_objects >= MAX_GDSTK_OBJECTS:
                metadata_truncated = True
                break
            layer_pairs.add((int(path_obj.layers[0]), int(path_obj.datatypes[0])))
            processed_objects += 1
        counts = {
            "polygons": len(cell.polygons), "paths": len(cell.paths),
            "labels": len(cell.labels), "references": len(cell.references),
        }
        for key, value in counts.items():
            total_counts[key] += value
        if sum(total_counts.values()) > MAX_GDSTK_OBJECTS:
            metadata_truncated = True
        cell_rows.append({"name": cell.name, "bounding_box_um": bbox_um, "object_counts": counts})
    return {
        **_base_provenance(filename, data, suffix.lstrip("."), "native layout metadata"),
        "status": "inspected",
        "parser": f"gdstk {gdstk.__version__}",
        "file_type": "GDSII" if suffix in {".gds", ".gdsii"} else "OASIS",
        "library_unit_meters": float(library.unit),
        "library_precision_meters": float(library.precision),
        "cell_count": len(library.cells),
        "cells": cell_rows,
        "top_level_cells": [cell.name for cell in library.top_level()[:100]],
        "layer_datatype_pairs": [{"layer": layer, "datatype": datatype} for layer, datatype in sorted(layer_pairs)],
        "total_object_counts": total_counts,
        "metadata_truncated": metadata_truncated,
        "interpretation": "Metadata only. No polygon DRC, LVS, connectivity, PEX, or foundry signoff was performed.",
    }


def inspect_native_layout(filename: str, data: bytes) -> dict[str, Any]:
    source = _filename(filename)
    suffix = Path(source).suffix.lower()
    if suffix not in LAYOUT_SUFFIXES:
        raise ArtifactInspectionError("Unsupported native layout extension. Use GDSII, OASIS, DEF, or LEF.")
    _decode(data)
    if suffix in {".gds", ".gdsii", ".oas", ".oasis"}:
        return _inspect_gdstk(source, data, suffix)

    text = _decode(data)
    if "\x00" in text:
        raise ArtifactInspectionError("The DEF/LEF file contains binary data and could not be inspected as text.")
    if len(data) > MAX_NATIVE_LAYOUT_TEXT_BYTES:
        raise ArtifactInspectionError("DEF/LEF text inspection is limited to 10 MB to bound parser work.")
    if suffix == ".def":
        design = re.search(r"\bDESIGN\s+([^\s;]+)", text, re.IGNORECASE)
        units = re.search(r"\bUNITS\s+DISTANCE\s+MICRONS\s+(\d+)", text, re.IGNORECASE)
        die = re.search(r"\bDIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)", text, re.IGNORECASE)
        components = re.search(r"\bCOMPONENTS\s+(\d+)\s*;(.*?)(?:END\s+COMPONENTS)", text, re.IGNORECASE | re.DOTALL)
        nets = re.search(r"\bNETS\s+(\d+)\s*;(.*?)(?:END\s+NETS)", text, re.IGNORECASE | re.DOTALL)
        component_rows = re.findall(r"^\s*-\s+([^\s]+)\s+([^\s]+)", components.group(2), re.MULTILINE) if components else []
        net_names = re.findall(r"^\s*-\s+([^\s]+)", nets.group(2), re.MULTILINE) if nets else []
        layer_names = sorted(set(re.findall(r"\+\s+ROUTED\s+([^\s(]+)", text, re.IGNORECASE)))
        if not any((design, units, die, components, nets)):
            raise ArtifactInspectionError("No supported DEF design, units, die area, components, or nets section was recognized.")
        return {
            **_base_provenance(source, data, "def", "native layout metadata"),
            "status": "inspected", "file_type": "DEF", "design_name": design.group(1) if design else None,
            "database_units_per_micron": int(units.group(1)) if units else None,
            "die_area_database_units": [int(value) for value in die.groups()] if die else None,
            "component_count_declared": int(components.group(1)) if components else None,
            "components_preview": [{"instance": name, "cell": cell} for name, cell in component_rows[:500]],
            "net_count_declared": int(nets.group(1)) if nets else None,
            "net_names_preview": net_names[:500], "routing_layers_observed": layer_names[:200],
            "interpretation": "Read-only DEF text subset; vendor extensions and full placement/routing semantics are not validated.",
        }

    macros = re.findall(r"\bMACRO\s+([^\s]+)(.*?)(?:END\s+\1)", text, re.IGNORECASE | re.DOTALL)
    layers = sorted(set(re.findall(r"\bLAYER\s+([^\s;]+)", text, re.IGNORECASE)))
    macro_rows = []
    for name, body in macros[:500]:
        size = re.search(r"\bSIZE\s+([\d.]+)\s+BY\s+([\d.]+)\s*;", body, re.IGNORECASE)
        macro_rows.append({
            "name": name,
            "size_microns": [float(size.group(1)), float(size.group(2))] if size else None,
        })
    if not macro_rows and not layers:
        raise ArtifactInspectionError("No supported LEF MACRO or LAYER section was recognized.")
    return {
        **_base_provenance(source, data, "lef", "native layout metadata"),
        "status": "inspected", "file_type": "LEF", "macro_count": len(macros),
        "macros": macro_rows, "layer_names": layers[:500],
        "interpretation": "Read-only LEF macro and layer metadata subset; the file was not used as a foundry rule deck.",
    }


def inspect_artifact(filename: str, data: bytes, artifact_kind: str) -> dict[str, Any]:
    """Dispatch one explicitly selected artifact type without side effects."""
    if artifact_kind == "EDA report":
        return inspect_eda_report(filename, data)
    if artifact_kind == "SPICE netlist":
        return inspect_spice_netlist(filename, data)
    if artifact_kind == "Native layout metadata":
        return inspect_native_layout(filename, data)
    raise ArtifactInspectionError("Choose an EDA report, SPICE netlist, or native layout metadata inspection mode.")
