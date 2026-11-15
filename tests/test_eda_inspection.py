from __future__ import annotations

import pytest

from src.eda.inspection import (
    ArtifactInspectionError,
    inspect_eda_report,
    inspect_native_layout,
    inspect_spice_netlist,
)


def test_text_and_json_reports_retain_import_provenance_and_missing_fields():
    text = b"Tool: Magic\nGenerated on: 2026-10-10\nDRC Errors: 2\nWarnings: 1\nRule: M1.SP.1\nInstance: M7\nNet: OUT\nCoordinate: (10, -20)\n"
    report = inspect_eda_report("review.log", text)
    assert report["report_type"] == "DRC"
    assert report["source_filename"] == "review.log"
    assert report["source_sha256"]
    assert report["error_count"] == 2
    assert report["warning_count"] == 1
    assert report["rule_names"] == ["M1.SP.1"]
    assert report["affected_instances"] == ["M7"]
    assert report["nets"] == ["OUT"]
    assert report["coordinates"] == ["(10, -20)"]
    assert report["report_timestamp"] == "2026-10-10"
    assert report["tool_name"] == "Magic"
    assert report["source_tool_result"].startswith("Imported evidence")
    assert report["measurements"] == []

    structured = inspect_eda_report("lvs.json", b'{"tool":"Netgen","lvs":{"error_count":0,"warnings":["one"]}}')
    assert structured["report_type"] == "LVS"
    assert structured["error_count"] == 0
    assert structured["tool_name"] == "Netgen"


@pytest.mark.parametrize(
    "filename,payload",
    [
        ("report.csv", b"Rule,Error Count\nM1.SP.1,2\n"),
        ("report.html", b"<html><head><title>Hidden</title></head><body><h1>PEX Summary</h1><p>Gain: 20 dB</p><script>fake error</script></body></html>"),
    ],
)
def test_csv_and_html_reports_are_read_without_claiming_execution(filename, payload):
    report = inspect_eda_report(filename, payload)
    assert report["status"] == "inspected"
    assert report["provenance"].startswith("Imported")


def test_malformed_and_unsupported_reports_fail_with_actionable_errors():
    with pytest.raises(ArtifactInspectionError, match="malformed"):
        inspect_eda_report("bad.json", b"{")
    with pytest.raises(ArtifactInspectionError, match="Unsupported report extension"):
        inspect_eda_report("secret.bin", b"data")
    with pytest.raises(ArtifactInspectionError, match="limited to 10 MB"):
        inspect_eda_report("large.csv", b"a,b\n" + b"x,y\n" * (5 * 1024 * 1024))


def test_spice_netlist_extracts_a_safe_subset_and_never_executes():
    netlist = b"""* differential pair example
.subckt OTA inp inn out vdd vss
M1 n1 inp tail vss nmos W=1u L=0.18u
M2 n2 inn tail vss nmos W=1u L=0.18u
R1 out n1 10k
C1 out vss 2p
XLOAD out vss load_cell
.include \"models.lib\"
.control
tran 1n 10n
.endc
.ends OTA
.end
"""
    result = inspect_spice_netlist("ota.cir", netlist)
    assert result["executed"] is False
    assert result["component_count"] == 5
    assert result["component_counts_by_prefix"] == {"M": 2, "R": 1, "C": 1, "X": 1}
    assert result["subcircuits"][0]["name"] == "OTA"
    assert "nmos" in result["model_references"]
    assert "load_cell" in result["model_references"]
    assert any(".include" in line for line in result["directives_not_executed"])


def test_spice_netlist_malformed_and_unsupported_formats_are_reported():
    with pytest.raises(ArtifactInspectionError, match="No supported SPICE"):
        inspect_spice_netlist("empty.cir", b"title only\n.end\n")
    with pytest.raises(ArtifactInspectionError, match="Unsupported SPICE"):
        inspect_spice_netlist("layout.gds", b"binary")
    with pytest.raises(ArtifactInspectionError, match="limited to 100,000 lines"):
        inspect_spice_netlist("too_many.cir", b"\n".join([b"* comment"] * 100_001))


def test_def_and_lef_extract_read_only_metadata():
    def_data = b"""DESIGN amp_top ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 10000 8000 ) ;
COMPONENTS 1 ;
 - U1 amp_cell + PLACED ( 10 20 ) N ;
END COMPONENTS
NETS 1 ;
 - OUT ( U1 Y ) + ROUTED M2 ( 0 0 ) ( 10 10 ) ;
END NETS
END DESIGN
"""
    result = inspect_native_layout("amp.def", def_data)
    assert result["file_type"] == "DEF"
    assert result["design_name"] == "amp_top"
    assert result["component_count_declared"] == 1
    assert result["net_count_declared"] == 1
    assert result["routing_layers_observed"] == ["M2"]
    assert "not validated" in result["interpretation"]

    lef = b"MACRO amp_cell\n  SIZE 12.5 BY 8.0 ;\n  LAYER M1 ;\nEND amp_cell\n"
    lef_result = inspect_native_layout("cells.lef", lef)
    assert lef_result["macros"] == [{"name": "amp_cell", "size_microns": [12.5, 8.0]}]
    assert "M1" in lef_result["layer_names"]
    with pytest.raises(ArtifactInspectionError, match="No supported DEF"):
        inspect_native_layout("invalid.def", b"this is not DEF")
    with pytest.raises(ArtifactInspectionError, match="No supported LEF"):
        inspect_native_layout("invalid.lef", b"this is not LEF")


@pytest.mark.parametrize("suffix", ["gds", "oas"])
def test_gdsii_and_oasis_metadata_are_read_without_modifying_source(suffix, tmp_path, monkeypatch):
    gdstk = pytest.importorskip("gdstk")
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = library.new_cell("TOP")
    cell.add(gdstk.rectangle((0, 0), (10, 5), layer=3, datatype=0))
    file_path = tmp_path / f"layout.{suffix}"
    if suffix == "gds":
        library.write_gds(str(file_path))
    else:
        library.write_oas(str(file_path))
    data = file_path.read_bytes()
    private_temp = tmp_path / "private_temp"
    monkeypatch.setenv("SILICONPILOT_PRIVATE_TEMP", str(private_temp))
    result = inspect_native_layout(file_path.name, data)
    assert result["cell_count"] == 1
    assert result["top_level_cells"] == ["TOP"]
    assert result["layer_datatype_pairs"] == [{"layer": 3, "datatype": 0}]
    assert result["cells"][0]["bounding_box_um"] == [[0.0, 0.0], [10.0, 5.0]]
    assert list(private_temp.iterdir()) == []

