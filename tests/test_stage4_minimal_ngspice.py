"""Live minimal RC smoke test for a process-configured ngspice DLL."""
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.ngspice_runner import NGSpiceRunner
from src.simulation.netlist_manager import NetlistManager
from src.simulation.simulation_parser import SimulationParser


class MinimalNgspiceSmokeTest(unittest.TestCase):
    def test_rc_circuit_ac_and_transient_vector_flow(self):
        dll_path = os.environ.get("NGSPICE_DLL")
        if not dll_path:
            self.skipTest("Set NGSPICE_DLL for the environment-specific live simulation test.")

        runner = NGSpiceRunner()
        self.assertTrue(runner.is_available(), runner.get_backend_info())
        self.assertIn("shared library", runner.get_backend_info().lower())

        netlist = """Minimal RC smoke test
V1 vin 0 DC 0 AC 1 PULSE(0 1 0 1n 1n 5n 10n)
R1 vin outp 1k
C1 outp 0 1n
.end
"""
        result = runner.run_netlist(netlist, run_op=False, run_ac=True, run_tran=True)
        self.assertTrue(result.success, f"{result.error_message}; statuses={result.analysis_statuses}; logs={result.log_messages}")
        self.assertEqual(result.ac_status, "SUCCESS")
        self.assertEqual(result.tran_status, "SUCCESS")

        parser = SimulationParser()
        ac_data = parser.parse_ac_file(result.ac_file)
        tran_data = parser.parse_tran_file(result.tran_file)
        try:
            self.assertTrue(ac_data.is_valid, ac_data.error_message)
            self.assertTrue(tran_data.is_valid, tran_data.error_message)
            self.assertGreater(len(ac_data.frequencies), 1)
            self.assertGreater(len(tran_data.time_s), 1)
        finally:
            for output_path in (result.ac_file, result.tran_file):
                if output_path and os.path.isfile(output_path):
                    os.remove(output_path)

    def test_all_project_topologies_produce_ac_and_transient_vectors(self):
        if not os.environ.get("NGSPICE_DLL"):
            self.skipTest("Set NGSPICE_DLL for topology simulations.")
        runner = NGSpiceRunner()
        manager = NetlistManager()
        parser = SimulationParser()
        params = {
            "transistor_width_W": 40.0,
            "transistor_length_L": 0.5,
            "finger_count": 8,
            "multiplicity": 2,
            "ibias_ua": 20.0,
            "cload_pf": 1.5,
            "vdd": 1.8,
        }
        for topology in manager.get_supported_topologies():
            with self.subTest(topology=topology):
                ac_result = runner.run_netlist(
                    manager.generate_netlist(topology, params, mode="ac"),
                    run_ac=True, run_tran=False, run_op=True,
                )
                self.assertTrue(ac_result.success, f"{topology} AC: {ac_result.error_message}; {ac_result.log_messages}")
                tran_result = runner.run_netlist(
                    manager.generate_netlist(topology, params, mode="tran"),
                    run_ac=False, run_tran=True, run_op=False,
                )
                self.assertTrue(tran_result.success, f"{topology} TRAN: {tran_result.error_message}; {tran_result.log_messages}")
                ac_data = parser.parse_ac_file(ac_result.ac_file)
                tran_data = parser.parse_tran_file(tran_result.tran_file)
                try:
                    self.assertTrue(ac_data.is_valid, ac_data.error_message)
                    self.assertTrue(tran_data.is_valid, tran_data.error_message)
                finally:
                    for output_path in (ac_result.ac_file, tran_result.tran_file):
                        if output_path and os.path.isfile(output_path):
                            os.remove(output_path)


if __name__ == "__main__":
    unittest.main()
