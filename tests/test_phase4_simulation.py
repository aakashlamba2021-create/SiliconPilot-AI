"""
Test Suite for SiliconPilot AI — Phase 4 SPICE Simulation Engine
Tests ngspice availability, parametric netlists, solver execution, vector parsing,
metric extraction, ML-vs-SPICE validation, simulation-in-the-loop optimization, and feedback DB.
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import numpy as np
import pandas as pd

from src.simulation.ngspice_runner import NGSpiceRunner, SimulationResult
from src.simulation.netlist_manager import NetlistManager
from src.simulation.simulation_parser import SimulationParser, ACSimulationData, TranSimulationData
from src.simulation.simulation_metrics import SimulationMetricsExtractor, ExtractedMetrics
from src.simulation.simulation_validator import SimulationValidator, ValidationReport
from src.simulation.feedback_db import FeedbackDatabase
from src.optimization.candidate_optimizer import CandidateOptimizer


class TestPhase4Simulation(unittest.TestCase):
    """Phase 4 unit and integration test suite."""

    @classmethod
    def setUpClass(cls):
        cls.runner = NGSpiceRunner()
        cls.net_mgr = NetlistManager()
        cls.parser = SimulationParser()
        cls.extractor = SimulationMetricsExtractor()
        cls.validator = SimulationValidator()
        cls.db_path = os.path.join(tempfile.gettempdir(), f"test_sim_results_{os.getpid()}_{uuid.uuid4().hex}.db")
        cls.db = FeedbackDatabase(db_path=cls.db_path)
        cls.optimizer = CandidateOptimizer()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.db_path):
            try:
                os.remove(cls.db_path)
            except Exception:
                pass

    def test_01_ngspice_availability(self):
        """Checks whether ngspice backend is detected and ready."""
        self.assertTrue(self.runner.is_available(), "ngspice backend must be available.")
        info = self.runner.get_backend_info()
        self.assertIn("ngspice", info.lower())
        print(f"✅ test_01_ngspice_availability: {info}")

    def test_02_netlist_generation_all_topologies(self):
        """Verifies netlist generation for all 4 supported analog topologies."""
        topologies = ["miller_2stage", "five_transistors_ota", "folded_cascode", "telescopic"]
        test_params = {
            "transistor_width_W": 45.0,
            "transistor_length_L": 0.5,
            "finger_count": 8,
            "multiplicity": 2,
            "cload_pf": 2.0,
            "ibias_ua": 25.0,
        }

        for top in topologies:
            ac_net = self.net_mgr.generate_netlist(top, test_params, mode="ac")
            self.assertIn(".ac", ac_net)
            self.assertIn(".end", ac_net)
            self.assertIn("NMOS_180", ac_net)

            tran_net = self.net_mgr.generate_netlist(top, test_params, mode="tran")
            self.assertIn(".tran", tran_net)
            self.assertIn("pulse", tran_net.lower())

        print("✅ test_02_netlist_generation_all_topologies passed.")

    def test_03_simulation_execution_ac_and_tran(self):
        """Executes actual ngspice AC and Transient simulation on Miller OTA."""
        params = {
            "transistor_width_W": 40.0,
            "transistor_length_L": 0.5,
            "multiplicity": 2,
            "ibias_ua": 20.0,
            "cload_pf": 1.5,
        }
        # Run AC
        ac_net = self.net_mgr.generate_netlist("miller_2stage", params, mode="ac")
        res_ac = self.runner.run_netlist(ac_net, run_ac=True, run_tran=False, run_op=True)
        self.assertTrue(res_ac.success, f"AC Simulation failed: {res_ac.error_message}")
        self.assertIsNotNone(res_ac.ac_file)
        self.assertTrue(os.path.exists(res_ac.ac_file))

        # Run TRAN
        tran_net = self.net_mgr.generate_netlist("miller_2stage", params, mode="tran")
        res_tran = self.runner.run_netlist(tran_net, run_ac=False, run_tran=True, run_op=False)
        self.assertTrue(res_tran.success, f"TRAN Simulation failed: {res_tran.error_message}")
        self.assertIsNotNone(res_tran.tran_file)
        self.assertTrue(os.path.exists(res_tran.tran_file))

        print("✅ test_03_simulation_execution_ac_and_tran passed.")

    def test_04_simulation_parsing(self):
        """Verifies vector parsing of frequency and time-domain simulation outputs."""
        params = {"transistor_width_W": 35.0, "transistor_length_L": 0.5, "multiplicity": 2}
        ac_net = self.net_mgr.generate_netlist("miller_2stage", params, mode="ac")
        res_ac = self.runner.run_netlist(ac_net, run_ac=True, run_tran=False)
        ac_data = self.parser.parse_ac_file(res_ac.ac_file)

        self.assertTrue(ac_data.is_valid)
        self.assertGreater(len(ac_data.frequencies), 10)
        self.assertEqual(len(ac_data.frequencies), len(ac_data.gain_db))
        self.assertEqual(len(ac_data.frequencies), len(ac_data.phase_deg))

        tran_net = self.net_mgr.generate_netlist("miller_2stage", params, mode="tran")
        res_tran = self.runner.run_netlist(tran_net, run_ac=False, run_tran=True)
        tran_data = self.parser.parse_tran_file(res_tran.tran_file)

        self.assertTrue(tran_data.is_valid)
        self.assertGreater(len(tran_data.time_s), 10)
        self.assertEqual(len(tran_data.time_s), len(tran_data.vout_v))

        print("✅ test_04_simulation_parsing passed.")

    def test_05_metric_extraction(self):
        """Tests extraction of standard analog IC performance metrics."""
        params = {"transistor_width_W": 40.0, "transistor_length_L": 0.5, "multiplicity": 2}
        ac_net = self.net_mgr.generate_netlist("miller_2stage", params, mode="ac")
        res_ac = self.runner.run_netlist(ac_net, run_ac=True, run_tran=False, run_op=True)
        ac_data = self.parser.parse_ac_file(res_ac.ac_file)

        tran_net = self.net_mgr.generate_netlist("miller_2stage", params, mode="tran")
        res_tran = self.runner.run_netlist(tran_net, run_ac=False, run_tran=True, run_op=False)
        tran_data = self.parser.parse_tran_file(res_tran.tran_file)

        metrics = self.extractor.extract(res_ac, ac_data, tran_data, vdd=1.8)

        self.assertEqual(metrics.convergence_status, "Converged")
        self.assertGreater(metrics.dc_gain_db, 30.0, "DC Gain should be >30 dB for 2-stage OTA")
        self.assertGreater(metrics.gain_bandwidth_product_hz, 1.0e6, "GBW should be >1 MHz")
        self.assertGreater(metrics.phase_margin_deg, 20.0, "Phase Margin should be >20°")
        self.assertGreater(metrics.slew_rate_v_per_us, 1.0, "Slew rate should be >1 V/µs")
        self.assertGreater(metrics.power_consumption_mw, 0.01, "Power dissipation should be positive")

        print(f"✅ test_05_metric_extraction passed: Gain={metrics.dc_gain_db:.1f}dB, GBW={metrics.gain_bandwidth_product_hz/1e6:.1f}MHz, PM={metrics.phase_margin_deg:.1f}°")

    def test_06_failure_handling(self):
        """Tests robust handling of non-convergence or invalid netlist syntax."""
        bad_netlist = "* Invalid Netlist\nMN1 outp in tail vss UNKNOWN_MODEL_XYZ W=1u L=1u\n.ac dec 10 1 100k\n.end\n"
        res = self.runner.run_netlist(bad_netlist)
        # Should gracefully return success=False without throwing unhandled exceptions
        self.assertFalse(res.success)
        self.assertIsNotNone(res.error_message)

        metrics = self.extractor.extract(res, None, None)
        self.assertIn("Failed", metrics.convergence_status)
        print("✅ test_06_failure_handling passed.")

    def test_07_prediction_vs_simulation_validation(self):
        """Tests validation report comparing ML estimates to SPICE simulated truths."""
        mock_metrics = ExtractedMetrics(
            dc_gain_db=54.2,
            gain_bandwidth_product_hz=42.0e6,
            phase_margin_deg=58.5,
            cutoff_freq_3db_hz=100.0e3,
            slew_rate_v_per_us=35.0,
            power_consumption_w=0.00018,
            power_consumption_mw=0.18,
            supply_current_ua=100.0,
            settling_time_ns=25.0,
            convergence_status="Converged",
        )

        mock_ml = {
            "dc_gain": 52.0,
            "gain_bandwidth_product": 40.0e6,
            "phase_margin": 60.0,
            "slew_rate": 32.0,
            "power_consumption": 0.00015,
        }

        report = self.validator.validate_candidate(mock_ml, mock_metrics, candidate_id="VAL_001", topology="miller_2stage")
        self.assertIn("VALIDATED", report.validation_verdict)
        self.assertLess(report.mean_relative_error_pct, 20.0)

        df = report.to_dataframe()
        self.assertEqual(len(df), 5)
        self.assertIn("Rel Error (%)", df.columns)
        print(f"✅ test_07_prediction_vs_simulation_validation passed: Mean Error {report.mean_relative_error_pct:.1f}%")

    def test_08_simulation_in_the_loop_candidate_selection(self):
        """Tests end-to-end multi-candidate SPICE ranking and winner selection."""
        candidates = pd.DataFrame([
            {
                "candidate_id": "Cand_A",
                "transistor_width_W": 40.0,
                "transistor_length_L": 0.5,
                "multiplicity": 2,
                "dc_gain": 55.0,
                "gain_bandwidth_product": 45.0e6,
                "phase_margin": 55.0,
                "slew_rate": 40.0,
                "power_consumption": 0.00015,
                "pareto_rank": 1,
            },
            {
                "candidate_id": "Cand_B",
                "transistor_width_W": 30.0,
                "transistor_length_L": 0.6,
                "multiplicity": 2,
                "dc_gain": 50.0,
                "gain_bandwidth_product": 35.0e6,
                "phase_margin": 65.0,
                "slew_rate": 30.0,
                "power_consumption": 0.00012,
                "pareto_rank": 1,
            },
            {
                "candidate_id": "Cand_C",
                "transistor_width_W": 20.0,
                "transistor_length_L": 0.4,
                "multiplicity": 1,
                "dc_gain": 45.0,
                "gain_bandwidth_product": 25.0e6,
                "phase_margin": 70.0,
                "slew_rate": 20.0,
                "power_consumption": 0.00010,
                "pareto_rank": 2,
            },
        ])

        sim_results = self.optimizer.run_simulation_validation(
            ranked_candidates=candidates,
            top_k=3,
            topology="miller_2stage",
            record_in_db=False,
        )

        self.assertTrue(sim_results["success"])
        # Pareto-front filtering ensures only rank-1 candidates (Cand_A and Cand_B) are simulated
        self.assertEqual(len(sim_results["evaluated_candidates"]), 2)
        self.assertIsNotNone(sim_results["winner_id"])
        self.assertIn("winner_justification", sim_results)
        print(f"✅ test_08_simulation_in_the_loop_candidate_selection passed: Winner is {sim_results['winner_id']}")

    def test_09_feedback_database(self):
        """Tests SQLite logging of simulation validation runs and retraining curation."""
        params = {"W": 40.0, "L": 0.5}
        ml_preds = {"dc_gain": 55.0}
        spice_res = {"dc_gain_db": 54.0}
        errors = {"mean_relative_error_pct": 5.2}

        rec_id = self.db.record_simulation(
            circuit_id="UNIT_TEST_01",
            topology="miller_2stage",
            parameters=params,
            ml_predictions=ml_preds,
            spice_results=spice_res,
            errors=errors,
            simulation_status="Converged",
            verified_for_training=True,
        )

        self.assertIsNotNone(rec_id)
        all_recs = self.db.get_all_records()
        self.assertGreaterEqual(len(all_recs), 1)

        verified_recs = self.db.get_verified_records_for_retraining()
        self.assertGreaterEqual(len(verified_recs), 1)
        self.assertEqual(verified_recs.iloc[0]["record_id"], rec_id)

        print(f"✅ test_09_feedback_database passed: Logged {rec_id}")

    def test_10_dashboard_pages_importable(self):
        """Verifies that all 8 dashboard pages are properly integrated and importable."""
        from dashboard.pages import (
            overview,
            dataset_explorer,
            model_training,
            layout_prediction,
            candidate_comparison,
            ai_assistant,
            system_info,
            simulation_studio,
        )
        self.assertTrue(callable(getattr(simulation_studio, "render", None)))
        print("✅ test_10_dashboard_pages_importable passed.")


if __name__ == "__main__":
    unittest.main()
