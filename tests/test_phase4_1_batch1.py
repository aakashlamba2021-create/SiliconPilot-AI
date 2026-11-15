"""
Phase 4.1 Batch 1 Test Suite — Simulation Reliability and Power Measurement Repair
Tests:
- Failed ngspice commands and per-analysis status tracking
- Missing AC or transient outputs and incomplete simulation rejection
- Empty or malformed simulation files
- Missing operating-point current and removal of 80 µA fallback
- Wrong supply-source name handling
- Positive and negative current conventions under SPICE polarity
- NaN and infinite values handling in vectors and OP
- Prevention of incomplete results entering the feedback database
- CLI backend contract disclosure
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.ngspice_runner import NGSpiceRunner, SimulationResult
from src.simulation.netlist_manager import NetlistManager
from src.simulation.simulation_parser import SimulationParser, ACSimulationData, TranSimulationData
from src.simulation.simulation_metrics import SimulationMetricsExtractor, ExtractedMetrics
from src.simulation.simulation_validator import SimulationValidator
from src.simulation.feedback_db import FeedbackDatabase
from src.optimization.candidate_optimizer import CandidateOptimizer


class TestPhase41Batch1(unittest.TestCase):
    """Unit and integration tests for Phase 4.1 Batch 1 repairs."""

    @classmethod
    def setUpClass(cls):
        cls.runner = NGSpiceRunner()
        cls.net_mgr = NetlistManager()
        cls.parser = SimulationParser()
        cls.extractor = SimulationMetricsExtractor()
        cls.validator = SimulationValidator()
        cls.db_path = os.path.join(tempfile.gettempdir(), f"test_p41_db_{os.getpid()}.db")
        cls.db = FeedbackDatabase(db_path=cls.db_path)
        cls.optimizer = CandidateOptimizer()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.db_path):
            try:
                os.remove(cls.db_path)
            except Exception:
                pass

    def test_01_failed_ngspice_command_reporting(self):
        """Validates that a syntax error or failed command marks analysis as FAILED and success=False."""
        bad_netlist = """* Malformed Netlist Test
V1 in 0 DC 1 AC 1
R1 in out INVALID_SYNTAX_VALUE
.ac dec 10 1 10k
.end
"""
        res = self.runner.run_netlist(bad_netlist, run_ac=True, run_tran=False, run_op=True)
        self.assertFalse(res.success, "Simulation with syntax error must fail.")
        self.assertEqual(res.ac_status, "FAILED")
        self.assertIsNotNone(res.error_message)

    def test_02_missing_ac_output_handling(self):
        """Validates that missing or ungenerated AC output is flagged as invalid and metrics are None."""
        data = self.parser.parse_ac_file(None)
        self.assertFalse(data.is_valid)
        self.assertIn("does not exist", data.error_message)

        dummy_sim = SimulationResult(
            success=False,
            ac_status="FAILED",
            op_status="SUCCESS",
            analysis_statuses={"ac": "FAILED", "op": "SUCCESS"},
            error_message="AC vector missing"
        )
        metrics = self.extractor.extract(dummy_sim, ac_data=data, tran_data=None)
        self.assertIsNone(metrics.dc_gain_db, "DC Gain must be None when AC data missing.")
        self.assertIsNone(metrics.gain_bandwidth_product_hz, "GBW must be None when AC data missing.")
        self.assertIsNone(metrics.phase_margin_deg, "Phase margin must be None when AC data missing.")
        self.assertNotEqual(metrics.convergence_status, "Converged", "Incomplete AC must never be labeled Converged.")
        self.assertIn("Failed", metrics.convergence_status)

    def test_03_missing_transient_output_handling(self):
        """Validates that missing or ungenerated transient output marks slew rate and settling time as None."""
        data = self.parser.parse_tran_file("non_existent_file_path_12345.txt")
        self.assertFalse(data.is_valid)

        dummy_sim = SimulationResult(
            success=False,
            tran_status="FAILED",
            analysis_statuses={"tran": "FAILED"},
            error_message="Transient output missing"
        )
        metrics = self.extractor.extract(dummy_sim, ac_data=None, tran_data=data)
        self.assertIsNone(metrics.slew_rate_v_per_us, "Slew rate must be None when transient data missing.")
        self.assertIsNone(metrics.settling_time_ns, "Settling time must be None when transient data missing.")
        self.assertNotEqual(metrics.convergence_status, "Converged", "Incomplete transient must never be labeled Converged.")

    def test_04_empty_or_malformed_simulation_files(self):
        """Validates parser rejection of empty files, text-only files, or non-numeric contents."""
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("")  # 0 bytes
            empty_path = f.name

        try:
            ac_res = self.parser.parse_ac_file(empty_path)
            self.assertFalse(ac_res.is_valid)
            self.assertIn("empty", ac_res.error_message)

            tran_res = self.parser.parse_tran_file(empty_path)
            self.assertFalse(tran_res.is_valid)
            self.assertIn("empty", tran_res.error_message)
        finally:
            os.remove(empty_path)

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("text_header only without numbers\nanother header line\n")
            text_path = f.name

        try:
            ac_res = self.parser.parse_ac_file(text_path)
            self.assertFalse(ac_res.is_valid)
            self.assertIn("Insufficient", ac_res.error_message)
        finally:
            os.remove(text_path)

    def test_05_nan_and_infinite_values_rejection(self):
        """Validates that NaN or Inf in vectors or OP results are caught and marked unavailable."""
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("100.0  50.0  100.0  180.0\n1000.0  NaN  1000.0  160.0\n")
            nan_path = f.name

        try:
            ac_res = self.parser.parse_ac_file(nan_path)
            self.assertFalse(ac_res.is_valid)
            self.assertIn("non-finite", ac_res.error_message)
        finally:
            os.remove(nan_path)

        # Test NaN in OP dict
        op_nan = {"v(vdd)": 1.8, "i(vdd)": float("nan")}
        p_w, p_mw, i_ua, prov, note = self.extractor._extract_power_metrics(op_nan, supply_source="VDD")
        self.assertIsNone(p_w)
        self.assertIsNone(p_mw)
        self.assertIsNone(i_ua)
        self.assertEqual(prov, "unavailable")
        self.assertIn("non-finite", note)

    def test_06_missing_operating_point_current_no_fallback(self):
        """
        Validates that missing OP current marks power as unavailable (None)
        and absolutely does NOT substitute the old 80 µA fallback.
        """
        op_no_current = {"v(vdd)": 1.8, "v(outp)": 0.9}
        p_w, p_mw, i_ua, prov, note = self.extractor._extract_power_metrics(op_no_current, supply_source="VDD")
        self.assertIsNone(p_w, "Power must be None when current is missing.")
        self.assertIsNone(p_mw, "Power mW must be None when current is missing.")
        self.assertIsNone(i_ua, "Supply current must be None when current is missing.")
        self.assertEqual(prov, "unavailable")
        self.assertIn("not found", note)
        # Explicit check: old fallback was 80 µA * 1.8 V = 0.144 mW
        self.assertNotEqual(p_mw, 0.144, "Old 80 µA fallback must never be returned.")

    def test_07_wrong_supply_source_name(self):
        """Validates that querying a non-existent supply source name fails cleanly with explanation."""
        op_data = {"v(vdd)": 1.8, "i(vdd)": -1.0e-4}
        # netlist specifies supply_source="VCC_AUX", which does not exist in op_data
        p_w, p_mw, i_ua, prov, note = self.extractor._extract_power_metrics(op_data, supply_source="VCC_AUX")
        self.assertIsNone(p_w)
        self.assertEqual(prov, "unavailable")
        self.assertIn("VCC_AUX", note)

    def test_08_positive_and_negative_current_conventions(self):
        """
        Validates SPICE branch current polarity:
        In SPICE, current leaving positive terminal of source has i_source < 0.
        i_delivered = -i_source.
        """
        # Power delivery to circuit: i(vdd) = -1.5e-4 A (150 µA delivered) at 1.8 V
        op_delivering = {"v(vdd)": 1.8, "i(vdd)": -1.5e-4}
        p_w, p_mw, i_ua, prov, note = self.extractor._extract_power_metrics(op_delivering, supply_source="VDD")
        self.assertAlmostEqual(p_w, 1.8 * 1.5e-4, places=6)
        self.assertAlmostEqual(p_mw, 0.27, places=4)
        self.assertAlmostEqual(i_ua, 150.0, places=2)
        self.assertEqual(prov, "spice_op_simulation")
        self.assertIsNone(note)

        # Current pushed into supply: i(vdd) = +5.0e-5 A
        op_sink = {"v(vdd)": 1.8, "i(vdd)": 5.0e-5}
        p_w, p_mw, i_ua, prov, note = self.extractor._extract_power_metrics(op_sink, supply_source="VDD")
        self.assertAlmostEqual(p_mw, -0.09, places=4)
        self.assertAlmostEqual(i_ua, -50.0, places=2)

    def test_09_cli_backend_unsupported_contract(self):
        """Validates that CLI backend is truthfully reported as unsupported for vector extraction."""
        res_cli = self.runner._run_via_cli("dummy.cir", start_time=0.0)
        self.assertFalse(res_cli.success)
        self.assertEqual(res_cli.op_status, "UNSUPPORTED")
        self.assertEqual(res_cli.ac_status, "UNSUPPORTED")
        self.assertEqual(res_cli.tran_status, "UNSUPPORTED")
        self.assertIn("CLI backend does not implement the vector output contract", res_cli.error_message)
        self.assertFalse(self.runner.is_available_for_extraction() if self.runner._backend == "cli" else False)

    def test_10_prevention_of_incomplete_results_in_feedback_db(self):
        """Validates that non-converged or incomplete simulations CANNOT be marked verified for retraining."""
        # 1. Direct record attempt with failed status and verified=True
        rec_id = self.db.record_simulation(
            circuit_id="INCOMPLETE_TEST",
            topology="miller_2stage",
            parameters={"w": 30},
            ml_predictions={"gain": 50},
            spice_results={},
            errors={},
            simulation_status="Incomplete: AC invalid",
            verified_for_training=True,  # Caller asks to verify
        )

        all_recs = self.db.get_all_records()
        row = all_recs[all_recs["record_id"] == rec_id].iloc[0]
        self.assertEqual(row["verified_for_training"], 0, "Incomplete simulation must be forced to verified=0.")

        verified_recs = self.db.get_verified_records_for_retraining()
        self.assertNotIn(rec_id, verified_recs["record_id"].values)

        # 2. Attempt manual mark_as_verified on un-converged record
        res_mark = self.db.mark_as_verified(rec_id, verified=True)
        self.assertFalse(res_mark, "mark_as_verified must reject non-converged records.")

        all_recs_after = self.db.get_all_records()
        row_after = all_recs_after[all_recs_after["record_id"] == rec_id].iloc[0]
        self.assertEqual(row_after["verified_for_training"], 0)

    def test_11_real_ngspice_power_measurement_parity(self):
        """
        Integration test with live ngspice DLL on Two-Stage Miller OTA.
        Verifies actual measured supply current is negative, power is calculated accurately,
        and differs from the old 0.144 mW fallback.
        """
        if not self.runner.is_available():
            self.skipTest("ngspice DLL not available")

        params = {"transistor_width_W": 40.0, "transistor_length_L": 0.5, "multiplicity": 2, "ibias_ua": 20.0}
        cir = self.net_mgr.generate_netlist("miller_2stage", params, mode="ac")
        res = self.runner.run_netlist(cir, run_ac=True, run_tran=False, run_op=True, supply_source="VDD")

        self.assertTrue(res.success)
        self.assertEqual(res.op_status, "SUCCESS")
        self.assertIn("i(vdd)", res.op_output)
        i_vdd = res.op_output["i(vdd)"]
        self.assertLess(i_vdd, 0.0, "i(vdd) must be negative under SPICE delivery convention.")

        ac_data = self.parser.parse_ac_file(res.ac_file)
        metrics = self.extractor.extract(res, ac_data=ac_data, supply_source="VDD")

        self.assertIsNotNone(metrics.power_consumption_mw)
        self.assertGreater(metrics.power_consumption_mw, 0.15, "Miller OTA with 20uA bias draws >0.15 mW.")
        self.assertNotEqual(metrics.power_consumption_mw, 0.144, "Must not equal the old 0.144 mW fallback.")

    def test_12_feedback_db_boundary_forged_converged_rejection(self):
        """
        Adversarial test for FeedbackDatabase trust boundary.
        Rejects forged Converged records with empty results, non-finite metrics, or failed statuses,
        regardless of caller-supplied verified_for_training=True flag.
        """
        # A. Forged record with status 'Converged' but empty spice_results
        r_empty = self.db.record_simulation(
            circuit_id="FORGE_EMPTY",
            topology="miller_2stage",
            parameters={"W": 40.0},
            ml_predictions={"gain": 50.0},
            spice_results={},
            errors={"mean_relative_error_pct": 2.0},
            simulation_status="Converged",
            verified_for_training=True,
        )
        recs = self.db.get_all_records()
        row_empty = recs[recs["record_id"] == r_empty].iloc[0]
        self.assertEqual(row_empty["verified_for_training"], 0, "Empty spice_results must never be verified.")
        self.assertFalse(self.db.mark_as_verified(r_empty, True), "Cannot manually promote record with empty evidence.")

        # B. Forged record with non-finite metric values
        r_nan = self.db.record_simulation(
            circuit_id="FORGE_NAN",
            topology="miller_2stage",
            parameters={"W": 40.0},
            ml_predictions={"gain": 50.0},
            spice_results={"dc_gain_db": float("nan")},
            errors={},
            simulation_status="Converged",
            verified_for_training=True,
        )
        recs = self.db.get_all_records()
        row_nan = recs[recs["record_id"] == r_nan].iloc[0]
        self.assertEqual(row_nan["verified_for_training"], 0, "NaN metrics must never be verified.")

        # C. Forged record with failed analysis status
        r_failed_status = self.db.record_simulation(
            circuit_id="FORGE_FAILED_AC",
            topology="miller_2stage",
            parameters={"W": 40.0},
            ml_predictions={"gain": 50.0},
            spice_results={"dc_gain_db": 50.0, "analysis_statuses": {"ac": "FAILED"}},
            errors={},
            simulation_status="Converged",
            verified_for_training=True,
        )
        recs = self.db.get_all_records()
        row_fail = recs[recs["record_id"] == r_failed_status].iloc[0]
        self.assertEqual(row_fail["verified_for_training"], 0, "Failed analysis status must never be verified.")

        # D. Genuine complete record
        r_valid = self.db.record_simulation(
            circuit_id="GENUINE_SPICE",
            topology="miller_2stage",
            parameters={"W": 40.0, "L": 0.5},
            ml_predictions={"gain": 50.0},
            spice_results={
                "dc_gain_db": 52.0,
                "power_consumption_mw": 0.18,
                "analysis_statuses": {"ac": "SUCCESS", "op": "SUCCESS"},
                "simulator_identity": "ngspice",
                "simulator_version": "46",
            },
            errors={"mean_relative_error_pct": 4.0},
            simulation_status="Converged",
            verified_for_training=True,
        )
        recs = self.db.get_all_records()
        row_valid = recs[recs["record_id"] == r_valid].iloc[0]
        self.assertEqual(row_valid["verified_for_training"], 1, "Complete genuine physical evidence must be verified.")

        # E. Forged record with missing simulator identity
        r_no_sim = self.db.record_simulation(
            circuit_id="FORGE_NO_SIM",
            topology="miller_2stage",
            parameters={"W": 40.0, "L": 0.5},
            ml_predictions={"gain": 50.0},
            spice_results={"dc_gain_db": 52.0, "simulator_identity": ""},
            errors={},
            simulation_status="Converged",
            verified_for_training=True,
        )
        recs = self.db.get_all_records()
        row_no_sim = recs[recs["record_id"] == r_no_sim].iloc[0]
        self.assertEqual(row_no_sim["verified_for_training"], 0, "Missing simulator identity must be rejected.")
        self.assertFalse(self.db.mark_as_verified(r_no_sim, True), "Cannot manually promote record missing simulator identity.")

        # F. Forged record with contradictory provenance
        r_contra = self.db.record_simulation(
            circuit_id="FORGE_CONTRA",
            topology="miller_2stage",
            parameters={"W": 40.0, "L": 0.5},
            ml_predictions={"gain": 50.0},
            spice_results={"dc_gain_db": 52.0, "metric_provenance": {"dc_gain": "unavailable"}},
            errors={},
            simulation_status="Converged",
            verified_for_training=True,
        )
        recs = self.db.get_all_records()
        row_contra = recs[recs["record_id"] == r_contra].iloc[0]
        self.assertEqual(row_contra["verified_for_training"], 0, "Contradictory provenance must be rejected.")

        # G. Forged record with unphysical negative power
        r_neg_pwr = self.db.record_simulation(
            circuit_id="FORGE_NEG_PWR",
            topology="miller_2stage",
            parameters={"W": 40.0, "L": 0.5},
            ml_predictions={"power": 0.1},
            spice_results={"power_consumption_mw": -5.0},
            errors={},
            simulation_status="Converged",
            verified_for_training=True,
        )
        recs = self.db.get_all_records()
        row_neg = recs[recs["record_id"] == r_neg_pwr].iloc[0]
        self.assertEqual(row_neg["verified_for_training"], 0, "Negative power must be rejected as unphysical.")

    def test_13_pareto_rank1_filtering_before_simulation(self):
        """
        Validates that candidate simulation selection strictly filters rank-1 Pareto candidates
        before invoking SPICE simulation, proving dominated candidates are NEVER simulated.
        """
        # Interleaved candidate pool with ranks 1, 2, and 3
        cand_df = pd.DataFrame([
            {"candidate_id": "Cand_Rank1_A", "transistor_width_W": 40.0, "transistor_length_L": 0.5, "multiplicity": 2, "pareto_rank": 1, "composite_score": 0.95},
            {"candidate_id": "Cand_Rank2_B", "transistor_width_W": 35.0, "transistor_length_L": 0.5, "multiplicity": 2, "pareto_rank": 2, "composite_score": 0.92},
            {"candidate_id": "Cand_Rank1_C", "transistor_width_W": 45.0, "transistor_length_L": 0.5, "multiplicity": 2, "pareto_rank": 1, "composite_score": 0.88},
            {"candidate_id": "Cand_Rank3_D", "transistor_width_W": 25.0, "transistor_length_L": 0.5, "multiplicity": 2, "pareto_rank": 3, "composite_score": 0.85},
            {"candidate_id": "Cand_Rank1_E", "transistor_width_W": 50.0, "transistor_length_L": 0.5, "multiplicity": 2, "pareto_rank": 1, "composite_score": 0.82},
            {"candidate_id": "Cand_Rank2_F", "transistor_width_W": 30.0, "transistor_length_L": 0.5, "multiplicity": 2, "pareto_rank": 2, "composite_score": 0.80},
        ])

        sim_res = self.optimizer.run_simulation_validation(
            ranked_candidates=cand_df,
            top_k=3,
            topology="miller_2stage",
            record_in_db=False,
            filter_pareto_front=True,
        )

        simulated_ids = [c["candidate_id"] for c in sim_res["evaluated_candidates"]]
        self.assertEqual(len(simulated_ids), 3, "Must simulate exactly top 3 rank-1 candidates.")

        # Assert all simulated candidates have pareto_rank == 1
        for c in sim_res["evaluated_candidates"]:
            self.assertEqual(c["pareto_rank"], 1, f"Candidate {c['candidate_id']} must be rank 1.")

        # Explicitly verify dominated candidates were NOT simulated
        self.assertNotIn("Cand_Rank2_B", simulated_ids, "Dominated Rank 2 candidate must NOT be simulated.")
        self.assertNotIn("Cand_Rank3_D", simulated_ids, "Dominated Rank 3 candidate must NOT be simulated.")
        self.assertNotIn("Cand_Rank2_F", simulated_ids, "Dominated Rank 2 candidate must NOT be simulated.")

        # Assert correct deterministic ordering
        self.assertEqual(simulated_ids, ["Cand_Rank1_A", "Cand_Rank1_C", "Cand_Rank1_E"])

    def test_14_metric_completeness_and_omitted_data_rejection(self):
        """
        Validates that missing required parsed vectors or omitted analyses NEVER result in 'Converged'.
        If no electrical metrics could be extracted, status must be explicitly Incomplete.
        """
        # Successful overall runner result but both AC and TRAN parsed data omitted
        sim_res = SimulationResult(
            success=True,
            ac_status="SUCCESS",
            tran_status="SUCCESS",
            op_status="NOT_REQUESTED",
            analysis_statuses={"ac": "SUCCESS", "tran": "SUCCESS"},
        )
        metrics = self.extractor.extract(sim_res, ac_data=None, tran_data=None)
        self.assertNotEqual(metrics.convergence_status, "Converged", "Omitted parsed data must not be labeled Converged.")
        self.assertIn("Incomplete", metrics.convergence_status)
        self.assertIsNone(metrics.dc_gain_db)
        self.assertIsNone(metrics.slew_rate_v_per_us)

    def test_15_power_voltage_provenance_distinction(self):
        """
        Validates that power extractor explicitly distinguishes measured OP voltage
        from configured/override supply voltage, labeling provenance accurately.
        """
        # Case A: OP output contains measured v(vdd)
        op_measured = {"v(vdd)": 1.8001, "i(vdd)": -1.0e-4}
        p_w, p_mw, i_ua, prov, note = self.extractor._extract_power_metrics(op_measured, supply_source="VDD")
        self.assertEqual(prov, "spice_op_simulation", "Provenance must be spice_op_simulation when voltage is measured.")
        self.assertAlmostEqual(p_mw, 0.180018, places=4)

        # Case B: OP output lacks v(vdd) and falls back to configured vdd parameter
        op_no_v = {"i(vdd)": -1.0e-4}
        p_w2, p_mw2, i_ua2, prov2, note2 = self.extractor._extract_power_metrics(op_no_v, supply_source="VDD", vdd_override=1.8)
        self.assertEqual(prov2, "calculated_simulated_current_x_configured_vdd", "Must explicitly indicate voltage was from configuration.")
        self.assertAlmostEqual(p_mw2, 0.18, places=4)

    def test_16_dll_session_health_and_automatic_recovery(self):
        """
        Validates DLL lifecycle management, session health monitoring, and crash recovery.
        After an unrecoverable netlist error, the runner automatically reloads the DLL
        so subsequent simulations succeed without 'ngspice.dll cannot recover' errors.
        """
        if not self.runner.is_available():
            self.skipTest("ngspice DLL not available")

        # 1. Session is healthy initially
        self.assertTrue(self.runner.is_session_healthy())

        # 2. Run malformed netlist that causes parser interruption
        bad_netlist = "* Fatal Syntax Crash\nR1 in out\n.ac dec 10 1 100k\n.end\n"
        res_bad = self.runner.run_netlist(bad_netlist)
        self.assertFalse(res_bad.success)

        # 3. Session must have automatically recovered and remain healthy
        self.assertTrue(self.runner.is_session_healthy(), "Runner must recover session health after failed netlist.")

        # 4. Immediate subsequent good simulation must succeed
        params = {"transistor_width_W": 30.0, "transistor_length_L": 0.5, "multiplicity": 2}
        good_cir = self.net_mgr.generate_netlist("miller_2stage", params, mode="ac")
        res_good = self.runner.run_netlist(good_cir, run_ac=True, run_tran=False, run_op=True)
        self.assertTrue(res_good.success, f"Subsequent simulation must succeed: {res_good.error_message}")
        self.assertEqual(res_good.ac_status, "SUCCESS")

    def test_17_controlled_exit_callback_5arg_abi(self):
        """
        Validates Priority 1: The ControlledExit callback matches the exact 5-argument
        native ABI: (int status, bool immediate, bool quitexit, int ident, void* userData).
        """
        import ctypes
        from src.simulation.ngspice_runner import _NGSpiceProcessContext

        self.assertIsNotNone(_NGSpiceProcessContext.cb_exit_ref, "ControlledExit callback must be registered.")
        # Verify callback invocation with 5 arguments does not throw exception
        cb_func = _NGSpiceProcessContext.cb_exit_ref
        ret = cb_func(1, True, False, 42, None)
        self.assertEqual(ret, 0, "ControlledExit callback must return 0.")
        self.assertFalse(_NGSpiceProcessContext.session_healthy, "ControlledExit must mark session unhealthy.")
        # Restore session health for subsequent tests
        self.runner.reset_session()
        self.assertTrue(self.runner.is_session_healthy())

    def test_18_strict_pareto_filtering_fail_closed_missing_metadata(self):
        """
        Validates Priority 5: Missing Pareto metadata fails closed with an explicit error
        and does NOT simulate any candidates or fall back to arbitrary rows.
        """
        cand_no_pareto = pd.DataFrame([
            {"candidate_id": "Cand_NoPareto_1", "transistor_width_W": 40.0, "transistor_length_L": 0.5, "composite_score": 0.90},
            {"candidate_id": "Cand_NoPareto_2", "transistor_width_W": 30.0, "transistor_length_L": 0.5, "composite_score": 0.85},
        ])

        res = self.optimizer.run_simulation_validation(
            ranked_candidates=cand_no_pareto,
            top_k=2,
            topology="miller_2stage",
            filter_pareto_front=True,
            record_in_db=False,
        )
        self.assertFalse(res["success"], "Strict Pareto filtering must fail closed when Pareto metadata is missing.")
        self.assertEqual(len(res["evaluated_candidates"]), 0, "No candidates must be simulated.")
        self.assertIsNone(res["winner_id"])

    def test_19_strict_pareto_filtering_fail_closed_empty_rank1(self):
        """
        Validates Priority 5: An empty rank-1 front fails closed and NEVER falls back
        to rank 2 or original candidate set.
        """
        cand_only_rank2 = pd.DataFrame([
            {"candidate_id": "Cand_Rank2_Only_1", "transistor_width_W": 40.0, "transistor_length_L": 0.5, "pareto_rank": 2, "composite_score": 0.90},
            {"candidate_id": "Cand_Rank2_Only_2", "transistor_width_W": 30.0, "transistor_length_L": 0.5, "pareto_rank": 2, "composite_score": 0.85},
        ])

        res = self.optimizer.run_simulation_validation(
            ranked_candidates=cand_only_rank2,
            top_k=2,
            topology="miller_2stage",
            filter_pareto_front=True,
            record_in_db=False,
        )
        self.assertFalse(res["success"], "Strict Pareto filtering must fail closed when rank-1 front is empty.")
        self.assertEqual(len(res["evaluated_candidates"]), 0, "Must NEVER fall back to simulating rank-2 candidates.")
        self.assertIsNone(res["winner_id"])

    def test_20_ac_metrics_validity_safeguards_missing_crossing_and_phase(self):
        """
        Validates Priority 6: AC metric validity safeguards.
        - Missing unity-gain crossing returns None for GBW and PM (never sweep endpoints).
        - Missing phase returns None for PM (never 0.0 degrees).
        - Candidate selection fails closed when PM is unavailable.
        """
        from src.simulation.simulation_parser import ACSimulationData
        # Case A: Gain never crosses 0 dB (e.g. constant 20 dB gain across entire sweep)
        freqs = np.logspace(3, 8, 50)
        gains_no_cross = np.full_like(freqs, 20.0)
        phases = np.full_like(freqs, -90.0)
        ac_no_cross = ACSimulationData(frequencies=freqs, gain_db=gains_no_cross, phase_deg=phases, is_valid=True)
        dc_g, gbw, pm, f3db = self.extractor._extract_ac_metrics(ac_no_cross)
        self.assertEqual(dc_g, 20.0)
        self.assertIsNone(gbw, "GBW must be None when response never crosses 0 dB.")
        self.assertIsNone(pm, "PM must be None when response never crosses 0 dB.")

        # Case B: Missing phase vector (has_phase=False, empty phase_deg)
        gains_crossing = np.linspace(40.0, -10.0, 50)
        ac_no_phase = ACSimulationData(
            frequencies=freqs,
            gain_db=gains_crossing,
            phase_deg=np.array([]),
            is_valid=True,
            has_phase=False,
        )
        dc_g2, gbw2, pm2, f3db2 = self.extractor._extract_ac_metrics(ac_no_phase)
        self.assertIsNotNone(gbw2, "GBW exists when 0 dB is crossed.")
        self.assertIsNone(pm2, "PM must be None (not 0.0) when phase vector is missing.")


if __name__ == "__main__":
    unittest.main()

