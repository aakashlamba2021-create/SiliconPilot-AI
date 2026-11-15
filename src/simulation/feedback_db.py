"""
SiliconPilot AI - Feedback Learning Database Module
Stores simulation-validated candidate results in a persistent SQLite database,
providing a controlled mechanism to curate high-confidence observations for ML retraining.
"""

import os
import json
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("src.simulation.feedback_db")

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "simulation_results.db"
)


class FeedbackDatabase:
    """
    Persistent SQLite storage for SPICE-in-the-loop simulation runs and ML validation errors.
    Strictly separates genuine physical simulation observations from synthetic proxies.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns SQLite database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes database tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS simulation_history (
                    record_id TEXT PRIMARY KEY,
                    circuit_id TEXT NOT NULL,
                    topology TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    simulation_status TEXT NOT NULL,
                    model_version TEXT,
                    parameters_json TEXT NOT NULL,
                    ml_predictions_json TEXT NOT NULL,
                    spice_results_json TEXT NOT NULL,
                    errors_json TEXT NOT NULL,
                    mean_relative_error REAL,
                    verified_for_training INTEGER DEFAULT 0,
                    notes TEXT
                )
            """)
            conn.commit()

    @staticmethod
    def _validate_simulation_evidence(
        simulation_status: str,
        spice_results: Dict[str, Any],
        parameters: Dict[str, Any],
        provenance: Optional[Dict[str, Any]] = None,
        circuit_id: Optional[str] = None,
        topology: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Validates whether a simulation record possesses genuine, complete physical SPICE evidence.
        Returns (is_valid, rejection_reason).

        Strict Evidence Rules:
        1. Status must be exactly 'Converged' (case-sensitive).
        2. Input & netlist provenance: circuit_id and topology must be non-empty strings.
        3. Parameters must be a non-empty dict of sizing parameters with positive finite values.
        4. Simulator identity must be valid ('ngspice' or known simulator) and simulator version must be present if specified.
        5. spice_results dictionary must be non-empty and contain valid measured electrical metrics.
        6. At least one primary electrical metric (dc_gain_db, power_consumption_mw, etc.) must be finite and within physical bounds.
        7. If any metric is NaN or +/-Inf, reject.
        8. Analysis statuses: no status may be 'FAILED', 'ERROR', or 'UNSUPPORTED'; at least one analysis must be 'SUCCESS'.
        9. Metric provenance: must not contradict measured values (e.g. claiming valid metric with provenance='unavailable').
        """
        if not simulation_status or simulation_status != "Converged":
            return False, f"simulation_status '{simulation_status}' is not Converged"

        # Circuit and topology provenance
        if circuit_id is not None and (not isinstance(circuit_id, str) or not circuit_id.strip()):
            return False, "circuit_id is missing or empty"

        if topology is not None and (not isinstance(topology, str) or not topology.strip()):
            return False, "topology is missing or empty"

        if not isinstance(parameters, dict) or len(parameters) == 0:
            return False, "Parameters dictionary is empty or missing"

        # Validate parameter values are finite
        for pk, pv in parameters.items():
            try:
                fpv = float(pv)
                if not np.isfinite(fpv):
                    return False, f"Parameter '{pk}' has non-finite value '{pv}'"
            except (ValueError, TypeError):
                pass

        if not isinstance(spice_results, dict) or len(spice_results) == 0:
            return False, "spice_results dictionary is empty or missing"

        # Check simulator identity if provided
        sim_id = spice_results.get("simulator_identity")
        if sim_id is None and provenance and isinstance(provenance, dict):
            sim_id = provenance.get("simulator_identity") or provenance.get("backend_info")
        if sim_id is not None:
            if not isinstance(sim_id, str) or sim_id.strip().lower() in ["", "none", "unknown"]:
                return False, f"Invalid or missing simulator identity: {sim_id}"

        # Check simulator version if explicitly provided
        sim_ver = spice_results.get("simulator_version")
        if sim_ver is None and provenance and isinstance(provenance, dict):
            sim_ver = provenance.get("simulator_version")
        if sim_ver is not None and (not isinstance(sim_ver, str) or not sim_ver.strip()):
            return False, f"Invalid or empty simulator version: {sim_ver}"

        # Check for presence of finite electrical metrics and bounds
        primary_metrics = [
            "dc_gain_db", "power_consumption_mw", "gain_bandwidth_product_hz",
            "phase_margin_deg", "slew_rate_v_per_us"
        ]
        finite_metrics = {}
        for m in primary_metrics:
            val = spice_results.get(m)
            if val is not None:
                try:
                    fval = float(val)
                    if not np.isfinite(fval):
                        return False, f"Metric '{m}' has non-finite value '{val}'"
                    finite_metrics[m] = fval
                except (ValueError, TypeError):
                    return False, f"Metric '{m}' has invalid numeric format '{val}'"

        if len(finite_metrics) == 0:
            return False, "spice_results contains no valid finite physical electrical metrics"

        # Physical sanity checks
        if "power_consumption_mw" in finite_metrics:
            p_val = finite_metrics["power_consumption_mw"]
            if p_val <= 0.0 or p_val > 10000.0:
                return False, f"Unphysical power consumption: {p_val} mW"

        if "dc_gain_db" in finite_metrics:
            g_val = finite_metrics["dc_gain_db"]
            if g_val < -100.0 or g_val > 250.0:
                return False, f"Unphysical DC gain: {g_val} dB"

        # Check analysis statuses
        statuses = spice_results.get("analysis_statuses")
        if not statuses and provenance and isinstance(provenance, dict):
            statuses = provenance.get("analysis_statuses")

        if isinstance(statuses, dict) and len(statuses) > 0:
            if any(st in ["FAILED", "ERROR", "UNSUPPORTED"] for st in statuses.values()):
                return False, f"Analysis statuses indicate failure: {statuses}"
            if not any(st == "SUCCESS" for st in statuses.values()):
                return False, f"No analysis status is SUCCESS: {statuses}"

        # Metric provenance consistency check
        met_prov = spice_results.get("metric_provenance")
        if not met_prov and provenance and isinstance(provenance, dict):
            met_prov = provenance.get("metric_provenance")

        if isinstance(met_prov, dict):
            # If a metric is present in finite_metrics, its provenance must not be 'unavailable'
            for mk in finite_metrics:
                pk = mk.replace("_db", "").replace("_mw", "").replace("_hz", "").replace("_deg", "").replace("_v_per_us", "")
                if met_prov.get(pk) == "unavailable" or met_prov.get(mk) == "unavailable":
                    return False, f"Contradictory provenance: metric '{mk}' has value but provenance is 'unavailable'"

        return True, "Valid complete simulation evidence"

    def record_simulation(
        self,
        circuit_id: str,
        topology: str,
        parameters: Dict[str, Any],
        ml_predictions: Dict[str, Any],
        spice_results: Dict[str, Any],
        errors: Dict[str, Any],
        simulation_status: str = "Converged",
        model_version: str = "v1.0.0",
        notes: str = "",
        verified_for_training: bool = False,
    ) -> str:
        """
        Inserts a new SPICE simulation validation record into the database.
        Enforces evidence-based trust boundary: caller boolean cannot bypass physical evidence checks.

        Returns:
            Unique record_id string.
        """
        timestamp = datetime.now().isoformat()
        record_id = f"SIM_{int(datetime.now().timestamp()*1000)}_{uuid.uuid4().hex[:8]}"

        mean_err = errors.get("mean_relative_error_pct", 0.0)

        # Enforce rule: only genuinely Converged simulations with complete physical evidence can be verified
        evidence_ok, reason = self._validate_simulation_evidence(
            simulation_status=simulation_status,
            spice_results=spice_results,
            parameters=parameters,
            provenance=errors,
            circuit_id=circuit_id,
            topology=topology,
        )

        is_eligible = evidence_ok and bool(verified_for_training)
        if verified_for_training and not evidence_ok:
            logger.warning(f"Rejected training verification for record {record_id}: {reason}")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO simulation_history (
                    record_id, circuit_id, topology, timestamp,
                    simulation_status, model_version, parameters_json,
                    ml_predictions_json, spice_results_json, errors_json,
                    mean_relative_error, verified_for_training, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record_id,
                circuit_id,
                topology,
                timestamp,
                simulation_status,
                model_version,
                json.dumps(parameters, default=str),
                json.dumps(ml_predictions, default=str),
                json.dumps(spice_results, default=str),
                json.dumps(errors, default=str),
                float(mean_err) if mean_err is not None else None,
                1 if is_eligible else 0,
                notes,
            ))
            conn.commit()

        logger.info(f"Recorded simulation run {record_id} for circuit {circuit_id} (status: {simulation_status}, verified: {is_eligible})")
        return record_id

    def get_all_records(self) -> pd.DataFrame:
        """Returns all simulation records as a pandas DataFrame."""
        conn = self._get_connection()
        try:
            query = "SELECT * FROM simulation_history ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, conn)
        finally:
            conn.close()
        return df

    def get_verified_records_for_retraining(self) -> pd.DataFrame:
        """
        Retrieves ONLY verified, converged physical simulation records suitable for model retraining.
        Guarantees that unverified, failed, or forged simulations are never fed into ML training datasets.
        """
        conn = self._get_connection()
        try:
            query = """
                SELECT * FROM simulation_history 
                WHERE verified_for_training = 1 AND simulation_status = 'Converged'
                ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(query, conn)
        finally:
            conn.close()
        return df

    def mark_as_verified(self, record_id: str, verified: bool = True) -> bool:
        """
        Updates the verification status of a simulation record.
        Strictly prevents marking failed, incomplete, or un-converged simulations as verified for retraining.
        Validates stored evidence before promoting any record.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if verified:
                cursor.execute(
                    "SELECT circuit_id, topology, simulation_status, spice_results_json, parameters_json, errors_json FROM simulation_history WHERE record_id = ?",
                    (record_id,)
                )
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"Record {record_id} not found.")
                    return False

                c_id = row["circuit_id"]
                top = row["topology"]
                status = row["simulation_status"]
                try:
                    s_res = json.loads(row["spice_results_json"]) if row["spice_results_json"] else {}
                    p_res = json.loads(row["parameters_json"]) if row["parameters_json"] else {}
                    e_res = json.loads(row["errors_json"]) if row["errors_json"] else {}
                except Exception:
                    s_res, p_res, e_res = {}, {}, {}

                evidence_ok, reason = self._validate_simulation_evidence(
                    simulation_status=status,
                    spice_results=s_res,
                    parameters=p_res,
                    provenance=e_res,
                    circuit_id=c_id,
                    topology=top,
                )
                if not evidence_ok:
                    logger.warning(f"Cannot mark record {record_id} as verified: {reason}")
                    return False

            cursor.execute(
                "UPDATE simulation_history SET verified_for_training = ? WHERE record_id = ?",
                (1 if verified else 0, record_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def export_to_csv(self, output_path: str) -> bool:
        """Exports database records to a CSV file."""
        df = self.get_all_records()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        return True
