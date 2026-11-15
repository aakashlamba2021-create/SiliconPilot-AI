"""
SiliconPilot AI — Baseline Model Training & Evaluation Script
Trains:
  1. Layout Geometry Predictor (ICLR-ADAPT 35D actions -> Aeval, Leq_signal, Leq_power, Ncut_intermetal) [3,760 Real Samples]
  2. OSIRIS Area & PEX Score Predictor (OSIRIS variants -> Area, PEX Low, Mid, High) [89,108 Real Samples]
  3. Electrical QoS Predictor (OSIRIS sizing & placement -> Gain, GBW, PM, Power, Slew Rate, CMRR)
  4. Layout Verification Classifier (DRC clean pass/fail)
Saves models to registry and outputs evaluation reports.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODEL_DIR, REPORTS_DIR
from src.data.iclr_loader import ICLRAdaptLoader
from src.data.osiris_loader import OsirisLoader
from src.data.preprocessing import DataPreprocessor
from src.evaluation.model_evaluator import ModelEvaluator
from src.models.layout_predictor import LayoutPredictor
from src.models.model_registry import ModelRegistry
from src.models.performance_predictor import PerformancePredictor
from src.models.quality_classifier import QualityClassifier
from src.utils.logger import get_logger

logger = get_logger("train_pipeline")


def train_all_models():
    registry = ModelRegistry(registry_dir=MODEL_DIR)
    eval_summary = {}

    print("\n" + "=" * 70)
    print("  SILICONPILOT AI — MACHINE LEARNING MODEL TRAINING PIPELINE")
    print("=" * 70 + "\n")

    # -----------------------------------------------------------------------
    # MODEL 1: Layout Physical Geometry Predictor (ICLR-ADAPT Real 3,760 obs)
    # -----------------------------------------------------------------------
    print("[1/4] Loading ICLR-ADAPT Dataset & Training Layout Geometry Predictor (3,760 Real Obs)...")
    iclr_loader = ICLRAdaptLoader()
    df_iclr = iclr_loader.load()
    X_iclr, y_iclr, groups_iclr = iclr_loader.get_features_and_targets(target_type="geometry")

    preprocessor_iclr = DataPreprocessor()
    X_tr_iclr, X_te_iclr, y_tr_iclr, y_te_iclr = preprocessor_iclr.split_data(
        X_iclr, y_iclr, groups=groups_iclr, test_size=0.2, random_state=42
    )

    layout_pred = LayoutPredictor(algorithm="rf", n_estimators=100)
    iclr_metrics = layout_pred.train_and_evaluate(X_tr_iclr, X_te_iclr, y_tr_iclr, y_te_iclr)
    layout_path = layout_pred.save(registry, model_name="iclr_layout_geometry_predictor")
    eval_summary["iclr_layout_geometry_predictor"] = {
        "model_file": layout_path,
        "dataset": "ICLR-ADAPT (Official 3,760 Observations)",
        "train_samples": len(X_tr_iclr),
        "test_samples": len(X_te_iclr),
        "metrics": iclr_metrics,
    }
    print(f"  -> Model 1 saved. Mean Test MAE: {iclr_metrics['overall'].get('mean_MAE', 0):.2f}")

    # -----------------------------------------------------------------------
    # MODEL 2: OSIRIS Area & PEX Score Predictor (Official Parquet 89,108 obs)
    # -----------------------------------------------------------------------
    print("\n[2/4] Loading OSIRIS Official Parquet & Training Area/PEX Predictor (89,108 Real Variants)...")
    osiris_loader = OsirisLoader(sample_mode=False)
    df_osiris_pq = osiris_loader.load(source="parquet")
    X_pq, y_pq, groups_pq = osiris_loader.get_features_and_targets(task="area")

    preprocessor_pq = DataPreprocessor()
    X_tr_pq, X_te_pq, y_tr_pq, y_te_pq = preprocessor_pq.split_data(
        X_pq, y_pq, groups=groups_pq, test_size=0.2, random_state=42
    )

    rf_pq = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    rf_pq.fit(preprocessor_pq.fit_transform(X_tr_pq), y_tr_pq)
    y_pred_pq = rf_pq.predict(preprocessor_pq.transform(X_te_pq))
    osiris_pq_metrics = ModelEvaluator.evaluate_regression(y_te_pq, y_pred_pq, target_names=list(y_pq.columns))

    pq_path = registry.save_model(
        model_name="osiris_area_pex_predictor",
        model_obj=rf_pq,
        preprocessor=preprocessor_pq,
        feature_names=list(X_pq.columns),
        target_names=list(y_pq.columns),
        metrics=osiris_pq_metrics,
        dataset_name="OSIRIS Official Parquet (89,108 Variants)",
        description="Physical Area & Multi-Interval Parasitic Extraction (PEX) Score Predictor",
    )
    eval_summary["osiris_area_pex_predictor"] = {
        "model_file": str(pq_path),
        "dataset": "OSIRIS Official Parquet (89,108 Variants)",
        "train_samples": len(X_tr_pq),
        "test_samples": len(X_te_pq),
        "metrics": osiris_pq_metrics,
    }
    print(f"  -> Model 2 saved. Area R²: {osiris_pq_metrics['per_target']['area']['R2']:.4f}, PEX Mid R²: {osiris_pq_metrics['per_target']['pex_score_mid']['R2']:.4f}")

    # -----------------------------------------------------------------------
    # MODEL 3: Parasitic-Aware Electrical Performance Predictor (OSIRIS Sizing & QoS)
    # -----------------------------------------------------------------------
    print("\n[3/4] Loading OSIRIS QoS Benchmark & Training Electrical Performance Predictor...")
    df_osiris = osiris_loader.load(source="sample")
    X_osiris, y_osiris, groups_osiris = osiris_loader.get_features_and_targets(task="performance")

    preprocessor_osiris = DataPreprocessor()
    X_tr_os, X_te_os, y_tr_os, y_te_os = preprocessor_osiris.split_data(
        X_osiris, y_osiris, groups=groups_osiris, test_size=0.2, random_state=42
    )

    perf_pred = PerformancePredictor(algorithm="et", n_estimators=120)
    osiris_metrics = perf_pred.train_and_evaluate(X_tr_os, X_te_os, y_tr_os, y_te_os)
    perf_path = perf_pred.save(registry, model_name="osiris_electrical_performance_predictor")
    eval_summary["osiris_electrical_performance_predictor"] = {
        "model_file": perf_path,
        "dataset": "OSIRIS Sizing & QoS Benchmark",
        "train_samples": len(X_tr_os),
        "test_samples": len(X_te_os),
        "metrics": osiris_metrics,
    }
    print(f"  -> Model 3 saved. Test Mean R²: {osiris_metrics['overall'].get('mean_R2', 0):.4f} (DC Gain R²: {osiris_metrics['per_target']['dc_gain']['R2']:.4f})")

    # -----------------------------------------------------------------------
    # MODEL 4: Layout Verification / DRC Clean Classifier
    # -----------------------------------------------------------------------
    print("\n[4/4] Training Layout Verification (DRC Clean) Classifier...")
    X_qual, y_qual, groups_qual = osiris_loader.get_features_and_targets(task="quality")
    preprocessor_qual = DataPreprocessor()
    X_tr_q, X_te_q, y_tr_q, y_te_q = preprocessor_qual.split_data(
        X_qual, y_qual, groups=groups_qual, test_size=0.2, random_state=42
    )

    qual_clf = QualityClassifier(algorithm="rf", n_estimators=100)
    qual_metrics = qual_clf.train_and_evaluate(X_tr_q, X_te_q, y_tr_q, y_te_q)
    qual_path = qual_clf.save(registry, model_name="layout_verification_classifier")
    eval_summary["layout_verification_classifier"] = {
        "model_file": qual_path,
        "dataset": "OSIRIS Layout Verification",
        "train_samples": len(X_tr_q),
        "test_samples": len(X_te_q),
        "metrics": qual_metrics,
    }
    print(f"  -> Model 4 saved. Test Accuracy: {qual_metrics['Accuracy']*100:.2f}%, F1: {qual_metrics['F1']:.4f}")

    # -----------------------------------------------------------------------
    # Save Full Evaluation Report
    # -----------------------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_json = REPORTS_DIR / "model_eval_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    report_md = REPORTS_DIR / "model_eval_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# SiliconPilot AI — ML Model Training & Evaluation Report\n\n")
        f.write(f"Generated at: {json.dumps(eval_summary.get('timestamp', ''))}\n\n")
        for m_name, data in eval_summary.items():
            f.write(f"## {m_name.replace('_', ' ').title()}\n")
            f.write(f"- **Dataset:** {data['dataset']}\n")
            f.write(f"- **Train / Test Split:** {data['train_samples']} train / {data['test_samples']} test\n")
            f.write("### Metrics:\n")
            f.write(f"```json\n{json.dumps(data['metrics'], indent=2)}\n```\n\n")

    print("\n" + "=" * 70)
    print(f"  All 4 models successfully trained and registered in {MODEL_DIR}")
    print(f"  Evaluation reports written to {report_json} and {report_md}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    train_all_models()
