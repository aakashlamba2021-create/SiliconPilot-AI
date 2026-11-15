"""
Comprehensive Unit Tests — SiliconPilot AI Pipeline
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR, MODEL_DIR, RAW_DIR
from src.data.iclr_loader import ICLRAdaptLoader
from src.data.osiris_loader import OsirisLoader
from src.data.preprocessing import DataPreprocessor
from src.data.schema_inspector import SchemaInspector
from src.evaluation.model_evaluator import ModelEvaluator
from src.features.feature_engineering import AnalogFeatureEngineer
from src.models.layout_predictor import LayoutPredictor
from src.models.model_registry import ModelRegistry
from src.models.performance_predictor import PerformancePredictor
from src.models.quality_classifier import QualityClassifier
from src.optimization.candidate_optimizer import CandidateOptimizer
from dashboard.chat import SiliconPilotChatbot


def test_config():
    assert PROJECT_ROOT.exists()
    assert DATA_DIR.exists()
    assert RAW_DIR.exists()
    assert MODEL_DIR.exists()
    print("✅ test_config passed")


def test_schema_inspector():
    df = pd.DataFrame({
        "w": [1.0, 2.0, 3.0],
        "l": [0.18, 0.18, 0.5],
        "dc_gain": [50.0, 60.0, 70.0],
        "drc_clean": [True, True, False],
    })
    inspector = SchemaInspector(df, name="TestDF")
    report = inspector.run()
    assert report["shape"]["rows"] == 3
    assert report["shape"]["columns"] == 4
    assert report["data_quality_score"] > 80
    assert "dc_gain" in report["feature_target_candidates"]["suggested_targets"]
    print("✅ test_schema_inspector passed")


def test_feature_engineering():
    df = pd.DataFrame({
        "transistor_width_W": [10.0, 20.0],
        "transistor_length_L": [0.5, 1.0],
        "finger_count": [2, 4],
        "multiplicity": [1, 2],
        "delta_x": [1.0, 3.0],
        "delta_y": [1.0, 4.0],
    })
    engineered = AnalogFeatureEngineer.engineer_osiris_features(df)
    assert "aspect_ratio_W_L" in engineered.columns
    assert "total_active_gate_area" in engineered.columns
    assert "perturbation_displacement" in engineered.columns
    assert engineered["perturbation_displacement"].iloc[1] == 5.0
    print("✅ test_feature_engineering passed")


def test_data_preprocessor_and_split():
    df_X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "f2": [10, 20, 30, 40, 50, 60]})
    df_y = pd.Series([0, 1, 0, 1, 0, 1])
    groups = pd.Series(["A", "A", "B", "B", "C", "C"])

    prep = DataPreprocessor()
    X_tr, X_te, y_tr, y_te = prep.split_data(df_X, df_y, groups=groups, test_size=0.33)
    assert len(X_tr) > 0 and len(X_te) > 0
    # Verify group separation
    tr_groups = groups.iloc[X_tr.index]
    te_groups = groups.iloc[X_te.index]
    assert set(tr_groups).isdisjoint(set(te_groups))
    print("✅ test_data_preprocessor_and_split passed")


def test_model_registry_and_evaluator():
    reg = ModelRegistry(registry_dir=MODEL_DIR)
    models = reg.list_models()
    assert len(models) >= 3
    assert "osiris_electrical_performance_predictor" in models
    assert "layout_verification_classifier" in models

    loaded = reg.load_model_for_permitted_use(
        "osiris_electrical_performance_predictor", "exploratory_candidate_generation"
    )
    assert loaded is not None
    assert "model" in loaded
    assert "preprocessor" in loaded
    print("✅ test_model_registry_and_evaluator passed")


def test_candidate_optimizer():
    df = pd.DataFrame({
        "dc_gain": [45.0, 65.0, 75.0],
        "phase_margin": [40.0, 65.0, 55.0],
        "area": [2000.0, 1500.0, 3000.0],
        "area_evidence_origin": ["parameter_proxy"] * 3,
        "electrical_evidence_origin": ["simulated"] * 3,
        "power_consumption": [0.001, 0.0005, 0.0008],
        "drc_clean": [True, True, False],
    })
    opt = CandidateOptimizer(min_phase_margin=50.0, min_dc_gain=50.0)
    ranked = opt.rank_candidates(df)
    assert "siliconpilot_score" in ranked.columns
    assert "is_clean_candidate" in ranked.columns
    # DRC labels are not verification evidence and do not affect feasibility.
    assert ranked["is_clean_candidate"].sum() == 2
    assert "DRC Violation detected" not in " ".join(ranked["constraint_status"])
    print("✅ test_candidate_optimizer passed")


def test_ai_assistant():
    bot = SiliconPilotChatbot()
    r1 = bot.respond("How do I layout a differential pair?")
    assert "Common-Centroid" in r1
    r2 = bot.respond("What models are currently trained?")
    assert "Trained Model Registry Status" in r2
    print("✅ test_ai_assistant passed")


if __name__ == "__main__":
    print("\n--- Running SiliconPilot AI Test Suite ---")
    test_config()
    test_schema_inspector()
    test_feature_engineering()
    test_data_preprocessor_and_split()
    test_model_registry_and_evaluator()
    test_candidate_optimizer()
    test_ai_assistant()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!\n")

