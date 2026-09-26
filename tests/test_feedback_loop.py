"""Unit tests for Adaptive Feedback Loop orchestrator."""

import pytest
import json
import pandas as pd
from pathlib import Path

from src.adversarial.feedback_loop import AdaptiveFeedbackLoop
from src.detection.predict import FraudDetector
from src.utils.config import MODELS_DIR


def test_feedback_loop_execution(tmp_path):
    """Verify end-to-end feedback loop runs and generates required artifacts."""
    baseline_path = MODELS_DIR / "baseline_detector.joblib"
    backup = baseline_path.read_bytes() if baseline_path.exists() else None
    try:
        loop = AdaptiveFeedbackLoop(
            baseline_seed=42,
            adversarial_train_seed=101,
            unseen_test_seed=1337,
            output_dir=tmp_path,
        )

        # Run quick cycle with 400 samples
        results = loop.run_cycle(n_samples=400, fraud_ratio=0.20, mutation_intensity=0.60)
    finally:
        if backup is not None:
            baseline_path.write_bytes(backup)

    assert "seeds" in results
    assert "normal_test_set_performance" in results
    assert "unseen_adversarial_test_performance" in results
    assert "adversarial_false_negatives" in results

    # Verify JSON artifact
    json_path = tmp_path / "baseline_vs_adversarial.json"
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json["seeds"]["baseline_dataset_a"] == 42

    # Verify CSV report
    csv_path = tmp_path / "adversarial_report.csv"
    assert csv_path.exists()
    df_report = pd.read_csv(csv_path)
    assert "detection_rate_delta" in df_report.columns

    # Verify Feature Importance CSV
    feat_path = tmp_path / "feature_importance.csv"
    assert feat_path.exists()

    # Verify Hardened Model exists and loads
    hardened_model_path = MODELS_DIR / "hardened_detector.joblib"
    assert hardened_model_path.exists()
    detector = FraudDetector(artifact_path=hardened_model_path)
    assert detector.model is not None
