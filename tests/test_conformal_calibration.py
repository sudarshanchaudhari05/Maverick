"""Unit and integration tests for Conformal Calibration & Risk-Controlled Abstention."""

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from src.api.app import app
from src.detection.predict import FraudDetector
from src.detection.conformal_calibration import (
    ConformalCalibrator,
    ConformalPredictionSet,
    get_default_calibration_dataset,
)
from src.detection.abstention_engine import (
    RiskControlledAbstentionEngine,
    SelectiveDecision,
    AbstentionEvaluationMetrics,
)
from src.utils.config import MODELS_DIR


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with application lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def baseline_detector():
    """Load baseline detector artifact."""
    baseline_path = MODELS_DIR / "baseline_detector.joblib"
    return FraudDetector(artifact_path=baseline_path)


@pytest.fixture(scope="module")
def sample_dataset():
    """Generate a consistent small dataset for calibration and testing."""
    return get_default_calibration_dataset(n_samples=300, seed=42)


def test_conformal_calibrator_quantile(baseline_detector, sample_dataset):
    """Verify split conformal calibration computes valid finite-sample quantile q_hat."""
    calibrator = ConformalCalibrator(detector=baseline_detector, default_alpha=0.05)
    assert not calibrator.is_calibrated

    cal_state = calibrator.calibrate(sample_dataset, target_alpha=0.05, detector=baseline_detector)
    assert calibrator.is_calibrated
    assert cal_state.is_calibrated
    assert 0.0 < cal_state.q_hat <= 1.0
    assert cal_state.n_calibration_samples == len(sample_dataset)
    assert "exchangeab" in cal_state.exchangeability_assumption.lower()


def test_conformal_prediction_sets(baseline_detector, sample_dataset):
    """Verify prediction sets {0}, {1}, {0, 1} and singleton/ambiguous flags."""
    calibrator = ConformalCalibrator(detector=baseline_detector, default_alpha=0.05)
    calibrator.calibrate(sample_dataset, target_alpha=0.05, detector=baseline_detector)

    # Low fraud probability -> should include 0
    res_legit = calibrator.predict_single({"fraud_probability": 0.01})
    assert 0 in res_legit.prediction_set
    assert res_legit.p_value_0 > 0.05

    # High fraud probability -> should include 1
    res_fraud = calibrator.predict_single({"fraud_probability": 0.99})
    assert 1 in res_fraud.prediction_set
    assert res_fraud.p_value_1 > 0.05

    # Ambiguous intermediate probability (e.g. 0.50) -> yields ambiguous or empty set (both require abstention)
    res_ambig = calibrator.predict_single({"fraud_probability": 0.50})
    assert isinstance(res_ambig.prediction_set, list)
    assert res_ambig.is_empty or res_ambig.is_ambiguous or len(res_ambig.prediction_set) in [0, 2]


def test_risk_controlled_threshold_calibration(baseline_detector, sample_dataset):
    """Verify acceptance threshold calibration bounds and sensitivity to target FNR."""
    calibrator = ConformalCalibrator(detector=baseline_detector, default_alpha=0.05)
    calibrator.calibrate(sample_dataset, target_alpha=0.05, detector=baseline_detector)

    engine = RiskControlledAbstentionEngine(calibrator=calibrator, default_target_fnr=0.05)
    tau_05 = engine.calibrate_acceptance_threshold(sample_dataset, target_fnr=0.05, detector=baseline_detector)
    tau_01 = engine.calibrate_acceptance_threshold(sample_dataset, target_fnr=0.01, detector=baseline_detector)

    assert 0.01 <= tau_01 <= 0.40
    assert 0.01 <= tau_05 <= 0.40
    # Conservative target FNR (1%) should result in lower or equal acceptance threshold
    assert tau_01 <= tau_05


def test_selective_decision_routing(baseline_detector, sample_dataset):
    """Verify decision routing: ALLOW/BLOCK are AUTOMATIC, ambiguous/marginal are HUMAN_REVIEW."""
    calibrator = ConformalCalibrator(detector=baseline_detector, default_alpha=0.05)
    calibrator.calibrate(sample_dataset, target_alpha=0.05, detector=baseline_detector)
    engine = RiskControlledAbstentionEngine(calibrator=calibrator, default_target_fnr=0.05)
    engine.calibrate_acceptance_threshold(sample_dataset, target_fnr=0.05, detector=baseline_detector)

    # 1. Very low risk transaction -> ALLOW, AUTOMATIC_DECISION
    dec_allow = engine.evaluate_transaction({"fraud_probability": 0.005}, detector=baseline_detector)
    assert dec_allow.decision == "ALLOW"
    assert dec_allow.routing == "AUTOMATIC_DECISION"
    assert not dec_allow.is_abstained

    # 2. Very high risk transaction -> BLOCK, AUTOMATIC_DECISION
    dec_block = engine.evaluate_transaction({"fraud_probability": 0.98}, detector=baseline_detector)
    assert dec_block.decision == "BLOCK"
    assert dec_block.routing == "AUTOMATIC_DECISION"
    assert not dec_block.is_abstained

    # 3. Intermediate/uncertain transaction -> ABSTAIN, HUMAN_REVIEW
    dec_abstain = engine.evaluate_transaction({"fraud_probability": 0.45}, detector=baseline_detector)
    assert dec_abstain.decision == "ABSTAIN"
    assert dec_abstain.routing == "HUMAN_REVIEW"
    assert dec_abstain.is_abstained


def test_dataset_evaluation_metrics(baseline_detector, sample_dataset):
    """Verify dataset-level evaluation metrics calculation under selective classification."""
    calibrator = ConformalCalibrator(detector=baseline_detector, default_alpha=0.05)
    calibrator.calibrate(sample_dataset, target_alpha=0.05, detector=baseline_detector)
    engine = RiskControlledAbstentionEngine(calibrator=calibrator, default_target_fnr=0.05)
    engine.calibrate_acceptance_threshold(sample_dataset, target_fnr=0.05, detector=baseline_detector)

    decisions, metrics = engine.evaluate_dataset(sample_dataset, detector=baseline_detector)
    assert len(decisions) == len(sample_dataset)
    assert isinstance(metrics, AbstentionEvaluationMetrics)
    assert metrics.total_evaluated == len(sample_dataset)
    assert metrics.automatic_decision_count + metrics.human_review_count == len(sample_dataset)
    assert pytest.approx(metrics.coverage + metrics.abstention_rate, 0.001) == 1.0
    assert 0.0 <= metrics.observed_fnr <= 1.0
    assert "exchangeab" in metrics.exchangeability_disclaimer.lower()


def test_api_conformal_calibrate_endpoint(client):
    """Verify POST /api/v1/conformal/calibrate endpoint."""
    payload = {
        "target_alpha": 0.05,
        "target_fnr": 0.05,
        "calibration_sample_size": 300,
    }
    response = client.post("/api/v1/conformal/calibrate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["is_calibrated"] is True
    assert data["target_alpha"] == 0.05
    assert data["target_fnr"] == 0.05
    assert data["q_hat"] > 0.0
    assert data["acceptance_threshold"] > 0.0
    assert "exchangeab" in data["exchangeability_disclaimer"].lower()


def test_api_conformal_predict_endpoint(client):
    """Verify POST /api/v1/conformal/predict endpoint for single and batch transactions."""
    payload = {
        "transaction": {
            "transaction_id": "TX-CONF-TEST-01",
            "amount": 1250.0,
            "payment_channel": "pos_chip",
            "velocity_1h": 1,
            "velocity_24h": 2,
        },
        "target_fnr": 0.05,
        "target_alpha": 0.05,
    }
    response = client.post("/api/v1/conformal/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_evaluated"] == 1
    assert len(data["results"]) == 1
    r = data["results"][0]
    assert r["transaction_id"] == "TX-CONF-TEST-01"
    assert r["decision"] in ["ALLOW", "BLOCK", "ABSTAIN"]
    assert r["routing"] in ["AUTOMATIC_DECISION", "HUMAN_REVIEW"]
    assert isinstance(r["prediction_set"], list)
    assert 0.0 <= r["p_value_0"] <= 1.0
    assert 0.0 <= r["p_value_1"] <= 1.0


def test_api_conformal_metrics_endpoint(client):
    """Verify GET /api/v1/conformal/metrics endpoint."""
    response = client.get("/api/v1/conformal/metrics?target_fnr=0.05")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["target_fnr"] == 0.05
    assert "coverage" in data
    assert "abstention_rate" in data
    assert "false_negatives" in data
    assert "exchangeab" in data["exchangeability_disclaimer"].lower()
