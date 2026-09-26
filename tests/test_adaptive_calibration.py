"""Unit tests for DelayedLabelStreamBuffer and AdaptiveCalibrationManager (Phase E.1)."""

import pytest
import numpy as np
import pandas as pd
from src.detection.adaptive_calibration import (
    DelayedLabelStreamBuffer,
    AdaptiveCalibrationManager,
)
from src.detection.conformal_calibration import ConformalCalibrator
from src.detection.predict import FraudDetector
from src.utils.config import MODELS_DIR


def _make_dummy_batch(batch_id: int, size: int = 100) -> pd.DataFrame:
    """Helper to generate dummy transaction DataFrame."""
    rng = np.random.RandomState(batch_id)
    return pd.DataFrame({
        "transaction_id": [f"tx_{batch_id}_{i}" for i in range(size)],
        "transaction_amount": rng.uniform(10, 500, size),
        "transaction_hour": rng.randint(0, 24, size),
        "account_age_days": rng.uniform(1, 1000, size),
        "device_age_days": rng.uniform(1, 500, size),
        "device_change": rng.choice([0, 1], size),
        "IP_risk_score": rng.uniform(0, 1, size),
        "merchant_risk_score": rng.uniform(0, 1, size),
        "transaction_velocity_1h": rng.randint(1, 10, size),
        "transaction_velocity_24h": rng.randint(1, 20, size),
        "average_customer_amount": [100.0] * size,
        "amount_deviation": [0.0] * size,
        "geographic_deviation": [0.0] * size,
        "behavioral_deviation": [0.0] * size,
        "failed_authentication_count": [0] * size,
        "identity_risk_score": [0.1] * size,
        "merchant_category": ["retail"] * size,
        "payment_channel": ["web"] * size,
        "device_type": ["mobile"] * size,
        "location_country": ["US"] * size,
        "card_type": ["visa"] * size,
        "fraud_label": rng.choice([0, 1], p=[0.95, 0.05], size=size),
    })


def test_buffer_label_isolation_and_delay():
    """Verify ground truth is strictly withheld until delay batches elapse."""
    buffer = DelayedLabelStreamBuffer(label_delay_batches=3)

    # Batch 1 arrives
    df_b1 = _make_dummy_batch(1, 50)
    buffer.add_batch(1, df_b1)

    # At batch 1, 2, 3: eligible count is 0
    assert buffer.get_eligible_count(current_batch_id=1) == 0
    assert buffer.get_eligible_count(current_batch_id=2) == 0
    assert buffer.get_eligible_count(current_batch_id=3) == 0

    # Add Batch 2 and 3
    buffer.add_batch(2, _make_dummy_batch(2, 50))
    buffer.add_batch(3, _make_dummy_batch(3, 50))
    assert buffer.get_eligible_count(current_batch_id=3) == 0

    # Batch 4 arrives -> Batch 1 becomes eligible (4 - 3 = 1)
    buffer.add_batch(4, _make_dummy_batch(4, 50))
    assert buffer.get_eligible_count(current_batch_id=4) == 50

    eligible_df = buffer.get_eligible_labeled_samples(current_batch_id=4)
    assert len(eligible_df) == 50
    assert set(eligible_df["_stream_batch_id"].unique()) == {1}


def test_adaptive_calibration_manager_window_enforcement():
    """Recalibration is refused until exactly recalibration_window eligible samples exist."""
    calibrator = ConformalCalibrator(default_alpha=0.05)
    detector = FraudDetector().load(MODELS_DIR / "baseline_detector.joblib")
    manager = AdaptiveCalibrationManager(
        initial_calibrator=calibrator,
        target_fnr=0.05,
        recalibration_window=500,
    )

    buffer = DelayedLabelStreamBuffer(label_delay_batches=3)
    for b in range(1, 5):
        buffer.add_batch(b, _make_dummy_batch(b, 100))

    # At Batch 4: eligible count is 100 (< 500)
    outcome = manager.trigger_recalibration(
        current_batch_id=4,
        stream_buffer=buffer,
        detector=detector,
        reason="martingale_alarm",
    )

    assert outcome["status"] == "STALE_PENDING_LABELS"
    assert manager.is_stale is True
    assert manager.pending_recalibration is True


def test_adaptive_calibration_manager_successful_adaptation():
    """Recalibration succeeds when >= 500 eligible samples exist, executing adaptation."""
    calibrator = ConformalCalibrator(default_alpha=0.05)
    detector = FraudDetector().load(MODELS_DIR / "baseline_detector.joblib")
    manager = AdaptiveCalibrationManager(
        initial_calibrator=calibrator,
        target_fnr=0.05,
        recalibration_window=500,
    )

    buffer = DelayedLabelStreamBuffer(label_delay_batches=3)
    # Add 8 batches of 100 samples -> at batch 8, batches 1..5 eligible = 500 samples
    for b in range(1, 9):
        buffer.add_batch(b, _make_dummy_batch(b, 100))

    outcome = manager.trigger_recalibration(
        current_batch_id=8,
        stream_buffer=buffer,
        detector=detector,
        reason="martingale_alarm",
    )

    assert outcome["status"] == "RECALIBRATED"
    assert outcome["new_version"] == "v1"
    assert manager.version_index == 1
    assert manager.calibration_version == "v1"
    assert manager.is_stale is False
