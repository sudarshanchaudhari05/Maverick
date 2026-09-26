"""Unit tests for ConformalTestMartingale drift detector (Phase E.1)."""

import pytest
import numpy as np
from src.detection.drift_martingale import ConformalTestMartingale


def test_martingale_initialization():
    """Verify initialization and parameter storage."""
    ref_scores = np.random.RandomState(42).uniform(0, 1, 500)
    martingale = ConformalTestMartingale(
        reference_scores=ref_scores,
        epsilon=0.80,
        alpha_drift=0.01,
    )
    assert martingale.alarm_threshold == 100.0
    assert martingale.epsilon == 0.80
    assert martingale.log_evidence == 0.0
    assert martingale.evidence == 1.0
    assert len(martingale.ref_scores) == 500


def test_conformal_p_value_calculation():
    """Verify non-parametric p-value formulation."""
    ref_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    martingale = ConformalTestMartingale(
        reference_scores=ref_scores,
        alpha_drift=0.05,
    )
    # Score 0.6 is greater than all 5 reference scores -> (0 + 1) / (5 + 1) = 1/6
    p_high = martingale.compute_p_value(0.6)
    assert abs(p_high - (1.0 / 6.0)) < 1e-5

    # Score 0.0 is less than all 5 reference scores -> (5 + 1) / (5 + 1) = 1.0
    p_low = martingale.compute_p_value(0.0)
    assert abs(p_low - 1.0) < 1e-5


def test_supermartingale_property_under_null():
    """Under H0 (exchangeable data), M_t is a non-negative supermartingale with no false alarms."""
    rng = np.random.RandomState(123)
    ref_scores = rng.normal(loc=0.0, scale=1.0, size=1000)
    test_scores = rng.normal(loc=0.0, scale=1.0, size=500)

    martingale = ConformalTestMartingale(
        reference_scores=ref_scores,
        epsilon=0.80,
        alpha_drift=0.01,
    )

    evidences = []
    alarms = []
    for s in test_scores:
        step = martingale.update_observation(float(s))
        evidences.append(step["evidence"])
        if step["is_alarm"]:
            alarms.append(step)

    assert len(alarms) == 0
    assert np.mean(evidences) < 10.0


def test_martingale_alarm_and_reset_under_drift():
    """Under strong shift, evidence crosses threshold tau=100 and can be reset."""
    rng = np.random.RandomState(456)
    # Reference: mostly low scores
    ref_scores = rng.exponential(scale=0.05, size=1000)

    martingale = ConformalTestMartingale(
        reference_scores=ref_scores,
        epsilon=0.80,
        alpha_drift=0.01,
    )

    # Shift: high scores
    drift_scores = rng.uniform(0.8, 1.0, size=100)

    alarm_triggered = False
    for s in drift_scores:
        step = martingale.update_observation(float(s))
        if step["is_alarm"]:
            alarm_triggered = True
            break

    assert alarm_triggered is True
    assert martingale.is_alarm is True

    # Test reset
    martingale.reset()
    assert martingale.log_evidence == 0.0
    assert martingale.evidence == 1.0
    assert martingale.is_alarm is False
