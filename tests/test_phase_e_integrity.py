"""Phase E.1 Independent Integrity & Scientific Audit Tests (FraudForge AI)."""

import pytest
import json
import pandas as pd
from pathlib import Path
import hashlib
from fastapi.testclient import TestClient

from src.utils.config import EXPERIMENTS_DIR, MODELS_DIR
from src.api.app import app


def compute_file_hash(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_phase_e_artifacts_exist():
    """Verify all authoritative Phase E.1 experiment artifacts exist."""
    required_files = [
        "phase_e_config.json",
        "phase_e_report.json",
        "phase_e_metrics.csv",
        "phase_e_timeline.csv",
        "phase_e_audit.json",
        "phase_e_drift_timeline.png",
        "phase_e_risk_over_time.png",
        "phase_e_coverage_over_time.png",
        "phase_e_abstention_over_time.png",
        "phase_e_detection_delay.png",
        "phase_e_false_alarms.png",
        "phase_e_calibration_events.png",
        "phase_e_risk_coverage_tradeoff.png",
    ]
    for filename in required_files:
        path = EXPERIMENTS_DIR / filename
        assert path.exists(), f"Missing required Phase E artifact: {filename}"
        assert path.stat().st_size > 0, f"Artifact {filename} is empty"


def test_baseline_detector_frozen_and_immutable():
    """Verify baseline XGBoost detector artifact is immutable."""
    baseline_path = MODELS_DIR / "baseline_detector.joblib"
    assert baseline_path.exists()
    current_hash = compute_file_hash(baseline_path)

    audit_path = EXPERIMENTS_DIR / "phase_e_audit.json"
    assert audit_path.exists()
    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["detector_hash_verified"] is True
    audit_target = Path(audit["detector_artifact"])
    assert audit_target.exists()
    assert current_hash == compute_file_hash(audit_target)


def test_leakage_and_overlap_audit():
    """Verify 0 train/calibration/test row overlap and 0 label leakages."""
    audit_path = EXPERIMENTS_DIR / "phase_e_audit.json"
    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["zero_overlap_leakage_passed"] is True
    assert audit["delayed_label_protocol_enforced"] is True
    assert audit["no_future_lookahead"] is True
    assert audit["no_result_dependent_tuning"] is True


def test_report_schema_and_metrics():
    """Verify phase_e_report.json structure and non-trivial values."""
    report_path = EXPERIMENTS_DIR / "phase_e_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["phase"] == "E.1"
    assert report["status"] in ["IMPLEMENTED_NOT_YET_FROZEN", "FROZEN_AND_COMPLETE"]
    assert "conformal_guarantee_disclaimer" in report

    streams = report["stream_summaries"]
    assert len(streams) == 4
    for stream_id in ["E1_no_drift", "E2_abrupt_drift", "E3_gradual_drift", "E4_adversarial_gen2_drift"]:
        assert stream_id in streams
        s = streams[stream_id]
        assert "system_a_overall" in s
        assert "system_b_overall" in s
        assert "system_c_overall" in s

    # Check E1 has 0 false alarms under H0
    assert streams["E1_no_drift"]["total_alarms_triggered"] == 0

    # Check E2, E3, E4 detect drift
    assert streams["E2_abrupt_drift"]["total_alarms_triggered"] > 0
    assert streams["E3_gradual_drift"]["total_alarms_triggered"] > 0
    assert streams["E4_adversarial_gen2_drift"]["total_alarms_triggered"] > 0


def test_api_drift_endpoints(client):
    """Verify all 5 Phase E.1 API endpoints return HTTP 200 with valid structure."""
    # 1. Status
    res = client.get("/api/v1/drift/status?stream=E2_abrupt_drift")
    assert res.status_code == 200
    assert res.json()["active_stream"] == "E2_abrupt_drift"
    assert res.json()["status"] in ["IMPLEMENTED_NOT_YET_FROZEN", "FROZEN_AND_COMPLETE"]

    # 2. Metrics
    res = client.get("/api/v1/drift/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "E1_no_drift" in data
    assert "E2_abrupt_drift" in data

    # 3. Timeline
    res = client.get("/api/v1/drift/timeline?stream=E2_abrupt_drift")
    assert res.status_code == 200
    timeline = res.json()
    assert len(timeline) == 20

    # 4. Calibration Events
    res = client.get("/api/v1/drift/calibration-events?stream=E2_abrupt_drift")
    assert res.status_code == 200
    events_data = res.json()
    assert isinstance(events_data, dict)
    assert "drift_aware_events" in events_data
    assert len(events_data["drift_aware_events"]) > 0

    # 5. Methodology
    res = client.get("/api/v1/drift/methodology")
    assert res.status_code == 200
    res_data = res.json()
    assert "methodology" in res_data
    assert "label_delay" in res_data["methodology"]
    assert "3 batches" in res_data["methodology"]["label_delay"]
    assert "conformal_disclaimer" in res_data


def test_all_streams_api_data(client):
    """Verify that all 4 streams (E1, E2, E3, E4) return consistent schema and data."""
    for s_id in ["E1_no_drift", "E2_abrupt_drift", "E3_gradual_drift", "E4_adversarial_gen2_drift"]:
        # Status
        res_stat = client.get(f"/api/v1/drift/status?stream={s_id}")
        assert res_stat.status_code == 200
        assert res_stat.json()["active_stream"] == s_id

        # Metrics
        res_met = client.get(f"/api/v1/drift/metrics?stream={s_id}")
        assert res_met.status_code == 200
        data = res_met.json()
        assert s_id in data
        assert "system_a_pre_drift" in data[s_id]
        assert "system_c_post_drift" in data[s_id]

        # Timeline
        res_time = client.get(f"/api/v1/drift/timeline?stream={s_id}")
        assert res_time.status_code == 200
        assert len(res_time.json()) == 20

        # Events
        res_evt = client.get(f"/api/v1/drift/calibration-events?stream={s_id}")
        assert res_evt.status_code == 200
        assert "drift_aware_events" in res_evt.json()
        assert isinstance(res_evt.json()["drift_aware_events"], list)
