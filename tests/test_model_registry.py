"""Tests for Volatile / Session-Only ModelRegistry and Dynamic In-Memory Model Promotion."""

import shutil
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from src.detection.model_registry import ModelRegistry
from src.detection.predict import FraudDetector
from src.attacks.attack_genome import AttackGenome, KNOWN_ATTACK_GENOMES
from src.attacks.attack_library import get_default_attack_library
from src.simulation.transaction_generator import TransactionGenerator
from src.api.app import app
from src.utils.config import MODELS_DIR


@pytest.fixture
def temp_models_dir(tmp_path):
    """Fixture providing an isolated temporary models directory."""
    base_src = MODELS_DIR / "baseline_detector.joblib"
    base_dst = tmp_path / "baseline_detector.joblib"
    if base_src.exists():
        shutil.copyfile(base_src, base_dst)
    return tmp_path


def test_registry_initialization(temp_models_dir):
    """Test that model registry initializes default baseline_v1 state in memory without disk artifacts."""
    registry = ModelRegistry(models_dir=temp_models_dir)
    assert registry.get_active_version() == "baseline_v1"
    assert registry.get_hardening_round() == 0
    
    # Verify no persistent files created on disk
    assert not (temp_models_dir / "active_detector.joblib").exists()
    assert not (temp_models_dir / "model_registry.json").exists()

    data = registry.get_registry_data()
    assert data["active_version"] == "baseline_v1"
    assert data["hardening_round"] == 0
    assert len(data["history"]) == 1
    assert data["history"][0]["version"] == "baseline_v1"


def test_active_and_baseline_detectors_loadable(temp_models_dir):
    """Test loading active and immutable baseline detectors."""
    registry = ModelRegistry(models_dir=temp_models_dir)
    detector_base = registry.load_baseline_detector()
    detector_active = registry.load_active_detector()

    assert isinstance(detector_base, FraudDetector)
    assert isinstance(detector_active, FraudDetector)

    # Test scoring on sample transaction
    gen = TransactionGenerator(seed=42)
    df = gen.generate_dataset(n_samples=10, fraud_ratio=0.5)
    probs_base = detector_base.predict_proba(df)
    probs_act = detector_active.predict_proba(df)

    assert len(probs_base) == 10
    assert np.allclose(probs_base, probs_act)


def test_model_promotion_session_only_no_disk_artifacts(temp_models_dir):
    """Test candidate validation and promotion increments version in memory with zero disk persistence."""
    registry = ModelRegistry(models_dir=temp_models_dir)
    baseline_bytes_before = (temp_models_dir / "baseline_detector.joblib").read_bytes()

    # Load baseline artifact to create a candidate detector
    detector_base = registry.load_baseline_detector()
    candidate_artifact = detector_base.artifact.copy()

    # Promote candidate
    new_version, active_det = registry.promote_candidate(
        candidate_artifact=candidate_artifact,
        trained_attack_ids=["ATK-012"],
        validation_metrics={"gain": 50.0},
    )

    assert new_version == "hardened_v1"
    assert registry.get_active_version() == "hardened_v1"
    assert registry.get_hardening_round() == 1
    assert isinstance(active_det, FraudDetector)

    # Verify baseline model file is untouched
    baseline_bytes_after = (temp_models_dir / "baseline_detector.joblib").read_bytes()
    assert baseline_bytes_before == baseline_bytes_after

    # Verify NO hardened files were written to disk
    assert not (temp_models_dir / "hardened_detector_v1.joblib").exists()
    assert not (temp_models_dir / "active_detector.joblib").exists()
    assert not (temp_models_dir / "model_registry.json").exists()

    # Promote second candidate
    new_version_2, _ = registry.promote_candidate(
        candidate_artifact=candidate_artifact,
        trained_attack_ids=["ATK-001"],
        validation_metrics={"gain": 60.0},
    )
    assert new_version_2 == "hardened_v2"
    assert registry.get_active_version() == "hardened_v2"
    assert registry.get_hardening_round() == 2
    assert not (temp_models_dir / "hardened_detector_v2.joblib").exists()


def test_registry_reset_to_baseline(temp_models_dir):
    """Test resetting active model back to baseline_v1 in memory."""
    registry = ModelRegistry(models_dir=temp_models_dir)
    detector_base = registry.load_baseline_detector()

    registry.promote_candidate(
        candidate_artifact=detector_base.artifact.copy(),
        trained_attack_ids=["ATK-012"],
    )
    assert registry.get_active_version() == "hardened_v1"

    res = registry.reset_to_baseline()
    assert res == "baseline_v1"
    assert registry.get_active_version() == "baseline_v1"
    assert registry.get_hardening_round() == 0


def test_backend_startup_and_restart_always_initializes_baseline():
    """Verify that a newly started backend instance always initializes baseline_v1 and round 0."""
    # Simulate a new backend session by creating a fresh TestClient
    with TestClient(app) as client:
        health_res = client.get("/api/v1/health")
        assert health_res.status_code == 200
        health_data = health_res.json()
        assert health_data["active_model_version"] == "baseline_v1"
        assert health_data["hardening_round"] == 0

        # Verify no volatile hardened artifacts exist in MODELS_DIR
        assert not (MODELS_DIR / "active_detector.joblib").exists()
        assert not (MODELS_DIR / "model_registry.json").exists()
        assert len(list(MODELS_DIR.glob("hardened_detector_v*.joblib"))) == 0


def test_frontend_does_not_use_browser_storage_for_models():
    """Verify that frontend/index.html does not store model state in localStorage or sessionStorage."""
    frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    assert frontend_path.exists()
    content = frontend_path.read_text(encoding="utf-8")
    assert "localStorage" not in content
    assert "sessionStorage" not in content
    assert "indexedDB" not in content


def test_end_to_end_closed_loop_feedback_via_api():
    """Test full API loop: Attack Lab (ATK-012) -> In-Memory Hardening -> Retest in Attack Lab."""
    with TestClient(app) as client:
        # 1. Reset active model to baseline for clean test state
        reset_res = client.post("/api/v1/models/reset-active")
        assert reset_res.status_code == 200
        assert reset_res.json()["active_version"] == "baseline_v1"

        # 2. Evaluate ATK-012 in Attack Lab using initial active detector (baseline)
        atk_012_genome = KNOWN_ATTACK_GENOMES["ATK-012"].get_genes()
        eval_req_1 = {
            "candidate_genome": atk_012_genome,
            "candidate_name": "ATK-012: Stealth Biometric Injection",
            "sample_count": 100,
            "seed": 42,
            "detector_role": "active",
        }
        eval_res_1 = client.post("/api/v1/discovery/evaluate-candidate", json=eval_req_1)
        assert eval_res_1.status_code == 200
        data_1 = eval_res_1.json()
        assert data_1["model_version"] == "baseline_v1"
        assert data_1["detector_role"] == "active"
        baseline_recall = data_1["detection_rate_pct"]
        baseline_misses = data_1["missed_count"]
        # Baseline detector has low detection on ATK-012
        assert baseline_recall <= 60.0
        assert baseline_misses >= 40

        # 3. Send ATK-012 to Defense Lab and execute Adaptive Hardening Cycle (In-Memory)
        harden_req = {
            "attack_genome": atk_012_genome,
            "attack_name": "ATK-012: Stealth Biometric Injection",
            "target_attack_id": "ATK-012",
            "sample_count": 100,
            "seed": 42,
            "auto_promote": True,
        }
        harden_res = client.post("/api/v1/hardening/compare-defense", json=harden_req)
        assert harden_res.status_code == 200
        harden_data = harden_res.json()
        assert harden_data["promoted"] is True
        assert harden_data["validation_passed"] is True
        new_active_ver = harden_data["active_version"]
        assert new_active_ver.startswith("hardened_v")
        assert harden_data["hardened_detector"]["detection_rate_pct"] > baseline_recall

        # 4. Check Health & Model Registry reflect the newly promoted model in memory
        health_res = client.get("/api/v1/health")
        assert health_res.status_code == 200
        assert health_res.json()["active_model_version"] == new_active_ver

        reg_res = client.get("/api/v1/models/registry")
        assert reg_res.status_code == 200
        assert reg_res.json()["active_version"] == new_active_ver

        # 5. Return to Attack Lab and Re-Evaluate the EXACT SAME attack (ATK-012)
        eval_res_2 = client.post("/api/v1/discovery/evaluate-candidate", json=eval_req_1)
        assert eval_res_2.status_code == 200
        data_2 = eval_res_2.json()
        assert data_2["model_version"] == new_active_ver
        hardened_recall_attack_lab = data_2["detection_rate_pct"]
        hardened_misses_attack_lab = data_2["missed_count"]

        # Genuinely improved result from the newly promoted active in-memory model!
        assert hardened_recall_attack_lab >= 85.0
        assert hardened_misses_attack_lab < baseline_misses
        assert hardened_recall_attack_lab > baseline_recall

        # 6. Risk Decision Engine also uses the in-memory active model
        risk_req = {
            "transaction_amount": 750.0,
            "transaction_hour": 3,
            "account_age_days": 120,
            "device_age_days": 1,
            "device_change": 1,
            "IP_risk_score": 0.82,
            "merchant_risk_score": 0.65,
            "transaction_velocity_1h": 7,
            "transaction_velocity_24h": 15,
            "average_customer_amount": 65.0,
            "amount_deviation": 10.5,
            "geographic_deviation": 1,
            "behavioral_deviation": 0.72,
            "failed_authentication_count": 2,
            "identity_risk_score": 0.45,
            "merchant_category": "digital_goods",
            "payment_channel": "mobile_app",
            "authentication_method": "sms_otp",
            "transaction_country": "US",
            "customer_country": "US",
            "policy_mode": "BALANCED",
        }
        risk_res = client.post("/api/v1/risk/evaluate-transaction", json=risk_req)
        assert risk_res.status_code == 200
        risk_data = risk_res.json()
        assert risk_data["model_version"] == new_active_ver

        # 7. Verify no hardened files were written to disk
        assert not (MODELS_DIR / "active_detector.joblib").exists()
        assert not (MODELS_DIR / "model_registry.json").exists()
        assert len(list(MODELS_DIR.glob("hardened_detector_v*.joblib"))) == 0

        # Clean up by resetting to baseline
        client.post("/api/v1/models/reset-active")
