"""Integration unit tests for FraudForge AI FastAPI REST API."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifecycle startup triggered."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify GET /api/v1/health returns healthy status and model statuses."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "healthy"]
    assert "baseline_model" in data
    assert "hardened_model" in data
    assert data["active_test_count"] >= 64


def test_get_known_attacks_catalog(client):
    """Verify GET /api/v1/catalog/known-attacks returns all 28 archetypes."""
    response = client.get("/api/v1/catalog/known-attacks")
    assert response.status_code == 200
    attacks = response.json()
    assert len(attacks) == 28
    first = attacks[0]
    assert "attack_id" in first
    assert "genome" in first
    assert len(first["genome"]) == 10


def test_get_genome_vocabulary(client):
    """Verify GET /api/v1/catalog/genome-vocabulary returns all 10 dimensions."""
    response = client.get("/api/v1/catalog/genome-vocabulary")
    assert response.status_code == 200
    data = response.json()
    assert len(data["dimensions"]) == 10
    assert len(data["vocabulary"]) == 10
    assert "target" in data["vocabulary"]
    assert "entry_vector" in data["vocabulary"]


def test_generate_candidate_mutation_and_crossover(client):
    """Verify POST /api/v1/discovery/generate-candidate via mutation and crossover."""
    # 1. Mutation
    mut_payload = {
        "generation_method": "mutation",
        "parent_attack_id": "ATK-001",
        "n_mutations": 1,
        "seed": 42,
    }
    res_mut = client.post("/api/v1/discovery/generate-candidate", json=mut_payload)
    assert res_mut.status_code == 200
    data_mut = res_mut.json()
    assert "candidate_id" in data_mut
    assert "scores" in data_mut
    assert 0.0 <= data_mut["scores"]["novelty_score"] <= 1.0
    assert 0.0 <= data_mut["scores"]["realism_score"] <= 1.0
    assert 0.0 <= data_mut["scores"]["evasion_potential"] <= 1.0

    # 2. Crossover
    cross_payload = {
        "generation_method": "crossover",
        "parent_1_id": "ATK-001",
        "parent_2_id": "ATK-021",
        "seed": 101,
    }
    res_cross = client.post("/api/v1/discovery/generate-candidate", json=cross_payload)
    assert res_cross.status_code == 200
    data_cross = res_cross.json()
    assert data_cross["lineage"]["mutation_type"] == "crossover"


def test_evaluate_candidate_against_baseline(client):
    """Verify POST /api/v1/discovery/evaluate-candidate simulates and detects attacks."""
    sample_genome = {
        "target": "customer",
        "entry_vector": "social_engineering",
        "behavior": "low_and_slow",
        "evasion_strategy": "trusted_device_masking",
        "payment_channel": "p2p_transfer",
        "amount_strategy": "sub_threshold_structuring",
        "temporal_strategy": "distributed",
        "identity_strategy": "authorized_victim",
        "merchant_strategy": "single_target",
        "geographic_strategy": "domestic_matching",
    }
    payload = {
        "candidate_genome": sample_genome,
        "candidate_name": "Test Social Engineering Candidate",
        "sample_count": 20,
        "seed": 42,
    }
    response = client.post("/api/v1/discovery/evaluate-candidate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_tested"] == 20
    assert data["detected_count"] + data["missed_count"] == 20
    assert 0.0 <= data["detection_rate_pct"] <= 100.0
    assert data["blind_spot_level"] in [
        "CRITICAL BLIND SPOT",
        "MODERATE BLIND SPOT",
        "CONTROLLED SCENARIO",
    ]
    assert len(data["sample_transactions"]) > 0


def test_evolve_gen2_variant(client):
    """Verify POST /api/v1/hardening/evolve-gen2 produces an evolved Gen-2 variant."""
    parent_genome = {
        "target": "merchant",
        "entry_vector": "credential_stuffing",
        "behavior": "burst_spike",
        "evasion_strategy": "trusted_session_behavior",
        "payment_channel": "e-commerce",
        "amount_strategy": "sub_threshold_structuring",
        "temporal_strategy": "distributed",
        "identity_strategy": "synthetic_sleeper",
        "merchant_strategy": "single_target",
        "geographic_strategy": "domestic_matching",
    }
    payload = {
        "parent_genome": parent_genome,
        "candidate_id": "NSA-TEST",
        "seed": 2026,
    }
    response = client.post("/api/v1/hardening/evolve-gen2", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["variant_id"] == "NSA-TEST-V2"
    assert "(Gen-2 Evolved)" in data["variant_name"]
    assert len(data["evolved_genes"]) > 0


def test_compare_defense_baseline_vs_hardened(client):
    """Verify POST /api/v1/hardening/compare-defense returns side-by-side stats."""
    test_genome = {
        "target": "customer",
        "entry_vector": "social_engineering",
        "behavior": "low_and_slow",
        "evasion_strategy": "trusted_device_masking",
        "payment_channel": "p2p_transfer",
        "amount_strategy": "stealth_discounted",
        "temporal_strategy": "off_hours_window",
        "identity_strategy": "authorized_victim",
        "merchant_strategy": "single_target",
        "geographic_strategy": "domestic_matching",
    }
    payload = {
        "attack_genome": test_genome,
        "attack_name": "Test Gen-2 Scenario",
        "sample_count": 20,
        "seed": 2026,
    }
    response = client.post("/api/v1/hardening/compare-defense", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "baseline_detector" in data
    assert "hardened_detector" in data
    assert "defense_improvement" in data
    assert data["total_simulated"] == 20


def test_evaluate_transaction_risk_engine(client):
    """Verify POST /api/v1/risk/evaluate-transaction scores transaction through the full pipeline."""
    # 1. Suspicious / Fraudulent Transaction
    fraud_tx = {
        "transaction_id": "TX-SUSPICIOUS-01",
        "transaction_amount": 950.0,
        "transaction_hour": 3,
        "account_age_days": 10,
        "device_age_days": 1,
        "device_change": 1,
        "IP_risk_score": 0.88,
        "merchant_risk_score": 0.75,
        "transaction_velocity_1h": 8,
        "transaction_velocity_24h": 16,
        "average_customer_amount": 40.0,
        "amount_deviation": 20.0,
        "geographic_deviation": 1,
        "behavioral_deviation": 0.85,
        "failed_authentication_count": 3,
        "identity_risk_score": 0.80,
        "merchant_category": "digital_goods",
        "payment_channel": "mobile_app",
        "authentication_method": "sms_otp",
        "transaction_country": "US",
        "customer_country": "US",
        "policy_mode": "BALANCED",
    }
    res_fraud = client.post("/api/v1/risk/evaluate-transaction", json=fraud_tx)
    assert res_fraud.status_code == 200
    data_f = res_fraud.json()
    assert data_f["transaction_id"] == "TX-SUSPICIOUS-01"
    assert data_f["risk_level"] in ["HIGH", "CRITICAL"]
    assert data_f["action"] in ["STEP_UP_AUTH", "BLOCK"]
    assert len(data_f["reason_codes"]) > 0
    assert "mitigation" in data_f

    # 2. Benign / Legitimate Transaction
    benign_tx = {
        "transaction_id": "TX-BENIGN-01",
        "transaction_amount": 35.0,
        "transaction_hour": 14,
        "account_age_days": 300,
        "device_age_days": 150,
        "device_change": 0,
        "IP_risk_score": 0.05,
        "merchant_risk_score": 0.10,
        "transaction_velocity_1h": 1,
        "transaction_velocity_24h": 2,
        "average_customer_amount": 40.0,
        "amount_deviation": 0.05,
        "geographic_deviation": 0,
        "behavioral_deviation": 0.02,
        "failed_authentication_count": 0,
        "identity_risk_score": 0.05,
        "merchant_category": "groceries",
        "payment_channel": "pos_chip",
        "authentication_method": "pin",
        "transaction_country": "US",
        "customer_country": "US",
        "policy_mode": "BALANCED",
    }
    res_benign = client.post("/api/v1/risk/evaluate-transaction", json=benign_tx)
    assert res_benign.status_code == 200
    data_b = res_benign.json()
    assert data_b["transaction_id"] == "TX-BENIGN-01"
    assert data_b["risk_level"] == "LOW"
    assert data_b["action"] == "ALLOW"
    assert data_b["mitigation"]["status"] == "APPROVED"


def test_provenance_endpoints(client):
    """Verify GET /api/v1/provenance/metrics and POST /api/v1/provenance/extract-features."""
    res_metrics = client.get("/api/v1/provenance/metrics")
    assert res_metrics.status_code == 200
    data_m = res_metrics.json()
    assert "experiment_name" in data_m
    assert "model_a_results_d3" in data_m
    assert "model_b_results_d3" in data_m

    res_extract = client.post("/api/v1/provenance/extract-features", json={
        "provenance_chain": [
            "https://store.merchanta.com/product/123",
            "https://malicious-context.top/prompt?override=true",
        ],
        "primary_merchant_domain": "merchanta.com",
    })
    assert res_extract.status_code == 200
    data_e = res_extract.json()
    assert "features" in data_e
    assert data_e["feature_count"] == 10
    assert data_e["features"]["instruction_source_present"] == 1.0
    assert data_e["features"]["suspicious_source_indicator"] == 1.0

