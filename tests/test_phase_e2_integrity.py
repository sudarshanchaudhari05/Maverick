"""FraudForge AI: Comprehensive Unit & Integrity Tests for Phase E.2.

Verifies:
1. Generator label independence (strict isolation from labels and target metadata)
2. Canonical 20-feature transaction + 10-feature provenance schemas
3. Zero dataset overlap across all 5 generated datasets (E2-A to E2-E)
4. Presence and evaluation of Hard Negatives & Deceptive Positives
5. Metric correctness & conformal metric nomenclature (FOR vs FNR)
6. Phase E.2 API endpoints (/status, /metrics, /e2-metrics, /risk-coverage, /ablation, /judge-samples)
7. Seed reproducibility and artifact integrity
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.attacks.attack_discovery import AttackDiscoveryEngine
from src.simulation.transaction_generator import TransactionGenerator
from src.provenance.schema import ProvenanceChain, AgenticPaymentRecord, ProvenanceSource
from src.provenance.features import ProvenanceFeatureExtractor, PROVENANCE_FEATURE_NAMES
from src.provenance.generator import SyntheticProvenanceGenerator
from experiments.phase_e2_provenance_experiment import (
    generate_phase_e2_datasets,
    run_phase_e2_leakage_and_independence_audit,
    STRUCTURAL_PROVENANCE_FEATURES,
    wilson_score_interval,
    mcnemar_paired_test,
)
from src.utils.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, EXPERIMENTS_DIR
from src.api.app import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_generator_label_independence():
    """Verify that SyntheticProvenanceGenerator is strictly independent of labels."""
    gen = SyntheticProvenanceGenerator(seed=42)
    
    # 1. Inspect API signatures — generator methods do not take label or target
    import inspect
    sig = inspect.signature(gen.attach_provenance_to_dataset)
    params = list(sig.parameters.keys())
    assert "df" in params
    assert "mixture" in params
    assert "label" not in params
    assert "fraud_label" not in params
    assert "target" not in params

    # 2. Attach provenance to a dataframe with varying fraud labels
    df_dummy = pd.DataFrame({
        "transaction_amount": [10.0, 20.0, 30.0, 40.0],
        "fraud_label": [0, 1, 0, 1],
    })
    df_attached = gen.attach_provenance_to_dataset(df_dummy)
    assert "provenance_chain" in df_attached.columns
    assert "provenance_category" in df_attached.columns

    # 3. Prove identical provenance is generated regardless of inverted fraud labels
    gen_a = SyntheticProvenanceGenerator(seed=100)
    gen_b = SyntheticProvenanceGenerator(seed=100)

    df_a = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "fraud_label": [0, 0, 0]})
    df_b = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "fraud_label": [1, 1, 1]})

    out_a = gen_a.attach_provenance_to_dataset(df_a)
    out_b = gen_b.attach_provenance_to_dataset(df_b)

    assert out_a["provenance_chain"].tolist() == out_b["provenance_chain"].tolist()
    assert out_a["provenance_category"].tolist() == out_b["provenance_category"].tolist()


def test_feature_schemas_and_dimensions():
    """Verify canonical feature schemas and counts for Model A (20) and Model B (30)."""
    assert len(NUMERICAL_FEATURES) == 15
    assert len(CATEGORICAL_FEATURES) == 5
    txn_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    assert len(txn_features) == 20

    assert len(PROVENANCE_FEATURE_NAMES) == 10
    full_features = txn_features + PROVENANCE_FEATURE_NAMES
    assert len(full_features) == 30

    assert len(STRUCTURAL_PROVENANCE_FEATURES) == 5
    for feat in STRUCTURAL_PROVENANCE_FEATURES:
        assert feat in PROVENANCE_FEATURE_NAMES


def test_zero_dataset_overlap():
    """Verify that E2-A, E2-B, E2-C, E2-D, E2-E have exactly zero tuple overlap across 20 txn and 30 joint features."""
    datasets = generate_phase_e2_datasets()
    assert len(datasets) == 5
    assert len(datasets["E2-A"]) == 3000
    assert len(datasets["E2-B"]) == 1000
    assert len(datasets["E2-C"]) == 2000
    assert len(datasets["E2-D"]) == 1000
    assert len(datasets["E2-E"]) == 1000

    audit_result = run_phase_e2_leakage_and_independence_audit(datasets)
    assert audit_result["status"] == "PASS"

    # Level 1: Verify all 10 pairs have 0 overlap on 20 canonical transaction features
    overlaps_20 = audit_result["canonical_20_transaction_feature_overlap"]
    assert len(overlaps_20) == 10
    for pair, count in overlaps_20.items():
        assert count == 0, f"20-feature leakage detected between {pair}: count={count}"

    # Level 2: Verify all 10 pairs have 0 overlap on 30 joint features
    overlaps_30 = audit_result["complete_30_joint_feature_overlap"]
    assert len(overlaps_30) == 10
    for pair, count in overlaps_30.items():
        assert count == 0, f"30-feature leakage detected between {pair}: count={count}"

    # Level 3: Verify 10 standalone provenance feature overlaps are audited and explained
    overlaps_10 = audit_result["standalone_10_provenance_feature_overlap"]
    assert len(overlaps_10) == 10
    assert "shared_url_pool_explanation" in audit_result
    assert len(audit_result["shared_url_pool_explanation"]) > 20

    # Verify calibration / evaluation separation flag
    assert audit_result["calibration_evaluation_separation_verified"] is True


def test_conformal_calibration_evaluation_separation():
    """Verify that E2-B calibration and evaluation partitions are strictly disjoint."""
    datasets = generate_phase_e2_datasets()
    df_b = datasets["E2-B"]
    df_b_cal = df_b.iloc[:500]
    df_b_eval = df_b.iloc[500:]

    # Index separation
    cal_indices = set(df_b_cal.index)
    eval_indices = set(df_b_eval.index)
    assert len(cal_indices.intersection(eval_indices)) == 0

    # 20-feature transaction tuple separation
    txn_cols = list(NUMERICAL_FEATURES) + list(CATEGORICAL_FEATURES)
    cal_txns = set(tuple(r) for r in df_b_cal[txn_cols].to_numpy())
    eval_txns = set(tuple(r) for r in df_b_eval[txn_cols].to_numpy())
    assert len(cal_txns.intersection(eval_txns)) == 0

    # Report verification
    report_path = EXPERIMENTS_DIR / "phase_e2_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    e2b_conf = report["primary_results"]["E2-B_InDistribution"]["conformal_selective_eval"]
    assert e2b_conf.get("disjoint_separation_verified") is True
    assert "E2-B_cal" in e2b_conf.get("calibration_partition", "")
    assert "E2-B_eval" in e2b_conf.get("evaluation_partition", "")


def test_hard_negatives_and_deceptive_fraud():
    """Verify the presence of Hard Negatives and Deceptive Frauds in evaluation sets."""
    report_path = EXPERIMENTS_DIR / "phase_e2_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # In E2-B
    slices_b = report["primary_results"]["E2-B_InDistribution"]["four_way_slices"]
    assert "1_legit_benign" in slices_b
    assert "2_legit_suspicious_or_poisoned_hard_negative" in slices_b
    assert "3_fraud_benign_deceptive" in slices_b
    assert "4_fraud_suspicious_or_poisoned" in slices_b

    # Ensure hard negatives exist (legitimate transactions with unusual/poisoned provenance)
    hard_negs = slices_b["2_legit_suspicious_or_poisoned_hard_negative"]
    assert hard_negs["sample_count"] > 0
    assert "model_b_fpr" in hard_negs

    # Ensure deceptive frauds exist (fraudulent transactions with clean/benign provenance)
    deceptive = slices_b["3_fraud_benign_deceptive"]
    assert deceptive["sample_count"] > 0
    assert "model_b_recall" in deceptive


def test_conformal_metric_nomenclature():
    """Verify strict mathematical nomenclature: FOR vs Selective FNR vs Observed FNR."""
    report_path = EXPERIMENTS_DIR / "phase_e2_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    conf_eval = report["primary_results"]["E2-B_InDistribution"]["conformal_selective_eval"]["model_b"]
    for row in conf_eval:
        assert "false_omission_rate" in row
        assert "selective_fnr" in row
        assert "observed_fnr" in row
        assert "coverage" in row
        assert "abstention_rate" in row

        # Mathematical constraints:
        # FOR = FN / Auto-Accept Count
        if row["auto_accept_count"] > 0:
            calc_for = round(row["false_negatives"] / row["auto_accept_count"], 4)
            assert abs(row["false_omission_rate"] - calc_for) <= 0.001

        # Coverage + Abstention Rate = 1.0 (approx due to rounding)
        assert abs((row["coverage"] + row["abstention_rate"]) - 1.0) <= 0.01


def test_statistical_test_utilities():
    """Verify Wilson score CI and McNemar paired test calculations."""
    # Wilson interval
    ci = wilson_score_interval(10, 100, confidence=0.95)
    assert ci["proportion"] == 0.10
    assert 0.05 < ci["ci_lower"] < 0.10
    assert 0.10 < ci["ci_upper"] < 0.20

    # McNemar test
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred_a = np.array([1, 0, 0, 0, 1, 0, 0, 0])
    y_pred_b = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    res = mcnemar_paired_test(y_true, y_pred_a, y_pred_b)
    assert "mcnemar_statistic" in res
    assert "p_value" in res
    assert res["table"]["model_b_only"] >= 1


def test_api_provenance_endpoints(client):
    """Verify all Phase E.2 API endpoints return HTTP 200 and conform to schema."""
    # 1. Status
    res = client.get("/api/v1/provenance/status")
    assert res.status_code == 200
    status_data = res.json()
    assert status_data["phase"] == "E.2"
    assert status_data["status"] == "FROZEN"
    assert status_data["audit_passed"] is True
    assert status_data["label_independence_passed"] is True

    # 2. Metrics (Phase D backward compatibility)
    res_d = client.get("/api/v1/provenance/metrics")
    assert res_d.status_code == 200
    assert "experiment_name" in res_d.json()

    # 3. Metrics (Phase E.2 query parameter)
    res_e2_q = client.get("/api/v1/provenance/metrics?phase=e2")
    assert res_e2_q.status_code == 200
    assert res_e2_q.json()["phase"] == "E.2"

    # 4. E2-Metrics endpoint
    res_e2 = client.get("/api/v1/provenance/e2-metrics")
    assert res_e2.status_code == 200
    data_e2 = res_e2.json()
    assert data_e2["phase"] == "E.2"
    assert "E2-B_InDistribution" in data_e2["primary_results"]
    assert "E2-C_DistributionShift" in data_e2["primary_results"]
    assert "E2-D_PoisoningStress" in data_e2["primary_results"]

    # 5. Risk-Coverage curves
    res_rc = client.get("/api/v1/provenance/risk-coverage")
    assert res_rc.status_code == 200
    data_rc = res_rc.json()
    assert "E2-B" in data_rc["curves"]
    assert "Model_A_TxOnly" in data_rc["curves"]["E2-B"]
    assert "Model_B_TxProv" in data_rc["curves"]["E2-B"]

    # 6. Ablation records
    res_abl = client.get("/api/v1/provenance/ablation")
    assert res_abl.status_code == 200
    assert len(res_abl.json()["ablation_records"]) >= 4

    # 7. Judge Samples
    res_j = client.get("/api/v1/provenance/judge-samples")
    assert res_j.status_code == 200
    data_j = res_j.json()
    assert data_j["case_count"] == 4
    assert len(data_j["samples"]) == 4
    for sample in data_j["samples"]:
        assert "transaction" in sample
        assert "provenance" in sample
        assert "evaluation" in sample

    # 8. Feature Extraction
    res_ext = client.post("/api/v1/provenance/extract-features", json={
        "provenance_chain": [
            "https://store.merchanta.com/item/100",
            "https://malicious-context-relay.top/task?override_instruction=pay"
        ],
        "primary_merchant_domain": "merchanta.com"
    })
    assert res_ext.status_code == 200
    assert res_ext.json()["features"]["instruction_source_present"] == 1.0


def test_scientific_wording_and_claim_discipline():
    """Verify strictly required scientific wording and discipline in report disclaimers."""
    report_path = EXPERIMENTS_DIR / "phase_e2_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()
        report = json.loads(report_text)

    # 1. Exact required phrasing for poisoning significance
    disclaimers = report["scientific_disclaimers"]
    assert "The observed poisoning-scenario differences did not reach statistical significance at α=0.05 (McNemar p=0.1306)." in disclaimers["statistical_significance"]

    # 2. Exact required phrasing for poisoning slice claim discipline
    assert "On the tested poisoning slice, Model B produced the measured recall shown in the experiment; the paired difference was not statistically significant." in disclaimers["poisoning_claim_discipline"]

    # 3. Disallowed overclaims
    assert "within statistical noise" not in report_text.lower()
    assert "statistically meaningful improvement" not in report_text.lower()
    assert "positive signal" not in report_text.lower()
    assert "localized resilience" not in report_text.lower()
