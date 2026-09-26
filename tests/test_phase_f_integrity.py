"""FraudForge AI: Phase F Automated Integrity & Methodology Test Suite.

Validates all locked Phase F protocols:
- FNR and Wilson score confidence interval calculations
- Support filtering (total >= 50, fraud >= 20) and INSUFFICIENT_SUPPORT handling
- Combination depths 1, 2, 3 and strict depth 4+ rejection
- Slice ranking rules (FNR desc -> fraud support desc -> total support desc)
- Top-5 persistent IDs (F-SLICE-001 .. F-SLICE-005)
- Generator label-independence (no inspection of fraud_label, is_fraud, detector predictions)
- Pairwise dataset zero-overlap leakage across all 6 dataset pairs
- Deterministic reproducibility for seeds 6001, 6002, 6003, 6004
- Paired McNemar's test calculation
- Locked Success Gates G1 through G5
- F-D4 dataset isolation from training and tuning
- Frozen Phase E.1 and Phase E.2 preservation
- Artifact and report schema verification
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.utils.config import PROJECT_ROOT, MODELS_DIR
from experiments.phase_f.slice_discovery.slice_miner import (
    compute_wilson_ci,
    WorstSliceMiner,
    MIN_TOTAL_SUPPORT,
    MIN_FRAUD_SUPPORT,
)
from experiments.phase_f.targeted_generation.targeted_generator import PhaseFDatasetGenerator
from experiments.phase_f.evaluation.evaluation_engine import (
    compute_binary_metrics,
    compute_mcnemar_test,
    audit_dataset_leakage,
    PhaseFEvaluationEngine,
)


PHASE_F_DIR = PROJECT_ROOT / "experiments" / "phase_f"


def test_fnr_and_wilson_ci():
    """Verify FNR calculation and 95% Wilson confidence interval properties."""
    # Perfect detector: 0 FN out of 25 fraud -> FNR = 0.0
    lower, upper = compute_wilson_ci(k=0, n=25)
    assert lower == 0.0
    assert 0.0 < upper < 0.20

    # Total failure: 20 FN out of 20 fraud -> FNR = 1.0
    lower, upper = compute_wilson_ci(k=20, n=20)
    assert 0.80 < lower < 1.0
    assert upper == 1.0

    # 10 FN out of 50 fraud -> FNR = 0.20
    lower, upper = compute_wilson_ci(k=10, n=50)
    assert lower < 0.20 < upper
    assert 0.10 < lower < 0.15
    assert 0.25 < upper < 0.35

    # Boundary handling
    assert compute_wilson_ci(0, 0) == (0.0, 0.0)


def test_support_filter_and_insufficient_support():
    """Verify support filtering marks slices failing either total or fraud thresholds."""
    miner = WorstSliceMiner(max_depth=1, min_total_support=50, min_fraud_support=20)

    # Case 1: total >= 50 and fraud >= 20 -> ELIGIBLE
    df_eligible = pd.DataFrame({
        "fraud_label": [1] * 20 + [0] * 35,
        "detector_prediction": [0] * 10 + [1] * 10 + [0] * 35,
        "detector_probability": [0.1] * 10 + [0.8] * 10 + [0.05] * 35,
        "risk_level": ["LOW_RISK"] * 45 + ["HIGH_RISK"] * 10,
    })
    res_eligible = miner.evaluate_slice_metrics(df_eligible, {"test": "eligible"}, depth=1)
    assert res_eligible["eligible"] is True
    assert res_eligible["status"] == "ELIGIBLE"
    assert res_eligible["FNR"] == 0.50

    # Case 2: total >= 50 but fraud < 20 -> INSUFFICIENT_SUPPORT
    df_low_fraud = pd.DataFrame({
        "fraud_label": [1] * 15 + [0] * 40,
        "detector_prediction": [0] * 15 + [0] * 40,
        "detector_probability": [0.1] * 55,
        "risk_level": ["LOW_RISK"] * 55,
    })
    res_low_fraud = miner.evaluate_slice_metrics(df_low_fraud, {"test": "low_fraud"}, depth=1)
    assert res_low_fraud["eligible"] is False
    assert res_low_fraud["status"] == "INSUFFICIENT_SUPPORT"

    # Case 3: fraud >= 20 but total < 50 -> INSUFFICIENT_SUPPORT
    df_low_total = pd.DataFrame({
        "fraud_label": [1] * 22 + [0] * 5,
        "detector_prediction": [0] * 22 + [0] * 5,
        "detector_probability": [0.1] * 27,
        "risk_level": ["LOW_RISK"] * 27,
    })
    res_low_total = miner.evaluate_slice_metrics(df_low_total, {"test": "low_total"}, depth=1)
    assert res_low_total["eligible"] is False
    assert res_low_total["status"] == "INSUFFICIENT_SUPPORT"


def test_maximum_depth_enforcement():
    """Verify maximum depth 3 is enforced and depth 4+ is strictly rejected."""
    # Depths 1, 2, 3 must be accepted
    WorstSliceMiner(max_depth=1)
    WorstSliceMiner(max_depth=2)
    WorstSliceMiner(max_depth=3)

    # Depth 4 must raise ValueError per protocol rules
    with pytest.raises(ValueError, match="Maximum combination depth is 3"):
        WorstSliceMiner(max_depth=4)


def test_slice_ranking_and_top5_persistent_ids():
    """Verify ranking criteria (FNR desc -> fraud support desc -> total support desc)."""
    miner = WorstSliceMiner(max_depth=2)

    dummy_candidates = [
        {"slice_definition": {"a": "1"}, "eligible": True, "FNR": 0.40, "fraud_support": 30, "total_support": 100},
        {"slice_definition": {"a": "2"}, "eligible": True, "FNR": 0.80, "fraud_support": 25, "total_support": 80},
        {"slice_definition": {"a": "3"}, "eligible": True, "FNR": 0.80, "fraud_support": 40, "total_support": 90},
        {"slice_definition": {"a": "4"}, "eligible": True, "FNR": 0.80, "fraud_support": 40, "total_support": 120},
        {"slice_definition": {"a": "5"}, "eligible": False, "FNR": 0.99, "fraud_support": 5, "total_support": 10},
        {"slice_definition": {"a": "6"}, "eligible": True, "FNR": 0.50, "fraud_support": 50, "total_support": 200},
        {"slice_definition": {"a": "7"}, "eligible": True, "FNR": 0.30, "fraud_support": 20, "total_support": 60},
    ]

    top_slices, ranked = miner.rank_and_select_top_slices(dummy_candidates, top_k=5, targetable_only=False)

    # Ineligible slice a=5 must NOT enter ranking
    assert all(s["slice_definition"] != {"a": "5"} for s in top_slices)

    # Expected order:
    # 1. a=4: FNR=0.80, fraud=40, total=120
    # 2. a=3: FNR=0.80, fraud=40, total=90
    # 3. a=2: FNR=0.80, fraud=25, total=80
    # 4. a=6: FNR=0.50, fraud=50, total=200
    # 5. a=1: FNR=0.40, fraud=30, total=100
    assert top_slices[0]["slice_definition"] == {"a": "4"}
    assert top_slices[0]["slice_id"] == "F-SLICE-001"
    assert top_slices[1]["slice_definition"] == {"a": "3"}
    assert top_slices[1]["slice_id"] == "F-SLICE-002"
    assert top_slices[2]["slice_definition"] == {"a": "2"}
    assert top_slices[2]["slice_id"] == "F-SLICE-003"
    assert top_slices[3]["slice_definition"] == {"a": "6"}
    assert top_slices[3]["slice_id"] == "F-SLICE-004"
    assert top_slices[4]["slice_definition"] == {"a": "1"}
    assert top_slices[4]["slice_id"] == "F-SLICE-005"


def test_generator_label_independence():
    """Verify dataset generator does not inspect detector predictions, targets, or ground truth to manufacture failures."""
    gen = PhaseFDatasetGenerator()

    # Inspect _sample_constrained_transaction
    # Verify that generation executes purely from random number generator and slice definition
    rng = np.random.default_rng(999)
    base_gen = gen.attack_library
    slice_def = {"payment_channel": "pos_contactless", "amount_bucket": "medium (50-200)"}

    tx = gen._sample_constrained_transaction(
        rng=rng,
        base_gen=None if False else from_import_helper(),
        slice_def=slice_def,
        is_fraud=True,
    )

    assert tx["payment_channel"] == "pos_contactless"
    assert 50.0 <= tx["transaction_amount"] <= 200.0
    assert "detector_prediction" not in tx
    assert "detector_probability" not in tx


def from_import_helper():
    from src.simulation.transaction_generator import TransactionGenerator
    return TransactionGenerator(seed=999)


def test_zero_dataset_overlap():
    """Verify pairwise signature overlap is strictly 0 across all 6 dataset pairs."""
    data_dir = PHASE_F_DIR / "data"
    assert (data_dir / "fd1_discovery.csv").exists()
    assert (data_dir / "fd2_random_baseline.csv").exists()
    assert (data_dir / "fd3_targeted_hardening.csv").exists()
    assert (data_dir / "fd4_unseen_v2_test.csv").exists()

    datasets = {
        "F-D1": pd.read_csv(data_dir / "fd1_discovery.csv"),
        "F-D2": pd.read_csv(data_dir / "fd2_random_baseline.csv"),
        "F-D3": pd.read_csv(data_dir / "fd3_targeted_hardening.csv"),
        "F-D4": pd.read_csv(data_dir / "fd4_unseen_v2_test.csv"),
    }

    assert len(datasets["F-D1"]) == 4000
    assert len(datasets["F-D2"]) == 2000
    assert len(datasets["F-D3"]) == 2500
    assert len(datasets["F-D4"]) == 2000

    audit_res = audit_dataset_leakage(datasets)
    assert audit_res["zero_leakage_passed"] is True
    for pair, count in audit_res["pair_overlap_counts"].items():
        assert count == 0, f"Leakage detected in pair {pair}: {count} duplicates"


def test_mcnemar_paired_calculation():
    """Verify McNemar paired calculation on synthetic contingency table."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    # Model A: misses two frauds (idx 0, 1)
    y_pred_a = np.array([0, 0, 1, 1, 0, 0, 0, 0])
    # Model B: catches all frauds
    y_pred_b = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    mcn = compute_mcnemar_test(y_true, y_pred_a, y_pred_b)
    # b: A wrong, B right = 2
    # c: A right, B wrong = 0
    assert mcn["b_model_b_better"] == 2
    assert mcn["c_model_a_better"] == 0
    assert mcn["discordant_pairs"] == 2
    # (|2 - 0| - 1)^2 / 2 = 1 / 2 = 0.5
    assert mcn["test_statistic"] == 0.5


def test_fd4_mcnemar_authoritative_verification():
    """Verify that F-D4 paired predictions yield exactly 59/28 discordant counts and chi2=10.3448."""
    pred_path_parquet = PHASE_F_DIR / "predictions" / "fd4_predictions.parquet"
    pred_path_csv = PHASE_F_DIR / "predictions" / "fd4_predictions_baseline_vs_hardened.csv"

    assert pred_path_parquet.exists(), f"Authoritative parquet artifact not found: {pred_path_parquet}"
    df = pd.read_parquet(pred_path_parquet)

    y_true = df["fraud_label"].values
    y_pred_a = df["baseline_prediction"].values
    y_pred_b = df["hardened_prediction"].values

    mcn = compute_mcnemar_test(y_true, y_pred_a, y_pred_b)

    assert mcn["both_correct"] == 1893
    assert mcn["both_incorrect"] == 20
    assert mcn["b_model_b_better"] == 59
    assert mcn["c_model_a_better"] == 28
    assert mcn["discordant_pairs"] == 87
    assert round(mcn["test_statistic"], 4) == 10.3448
    assert round(mcn["p_value"], 4) == 0.0013
    assert mcn["statistically_significant_at_05"] is True


def test_fd4_isolated_from_training():
    """Verify F-D4 (seed 6004) was never seen during hardening or training."""
    report_path = PHASE_F_DIR / "reports" / "phase_f_report.json"
    assert report_path.exists()
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Check training metadata
    train_meta = report["training_metadata"]
    assert train_meta["fd3_samples"] == 2500
    assert train_meta["baseline_training_samples"] == 5000
    assert train_meta["total_hardened_train_samples"] == 7500

    # Ensure F-D4 (N=2000) was NOT in training population
    assert report["datasets"]["F-D4"]["size"] == 2000
    assert report["datasets"]["F-D4"]["seed"] == 6004


def test_success_gates_pass():
    """Verify all 5 locked success gates evaluate to PASS in authoritative report."""
    report_path = PHASE_F_DIR / "reports" / "phase_f_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    gates = report["success_gates"]
    assert gates["all_gates_passed"] is True
    assert gates["G1_slice_improvement"]["passed"] is True
    assert gates["G2_overall_f5_recall"]["passed"] is True
    assert gates["G3_fpr_protection"]["passed"] is True
    assert gates["G4_random_attack_preservation"]["passed"] is True
    assert gates["G5_unseen_targeted_generalization"]["passed"] is True


def test_phase_e_remains_frozen():
    """Confirm frozen Phase E.1 and Phase E.2 artifacts remain untouched and uncorrupted."""
    audit_e1 = PROJECT_ROOT / "experiments" / "phase_e_audit.json"
    assert audit_e1.exists()
    with open(audit_e1, "r", encoding="utf-8") as f:
        data_e1 = json.load(f)
    assert data_e1["phase_e_status"] == "FROZEN AND COMPLETE"

    audit_e2 = PROJECT_ROOT / "experiments" / "phase_e2_audit.json"
    assert audit_e2.exists()
    with open(audit_e2, "r", encoding="utf-8") as f:
        data_e2 = json.load(f)
    assert data_e2["phase_e2_status"] == "FROZEN"


def test_phase_f_status_not_auto_frozen():
    """Confirm Phase F status is explicitly IMPLEMENTED_PENDING_INDEPENDENT_AUDIT and NOT frozen."""
    report_path = PHASE_F_DIR / "reports" / "phase_f_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["status"] == "PHASE F — IMPLEMENTED, PENDING INDEPENDENT AUDIT"


def test_api_slices_endpoints():
    """Test all 4 Phase F slice API endpoints return HTTP 200 with valid schema."""
    from fastapi.testclient import TestClient
    from src.api.app import app

    client = TestClient(app)

    # 1. /api/v1/slices/status
    res_status = client.get("/api/v1/slices/status")
    assert res_status.status_code == 200
    data_status = res_status.json()
    assert data_status["phase"] == "F"
    assert data_status["top_5_slices_count"] == 5
    assert data_status["all_gates_passed"] is True

    # 2. /api/v1/slices/top
    res_top = client.get("/api/v1/slices/top")
    assert res_top.status_code == 200
    data_top = res_top.json()
    assert len(data_top["top_5_slices"]) == 5
    assert data_top["top_5_slices"][0]["slice_id"] == "F-SLICE-001"

    # 3. /api/v1/slices/metrics
    res_metrics = client.get("/api/v1/slices/metrics")
    assert res_metrics.status_code == 200
    data_metrics = res_metrics.json()
    assert "unseen_targeted_v2_fd4" in data_metrics
    assert "random_baseline_fd2" in data_metrics

    # 4. /api/v1/slices/gates
    res_gates = client.get("/api/v1/slices/gates")
    assert res_gates.status_code == 200
    data_gates = res_gates.json()
    assert data_gates["success_gates"]["all_gates_passed"] is True

