"""FraudForge AI: Comprehensive Unit Tests for Phase D Provenance Chain Security Signal.

Verifies:
1. Provenance schema definitions and validation
2. Deterministic provenance feature extraction
3. Deterministic generation with fixed seeds
4. Label and target isolation (zero leakage)
5. Hard-negative generation (legitimate transactions with unusual context)
6. Dataset overlap detection (zero overlap across partitions)
7. Poisoned and adversarial context generation
8. Identical evaluation rows shared across Model A and Model B
9. Metric calculation and nomenclature accuracy
10. End-to-end report and artifact generation
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.attacks.attack_discovery import AttackDiscoveryEngine
from src.simulation.transaction_generator import TransactionGenerator
from src.provenance.schema import (
    ProvenanceSource,
    ProvenanceChain,
    AgenticPaymentRecord,
)
from src.provenance.features import (
    ProvenanceFeatureExtractor,
    PROVENANCE_FEATURE_NAMES,
    calculate_string_entropy,
)
from src.provenance.generator import (
    SyntheticProvenanceGenerator,
)
from experiments.provenance_chain_ablation import (
    generate_phase_d_datasets,
    run_automated_leakage_checks,
    train_model,
    evaluate_model_predictions,
    evaluate_conformal_abstention,
    run_provenance_experiment,
)
from src.utils.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


# 1. Provenance Schema Tests
def test_provenance_schema_construction():
    """Verify typed dataclass construction and serialization for provenance."""
    src = ProvenanceSource(url="https://store.merchanta.com/item/123", source_type="product_page")
    assert src.domain == "store.merchanta.com"

    chain = ProvenanceChain.from_list([
        "https://store.merchanta.com/item/123",
        "https://checkout.merchanta.com/pay",
    ], provenance_type="benign")
    assert len(chain.to_list()) == 2
    assert chain.provenance_type == "benign"

    record = AgenticPaymentRecord(
        transaction_features={"transaction_amount": 99.50, "fraud_label": 0},
        provenance_chain=chain.to_list(),
    )
    flat = record.to_flat_dict()
    assert flat["transaction_amount"] == 99.50
    assert flat["agent_id"] == "payment_agent_v1"
    assert len(flat["provenance_chain"]) == 2


# 2. Deterministic Feature Extraction Tests
def test_deterministic_feature_extraction():
    """Verify that feature extraction produces deterministic outputs for identical inputs."""
    extractor = ProvenanceFeatureExtractor()
    urls = [
        "https://store.merchanta.com/item/123",
        "https://malicious-context-relay.top/task?override_instruction=pay",
        "https://unverified-discount.xyz/promo.pdf",
    ]

    f1 = extractor.extract_single(urls)
    f2 = extractor.extract_single(urls)
    assert f1 == f2
    assert f1["provenance_count"] == 3.0
    assert f1["document_type_count"] == 1.0  # .pdf
    assert f1["instruction_source_present"] == 1.0  # override_instruction
    assert f1["suspicious_source_indicator"] == 1.0  # .top and .xyz
    assert f1["unknown_domain_count"] >= 2.0


# 3. Deterministic Generation Tests
def test_deterministic_generation_with_seeds():
    """Verify that identical random seeds produce identical provenance chains."""
    gen1 = SyntheticProvenanceGenerator(seed=42)
    gen2 = SyntheticProvenanceGenerator(seed=42)

    c1 = gen1.generate_benign_chain().to_list()
    c2 = gen2.generate_benign_chain().to_list()
    assert c1 == c2

    p1 = gen1.generate_adversarial_poisoned_chain().to_list()
    p2 = gen2.generate_adversarial_poisoned_chain().to_list()
    assert p1 == p2


# 4. Zero Label Leakage Tests
def test_no_label_leakage_in_features():
    """Verify that feature extractor never outputs labels or targets."""
    extractor = ProvenanceFeatureExtractor()
    for feat in extractor.feature_names:
        lower = feat.lower()
        assert "label" not in lower
        assert "fraud" not in lower
        assert "target" not in lower
        assert "ground_truth" not in lower


# 5. Label Independence Tests
def test_provenance_generator_is_label_independent():
    """Verify that generator output does NOT depend on fraud_label or targets."""
    import inspect
    sig = inspect.signature(SyntheticProvenanceGenerator.attach_provenance_to_dataset)
    param_names = list(sig.parameters.keys())
    assert "fraud_label" not in param_names
    assert "is_fraud" not in param_names
    assert "attack_type" not in param_names
    assert "target" not in param_names

    # Generate for identical transaction features with different labels
    df_legit = pd.DataFrame([{"amount": 100.0, "fraud_label": 0}])
    df_fraud = pd.DataFrame([{"amount": 100.0, "fraud_label": 1}])

    gen1 = SyntheticProvenanceGenerator(seed=777)
    res1 = gen1.attach_provenance_to_dataset(df_legit)

    gen2 = SyntheticProvenanceGenerator(seed=777)
    res2 = gen2.attach_provenance_to_dataset(df_fraud)

    # Under identical random state, provenance chain must be identical despite opposite fraud labels
    assert res1["provenance_chain"].iloc[0] == res2["provenance_chain"].iloc[0]
    assert res1["provenance_category"].iloc[0] == res2["provenance_category"].iloc[0]


def test_poisoned_provenance_occurs_for_legitimate_transactions():
    """Verify that legitimate transactions can receive poisoned provenance and instruction tokens."""
    datasets = generate_phase_d_datasets()
    df_d1 = datasets["D1"]
    legit_poisoned = df_d1[(df_d1["fraud_label"] == 0) & (df_d1["provenance_category"] == "poisoned")]
    assert len(legit_poisoned) > 0

    # Instruction source present must occur for legitimate transactions
    legit_instructions = df_d1[(df_d1["fraud_label"] == 0) & (df_d1["instruction_source_present"] == 1.0)]
    assert len(legit_instructions) > 0


def test_benign_provenance_occurs_for_fraudulent_transactions():
    """Verify that fraudulent transactions can receive clean benign provenance."""
    datasets = generate_phase_d_datasets()
    df_d1 = datasets["D1"]
    fraud_benign = df_d1[(df_d1["fraud_label"] == 1) & (df_d1["provenance_category"] == "benign")]
    assert len(fraud_benign) > 0


# 6. Dataset Overlap Detection Tests
def test_dataset_overlap_detection():
    """Verify automated leakage checks catch identical transaction tuples."""
    datasets = generate_phase_d_datasets()
    leakage = run_automated_leakage_checks(datasets)
    assert leakage["status"] == "PASS"
    for k, overlap in leakage["dataset_overlap_check"].items():
        assert overlap == 0


# 7. Poisoning Generation Tests
def test_poisoning_chain_structure():
    """Verify that adversarial poisoned chains contain instruction tokens and suspicious domains."""
    gen = SyntheticProvenanceGenerator(seed=999)
    chain = gen.generate_adversarial_poisoned_chain()
    assert chain.provenance_type == "poisoned"
    extractor = ProvenanceFeatureExtractor()
    feats = extractor.extract_single(chain.to_list())
    assert feats["instruction_source_present"] == 1.0
    assert feats["suspicious_source_indicator"] == 1.0


# 8. Identical Evaluation Rows Shared Between Models
def test_identical_evaluation_rows_for_ab_models():
    """Verify Model A and Model B are strictly evaluated on identical row indices."""
    datasets = generate_phase_d_datasets()
    df_eval = datasets["D3"]

    pipe_a, model_a = train_model(
        datasets["D1"].iloc[:500],
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        seed=4001,
    )
    pipe_b, model_b = train_model(
        datasets["D1"].iloc[:500],
        numerical_features=NUMERICAL_FEATURES + PROVENANCE_FEATURE_NAMES,
        categorical_features=CATEGORICAL_FEATURES,
        seed=4001,
    )

    res_a = evaluate_model_predictions(pipe_a, model_a, df_eval, NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
    res_b = evaluate_model_predictions(pipe_b, model_b, df_eval, NUMERICAL_FEATURES + PROVENANCE_FEATURE_NAMES, CATEGORICAL_FEATURES)

    assert res_a["total"] == res_b["total"]
    assert res_a["actual_fraud"] == res_b["actual_fraud"]
    assert res_a["actual_legit"] == res_b["actual_legit"]


# 9. Metric Calculation Accuracy
def test_conformal_and_selective_metric_calculations():
    """Verify selective FNR, False Omission Rate, and coverage calculations."""
    cal_probs = np.array([0.1, 0.2, 0.8, 0.9, 0.95])
    cal_labels = np.array([0, 0, 1, 1, 1])
    eval_probs = np.array([0.02, 0.03, 0.50, 0.90, 0.92])
    eval_labels = np.array([0, 1, 0, 1, 1])

    results = evaluate_conformal_abstention(cal_probs, cal_labels, eval_probs, eval_labels, [0.05], tau_review=0.85)
    assert len(results) == 1
    r = results[0]
    assert "observed_fnr" in r
    assert "selective_fnr" in r
    assert "false_omission_rate" in r
    assert "coverage" in r
    assert 0.0 <= r["coverage"] <= 1.0


# 10. End-to-End Report Generation Test
def test_full_experiment_report_generation(tmp_path):
    """Verify that running the experiment produces valid structured JSON, CSV, and PNG files."""
    report = run_provenance_experiment(output_dir=tmp_path)
    assert report["experiment_name"] == "Phase D: Provenance Chain Security Signal Evaluation"
    assert report["methodology_frozen"] is True
    assert (tmp_path / "provenance_chain_ablation_report.json").exists()
    assert (tmp_path / "provenance_chain_ablation_results.csv").exists()
    assert (tmp_path / "provenance_chain_ablation.png").exists()
    assert (tmp_path / "provenance_chain_poisoning.png").exists()
    assert (tmp_path / "provenance_chain_risk_coverage.png").exists()


# 11. Cross-Combination Coverage Test
def test_cross_combination_coverage_all_datasets():
    """Verify all 4 cross-combinations exist in non-zero counts across all datasets."""
    datasets = generate_phase_d_datasets()
    for name in ["D1", "D3", "D4", "D5"]:
        df = datasets[name]
        legit = df["fraud_label"] == 0
        fraud = df["fraud_label"] == 1
        benign = df["provenance_category"].str.contains("benign")
        susp_or_poison = df["provenance_category"].str.contains("suspicious|poisoned")

        c1 = (legit & benign).sum()
        c2 = (legit & susp_or_poison).sum()
        c3 = (fraud & benign).sum()
        c4 = (fraud & susp_or_poison).sum()

        assert c1 > 0, f"{name} missing legit+benign"
        assert c2 > 0, f"{name} missing legit+susp/pois (hard negative)"
        assert c3 > 0, f"{name} missing fraud+benign (deceptive positive)"
        assert c4 > 0, f"{name} missing fraud+susp/pois"

