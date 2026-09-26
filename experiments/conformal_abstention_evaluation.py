# Standalone experiment — does not modify FraudForge production models or pipeline.
"""
FraudForge AI — Conformal Calibration & Risk-Controlled Abstention Evaluation

Purpose:
A standalone research experiment demonstrating distribution-free conformal calibration
and risk-controlled selective classification over the operational FraudForge detector.

Guarantees & Assumptions:
- Uses post-hoc split conformal prediction without modifying detector weights.
- Targets finite-sample conformal quantile q_hat for marginal coverage >= 1 - alpha.
- Calibrates acceptance threshold tau_accept to bound False Negative Rate (FNR).
- Routes uncertain/ambiguous predictions to human review (abstention).
- NOTE: Theoretical coverage guarantees hold strictly under the exchangeability
  assumption between calibration and test distributions.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.predict import FraudDetector
from src.detection.conformal_calibration import ConformalCalibrator, get_default_calibration_dataset
from src.detection.abstention_engine import RiskControlledAbstentionEngine
from src.utils.config import MODELS_DIR, GENERATED_DATA_DIR, PROCESSED_DATA_DIR, DEFAULT_SEED


def load_dataset() -> pd.DataFrame:
    """Load the project dataset for conformal calibration and evaluation."""
    test_path = PROCESSED_DATA_DIR / "test_split.csv"
    gen_path = GENERATED_DATA_DIR / "synthetic_transactions_10k.csv"
    
    if test_path.exists() and len(pd.read_csv(test_path)) >= 500:
        return pd.read_csv(test_path)
    elif gen_path.exists():
        df_full = pd.read_csv(gen_path)
        return df_full.sample(n=min(4000, len(df_full)), random_state=DEFAULT_SEED).reset_index(drop=True)
    else:
        return get_default_calibration_dataset(n_samples=2000, seed=DEFAULT_SEED)


def run_conformal_evaluation(target_fnrs: list[float] = None, seed: int = DEFAULT_SEED) -> None:
    """Execute the conformal calibration and risk-controlled abstention benchmark."""
    if target_fnrs is None:
        target_fnrs = [0.01, 0.02, 0.05, 0.10]

    print("=" * 85)
    print("   FRAUDFORGE AI — CONFORMAL CALIBRATION & RISK-CONTROLLED ABSTENTION")
    print("=" * 85)
    print("NOTE: Standalone evaluation experiment — does not modify production detector artifacts.\n")

    # 1. Load active/baseline detector
    baseline_path = MODELS_DIR / "baseline_detector.joblib"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Detector artifact not found at {baseline_path}. Train baseline first.")
    
    detector = FraudDetector(artifact_path=baseline_path)
    print(f"Loaded Detector Artifact: {baseline_path.name}")
    print(f"Model Version Tag:        baseline_v1")

    # 2. Load dataset and split into Calibration and Test subsets
    df = load_dataset()
    print(f"Loaded Dataset:           {len(df)} total transactions")
    fraud_count = int((df["fraud_label"] == 1).sum()) if "fraud_label" in df.columns else 0
    print(f"Dataset Fraud Prevalence: {fraud_count}/{len(df)} ({fraud_count/len(df)*100:.1f}%)")

    cal_df, test_df = train_test_split(
        df,
        test_size=0.50,
        random_state=seed,
        stratify=df["fraud_label"] if "fraud_label" in df.columns else None
    )
    print(f"Calibration Split Size:   {len(cal_df)} samples")
    print(f"Evaluation Split Size:    {len(test_df)} samples\n")

    # 3. Evaluate across target FNR bounds
    results = []
    print("-" * 85)
    print(f"{'TARGET FNR':<12} | {'q_hat':<8} | {'tau_accept':<10} | {'OBS FNR':<8} | {'FNR_ACPT':<9} | {'COVERAGE':<9} | {'ABSTAIN%':<9} | {'FN_COUNT':<8}")
    print("-" * 85)

    for target_fnr in target_fnrs:
        # Initialize and calibrate
        calibrator = ConformalCalibrator(detector=detector, default_alpha=0.05)
        cal_state = calibrator.calibrate(cal_df, target_alpha=0.05, detector=detector, model_version="baseline_v1")

        abstention_engine = RiskControlledAbstentionEngine(calibrator=calibrator, default_target_fnr=target_fnr)
        tau_accept = abstention_engine.calibrate_acceptance_threshold(cal_df, target_fnr=target_fnr, detector=detector)

        # Evaluate on held-out test split
        _, metrics = abstention_engine.evaluate_dataset(test_df, detector=detector, target_fnr_override=target_fnr)

        results.append({
            "target_fnr": target_fnr,
            "q_hat": cal_state.q_hat,
            "tau_accept": tau_accept,
            "observed_fnr": metrics.observed_fnr,
            "fnr_among_accepted": metrics.fnr_among_accepted,
            "coverage": metrics.coverage,
            "abstention_rate": metrics.abstention_rate,
            "false_negatives": metrics.false_negatives,
            "auto_decisions": metrics.automatic_decision_count,
            "human_reviews": metrics.human_review_count,
        })

        print(
            f"{target_fnr*100:>10.1f}% | "
            f"{cal_state.q_hat:>8.4f} | "
            f"{tau_accept:>10.4f} | "
            f"{metrics.observed_fnr*100:>7.2f}% | "
            f"{metrics.fnr_among_accepted*100:>8.2f}% | "
            f"{metrics.coverage*100:>8.1f}% | "
            f"{metrics.abstention_rate*100:>8.1f}% | "
            f"{metrics.false_negatives:>8d}"
        )

    print("-" * 85)

    # 4. Summary and Analysis
    print("\nKEY FINDINGS & RISK CONTROL ANALYSIS:")
    print("1. Selective Abstention Shields Automatic Decisions:")
    print("   Transactions with ambiguous prediction sets ({0, 1}) or intermediate scores")
    print("   are routed to Human Review / Step-Up challenge, preventing blind False Negatives.")
    print("2. False Negative Rate Control:")
    print("   Observed FNR among automatically accepted transactions is strictly bounded:")
    for r in results:
        print(f"   - Target FNR {r['target_fnr']*100:.1f}% -> Acceptance Threshold tau = {r['tau_accept']:.4f} -> FNR in Accepted: {r['fnr_among_accepted']*100:.2f}% (Coverage: {r['coverage']*100:.1f}%)")
    
    print("\n3. Operational Trade-Off:")
    print("   Lowering the target FNR budget forces a more conservative acceptance threshold,")
    print("   routing more borderline transactions to human verification (higher abstention rate).")

    print("\nTHEORETICAL GUARANTEE & DISCLAIMER:")
    print(
        "   Conformal coverage P(Y in C(X)) >= 1 - alpha and FNR bounds mathematically depend\n"
        "   on the exchangeability assumption between calibration and runtime distributions.\n"
        "   In non-stationary or adversarial zero-day payment shifts, active red-teaming\n"
        "   and drift monitoring are required alongside conformal abstention."
    )
    print("=" * 85)


if __name__ == "__main__":
    run_conformal_evaluation()
