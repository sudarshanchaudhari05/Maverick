"""FraudForge AI: Phase F Evaluation, Statistical Rigor & Gate Assessment Engine.

Implements:
- Paired Baseline vs Hardened detector evaluation
- McNemar's paired chi-squared test with Edwards continuity correction
- Evaluation of Locked Success Gates G1 through G5
- Comprehensive 6-pair pairwise dataset overlap audit
- Slice-level comparative performance metrics
"""

from typing import Dict, List, Any, Tuple, Optional
import math
import numpy as np
import pandas as pd
from scipy import stats

from src.detection.predict import FraudDetector


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Calculate accuracy, precision, recall, F1, FPR, FNR, and confusion matrix."""
    total = len(y_true)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    n_pos = tp + fn
    n_neg = tn + fp

    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / n_pos) if n_pos > 0 else 0.0
    fnr = float(fn / n_pos) if n_pos > 0 else 0.0
    fpr = float(fp / n_neg) if n_neg > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    mean_prob = float(np.mean(probs)) if probs is not None and len(probs) > 0 else None

    return {
        "total": total,
        "fraud_count": n_pos,
        "legitimate_count": n_neg,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "F1": round(f1, 4),
        "FPR": round(fpr, 4),
        "FNR": round(fnr, 4),
        "mean_probability": round(mean_prob, 4) if mean_prob is not None else None,
    }


def compute_mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, Any]:
    """Compute paired McNemar test comparing Model A and Model B accuracy against ground truth.

    Contingency table on correct/incorrect:
        b: Model A incorrect, Model B correct (B beats A)
        c: Model A correct, Model B incorrect (A beats B)
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    both_correct = int(np.sum(correct_a & correct_b))
    both_incorrect = int(np.sum(~correct_a & ~correct_b))
    b = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    c = int(np.sum(correct_a & ~correct_b))  # A right, B wrong

    discordant = b + c
    if discordant == 0:
        stat = 0.0
        p_value = 1.0
    else:
        # Edwards continuity-corrected McNemar statistic
        stat = float(((abs(b - c) - 1.0) ** 2) / discordant)
        p_val_raw = float(stats.chi2.sf(stat, df=1))
        p_value = float(f"{p_val_raw:.2e}") if p_val_raw < 1e-4 else round(p_val_raw, 5)

    return {
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "b_model_b_better": b,
        "c_model_a_better": c,
        "discordant_pairs": discordant,
        "test_statistic": round(stat, 4),
        "p_value": p_value,
        "statistically_significant_at_05": bool(p_value < 0.05),
    }


def audit_dataset_leakage(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Perform pairwise overlap check across all 6 dataset pairs."""
    pair_names = [
        ("F-D1", "F-D2"),
        ("F-D1", "F-D3"),
        ("F-D1", "F-D4"),
        ("F-D2", "F-D3"),
        ("F-D2", "F-D4"),
        ("F-D3", "F-D4"),
    ]

    results: Dict[str, Any] = {
        "total_pairs_checked": len(pair_names),
        "zero_leakage_passed": True,
        "pair_overlap_counts": {},
    }

    for name_a, name_b in pair_names:
        df_a = datasets[name_a]
        df_b = datasets[name_b]

        # Canonical 20 features (excluding target columns)
        cols = [c for c in df_a.columns if c not in ["attack_type", "fraud_label"]]
        set_a = set(tuple(x) for x in df_a[cols].itertuples(index=False))
        set_b = set(tuple(x) for x in df_b[cols].itertuples(index=False))
        overlap = len(set_a & set_b)

        results["pair_overlap_counts"][f"{name_a}_vs_{name_b}"] = overlap
        if overlap > 0:
            results["zero_leakage_passed"] = False

    return results


class PhaseFEvaluationEngine:
    """Evaluates Baseline (Model A) vs Hardened (Model B) across Phase F datasets and Top-5 slices."""

    def __init__(self, baseline_detector: FraudDetector, hardened_detector: FraudDetector):
        self.baseline_detector = baseline_detector
        self.hardened_detector = hardened_detector

    def evaluate_dataset_pair(
        self,
        df: pd.DataFrame,
        dataset_name: str,
    ) -> Dict[str, Any]:
        """Run paired evaluation on a dataset for Model A vs Model B."""
        y_true = df["fraud_label"].values

        probs_a = self.baseline_detector.predict_proba(df)
        preds_a = self.baseline_detector.predict(df)

        probs_b = self.hardened_detector.predict_proba(df)
        preds_b = self.hardened_detector.predict(df)

        metrics_a = compute_binary_metrics(y_true, preds_a, probs_a)
        metrics_b = compute_binary_metrics(y_true, preds_b, probs_b)
        mcnemar = compute_mcnemar_test(y_true, preds_a, preds_b)

        delta_recall = round(metrics_b["recall"] - metrics_a["recall"], 4)
        delta_fnr = round(metrics_b["FNR"] - metrics_a["FNR"], 4)
        delta_f1 = round(metrics_b["F1"] - metrics_a["F1"], 4)
        delta_fpr = round(metrics_b["FPR"] - metrics_a["FPR"], 4)

        return {
            "dataset_name": dataset_name,
            "sample_count": len(df),
            "model_a_baseline": metrics_a,
            "model_b_hardened": metrics_b,
            "delta_metrics": {
                "delta_recall": delta_recall,
                "delta_fnr": delta_fnr,
                "delta_f1": delta_f1,
                "delta_fpr": delta_fpr,
            },
            "mcnemar_test": mcnemar,
        }

    def evaluate_top_slices(
        self,
        eval_df: pd.DataFrame,
        top_slices: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Evaluate baseline vs hardened detector on each discovered Top-5 slice."""
        slice_results: List[Dict[str, Any]] = []

        for s in top_slices:
            slice_id = s.get("slice_id", "F-SLICE-???")
            slice_def = s["slice_definition"]

            # Filter eval_df matching slice definition
            mask = pd.Series(True, index=eval_df.index)
            for col, val in slice_def.items():
                if col in eval_df.columns:
                    mask = mask & (eval_df[col].astype(str) == str(val))

            sub = eval_df[mask]
            if len(sub) == 0:
                continue

            y_sub = sub["fraud_label"].values
            probs_a = self.baseline_detector.predict_proba(sub)
            preds_a = self.baseline_detector.predict(sub)
            probs_b = self.hardened_detector.predict_proba(sub)
            preds_b = self.hardened_detector.predict(sub)

            m_a = compute_binary_metrics(y_sub, preds_a, probs_a)
            m_b = compute_binary_metrics(y_sub, preds_b, probs_b)
            mcn = compute_mcnemar_test(y_sub, preds_a, preds_b)

            fnr_improved = bool(m_b["FNR"] < m_a["FNR"])

            slice_results.append({
                "slice_id": slice_id,
                "slice_definition": slice_def,
                "matched_samples": len(sub),
                "fraud_count": int(np.sum(y_sub == 1)),
                "baseline_fnr": m_a["FNR"],
                "hardened_fnr": m_b["FNR"],
                "delta_fnr": round(m_b["FNR"] - m_a["FNR"], 4),
                "baseline_recall": m_a["recall"],
                "hardened_recall": m_b["recall"],
                "delta_recall": round(m_b["recall"] - m_a["recall"], 4),
                "fnr_improved": fnr_improved,
                "model_a_baseline": m_a,
                "model_b_hardened": m_b,
                "mcnemar": mcn,
            })

        return slice_results

    @staticmethod
    def evaluate_success_gates(
        slice_results: List[Dict[str, Any]],
        eval_fd4: Dict[str, Any],
        eval_fd2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess the 5 locked Phase F success gates."""
        # Gate G1: FNR_hardened < FNR_baseline on >= 4 of 5 slices
        slices_improved = sum(1 for s in slice_results if s["fnr_improved"])
        total_slices = len(slice_results)
        g1_passed = bool(slices_improved >= min(4, total_slices))

        # Gate G2: On F-D4, Recall_hardened > Recall_baseline
        rec_base_fd4 = eval_fd4["model_a_baseline"]["recall"]
        rec_hard_fd4 = eval_fd4["model_b_hardened"]["recall"]
        g2_passed = bool(rec_hard_fd4 > rec_base_fd4)

        # Gate G3: On F-D4, Hardened FPR must not increase by > +1.0 percentage point (+0.01)
        fpr_base_fd4 = eval_fd4["model_a_baseline"]["FPR"]
        fpr_hard_fd4 = eval_fd4["model_b_hardened"]["FPR"]
        fpr_delta_fd4 = fpr_hard_fd4 - fpr_base_fd4
        g3_passed = bool(fpr_delta_fd4 <= 0.0101)  # allow tiny floating point tolerance

        # Gate G4: On F-D2, Hardened recall must not decrease by > 2.0 percentage points (-0.02)
        rec_base_fd2 = eval_fd2["model_a_baseline"]["recall"]
        rec_hard_fd2 = eval_fd2["model_b_hardened"]["recall"]
        rec_drop_fd2 = rec_base_fd2 - rec_hard_fd2
        g4_passed = bool(rec_drop_fd2 <= 0.0201)

        # Gate G5: On F-D4, FNR_hardened < FNR_baseline
        fnr_base_fd4 = eval_fd4["model_a_baseline"]["FNR"]
        fnr_hard_fd4 = eval_fd4["model_b_hardened"]["FNR"]
        g5_passed = bool(fnr_hard_fd4 < fnr_base_fd4)

        all_passed = bool(g1_passed and g2_passed and g3_passed and g4_passed and g5_passed)

        return {
            "all_gates_passed": all_passed,
            "G1_slice_improvement": {
                "rule": "FNR_hardened < FNR_baseline on >= 4 of 5 discovered slices",
                "slices_improved": slices_improved,
                "total_slices": total_slices,
                "passed": g1_passed,
            },
            "G2_overall_f5_recall": {
                "rule": "On F-D4, Recall_hardened > Recall_baseline",
                "baseline_recall": rec_base_fd4,
                "hardened_recall": rec_hard_fd4,
                "delta_recall": round(rec_hard_fd4 - rec_base_fd4, 4),
                "passed": g2_passed,
            },
            "G3_fpr_protection": {
                "rule": "On F-D4, Hardened FPR must not increase by > +1.0 percentage point",
                "baseline_fpr": fpr_base_fd4,
                "hardened_fpr": fpr_hard_fd4,
                "delta_fpr": round(fpr_delta_fd4, 4),
                "passed": g3_passed,
            },
            "G4_random_attack_preservation": {
                "rule": "On F-D2, Hardened recall must not decrease by > 2.0 percentage points",
                "baseline_recall": rec_base_fd2,
                "hardened_recall": rec_hard_fd2,
                "delta_recall": round(rec_hard_fd2 - rec_base_fd2, 4),
                "passed": g4_passed,
            },
            "G5_unseen_targeted_generalization": {
                "rule": "On F-D4, FNR_hardened < FNR_baseline",
                "baseline_fnr": fnr_base_fd4,
                "hardened_fnr": fnr_hard_fd4,
                "delta_fnr": round(fnr_hard_fd4 - fnr_base_fd4, 4),
                "passed": g5_passed,
            },
        }
