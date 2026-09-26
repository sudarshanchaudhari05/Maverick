"""FraudForge AI: Risk-Controlled Selective Classification & Abstention Engine.

Implements risk-controlled abstention for high-stakes payment decisions:
- Automatic decisions (ALLOW / BLOCK) are made only when conformal uncertainty is low
  and risk is strictly within the user-specified Target FNR budget.
- Ambiguous ({0, 1}), empty set, or marginal transactions trigger:
    decision = "ABSTAIN"
- "ABSTAIN" routes the transaction to Human Review / Step-up challenge, NOT treating it as fraud.

EXPLICIT TELEMETRY & AUDITING:
- target_fnr: User-specified acceptable false-negative risk bound.
- observed_fnr: Empirical false-negative rate among evaluated fraud transactions.
- coverage: Proportion of transactions decided automatically without abstention.
- abstention_rate: Proportion of transactions routed for human review.
- false_negatives: Count of actual frauds mistakenly auto-accepted.
- automatic_decision_count: Total automated ALLOW and BLOCK decisions.
- human_review_count: Total transactions abstained for human inspection.

CRITICAL STATISTICAL NOTICE:
Ordinary conformal prediction does not inherently guarantee unconditional false-negative
rate bounds without risk-controlling selective classification and exchangeable (i.i.d.)
data distributions. Performance may degrade under adversarial distribution shifts.
"""

from dataclasses import dataclass, asdict
import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd

from src.detection.predict import FraudDetector
from src.detection.conformal_calibration import (
    ConformalCalibrator,
    ConformalPredictionSet,
    normalize_transaction_dataframe,
)


@dataclass
class SelectiveDecision:
    """Individual decision result incorporating risk-controlled abstention."""
    transaction_id: str
    decision: str                   # "ALLOW", "BLOCK", or "ABSTAIN"
    routing: str                    # "AUTOMATIC_DECISION" or "HUMAN_REVIEW"
    is_abstained: bool              # True if routed for human review
    fraud_probability: float        # raw model score
    conformal_set: List[int]        # [0], [1], [0, 1], or []
    conformal_set_size: int         # 0, 1, or 2
    acceptance_threshold: float     # tau_accept calibrated for target FNR
    block_threshold: float          # tau_block
    target_fnr: float               # nominal target FNR budget
    reason: str                     # explainability explanation
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AbstentionEvaluationMetrics:
    """Comprehensive performance metrics under risk-controlled selective classification."""
    target_fnr: float
    observed_fnr: float
    fnr_among_accepted: float
    coverage: float
    abstention_rate: float
    total_evaluated: int
    automatic_decision_count: int
    human_review_count: int
    accepted_count: int
    blocked_count: int
    false_negatives: int
    false_positives: int
    true_positives: int
    true_negatives: int
    precision_auto: float
    recall_auto: float
    model_version: str
    exchangeability_disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RiskControlledAbstentionEngine:
    """Controls false-negative risk by selectively deciding or abstaining."""

    def __init__(
        self,
        calibrator: ConformalCalibrator,
        default_target_fnr: float = 0.05,
        default_block_threshold: float = 0.60,
    ):
        self.calibrator = calibrator
        self.target_fnr = default_target_fnr
        self.block_threshold = default_block_threshold
        self.acceptance_threshold = 0.30  # Default initial boundary
        self._exchangeability_disclaimer = (
            "Statistical false-negative bounds hold under exchangeable (i.i.d.) calibration and test data. "
            "Under novel zero-day attacks or adversarial distribution shifts, observed FNR may exceed target FNR."
        )

    def calibrate_acceptance_threshold(
        self,
        calibration_df: pd.DataFrame,
        target_fnr: Optional[float] = None,
        detector: Optional[FraudDetector] = None,
    ) -> float:
        """Calibrate the maximum fraud probability threshold tau_accept for automatic acceptance
        such that the empirical FNR among fraud samples is bounded by target_fnr under exchangeability."""
        t_fnr = float(target_fnr if target_fnr is not None else self.target_fnr)
        self.target_fnr = t_fnr

        active_detector = detector or self.calibrator.detector
        if active_detector is None:
            raise RuntimeError("No FraudDetector available for threshold calibration.")

        if "fraud_label" not in calibration_df.columns:
            raise ValueError("Calibration dataset must contain 'fraud_label' column.")

        y_true = calibration_df["fraud_label"].to_numpy().astype(int)
        df_norm = normalize_transaction_dataframe(calibration_df)
        if "fraud_probability" in calibration_df.columns and calibration_df["fraud_probability"].notnull().all():
            probs = calibration_df["fraud_probability"].to_numpy(dtype=float)
        else:
            probs = active_detector.predict_proba(df_norm)

        fraud_probs = probs[y_true == 1]
        if len(fraud_probs) == 0:
            self.acceptance_threshold = 0.15
            return self.acceptance_threshold

        # For fraud transactions, any score <= tau_accept results in a False Negative (auto-allow).
        # We bound tau_accept by both the empirical fraud score quantile and the conformal risk scale:
        tau_fraud_quantile = float(np.quantile(fraud_probs, t_fnr))
        q_ref = getattr(self.calibrator, "q_hat", None) or 0.30
        alpha_ref = getattr(self.calibrator, "target_alpha", 0.05)
        conformal_risk_bound = q_ref * (t_fnr / max(alpha_ref, 0.01))

        tau_val = min(tau_fraud_quantile, conformal_risk_bound)
        # Ensure reasonable defensive bounds [0.01, 0.40]
        self.acceptance_threshold = round(float(np.clip(tau_val, 0.01, 0.40)), 4)
        return self.acceptance_threshold

    def evaluate_transaction(
        self,
        transaction: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        detector: Optional[FraudDetector] = None,
        target_fnr_override: Optional[float] = None,
    ) -> SelectiveDecision:
        """Evaluate a single transaction with conformal prediction and risk-controlled abstention."""
        # 1. Conformal Prediction Set
        conformal_res = self.calibrator.predict_single(transaction, detector=detector)
        prob = conformal_res.fraud_probability
        pred_set = conformal_res.prediction_set
        tx_id = conformal_res.transaction_id
        t_fnr = target_fnr_override if target_fnr_override is not None else self.target_fnr

        tau_acc = self.acceptance_threshold
        tau_blk = self.block_threshold
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Decision Logic:
        # Rule 1: Ambiguous ({0, 1}) or Empty ([]) prediction sets must ABSTAIN
        if conformal_res.is_ambiguous:
            decision = "ABSTAIN"
            routing = "HUMAN_REVIEW"
            reason = f"Conformal prediction set is ambiguous {{0, 1}} with high epistemic uncertainty (p0={conformal_res.p_value_0}, p1={conformal_res.p_value_1})."

        elif conformal_res.is_empty:
            decision = "ABSTAIN"
            routing = "HUMAN_REVIEW"
            reason = "Conformal prediction set is empty (out-of-distribution anomaly detected)."

        # Rule 2: Single label {0} (Legitimate) - check acceptance threshold
        elif pred_set == [0]:
            if prob <= tau_acc:
                decision = "ALLOW"
                routing = "AUTOMATIC_DECISION"
                reason = f"Conformal set {{0}} with fraud score ({prob:.4f}) within target FNR budget (tau <= {tau_acc})."
            else:
                decision = "ABSTAIN"
                routing = "HUMAN_REVIEW"
                reason = f"Conformal set {{0}} but fraud probability ({prob:.4f}) exceeds controlled acceptance cutoff ({tau_acc})."

        # Rule 3: Single label {1} (Fraud) - check block threshold
        elif pred_set == [1]:
            if prob >= tau_blk:
                decision = "BLOCK"
                routing = "AUTOMATIC_DECISION"
                reason = f"Conformal set {{1}} with confident fraud probability ({prob:.4f} >= {tau_blk})."
            else:
                decision = "ABSTAIN"
                routing = "HUMAN_REVIEW"
                reason = f"Conformal set {{1}} but fraud probability ({prob:.4f}) falls below confident block threshold ({tau_blk})."

        else:
            decision = "ABSTAIN"
            routing = "HUMAN_REVIEW"
            reason = "Uncertain risk margin. Routed for human investigation."

        return SelectiveDecision(
            transaction_id=tx_id,
            decision=decision,
            routing=routing,
            is_abstained=(decision == "ABSTAIN"),
            fraud_probability=prob,
            conformal_set=pred_set,
            conformal_set_size=len(pred_set),
            acceptance_threshold=tau_acc,
            block_threshold=tau_blk,
            target_fnr=t_fnr,
            reason=reason,
            timestamp=now_str,
        )

    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        detector: Optional[FraudDetector] = None,
        target_fnr_override: Optional[float] = None,
        model_version: str = "active",
    ) -> Tuple[List[SelectiveDecision], AbstentionEvaluationMetrics]:
        """Evaluate a full dataset and compute comprehensive risk-controlled metrics."""
        t_fnr = target_fnr_override if target_fnr_override is not None else self.target_fnr
        decisions: List[SelectiveDecision] = []

        conformal_sets = self.calibrator.predict_batch(df, detector=detector)
        records = df.to_dict(orient="records")

        has_ground_truth = "fraud_label" in df.columns
        y_true = df["fraud_label"].to_numpy().astype(int) if has_ground_truth else np.zeros(len(df))

        n_total = len(df)
        n_auto = 0
        n_review = 0
        n_accepted = 0
        n_blocked = 0

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        tau_acc = self.acceptance_threshold
        tau_blk = self.block_threshold
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        for i, c_set in enumerate(conformal_sets):
            prob = c_set.fraud_probability
            pred_set = c_set.prediction_set
            tx_id = c_set.transaction_id

            if c_set.is_ambiguous:
                dec = "ABSTAIN"
                routing = "HUMAN_REVIEW"
                reason = "Ambiguous conformal set {0, 1}."
            elif c_set.is_empty:
                dec = "ABSTAIN"
                routing = "HUMAN_REVIEW"
                reason = "Out-of-distribution empty prediction set."
            elif pred_set == [0]:
                if prob <= tau_acc:
                    dec = "ALLOW"
                    routing = "AUTOMATIC_DECISION"
                    reason = f"Conformal set {{0}} within acceptance threshold ({prob:.4f} <= {tau_acc})."
                else:
                    dec = "ABSTAIN"
                    routing = "HUMAN_REVIEW"
                    reason = f"Conformal set {{0}} but prob ({prob:.4f}) exceeds acceptance threshold ({tau_acc})."
            elif pred_set == [1]:
                if prob >= tau_blk:
                    dec = "BLOCK"
                    routing = "AUTOMATIC_DECISION"
                    reason = f"Conformal set {{1}} with high fraud confidence ({prob:.4f} >= {tau_blk})."
                else:
                    dec = "ABSTAIN"
                    routing = "HUMAN_REVIEW"
                    reason = f"Conformal set {{1}} but prob ({prob:.4f}) < block threshold ({tau_blk})."
            else:
                dec = "ABSTAIN"
                routing = "HUMAN_REVIEW"
                reason = "Marginal score."

            is_abs = (dec == "ABSTAIN")
            if is_abs:
                n_review += 1
            else:
                n_auto += 1
                if dec == "ALLOW":
                    n_accepted += 1
                    if has_ground_truth:
                        if y_true[i] == 1:
                            fn += 1  # Fraud was mistakenly allowed (False Negative)
                        else:
                            tn += 1  # Legitimate was correctly allowed (True Negative)
                elif dec == "BLOCK":
                    n_blocked += 1
                    if has_ground_truth:
                        if y_true[i] == 1:
                            tp += 1  # Fraud was correctly blocked (True Positive)
                        else:
                            fp += 1  # Legitimate was mistakenly blocked (False Positive)

            decisions.append(
                SelectiveDecision(
                    transaction_id=tx_id,
                    decision=dec,
                    routing=routing,
                    is_abstained=is_abs,
                    fraud_probability=prob,
                    conformal_set=pred_set,
                    conformal_set_size=len(pred_set),
                    acceptance_threshold=tau_acc,
                    block_threshold=tau_blk,
                    target_fnr=t_fnr,
                    reason=reason,
                    timestamp=now_str,
                )
            )

        # Performance metric calculations
        coverage = float(n_auto / max(1, n_total))
        abstention_rate = float(n_review / max(1, n_total))

        total_actual_fraud = int(np.sum(y_true == 1)) if has_ground_truth else 0
        observed_fnr = float(fn / max(1, total_actual_fraud)) if total_actual_fraud > 0 else 0.0
        fnr_among_accepted = float(fn / max(1, n_accepted)) if n_accepted > 0 else 0.0

        prec_auto = float(tp / max(1, tp + fp)) if (tp + fp) > 0 else 0.0
        rec_auto = float(tp / max(1, tp + fn)) if (tp + fn) > 0 else 0.0

        metrics = AbstentionEvaluationMetrics(
            target_fnr=round(t_fnr, 4),
            observed_fnr=round(observed_fnr, 4),
            fnr_among_accepted=round(fnr_among_accepted, 4),
            coverage=round(coverage, 4),
            abstention_rate=round(abstention_rate, 4),
            total_evaluated=n_total,
            automatic_decision_count=n_auto,
            human_review_count=n_review,
            accepted_count=n_accepted,
            blocked_count=n_blocked,
            false_negatives=fn,
            false_positives=fp,
            true_positives=tp,
            true_negatives=tn,
            precision_auto=round(prec_auto, 4),
            recall_auto=round(rec_auto, 4),
            model_version=model_version,
            exchangeability_disclaimer=self._exchangeability_disclaimer,
        )

        return decisions, metrics
