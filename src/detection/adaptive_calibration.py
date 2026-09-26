"""FraudForge AI: Adaptive Recalibration & Delayed Label Buffer Engine.

Manages controlled post-hoc recalibration of conformal decision thresholds
under a realistic delayed-label streaming protocol:
- LABEL_DELAY protocol: Labels for batch t are strictly sequestered until batch t + delay (e.g. delay = 3).
- RECALIBRATION_WINDOW: Requires a fixed quota of eligible labeled transactions (e.g. 500 samples)
  to perform empirical threshold recalibration.
- FALLBACK POLICY: If drift is flagged but eligible delayed labels are insufficient (< 500),
  the system flags CALIBRATION_STALE = True, preserves existing thresholds, and executes
  recalibration as soon as the quota of delayed labels arrives.
- SAFETY INVARIANCE: XGBoost model weights are never retrained or modified; only post-hoc
  conformal nonconformity quantiles and selective acceptance thresholds are updated.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.detection.predict import FraudDetector
from src.detection.conformal_calibration import (
    ConformalCalibrator,
    normalize_transaction_dataframe,
)


class DelayedLabelStreamBuffer:
    """Buffers streaming batches and enforces strict temporal label-delay visibility."""

    def __init__(self, label_delay_batches: int = 3):
        """Initialize the buffer with a fixed batch delay."""
        self.label_delay = int(label_delay_batches)
        self.batches: Dict[int, pd.DataFrame] = {}
        self.total_buffered_records = 0

    def add_batch(self, batch_id: int, df_batch: pd.DataFrame) -> None:
        """Store an incoming batch with its hidden ground-truth labels."""
        df_copy = df_batch.copy().reset_index(drop=True)
        df_copy["_stream_batch_id"] = int(batch_id)
        self.batches[int(batch_id)] = df_copy
        self.total_buffered_records += len(df_copy)

    def get_eligible_batch_ids(self, current_batch_id: int) -> List[int]:
        """Return list of batch IDs whose labels are available at current_batch_id."""
        max_eligible_id = int(current_batch_id) - self.label_delay
        return [b_id for b_id in sorted(self.batches.keys()) if b_id <= max_eligible_id]

    def get_eligible_labeled_samples(
        self,
        current_batch_id: int,
        max_samples: Optional[int] = 500,
    ) -> pd.DataFrame:
        """Extract eligible labeled observations whose label delay has elapsed.

        Args:
            current_batch_id: Current batch index (1-indexed).
            max_samples: If specified, returns the most recent up to max_samples eligible rows.

        Returns:
            DataFrame of eligible labeled observations.
        """
        eligible_ids = self.get_eligible_batch_ids(current_batch_id)
        if not eligible_ids:
            return pd.DataFrame()

        eligible_dfs = [self.batches[b_id] for b_id in eligible_ids]
        combined = pd.concat(eligible_dfs, ignore_index=True)

        if max_samples is not None and len(combined) > max_samples:
            # Take the most recent max_samples
            combined = combined.iloc[-max_samples:].reset_index(drop=True)

        return combined

    def get_eligible_count(self, current_batch_id: int) -> int:
        """Count total eligible labeled observations available at current_batch_id."""
        eligible_ids = self.get_eligible_batch_ids(current_batch_id)
        return sum(len(self.batches[b_id]) for b_id in eligible_ids)


class AdaptiveCalibrationManager:
    """Coordinates post-hoc conformal recalibration under streaming drift alarms."""

    def __init__(
        self,
        initial_calibrator: ConformalCalibrator,
        target_fnr: float = 0.05,
        recalibration_window: int = 500,
        tau_review: float = 0.85,
        tau_accept_min: float = 0.01,
        tau_accept_max: float = 0.40,
    ):
        """Initialize the adaptive calibration manager.

        Args:
            initial_calibrator: Frozen baseline ConformalCalibrator instance.
            target_fnr: Nominal false-negative rate target.
            recalibration_window: Minimum eligible labeled samples required for recalibration.
            tau_review: Upper threshold for automatic review routing.
            tau_accept_min: Defensively clipped minimum acceptance threshold.
            tau_accept_max: Defensively clipped maximum acceptance threshold.
        """
        self.calibrator = initial_calibrator
        self.target_fnr = float(target_fnr)
        self.recalibration_window = int(recalibration_window)
        self.tau_review = float(tau_review)
        self.tau_accept_min = float(tau_accept_min)
        self.tau_accept_max = float(tau_accept_max)

        # Initial state (v0)
        self.version_index = 0
        self.calibration_version = "v0"
        self.is_stale = False
        self.pending_recalibration = False
        self.pending_trigger_batch: Optional[int] = None
        self.pending_reason: str = ""

        # Compute initial tau_accept from initial calibrator
        q_init = getattr(self.calibrator, "q_hat", 0.30) or 0.30
        nominal_alpha = getattr(self.calibrator, "target_alpha", 0.05)
        raw_tau = q_init * (self.target_fnr / max(0.01, nominal_alpha))
        self.current_tau_accept = round(float(np.clip(raw_tau, self.tau_accept_min, self.tau_accept_max)), 4)

        self.recalibration_events: List[Dict[str, Any]] = []

    def compute_tau_accept(
        self,
        fraud_probs: np.ndarray,
        q_hat: float,
        nominal_alpha: float = 0.05,
    ) -> float:
        """Compute the bounded acceptance threshold for target FNR."""
        if len(fraud_probs) == 0:
            return 0.15
        tau_fraud_quantile = float(np.quantile(fraud_probs, self.target_fnr))
        conformal_bound = q_hat * (self.target_fnr / max(0.01, nominal_alpha))
        tau_val = min(tau_fraud_quantile, conformal_bound)
        return round(float(np.clip(tau_val, self.tau_accept_min, self.tau_accept_max)), 4)

    def trigger_recalibration(
        self,
        current_batch_id: int,
        stream_buffer: DelayedLabelStreamBuffer,
        detector: FraudDetector,
        reason: str = "martingale_alarm",
    ) -> Dict[str, Any]:
        """Attempt to recalibrate conformal thresholds using eligible delayed labels.

        Args:
            current_batch_id: Batch index at which alarm or recalibration is requested.
            stream_buffer: The buffer holding streaming data with label delay.
            detector: Frozen detector used to score calibration samples.
            reason: Trigger cause ("martingale_alarm" or "oracle_transition").

        Returns:
            Dict recording event outcome (success or fallback pending).
        """
        eligible_count = stream_buffer.get_eligible_count(current_batch_id)

        if eligible_count < self.recalibration_window:
            # Fallback: Insufficient delayed labels available
            self.is_stale = True
            self.pending_recalibration = True
            self.pending_trigger_batch = current_batch_id
            self.pending_reason = reason

            event = {
                "event_type": "recalibration_deferred",
                "trigger_batch": current_batch_id,
                "reason": reason,
                "status": "STALE_PENDING_LABELS",
                "eligible_samples_available": eligible_count,
                "required_samples": self.recalibration_window,
                "active_version": self.calibration_version,
                "tau_accept": self.current_tau_accept,
            }
            self.recalibration_events.append(event)
            return event

        # Execute recalibration with eligible samples
        recal_df = stream_buffer.get_eligible_labeled_samples(
            current_batch_id=current_batch_id,
            max_samples=self.recalibration_window,
        )

        y_true = recal_df["fraud_label"].to_numpy().astype(int)
        df_norm = normalize_transaction_dataframe(recal_df)
        probs = detector.predict_proba(df_norm)

        # Calibrate nonconformity scores
        fraud_mask = (y_true == 1)
        fraud_probs = probs[fraud_mask]
        n_fraud = len(fraud_probs)

        if n_fraud > 5:
            alpha_scores = 1.0 - fraud_probs
            nominal_alpha = 0.05
            q_level = min(1.0, float(np.ceil((n_fraud + 1) * (1.0 - nominal_alpha)) / n_fraud))
            new_q_hat = float(np.quantile(alpha_scores, q_level))
            new_tau_accept = self.compute_tau_accept(fraud_probs, new_q_hat, nominal_alpha)
        else:
            # If very few fraud samples, retain safe lower bound
            new_q_hat = self.calibrator.q_hat or 0.30
            new_tau_accept = self.tau_accept_min

        prev_version = self.calibration_version
        prev_tau = self.current_tau_accept

        self.version_index += 1
        self.calibration_version = f"v{self.version_index}"
        self.current_tau_accept = new_tau_accept
        self.is_stale = False
        self.pending_recalibration = False
        self.pending_trigger_batch = None
        self.pending_reason = ""

        event = {
            "event_type": "recalibration_executed",
            "trigger_batch": current_batch_id,
            "reason": reason,
            "status": "RECALIBRATED",
            "previous_version": prev_version,
            "new_version": self.calibration_version,
            "previous_tau_accept": prev_tau,
            "new_tau_accept": new_tau_accept,
            "q_hat": round(new_q_hat, 5),
            "eligible_samples_used": len(recal_df),
            "fraud_samples_in_window": int(n_fraud),
        }
        self.recalibration_events.append(event)
        return event

    def check_and_execute_pending(
        self,
        current_batch_id: int,
        stream_buffer: DelayedLabelStreamBuffer,
        detector: FraudDetector,
    ) -> Optional[Dict[str, Any]]:
        """Check if pending recalibration can now be executed as newly delayed labels arrived."""
        if not self.pending_recalibration:
            return None

        eligible_count = stream_buffer.get_eligible_count(current_batch_id)
        if eligible_count >= self.recalibration_window:
            reason = f"deferred_from_b{self.pending_trigger_batch}_{self.pending_reason}"
            return self.trigger_recalibration(
                current_batch_id=current_batch_id,
                stream_buffer=stream_buffer,
                detector=detector,
                reason=reason,
            )
        return None

    def get_state(self) -> Dict[str, Any]:
        """Return the current diagnostic state of the adaptive manager."""
        return {
            "calibration_version": self.calibration_version,
            "version_index": self.version_index,
            "tau_accept": self.current_tau_accept,
            "tau_review": self.tau_review,
            "is_stale": self.is_stale,
            "pending_recalibration": self.pending_recalibration,
            "target_fnr": self.target_fnr,
            "recalibration_window": self.recalibration_window,
            "recalibration_event_count": len(self.recalibration_events),
        }
