"""FraudForge AI: Conformal Test-Martingale Drift Detection Engine.

Implements sequential drift detection using conformal test-martingales:
- Evaluates exchangeability between incoming streaming predictions and the reference calibration distribution.
- Constructs empirical conformal p-values from calibration nonconformity scores.
- Accumulates evidence sequentially using the Power Martingale betting function (Vovk et al., 2005):
    f(u) = epsilon * u^(epsilon - 1),  with 0 < epsilon < 1.
- Computes evidence in log-space to guarantee numerical stability across extended streams.
- Derives the alarm threshold deterministically from Ville's inequality:
    tau = 1 / alpha_drift (e.g. tau = 100.0 for alpha_drift = 0.01).

Mathematical Foundation & Guarantees:
- Under the null hypothesis H_0 (exchangeability with calibration data), M_n is a non-negative
  supermartingale with E[M_n] = 1.
- By Ville's inequality: P_H0(exists n >= 1: M_n >= 1 / alpha_drift) <= alpha_drift.
- When distribution drift occurs (e.g. concept drift, evasive attacks, or fraud prevalence shifts),
  conformal nonconformity scores increase, p-values concentrate near 0, and M_n grows exponentially.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np


class ConformalTestMartingale:
    """Sequential Conformal Test-Martingale for streaming distribution shift detection."""

    def __init__(
        self,
        reference_scores: Union[List[float], np.ndarray],
        epsilon: float = 0.80,
        alpha_drift: float = 0.01,
        min_p_value: float = 1e-6,
    ):
        """Initialize the conformal test-martingale with reference calibration scores.

        Args:
            reference_scores: Array of nonconformity scores from the reference calibration set.
            epsilon: Power martingale betting parameter (0 < epsilon < 1). Default 0.80.
            alpha_drift: Sequential drift significance level. Default 0.01 (1% false alarm rate bound).
            min_p_value: Numerical lower bound for p-values to prevent log(0) singularities.
        """
        if not (0.0 < epsilon < 1.0):
            raise ValueError(f"Epsilon must be in (0, 1), got {epsilon}")
        if not (0.0 < alpha_drift < 1.0):
            raise ValueError(f"alpha_drift must be in (0, 1), got {alpha_drift}")

        self.ref_scores = np.sort(np.asarray(reference_scores, dtype=np.float64))
        self.m = len(self.ref_scores)
        if self.m == 0:
            raise ValueError("reference_scores cannot be empty.")

        self.epsilon = float(epsilon)
        self.alpha_drift = float(alpha_drift)
        self.min_p_value = float(min_p_value)

        # Ville's inequality alarm threshold: tau = 1 / alpha_drift
        self.alarm_threshold = float(1.0 / self.alpha_drift)
        self.log_alarm_threshold = float(np.log(self.alarm_threshold))

        # Log evidence state: log(M_0) = 0.0 (M_0 = 1.0)
        self.log_evidence = 0.0
        self.total_observations = 0
        self.alarm_count = 0
        self.last_alarm_observation: Optional[int] = None

    @property
    def evidence(self) -> float:
        """Current martingale value M_t = exp(log_evidence), safely clipped."""
        return float(np.exp(np.clip(self.log_evidence, -50.0, 50.0)))

    @property
    def is_alarm(self) -> bool:
        """Check if accumulated evidence exceeds the Ville alarm threshold."""
        return bool(self.log_evidence >= self.log_alarm_threshold)

    def compute_p_value(self, nonconf_score: float) -> float:
        """Compute the conformal p-value of an observation against reference scores.

        p_t = (sum_{j=1}^m I(s_j^ref >= s_t) + 1) / (m + 1)
        """
        score = float(nonconf_score)
        # Count reference scores >= test score
        count_greater = int(np.sum(self.ref_scores >= score))
        p_val = float((count_greater + 1.0) / (self.m + 1.0))
        return float(np.clip(p_val, self.min_p_value, 1.0))

    def betting_factor(self, p_value: float) -> float:
        """Compute the Power Martingale multiplier f(u) = epsilon * u^(epsilon - 1)."""
        u = max(self.min_p_value, float(p_value))
        return float(self.epsilon * (u ** (self.epsilon - 1.0)))

    def update_observation(self, nonconf_score: float) -> Dict[str, Any]:
        """Update the test-martingale with a single incoming observation's nonconformity score.

        Args:
            nonconf_score: Nonconformity score s_t computed for the incoming observation.

        Returns:
            Dict containing step p-value, log step increment, updated evidence, and alarm flag.
        """
        p_val = self.compute_p_value(nonconf_score)
        log_step = np.log(self.epsilon) + (self.epsilon - 1.0) * np.log(p_val)
        
        self.log_evidence += float(log_step)
        self.total_observations += 1

        alarm_fired = self.is_alarm
        if alarm_fired and (self.last_alarm_observation != self.total_observations):
            self.alarm_count += 1
            self.last_alarm_observation = self.total_observations

        return {
            "observation_index": self.total_observations,
            "nonconf_score": round(float(nonconf_score), 5),
            "p_value": round(p_val, 6),
            "log_step": round(float(log_step), 5),
            "log_evidence": round(self.log_evidence, 5),
            "evidence": round(self.evidence, 4),
            "is_alarm": alarm_fired,
            "alarm_threshold": self.alarm_threshold,
        }

    def update_batch(self, nonconf_scores: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """Update the martingale with a batch of nonconformity scores.

        Args:
            nonconf_scores: Array or list of nonconformity scores for the batch.

        Returns:
            Dict summarizing batch p-values, ending evidence, and whether an alarm triggered.
        """
        scores = np.asarray(nonconf_scores, dtype=np.float64)
        p_vals: List[float] = []
        step_alarms = False
        first_alarm_idx: Optional[int] = None

        for idx, s in enumerate(scores):
            step_res = self.update_observation(float(s))
            p_vals.append(step_res["p_value"])
            if step_res["is_alarm"] and not step_alarms:
                step_alarms = True
                first_alarm_idx = idx

        return {
            "batch_size": len(scores),
            "mean_p_value": round(float(np.mean(p_vals)), 5) if p_vals else 1.0,
            "min_p_value": round(float(np.min(p_vals)), 6) if p_vals else 1.0,
            "log_evidence": round(self.log_evidence, 5),
            "evidence": round(self.evidence, 4),
            "alarm_triggered": self.is_alarm,
            "first_alarm_index_in_batch": first_alarm_idx,
            "total_observations": self.total_observations,
            "p_values": p_vals,
        }

    def reset(self) -> None:
        """Reset the martingale evidence accumulator (M_t = 1.0, log(M_t) = 0.0)."""
        self.log_evidence = 0.0

    def get_state(self) -> Dict[str, Any]:
        """Return the current diagnostic state of the martingale detector."""
        return {
            "log_evidence": round(self.log_evidence, 5),
            "evidence": round(self.evidence, 4),
            "is_alarm": self.is_alarm,
            "alarm_threshold": self.alarm_threshold,
            "log_alarm_threshold": round(self.log_alarm_threshold, 5),
            "alpha_drift": self.alpha_drift,
            "epsilon": self.epsilon,
            "reference_sample_count": self.m,
            "total_observations": self.total_observations,
            "alarm_count": self.alarm_count,
            "last_alarm_observation": self.last_alarm_observation,
        }
