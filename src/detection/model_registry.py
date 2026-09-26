"""FraudForge AI: Volatile / Session-Only Model Registry & Promotion Engine.

Manages the session lifecycle of ML detectors:
- Immutable Baseline Detector (baseline_detector.joblib) - permanent reference
- Active Operational Detector - session/in-memory instance
- Session Hardened Version History - runtime memory only
- Registry Metadata - initialized fresh on every application startup
"""

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.config import MODELS_DIR
from src.detection.predict import FraudDetector
from src.detection.evaluate import evaluate_global_metrics


class ModelRegistry:
    """Manages active detector resolution, validation gates, and session-only version promotion."""

    BASELINE_FILE = "baseline_detector.joblib"

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.baseline_path = self.models_dir / self.BASELINE_FILE
        
        # Session/volatile state - always initializes fresh on startup
        self.active_version: str = "baseline_v1"
        self.previous_version: Optional[str] = None
        self.hardening_round: int = 0
        self.active_detector: Optional[FraudDetector] = None
        self.last_updated: str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.history: List[Dict[str, Any]] = []
        
        self._reset_session_state()

    def _reset_session_state(self) -> None:
        """Reset in-memory state to baseline_v1 without touching disk."""
        self.active_version = "baseline_v1"
        self.previous_version = None
        self.hardening_round = 0
        self.last_updated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.history = [
            {
                "version": "baseline_v1",
                "model_path": "models/baseline_detector.joblib (permanent immutable baseline)",
                "hardening_round": 0,
                "trained_attack_ids": [],
                "created_at": self.last_updated,
                "description": "Permanent baseline XGBoost detector trained on normal + baseline synthetic transactions",
            }
        ]
        if self.baseline_path.exists():
            self.active_detector = FraudDetector(artifact_path=self.baseline_path)
        else:
            self.active_detector = None

    def get_registry_data(self) -> Dict[str, Any]:
        """Read current session registry state from memory."""
        return {
            "active_version": self.active_version,
            "previous_version": self.previous_version,
            "hardening_round": self.hardening_round,
            "active_model_path": "in-memory (session-only)",
            "baseline_model_path": str(self.baseline_path.name),
            "last_updated": self.last_updated,
            "history": list(self.history),
        }

    def get_active_version(self) -> str:
        """Get the identifier string of the current active detector."""
        return self.active_version

    def get_hardening_round(self) -> int:
        """Get the current session hardening round count."""
        return self.hardening_round

    def load_active_detector(self) -> FraudDetector:
        """Get the current active FraudDetector instance."""
        if self.active_detector is None:
            self.active_detector = self.load_baseline_detector()
        return self.active_detector

    def load_baseline_detector(self) -> FraudDetector:
        """Load and return the permanent immutable baseline FraudDetector instance."""
        return FraudDetector(artifact_path=self.baseline_path)

    def validate_candidate(
        self,
        candidate_detector: FraudDetector,
        target_attack_df: pd.DataFrame,
        normal_test_df: pd.DataFrame,
        baseline_detector: Optional[FraudDetector] = None,
        min_recall_gain: float = 0.0,
        min_normal_recall: float = 0.85,
        max_fpr_increase: float = 0.08,
    ) -> Tuple[bool, Dict[str, Any], str]:
        """Validate candidate hardened detector against promotion gates.
        
        Gates:
        1. Target attack recall must show improvement over baseline (or achieve >= 80%).
        2. Normal test set recall must remain above min_normal_recall (no catastrophic forgetting).
        3. False Positive Rate must not explode beyond max_fpr_increase.
        """
        if baseline_detector is None:
            baseline_detector = self.load_baseline_detector()

        # 1. Target attack evaluation
        cand_attack_probs = candidate_detector.predict_proba(target_attack_df)
        cand_attack_preds = (cand_attack_probs >= 0.50).astype(int)
        cand_attack_recall = float(np.mean(cand_attack_preds == 1)) * 100.0

        base_attack_probs = baseline_detector.predict_proba(target_attack_df)
        base_attack_preds = (base_attack_probs >= 0.50).astype(int)
        base_attack_recall = float(np.mean(base_attack_preds == 1)) * 100.0

        recall_gain = cand_attack_recall - base_attack_recall

        # 2. Normal/held-out test evaluation
        y_normal_true = normal_test_df["fraud_label"].to_numpy()
        cand_norm_preds = candidate_detector.predict(normal_test_df)
        cand_norm_probs = candidate_detector.predict_proba(normal_test_df)
        cand_norm_metrics = evaluate_global_metrics(y_normal_true, cand_norm_preds, cand_norm_probs)

        base_norm_preds = baseline_detector.predict(normal_test_df)
        base_norm_probs = baseline_detector.predict_proba(normal_test_df)
        base_norm_metrics = evaluate_global_metrics(y_normal_true, base_norm_preds, base_norm_probs)

        fpr_diff = cand_norm_metrics["false_positive_rate"] - base_norm_metrics["false_positive_rate"]

        validation_metrics = {
            "target_attack": {
                "baseline_recall_pct": round(base_attack_recall, 2),
                "hardened_recall_pct": round(cand_attack_recall, 2),
                "recall_gain_pct_points": round(recall_gain, 2),
            },
            "normal_test": {
                "baseline_recall": round(base_norm_metrics["recall"], 4),
                "hardened_recall": round(cand_norm_metrics["recall"], 4),
                "baseline_fpr": round(base_norm_metrics["false_positive_rate"], 4),
                "hardened_fpr": round(cand_norm_metrics["false_positive_rate"], 4),
                "fpr_increase": round(fpr_diff, 4),
            },
        }

        # Gate Checks
        if cand_attack_recall < base_attack_recall and cand_attack_recall < 75.0:
            msg = f"Validation Failed: Candidate recall ({cand_attack_recall:.1f}%) did not improve over baseline ({base_attack_recall:.1f}%)."
            return False, validation_metrics, msg

        if cand_norm_metrics["recall"] < min_normal_recall:
            msg = f"Validation Failed: Normal recall ({cand_norm_metrics['recall']*100:.1f}%) degraded below minimum threshold ({min_normal_recall*100:.1f}%)."
            return False, validation_metrics, msg

        if fpr_diff > max_fpr_increase:
            msg = f"Validation Failed: False positive rate increased by {fpr_diff*100:.1f}%, exceeding {max_fpr_increase*100:.1f}% limit."
            return False, validation_metrics, msg

        msg = f"Validation Passed: Attack recall {cand_attack_recall:.1f}% (+{recall_gain:.1f} pts), normal recall {cand_norm_metrics['recall']*100:.1f}%."
        return True, validation_metrics, msg

    def promote_candidate(
        self,
        candidate_artifact: Optional[Dict[str, Any]] = None,
        candidate_detector: Optional[FraudDetector] = None,
        trained_attack_ids: Optional[List[str]] = None,
        validation_metrics: Optional[Dict[str, Any]] = None,
        custom_version_name: Optional[str] = None,
    ) -> Tuple[str, FraudDetector]:
        """Promote candidate artifact/detector IN MEMORY ONLY without saving to disk."""
        if candidate_detector is None:
            if candidate_artifact is None:
                raise ValueError("Either candidate_detector or candidate_artifact must be provided.")
            candidate_detector = FraudDetector(artifact=candidate_artifact)

        current_round = self.hardening_round + 1
        new_version = custom_version_name or f"hardened_v{current_round}"
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Update in-memory session state
        self.previous_version = self.active_version
        self.active_version = new_version
        self.hardening_round = current_round
        self.active_detector = candidate_detector
        self.last_updated = now_str

        history_entry = {
            "version": new_version,
            "model_path": f"in-memory (session round {current_round})",
            "hardening_round": current_round,
            "trained_attack_ids": trained_attack_ids or [],
            "created_at": now_str,
            "validation_metrics": validation_metrics or {},
        }
        self.history.append(history_entry)

        return new_version, candidate_detector

    def reset_to_baseline(self) -> str:
        """Reset session active detector back to immutable baseline."""
        self._reset_session_state()
        return "baseline_v1"
