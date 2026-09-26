"""FraudForge AI: Inference and Scoring Engine.

Loads trained detector artifacts and provides prediction and fraud scoring APIs
for single transactions or batch DataFrames.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import joblib
import numpy as np
import pandas as pd

from src.features.feature_engineering import FraudFeaturePipeline, extract_features_and_targets
from src.utils.config import MODELS_DIR


class FraudDetector:
    """Inference wrapper for trained fraud detection models."""

    def __init__(
        self,
        artifact_path: Optional[Union[str, Path]] = None,
        artifact: Optional[Dict[str, Any]] = None,
        threshold: float = 0.50,
    ):
        self.threshold = threshold
        self.pipeline: Optional[FraudFeaturePipeline] = None
        self.model = None
        self.model_type: str = ""
        self.feature_names: List[str] = []

        if artifact is not None:
            self._load_from_dict(artifact)
        elif artifact_path is not None:
            self.load(Path(artifact_path))

    def _load_from_dict(self, artifact: Dict[str, Any]) -> None:
        self.pipeline = artifact["pipeline"]
        self.model = artifact["model"]
        self.model_type = artifact.get("model_type", "unknown")
        self.feature_names = artifact.get("feature_names", [])

    @property
    def artifact(self) -> Dict[str, Any]:
        """Return serialized dictionary representation of model artifact."""
        return {
            "pipeline": self.pipeline,
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
        }

    def load(self, filepath: Path) -> "FraudDetector":
        """Load detector artifact from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Model artifact not found at: {filepath}")
        artifact = joblib.load(filepath)
        self._load_from_dict(artifact)
        return self

    def _ensure_loaded(self) -> None:
        if self.pipeline is None or self.model is None:
            raise RuntimeError("FraudDetector model is not loaded. Call load() or pass artifact.")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate fraud probability score (0.0 to 1.0) for each transaction row."""
        self._ensure_loaded()
        X, _, _ = extract_features_and_targets(df)
        X_trans = self.pipeline.transform(X)
        probs = self.model.predict_proba(X_trans)[:, 1]
        return np.asarray(probs, dtype=np.float64)

    def predict(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Predict binary class (0: legitimate, 1: fraud) with configurable threshold."""
        t = self.threshold if threshold is None else threshold
        probs = self.predict_proba(df)
        return (probs >= t).astype(int)

    def score_transaction(
        self,
        tx: Dict[str, Any],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Score a single transaction dictionary."""
        df_single = pd.DataFrame([tx])
        prob = float(self.predict_proba(df_single)[0])
        t = self.threshold if threshold is None else threshold
        is_fraud = bool(prob >= t)

        if prob >= 0.75:
            tier = "CRITICAL_RISK"
        elif prob >= 0.50:
            tier = "HIGH_RISK"
        elif prob >= 0.25:
            tier = "MODERATE_RISK"
        else:
            tier = "LOW_RISK"

        return {
            "fraud_probability": round(prob, 4),
            "is_fraud": is_fraud,
            "decision": "DECLINE" if is_fraud else "APPROVE",
            "risk_tier": tier,
            "threshold_used": t,
        }

    def get_feature_importances(self, top_n: int = 15) -> pd.DataFrame:
        """Extract sorted feature importances from underlying model."""
        self._ensure_loaded()
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feat_df = pd.DataFrame({
                "feature": self.feature_names,
                "importance": importances,
            }).sort_values(by="importance", ascending=False).reset_index(drop=True)
            return feat_df.head(top_n)
        return pd.DataFrame(columns=["feature", "importance"])
