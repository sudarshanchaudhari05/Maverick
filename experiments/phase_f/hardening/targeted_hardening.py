"""FraudForge AI: Phase F Targeted Adversarial Hardening Module.

Constructs hardened training population from:
    EXISTING BASELINE TRAINING DATA (Legitimate + Fraud)
    +
    F-D3 TARGETED ADVERSARIAL SAMPLES
and trains hardened XGBoost classifier using locked architecture and hyperparameters.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.features.feature_engineering import FraudFeaturePipeline, extract_features_and_targets
from src.simulation.transaction_generator import TransactionGenerator
from src.detection.predict import FraudDetector
from src.utils.config import DEFAULT_SEED


class TargetedHardeningTrainer:
    """Trains hardened detector using baseline training data + targeted adversarial samples."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.08,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        seed: int = 6003,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.seed = seed

    def train_hardened_detector(
        self,
        fd3_dataset: pd.DataFrame,
        base_training_samples: int = 5000,
        base_seed: int = DEFAULT_SEED,
    ) -> Tuple[FraudDetector, Dict[str, Any], pd.DataFrame]:
        """Train hardened XGBoost detector on combined baseline + F-D3 dataset."""
        # 1. Generate baseline training population
        gen_base = TransactionGenerator(seed=base_seed)
        df_base = gen_base.generate_dataset(n_samples=base_training_samples, fraud_ratio=0.15)

        # 2. Combine with F-D3 targeted adversarial samples
        df_hardened = pd.concat([df_base, fd3_dataset], ignore_index=True)
        # Shuffle deterministically
        df_hardened = df_hardened.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

        # 3. Fit Feature Pipeline
        X_train, y_train, _ = extract_features_and_targets(df_hardened)
        pipeline = FraudFeaturePipeline()
        X_trans = pipeline.fit_transform(X_train, y_train)

        # 4. Class Imbalance Scale
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = float(n_neg / max(1, n_pos))

        # 5. Train XGBoost model
        model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=self.seed,
            eval_metric="logloss",
        )
        model.fit(X_trans, y_train)

        artifact = {
            "pipeline": pipeline,
            "model": model,
            "model_type": "xgboost",
            "feature_names": pipeline.get_feature_names_out(),
            "seed": self.seed,
            "scale_pos_weight": scale_pos_weight,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        }

        metadata = {
            "baseline_training_samples": len(df_base),
            "baseline_fraud_count": int(df_base["fraud_label"].sum()),
            "fd3_samples": len(fd3_dataset),
            "fd3_fraud_count": int(fd3_dataset["fraud_label"].sum()),
            "total_hardened_train_samples": len(df_hardened),
            "total_hardened_fraud_count": int(y_train.sum()),
            "scale_pos_weight": round(scale_pos_weight, 4),
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "random_state": self.seed,
            },
        }

        detector = FraudDetector(artifact=artifact)
        return detector, metadata, df_hardened

    @staticmethod
    def save_hardened_model(detector: FraudDetector, save_path: Path) -> None:
        """Persist hardened detector artifact to disk."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(detector.artifact, save_path)
