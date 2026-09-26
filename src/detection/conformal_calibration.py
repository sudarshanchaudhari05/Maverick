"""FraudForge AI: Post-Hoc Split Conformal Calibration Engine.

Provides distribution-free conformal prediction sets for binary fraud detection:
- Computes nonconformity scores on held-out calibration transactions.
- Derives finite-sample empirical quantile q_hat at significance level alpha.
- Constructs formal prediction sets: {0}, {1}, {0, 1} (ambiguous), or empty set.
- Quantifies decision uncertainty post-hoc over the active FraudDetector.

IMPORTANT STATISTICAL DISCLAIMER:
Marginal coverage guarantees (P(Y in C(X)) >= 1 - alpha) depend strictly on the
exchangeability (i.i.d.) assumption between calibration and test distributions.
Ordinary conformal prediction does not inherently guarantee unconditional false-negative
bounds under adversarial distribution shift.
"""

from dataclasses import dataclass, asdict
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd

from src.detection.predict import FraudDetector
from src.utils.config import (
    DEFAULT_SEED,
    MODELS_DIR,
    GENERATED_DATA_DIR,
    PROCESSED_DATA_DIR,
)


@dataclass
class ConformalPredictionSet:
    """Prediction set and conformal uncertainty scores for a single transaction."""
    transaction_id: str
    prediction_set: List[int]       # e.g. [0], [1], [0, 1], or []
    set_size: int                   # 0, 1, or 2
    is_singleton: bool              # True if set_size == 1
    is_ambiguous: bool              # True if set_size == 2
    is_empty: bool                  # True if set_size == 0
    fraud_probability: float        # raw model probability f_hat(x)
    nonconformity_0: float          # nonconformity score if true label were 0
    nonconformity_1: float          # nonconformity score if true label were 1
    p_value_0: float                # empirical p-value for hypothesis Y=0
    p_value_1: float                # empirical p-value for hypothesis Y=1
    threshold_used: float           # conformal quantile q_hat
    significance_level: float       # alpha

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationState:
    """Summary state of the conformal calibrator."""
    is_calibrated: bool
    target_alpha: float
    nominal_coverage: float
    q_hat: float
    n_calibration_samples: int
    calibrated_at: str
    model_version: str
    exchangeability_assumption: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def normalize_transaction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required pipeline features exist with valid default values."""
    df_out = df.copy()
    alias_map = {
        "amount": "transaction_amount",
        "hour": "transaction_hour",
        "hour_of_day": "transaction_hour",
        "velocity_1h": "transaction_velocity_1h",
        "velocity_24h": "transaction_velocity_24h",
    }
    for alias, target in alias_map.items():
        if alias in df_out.columns and target not in df_out.columns:
            df_out[target] = df_out[alias]

    defaults_num = {
        "transaction_amount": 100.0,
        "transaction_hour": 14,
        "account_age_days": 365.0,
        "device_age_days": 180.0,
        "device_change": 0,
        "IP_risk_score": 0.10,
        "merchant_risk_score": 0.10,
        "transaction_velocity_1h": 1,
        "transaction_velocity_24h": 2,
        "average_customer_amount": 100.0,
        "amount_deviation": 0.0,
        "geographic_deviation": 0.0,
        "behavioral_deviation": 0.0,
        "failed_authentication_count": 0,
        "identity_risk_score": 0.10,
    }
    for col, default_val in defaults_num.items():
        if col not in df_out.columns:
            df_out[col] = default_val

    defaults_cat = {
        "merchant_category": "retail",
        "payment_channel": "p2p_transfer",
        "authentication_method": "pin",
        "transaction_country": "IND",
        "customer_country": "IND",
    }
    for col, default_val in defaults_cat.items():
        if col not in df_out.columns:
            df_out[col] = default_val

    return df_out


class ConformalCalibrator:
    """Post-hoc split conformal calibration engine for payment fraud detection."""

    def __init__(
        self,
        detector: Optional[FraudDetector] = None,
        default_alpha: float = 0.05,
    ):
        self.detector = detector
        self.default_alpha = default_alpha
        self.q_hat: Optional[float] = None
        self.cal_scores: Optional[np.ndarray] = None
        self.target_alpha: float = default_alpha
        self.n_cal_samples: int = 0
        self.calibrated_at: str = ""
        self.model_version: str = "unknown"
        self._exchangeability_notice: str = (
            "Guarantees hold under the assumption of exchangeable (i.i.d.) calibration and test distributions. "
            "Distribution shifts and adversarial mutations may cause empirical coverage to deviate from nominal."
        )

    @property
    def is_calibrated(self) -> bool:
        """Check if calibrator has computed q_hat."""
        return self.q_hat is not None

    def calibrate(
        self,
        calibration_df: pd.DataFrame,
        target_alpha: Optional[float] = None,
        detector: Optional[FraudDetector] = None,
        model_version: str = "baseline_v1",
    ) -> CalibrationState:
        """Calibrate nonconformity threshold q_hat on a held-out calibration dataset."""
        active_detector = detector or self.detector
        if active_detector is None:
            raise RuntimeError("No FraudDetector available for conformal calibration.")

        alpha = float(target_alpha if target_alpha is not None else self.default_alpha)
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")

        if "fraud_label" not in calibration_df.columns:
            raise ValueError("Calibration dataset must contain 'fraud_label' column.")

        y_true = calibration_df["fraud_label"].to_numpy().astype(int)
        df_norm = normalize_transaction_dataframe(calibration_df)
        probs = active_detector.predict_proba(df_norm)

        # Nonconformity score: s_i = 1 - P_hat(Y = y_i | X_i)
        # If y_i = 1: s_i = 1 - probs[i]
        # If y_i = 0: s_i = probs[i] (which is 1 - (1 - probs[i]))
        scores = np.where(y_true == 1, 1.0 - probs, probs)
        scores = np.asarray(scores, dtype=np.float64)

        n = len(scores)
        if n == 0:
            raise ValueError("Calibration dataset cannot be empty.")

        # Finite-sample adjusted conformal quantile:
        # q_level = ceil((n + 1) * (1 - alpha)) / n, clipped to 1.0
        q_level = float(np.clip(np.ceil((n + 1) * (1.0 - alpha)) / n, 0.0, 1.0))
        q_val = float(np.quantile(scores, q_level, method="higher" if hasattr(np, "quantile") else "linear"))

        self.q_hat = round(q_val, 5)
        self.cal_scores = np.sort(scores)
        self.target_alpha = alpha
        self.n_cal_samples = n
        self.calibrated_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.model_version = model_version

        return self.get_state()

    def get_state(self) -> CalibrationState:
        """Return the current calibration state."""
        return CalibrationState(
            is_calibrated=self.q_hat is not None,
            target_alpha=self.target_alpha,
            nominal_coverage=round(1.0 - self.target_alpha, 4),
            q_hat=self.q_hat if self.q_hat is not None else 0.0,
            n_calibration_samples=self.n_cal_samples,
            calibrated_at=self.calibrated_at,
            model_version=self.model_version,
            exchangeability_assumption=self._exchangeability_notice,
        )

    def _calculate_p_values(self, nonconf_0: float, nonconf_1: float) -> Tuple[float, float]:
        """Compute empirical conformal p-values for y=0 and y=1."""
        if self.cal_scores is None or len(self.cal_scores) == 0:
            return (1.0 - nonconf_0, 1.0 - nonconf_1)
        n = len(self.cal_scores)
        # Proportion of calibration scores >= nonconformity score
        p0 = float((np.sum(self.cal_scores >= nonconf_0) + 1.0) / (n + 1.0))
        p1 = float((np.sum(self.cal_scores >= nonconf_1) + 1.0) / (n + 1.0))
        return round(float(np.clip(p0, 0.0, 1.0)), 4), round(float(np.clip(p1, 0.0, 1.0)), 4)

    def predict_single(
        self,
        transaction: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        detector: Optional[FraudDetector] = None,
        alpha_override: Optional[float] = None,
    ) -> ConformalPredictionSet:
        """Construct a conformal prediction set for a single candidate transaction."""
        active_detector = detector or self.detector
        if active_detector is None:
            raise RuntimeError("No FraudDetector available for conformal prediction.")

        if self.q_hat is None:
            raise RuntimeError("ConformalCalibrator has not been calibrated. Call calibrate() first.")

        if isinstance(transaction, pd.Series):
            tx_dict = transaction.to_dict()
        elif isinstance(transaction, pd.DataFrame):
            tx_dict = transaction.iloc[0].to_dict()
        elif isinstance(transaction, dict):
            tx_dict = transaction.copy()
        else:
            tx_dict = dict(transaction)

        if "fraud_probability" in tx_dict and tx_dict["fraud_probability"] is not None:
            prob = float(tx_dict["fraud_probability"])
        else:
            df_single = pd.DataFrame([tx_dict])
            df_norm = normalize_transaction_dataframe(df_single)
            prob = float(active_detector.predict_proba(df_norm)[0])
        tx_id = str(tx_dict.get("transaction_id", "TX-PREDICT-001"))

        # Determine quantile to use
        q = self.q_hat
        alpha = self.target_alpha
        if alpha_override is not None and self.cal_scores is not None:
            alpha = float(alpha_override)
            n = len(self.cal_scores)
            q_level = float(np.clip(np.ceil((n + 1) * (1.0 - alpha)) / n, 0.0, 1.0))
            q = float(np.quantile(self.cal_scores, q_level))

        # Nonconformity scores for candidate labels:
        # If true label were 0 (legitimate), nonconformity is prob (fraud probability)
        # If true label were 1 (fraud), nonconformity is 1 - prob (legitimate probability)
        s0 = float(prob)
        s1 = float(1.0 - prob)

        # Form prediction set: include label if nonconformity <= q
        pred_set = []
        if s0 <= q:
            pred_set.append(0)
        if s1 <= q:
            pred_set.append(1)

        p0, p1 = self._calculate_p_values(s0, s1)
        size = len(pred_set)

        return ConformalPredictionSet(
            transaction_id=tx_id,
            prediction_set=pred_set,
            set_size=size,
            is_singleton=(size == 1),
            is_ambiguous=(size == 2),
            is_empty=(size == 0),
            fraud_probability=round(prob, 4),
            nonconformity_0=round(s0, 4),
            nonconformity_1=round(s1, 4),
            p_value_0=p0,
            p_value_1=p1,
            threshold_used=round(q, 4),
            significance_level=round(alpha, 4),
        )

    def predict_batch(
        self,
        df: pd.DataFrame,
        detector: Optional[FraudDetector] = None,
        alpha_override: Optional[float] = None,
    ) -> List[ConformalPredictionSet]:
        """Construct conformal prediction sets for a DataFrame batch."""
        active_detector = detector or self.detector
        if active_detector is None:
            raise RuntimeError("No FraudDetector available for conformal prediction.")

        df_norm = normalize_transaction_dataframe(df)
        if "fraud_probability" in df.columns and df["fraud_probability"].notnull().all():
            probs = df["fraud_probability"].to_numpy(dtype=float)
        else:
            probs = active_detector.predict_proba(df_norm)
        q = self.q_hat
        alpha = self.target_alpha
        if alpha_override is not None and self.cal_scores is not None:
            alpha = float(alpha_override)
            n = len(self.cal_scores)
            q_level = float(np.clip(np.ceil((n + 1) * (1.0 - alpha)) / n, 0.0, 1.0))
            q = float(np.quantile(self.cal_scores, q_level))

        results: List[ConformalPredictionSet] = []
        records = df.to_dict(orient="records")

        for i, row in enumerate(records):
            prob = float(probs[i])
            tx_id = str(row.get("transaction_id", f"TX-BATCH-{i+1:04d}"))
            s0 = float(prob)
            s1 = float(1.0 - prob)

            pred_set = []
            if s0 <= q:
                pred_set.append(0)
            if s1 <= q:
                pred_set.append(1)

            p0, p1 = self._calculate_p_values(s0, s1)
            size = len(pred_set)

            results.append(
                ConformalPredictionSet(
                    transaction_id=tx_id,
                    prediction_set=pred_set,
                    set_size=size,
                    is_singleton=(size == 1),
                    is_ambiguous=(size == 2),
                    is_empty=(size == 0),
                    fraud_probability=round(prob, 4),
                    nonconformity_0=round(s0, 4),
                    nonconformity_1=round(s1, 4),
                    p_value_0=p0,
                    p_value_1=p1,
                    threshold_used=round(q, 4),
                    significance_level=round(alpha, 4),
                )
            )

        return results


def get_default_calibration_dataset(n_samples: int = 1000, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Load or generate standard calibration data from project artifacts."""
    # 1. Prefer existing 10k dataset split
    path_10k = GENERATED_DATA_DIR / "synthetic_transactions_10k.csv"
    if path_10k.exists():
        df = pd.read_csv(path_10k)
        return df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)

    # 2. Fall back to processed train split
    path_train = PROCESSED_DATA_DIR / "train_split.csv"
    if path_train.exists():
        df = pd.read_csv(path_train)
        return df

    # 3. Fall back to minimal synthetic generation if needed
    from src.simulation.transaction_generator import TransactionGenerator
    gen = TransactionGenerator(seed=seed)
    return gen.generate_dataset(n_samples=n_samples, fraud_ratio=0.15)
