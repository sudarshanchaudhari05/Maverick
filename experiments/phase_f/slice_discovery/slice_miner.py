"""FraudForge AI: Phase F Worst-Slice Mining Engine.

Implements systematic multi-dimensional slice mining across 17 dimensions (Families S1, S2, S3),
evaluating combination depths 1 to 3 with strict support filtering (>=50 total, >=20 fraud),
calculates comprehensive slice metrics with 95% Wilson score confidence intervals,
and produces the authoritative Top-5 worst-performing slices ranked by FNR descending.
"""

from typing import Dict, List, Tuple, Any, Optional
import itertools
import math
import numpy as np
import pandas as pd

from src.attacks.attack_genome import AttackGenome, KNOWN_ATTACK_GENOMES
from src.attacks.attack_library import get_default_attack_library


# =============================================================================
# 17 CANONICAL SLICE DIMENSIONS (LOCKED SPECIFICATION)
# =============================================================================

FAMILY_S1_DIMENSIONS = [
    "attack_family",
    "attack_archetype",
    "entry/payment_channel",
    "evasion_strategy",
    "amount_strategy",
    "temporal_strategy",
    "identity_strategy",
    "merchant_strategy",
    "geographic_strategy",
]

FAMILY_S2_DIMENSIONS = [
    "amount_bucket",
    "payment_channel",
    "merchant_category",
    "geographic_region_or_strategy",
    "temporal_bucket",
    "identity_strategy",
]

FAMILY_S3_DIMENSIONS = [
    "fraud_probability_bucket",
    "risk_level",
]

# All unique slice dimensions (exactly 17 distinct dimension keys)
ALL_SLICE_DIMENSIONS = list(dict.fromkeys(
    FAMILY_S1_DIMENSIONS + FAMILY_S2_DIMENSIONS + FAMILY_S3_DIMENSIONS
))

# Support thresholds (Locked)
MIN_TOTAL_SUPPORT = 50
MIN_FRAUD_SUPPORT = 20
MAX_SLICE_DEPTH = 3


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute two-sided Wilson score confidence interval for a proportion k / n."""
    if n <= 0:
        return 0.0, 0.0
    z = 1.959963984540054  # 95% two-sided normal quantile
    p = float(k) / float(n)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return float(np.round(lower, 4)), float(np.round(upper, 4))


class SliceFeatureExtractor:
    """Discretizes and enriches transactions into the 17 canonical slice dimensions."""

    def __init__(self):
        self.attack_library = get_default_attack_library()
        # Build mapping from archetype name / attack_id to AttackGenome
        self.name_to_genome: Dict[str, AttackGenome] = {}
        for archetype in self.attack_library.get_all():
            genome = KNOWN_ATTACK_GENOMES.get(archetype.attack_id)
            if genome:
                self.name_to_genome[archetype.name] = genome
                self.name_to_genome[archetype.attack_id] = genome

    @staticmethod
    def discretize_amount(amount: float) -> str:
        if amount < 50.0:
            return "micro_low (<50)"
        elif amount < 200.0:
            return "medium (50-200)"
        elif amount < 800.0:
            return "high (200-800)"
        else:
            return "very_high (>=800)"

    @staticmethod
    def discretize_hour(hour: int) -> str:
        if 0 <= hour < 6:
            return "night_off_hours (00-05)"
        elif 6 <= hour < 12:
            return "morning (06-11)"
        elif 12 <= hour < 18:
            return "afternoon (12-17)"
        else:
            return "evening (18-23)"

    @staticmethod
    def discretize_probability(prob: float) -> str:
        if prob < 0.20:
            return "prob_0.00-0.20"
        elif prob < 0.50:
            return "prob_0.20-0.50"
        elif prob < 0.75:
            return "prob_0.50-0.75"
        else:
            return "prob_0.75-1.00"

    @staticmethod
    def discretize_risk_level(prob: float) -> str:
        if prob >= 0.75:
            return "CRITICAL_RISK"
        elif prob >= 0.50:
            return "HIGH_RISK"
        elif prob >= 0.25:
            return "MODERATE_RISK"
        else:
            return "LOW_RISK"

    def enrich_dataset(
        self,
        df: pd.DataFrame,
        probabilities: np.ndarray,
        predictions: np.ndarray,
    ) -> pd.DataFrame:
        """Add 17 slice dimension columns to dataframe alongside detector predictions."""
        enriched = df.copy()
        enriched["detector_probability"] = probabilities
        enriched["detector_prediction"] = predictions

        # Family S3 — Detector
        enriched["fraud_probability_bucket"] = [
            self.discretize_probability(p) for p in probabilities
        ]
        enriched["risk_level"] = [
            self.discretize_risk_level(p) for p in probabilities
        ]

        # Family S2 — Transaction
        enriched["amount_bucket"] = [
            self.discretize_amount(a) for a in enriched["transaction_amount"]
        ]
        enriched["temporal_bucket"] = [
            self.discretize_hour(int(h)) for h in enriched["transaction_hour"]
        ]

        # Geographic region or strategy
        def get_geo(row):
            is_cross = (
                row.get("geographic_deviation", 0) == 1
                or row.get("transaction_country") != row.get("customer_country")
            )
            return "cross_border" if is_cross else "domestic"

        enriched["geographic_region_or_strategy"] = [
            get_geo(r) for _, r in enriched.iterrows()
        ]

        # Family S1 & Genome mappings
        attack_families = []
        attack_archetypes = []
        entry_channels = []
        evasion_strategies = []
        amount_strategies = []
        temporal_strategies = []
        identity_strategies = []
        merchant_strategies = []
        geographic_strategies = []

        for _, row in enriched.iterrows():
            is_fraud = int(row.get("fraud_label", 0)) == 1
            atk_type = str(row.get("attack_type", "LEGITIMATE"))

            if is_fraud and atk_type in self.name_to_genome:
                g = self.name_to_genome[atk_type]
                attack_families.append(g.category or "UNKNOWN_ATTACK_FAMILY")
                attack_archetypes.append(g.attack_id or atk_type)
                entry_channels.append(f"{g.entry_vector}/{g.payment_channel}")
                evasion_strategies.append(g.evasion_strategy)
                amount_strategies.append(g.amount_strategy)
                temporal_strategies.append(g.temporal_strategy)
                identity_strategies.append(g.identity_strategy)
                merchant_strategies.append(g.merchant_strategy)
                geographic_strategies.append(g.geographic_strategy)
            else:
                # Legitimate / Benign defaults
                attack_families.append("LEGITIMATE")
                attack_archetypes.append("LEGITIMATE")
                entry_channels.append(f"legitimate/{row.get('payment_channel', 'unknown')}")
                evasion_strategies.append("NONE")
                amount_strategies.append("ORGANIC")
                temporal_strategies.append("ORGANIC")
                identity_strategies.append("LEGITIMATE_OWNER")
                merchant_strategies.append("ORGANIC")
                geographic_strategies.append("ORGANIC")

        enriched["attack_family"] = attack_families
        enriched["attack_archetype"] = attack_archetypes
        enriched["entry/payment_channel"] = entry_channels
        enriched["evasion_strategy"] = evasion_strategies
        enriched["amount_strategy"] = amount_strategies
        enriched["temporal_strategy"] = temporal_strategies
        enriched["identity_strategy"] = identity_strategies
        enriched["merchant_strategy"] = merchant_strategies
        enriched["geographic_strategy"] = geographic_strategies

        return enriched


class WorstSliceMiner:
    """Mines and ranks worst-performing slices from an enriched dataset."""

    def __init__(
        self,
        max_depth: int = MAX_SLICE_DEPTH,
        min_total_support: int = MIN_TOTAL_SUPPORT,
        min_fraud_support: int = MIN_FRAUD_SUPPORT,
    ):
        if max_depth > 3:
            raise ValueError(
                f"Protocol violation: Maximum combination depth is 3. Got {max_depth}."
            )
        self.max_depth = max_depth
        self.min_total_support = min_total_support
        self.min_fraud_support = min_fraud_support
        self.extractor = SliceFeatureExtractor()

    def evaluate_slice_metrics(
        self,
        sub_df: pd.DataFrame,
        slice_definition: Dict[str, str],
        depth: int,
    ) -> Dict[str, Any]:
        """Compute all locked metrics for a candidate slice."""
        total_support = len(sub_df)
        y_true = sub_df["fraud_label"].values
        y_pred = sub_df["detector_prediction"].values
        probs = sub_df["detector_probability"].values

        fraud_support = int(np.sum(y_true == 1))
        legit_support = int(np.sum(y_true == 0))

        # Support check
        eligible = (
            total_support >= self.min_total_support
            and fraud_support >= self.min_fraud_support
        )
        status = "ELIGIBLE" if eligible else "INSUFFICIENT_SUPPORT"

        # Confusion Matrix
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        # Rates
        fnr = float(fn / fraud_support) if fraud_support > 0 else 0.0
        recall = float(tp / fraud_support) if fraud_support > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        fpr = float(fp / legit_support) if legit_support > 0 else 0.0

        mean_fraud_prob = float(np.mean(probs)) if len(probs) > 0 else 0.0

        # Risk distribution
        risk_dist = sub_df["risk_level"].value_counts().to_dict()

        # 95% Wilson confidence interval for FNR
        fnr_ci_lower, fnr_ci_upper = compute_wilson_ci(fn, fraud_support)

        return {
            "slice_definition": slice_definition,
            "depth": depth,
            "status": status,
            "eligible": eligible,
            "total_support": total_support,
            "fraud_support": fraud_support,
            "legitimate_support": legit_support,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "FNR": round(fnr, 4),
            "FNR_ci_lower": fnr_ci_lower,
            "FNR_ci_upper": fnr_ci_upper,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "F1": round(f1, 4),
            "FPR": round(fpr, 4),
            "mean_fraud_probability": round(mean_fraud_prob, 4),
            "risk_level_distribution": risk_dist,
        }

    def mine_slices(
        self,
        df: pd.DataFrame,
        probabilities: np.ndarray,
        predictions: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Mine candidate slices across Levels 1, 2, and 3 combinations."""
        enriched = self.extractor.enrich_dataset(df, probabilities, predictions)
        all_candidates: List[Dict[str, Any]] = []

        # Available slice dimensions in enriched dataframe
        available_dims = [d for d in ALL_SLICE_DIMENSIONS if d in enriched.columns]

        # 1. Level 1: Single dimension
        for dim in available_dims:
            for val, group in enriched.groupby(dim, observed=True):
                # Skip trivial all-legitimate or empty slices
                if len(group) < self.min_total_support:
                    continue
                metrics = self.evaluate_slice_metrics(
                    group, {dim: str(val)}, depth=1
                )
                all_candidates.append(metrics)

        # 2. Level 2: Two-dimensional combinations
        if self.max_depth >= 2:
            dim_pairs = list(itertools.combinations(available_dims, 2))
            for d1, d2 in dim_pairs:
                for (v1, v2), group in enriched.groupby([d1, d2], observed=True):
                    if len(group) < self.min_total_support:
                        continue
                    metrics = self.evaluate_slice_metrics(
                        group, {d1: str(v1), d2: str(v2)}, depth=2
                    )
                    all_candidates.append(metrics)

        # 3. Level 3: Three-dimensional combinations
        if self.max_depth >= 3:
            dim_triples = list(itertools.combinations(available_dims, 3))
            for d1, d2, d3 in dim_triples:
                for (v1, v2, v3), group in enriched.groupby([d1, d2, d3], observed=True):
                    if len(group) < self.min_total_support:
                        continue
                    metrics = self.evaluate_slice_metrics(
                        group, {d1: str(v1), d2: str(v2), d3: str(v3)}, depth=3
                    )
                    all_candidates.append(metrics)

        return all_candidates

    def rank_and_select_top_slices(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        targetable_only: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filter eligible slices and rank by:
        1. FNR descending
        2. fraud_support descending
        3. total_support descending

        Args:
            candidates: All mined slice candidates.
            top_k: Number of top slices to select (default 5).
            targetable_only: If True, selects slices composed solely of transaction and
                attack/genome dimensions (Families S1 & S2), ensuring every dimension can
                serve as a valid generative target constraint under Section 11/18 rules.

        Returns:
            (top_k_slices, all_ranked_eligible_slices)
        """
        # Filter eligible only
        eligible_slices = [c for c in candidates if c["eligible"]]

        if targetable_only:
            s3_keys = set(FAMILY_S3_DIMENSIONS)
            eligible_slices = [
                c for c in eligible_slices
                if not any(k in s3_keys for k in c["slice_definition"].keys())
            ]

        # Sort with locked criteria
        ranked = sorted(
            eligible_slices,
            key=lambda x: (x["FNR"], x["fraud_support"], x["total_support"]),
            reverse=True,
        )

        # Assign persistent IDs F-SLICE-001 .. F-SLICE-005
        top_slices = []
        for i, sl in enumerate(ranked[:top_k], start=1):
            sl_copy = dict(sl)
            sl_copy["slice_id"] = f"F-SLICE-{i:03d}"
            top_slices.append(sl_copy)

        return top_slices, ranked
