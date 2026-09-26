"""FraudForge AI: Phase F Targeted Attack & Dataset Generation Module.

Implements:
- F2: Random Attack Baseline Dataset (F-D2, Seed 6002, N=2000, 15% fraud)
- F3: Targeted Adversarial Hardening Dataset (F-D3, Seed 6003, N=2500, 50% fraud)
- F5: Unseen Targeted V2 Generalization Dataset (F-D4, Seed 6004, N=2000, 15% fraud)

Enforces strict generator independence: generator NEVER branches on or inspects
fraud_label, is_fraud, detector_prediction, detector_probability, or target labels.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from src.attacks.attack_library import get_default_attack_library, AttackArchetype
from src.attacks.attack_genome import AttackGenome, KNOWN_ATTACK_GENOMES
from src.simulation.transaction_generator import TransactionGenerator
from src.simulation.distributions import (
    CATEGORY_AMOUNT_PARAMS,
    CATEGORY_BASE_RISK,
    COUNTRY_WEIGHTS,
    clip_score,
)
from src.utils.config import ALL_COLUMNS


class PhaseFDatasetGenerator:
    """Orchestrates dataset generation for Experiments F2, F3, and F5."""

    def __init__(self):
        self.attack_library = get_default_attack_library()
        self.all_archetypes = self.attack_library.get_all()

    def generate_fd1_discovery(
        self,
        seed: int = 6001,
        n_samples: int = 4000,
        fraud_ratio: float = 0.15,
    ) -> pd.DataFrame:
        """Generate F-D1 Discovery Dataset (Seed 6001, N=4000, 15% fraud)."""
        gen = TransactionGenerator(seed=seed)
        df = gen.generate_dataset(n_samples=n_samples, fraud_ratio=fraud_ratio, shuffle=True)
        return df[ALL_COLUMNS]

    def generate_fd2_random_baseline(
        self,
        seed: int = 6002,
        n_samples: int = 2000,
        fraud_ratio: float = 0.15,
    ) -> pd.DataFrame:
        """Generate F-D2 Random Attack Baseline (Seed 6002, N=2000, 15% fraud).

        Strictly independent: receives NO knowledge of Top-5 slices or F1 ranking.
        """
        gen = TransactionGenerator(seed=seed)
        df = gen.generate_dataset(n_samples=n_samples, fraud_ratio=fraud_ratio, shuffle=True)
        return df[ALL_COLUMNS]

    def _sample_constrained_transaction(
        self,
        rng: np.random.Generator,
        base_gen: TransactionGenerator,
        slice_def: Dict[str, str],
        is_fraud: bool,
        intensified: bool = False,
    ) -> Dict[str, Any]:
        """Synthesize a single transaction matching slice constraints without detector inspection."""
        # 1. Select Archetype if fraud
        archetype: Optional[AttackArchetype] = None
        if is_fraud:
            # Find archetypes matching slice genome constraints if any
            matching_archetypes = []
            for arc in self.all_archetypes:
                g = KNOWN_ATTACK_GENOMES.get(arc.attack_id)
                if not g:
                    continue
                match = True
                if "attack_family" in slice_def and g.category != slice_def["attack_family"]:
                    match = False
                if "attack_archetype" in slice_def and arc.attack_id != slice_def["attack_archetype"]:
                    match = False
                if "evasion_strategy" in slice_def and g.evasion_strategy != slice_def["evasion_strategy"]:
                    match = False
                if "amount_strategy" in slice_def and g.amount_strategy != slice_def["amount_strategy"]:
                    match = False
                if "temporal_strategy" in slice_def and g.temporal_strategy != slice_def["temporal_strategy"]:
                    match = False
                if "identity_strategy" in slice_def and g.identity_strategy != slice_def["identity_strategy"]:
                    match = False
                if "merchant_strategy" in slice_def and g.merchant_strategy != slice_def["merchant_strategy"]:
                    match = False
                if "geographic_strategy" in slice_def and g.geographic_strategy != slice_def["geographic_strategy"]:
                    match = False
                if match:
                    matching_archetypes.append(arc)

            if matching_archetypes:
                archetype = rng.choice(matching_archetypes)
            else:
                archetype = rng.choice(self.all_archetypes)

            tx = base_gen.generate_fraud_transaction(archetype=archetype)
        else:
            tx = base_gen.generate_legitimate_transaction()

        # 2. Apply Slice Constraints Deterministically
        # Amount bucket constraint
        if "amount_bucket" in slice_def:
            ab = slice_def["amount_bucket"]
            if "micro_low" in ab or "<50" in ab:
                tx["transaction_amount"] = float(np.round(rng.uniform(5.0, 48.5), 2))
            elif "medium" in ab or "50-200" in ab:
                tx["transaction_amount"] = float(np.round(rng.uniform(52.0, 195.0), 2))
            elif "high" in ab or "200-800" in ab:
                tx["transaction_amount"] = float(np.round(rng.uniform(210.0, 785.0), 2))
            elif "very_high" in ab or ">=800" in ab:
                tx["transaction_amount"] = float(np.round(rng.uniform(820.0, 2950.0), 2))
            # Recalculate amount deviation
            avg_amt = tx.get("average_customer_amount", 50.0)
            tx["amount_deviation"] = float(np.round((tx["transaction_amount"] - avg_amt) / avg_amt, 4))

        # Payment channel constraint
        if "payment_channel" in slice_def:
            tx["payment_channel"] = slice_def["payment_channel"]

        # Merchant category constraint
        if "merchant_category" in slice_def:
            tx["merchant_category"] = slice_def["merchant_category"]

        # Temporal bucket constraint
        if "temporal_bucket" in slice_def:
            tb = slice_def["temporal_bucket"]
            if "night_off_hours" in tb or "00-05" in tb:
                tx["transaction_hour"] = int(rng.integers(0, 6))
            elif "morning" in tb or "06-11" in tb:
                tx["transaction_hour"] = int(rng.integers(6, 12))
            elif "afternoon" in tb or "12-17" in tb:
                tx["transaction_hour"] = int(rng.integers(12, 18))
            elif "evening" in tb or "18-23" in tb:
                tx["transaction_hour"] = int(rng.integers(18, 24))

        # Geographic constraint
        if "geographic_region_or_strategy" in slice_def or "geographic_strategy" in slice_def:
            geo = slice_def.get("geographic_region_or_strategy") or slice_def.get("geographic_strategy")
            if geo in ["domestic", "domestic_matching"]:
                tx["transaction_country"] = tx["customer_country"]
                tx["geographic_deviation"] = 0
            elif geo in ["cross_border", "cross_border_arbitrage"]:
                other_countries = [c for c in COUNTRY_WEIGHTS if c != tx["customer_country"]]
                tx["transaction_country"] = str(rng.choice(other_countries))
                tx["geographic_deviation"] = 1

        # Intensified mutation for F5 Unseen V2
        if intensified:
            # Add behavioral variance while preserving slice constraints
            if is_fraud:
                # Perturb IP risk and behavioral deviation subtly to test generalization
                tx["IP_risk_score"] = float(np.round(clip_score(tx["IP_risk_score"] + rng.normal(0, 0.05)), 4))
                tx["behavioral_deviation"] = float(np.round(clip_score(tx["behavioral_deviation"] + rng.normal(0, 0.06)), 4))
                tx["identity_risk_score"] = float(np.round(clip_score(tx["identity_risk_score"] + rng.normal(0, 0.04)), 4))
                # Slight velocity variation
                v1h_jitter = int(rng.choice([-1, 0, 1]))
                tx["transaction_velocity_1h"] = max(1, tx["transaction_velocity_1h"] + v1h_jitter)
                tx["transaction_velocity_24h"] = max(tx["transaction_velocity_1h"], tx["transaction_velocity_24h"] + int(rng.choice([0, 1, 2])))

        return tx

    def generate_fd3_targeted_hardening(
        self,
        top_slices: List[Dict[str, Any]],
        seed: int = 6003,
        total_samples: int = 2500,
        fraud_ratio: float = 0.50,
    ) -> pd.DataFrame:
        """Generate F-D3 Targeted Adversarial Hardening Set (Seed 6003, N=2500, 50% fraud).

        Produces 500 samples per Top-5 slice (250 fraud + 250 legitimate per slice).
        5 x 500 = 2,500 total samples.
        Note: 50% fraud is an adversarial hardening distribution, not realistic payment traffic.
        """
        rng = np.random.default_rng(seed)
        base_gen = TransactionGenerator(seed=seed)
        n_slices = len(top_slices)
        samples_per_slice = total_samples // n_slices
        fraud_per_slice = int(samples_per_slice * fraud_ratio)
        legit_per_slice = samples_per_slice - fraud_per_slice

        records: List[Dict[str, Any]] = []

        for slice_info in top_slices:
            slice_def = slice_info["slice_definition"]

            # Generate Legitimate samples with slice constraints
            for _ in range(legit_per_slice):
                tx = self._sample_constrained_transaction(
                    rng=rng,
                    base_gen=base_gen,
                    slice_def=slice_def,
                    is_fraud=False,
                    intensified=False,
                )
                records.append(tx)

            # Generate Fraud samples with slice constraints
            for _ in range(fraud_per_slice):
                tx = self._sample_constrained_transaction(
                    rng=rng,
                    base_gen=base_gen,
                    slice_def=slice_def,
                    is_fraud=True,
                    intensified=False,
                )
                records.append(tx)

        df = pd.DataFrame.from_records(records)
        df = df[ALL_COLUMNS]
        # Shuffle deterministically
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return df

    def generate_fd4_unseen_v2_test(
        self,
        top_slices: List[Dict[str, Any]],
        seed: int = 6004,
        total_samples: int = 2000,
        fraud_ratio: float = 0.15,
    ) -> pd.DataFrame:
        """Generate F-D4 Unseen Targeted V2 Test (Seed 6004, N=2000, 15% fraud = 300 fraud, 1700 legit).

        Targets discovered weak-slice characteristics using independent seed, independent
        transaction samples, higher mutation intensity, and diverse genomes.
        Evaluation-only: MUST NEVER be used during training, tuning, or threshold selection.
        """
        rng = np.random.default_rng(seed)
        base_gen = TransactionGenerator(seed=seed)
        n_slices = len(top_slices)

        total_fraud = int(total_samples * fraud_ratio)
        total_legit = total_samples - total_fraud

        fraud_per_slice = total_fraud // n_slices
        legit_per_slice = total_legit // n_slices

        records: List[Dict[str, Any]] = []

        for i, slice_info in enumerate(top_slices):
            slice_def = slice_info["slice_definition"]
            # Handle remainder on last slice
            n_f = fraud_per_slice + (total_fraud % n_slices if i == n_slices - 1 else 0)
            n_l = legit_per_slice + (total_legit % n_slices if i == n_slices - 1 else 0)

            for _ in range(n_l):
                tx = self._sample_constrained_transaction(
                    rng=rng,
                    base_gen=base_gen,
                    slice_def=slice_def,
                    is_fraud=False,
                    intensified=True,
                )
                records.append(tx)

            for _ in range(n_f):
                tx = self._sample_constrained_transaction(
                    rng=rng,
                    base_gen=base_gen,
                    slice_def=slice_def,
                    is_fraud=True,
                    intensified=True,
                )
                records.append(tx)

        df = pd.DataFrame.from_records(records)
        df = df[ALL_COLUMNS]
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return df
