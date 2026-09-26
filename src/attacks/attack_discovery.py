"""FraudForge AI: Novel Synthetic Attack Discovery Engine.

Generates novel synthetic attack candidate genomes via mutation and crossover,
filters them using domain compatibility rules and near-duplicate rejection,
evaluates them against the baseline detector, discovers critical blind spots,
and exports structured experiment artifacts.
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

from src.attacks.attack_library import AttackArchetype
from src.attacks.attack_genome import (
    AttackGenome,
    GENOME_VOCABULARY,
    GENE_KEYS,
    get_all_known_genomes,
    get_archetype_genome,
)
from src.attacks.novelty_engine import (
    check_compatibility,
    calculate_genome_similarity,
    calculate_novelty_score,
    calculate_realism_score,
    calculate_evasion_potential,
    calculate_priority_score,
    generate_candidate_name,
)
from src.detection.predict import FraudDetector
from src.utils.config import (
    DEFAULT_SEED,
    EXPERIMENTS_DIR,
    MODELS_DIR,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NovelAttackCandidate:
    """Represents a discovered novel synthetic attack pattern."""

    candidate_id: str
    candidate_name: str
    genome: AttackGenome
    novelty_score: float
    realism_score: float
    evasion_potential: float
    priority_score: float
    nearest_known_attack: str
    nearest_known_similarity: float
    lineage: Dict[str, Any] = field(default_factory=dict)
    simulation_parameters: Dict[str, Any] = field(default_factory=dict)

    # Evaluation results against detector
    number_tested: int = 0
    number_detected: int = 0
    number_missed: int = 0
    detection_rate: float = 0.0  # Percentage 0.0 to 100.0
    blind_spot_level: str = "UNTESTED"

    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate to serialized dictionary."""
        d = asdict(self)
        d["genome"] = self.genome.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NovelAttackCandidate":
        """Deserialize from dictionary."""
        d = data.copy()
        if isinstance(d.get("genome"), dict):
            d["genome"] = AttackGenome.from_dict(d["genome"])
        return cls(**d)


# =============================================================================
# ATTACK DISCOVERY ENGINE
# =============================================================================

class AttackDiscoveryEngine:
    """Engine for generating, ranking, simulating, and evaluating novel synthetic attacks."""

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        known_genomes: Optional[List[AttackGenome]] = None,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.known_genomes = known_genomes or get_all_known_genomes()

    def mutate_genome(
        self,
        parent: AttackGenome,
        n_mutations: int = 1,
    ) -> Tuple[AttackGenome, List[str]]:
        """Create a mutated candidate by randomly modifying 1 to n genes."""
        genes = parent.get_genes()
        mutated_genes = genes.copy()
        keys_to_mutate = list(self.rng.choice(GENE_KEYS, size=min(n_mutations, len(GENE_KEYS)), replace=False))

        mutated_log: List[str] = []
        for key in keys_to_mutate:
            options = [v for v in GENOME_VOCABULARY[key] if v != genes[key]]
            if options:
                new_val = str(self.rng.choice(options))
                mutated_genes[key] = new_val
                mutated_log.append(f"{key}: {genes[key]} -> {new_val}")

        child = AttackGenome(
            target=mutated_genes["target"],
            entry_vector=mutated_genes["entry_vector"],
            behavior=mutated_genes["behavior"],
            evasion_strategy=mutated_genes["evasion_strategy"],
            payment_channel=mutated_genes["payment_channel"],
            amount_strategy=mutated_genes["amount_strategy"],
            temporal_strategy=mutated_genes["temporal_strategy"],
            identity_strategy=mutated_genes["identity_strategy"],
            merchant_strategy=mutated_genes["merchant_strategy"],
            geographic_strategy=mutated_genes["geographic_strategy"],
            category="Novel Synthetic Attack Candidate",
        )
        return child, mutated_log

    def crossover_genomes(
        self,
        parent1: AttackGenome,
        parent2: AttackGenome,
    ) -> Tuple[AttackGenome, List[str]]:
        """Combine genes from two parent genomes to form a crossover candidate."""
        g1 = parent1.get_genes()
        g2 = parent2.get_genes()
        child_genes: Dict[str, str] = {}
        crossover_log: List[str] = []

        for key in GENE_KEYS:
            if self.rng.random() < 0.50:
                child_genes[key] = g1[key]
                crossover_log.append(f"{key} from {parent1.attack_id or 'Parent1'}")
            else:
                child_genes[key] = g2[key]
                crossover_log.append(f"{key} from {parent2.attack_id or 'Parent2'}")

        child = AttackGenome(
            target=child_genes["target"],
            entry_vector=child_genes["entry_vector"],
            behavior=child_genes["behavior"],
            evasion_strategy=child_genes["evasion_strategy"],
            payment_channel=child_genes["payment_channel"],
            amount_strategy=child_genes["amount_strategy"],
            temporal_strategy=child_genes["temporal_strategy"],
            identity_strategy=child_genes["identity_strategy"],
            merchant_strategy=child_genes["merchant_strategy"],
            geographic_strategy=child_genes["geographic_strategy"],
            category="Novel Synthetic Attack Candidate",
        )
        return child, crossover_log

    def generate_raw_candidates(
        self,
        n_candidates: int = 80,
    ) -> List[Tuple[AttackGenome, Dict[str, Any]]]:
        """Generate raw candidate genomes via mutation and crossover."""
        raw: List[Tuple[AttackGenome, Dict[str, Any]]] = []
        n_known = len(self.known_genomes)

        for i in range(n_candidates):
            # 55% crossover, 45% mutation
            if self.rng.random() < 0.55 and n_known >= 2:
                idx1, idx2 = self.rng.choice(n_known, size=2, replace=False)
                p1, p2 = self.known_genomes[idx1], self.known_genomes[idx2]
                child, log = self.crossover_genomes(p1, p2)
                lineage = {
                    "mutation_type": "crossover",
                    "parent_1": p1.attack_id or p1.attack_name or f"Known-{idx1}",
                    "parent_2": p2.attack_id or p2.attack_name or f"Known-{idx2}",
                    "gene_sources": log,
                }
            else:
                idx = int(self.rng.choice(n_known))
                parent = self.known_genomes[idx]
                n_mut = int(self.rng.choice([1, 2, 3], p=[0.50, 0.35, 0.15]))
                child, log = self.mutate_genome(parent, n_mutations=n_mut)
                lineage = {
                    "mutation_type": "mutation",
                    "parent_1": parent.attack_id or parent.attack_name or f"Known-{idx}",
                    "parent_2": None,
                    "mutations": log,
                }
            raw.append((child, lineage))

        return raw

    def filter_and_rank_candidates(
        self,
        raw_candidates: List[Tuple[AttackGenome, Dict[str, Any]]],
        max_retained: int = 15,
        max_similarity_to_known: float = 0.85,
        pairwise_similarity_threshold: float = 0.90,
    ) -> Tuple[List[NovelAttackCandidate], Dict[str, int]]:
        """Filter candidates for compatibility & duplicates, rank by priority score, and retain top N."""
        stats = {
            "total_generated": len(raw_candidates),
            "rejected_incompatible": 0,
            "rejected_near_duplicate_known": 0,
            "rejected_near_duplicate_candidate": 0,
            "compatible_candidates": 0,
            "retained_candidates": 0,
        }

        scored_pool: List[NovelAttackCandidate] = []

        for genome, lineage in raw_candidates:
            # 1. Compatibility Check
            is_compat, _ = check_compatibility(genome)
            if not is_compat:
                stats["rejected_incompatible"] += 1
                continue

            stats["compatible_candidates"] += 1

            # 2. Novelty & Similarity against Known Archetypes
            novelty, nearest_known, max_sim = calculate_novelty_score(genome, self.known_genomes)
            if max_sim > max_similarity_to_known:
                stats["rejected_near_duplicate_known"] += 1
                continue

            # 3. Realism & Evasion Potential
            realism = calculate_realism_score(genome)
            evasion = calculate_evasion_potential(genome)
            priority = calculate_priority_score(novelty, realism, evasion)

            candidate_name = generate_candidate_name(genome)
            sim_params = genome.to_simulation_parameters()

            nearest_name = (
                f"{nearest_known.attack_id} ({nearest_known.attack_name})"
                if nearest_known and nearest_known.attack_id
                else (nearest_known.attack_name if nearest_known else "Unknown")
            )

            candidate = NovelAttackCandidate(
                candidate_id="",  # Assigned after ranking & deduplication
                candidate_name=candidate_name,
                genome=genome,
                novelty_score=novelty,
                realism_score=realism,
                evasion_potential=evasion,
                priority_score=priority,
                nearest_known_attack=nearest_name,
                nearest_known_similarity=max_sim,
                lineage=lineage,
                simulation_parameters=sim_params,
            )
            scored_pool.append(candidate)

        # 4. Sort candidates by Priority Score descending
        scored_pool.sort(key=lambda c: c.priority_score, reverse=True)

        # 5. Pairwise deduplication among candidates
        retained: List[NovelAttackCandidate] = []
        for cand in scored_pool:
            if len(retained) >= max_retained:
                break

            # Check if too close to an already selected candidate
            is_dup = False
            for prev in retained:
                sim = calculate_genome_similarity(cand.genome, prev.genome)
                if sim >= pairwise_similarity_threshold:
                    is_dup = True
                    break

            if is_dup:
                stats["rejected_near_duplicate_candidate"] += 1
                continue

            # Assign clean ID
            cand_id = f"NSA-{len(retained) + 1:03d}"
            cand.candidate_id = cand_id
            cand.genome.attack_id = cand_id
            cand.genome.attack_name = cand.candidate_name
            retained.append(cand)

        stats["retained_candidates"] = len(retained)
        return retained, stats

    def simulate_candidate_transactions(
        self,
        candidates: List[NovelAttackCandidate],
        samples_per_candidate: int = 200,
        seed: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Generate synthetic transactions for each novel candidate."""
        from src.simulation.transaction_generator import TransactionGenerator

        tx_seed = seed or (self.seed + 999)
        generator = TransactionGenerator(seed=tx_seed)
        datasets: Dict[str, pd.DataFrame] = {}

        for cand in candidates:
            cand_archetype = AttackArchetype(
                attack_id=cand.candidate_id,
                name=cand.candidate_name,
                category="Novel Synthetic Attack Candidate",
                description=f"Synthetic attack generated via {cand.lineage.get('mutation_type', 'mutation')}.",
                severity="CRITICAL" if cand.evasion_potential >= 0.75 else "HIGH",
                novelty_score=cand.novelty_score,
                detectability_score=float(np.round(1.0 - cand.evasion_potential, 2)),
                behavioral_indicators=[f"{k}={v}" for k, v in cand.genome.get_genes().items()],
                affected_payment_surface=cand.genome.payment_channel,
                simulation_parameters=cand.simulation_parameters,
            )

            records: List[Dict[str, Any]] = []
            for _ in range(samples_per_candidate):
                tx = generator.generate_fraud_transaction(archetype=cand_archetype)
                records.append(tx)

            datasets[cand.candidate_id] = pd.DataFrame(records)

        return datasets

    def evaluate_against_detector(
        self,
        detector: FraudDetector,
        candidate_datasets: Dict[str, pd.DataFrame],
        candidates: List[NovelAttackCandidate],
    ) -> List[NovelAttackCandidate]:
        """Test candidate transaction streams against baseline detector and classify blind spots."""
        for cand in candidates:
            df = candidate_datasets.get(cand.candidate_id)
            if df is None or len(df) == 0:
                continue

            preds = detector.predict(df)
            n_tested = len(preds)
            n_detected = int(np.sum(preds == 1))
            n_missed = n_tested - n_detected
            det_rate = float(np.round((n_detected / n_tested) * 100.0, 2))

            # Classify blind spot severity
            if det_rate < 40.0:
                blind_spot = "CRITICAL BLIND SPOT"
            elif det_rate < 70.0:
                blind_spot = "HIGH BLIND SPOT"
            elif det_rate < 90.0:
                blind_spot = "MODERATE BLIND SPOT"
            else:
                blind_spot = "LOW PRIORITY"

            cand.number_tested = n_tested
            cand.number_detected = n_detected
            cand.number_missed = n_missed
            cand.detection_rate = det_rate
            cand.blind_spot_level = blind_spot

        return candidates

    def export_artifacts(
        self,
        candidates: List[NovelAttackCandidate],
        output_dir: Optional[Path] = None,
    ) -> Tuple[Path, Path]:
        """Save structured JSON and summary CSV artifacts."""
        out_dir = output_dir or EXPERIMENTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "novel_attack_candidates.json"
        csv_path = out_dir / "novel_attack_report.csv"

        # 1. JSON Export
        json_data = [cand.to_dict() for cand in candidates]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        # 2. CSV Export
        csv_records: List[Dict[str, Any]] = []
        for cand in candidates:
            csv_records.append({
                "candidate_id": cand.candidate_id,
                "candidate_name": cand.candidate_name,
                "genome": json.dumps(cand.genome.get_genes()),
                "novelty_score": cand.novelty_score,
                "realism_score": cand.realism_score,
                "evasion_potential": cand.evasion_potential,
                "priority_score": cand.priority_score,
                "detection_rate": cand.detection_rate,
                "blind_spot_level": cand.blind_spot_level,
                "nearest_known_attack": cand.nearest_known_attack,
                "nearest_known_similarity": cand.nearest_known_similarity,
            })

        df_csv = pd.DataFrame(csv_records)
        df_csv.to_csv(csv_path, index=False)

        return json_path, csv_path

    def run_discovery(
        self,
        n_raw: int = 80,
        max_retained: int = 15,
        samples_per_candidate: int = 200,
        detector: Optional[FraudDetector] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Execute end-to-end Novel Synthetic Attack Discovery pipeline."""
        if verbose:
            print("=" * 65)
            print("        FRAUDFORGE AI — NOVEL ATTACK DISCOVERY")
            print("=" * 65)

        # 1. Generate & Filter
        raw = self.generate_raw_candidates(n_candidates=n_raw)
        retained, stats = self.filter_and_rank_candidates(raw, max_retained=max_retained)

        if verbose:
            print(f"Known Attack Archetypes:        {len(self.known_genomes)}")
            print(f"Candidate Genomes Generated:    {stats['total_generated']}")
            print(f"Compatible Candidates:          {stats['compatible_candidates']}")
            print(f"Novel Candidates Retained:      {stats['retained_candidates']}")

        # 2. Simulate Synthetic Datasets
        datasets = self.simulate_candidate_transactions(
            retained,
            samples_per_candidate=samples_per_candidate,
        )

        # 3. Load / Use Detector
        eval_detector = detector
        if eval_detector is None:
            model_path = MODELS_DIR / "baseline_detector.joblib"
            if model_path.exists():
                eval_detector = FraudDetector(artifact_path=model_path)
            else:
                raise FileNotFoundError(f"Baseline detector not found at: {model_path}")

        # 4. Evaluate against Detector
        evaluated = self.evaluate_against_detector(eval_detector, datasets, retained)

        # 5. Export Artifacts
        json_path, csv_path = self.export_artifacts(evaluated)

        # 6. Terminal Display
        if verbose:
            print("\n" + "-" * 65)
            print("TOP NOVEL SYNTHETIC ATTACK CANDIDATES")
            print("-" * 65)
            for cand in evaluated:
                print(f"\n{cand.candidate_id}  {cand.candidate_name}")
                print(f"Novelty:           {cand.novelty_score:.2f}")
                print(f"Realism:           {cand.realism_score:.2f}")
                print(f"Evasion Potential: {cand.evasion_potential:.2f}")
                print(f"Priority Score:    {cand.priority_score:.2f}")
                print(f"Detector Detection Rate: {cand.detection_rate:.1f}% ({cand.number_detected}/{cand.number_tested})")
                print(f"Status:\n{cand.blind_spot_level}")
                print(f"Nearest Known:     {cand.nearest_known_attack} (Sim: {cand.nearest_known_similarity:.2f})")
            print("=" * 65)

        return {
            "stats": stats,
            "candidates": [c.to_dict() for c in evaluated],
            "json_path": str(json_path),
            "csv_path": str(csv_path),
        }


def main():
    """CLI entry point for attack discovery runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="FraudForge AI: Novel Synthetic Attack Discovery Engine"
    )
    parser.add_argument(
        "--raw-candidates",
        type=int,
        default=80,
        help="Number of raw candidate genomes to generate (default: 80)",
    )
    parser.add_argument(
        "--retained",
        type=int,
        default=15,
        help="Number of top priority candidates to retain (default: 15)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of synthetic transactions to generate per candidate (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic random seed (default: {DEFAULT_SEED})",
    )

    args = parser.parse_args()

    engine = AttackDiscoveryEngine(seed=args.seed)
    engine.run_discovery(
        n_raw=args.raw_candidates,
        max_retained=args.retained,
        samples_per_candidate=args.samples,
        verbose=True,
    )


if __name__ == "__main__":
    main()
