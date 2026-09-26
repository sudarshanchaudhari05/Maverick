"""Novel Attack Discovery & Blind-Spot Evaluation routes."""

import random
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Request
import pandas as pd
import numpy as np

from src.api.schemas import (
    GenerateCandidateRequest,
    GenerateCandidateResponse,
    NovelCandidateScores,
    LineageInfo,
    EvaluateCandidateRequest,
    EvaluateCandidateResponse,
    SampleTransactionView,
)
from src.attacks.attack_genome import (
    AttackGenome,
    get_all_known_genomes,
    KNOWN_ATTACK_GENOMES,
)
from src.attacks.attack_library import AttackArchetype, get_default_attack_library
from src.attacks.novelty_engine import (
    calculate_novelty_score,
    calculate_realism_score,
    calculate_evasion_potential,
    calculate_priority_score,
    generate_candidate_name,
)
from src.attacks.attack_discovery import AttackDiscoveryEngine
from src.simulation.transaction_generator import TransactionGenerator
from src.detection.model_registry import ModelRegistry

router = APIRouter(prefix="/discovery", tags=["Novel Attack Discovery"])


@router.post("/generate-candidate", response_model=GenerateCandidateResponse)
def generate_candidate(req: GenerateCandidateRequest) -> GenerateCandidateResponse:
    """Generate and score a novel synthetic attack candidate via mutation or crossover."""
    seed = req.seed if req.seed is not None else random.randint(1000, 99999)
    rng = random.Random(seed)

    lib = get_default_attack_library()
    known_genomes_dict = KNOWN_ATTACK_GENOMES
    known_genomes_list = get_all_known_genomes()
    engine = AttackDiscoveryEngine(seed=seed)

    if req.custom_genome is not None or req.generation_method == "custom":
        if req.custom_genome:
            candidate_genome = AttackGenome.from_dict(req.custom_genome)
        else:
            p_id = req.parent_attack_id or "ATK-001"
            candidate_genome = known_genomes_dict.get(p_id, list(known_genomes_dict.values())[0])
        
        lineage = LineageInfo(
            mutation_type="custom_configuration",
            parents=[req.parent_attack_id or "Custom_Config"],
            mutations=["10-dimensional custom genome profile synthesized"],
        )
    elif req.generation_method == "crossover":
        p1_id = req.parent_1_id or "ATK-001"
        p2_id = req.parent_2_id or "ATK-021"

        if p1_id not in known_genomes_dict:
            p1_id = list(known_genomes_dict.keys())[0]
        if p2_id not in known_genomes_dict:
            p2_id = list(known_genomes_dict.keys())[1]

        g1 = known_genomes_dict[p1_id]
        g2 = known_genomes_dict[p2_id]

        child_genome, logs = engine.crossover_genomes(g1, g2)
        lineage = LineageInfo(
            mutation_type="crossover",
            parents=[p1_id, p2_id],
            mutations=logs,
        )
        candidate_genome = child_genome

    else:  # Default to mutation
        parent_id = req.parent_attack_id or "ATK-001"
        if parent_id not in known_genomes_dict:
            parent_id = list(known_genomes_dict.keys())[0]

        parent_genome = known_genomes_dict[parent_id]
        mutated_genome, logs = engine.mutate_genome(parent_genome, req.n_mutations)
        lineage = LineageInfo(
            mutation_type="mutation",
            parents=[parent_id],
            mutations=logs,
        )
        candidate_genome = mutated_genome

    # Compute novelty, realism, evasion and priority scores
    novelty, nearest_known, nearest_sim = calculate_novelty_score(candidate_genome, known_genomes_list)
    realism = calculate_realism_score(candidate_genome)
    evasion = calculate_evasion_potential(candidate_genome)
    priority = calculate_priority_score(novelty, realism, evasion)

    candidate_name = generate_candidate_name(candidate_genome)
    nearest_id = nearest_known.attack_id if nearest_known and hasattr(nearest_known, "attack_id") and nearest_known.attack_id else "ATK-001"
    nearest_arch = lib.get_by_id(nearest_id)
    nearest_desc = f"{nearest_id} ({nearest_arch.name if nearest_arch else 'Known Archetype'})"

    candidate_id = f"NSA-GEN-{seed % 10000:04d}"

    return GenerateCandidateResponse(
        candidate_id=candidate_id,
        candidate_name=candidate_name,
        genome=candidate_genome.get_genes(),
        scores=NovelCandidateScores(
            novelty_score=round(novelty, 4),
            realism_score=round(realism, 4),
            evasion_potential=round(evasion, 4),
            priority_score=round(priority, 4),
        ),
        lineage=lineage,
        nearest_known_archetype=nearest_desc,
        similarity_to_nearest=round(nearest_sim, 4),
    )


@router.post("/evaluate-candidate", response_model=EvaluateCandidateResponse)
def evaluate_candidate(req: EvaluateCandidateRequest, request: Request) -> EvaluateCandidateResponse:
    """Simulate transactions for the candidate genome and test against the active detector."""
    role = (req.detector_role or "active").lower()
    registry = getattr(request.app.state, "model_registry", None) or ModelRegistry()

    if role == "baseline":
        detector = getattr(request.app.state, "detector_baseline", None)
        model_ver = "baseline_v1"
        if detector is None:
            detector = registry.load_baseline_detector()
            request.app.state.detector_baseline = detector
    else:
        role = "active"
        detector = getattr(request.app.state, "detector_active", None)
        model_ver = getattr(request.app.state, "active_model_version", None) or registry.get_active_version()
        if detector is None:
            try:
                detector = registry.load_active_detector()
                request.app.state.detector_active = detector
                request.app.state.active_model_version = model_ver
            except Exception:
                detector = getattr(request.app.state, "detector_baseline", None)
                model_ver = "baseline_v1"

    if detector is None:
        raise HTTPException(status_code=503, detail="Detector model is not available in application state.")

    # Validate and build genome object
    try:
        genome = AttackGenome.from_dict(req.candidate_genome)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid genome schema: {exc}")

    sim_params = genome.to_simulation_parameters()
    gen = TransactionGenerator(seed=req.seed or 42)

    cand_archetype = AttackArchetype(
        attack_id="NSA-CUSTOM",
        name=req.candidate_name or "Novel Synthetic Attack",
        category="Novel Attack Discovery",
        description="Synthetic attack dynamically synthesized from genome representation.",
        severity="CRITICAL",
        novelty_score=0.85,
        detectability_score=0.35,
        behavioral_indicators=[],
        affected_payment_surface=str(sim_params.get("payment_channel", "e-commerce")),
        simulation_parameters=sim_params,
    )

    # Generate synthetic transactions matching this genome profile
    txs = []
    sample_views = []
    n_samples = req.sample_count

    for _ in range(n_samples):
        tx = gen.generate_fraud_transaction(archetype=cand_archetype)
        txs.append(tx)

    df_test = pd.DataFrame(txs)
    probs = detector.predict_proba(df_test)
    preds = (probs >= 0.50).astype(int)

    detected_count = int(np.sum(preds == 1))
    missed_count = int(np.sum(preds == 0))
    detection_rate_pct = round((detected_count / n_samples) * 100.0, 2)

    # Determine blind spot severity
    if detection_rate_pct < 50.0:
        blind_spot_level = "CRITICAL BLIND SPOT"
    elif detection_rate_pct < 80.0:
        blind_spot_level = "MODERATE BLIND SPOT"
    else:
        blind_spot_level = "CONTROLLED SCENARIO"

    # Select up to 10 representative transactions for frontend display
    for i in range(min(10, n_samples)):
        row = df_test.iloc[i]
        sample_views.append(
            SampleTransactionView(
                transaction_id=str(row.get("transaction_id", f"TX-NSA-{i:03d}")),
                transaction_amount=round(float(row.get("transaction_amount", 0.0)), 2),
                device_change=int(row.get("device_change", 0)),
                IP_risk_score=round(float(row.get("IP_risk_score", 0.0)), 4),
                merchant_risk_score=round(float(row.get("merchant_risk_score", 0.0)), 4),
                transaction_velocity_1h=int(row.get("transaction_velocity_1h", 0)),
                is_detected=bool(preds[i] == 1),
                fraud_probability=round(float(probs[i]), 4),
            )
        )

    return EvaluateCandidateResponse(
        candidate_name=req.candidate_name or "Synthetic Attack Candidate",
        detector_role=role,
        model_version=model_ver,
        total_tested=n_samples,
        detected_count=detected_count,
        missed_count=missed_count,
        detection_rate_pct=detection_rate_pct,
        blind_spot_level=blind_spot_level,
        sample_transactions=sample_views,
    )
