"""Zero-Day Adaptive Hardening & Defense Comparison routes."""

import random
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Request
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

from src.api.schemas import (
    EvolveGen2Request,
    EvolveGen2Response,
    CompareDefenseRequest,
    CompareDefenseResponse,
    DetectorEvaluationStats,
    DefenseImprovementStats,
)
from src.attacks.attack_genome import AttackGenome
from src.attacks.attack_library import AttackArchetype
from src.attacks.novelty_engine import generate_candidate_name
from src.simulation.transaction_generator import TransactionGenerator
from src.features.feature_engineering import FraudFeaturePipeline, extract_features_and_targets
from src.detection.predict import FraudDetector
from src.detection.model_registry import ModelRegistry

router = APIRouter(prefix="/hardening", tags=["Adaptive Hardening & Defense"])


@router.post("/evolve-gen2", response_model=EvolveGen2Response)
def evolve_gen2(req: EvolveGen2Request) -> EvolveGen2Response:
    """Evolve a discovered novel attack into an unseen Generation-2 (V2) variant."""
    try:
        parent_genome = AttackGenome.from_dict(req.parent_genome)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid parent genome: {exc}")

    evolved_dict = parent_genome.to_dict().copy()
    evolved_genes = []

    # Evolution rules matching ZeroDayHardeningPipeline Generation-2 synthesis
    # 1. Evolve Amount Strategy
    if evolved_dict["amount_strategy"] == "sub_threshold_structuring":
        evolved_dict["amount_strategy"] = "stealth_discounted"
        evolved_genes.append("amount_strategy: sub_threshold_structuring -> stealth_discounted")
    elif evolved_dict["amount_strategy"] == "high_ticket_burst":
        evolved_dict["amount_strategy"] = "micro_charge_testing"
        evolved_genes.append("amount_strategy: high_ticket_burst -> micro_charge_testing")
    else:
        evolved_dict["amount_strategy"] = "sub_threshold_structuring"
        evolved_genes.append("amount_strategy: -> sub_threshold_structuring")

    # 2. Evolve Temporal Strategy
    if evolved_dict["temporal_strategy"] == "distributed":
        evolved_dict["temporal_strategy"] = "off_hours_window"
        evolved_genes.append("temporal_strategy: distributed -> off_hours_window")
    elif evolved_dict["temporal_strategy"] == "burst_rapid":
        evolved_dict["temporal_strategy"] = "gradual_escalation"
        evolved_genes.append("temporal_strategy: burst_rapid -> gradual_escalation")

    # 3. Evolve Evasion Strategy if possible
    if evolved_dict["evasion_strategy"] == "trusted_session_behavior":
        evolved_dict["evasion_strategy"] = "trusted_device_masking"
        evolved_genes.append("evasion_strategy: trusted_session_behavior -> trusted_device_masking")

    variant_genome = AttackGenome.from_dict(evolved_dict)
    variant_name = f"{generate_candidate_name(variant_genome)} (Gen-2 Evolved)"
    variant_id = f"{req.candidate_id or 'NSA-001'}-V2"

    return EvolveGen2Response(
        variant_id=variant_id,
        variant_name=variant_name,
        genome=variant_genome.get_genes(),
        parent_id=req.candidate_id or "NSA-001",
        evolved_genes=evolved_genes,
    )


def _train_and_validate_hardened_candidate(
    attack_genome: AttackGenome,
    attack_name: str,
    target_attack_id: str,
    registry: ModelRegistry,
    baseline_detector: FraudDetector,
    seed: int = 42,
) -> tuple[Optional[Dict[str, Any]], Optional[FraudDetector], bool, Dict[str, Any], str]:
    """Dynamically synthesize adversarial training data and retrain candidate XGBoost model."""
    sim_params = attack_genome.to_simulation_parameters()
    gen = TransactionGenerator(seed=seed)

    target_archetype = AttackArchetype(
        attack_id=target_attack_id,
        name=attack_name,
        category="Targeted Adversarial Hardening",
        description="Targeted attack scenario for runtime defense hardening.",
        severity="CRITICAL",
        novelty_score=0.85,
        detectability_score=0.35,
        behavioral_indicators=[],
        affected_payment_surface=str(sim_params.get("payment_channel", "e-commerce")),
        simulation_parameters=sim_params,
    )

    # 1. Generate base legitimate & background fraud transactions (Dataset A slice)
    df_normal_train = gen.generate_dataset(n_samples=1800, fraud_ratio=0.10)

    # 2. Generate target attack adversarial training samples (augmented)
    target_txs = [gen.generate_fraud_transaction(archetype=target_archetype) for _ in range(400)]
    df_target_fraud = pd.DataFrame(target_txs)

    # 3. Assemble augmented training set
    df_augmented_train = pd.concat([df_normal_train, df_target_fraud], ignore_index=True).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)

    # 4. Train pipeline and model
    X_train, y_train, _ = extract_features_and_targets(df_augmented_train)
    feature_pipeline = FraudFeaturePipeline()
    X_trans = feature_pipeline.fit_transform(X_train, y_train)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos = float(n_neg / max(1, n_pos))

    candidate_xgb = XGBClassifier(
        n_estimators=130,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        scale_pos_weight=scale_pos,
        random_state=seed,
        eval_metric="logloss",
        n_jobs=-1,
    )
    candidate_xgb.fit(X_trans, y_train)

    candidate_artifact = {
        "pipeline": feature_pipeline,
        "model": candidate_xgb,
        "model_type": "xgboost_targeted_hardened",
        "feature_names": feature_pipeline.get_feature_names_out(),
        "seed": seed,
    }
    candidate_detector = FraudDetector(artifact=candidate_artifact)

    # 5. Validation Gates
    gen_val = TransactionGenerator(seed=seed + 999)
    val_target_txs = [gen_val.generate_fraud_transaction(archetype=target_archetype) for _ in range(100)]
    df_val_target = pd.DataFrame(val_target_txs)
    df_val_normal = gen_val.generate_dataset(n_samples=500, fraud_ratio=0.15)

    is_valid, val_metrics, val_msg = registry.validate_candidate(
        candidate_detector=candidate_detector,
        target_attack_df=df_val_target,
        normal_test_df=df_val_normal,
        baseline_detector=baseline_detector,
        min_recall_gain=0.0,
        min_normal_recall=0.85,
    )

    return candidate_artifact, candidate_detector, is_valid, val_metrics, val_msg


@router.post("/compare-defense", response_model=CompareDefenseResponse)
def compare_defense(req: CompareDefenseRequest, request: Request) -> CompareDefenseResponse:
    """Evaluate an attack scenario against Baseline and Hardened detectors side-by-side,
    dynamically retraining and promoting the active detector upon successful hardening."""
    registry = getattr(request.app.state, "model_registry", None) or ModelRegistry()
    detector_baseline = getattr(request.app.state, "detector_baseline", None)

    if detector_baseline is None:
        try:
            detector_baseline = registry.load_baseline_detector()
            request.app.state.detector_baseline = detector_baseline
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Baseline detector not available: {exc}")

    try:
        genome = AttackGenome.from_dict(req.attack_genome)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid attack genome: {exc}")

    sim_params = genome.to_simulation_parameters()
    target_id = req.target_attack_id or "ATK-CUSTOM"
    scenario_name = req.attack_name or "Custom Attack Scenario"

    prev_version = getattr(request.app.state, "active_model_version", None) or registry.get_active_version()
    active_version = prev_version
    promoted = False
    val_passed = True
    val_msg = "Evaluated against existing models."

    # 1. Dynamic Hardening Retraining & Validation
    if req.auto_promote:
        cand_artifact, cand_detector, is_valid, val_metrics, val_msg = _train_and_validate_hardened_candidate(
            attack_genome=genome,
            attack_name=scenario_name,
            target_attack_id=target_id,
            registry=registry,
            baseline_detector=detector_baseline,
            seed=req.seed or 42,
        )
        val_passed = is_valid

        if is_valid and cand_artifact is not None and cand_detector is not None:
            # Promote candidate model
            new_ver, _ = registry.promote_candidate(
                candidate_artifact=cand_artifact,
                trained_attack_ids=[target_id],
                validation_metrics=val_metrics,
            )
            promoted = True
            active_version = new_ver

            # Hot-swap in-memory detector state in FastAPI immediately
            request.app.state.detector_active = cand_detector
            request.app.state.detector_hardened = cand_detector
            request.app.state.active_model_version = new_ver
            detector_hardened = cand_detector
        else:
            detector_hardened = getattr(request.app.state, "detector_hardened", None) or detector_baseline
    else:
        detector_hardened = getattr(request.app.state, "detector_hardened", None) or detector_baseline

    # 2. Side-by-Side Inference on Test Transactions
    gen = TransactionGenerator(seed=req.seed or 2026)
    test_archetype = AttackArchetype(
        attack_id="GEN2-COMPARE",
        name=scenario_name,
        category="Hardening Evaluation",
        description="Attack scenario evaluation.",
        severity="CRITICAL",
        novelty_score=0.85,
        detectability_score=0.35,
        behavioral_indicators=[],
        affected_payment_surface=str(sim_params.get("payment_channel", "e-commerce")),
        simulation_parameters=sim_params,
    )

    n_samples = req.sample_count
    txs = [gen.generate_fraud_transaction(archetype=test_archetype) for _ in range(n_samples)]
    df_test = pd.DataFrame(txs)

    # Baseline Detector Inference
    probs_base = detector_baseline.predict_proba(df_test)
    preds_base = (probs_base >= 0.50).astype(int)
    base_detected = int(np.sum(preds_base == 1))
    base_missed = int(np.sum(preds_base == 0))
    base_rate = round((base_detected / n_samples) * 100.0, 2)

    # Hardened Detector Inference
    probs_hard = detector_hardened.predict_proba(df_test)
    preds_hard = (probs_hard >= 0.50).astype(int)
    hard_detected = int(np.sum(preds_hard == 1))
    hard_missed = int(np.sum(preds_hard == 0))
    hard_rate = round((hard_detected / n_samples) * 100.0, 2)

    # Delta Metrics
    gen_gain = round(hard_rate - base_rate, 2)
    miss_reduction = base_missed - hard_missed
    fn_reduction_pct = round(
        ((base_missed - hard_missed) / max(base_missed, 1)) * 100.0, 2
    ) if base_missed > 0 else 0.0

    return CompareDefenseResponse(
        scenario_name=scenario_name,
        total_simulated=n_samples,
        baseline_detector=DetectorEvaluationStats(
            detected=base_detected,
            missed=base_missed,
            detection_rate_pct=base_rate,
        ),
        hardened_detector=DetectorEvaluationStats(
            detected=hard_detected,
            missed=hard_missed,
            detection_rate_pct=hard_rate,
        ),
        defense_improvement=DefenseImprovementStats(
            generalization_gain_pct_points=gen_gain,
            missed_attacks_reduction=miss_reduction,
            false_negative_reduction_pct=fn_reduction_pct,
        ),
        promoted=promoted,
        validation_passed=val_passed,
        active_version=active_version,
        previous_version=prev_version,
        validation_message=val_msg,
        hardening_round=registry.get_hardening_round(),
    )
