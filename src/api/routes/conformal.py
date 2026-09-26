"""Conformal Calibration & Risk-Controlled Abstention API endpoints."""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Request
import pandas as pd
import numpy as np

from src.api.schemas import (
    ConformalCalibrateRequest,
    ConformalCalibrateResponse,
    ConformalPredictRequest,
    ConformalPredictResponse,
    ConformalPredictionResult,
    ConformalMetricsResponse,
)
from src.detection.conformal_calibration import (
    ConformalCalibrator,
    get_default_calibration_dataset,
)
from src.detection.abstention_engine import (
    RiskControlledAbstentionEngine,
    AbstentionEvaluationMetrics,
)
from src.detection.model_registry import ModelRegistry
from src.utils.config import GENERATED_DATA_DIR, PROCESSED_DATA_DIR

router = APIRouter(prefix="/conformal", tags=["Conformal Calibration & Abstention"])


def _ensure_conformal_state(request: Request) -> tuple[ConformalCalibrator, RiskControlledAbstentionEngine]:
    """Ensure in-memory calibrator and abstention engine exist in application state."""
    registry = getattr(request.app.state, "model_registry", None) or ModelRegistry()
    active_detector = getattr(request.app.state, "detector_active", None)
    if active_detector is None:
        active_detector = registry.load_active_detector()
        request.app.state.detector_active = active_detector

    calibrator: Optional[ConformalCalibrator] = getattr(request.app.state, "conformal_calibrator", None)
    abstention_engine: Optional[RiskControlledAbstentionEngine] = getattr(request.app.state, "abstention_engine", None)

    active_ver = getattr(request.app.state, "active_model_version", None) or registry.get_active_version()

    # Re-initialize or calibrate if not present or if model changed
    if calibrator is None or calibrator.detector != active_detector:
        calibrator = ConformalCalibrator(detector=active_detector, default_alpha=0.05)
        cal_df = get_default_calibration_dataset(n_samples=1000)
        calibrator.calibrate(cal_df, target_alpha=0.05, detector=active_detector, model_version=active_ver)
        request.app.state.conformal_calibrator = calibrator

    if abstention_engine is None or abstention_engine.calibrator != calibrator:
        abstention_engine = RiskControlledAbstentionEngine(calibrator=calibrator, default_target_fnr=0.05)
        cal_df = get_default_calibration_dataset(n_samples=1000)
        abstention_engine.calibrate_acceptance_threshold(cal_df, target_fnr=0.05, detector=active_detector)
        request.app.state.abstention_engine = abstention_engine

    return calibrator, abstention_engine


@router.post("/calibrate", response_model=ConformalCalibrateResponse)
def calibrate_conformal(req: ConformalCalibrateRequest, request: Request) -> ConformalCalibrateResponse:
    """Calibrate conformal nonconformity bounds and target FNR acceptance threshold."""
    registry = getattr(request.app.state, "model_registry", None) or ModelRegistry()
    active_detector = getattr(request.app.state, "detector_active", None)
    if active_detector is None:
        active_detector = registry.load_active_detector()
        request.app.state.detector_active = active_detector

    active_ver = getattr(request.app.state, "active_model_version", None) or registry.get_active_version()

    calibrator = ConformalCalibrator(detector=active_detector, default_alpha=req.target_alpha)
    cal_df = get_default_calibration_dataset(n_samples=req.calibration_sample_size)
    cal_state = calibrator.calibrate(
        cal_df,
        target_alpha=req.target_alpha,
        detector=active_detector,
        model_version=active_ver,
    )

    abstention_engine = RiskControlledAbstentionEngine(calibrator=calibrator, default_target_fnr=req.target_fnr)
    tau_accept = abstention_engine.calibrate_acceptance_threshold(
        cal_df,
        target_fnr=req.target_fnr,
        detector=active_detector,
    )

    # Store in application state
    request.app.state.conformal_calibrator = calibrator
    request.app.state.abstention_engine = abstention_engine

    return ConformalCalibrateResponse(
        status="ok",
        is_calibrated=cal_state.is_calibrated,
        target_alpha=cal_state.target_alpha,
        nominal_coverage=cal_state.nominal_coverage,
        target_fnr=req.target_fnr,
        q_hat=cal_state.q_hat,
        acceptance_threshold=tau_accept,
        n_calibration_samples=cal_state.n_calibration_samples,
        model_version=active_ver,
        calibrated_at=cal_state.calibrated_at,
        exchangeability_disclaimer=cal_state.exchangeability_assumption,
    )


@router.post("/predict", response_model=ConformalPredictResponse)
def predict_conformal(req: ConformalPredictRequest, request: Request) -> ConformalPredictResponse:
    """Evaluate transactions using conformal prediction sets and risk-controlled selective abstention."""
    calibrator, abstention_engine = _ensure_conformal_state(request)
    active_detector = getattr(request.app.state, "detector_active", None) or calibrator.detector
    active_ver = getattr(request.app.state, "active_model_version", None) or "baseline_v1"

    # Assemble transactions list
    tx_list: List[Dict[str, Any]] = []
    if req.transaction:
        tx_list.append(req.transaction)
    if req.transactions:
        tx_list.extend(req.transactions)

    if not tx_list:
        raise HTTPException(status_code=400, detail="Provide 'transaction' object or 'transactions' list.")

    df = pd.DataFrame(tx_list)
    decisions, metrics = abstention_engine.evaluate_dataset(
        df,
        detector=active_detector,
        target_fnr_override=req.target_fnr,
        model_version=active_ver,
    )

    results: List[ConformalPredictionResult] = []
    for d in decisions:
        c_set = calibrator.predict_single(
            {"fraud_probability": d.fraud_probability, "transaction_id": d.transaction_id},
            detector=active_detector,
            alpha_override=req.target_alpha,
        )
        results.append(
            ConformalPredictionResult(
                transaction_id=d.transaction_id,
                decision=d.decision,
                routing=d.routing,
                is_abstained=d.is_abstained,
                fraud_probability=d.fraud_probability,
                prediction_set=d.conformal_set,
                prediction_set_size=d.conformal_set_size,
                p_value_0=c_set.p_value_0,
                p_value_1=c_set.p_value_1,
                acceptance_threshold=d.acceptance_threshold,
                q_hat=calibrator.q_hat or 0.0,
                target_fnr=d.target_fnr,
                reason=d.reason,
            )
        )

    return ConformalPredictResponse(
        total_evaluated=len(results),
        automatic_decisions=metrics.automatic_decision_count,
        human_reviews=metrics.human_review_count,
        active_model_version=active_ver,
        results=results,
    )


@router.get("/metrics", response_model=ConformalMetricsResponse)
def get_conformal_metrics(request: Request, target_fnr: Optional[float] = None) -> ConformalMetricsResponse:
    """Compute and return empirical conformal abstention and false-negative metrics on test data."""
    calibrator, abstention_engine = _ensure_conformal_state(request)
    active_detector = getattr(request.app.state, "detector_active", None) or calibrator.detector
    active_ver = getattr(request.app.state, "active_model_version", None) or "baseline_v1"

    # Load test dataset for evaluation
    path_test = PROCESSED_DATA_DIR / "test_split.csv"
    path_10k = GENERATED_DATA_DIR / "synthetic_transactions_10k.csv"

    if path_test.exists() and len(pd.read_csv(path_test)) >= 50:
        df_test = pd.read_csv(path_test)
    elif path_10k.exists():
        df_full = pd.read_csv(path_10k)
        df_test = df_full.sample(n=min(2000, len(df_full)), random_state=1337).reset_index(drop=True)
    else:
        df_test = get_default_calibration_dataset(n_samples=500, seed=1337)

    t_fnr = target_fnr if target_fnr is not None else abstention_engine.target_fnr
    _, metrics = abstention_engine.evaluate_dataset(
        df_test,
        detector=active_detector,
        target_fnr_override=t_fnr,
        model_version=active_ver,
    )

    return ConformalMetricsResponse(
        status="ok",
        target_fnr=metrics.target_fnr,
        observed_fnr=metrics.observed_fnr,
        fnr_among_accepted=metrics.fnr_among_accepted,
        coverage=metrics.coverage,
        abstention_rate=metrics.abstention_rate,
        total_evaluated=metrics.total_evaluated,
        automatic_decision_count=metrics.automatic_decision_count,
        human_review_count=metrics.human_review_count,
        accepted_count=metrics.accepted_count,
        blocked_count=metrics.blocked_count,
        false_negatives=metrics.false_negatives,
        false_positives=metrics.false_positives,
        true_positives=metrics.true_positives,
        true_negatives=metrics.true_negatives,
        precision_auto=metrics.precision_auto,
        recall_auto=metrics.recall_auto,
        q_hat=calibrator.q_hat or 0.0,
        active_model_version=active_ver,
        exchangeability_disclaimer=metrics.exchangeability_disclaimer,
    )
