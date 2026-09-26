"""Risk-Aware Decision Engine & Mitigation routes."""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import (
    EvaluateTransactionRequest,
    EvaluateTransactionResponse,
    DetailedReason,
    MitigationPayload,
)
from src.detection.risk_engine import (
    RiskDecisionEngine,
    PolicyMode,
)
from src.detection.mitigation import MitigationSimulator
from src.detection.model_registry import ModelRegistry

router = APIRouter(prefix="/risk", tags=["Risk Decision Engine"])


@router.post("/evaluate-transaction", response_model=EvaluateTransactionResponse)
def evaluate_transaction(req: EvaluateTransactionRequest, request: Request) -> EvaluateTransactionResponse:
    registry = getattr(request.app.state, "model_registry", None) or ModelRegistry()
    detector = getattr(request.app.state, "detector_active", None)
    active_version = getattr(request.app.state, "active_model_version", None) or registry.get_active_version()

    if detector is None:
        try:
            detector = registry.load_active_detector()
            request.app.state.detector_active = detector
        except Exception:
            detector = getattr(request.app.state, "detector_hardened", None) or getattr(request.app.state, "detector_baseline", None)

    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Active detector model is not loaded in application state.",
        )

    # Resolve policy mode
    mode_str = req.policy_mode.upper()
    try:
        policy_mode = PolicyMode[mode_str]
    except KeyError:
        policy_mode = PolicyMode.BALANCED

    engine = RiskDecisionEngine(
        policy_mode=policy_mode,
        detector=detector,
        model_version=active_version,
    )

    tx_dict = req.model_dump()
    decision_dict = engine.evaluate_transaction(tx_dict)
    mitigation_dict = MitigationSimulator.execute_mitigation(decision_dict)

    # Build detailed reasons list
    detailed_reasons_list = [
        DetailedReason(
            code=r["code"],
            feature=r["feature"],
            value=r["value"],
            severity=r["severity"],
            description=r["description"],
        )
        for r in decision_dict.get("detailed_reasons", [])
    ]

    mitigation_payload = MitigationPayload(
        status=mitigation_dict["status"],
        message=mitigation_dict["message"],
        requires_customer_action=mitigation_dict["requires_customer_action"],
        challenge_type=mitigation_dict.get("challenge_type"),
        recommended_verification_methods=mitigation_dict.get("recommended_verification_methods", []),
    )

    return EvaluateTransactionResponse(
        transaction_id=decision_dict["transaction_id"],
        fraud_probability=round(decision_dict["fraud_probability"], 4),
        risk_score=round(decision_dict["risk_score"], 1),
        risk_level=decision_dict["risk_level"],
        action=decision_dict["action"],
        reason_codes=decision_dict["reason_codes"],
        detailed_reasons=detailed_reasons_list,
        mitigation=mitigation_payload,
        model_version=decision_dict["model_version"],
        policy_version=decision_dict["policy_version"],
    )
