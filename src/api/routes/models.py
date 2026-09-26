"""Model Registry & Active Model Lifecycle endpoints."""

from fastapi import APIRouter, Request, HTTPException
from src.api.schemas import ModelRegistryResponse, ResetActiveModelResponse, ModelRegistryHistoryEntry
from src.detection.model_registry import ModelRegistry

router = APIRouter(prefix="/models", tags=["Model Registry"])


@router.get("/registry", response_model=ModelRegistryResponse)
def get_model_registry(request: Request) -> ModelRegistryResponse:
    """Get the full session model registry state, active model version, and hardening history."""
    registry = getattr(request.app.state, "model_registry", None) or ModelRegistry()
    data = registry.get_registry_data()

    history_entries = [
        ModelRegistryHistoryEntry(
            version=h["version"],
            model_path=h["model_path"],
            hardening_round=h.get("hardening_round", 0),
            trained_attack_ids=h.get("trained_attack_ids", []),
            created_at=h["created_at"],
            validation_metrics=h.get("validation_metrics"),
            description=h.get("description"),
        )
        for h in data.get("history", [])
    ]

    return ModelRegistryResponse(
        active_version=data.get("active_version", "baseline_v1"),
        previous_version=data.get("previous_version"),
        hardening_round=int(data.get("hardening_round", 0)),
        active_model_path=data.get("active_model_path", "in-memory (session-only)"),
        baseline_model_path=data.get("baseline_model_path", "baseline_detector.joblib"),
        last_updated=data.get("last_updated", ""),
        history=history_entries,
    )


@router.post("/reset-active", response_model=ResetActiveModelResponse)
def reset_active_model(request: Request) -> ResetActiveModelResponse:
    """Reset the active detector back to the baseline model (baseline_v1)."""
    registry = getattr(request.app.state, "model_registry", None) or ModelRegistry()
    active_ver = registry.reset_to_baseline()

    # Update in-memory application state
    request.app.state.detector_active = registry.load_active_detector()
    request.app.state.active_model_version = active_ver

    return ResetActiveModelResponse(
        status="ok",
        message="Active detector reset to baseline_v1 successfully.",
        active_version=active_ver,
        hardening_round=0,
    )
