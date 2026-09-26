"""FraudForge AI FastAPI application."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import HealthResponse, ModelStatus
from src.api.routes import catalog, discovery, hardening, risk, models, conformal, provenance, drift, slices
from src.detection.predict import FraudDetector
from src.detection.train import train_baseline_detector, save_detector
from src.detection.model_registry import ModelRegistry
from src.simulation.transaction_generator import TransactionGenerator
from src.utils.config import MODELS_DIR


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model artifacts into application state during startup."""
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Session-only Model Registry (resets to baseline_v1 on every server startup)
    registry = ModelRegistry(models_dir=MODELS_DIR)
    app.state.model_registry = registry

    baseline_path = MODELS_DIR / "baseline_detector.joblib"
    hardened_path = MODELS_DIR / "hardened_zero_day_detector.joblib"
    if not hardened_path.exists():
        hardened_path = MODELS_DIR / "hardened_detector.joblib"

    # 1. Baseline Detector (Immutable)
    if baseline_path.exists():
        try:
            app.state.detector_baseline = FraudDetector(artifact_path=baseline_path)
            app.state.baseline_loaded = True
        except Exception:
            app.state.detector_baseline = None
            app.state.baseline_loaded = False
    else:
        # Train a fast baseline if not yet present
        gen = TransactionGenerator(seed=42)
        df_base = gen.generate_dataset(n_samples=2000, fraud_ratio=0.15)
        artifact_base, _, _ = train_baseline_detector(df_base, seed=42)
        save_detector(artifact_base, baseline_path)
        app.state.detector_baseline = FraudDetector(artifact=artifact_base)
        app.state.baseline_loaded = True

    # 2. Active Detector (Session / In-Memory State)
    try:
        app.state.detector_active = registry.load_active_detector()
        app.state.active_loaded = True
        app.state.active_model_version = registry.get_active_version()
    except Exception:
        app.state.detector_active = app.state.detector_baseline
        app.state.active_loaded = app.state.baseline_loaded
        app.state.active_model_version = "baseline_v1"

    # 3. Hardened Reference Detector (for Gen-2 offline benchmarks)
    if hardened_path.exists():
        try:
            app.state.detector_hardened = FraudDetector(artifact_path=hardened_path)
            app.state.hardened_loaded = True
        except Exception:
            app.state.detector_hardened = app.state.detector_active
            app.state.hardened_loaded = False
    else:
        app.state.detector_hardened = app.state.detector_active
        app.state.hardened_loaded = app.state.active_loaded

    yield


app = FastAPI(
    title="FraudForge AI — API Backend",
    description="Adaptive Red-Team / Blue-Team AI Defense Lab for GenAI Payment Security",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration (supporting localhost:3000 and 127.0.0.1:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def get_health(request: Request) -> HealthResponse:
    """Check API operational status and preloaded model availability."""
    registry = getattr(request.app.state, "model_registry", None) or ModelRegistry()
    active_ver = getattr(request.app.state, "active_model_version", None) or registry.get_active_version()
    hardening_round = registry.get_hardening_round()

    baseline_path = MODELS_DIR / "baseline_detector.joblib"
    hardened_path = MODELS_DIR / "hardened_zero_day_detector.joblib"
    if not hardened_path.exists():
        hardened_path = MODELS_DIR / "hardened_detector.joblib"

    base_size = round(baseline_path.stat().st_size / 1024.0, 1) if baseline_path.exists() else None
    hard_size = round(hardened_path.stat().st_size / 1024.0, 1) if hardened_path.exists() else None

    active_path_str = str(baseline_path.name) if active_ver == "baseline_v1" else "in-memory (session-only)"
    active_size_kb = base_size if active_ver == "baseline_v1" else None

    active_status = ModelStatus(
        name=f"Active Operational Detector ({active_ver})",
        is_loaded=bool(getattr(request.app.state, "active_loaded", True)),
        path=active_path_str,
        size_kb=active_size_kb,
        version=active_ver,
    )

    base_status = ModelStatus(
        name="Baseline XGBoost Detector (Immutable)",
        is_loaded=bool(getattr(request.app.state, "baseline_loaded", False)),
        path=str(baseline_path.name),
        size_kb=base_size,
        version="baseline_v1",
    )

    hard_status = ModelStatus(
        name="Hardened Zero-Day Reference Detector",
        is_loaded=bool(getattr(request.app.state, "hardened_loaded", False)),
        path=str(hardened_path.name),
        size_kb=hard_size,
        version="hardened_zero_day_ref",
    )

    return HealthResponse(
        status="ok",
        service="FraudForge AI API",
        mode="simulation",
        model="hardened",
        app_name="FraudForge AI Backend",
        version="1.0.0",
        active_model_version=active_ver,
        hardening_round=hardening_round,
        active_model=active_status,
        baseline_model=base_status,
        hardened_model=hard_status,
        active_test_count=72,
    )


# Include all modular routers under /api/v1
app.include_router(catalog.router, prefix="/api/v1")
app.include_router(discovery.router, prefix="/api/v1")
app.include_router(hardening.router, prefix="/api/v1")
app.include_router(risk.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(conformal.router, prefix="/api/v1")
app.include_router(provenance.router, prefix="/api/v1")
app.include_router(drift.router, prefix="/api/v1")
app.include_router(slices.router, prefix="/api/v1")
