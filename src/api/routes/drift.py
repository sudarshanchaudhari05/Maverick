"""FraudForge AI: Conformal Test-Martingale Drift Detection & Adaptive Recalibration API Routes."""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
import json
import pandas as pd
from pathlib import Path

from src.utils.config import EXPERIMENTS_DIR

router = APIRouter(prefix="/drift", tags=["Conformal Drift & Adaptive Recalibration"])


def _load_phase_e_report() -> Dict[str, Any]:
    report_path = EXPERIMENTS_DIR / "phase_e_report.json"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Phase E.1 experiment report not found. Run experiments/conformal_drift_adaptation.py first.",
        )
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/status")
def get_drift_status(stream: Optional[str] = Query("E2_abrupt_drift", description="Stream ID (E1_no_drift, E2_abrupt_drift, E3_gradual_drift, E4_adversarial_gen2_drift)")) -> Dict[str, Any]:
    """Retrieve the high-level Phase E.1 experiment status, active stream, and drift parameters."""
    report = _load_phase_e_report()
    summaries = report.get("stream_summaries", {})
    if stream not in summaries:
        stream = "E2_abrupt_drift"

    stream_data = summaries.get(stream, {})
    return {
        "phase": report.get("phase", "E.1"),
        "status": report.get("status", "IMPLEMENTED_NOT_YET_FROZEN"),
        "active_stream": stream,
        "true_drift_point": stream_data.get("true_drift_point"),
        "first_alarm_batch": stream_data.get("first_alarm_batch"),
        "detection_delay_batches": stream_data.get("detection_delay_batches"),
        "total_alarms": stream_data.get("total_alarms_triggered", 0),
        "recalibrations_executed": stream_data.get("recalibrations_executed_drift_aware", 0),
        "scientific_integrity": report.get("scientific_integrity", {}),
        "conformal_guarantee_disclaimer": report.get("conformal_guarantee_disclaimer", ""),
    }


@router.get("/metrics")
def get_drift_metrics(stream: Optional[str] = Query(None, description="Optional stream filter")) -> Dict[str, Any]:
    """Retrieve comparative metrics across System A (Static), System B (Oracle), and System C (Drift-Aware)."""
    report = _load_phase_e_report()
    summaries = report.get("stream_summaries", {})
    if stream and stream in summaries:
        return {stream: summaries[stream]}
    return summaries


@router.get("/timeline")
def get_drift_timeline(stream: Optional[str] = Query("E2_abrupt_drift", description="Stream to inspect")) -> List[Dict[str, Any]]:
    """Retrieve batch-by-batch timeline data for the selected experiment stream."""
    timeline_path = EXPERIMENTS_DIR / "phase_e_timeline.csv"
    if not timeline_path.exists():
        raise HTTPException(status_code=404, detail="Phase E.1 timeline CSV not found.")
    
    df = pd.read_csv(timeline_path)
    if stream:
        df = df[df["stream"] == stream]
    
    return df.to_dict(orient="records")


@router.get("/calibration-events")
def get_calibration_events(stream: Optional[str] = Query("E2_abrupt_drift", description="Stream to inspect")) -> Dict[str, Any]:
    """Retrieve the log of calibration and recalibration events for Oracle and Drift-Aware managers."""
    report = _load_phase_e_report()
    summaries = report.get("stream_summaries", {})
    s_data = summaries.get(stream, summaries.get("E2_abrupt_drift", {}))
    return {
        "stream": stream,
        "oracle_events": s_data.get("manager_b_events", []),
        "drift_aware_events": s_data.get("manager_c_events", []),
    }


@router.get("/methodology")
def get_drift_methodology() -> Dict[str, Any]:
    """Retrieve the formal mathematical definitions, assumptions, and limitations of Phase E.1."""
    report = _load_phase_e_report()
    return {
        "experiment_name": report.get("experiment_name"),
        "methodology": {
            "test_martingale": "Power Martingale f(u) = epsilon * u^(epsilon - 1) with epsilon = 0.80",
            "thresholding": "Ville's Inequality: tau = 1 / alpha_drift (alpha_drift = 0.01 -> tau = 100.0)",
            "label_delay": "LABEL_DELAY = 3 batches strictly enforced before true labels become accessible",
            "recalibration_window": "500 eligible delayed-label samples required for quantile recalibration",
            "fallback_policy": "If eligible labels < 500 upon alarm, set CALIBRATION_STALE = True and wait",
        },
        "metric_definitions": report.get("metric_definitions", {}),
        "limitations": report.get("limitations", []),
        "conformal_disclaimer": report.get("conformal_guarantee_disclaimer", ""),
        "risk_coverage_interpretation": report.get("risk_coverage_interpretation", ""),
    }
