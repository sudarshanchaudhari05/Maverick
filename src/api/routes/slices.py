"""FastAPI routes for Phase F: Worst-Slice Mining & Targeted Adaptive Defense."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

from src.utils.config import PROJECT_ROOT

router = APIRouter(prefix="/slices", tags=["Worst-Slice Mining"])

PHASE_F_DIR = PROJECT_ROOT / "experiments" / "phase_f"
REPORT_PATH = PHASE_F_DIR / "reports" / "phase_f_report.json"
AUDIT_PATH = PHASE_F_DIR / "audit" / "phase_f_audit.json"
CONFIG_PATH = PHASE_F_DIR / "config" / "phase_f_config.json"


def _load_report():
    if not REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="Phase F report not found. Execute run_phase_f.py first.")
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/status")
async def get_slice_mining_status():
    """Return Phase F execution status, dataset sizes, and Top-5 discovered slices."""
    report = _load_report()
    return {
        "phase": report.get("phase", "F"),
        "status": report.get("status", "PHASE F — IMPLEMENTED, PENDING INDEPENDENT AUDIT"),
        "title": report.get("title"),
        "timestamp": report.get("timestamp"),
        "datasets": report.get("datasets", {}),
        "top_5_slices_count": len(report.get("top_5_discovered_slices", [])),
        "all_gates_passed": report.get("success_gates", {}).get("all_gates_passed", False),
    }


@router.get("/top")
async def get_top_slices():
    """Return the Top-5 worst-performing slices discovered in Experiment F1."""
    report = _load_report()
    return {
        "top_5_slices": report.get("top_5_discovered_slices", []),
        "slice_evaluations_fd4": report.get("slice_evaluations_fd4", []),
    }


@router.get("/metrics")
async def get_slice_metrics():
    """Return comparative metrics for Baseline Model A vs Hardened Model B."""
    report = _load_report()
    return {
        "unseen_targeted_v2_fd4": report.get("eval_fd4_unseen_targeted_v2", {}),
        "random_baseline_fd2": report.get("eval_fd2_random_baseline", {}),
        "slice_evaluations": report.get("slice_evaluations_fd4", []),
    }


@router.get("/gates")
async def get_success_gates():
    """Return locked Success Gates G1-G5 evaluations and McNemar significance tests."""
    report = _load_report()
    return {
        "success_gates": report.get("success_gates", {}),
        "mcnemar_fd4": report.get("eval_fd4_unseen_targeted_v2", {}).get("mcnemar_test", {}),
        "mcnemar_fd2": report.get("eval_fd2_random_baseline", {}).get("mcnemar_test", {}),
    }
