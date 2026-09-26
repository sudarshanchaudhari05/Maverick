"""Focused unit tests for Conformal Distribution Shift experiment logic."""

import pytest
import pandas as pd
from src.utils.config import EXPERIMENTS_DIR
from experiments.conformal_distribution_shift import (
    load_evaluation_regimes,
    verify_data_non_overlap,
    run_distribution_shift_experiment,
)


def test_load_evaluation_regimes_and_non_overlap():
    """Verify that calibration and evaluation sets are loaded disjointly with zero overlap."""
    cal_df, same_df, adv_df, gen2_df = load_evaluation_regimes(cal_size=500, same_eval_size=500)
    
    assert len(cal_df) == 500
    assert len(same_df) == 500
    assert len(adv_df) > 0
    assert len(gen2_df) > 0

    audit = verify_data_non_overlap(cal_df, same_df, adv_df, gen2_df)
    assert audit["leakage_passed"] is True
    assert audit["same_distribution_overlap"] == 0
    assert audit["adversarial_c_overlap"] == 0
    assert audit["gen2_novel_d_overlap"] == 0


def test_distribution_shift_experiment_structure(tmp_path):
    """Verify that running the shift experiment produces structured reports, plots, and expected violations."""
    report = run_distribution_shift_experiment(target_fnrs=[0.05], output_dir=tmp_path)
    
    assert "experiment_name" in report
    assert "target_fnr_metrics" in report
    assert len(report["target_fnr_metrics"]) == 3  # 3 regimes for 1 target FNR

    # Verify each regime result
    regime_results = {r["regime_key"]: r for r in report["target_fnr_metrics"]}
    assert "same_distribution" in regime_results
    assert "adversarial" in regime_results
    assert "gen2_shifted" in regime_results

    # Same-distribution should satisfy target FNR (under exchangeability)
    same_res = regime_results["same_distribution"]
    assert same_res["observed_fnr"] <= same_res["target_fnr"]
    assert same_res["is_violation"] is False

    # Shifted distributions should exhibit violations
    gen2_res = regime_results["gen2_shifted"]
    assert gen2_res["is_violation"] is True
    assert gen2_res["observed_fnr"] > gen2_res["target_fnr"]

    # Verify files created in tmp_path
    assert (tmp_path / "conformal_distribution_shift_report.json").exists()
    assert (tmp_path / "conformal_distribution_shift_metrics.csv").exists()
    assert (tmp_path / "risk_coverage_curve.csv").exists()
    assert (tmp_path / "target_vs_observed_fnr.png").exists()
    assert (tmp_path / "risk_coverage_curve.png").exists()
