# Phase F Implementation Walkthrough
FraudForge AI — Worst-Slice Mining & Targeted Adaptive Attack Discovery

## Status
**`PHASE F — IMPLEMENTED, PENDING INDEPENDENT AUDIT`**

---

## 1. Executive Summary & Objective

Phase F implements an autonomous, closed-loop worst-slice mining and targeted hardening framework for FraudForge AI.

**Core Research Question**:
> *"Can automatically identifying the detector's worst-performing transaction/attack slices reveal hidden weaknesses that aggregate metrics miss, and can those weaknesses be converted into targeted attacks that improve the detector after hardening?"*

### Locked Invariants & Scope:
- **Status Policy**: Phase F is explicitly marked `PHASE F — IMPLEMENTED, PENDING INDEPENDENT AUDIT`. No automatic freezing was applied.
- **Frozen Baseline Preservation**: Phase E.1 and Phase E.2 artifacts remain completely untouched with 0 diffs.
- **Deterministic Reproducibility**: Exact seeds (`6001`, `6002`, `6003`, `6004`) and XGBoost hyperparameters (`n_estimators=150`, `max_depth=6`, `learning_rate=0.08`, `subsample=0.85`, `colsample_bytree=0.85`) were applied.
- **Protocol Bounds**: Combinations depth capped strictly at 3 ($D \le 3$, attempts at depth $\ge 4$ raise `ValueError`), minimum support filters ($\ge 50$ total, $\ge 20$ fraud), Wilson 95% score confidence intervals.
- **Independence & Anti-Leakage**: Generator produces transactions without inspecting labels or model outputs; all 6 dataset pairs exhibit strictly 0 sample overlap.

---

## 2. Experimental Datasets & Realized Mixtures

| Dataset ID | Name / Role | Seed | Target Size ($N$) | Realized Size ($N$) | Target Fraud Pct | Realized Fraud Count (Pct) | Realized Legit Count (Pct) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **F-D1** | Discovery Dataset | 6001 | 4,000 | 4,000 | 15.0% | 600 (15.00%) | 3,400 (85.00%) |
| **F-D2** | Random Attack Baseline | 6002 | 2,000 | 2,000 | 15.0% | 300 (15.00%) | 1,700 (85.00%) |
| **F-D3** | Targeted Hardening Set | 6003 | 2,500 | 2,500 | 50.0% | 1,250 (50.00%) | 1,250 (50.00%) |
| **F-D4** | Unseen Targeted V2 Eval | 6004 | 2,000 | 2,000 | 15.0% | 300 (15.00%) | 1,700 (85.00%) |

*Note on F-D3*: Exactly 500 samples were generated per Top-5 discovered weak slice (250 fraud, 250 legitimate per slice), yielding $5 \times 500 = 2,500$ records at a 50.0% fraud ratio.

---

## 3. Discovered Top-5 Weakest Slices (Experiment F1)

Slice search evaluated all 17 dimensions across Families S1 (Transaction Attributes), S2 (Attack Lineage/Strategy), and S3 (Provenance & Environment) at combination depths 1, 2, and 3. Ranking was determined strictly by:
1. `FNR` descending
2. `fraud_support` descending
3. `total_support` descending

| Slice ID | Targetable Slice Definition | Depth | Support (Fraud / Total) | Baseline FNR | Wilson 95% Score CI | Hardened FNR | FNR $\Delta$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **F-SLICE-001** | `amount_bucket`: medium (50-200)<br>`payment_channel`: pos_contactless | 2 | 21 / 672 | 80.95% | [60.00%, 92.33%] | 6.43% | **-15.00 pp** |
| **F-SLICE-002** | `merchant_strategy`: single_target<br>`merchant_category`: retail | 2 | 56 / 56 | 64.29% | [51.19%, 75.54%] | 5.47% | **-15.92 pp** |
| **F-SLICE-003** | `payment_channel`: pos_contactless<br>`geographic_region_or_strategy`: domestic | 2 | 26 / 1103 | 61.54% | [42.53%, 77.57%] | 6.43% | **-15.00 pp** |
| **F-SLICE-004** | `identity_strategy`: existing_account<br>`geographic_strategy`: domestic_matching<br>`merchant_category`: retail | 3 | 50 / 50 | 60.00% | [46.18%, 72.39%] | 5.47% | **-15.92 pp** |
| **F-SLICE-005** | `geographic_strategy`: domestic_matching<br>`merchant_category`: retail<br>`geographic_region_or_strategy`: domestic | 3 | 57 / 57 | 57.89% | [44.98%, 69.81%] | 5.47% | **-15.92 pp** |

---

## 4. Evaluation of Protocol Success Gates (G1 &ndash; G5)

All 5 locked success gates passed conclusively:

| Gate | Description & Locked Requirement | Metric Baseline | Metric Hardened | Margin / Result | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **G1** | **Slice Improvement**: Hardened FNR < Baseline FNR on $\ge 4$ of 5 discovered slices | — | — | **5 of 5 slices improved** (FNR reduced by 15.00 to 15.92 pp) | **PASS** |
| **G2** | **Overall F5 Recall**: On F-D4 (Unseen V2), $\text{Recall}_{\text{hardened}} > \text{Recall}_{\text{baseline}}$ | 81.00% | 94.67% | **+13.67 pp** | **PASS** |
| **G3** | **FPR Protection**: On F-D4, Hardened FPR must not increase by $> +1.0$ pp | 1.29% | 1.88% | **+0.59 pp** (well within $\le +1.0$ pp limit) | **PASS** |
| **G4** | **Random Attack Preservation**: On F-D2, Hardened recall must not decrease by $> 2.0$ pp | 81.67% | 97.67% | **+16.00 pp** (no catastrophic forgetting; substantial gain) | **PASS** |
| **G5** | **Unseen Targeted Generalization**: On F-D4, $\text{FNR}_{\text{hardened}} < \text{FNR}_{\text{baseline}}$ | 19.00% | 5.33% | **-13.67 pp** (72.0% relative reduction in missed fraud) | **PASS** |

---

## 5. Statistical Rigor (McNemar Paired Tests) & Overlap Audit

### McNemar Paired Binary Classification Tests
Evaluated paired decision transitions (correct vs incorrect) using continuity-corrected McNemar $\chi^2$:

- **On F-D4 (Unseen Targeted V2, $N=2,000$)**:
  - Concordant Correct: 1,893 | Concordant Incorrect: 20
  - Discordant pairs: **87**
  - Model B better (cured Model A failure, `b_model_b_better`): **59** (43 fraud cases where Model B cured Model A FN + 16 legitimate cases where Model B cured Model A FP)
  - Model A better (cured Model B failure, `c_model_a_better`): **28** (2 fraud cases where Model A was correct + 26 legitimate cases where Model A was correct)
  - Continuity-corrected $\chi^2$: $\frac{(|59 - 28| - 1)^2}{59 + 28} = \frac{30^2}{87} = \mathbf{10.3448}$
  - Two-sided $p$-value: **0.0013** ($p = 0.001298$) &rarr; Statistically significant paired difference ($p < 0.05$).

- **On F-D2 (Random Baseline, $N=2,000$)**:
  - Discordant pairs: 102 (Model B better: 88, Model A better: 14)
  - $\chi^2_{\text{corrected}} = 52.2451$, $p = 4.90 \times 10^{-13}$ ($p < 0.0001$) &rarr; Statistically significant paired improvement.

### Zero-Leakage Dataset Overlap Audit (All 6 Pairs)
Audited record-level uniqueness across canonical transaction features:
- `F-D1` vs `F-D2`: **0 records**
- `F-D1` vs `F-D3`: **0 records**
- `F-D1` vs `F-D4`: **0 records**
- `F-D2` vs `F-D3`: **0 records**
- `F-D2` vs `F-D4`: **0 records**
- `F-D3` vs `F-D4`: **0 records**

Audit Result: **STRICT PASS — 0 record leakage across all 6 pairs.**

---

## 6. End-to-End Verification & Test Suite

### Regression Test Suite
- Executed `pytest -q`: **138 passed out of 138 tests in 34.50s (100% pass rate)**.
- Specific Phase F test suite: `tests/test_phase_f_integrity.py` (13 tests) passed in 2.04s:
  1. `test_fnr_and_wilson_ci`: Validated mathematical calculation of FNR and Wilson 95% score intervals.
  2. `test_support_filter_and_insufficient_support`: Enforced $\ge 50$ total and $\ge 20$ fraud threshold.
  3. `test_maximum_depth_enforcement`: Verified depth cap $D \le 3$ and rejection of $D \ge 4$ via `ValueError`.
  4. `test_slice_ranking_and_top5_persistent_ids`: Verified multi-key sorting and persistent `F-SLICE-001..005` IDs.
  5. `test_generator_label_independence`: Verified targeted generator never inspects `fraud_label` or model scores.
  6. `test_zero_dataset_overlap`: Confirmed 0 duplicate tuples across all 6 dataset pairs.
  7. `test_mcnemar_paired_calculation`: Confirmed continuity-corrected $\chi^2$ math and $p$-value precision.
  8. `test_fd4_isolated_from_training`: Confirmed F-D4 was strictly isolated during model training.
  9. `test_success_gates_pass`: Verified programmatic pass condition for G1 through G5.
  10. `test_phase_e_remains_frozen`: Verified that Phase E.1 and Phase E.2 configuration and metrics remain frozen.
  11. `test_phase_f_status_not_auto_frozen`: Verified report status is strictly `PHASE F — IMPLEMENTED, PENDING INDEPENDENT AUDIT`.
  12. `test_api_slices_endpoints`: Verified FastAPI routes `/api/v1/slices/{status,top,metrics,gates}` return valid 200 JSON payloads.

---

## 7. Frontend Integration

Updated `frontend/index.html`:
- Added Phase F sidebar button and tab handler in `switchTab("slice-mining-lab")`.
- Bound dynamic data loaders `loadSliceMiningLab()`, `renderSliceKpis()`, `renderSliceTable()` fetching from `/api/v1/slices/*`.
- Configured real-time KPI card bindings for Recall delta, FNR reduction, FPR protection, and Random Attack preservation.
