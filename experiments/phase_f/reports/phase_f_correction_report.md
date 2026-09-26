# Phase F Correction Report: F-D4 McNemar Statistical Reconciliation
FraudForge AI — Worst-Slice Mining & Targeted Adaptive Defense

## Status
**`PHASE F — IMPLEMENTED, PENDING INDEPENDENT AUDIT`**

---

## 1. Executive Summary

This correction report documents the mathematical trace and reconciliation of the paired McNemar significance test on held-out dataset **F-D4** ($N=2,000$, Seed `6004`, 15.0% fraud prevalence) between Baseline Detector (Model A) and Targeted Hardened Detector (Model B).

An initial documentation transcription discrepancy noted discordant pair figures of `46 / 5` in draft markdown. A direct trace from the primary prediction artifacts (`fd4_predictions.parquet` and `fd4_predictions_baseline_vs_hardened.csv`) confirmed that the authoritative values are **`59 / 28`** (total discordant pairs = **`87`**).

All reports, artifacts, API endpoints, frontend bindings, and unit tests have been reconciled to ensure exactly one authoritative result across the system.

---

## 2. Contingency Trace from Authoritative Artifact (`fd4_predictions.parquet`)

Evaluating all $N=2,000$ paired predictions on F-D4 against true labels ($y_{\text{true}} \in \{0, 1\}$):

### Joint 2x2 Decision Matrix
| | Model B Correct ($\hat{y}_B = y$) | Model B Incorrect ($\hat{y}_B \neq y$) | Total |
| :--- | :---: | :---: | :---: |
| **Model A Correct ($\hat{y}_A = y$)** | **1,893** (`both_correct`) | **28** (`c_model_a_better`) | 1,921 |
| **Model A Incorrect ($\hat{y}_A \neq y$)** | **59** (`b_model_b_better`) | **20** (`both_incorrect`) | 79 |
| **Total** | 1,952 | 48 | **2,000** |

### Stratified Class-Level Breakdown
1. **Fraud Class ($N_{\text{fraud}} = 300$)**:
   - Model A Baseline: 243 True Positives, 57 False Negatives
   - Model B Hardened: 284 True Positives, 16 False Negatives
   - Model B cured Model A False Negative ($y=1, \hat{y}_A=0, \hat{y}_B=1$): **43**
   - Model A correct while Model B missed ($y=1, \hat{y}_A=1, \hat{y}_B=0$): **2**
   - Both correct ($y=1, \hat{y}_A=1, \hat{y}_B=1$): **241**
   - Both missed ($y=1, \hat{y}_A=0, \hat{y}_B=0$): **14**

2. **Legitimate Class ($N_{\text{legit}} = 1,700$)**:
   - Model A Baseline: 1,678 True Negatives, 22 False Positives
   - Model B Hardened: 1,668 True Negatives, 32 False Positives
   - Model B cured Model A False Positive ($y=0, \hat{y}_A=1, \hat{y}_B=0$): **16**
   - Model A correct while Model B raised False Positive ($y=0, \hat{y}_A=0, \hat{y}_B=1$): **26**
   - Both correct ($y=0, \hat{y}_A=0, \hat{y}_B=0$): **1,652**
   - Both raised false alarms ($y=0, \hat{y}_A=1, \hat{y}_B=1$): **6**

### Discordant Aggregation
- **$b$ (`b_model_b_better`)** = $43 \text{ (fraud cured)} + 16 \text{ (legit cured)} = \mathbf{59}$
- **$c$ (`c_model_a_better`)** = $2 \text{ (fraud missed)} + 26 \text{ (legit false alarm)} = \mathbf{28}$
- **Total Discordant Pairs ($b + c$)** = $59 + 28 = \mathbf{87}$

---

## 3. Authoritative Test Statistics

Using Edwards' continuity-corrected McNemar $\chi^2$ statistic:
$$\chi^2_{\text{corrected}} = \frac{(|b - c| - 1)^2}{b + c} = \frac{(|59 - 28| - 1)^2}{59 + 28} = \frac{(31 - 1)^2}{87} = \frac{900}{87} = \mathbf{10.3448}$$

- **Degrees of Freedom**: $df = 1$
- **Two-Sided $p$-value**: $p = \mathbf{0.001298} \approx \mathbf{0.0013}$
- **Significance at $\alpha = 0.05$**: **TRUE** ($p < 0.05$)
- **Significance at $\alpha = 0.01$**: **TRUE** ($p < 0.01$)

The draft value of $46 / 5$ was an assistant transcription error. The pipeline JSON report (`phase_f_report.json`) and raw prediction table have always reflected the true mathematical counts of $59 / 28$.

### Finite Precision Representation on F-D2 and F-D1
To prevent very small finite McNemar $p$-values from collapsing to literal `0.0` due to floating point truncation:
- **F-D2 (Random Baseline, $N=2,000$)**:
  - $b=88, c=14, \text{discordant}=102, \chi^2_{\text{corrected}}=52.2451$
  - Finite $p$-value stored as: **`4.90e-13`** (instead of literal `0.0`)
- **F-D1 (Discovery Dataset, $N=4,000$)**:
  - $b=171, c=32, \text{discordant}=203, \chi^2_{\text{corrected}}=93.8128$
  - Finite $p$-value stored as: **`3.47e-22`** (instead of literal `0.0`)

---

## 4. Reconciled System Artifacts

1. **Prediction Artifact**: `experiments/phase_f/predictions/fd4_predictions.parquet` (and CSV mirror).
2. **Authoritative JSON**: `experiments/phase_f/reports/phase_f_report.json` under `eval_fd4_unseen_targeted_v2.mcnemar_test`.
3. **Audit JSON**: `experiments/phase_f/audit/phase_f_audit.json` (zero-leakage audit verified).
4. **Walkthrough Document**: Artifact `walkthrough.md` updated with $b=59, c=28, \text{discordant}=87, \chi^2=10.3448, p=0.0013$.
5. **Frontend UI**: `frontend/index.html` updated with dynamic elements `id="mcnemar-fd4-*"` and fallback values.
6. **Backend API**: Route `/api/v1/slices/gates` returns authoritative `mcnemar_fd4` dictionary.
7. **Integrity Test**: `tests/test_phase_f_integrity.py::test_fd4_mcnemar_authoritative_verification` directly loads `fd4_predictions.parquet` and asserts all 7 parameters.

---

## 5. Invariant Confirmations

- **No experiment rerun**: Baseline and hardened runs preserved.
- **No dataset regenerated**: F-D1 through F-D4 hashes/sizes preserved.
- **No model retrained**: Artifacts `models/baseline_detector.joblib` and `models/hardened_slice_detector.joblib` preserved.
- **Locked seeds preserved**: 6001, 6002, 6003, 6004 unchanged.
- **Success gates**: All 5 locked gates (G1–G5) evaluate to PASS.
- **Phases E.1 & E.2**: Zero modifications (`git diff` confirms 0 lines modified).
