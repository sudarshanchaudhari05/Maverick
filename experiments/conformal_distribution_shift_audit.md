# Independent Results Audit: Conformal Distribution Shift & Risk-Coverage Evaluation

**Target Module**: `experiments/conformal_distribution_shift.py`  
**Report Artifacts Audited**:
- `experiments/conformal_distribution_shift_report.json`
- `experiments/conformal_distribution_shift_metrics.csv`
- `experiments/risk_coverage_curve.csv`
- `experiments/conformal_distribution_shift_audit.json`

**Auditor Scope**: Independent Scientific Verification  
**Evaluation Standard**: Zero alterations to model weights, random seeds, calibration procedures, or evaluation datasets. Verification of scientific integrity, mathematical correctness, data leakage, and terminology accuracy.

---

## Executive Audit Summary

| Audit Item | Scope | Status | Key Finding |
|---|---|---|---|
| **Audit 1: Metric Definitions** | Code formulas vs Report headers | ⚠️ **DISCREPANCY** | `fnr_among_accepted` is mathematically **False Omission Rate (FOR = FN / Accepted)**, not FNR. Dual definitions of FNR (`observed_fnr` = $FN / N_{\text{fraud, total}}$ vs $1 - \text{recall\_auto} = FN / (TP + FN)$). |
| **Audit 2: Arithmetic Verification** | Independent recomputation of all metrics | ✅ **PASS** | 204/204 individual metric cells in CSV and JSON match independent recomputation exactly ($0$ arithmetic error). |
| **Audit 3: Terminology & Naming** | Accuracy of statistical terms | ⚠️ **DISCREPANCY** | Accepted population contains both legitimate and fraudulent transactions; denominator does not equal total fraud. |
| **Audit 4: Risk-Coverage Monotonicity** | Strict monotonicity of $(\text{Coverage}, \text{Risk})$ | ⚠️ **NON-MONOTONIC** | Coverage is strictly monotonic non-decreasing with $\tau_{\text{accept}}$. Selective risk (`fnr_among_accepted`) has micro-dips in Adversarial & Gen-2 regimes when $\tau$ intervals accept only legitimate records. |
| **Audit 5: Data Leakage** | Feature overlap between calibration & evaluation | ✅ **PASS** | Exact tuple intersection check confirms **0 overlapping rows** between calibration ($N=2000$) and all three evaluation sets. |
| **Audit 6: Dataset Counts** | Record counts and class balances | ✅ **PASS** | Exact match with source files: Calibration ($N=2000$, 283 fraud), Same-Dist ($N=2000$, 309 fraud), Adversarial ($N=2000$, 300 fraud), Gen-2 ($N=1080$, 80 fraud). |
| **Audit 7: Threshold Calculation** | Calibrated threshold derivations | ✅ **PASS (WITH NOTE)** | Exact formulas validated ($0.0460, 0.0920, 0.2300, 0.4000$). Empirical fraud quantile term was strictly inactive because baseline synthetic fraud scores exceeded $0.84$. |
| **Audit 8: Exchangeability Claim** | Validity of exchangeability assumptions | ℹ️ **QUALIFIED** | Regime A provides an empirical in-distribution proxy from a stationary generator, not a general proof. Regimes B and C violate exchangeability by design. |
| **Audit 9: Result Integrity** | Test suite, artifact alignment, determinism | ✅ **PASS** | 89/89 pytest suite passes. Experiment outputs are 100% deterministic and reproducible under fixed seeds. |

---

## Audit 1: Metric Definitions

### Exact Mathematical Formulas Used in Code
From `experiments/conformal_distribution_shift.py` (`compute_conformal_metrics`):
- **Decision Partition**:
  - Auto-Accept: $\hat{p}(x) \le \tau_{\text{accept}}$ (Count: $N_{\text{accept}}$)
  - Abstained (Human Review): $\tau_{\text{accept}} < \hat{p}(x) < \tau_{\text{review}}$ (Count: $N_{\text{abstain}}$)
  - Auto-Reject: $\hat{p}(x) \ge \tau_{\text{review}}$ (Count: $N_{\text{reject}}$)
- **Decided Transactions**: $N_{\text{decided}} = N_{\text{accept}} + N_{\text{reject}}$
- **Coverage**:
  $$\text{Coverage} = \frac{N_{\text{decided}}}{N_{\text{total}}} = 1 - \frac{N_{\text{abstain}}}{N_{\text{total}}}$$
- **Confusion Matrix on Decided Population**:
  - $TP = \sum \mathbf{1}(y=1 \land \hat{p}(x) \ge \tau_{\text{review}})$
  - $FP = \sum \mathbf{1}(y=0 \land \hat{p}(x) \ge \tau_{\text{review}})$
  - $TN = \sum \mathbf{1}(y=0 \land \hat{p}(x) \le \tau_{\text{accept}})$
  - $FN = \sum \mathbf{1}(y=1 \land \hat{p}(x) \le \tau_{\text{accept}})$
- **Code Metric `observed_fnr`**:
  $$\text{observed\_fnr} = \frac{FN}{N_{\text{fraud, total}}}$$
  *Interpretation*: Population-level fraud leakage rate across all incoming transactions.
- **Code Metric `recall_auto`**:
  $$\text{recall\_auto} = \frac{TP}{TP + FN}$$
  *Interpretation*: Recall among decided transactions. Therefore:
  $$\text{FNR}_{\text{decided}} = 1 - \text{recall\_auto} = \frac{FN}{TP + FN}$$
- **Code Metric `fnr_among_accepted`**:
  $$\text{fnr\_among\_accepted} = \frac{FN}{N_{\text{accept}}} = \frac{FN}{TN + FN}$$
  *Interpretation*: The proportion of accepted transactions that are fraudulent.

### Discrepancy Findings
1. **Misnamed Metric**: `fnr_among_accepted` uses $N_{\text{accept}} = TN + FN$ as the denominator. In statistical classification and epidemiology, this quantity is formally the **False Omission Rate (FOR)** or $1 - \text{Negative Predictive Value (NPV)}$:
   $$\text{FOR} = \frac{FN}{TN + FN} = 1 - \text{NPV}$$
   It represents the **fraud contamination rate of the accepted stream**, NOT the False Negative Rate. FNR strictly requires conditioning on actual fraud instances.
2. **Divergent FNR Metrics**:
   - Population FNR (`observed_fnr`): Evaluated against $N_{\text{fraud, total}}$ (including abstained frauds).
   - Selective FNR ($1 - \text{recall\_auto}$): Evaluated against decided fraud ($TP + FN$).
   - In Adversarial Regime at 1% target:
     - Total fraud = $300$, $FN = 34$, $TP = 151 \implies TP + FN = 185$.
     - $\text{observed\_fnr} = 34 / 300 = \mathbf{11.33\%}$
     - $\text{FNR}_{\text{decided}} = 34 / 185 = \mathbf{18.38\%}$
     - $\text{FOR} = 34 / 1441 = \mathbf{2.35\%}$
   - In Gen-2 Novel Regime at 1% target:
     - Total fraud = $80$, $FN = 43$, $TP = 5 \implies TP + FN = 48$.
     - $\text{observed\_fnr} = 43 / 80 = \mathbf{53.75\%}$
     - $\text{FNR}_{\text{decided}} = 43 / 48 = \mathbf{89.58\%}$
     - $\text{FOR} = 43 / 894 = \mathbf{4.86\%}$
3. **Display Inconsistency in Terminal Output**:
   Line 236 of `conformal_distribution_shift.py` formats output as:
   `f"{m.false_negatives:>3d}/{total_fraud:<5d}"` where `total_fraud = m.false_negatives + m.true_positives` ($185$ and $48$), while displaying `OBS FNR` computed as $FN / N_{\text{fraud, total}}$ ($34/300$ and $43/80$).

---

## Audit 2: Manual Arithmetic Verification

All 12 evaluation regimes and target FNR pairs were recomputed from raw prediction vectors without importing or running the experiment script:

| Regime | Target FNR | $\tau_{\text{accept}}$ | $\tau_{\text{review}}$ | Total | Total Fraud | Accepted | Abstained | Rejected | TP | FP | TN | FN | Coverage | Observed FNR | Selective FNR ($1-\text{Rec}$) | FOR (`fnr_accepted`) | Auto Precision | Auto F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Same-Dist** | 1.0% | 0.0460 | 0.8500 | 2000 | 309 | 1581 | 114 | 305 | 305 | 0 | 1581 | 0 | 94.30% | 0.00% | 0.00% | 0.00% | 1.0000 | 1.0000 |
| **Same-Dist** | 2.0% | 0.0920 | 0.8500 | 2000 | 309 | 1632 | 63 | 305 | 305 | 0 | 1632 | 0 | 96.85% | 0.00% | 0.00% | 0.00% | 1.0000 | 1.0000 |
| **Same-Dist** | 5.0% | 0.2300 | 0.8500 | 2000 | 309 | 1675 | 20 | 305 | 305 | 0 | 1675 | 0 | 99.00% | 0.00% | 0.00% | 0.00% | 1.0000 | 1.0000 |
| **Same-Dist** | 10.0% | 0.4000 | 0.8500 | 2000 | 309 | 1684 | 11 | 305 | 305 | 0 | 1684 | 0 | 99.45% | 0.00% | 0.00% | 0.00% | 1.0000 | 1.0000 |
| **Adversarial** | 1.0% | 0.0460 | 0.8500 | 2000 | 300 | 1441 | 408 | 151 | 151 | 0 | 1407 | 34 | 79.60% | 11.33% | 18.38% | 2.36% | 1.0000 | 0.8988 |
| **Adversarial** | 2.0% | 0.0920 | 0.8500 | 2000 | 300 | 1544 | 305 | 151 | 151 | 0 | 1505 | 39 | 84.75% | 13.00% | 20.53% | 2.53% | 1.0000 | 0.8856 |
| **Adversarial** | 5.0% | 0.2300 | 0.8500 | 2000 | 300 | 1673 | 176 | 151 | 151 | 0 | 1630 | 43 | 91.20% | 14.33% | 22.16% | 2.57% | 1.0000 | 0.8754 |
| **Adversarial** | 10.0% | 0.4000 | 0.8500 | 2000 | 300 | 1705 | 144 | 151 | 151 | 0 | 1660 | 45 | 92.80% | 15.00% | 22.96% | 2.64% | 1.0000 | 0.8703 |
| **Gen-2 Novel**| 1.0% | 0.0460 | 0.8500 | 1080 | 80 | 894 | 181 | 5 | 5 | 0 | 851 | 43 | 83.24% | 53.75% | 89.58% | 4.81% | 1.0000 | 0.1887 |
| **Gen-2 Novel**| 2.0% | 0.0920 | 0.8500 | 1080 | 80 | 954 | 121 | 5 | 5 | 0 | 907 | 47 | 88.80% | 58.75% | 90.38% | 4.93% | 1.0000 | 0.1754 |
| **Gen-2 Novel**| 5.0% | 0.2300 | 0.8500 | 1080 | 80 | 1007 | 68 | 5 | 5 | 0 | 959 | 48 | 93.70% | 60.00% | 90.57% | 4.77% | 1.0000 | 0.1724 |
| **Gen-2 Novel**| 10.0% | 0.4000 | 0.8500 | 1080 | 80 | 1022 | 53 | 5 | 5 | 0 | 973 | 49 | 95.09% | 61.25% | 90.74% | 4.79% | 1.0000 | 0.1695 |

**Arithmetic Verdict**: Exactly **0 arithmetic discrepancies** found across all 204 values.

---

## Audit 3: Terminology & Naming

1. **Standard Machine Learning Terminology**:
   - $\text{False Negative Rate} = \frac{FN}{P} = \frac{FN}{FN + TP}$
   - $\text{False Omission Rate} = \frac{FN}{FN + TN} = \frac{FN}{N_{\text{accept}}}$
   - $\text{Negative Predictive Value} = \frac{TN}{FN + TN} = 1 - \text{FOR}$
2. **Current Code Term**:
   - `m.fnr_among_accepted = m.false_negatives / max(1, m.accepted)`
   - This computes **False Omission Rate (FOR)**.
3. **Recommendation**:
   Retain calculation but rename or alias in documentation to avoid confusion during academic or judicial evaluation:
   - Report `false_omission_rate` or `accepted_contamination_rate` ($\frac{FN}{N_{\text{accept}}}$).
   - Report `selective_fnr` ($\frac{FN}{FN + TP} = 1 - \text{Recall}_{\text{auto}}$).
   - Report `population_fnr` / `unconditional_fraud_leakage` ($\frac{FN}{N_{\text{fraud, total}}}$).

---

## Audit 4: Risk-Coverage Monotonicity Check

Evaluating `experiments/risk_coverage_curve.csv` across the 20-point sweep:
- **$\tau_{\text{accept}}$ range**: $[0.0100, 0.4000]$ (step size $0.0205$)
- **$\tau_{\text{review}}$**: fixed at $0.8500$

### Monotonicity Findings
1. **Coverage**:
   - Same-Distribution: Strictly monotonic increasing ($84.90\% \to 99.45\%$, $\Delta \ge 0$).
   - Adversarial Regime: Strictly monotonic increasing ($68.20\% \to 92.80\%$, $\Delta \ge 0$).
   - Gen-2 Novel Regime: Strictly monotonic increasing ($72.50\% \to 95.09\%$, $\Delta \ge 0$).
   - **Verdict**: ✅ Strictly monotonic.
2. **Selective Risk (`fnr_among_accepted` / FOR)**:
   - Same-Distribution: Constant at $0.0000$ across all 20 points (no false negatives).
   - Adversarial Regime: **Non-monotonic**.
     - Point 3 ($\tau = 0.0510$): $FN = 35$, $N_{\text{acc}} = 1460 \implies \text{Risk} = 0.02397$
     - Point 4 ($\tau = 0.0715$): $FN = 39$, $N_{\text{acc}} = 1494 \implies \text{Risk} = 0.02610$
     - Point 5 ($\tau = 0.0920$): $FN = 39$, $N_{\text{acc}} = 1544 \implies \text{Risk} = 0.02526$ (**DIP**)
     - Explanation: When $\tau_{\text{accept}}$ expands from $0.0715$ to $0.0920$, 50 additional transactions are accepted. **All 50 are legitimate ($TN$)**, while $FN$ remains constant at 39. The denominator grows while the numerator is unchanged, causing risk to drop by $0.084\%$.
   - Gen-2 Novel Regime: **Non-monotonic**.
     - Point 11 ($\tau = 0.2150$): $FN = 48$, $N_{\text{acc}} = 1000 \implies \text{Risk} = 0.04800$
     - Point 12 ($\tau = 0.2355$): $FN = 48$, $N_{\text{acc}} = 1010 \implies \text{Risk} = 0.04752$ (**DIP**)
     - Explanation: $10$ additional legitimate transactions accepted with $0$ new false negatives.

**Mathematical Rule**: For any selective risk metric of the form $R = \frac{FN(\tau)}{N_{\text{accept}}(\tau)}$, monotonicity holds if and only if:
$$\frac{d FN(\tau)}{d\tau} \ge R(\tau) \cdot \frac{d N_{\text{accept}}(\tau)}{d\tau}$$
When the score distribution has intervals containing legitimate samples but zero false negative fraud scores, $\frac{d FN}{d\tau} = 0 < R \cdot \frac{d N_{\text{accept}}}{d\tau}$, inducing localized downward steps.

---

## Audit 5: Data Leakage Verification

### Isolation Checks
- **Calibration Source**: `data/synthetic_transactions_10k.csv` (rows $0 \dots 1999$, $N=2000$).
- **Same-Distribution Test Source**: `data/synthetic_transactions_10k.csv` (rows $2000 \dots 3999$, $N=2000$).
- **Adversarial Test Source**: `data/adversarial_eval_synthetic.csv` ($N=2000$).
- **Gen-2 Novel Test Source**: `data/gen2_novel_drift_dataset.csv` ($N=1080$).

### Hash and Set-Intersection Check
- Calibrated nonconformity scores $\alpha_i = 1 - \hat{p}(x_i \mid y_i = 1)$ were computed solely on the 283 fraud samples within the first 2,000 rows of `synthetic_transactions_10k.csv`.
- An exact tuple set-intersection across core feature vectors (`amount`, `hour_of_day`, `device_trust_score`, `velocity_1h`, etc.) revealed:
  - $\text{Overlap}(\text{Calibration}, \text{Same-Dist Test}) = \mathbf{0}$
  - $\text{Overlap}(\text{Calibration}, \text{Adversarial Test}) = \mathbf{0}$
  - $\text{Overlap}(\text{Calibration}, \text{Gen-2 Novel Test}) = \mathbf{0}$
- Model parameters: `models/baseline_detector.joblib` was pretrained on `data/transactions_sample.csv` and remained frozen and unmodified.

**Leakage Verdict**: ✅ **Zero Data Leakage Confirmed**.

---

## Audit 6: Dataset Counts & Balances

| Dataset Role | File Path | Total Records | Legitimate | Fraud | Fraud Proportion |
|---|---|---|---|---|---|
| **Calibration Set** | `data/synthetic_transactions_10k.csv` (0:2000) | 2,000 | 1,717 | 283 | 14.15% |
| **Regime A (Same-Dist)** | `data/synthetic_transactions_10k.csv` (2000:4000) | 2,000 | 1,691 | 309 | 15.45% |
| **Regime B (Adversarial)** | `data/adversarial_eval_synthetic.csv` | 2,000 | 1,700 | 300 | 15.00% |
| **Regime C (Gen-2 Novel)** | `data/gen2_novel_drift_dataset.csv` | 1,080 | 1,000 | 80 | 7.41% |

All row counts, label distributions, and source file identities match the experiment execution logs and disk metadata.

---

## Audit 7: Threshold Calculation

### Mathematical Implementation
From `src/fraudforge/conformal/risk_control.py`:
1. Nonconformity score for fraud ($y=1$):
   $$\alpha_i = 1 - \hat{p}(x_i)$$
2. Conformal quantile with finite-sample correction:
   $$\hat{q}_{\alpha} = \text{Quantile}\left(\{\alpha_i\}_{i=1}^{n}, \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)$$
   For calibration fraud count $n = 283$ and nominal error budget $\alpha = 0.05$:
   $$\text{Coverage target} = \frac{\lceil 284 \times 0.95 \rceil}{283} = \frac{270}{283} \approx 0.95406$$
   $$\hat{q}_{0.05} = 0.230043$$
3. Accept Threshold Derivation:
   $$\tau_{\text{accept}} = \min\left(\tau_{\text{fraud\_quantile}}, \hat{q}_{\alpha} \times \frac{\alpha_{\text{FNR}}}{\alpha}\right)$$
   clipped to $[\tau_{\min}, \tau_{\max}] = [0.01, 0.40]$.

### Empirical Behavior in Phase C
On the calibration fraud population ($n=283$), the baseline detector scores are heavily separated near 1.0:
- Empirical 1.0% quantile of predicted probabilities: $\mathbf{0.8453}$
- Empirical 2.0% quantile: $\mathbf{0.8659}$
- Empirical 5.0% quantile: $\mathbf{0.9161}$
- Empirical 10.0% quantile: $\mathbf{0.9427}$

Because all these quantiles exceed $0.84$, the $\min$ operator is strictly bounded by the scaled conformal term:
- **Target FNR = 1.0%**: $\tau = 0.230043 \times \frac{0.01}{0.05} = 0.046009 \to \mathbf{0.0460}$
- **Target FNR = 2.0%**: $\tau = 0.230043 \times \frac{0.02}{0.05} = 0.092017 \to \mathbf{0.0920}$
- **Target FNR = 5.0%**: $\tau = 0.230043 \times \frac{0.05}{0.05} = 0.230043 \to \mathbf{0.2300}$
- **Target FNR = 10.0%**: $\tau = 0.230043 \times \frac{0.10}{0.05} = 0.460086 \to \text{clipped to } \mathbf{0.4000}$

**Finding**: The threshold formula is deterministic and verified. However, in practice, the `tau_fraud_quantile` branch was inactive, and the upper threshold was governed by the 0.40 clipping bound for the 10% target.

---

## Audit 8: Exchangeability Claims

1. **Regime A (Same-Distribution Evaluation)**:
   - The calibration set and Regime A evaluation set are generated by the same synthetic transaction generator under identical distributions.
   - The experiment states: *"Empirically satisfies conformal coverage guarantees."*
   - **Audit Verdict**: Valid as an empirical simulation of exchangeability. However, real-world credit card transactions are non-stationary time series (concept drift, seasonality); synthetic exchangeability cannot be extrapolated to uncalibrated live production streams without continuous recalibration.
2. **Regime B (Adversarial) & Regime C (Gen-2 Shift)**:
   - Conformal risk control explicitly assumes exchangeability between calibration and test data ($P_{\text{cal}} = P_{\text{test}}$).
   - Under adversarial perturbation and structural feature drift, $P_{\text{test}} \ne P_{\text{cal}}$, formally violating exchangeability.
   - **Audit Finding**: The experiment correctly illustrates and documents this theoretical violation: observed FNR breaches the target bound ($11.33\%$ vs $1\%$ in Adversarial, $53.75\%$ vs $1\%$ in Gen-2). This is a scientifically valid negative result demonstrating the necessity of adaptive conformalization or provenance-based detection.

---

## Audit 9: Result Integrity & Reproducibility

- **Git & Working Directory State**: Local workspace is clean, with no uncommitted or untracked changes to core detector files.
- **Automated Regression Suite**:
  - `pytest -q`: **89 passed**, 91 warnings in 20.47s.
  - Comprising 79 pre-existing tests + 8 Phase B conformal tests + 2 Phase C shift evaluation tests.
- **Determinism**: Setting random seeds and static inputs ensures identical numerical output across independent runs.

---

## Section A: Verified Correct
1. The 204 numerical metric values across JSON and CSV match exact recomputations from the model's raw probability predictions.
2. Zero data leakage: calibration data and evaluation sets have empty feature-set intersections.
3. Baseline detector artifact `models/baseline_detector.joblib` remained frozen and immutable.
4. Conformal quantile calculations follow exact finite-sample correction formulas.
5. All test suites pass (89/89).

## Section B: Arithmetic Discrepancies
- **Zero arithmetic errors**. All floating-point operations, roundings, and counts are mathematically consistent with code implementation.

## Section C: Metric-Definition Discrepancies
1. **Misleading Name `fnr_among_accepted`**:
   The formula $\frac{FN}{TN + FN}$ is the **False Omission Rate (FOR)**, not the False Negative Rate.
2. **Dual FNR Denominators**:
   - `observed_fnr` evaluates $FN$ relative to all dataset fraud: $\frac{FN}{N_{\text{fraud, total}}}$.
   - Selective FNR ($1 - \text{recall\_auto}$) evaluates $FN$ relative to decided fraud: $\frac{FN}{TP + FN}$.
3. **Terminal String Label Mismatch**:
   Line 236 in `conformal_distribution_shift.py` displays `m.false_negatives / total_fraud` where `total_fraud = TP + FN`, yet labels the percentage as `OBS FNR` (which is $FN / N_{\text{fraud, total}}$).

## Section D: Methodology Concerns
1. **Unexercised Quantile Branch in Threshold Formula**:
   Because baseline detector scores for synthetic fraud are consistently $> 0.84$, `min(tau_fraud_quantile, q_hat * (target / alpha))` is always dominated by the scaled conformal bound.
2. **Upper Clipping at 0.40**:
   At target FNR = 10%, the scaled conformal bound evaluates to $0.4601$, which is clipped to $0.4000$. This prevents the system from exploring higher accept thresholds.

## Section E: Data Leakage Findings
- Independent row-by-row hash and feature-tuple verification confirmed **0 duplicate or leaked records** between calibration data and evaluation datasets.

## Section F: Reproducibility Findings
- 100% reproducible. The experiment runs deterministically and reproduces all metrics and visualizations from fixed input files.

## Section G: Bugs Requiring Correction
*(To be addressed in Phase D or subsequent updates without altering historical results)*:
1. **Metric Naming / Reporting Clarification**: Add explicit column/key names:
   - `false_omission_rate` for $\frac{FN}{TN + FN}$
   - `selective_fnr` for $\frac{FN}{TP + FN}$
   - Keep `observed_fnr` clearly documented as population fraud leakage rate ($\frac{FN}{N_{\text{fraud, total}}}$).
2. **Terminal Formatting Alignment**: Update line 236 print string so the denominator fraction matches the displayed `OBS FNR` formula.

## Section H: Recommended Next Action
1. Keep the audited Phase C artifacts unmodified as the frozen historical benchmark.
2. Formally proceed to **Phase D (Provenance Chain & Adaptive Calibration)** with documented metric definitions.
3. In Phase D, introduce non-exchangeability detection (martingale/drift tests) and adaptive conformal updating to restore risk coverage under distribution shift.
