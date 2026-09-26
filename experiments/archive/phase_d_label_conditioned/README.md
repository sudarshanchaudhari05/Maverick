# PRELIMINARY — LABEL-CONDITIONED PROVENANCE GENERATOR (NOT AUTHORITATIVE)

These archived artifacts represent the initial implementation of Phase D, in which `SyntheticProvenanceGenerator` conditioned provenance branch selection on `fraud_label`.

As identified during the Phase D final integrity audit:
- Poisoned/injected chains were only assigned when `is_fraud == True`.
- `instruction_source_present = 0` for all legitimate transactions ($P(\text{fraud} \mid \text{instruction\_source\_present}=1) = 1.0$).
- This created an artificial synthetic shortcut.

These results are preserved strictly for historical scientific transparency and are **INVALID FOR PRIMARY SCIENTIFIC COMPARISON**.
The authoritative, label-independent results reside in `experiments/provenance_chain_ablation_report.json`.
