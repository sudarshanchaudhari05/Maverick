"""Pydantic schemas for FraudForge AI REST API."""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# System & Health
# -----------------------------------------------------------------------------

class ModelStatus(BaseModel):
    name: str
    is_loaded: bool
    path: str
    size_kb: Optional[float] = None
    version: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "FraudForge AI API"
    mode: str = "simulation"
    model: str = "hardened"
    app_name: str = "FraudForge AI Backend"
    version: str = "1.0.0"
    active_model_version: str = "baseline_v1"
    hardening_round: int = 0
    active_model: Optional[ModelStatus] = None
    baseline_model: Optional[ModelStatus] = None
    hardened_model: Optional[ModelStatus] = None
    active_test_count: int = 72


class ModelRegistryHistoryEntry(BaseModel):
    version: str
    model_path: str
    hardening_round: int
    trained_attack_ids: List[str]
    created_at: str
    validation_metrics: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


class ModelRegistryResponse(BaseModel):
    active_version: str
    previous_version: Optional[str] = None
    hardening_round: int
    active_model_path: str
    baseline_model_path: str
    last_updated: str
    history: List[ModelRegistryHistoryEntry]


class ResetActiveModelResponse(BaseModel):
    status: str
    message: str
    active_version: str
    hardening_round: int


# -----------------------------------------------------------------------------
# Catalog & Genome
# -----------------------------------------------------------------------------

class AttackArchetypeSchema(BaseModel):
    attack_id: str
    name: str
    category: str
    severity: str
    payment_channel: str
    indicators: List[str]
    description: str
    genome: Dict[str, str]


class GenomeVocabularyResponse(BaseModel):
    dimensions: List[str]
    vocabulary: Dict[str, List[str]]
    descriptions: Dict[str, str]


# -----------------------------------------------------------------------------
# Attack Discovery & Blind-Spot Evaluation
# -----------------------------------------------------------------------------

class GenerateCandidateRequest(BaseModel):
    generation_method: str = Field(
        default="mutation",
        description="Method to generate candidate: 'mutation', 'crossover', or 'custom'",
    )
    custom_genome: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional full 10-gene dictionary for custom candidate evaluation",
    )
    parent_attack_id: Optional[str] = Field(
        default=None,
        description="Parent attack archetype ID for mutation (e.g., 'ATK-001')",
    )
    parent_1_id: Optional[str] = Field(
        default=None,
        description="First parent attack ID for crossover",
    )
    parent_2_id: Optional[str] = Field(
        default=None,
        description="Second parent attack ID for crossover",
    )
    n_mutations: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of gene mutations to apply",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility",
    )


class NovelCandidateScores(BaseModel):
    novelty_score: float
    realism_score: float
    evasion_potential: float
    priority_score: float


class LineageInfo(BaseModel):
    mutation_type: str
    parents: List[str]
    mutations: List[str]


class GenerateCandidateResponse(BaseModel):
    candidate_id: str
    candidate_name: str
    genome: Dict[str, str]
    scores: NovelCandidateScores
    lineage: LineageInfo
    nearest_known_archetype: str
    similarity_to_nearest: float


class EvaluateCandidateRequest(BaseModel):
    candidate_genome: Dict[str, str]
    candidate_name: Optional[str] = "Synthetic Attack Candidate"
    sample_count: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of synthetic transactions to simulate",
    )
    seed: Optional[int] = Field(
        default=42,
        description="Random seed for simulation",
    )
    detector_role: Optional[str] = Field(
        default="active",
        description="'active' (default, uses current promoted active detector) or 'baseline' (immutable baseline)",
    )


class SampleTransactionView(BaseModel):
    transaction_id: str
    transaction_amount: float
    device_change: int
    IP_risk_score: float
    merchant_risk_score: float
    transaction_velocity_1h: int
    is_detected: bool
    fraud_probability: float


class EvaluateCandidateResponse(BaseModel):
    candidate_name: str
    detector_role: str = "active"
    model_version: str = "baseline_v1"
    total_tested: int
    detected_count: int
    missed_count: int
    detection_rate_pct: float
    blind_spot_level: str
    sample_transactions: List[SampleTransactionView]


# -----------------------------------------------------------------------------
# Hardening & Generalization Benchmark
# -----------------------------------------------------------------------------

class EvolveGen2Request(BaseModel):
    parent_genome: Dict[str, str]
    candidate_id: Optional[str] = "NSA-001"
    candidate_name: Optional[str] = "Novel Discovered Attack"
    seed: Optional[int] = Field(
        default=2026,
        description="Evolution seed for Gen-2 variant synthesis",
    )


class EvolveGen2Response(BaseModel):
    variant_id: str
    variant_name: str
    genome: Dict[str, str]
    parent_id: str
    evolved_genes: List[str]


class CompareDefenseRequest(BaseModel):
    attack_genome: Dict[str, str]
    attack_name: Optional[str] = "Attack Scenario"
    target_attack_id: Optional[str] = None
    sample_count: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of transactions to test against both detectors",
    )
    seed: Optional[int] = Field(
        default=2026,
        description="Simulation seed",
    )
    auto_promote: bool = Field(
        default=True,
        description="Whether to automatically train, validate, and promote the hardened model as active detector upon success",
    )


class DetectorEvaluationStats(BaseModel):
    detected: int
    missed: int
    detection_rate_pct: float


class DefenseImprovementStats(BaseModel):
    generalization_gain_pct_points: float
    missed_attacks_reduction: int
    false_negative_reduction_pct: float


class CompareDefenseResponse(BaseModel):
    scenario_name: str
    total_simulated: int
    baseline_detector: DetectorEvaluationStats
    hardened_detector: DetectorEvaluationStats
    defense_improvement: DefenseImprovementStats
    promoted: bool = True
    validation_passed: bool = True
    active_version: str = "hardened_v1"
    previous_version: Optional[str] = "baseline_v1"
    validation_message: Optional[str] = None
    hardening_round: int = 1


# -----------------------------------------------------------------------------
# Risk Decision Engine & Production Mitigation
# -----------------------------------------------------------------------------

class EvaluateTransactionRequest(BaseModel):
    transaction_id: Optional[str] = "TX-ONLINE-999"
    transaction_amount: float = Field(..., ge=0.01)
    transaction_hour: int = Field(12, ge=0, le=23)
    account_age_days: int = Field(90, ge=0)
    device_age_days: int = Field(30, ge=0)
    device_change: int = Field(0, ge=0, le=1)
    IP_risk_score: float = Field(0.15, ge=0.0, le=1.0)
    merchant_risk_score: float = Field(0.20, ge=0.0, le=1.0)
    transaction_velocity_1h: int = Field(1, ge=0)
    transaction_velocity_24h: int = Field(2, ge=0)
    average_customer_amount: float = Field(50.0, ge=0.01)
    amount_deviation: float = Field(0.10, ge=0.0)
    geographic_deviation: int = Field(0, ge=0, le=1)
    behavioral_deviation: float = Field(0.05, ge=0.0, le=1.0)
    failed_authentication_count: int = Field(0, ge=0)
    identity_risk_score: float = Field(0.10, ge=0.0, le=1.0)
    merchant_category: str = Field("groceries")
    payment_channel: str = Field("e-commerce")
    authentication_method: str = Field("password")
    transaction_country: str = Field("US")
    customer_country: str = Field("US")
    policy_mode: str = Field("BALANCED", description="'BALANCED' or 'STRICT_SECURITY'")


class DetailedReason(BaseModel):
    code: str
    feature: str
    value: Any
    severity: str
    description: str


class MitigationPayload(BaseModel):
    status: str
    message: str
    requires_customer_action: bool
    challenge_type: Optional[str] = None
    recommended_verification_methods: List[str]


class EvaluateTransactionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_score: float
    risk_level: str
    action: str
    reason_codes: List[str]
    detailed_reasons: List[DetailedReason]
    mitigation: MitigationPayload
    model_version: str
    policy_version: str


# -----------------------------------------------------------------------------
# Conformal Calibration & Risk-Controlled Abstention
# -----------------------------------------------------------------------------

class ConformalCalibrateRequest(BaseModel):
    target_alpha: float = Field(default=0.05, ge=0.001, le=0.50, description="Significance level alpha for (1 - alpha) coverage")
    target_fnr: float = Field(default=0.05, ge=0.001, le=0.50, description="Target False Negative Rate budget for automatic decisions")
    calibration_sample_size: int = Field(default=1000, ge=100, le=10000, description="Number of calibration samples to use")


class ConformalCalibrateResponse(BaseModel):
    status: str = "ok"
    is_calibrated: bool
    target_alpha: float
    nominal_coverage: float
    target_fnr: float
    q_hat: float
    acceptance_threshold: float
    n_calibration_samples: int
    model_version: str
    calibrated_at: str
    exchangeability_disclaimer: str


class ConformalPredictRequest(BaseModel):
    transaction: Optional[Dict[str, Any]] = None
    transactions: Optional[List[Dict[str, Any]]] = None
    target_alpha: Optional[float] = Field(default=None, ge=0.001, le=0.50)
    target_fnr: Optional[float] = Field(default=None, ge=0.001, le=0.50)


class ConformalPredictionResult(BaseModel):
    transaction_id: str
    decision: str                   # "ALLOW", "BLOCK", "ABSTAIN"
    routing: str                    # "AUTOMATIC_DECISION", "HUMAN_REVIEW"
    is_abstained: bool
    fraud_probability: float
    prediction_set: List[int]
    prediction_set_size: int
    p_value_0: float
    p_value_1: float
    acceptance_threshold: float
    q_hat: float
    target_fnr: float
    reason: str


class ConformalPredictResponse(BaseModel):
    total_evaluated: int
    automatic_decisions: int
    human_reviews: int
    active_model_version: str
    results: List[ConformalPredictionResult]


class ConformalMetricsResponse(BaseModel):
    status: str = "ok"
    target_fnr: float
    observed_fnr: float
    fnr_among_accepted: float
    coverage: float
    abstention_rate: float
    total_evaluated: int
    automatic_decision_count: int
    human_review_count: int
    accepted_count: int
    blocked_count: int
    false_negatives: int
    false_positives: int
    true_positives: int
    true_negatives: int
    precision_auto: float
    recall_auto: float
    q_hat: float
    active_model_version: str
    exchangeability_disclaimer: str

