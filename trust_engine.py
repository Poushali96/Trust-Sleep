
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SUBTYPE_NAMES = ("obstructive", "central", "mixed", "indeterminate")
SIGNAL_NAMES = ("chest", "flow", "pulse", "spo2")


@dataclass
class PhysiologicalAudit:
    predicted_subtype: str
    status: str
    rule_consistency: float
    airflow_reduction: float
    preserved_effort: float
    reduced_effort: float
    effort_return: float
    delayed_desaturation: float
    pulse_recovery: float
    evidence: List[str]
    conflicts: List[str]


@dataclass
class TrustScores:
    cdt: float
    cera: float
    cas: float
    cdt_level: str
    cera_level: str
    cas_level: str
    signal_quality: float
    domain_similarity: float
    uncertainty_reliability: float
    binary_conformal_reliability: float
    subtype_conformal_reliability: float
    explanation_stability: float
    temporal_consistency: float
    physiological_plausibility: float
    deletion_faithfulness: float
    rule_consistency: float


@dataclass
class ClinicalDecision:
    status: str
    review_required: bool
    reason_codes: List[str]
    reasons: List[str]


@dataclass
class ExplainedPrediction:
    binary_label: str
    binary_probability: float
    binary_conformal_set: List[str]
    subtype_hypothesis: str
    subtype_probability: float
    subtype_alternatives: Dict[str, float]
    subtype_conformal_set: List[str]
    subtype_externally_transferred: bool
    subtype_locally_validated: bool
    audit: PhysiologicalAudit
    trust: TrustScores
    patient_context: List[str]
    decision: ClinicalDecision
    doctor_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def trust_level(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.70:
        return "moderate"
    if score >= 0.55:
        return "low"
    return "very low"


def _robust_scale(channel: np.ndarray) -> np.ndarray:
    channel = np.asarray(channel, dtype=float)
    median = np.nanmedian(channel)
    mad = np.nanmedian(np.abs(channel - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        scale = np.nanstd(channel)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return np.nan_to_num((channel - median) / scale)


def signal_quality_components(signal: np.ndarray) -> Tuple[float, Dict[str, float]]:
    scores: Dict[str, float] = {}
    for index, name in enumerate(SIGNAL_NAMES):
        x = np.asarray(signal[:, index], dtype=float)
        finite = np.isfinite(x).mean()
        spread = np.nanstd(x)
        flat_score = clamp01(spread / (spread + 1e-3))
        dx = np.diff(np.nan_to_num(x))
        if len(dx):
            median_dx = np.median(np.abs(dx)) + 1e-6
            spike_fraction = np.mean(np.abs(dx) > 12 * median_dx)
        else:
            spike_fraction = 1.0
        artifact_score = 1.0 - clamp01(spike_fraction / 0.20)
        scores[name] = clamp01(0.45 * finite + 0.30 * flat_score + 0.25 * artifact_score)

    # Airflow and effort are indispensable for subtype inference.
    overall = (
        0.35 * scores["flow"]
        + 0.35 * scores["chest"]
        + 0.15 * scores["spo2"]
        + 0.15 * scores["pulse"]
    )
    return clamp01(overall), scores


def physiological_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Continuous visualization/audit features, not AASM scoring rules.
    """
    chest = _robust_scale(signal[:, 0])
    flow = _robust_scale(signal[:, 1])
    pulse = _robust_scale(signal[:, 2])
    spo2 = _robust_scale(signal[:, 3])
    n = len(signal)

    flow_envelope = np.abs(flow)
    chest_envelope = np.abs(chest)

    baseline_end = max(2, n // 3)
    event_start = max(1, n // 3)
    event_end = max(event_start + 1, 2 * n // 3)
    recovery_start = event_end

    flow_base = np.median(flow_envelope[:baseline_end]) + 1e-6
    flow_event = np.median(flow_envelope[event_start:event_end])
    chest_base = np.median(chest_envelope[:baseline_end]) + 1e-6
    chest_event = np.median(chest_envelope[event_start:event_end])

    airflow_reduction = clamp01(1.0 - flow_event / flow_base)
    effort_ratio = chest_event / chest_base
    preserved_effort = clamp01(effort_ratio)
    reduced_effort = clamp01(1.0 - effort_ratio)

    midpoint = event_start + max(1, (event_end - event_start) // 2)
    early_effort = np.median(chest_envelope[event_start:midpoint]) + 1e-6
    late_effort = np.median(chest_envelope[midpoint:event_end]) + 1e-6
    effort_return = clamp01((late_effort - early_effort) / (chest_base + 1e-6))

    baseline_spo2 = np.median(spo2[:baseline_end])
    recovery_spo2 = np.min(spo2[event_end:]) if event_end < n else np.min(spo2)
    delayed_desaturation = clamp01((baseline_spo2 - recovery_spo2) / 3.0)

    baseline_pulse = np.median(pulse[:baseline_end])
    recovery_pulse = np.max(pulse[recovery_start:]) if recovery_start < n else np.max(pulse)
    pulse_recovery = clamp01((recovery_pulse - baseline_pulse) / 3.0)

    return {
        "airflow_reduction": airflow_reduction,
        "preserved_effort": preserved_effort,
        "reduced_effort": reduced_effort,
        "effort_return": effort_return,
        "delayed_desaturation": delayed_desaturation,
        "pulse_recovery": pulse_recovery,
    }


def physiological_audit(signal: np.ndarray, predicted_subtype: str) -> PhysiologicalAudit:
    f = physiological_features(signal)
    evidence: List[str] = []
    conflicts: List[str] = []

    if f["airflow_reduction"] >= 0.45:
        evidence.append("Airflow decreased markedly during the candidate event.")
    else:
        conflicts.append("A marked airflow reduction was not clearly demonstrated.")

    if f["preserved_effort"] >= 0.55:
        evidence.append("Thoracic respiratory effort remained present during reduced airflow.")
    if f["reduced_effort"] >= 0.55:
        evidence.append("Thoracic effort decreased together with airflow.")
    if f["effort_return"] >= 0.35:
        evidence.append("Respiratory effort increased again before the event fully resolved.")
    if f["delayed_desaturation"] >= 0.35:
        evidence.append("Oxygen saturation decreased after the airflow disturbance.")
    if f["pulse_recovery"] >= 0.35:
        evidence.append("Pulse accelerated during or after recovery.")

    if predicted_subtype == "obstructive":
        consistency = np.sqrt(f["airflow_reduction"] * f["preserved_effort"])
        if f["preserved_effort"] < 0.45:
            conflicts.append("Continued respiratory effort was not clear enough for an obstructive pattern.")
    elif predicted_subtype == "central":
        consistency = np.sqrt(f["airflow_reduction"] * f["reduced_effort"])
        if f["reduced_effort"] < 0.45:
            conflicts.append("Simultaneous loss of respiratory effort was not clear enough for a central pattern.")
    elif predicted_subtype == "mixed":
        consistency = (
            f["airflow_reduction"]
            * f["reduced_effort"]
            * max(f["effort_return"], 0.05)
        ) ** (1.0 / 3.0)
        if f["effort_return"] < 0.30:
            conflicts.append("A central-to-obstructive effort transition was not clearly visible.")
    else:
        consistency = 0.5

    consistency = clamp01(consistency)
    status = "consistent" if consistency >= 0.65 and not conflicts else (
        "partially_consistent" if consistency >= 0.40 else "inconsistent"
    )
    return PhysiologicalAudit(
        predicted_subtype=predicted_subtype,
        status=status,
        rule_consistency=consistency,
        evidence=evidence,
        conflicts=conflicts,
        **f,
    )


def conformal_reliability(set_size: int, number_of_classes: int) -> float:
    if set_size <= 0:
        return 0.0
    if set_size == 1:
        return 1.0
    if number_of_classes <= 1:
        return 0.0
    return clamp01(1.0 - (set_size - 1) / number_of_classes)


def uncertainty_reliability(variance: float, reference: float) -> float:
    reference = max(float(reference), 1e-8)
    return clamp01(np.exp(-float(variance) / reference))


def domain_similarity(
    embedding: np.ndarray,
    mean: np.ndarray,
    precision: np.ndarray,
    q95: float,
    q99: float,
) -> Tuple[float, str, float]:
    delta = np.asarray(embedding, dtype=float) - np.asarray(mean, dtype=float)
    distance = float(np.sqrt(max(delta @ precision @ delta, 0.0)))
    if distance <= q95:
        score = 1.0 - 0.20 * distance / max(q95, 1e-6)
        status = "in_distribution"
    elif distance <= q99:
        score = 0.80 - 0.40 * (distance - q95) / max(q99 - q95, 1e-6)
        status = "near_distribution_boundary"
    else:
        score = 0.40 * np.exp(-(distance - q99) / max(q99, 1e-6))
        status = "out_of_distribution"
    return clamp01(score), status, distance


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    numerator = float(np.dot(a, b))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-12:
        return 0.0
    return clamp01((numerator / denominator + 1.0) / 2.0)


@torch.inference_mode()
def explanation_stability_score(
    model,
    signal_tensor: torch.Tensor,
    metadata_tensor: torch.Tensor,
    noise_std: float = 0.03,
    repetitions: int = 5,
) -> float:
    model.eval()
    base = model(signal_tensor, metadata_tensor).contributions.abs().cpu().numpy()[0]
    similarities = []
    for _ in range(repetitions):
        perturbed = signal_tensor + torch.randn_like(signal_tensor) * noise_std
        contribution = model(perturbed, metadata_tensor).contributions.abs().cpu().numpy()[0]
        similarities.append(cosine_similarity(base, contribution))
    return float(np.mean(similarities))


@torch.inference_mode()
def deletion_faithfulness_score(
    model,
    signal_tensor: torch.Tensor,
    metadata_tensor: torch.Tensor,
    top_fraction: float = 0.25,
) -> float:
    """
    Zero the most influential event descriptors and measure the binary probability change.
    """
    model.eval()
    output = model(signal_tensor, metadata_tensor)
    base_probability = torch.sigmoid(output.binary_logit)

    raw = model.raw(signal_tensor)
    contributions = output.contributions.clone()
    k = max(1, int(contributions.shape[1] * top_fraction))
    top_indices = contributions.abs().topk(k, dim=1).indices
    masked = contributions.clone()
    masked.scatter_(1, top_indices, 0.0)
    event = model.event_encoder(masked)
    signal_embedding = model.signal_projection(torch.cat([raw, event], dim=1))
    masked_probability = torch.sigmoid(model.binary_head(signal_embedding).squeeze(-1))

    change = torch.abs(base_probability - masked_probability).mean().item()
    # Map a probability change of 0.20 or more close to 1.
    return clamp01(change / 0.20)


def temporal_consistency_score(
    current_contribution: np.ndarray,
    previous_contribution: Optional[np.ndarray],
) -> float:
    if previous_contribution is None:
        return 0.5
    return cosine_similarity(current_contribution, previous_contribution)


def patient_context_notes(metadata: Mapping[str, Any]) -> List[str]:
    notes: List[str] = []
    age = metadata.get("age")
    bmi = metadata.get("bmi")

    if age is not None:
        age = float(age)
        if age < 18:
            notes.append("Pediatric context requires pediatric scoring rules and specialist review.")
        elif age >= 75:
            notes.append("Advanced age increases the importance of comorbidity, baseline oxygenation, and artifact review.")

    if bmi is not None and float(bmi) >= 30:
        notes.append("Elevated BMI supports an obstructive clinical context but does not determine subtype.")
    if metadata.get("heart_failure"):
        notes.append("Heart failure increases the clinical relevance of central or periodic breathing.")
    if metadata.get("stroke_history"):
        notes.append("Stroke history increases concern for altered respiratory control.")
    if metadata.get("opioid_use"):
        notes.append("Opioid exposure increases concern for central respiratory pauses.")
    if float(metadata.get("altitude_m", 0) or 0) >= 2500:
        notes.append("High-altitude exposure can increase central breathing instability.")
    if metadata.get("tonsillar_hypertrophy"):
        notes.append("Tonsillar hypertrophy supports an obstructive context, especially in pediatric patients.")
    return notes


def compute_trust_scores(
    binary_probability: float,
    binary_set_size: int,
    subtype_probability: float,
    subtype_set_size: int,
    mc_variance: float,
    variance_reference: float,
    signal_quality: float,
    domain_similarity_score: float,
    explanation_stability: float,
    temporal_consistency: float,
    audit: PhysiologicalAudit,
    deletion_faithfulness: float,
) -> TrustScores:
    prediction_confidence = max(binary_probability, 1.0 - binary_probability)
    uncertainty_rel = uncertainty_reliability(mc_variance, variance_reference)
    binary_conf_rel = conformal_reliability(binary_set_size, 2)
    subtype_conf_rel = conformal_reliability(subtype_set_size, 4)

    cdt = (
        0.25 * prediction_confidence
        + 0.15 * subtype_probability
        + 0.15 * uncertainty_rel
        + 0.15 * binary_conf_rel
        + 0.10 * subtype_conf_rel
        + 0.10 * signal_quality
        + 0.10 * domain_similarity_score
    )
    physiological_plausibility = clamp01(
        0.55 * audit.rule_consistency
        + 0.20 * audit.delayed_desaturation
        + 0.15 * audit.pulse_recovery
        + 0.10 * audit.airflow_reduction
    )
    cera = (
        0.20 * explanation_stability
        + 0.15 * temporal_consistency
        + 0.25 * physiological_plausibility
        + 0.25 * deletion_faithfulness
        + 0.15 * audit.rule_consistency
    )
    cas = 0.55 * cdt + 0.45 * cera

    return TrustScores(
        cdt=clamp01(cdt),
        cera=clamp01(cera),
        cas=clamp01(cas),
        cdt_level=trust_level(cdt),
        cera_level=trust_level(cera),
        cas_level=trust_level(cas),
        signal_quality=clamp01(signal_quality),
        domain_similarity=clamp01(domain_similarity_score),
        uncertainty_reliability=clamp01(uncertainty_rel),
        binary_conformal_reliability=clamp01(binary_conf_rel),
        subtype_conformal_reliability=clamp01(subtype_conf_rel),
        explanation_stability=clamp01(explanation_stability),
        temporal_consistency=clamp01(temporal_consistency),
        physiological_plausibility=clamp01(physiological_plausibility),
        deletion_faithfulness=clamp01(deletion_faithfulness),
        rule_consistency=clamp01(audit.rule_consistency),
    )


def clinical_decision(
    binary_label: str,
    binary_set: Sequence[str],
    subtype_set: Sequence[str],
    subtype_probability: float,
    audit: PhysiologicalAudit,
    trust: TrustScores,
    domain_status: str,
    metadata: Mapping[str, Any],
) -> ClinicalDecision:
    codes: List[str] = []
    reasons: List[str] = []

    if binary_label != "apnea":
        codes.append("BINARY_EVENT_NOT_APNEA")
        reasons.append("The locally calibrated binary model did not classify this window as apnea.")
    if list(binary_set) != ["apnea"]:
        codes.append("BINARY_CONFORMAL_AMBIGUITY")
        reasons.append("The binary conformal prediction set was not the singleton set {apnea}.")
    if len(subtype_set) != 1:
        codes.append("SUBTYPE_CONFORMAL_AMBIGUITY")
        reasons.append("The subtype conformal prediction set contained more than one mechanism.")
    if subtype_probability < 0.85:
        codes.append("SUBTYPE_CONFIDENCE_BELOW_POLICY")
        reasons.append("Transferred subtype confidence was below the assisted-review threshold.")
    if trust.cdt < 0.85:
        codes.append("CDT_BELOW_HIGH_TRUST")
        reasons.append("Clinical decision trust was below the high-trust threshold.")
    if trust.cera < 0.75:
        codes.append("CERA_BELOW_POLICY")
        reasons.append("Explanation reliability was below the assisted-review threshold.")
    if trust.cas < 0.82:
        codes.append("CAS_BELOW_POLICY")
        reasons.append("Combined assurance was below the assisted-review threshold.")
    if audit.status != "consistent":
        codes.append("PHYSIOLOGY_NOT_FULLY_CONSISTENT")
        reasons.append("The subtype hypothesis was not fully consistent with the waveform audit.")
    if trust.signal_quality < 0.80:
        codes.append("SIGNAL_QUALITY_LOW")
        reasons.append("Required airflow or respiratory-effort signal quality was insufficient.")
    if domain_status == "out_of_distribution":
        codes.append("OUT_OF_DISTRIBUTION")
        reasons.append("The signal representation was unlike the external subtype-training distribution.")
    if metadata.get("age") is not None and float(metadata["age"]) < 18:
        codes.append("PEDIATRIC_SPECIALIST_REVIEW")
        reasons.append("Pediatric cases require specialist review and pediatric scoring rules.")

    # External subtype hypotheses always require clinician confirmation.
    codes.append("LOCAL_SUBTYPE_GROUND_TRUTH_UNAVAILABLE")
    reasons.append("The subtype was transferred from external PSG data and lacks local subtype validation.")

    if (
        binary_label == "apnea"
        and list(binary_set) == ["apnea"]
        and len(subtype_set) == 1
        and subtype_probability >= 0.85
        and trust.cdt >= 0.85
        and trust.cera >= 0.75
        and trust.cas >= 0.82
        and audit.status == "consistent"
        and trust.signal_quality >= 0.80
        and domain_status != "out_of_distribution"
    ):
        status = "assisted_review"
    elif binary_label == "apnea" and list(binary_set) == ["apnea"]:
        status = "binary_event_supported_subtype_review_required"
    else:
        status = "clinician_review"

    return ClinicalDecision(
        status=status,
        review_required=True,
        reason_codes=codes,
        reasons=reasons,
    )


def doctor_summary(
    binary_label: str,
    subtype: str,
    subtype_probability: float,
    alternatives: Mapping[str, float],
    subtype_set: Sequence[str],
    audit: PhysiologicalAudit,
    trust: TrustScores,
    context: Sequence[str],
    decision: ClinicalDecision,
    domain_status: str,
) -> str:
    if binary_label != "apnea":
        return (
            "The current window was not classified as apnea by the locally calibrated binary model. "
            "Clinical review remains appropriate when symptoms or waveform findings remain concerning."
        )

    opening = (
        f"An apnea event was detected. The transferred subtype model considers the waveform "
        f"most compatible with a {subtype} pattern ({subtype_probability:.0%})."
    )

    evidence = " ".join(audit.evidence[:4])
    if not evidence:
        evidence = "The available waveform evidence was limited."

    alternative_text = ", ".join(
        f"{name} {probability:.0%}"
        for name, probability in sorted(
            alternatives.items(), key=lambda item: item[1], reverse=True
        )
        if name != subtype
    )
    context_text = " ".join(context[:2]) if context else (
        "No recorded patient-context factor materially changed the interpretation."
    )
    trust_text = (
        f"CDT was {trust.cdt:.2f} ({trust.cdt_level}), CERA was {trust.cera:.2f} "
        f"({trust.cera_level}), and CAS was {trust.cas:.2f} ({trust.cas_level})."
    )
    conformal_text = (
        f"The external-calibrated subtype conformal set was "
        f"{{{', '.join(subtype_set)}}}."
    )
    decision_text = decision.status.replace("_", " ").capitalize() + "."
    limitation = (
        "The subtype is an externally transferred hypothesis and requires clinician confirmation "
        "because local obstructive, central, and mixed ground-truth labels are unavailable."
    )
    return " ".join([
        opening, evidence, f"Alternative hypotheses were {alternative_text}.",
        conformal_text, trust_text,
        f"Domain status was {domain_status.replace('_', ' ')}.",
        context_text, decision_text, limitation,
    ])
