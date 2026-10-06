
from __future__ import annotations
from typing import Dict, List, Mapping, Sequence
import numpy as np


def signal_quality(signal: np.ndarray) -> Dict[str, str]:
    names = ("chest","flow","pulse","spo2")
    result = {}
    for i, name in enumerate(names):
        channel = signal[:, i]
        if not np.isfinite(channel).all():
            result[name] = "unusable"
        elif np.std(channel) < 1e-6:
            result[name] = "missing_or_flat"
        elif np.mean(np.abs(np.diff(channel))) > 10 * (np.median(np.abs(np.diff(channel))) + 1e-6):
            result[name] = "artifact_suspected"
        else:
            result[name] = "good"
    return result


def patient_context(meta: Mapping) -> List[str]:
    notes = []
    age = meta.get("age")
    bmi = meta.get("bmi")
    if age is not None:
        if age < 18:
            notes.append("Pediatric context: use pediatric scoring and specialist review.")
        elif age >= 75:
            notes.append("Advanced-age context: review comorbidity, baseline oxygenation, and artifacts.")
    if bmi is not None and bmi >= 30:
        notes.append("Elevated BMI increases obstructive-apnea context but does not determine subtype.")
    if meta.get("heart_failure"):
        notes.append("Heart failure raises the clinical relevance of central or periodic breathing.")
    if meta.get("stroke_history"):
        notes.append("Stroke history raises concern for altered respiratory control.")
    if meta.get("opioid_use"):
        notes.append("Opioid exposure raises concern for central respiratory pauses.")
    if meta.get("altitude_m", 0) >= 2500:
        notes.append("High-altitude exposure can increase central breathing instability.")
    if meta.get("tonsillar_hypertrophy"):
        notes.append("Tonsillar hypertrophy supports an obstructive context, especially in children.")
    return notes


def waveform_explanation(subtype: str, top_features: Sequence[str]) -> List[str]:
    joined = " ".join(top_features).lower()
    evidence = []
    if "flow" in joined:
        evidence.append("Airflow dynamics contributed strongly to the model output.")
    if "chest" in joined:
        evidence.append("Thoracic respiratory-effort dynamics contributed strongly.")
    if "spo2" in joined:
        evidence.append("Oxygen-saturation behavior contributed to the event interpretation.")
    if "pulse" in joined:
        evidence.append("Pulse response around the event contributed to the interpretation.")

    mechanism = {
        "obstructive": "The transferred subtype model found an obstructive-pattern representation, typically characterized by reduced airflow with preserved respiratory effort.",
        "central": "The transferred subtype model found a central-pattern representation, typically characterized by simultaneous reduction of airflow and respiratory effort.",
        "mixed": "The transferred subtype model found a mixed-pattern representation, which can include a central phase followed by obstructive effort.",
        "indeterminate": "The signal evidence did not support a reliable mechanism assignment.",
    }[subtype]
    return [mechanism] + evidence


def final_policy(
    event_confidence: float,
    subtype_confidence: float,
    variance: float,
    conformal_set: Sequence[str],
    quality: Mapping[str, str],
    meta: Mapping,
) -> Dict[str, str]:
    reasons = []
    if event_confidence < 0.85:
        reasons.append("event confidence below policy threshold")
    if subtype_confidence < 0.80:
        reasons.append("subtype confidence below policy threshold")
    if variance > 0.02:
        reasons.append("high model disagreement")
    if len(conformal_set) != 1:
        reasons.append("non-singleton subtype prediction set")
    if quality.get("flow") != "good":
        reasons.append("airflow channel inadequate")
    if quality.get("chest") != "good":
        reasons.append("respiratory-effort channel inadequate")
    if meta.get("age") is not None and meta["age"] < 18:
        reasons.append("pediatric case requires specialist review")

    return {
        "decision": "clinician_review" if reasons else "decision_support_permitted",
        "review_reason": "; ".join(reasons) if reasons else "locked acceptance policy satisfied",
        "disclaimer": "Screening decision support only; not a definitive diagnosis.",
    }
