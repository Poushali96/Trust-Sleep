
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
import json
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from calibration import (
    BinaryMondrianConformal,
    BinaryPlattScaler,
    MulticlassMondrianConformal,
    MulticlassTemperatureScaler,
)
from datasets import SignalStandardizer, metadata_vector
from metadata_scaler import MetadataStandardizer
from model_v4 import HierarchicalTrustSleepV4, SUBTYPE_CLASSES
from session_state import SessionStateStore
from alert_manager import AlertManager
from trust_engine import (
    clamp01,
    cosine_similarity,
    patient_context_notes,
    physiological_audit,
    signal_quality_components,
    trust_level,
)
from llm_explainer import deterministic_explanation, explain_with_endpoint


@dataclass
class PredictionV5:
    case_id: str
    session_id: str
    model_bundle_version: str
    binary_label: str
    binary_probability: float
    binary_conformal_set: List[str]
    subtype_hypothesis: str
    signal_only_subtype: str
    subtype_probability: float
    subtype_probabilities: Dict[str, float]
    subtype_conformal_set: List[str]
    context_changed_class: bool
    event_start_index: Optional[int]
    event_end_index: Optional[int]
    localization_peak: float
    ensemble_variance: float
    domain_similarity: float
    domain_status: str
    signal_quality: float
    cdt: float
    cera: float
    cas: float
    cdt_level: str
    cera_level: str
    cas_level: str
    physiological_audit: Dict[str, Any]
    patient_context: List[str]
    decision_status: str
    review_reasons: List[str]
    alert: Optional[Dict[str, Any]]
    explanation_payload: Dict[str, Any]
    deterministic_explanation: Dict[str, Any]
    llm_explanation: Optional[Dict[str, Any]] = None
    subtype_locally_validated: bool = False
    subtype_externally_transferred: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrustSleepDeploymentEngine:
    """
    Thread-safe, ensemble-calibrated inference engine.

    The ensemble bundle contains calibrators fitted to the ensemble mean, avoiding
    the invalid practice of applying one member's calibrator to an ensemble output.
    """
    def __init__(
        self,
        bundle_path: str,
        device: Optional[str] = None,
        llm_endpoint: Optional[str] = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.bundle = torch.load(bundle_path, map_location="cpu")
        self.bundle_path = str(bundle_path)
        self.bundle_version = str(self.bundle.get("bundle_version", "5.0.0"))
        self.llm_endpoint = llm_endpoint
        self._lock = threading.RLock()

        self.signal_standardizer = SignalStandardizer.from_dict(
            self.bundle["signal_standardizer"]
        )
        self.metadata_standardizer = MetadataStandardizer.from_dict(
            self.bundle["metadata_standardizer"]
        )
        self.binary_scaler = BinaryPlattScaler.from_dict(
            self.bundle["binary_platt"]
        )
        self.binary_conformal = BinaryMondrianConformal.from_dict(
            self.bundle["binary_conformal"]
        )
        self.subtype_scaler = MulticlassTemperatureScaler.from_dict(
            self.bundle["subtype_temperature"]
        )
        self.subtype_conformal = MulticlassMondrianConformal.from_dict(
            self.bundle["subtype_conformal"]
        )

        self.models = []
        bundle_directory = Path(self.bundle_path).resolve().parent
        for checkpoint_path in self.bundle["checkpoint_paths"]:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.is_absolute():
                checkpoint_path = bundle_directory / checkpoint_path
            payload = torch.load(checkpoint_path, map_location=self.device)
            model = HierarchicalTrustSleepV4(
                metadata_dim=24,
                max_context_weight=float(
                    payload.get("config", {}).get("max_context_weight", 0.20)
                ),
            ).to(self.device)
            model.load_state_dict(payload["model_state"])
            model.eval()
            self.models.append(model)

        self.state = SessionStateStore()
        self.alerts = AlertManager()

    def _prepare(
        self,
        signal: np.ndarray,
        metadata: Mapping[str, Any],
        modality_present: Optional[np.ndarray],
    ):
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim != 2 or signal.shape[1] != 4:
            raise ValueError(
                "signal must have shape [time,4]: chest, flow, pulse, SpO2"
            )
        if len(signal) < 8:
            raise ValueError("At least eight signal observations are required.")
        if not np.isfinite(signal).all():
            raise ValueError("Signal contains NaN or infinity.")

        present = (
            np.ones(4, dtype=np.float32)
            if modality_present is None
            else np.asarray(modality_present, dtype=np.float32)
        )
        if present.shape != (4,):
            raise ValueError("modality_present must have shape [4].")
        if not np.isin(present, [0, 1]).all():
            raise ValueError("modality_present values must be 0 or 1.")

        standardized = self.signal_standardizer.transform(signal)
        x = torch.from_numpy(standardized.T[None])
        x = F.interpolate(
            x, size=30, mode="linear", align_corners=False
        ).transpose(1, 2)

        meta = dict(metadata)
        meta.setdefault(
            "baseline_spo2",
            float(np.nanmedian(signal[:, 3])),
        )
        meta_vector = metadata_vector(meta)
        meta_vector = self.metadata_standardizer.transform(meta_vector)

        return (
            x.to(self.device),
            torch.from_numpy(meta_vector[None]).to(self.device),
            torch.from_numpy(present[None]).to(self.device),
        )

    @staticmethod
    def _localize(probability: np.ndarray, threshold: float = 0.5):
        active = probability >= threshold
        if not active.any():
            return None, None, float(probability.max())
        indices = np.flatnonzero(active)
        groups = np.split(
            indices,
            np.where(np.diff(indices) > 1)[0] + 1,
        )
        best = max(groups, key=len)
        return int(best[0]), int(best[-1] + 1), float(probability.max())

    def _domain(self, embedding: np.ndarray):
        reference = self.bundle["ood_reference"]
        mean = np.asarray(reference["mean"], dtype=float)
        precision = np.asarray(reference["precision"], dtype=float)
        delta = embedding - mean
        distance = float(np.sqrt(max(delta @ precision @ delta, 0.0)))
        q95 = float(reference["distance_q95"])
        q99 = float(reference["distance_q99"])

        if distance <= q95:
            return clamp01(1 - 0.20 * distance / max(q95, 1e-6)), "in_distribution"
        if distance <= q99:
            return (
                clamp01(
                    0.80
                    - 0.40 * (distance - q95) / max(q99 - q95, 1e-6)
                ),
                "near_distribution_boundary",
            )
        return (
            clamp01(0.40 * np.exp(-(distance - q99) / max(q99, 1e-6))),
            "out_of_distribution",
        )

    @torch.inference_mode()
    def predict(
        self,
        signal: np.ndarray,
        metadata: Mapping[str, Any],
        session_id: str,
        case_id: str,
        timestamp: Optional[float] = None,
        modality_present: Optional[np.ndarray] = None,
    ) -> PredictionV5:
        timestamp = float(timestamp if timestamp is not None else time.time())
        x, m, present = self._prepare(signal, metadata, modality_present)

        with self._lock:
            outputs = [
                model(x, m, present)
                for model in self.models
            ]

        raw_binary = np.asarray([
            float(torch.sigmoid(output.binary_logit).cpu()[0])
            for output in outputs
        ])
        ensemble_raw_binary = float(raw_binary.mean())
        binary_probability = float(
            self.binary_scaler.transform(
                np.asarray([ensemble_raw_binary])
            )[0]
        )

        subtype_logits = np.mean(
            [output.subtype_logits.cpu().numpy()[0] for output in outputs],
            axis=0,
        )
        signal_subtype_logits = np.mean(
            [
                output.signal_subtype_logits.cpu().numpy()[0]
                for output in outputs
            ],
            axis=0,
        )
        subtype_probabilities = self.subtype_scaler.transform_probabilities(
            subtype_logits[None]
        )[0]
        signal_subtype_probabilities = (
            self.subtype_scaler.transform_probabilities(
                signal_subtype_logits[None]
            )[0]
        )

        subtype_index = int(np.argmax(subtype_probabilities))
        signal_subtype_index = int(np.argmax(signal_subtype_probabilities))
        subtype = SUBTYPE_CLASSES[subtype_index]
        signal_subtype = SUBTYPE_CLASSES[signal_subtype_index]
        subtype_probability = float(subtype_probabilities[subtype_index])
        context_changed = subtype_index != signal_subtype_index

        threshold = float(self.bundle["operating_threshold"])
        binary_label = (
            "apnea" if binary_probability >= threshold else "no_apnea"
        )
        include0, include1, _ = self.binary_conformal.predict_sets(
            np.asarray([binary_probability])
        )
        binary_set = []
        if include0[0]:
            binary_set.append("no_apnea")
        if include1[0]:
            binary_set.append("apnea")

        subtype_indices = self.subtype_conformal.prediction_sets(
            subtype_probabilities[None]
        )[0]
        subtype_set = [SUBTYPE_CLASSES[index] for index in subtype_indices]

        localization = np.mean(
            [
                torch.sigmoid(output.localization_logits)
                .cpu()
                .numpy()[0]
                for output in outputs
            ],
            axis=0,
        )
        start, end, peak = self._localize(localization)
        if start is not None:
            start = int(round(start / 30 * len(signal)))
            end = int(round(end / 30 * len(signal)))

        event_signal = (
            signal[start:end]
            if start is not None and end is not None and end - start >= 8
            else signal
        )
        audit = physiological_audit(event_signal, subtype)
        quality, _ = signal_quality_components(signal)

        embeddings = np.asarray([
            output.signal_embedding.cpu().numpy()[0]
            for output in outputs
        ])
        mean_embedding = embeddings.mean(axis=0)
        domain_similarity, domain_status = self._domain(mean_embedding)

        subtype_member_probabilities = np.asarray([
            self.subtype_scaler.transform_probabilities(
                output.subtype_logits.cpu().numpy()
            )[0]
            for output in outputs
        ])
        ensemble_variance = float(
            raw_binary.var()
            + subtype_member_probabilities.var(axis=0).mean()
        )

        contribution = np.mean(
            [
                output.contributions.abs().cpu().numpy()[0]
                for output in outputs
            ],
            axis=0,
        )
        record = self.state.get(session_id)
        temporal = (
            0.5
            if record.previous_contribution is None
            else cosine_similarity(
                contribution,
                record.previous_contribution,
            )
        )
        self.state.update_contribution(
            session_id,
            contribution,
            timestamp,
        )

        binary_conformal_reliability = (
            1.0 if binary_set == ["apnea"] else 0.4 if len(binary_set) == 2 else 0.0
        )
        subtype_conformal_reliability = (
            1.0
            if len(subtype_set) == 1
            else max(0.0, 1 - (len(subtype_set)-1)/4)
        )
        uncertainty_reliability = clamp01(
            np.exp(-ensemble_variance / 0.02)
        )

        cdt = clamp01(
            0.22 * max(binary_probability, 1 - binary_probability)
            + 0.18 * subtype_probability
            + 0.15 * uncertainty_reliability
            + 0.15 * binary_conformal_reliability
            + 0.10 * subtype_conformal_reliability
            + 0.10 * quality
            + 0.10 * domain_similarity
        )
        cera = clamp01(
            0.20 * temporal
            + 0.30 * audit.rule_consistency
            + 0.20 * audit.airflow_reduction
            + 0.15 * audit.delayed_desaturation
            + 0.15 * (1.0 if not context_changed else 0.35)
        )
        cas = clamp01(0.55 * cdt + 0.45 * cera)

        policy = self.bundle["trust_policy"]
        reasons = []
        if binary_set != ["apnea"]:
            reasons.append("Binary conformal set is not the singleton {apnea}.")
        if len(subtype_set) != 1:
            reasons.append("Subtype conformal set is not a singleton.")
        if subtype_probability < float(
            policy["subtype_probability_threshold"]
        ):
            reasons.append(
                "Subtype probability is below the validation-selected threshold."
            )
        if cdt < float(policy["cdt_threshold"]):
            reasons.append("CDT is below the validation-selected threshold.")
        if cera < float(policy["cera_threshold"]):
            reasons.append("CERA is below the validation-selected threshold.")
        if cas < float(policy["cas_threshold"]):
            reasons.append("CAS is below the validation-selected threshold.")
        if context_changed:
            reasons.append(
                "Patient context changed the signal-only subtype class."
            )
        if audit.status != "consistent":
            reasons.append("Physiological audit is not fully consistent.")
        if quality < 0.80:
            reasons.append("Signal quality is below policy.")
        if domain_status == "out_of_distribution":
            reasons.append(
                "Signal is outside the external subtype-training distribution."
            )
        if present.cpu().numpy()[0, :2].min() == 0:
            reasons.append(
                "Airflow or thoracic-effort signal is unavailable."
            )
        reasons.append(
            "Local subtype ground truth is unavailable; clinician confirmation is required."
        )

        assisted = (
            binary_label == "apnea"
            and binary_set == ["apnea"]
            and len(subtype_set) == 1
            and subtype_probability
            >= float(policy["subtype_probability_threshold"])
            and cdt >= float(policy["cdt_threshold"])
            and cera >= float(policy["cera_threshold"])
            and cas >= float(policy["cas_threshold"])
            and not context_changed
            and audit.status == "consistent"
            and quality >= 0.80
            and domain_status != "out_of_distribution"
        )
        decision = (
            "assisted_review"
            if assisted
            else "binary_event_supported_subtype_review_required"
            if binary_label == "apnea" and binary_set == ["apnea"]
            else "clinician_review"
        )

        alert = self.alerts.update(
            session_id=session_id,
            event_active=binary_label == "apnea",
            subtype=subtype,
            probability=binary_probability,
            timestamp=timestamp,
        )

        context_notes = patient_context_notes(metadata)
        explanation_payload = {
            "subtype_hypothesis": subtype,
            "subtype_probability": subtype_probability,
            "audit": asdict(audit),
            "trust": {
                "cdt": cdt,
                "cera": cera,
                "cas": cas,
                "cdt_level": trust_level(cdt),
                "cera_level": trust_level(cera),
                "cas_level": trust_level(cas),
            },
            "decision": {"status": decision},
        }
        deterministic = deterministic_explanation(explanation_payload)

        return PredictionV5(
            case_id=case_id,
            session_id=session_id,
            model_bundle_version=self.bundle_version,
            binary_label=binary_label,
            binary_probability=binary_probability,
            binary_conformal_set=binary_set,
            subtype_hypothesis=subtype,
            signal_only_subtype=signal_subtype,
            subtype_probability=subtype_probability,
            subtype_probabilities=dict(
                zip(SUBTYPE_CLASSES, map(float, subtype_probabilities))
            ),
            subtype_conformal_set=subtype_set,
            context_changed_class=context_changed,
            event_start_index=start,
            event_end_index=end,
            localization_peak=peak,
            ensemble_variance=ensemble_variance,
            domain_similarity=domain_similarity,
            domain_status=domain_status,
            signal_quality=quality,
            cdt=cdt,
            cera=cera,
            cas=cas,
            cdt_level=trust_level(cdt),
            cera_level=trust_level(cera),
            cas_level=trust_level(cas),
            physiological_audit=asdict(audit),
            patient_context=context_notes,
            decision_status=decision,
            review_reasons=reasons,
            alert=asdict(alert) if alert else None,
            explanation_payload=explanation_payload,
            deterministic_explanation=asdict(deterministic),
        )

    def generate_llm_explanation(
        self,
        prediction: PredictionV5 | Mapping[str, Any],
        endpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = (
            prediction.explanation_payload
            if isinstance(prediction, PredictionV5)
            else dict(prediction).get(
                "explanation_payload",
                dict(prediction),
            )
        )
        return asdict(
            explain_with_endpoint(
                payload,
                endpoint=endpoint or self.llm_endpoint,
            )
        )

    @staticmethod
    def input_checksum(signal: np.ndarray) -> str:
        return sha256(
            np.asarray(signal, dtype=np.float32).tobytes()
        ).hexdigest()
