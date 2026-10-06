
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

EVENT_MAP = {
    "no_event": 0,
    "apnea": 1,
    "hypopnea": 2,
    "artifact_or_uncertain": 3,
}
SUBTYPE_MAP = {
    "obstructive": 0,
    "central": 1,
    "mixed": 2,
    "indeterminate": 3,
}
EVENT_NAMES = tuple(EVENT_MAP)
SUBTYPE_NAMES = tuple(SUBTYPE_MAP)

META_FIELDS = [
    "age", "bmi", "neck_circumference", "altitude_m", "baseline_spo2",
    "heart_failure", "stroke_history", "opioid_use", "smoking",
    "sedative_use", "pulmonary_disease", "tonsillar_hypertrophy",
]
LABEL_RE = re.compile(r"(?:^|[_/\-])(0|1)$")


@dataclass
class SignalStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, samples: Sequence["Sample"]) -> "SignalStandardizer":
        if not samples:
            raise ValueError("Cannot fit a standardizer with no samples.")
        total = np.zeros(4, dtype=np.float64)
        total_sq = np.zeros(4, dtype=np.float64)
        count = np.zeros(4, dtype=np.int64)

        for sample in samples:
            x = np.asarray(sample.signal, dtype=np.float64)
            finite = np.isfinite(x)
            total += np.where(finite, x, 0.0).sum(axis=0)
            total_sq += np.where(finite, x * x, 0.0).sum(axis=0)
            count += finite.sum(axis=0)

        if np.any(count == 0):
            raise ValueError("At least one signal channel has no finite training values.")

        mean = total / count
        variance = np.maximum(total_sq / count - mean * mean, 1e-8)
        std = np.sqrt(variance)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, signal: np.ndarray) -> np.ndarray:
        x = np.asarray(signal, dtype=np.float32)
        x = np.where(np.isfinite(x), x, self.mean[None, :])
        return ((x - self.mean[None, :]) / self.std[None, :]).astype(np.float32)

    def to_dict(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Sequence[float]]) -> "SignalStandardizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


@dataclass
class Sample:
    signal: np.ndarray
    metadata: np.ndarray
    metadata_raw: Dict[str, Any]
    binary_label: int
    event_label: int
    subtype_label: int
    domain_label: int
    sample_id: str
    subject_id: str
    recording_id: str


class Samples(Dataset):
    def __init__(self, samples: Sequence[Sample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        return {
            "signal": torch.from_numpy(sample.signal).float(),
            "metadata": torch.from_numpy(sample.metadata).float(),
            "binary_label": torch.tensor(sample.binary_label, dtype=torch.float32),
            "event_label": torch.tensor(sample.event_label, dtype=torch.long),
            "subtype_label": torch.tensor(sample.subtype_label, dtype=torch.long),
            "domain_label": torch.tensor(sample.domain_label, dtype=torch.long),
            "sample_id": sample.sample_id,
            "subject_id": sample.subject_id,
            "recording_id": sample.recording_id,
            "metadata_raw": sample.metadata_raw,
        }


def stable_subject_hash(subject_id: str) -> str:
    return sha256(str(subject_id).encode("utf-8")).hexdigest()


def metadata_vector(row: pd.Series | Mapping[str, Any]) -> np.ndarray:
    values: List[float] = []
    missing: List[float] = []
    for field in META_FIELDS:
        raw = row.get(field, np.nan)
        numeric = pd.to_numeric(raw, errors="coerce")
        absent = pd.isna(numeric)
        missing.append(float(absent))
        values.append(0.0 if absent else float(numeric))
    return np.asarray(values + missing, dtype=np.float32)


def infer_ids(key: str) -> Tuple[str, str]:
    """
    Fallback only. Explicit metadata is strongly preferred.

    The parser removes the terminal binary label and assumes the first remaining
    token identifies the subject. Verify this against the real HDF naming scheme.
    """
    stripped = re.sub(r"([_/\-])(0|1)$", "", key.strip("/"))
    tokens = [token for token in re.split(r"[/_\-]+", stripped) if token]
    subject_id = tokens[0] if tokens else stripped
    recording_id = "/".join(tokens[:-1]) if len(tokens) > 1 else stripped
    return subject_id, recording_id


def canonical_columns(frame: pd.DataFrame) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for column in frame.select_dtypes(include=[np.number]).columns:
        low = str(column).lower()
        if ("chest" in low or "thor" in low) and "chest" not in result:
            result["chest"] = column
        elif ("flow" in low or "nasal" in low or "airflow" in low) and "flow" not in result:
            result["flow"] = column
        elif ("pulse" in low or low.endswith("_hr") or "heart_rate" in low) and "pulse" not in result:
            result["pulse"] = column
        elif ("spo2" in low or "sao2" in low or "oxygen" in low) and "spo2" not in result:
            result["spo2"] = column
    return result


def _metadata_lookup(metadata_csv: Optional[str]) -> Optional[pd.DataFrame]:
    if not metadata_csv:
        return None
    path = Path(metadata_csv)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    metadata = pd.read_csv(path)
    if "key" not in metadata.columns:
        raise ValueError("Metadata CSV must contain a 'key' column.")
    return metadata.drop_duplicates("key").set_index("key")


def load_local_hdf(
    path: str,
    metadata_csv: Optional[str] = None,
    domain_label: int = 0,
) -> List[Sample]:
    metadata = _metadata_lookup(metadata_csv)
    samples: List[Sample] = []
    with pd.HDFStore(path, mode="r") as store:
        keys = sorted(store.keys())

    expected_length: Optional[int] = None
    for key in keys:
        match = LABEL_RE.search(key.rstrip("/"))
        if match is None:
            continue

        frame = pd.read_hdf(path, key=key)
        mapping = canonical_columns(frame)
        required = ("chest", "flow", "pulse", "spo2")
        if not all(name in mapping for name in required):
            continue

        signal = frame[[mapping[name] for name in required]].to_numpy(np.float32)
        if signal.ndim != 2 or signal.shape[0] < 4:
            continue
        if expected_length is None:
            expected_length = signal.shape[0]
        if signal.shape[0] != expected_length:
            continue
        signal = np.nan_to_num(signal)

        fallback_subject, fallback_recording = infer_ids(key)
        if metadata is not None and key in metadata.index:
            row = metadata.loc[key]
            subject_id = str(row.get("subject_id", fallback_subject))
            recording_id = str(row.get("recording_id", fallback_recording))
            raw_meta = row.to_dict()
        else:
            row = pd.Series(dtype=float)
            subject_id = fallback_subject
            recording_id = fallback_recording
            raw_meta = {}

        samples.append(Sample(
            signal=signal,
            metadata=metadata_vector(row),
            metadata_raw=raw_meta,
            binary_label=int(match.group(1)),
            event_label=-100,
            subtype_label=-100,
            domain_label=domain_label,
            sample_id=key,
            subject_id=subject_id,
            recording_id=recording_id,
        ))

    if not samples:
        raise RuntimeError(f"No valid local samples loaded from {path}")
    return samples


def load_external_manifest(
    manifest_path: str,
    domain_label: int = 1,
) -> List[Sample]:
    manifest = pd.read_csv(manifest_path)
    required = {
        "sample_id", "npz_path", "binary_label", "event_label",
        "subtype_label", "subject_id",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"External manifest missing columns: {sorted(missing)}")

    samples: List[Sample] = []
    for _, row in manifest.iterrows():
        payload = np.load(row["npz_path"])
        signal = np.asarray(payload["signal"], dtype=np.float32)
        if signal.ndim != 2 or signal.shape[1] != 4:
            raise ValueError(f"{row['npz_path']} must contain signal shaped [time, 4].")

        event_name = str(row["event_label"]).strip().lower()
        subtype_name = str(row["subtype_label"]).strip().lower()
        if event_name not in EVENT_MAP:
            raise ValueError(f"Unknown event label: {event_name}")
        if subtype_name not in SUBTYPE_MAP:
            raise ValueError(f"Unknown subtype label: {subtype_name}")

        samples.append(Sample(
            signal=np.nan_to_num(signal),
            metadata=metadata_vector(row),
            metadata_raw=row.to_dict(),
            binary_label=int(row["binary_label"]),
            event_label=EVENT_MAP[event_name],
            subtype_label=SUBTYPE_MAP[subtype_name],
            domain_label=domain_label,
            sample_id=str(row["sample_id"]),
            subject_id=str(row["subject_id"]),
            recording_id=str(row.get("recording_id", row["subject_id"])),
        ))

    if not samples:
        raise RuntimeError(f"No external samples loaded from {manifest_path}")
    return samples
