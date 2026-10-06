
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence
import numpy as np


@dataclass
class MetadataStandardizer:
    """
    Standardizes the observed-value half of the metadata vector while preserving
    the missingness-indicator half.

    The expected vector is [12 values, 12 missing indicators].
    Binary fields are still standardized because this prevents scale dominance,
    but missing indicators remain exactly 0/1.
    """
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, vectors: Sequence[np.ndarray]) -> "MetadataStandardizer":
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != 24:
            raise ValueError("Metadata vectors must have shape [n, 24].")
        values = matrix[:, :12]
        missing = matrix[:, 12:] > 0.5
        observed = np.where(missing, np.nan, values)
        mean = np.nanmean(observed, axis=0)
        std = np.nanstd(observed, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        values = vector[..., :12]
        missing = vector[..., 12:]
        standardized = (values - self.mean) / self.std
        standardized = np.where(missing > 0.5, 0.0, standardized)
        return np.concatenate([standardized, missing], axis=-1).astype(np.float32)

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MetadataStandardizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )
