
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Deque, Dict
import numpy as np


@dataclass
class DriftSummary:
    n: int
    mean_domain_similarity: float
    out_of_distribution_rate: float
    mean_signal_quality: float
    mean_binary_probability: float
    review_rate: float
    alert: bool


class DriftMonitor:
    """
    Operational drift monitor.

    Thresholds are policy defaults and must be tuned on a locked validation period.
    """
    def __init__(
        self,
        window_size: int = 1000,
        minimum_n: int = 100,
        minimum_domain_similarity: float = 0.55,
        maximum_ood_rate: float = 0.10,
        minimum_signal_quality: float = 0.70,
    ):
        self.window_size = int(window_size)
        self.minimum_n = int(minimum_n)
        self.minimum_domain_similarity = float(minimum_domain_similarity)
        self.maximum_ood_rate = float(maximum_ood_rate)
        self.minimum_signal_quality = float(minimum_signal_quality)
        self._records: Deque[Dict[str, float]] = deque(maxlen=self.window_size)
        self._lock = RLock()

    def update(self, prediction: Dict[str, object]) -> DriftSummary:
        record = {
            "domain_similarity": float(prediction["domain_similarity"]),
            "ood": float(prediction["domain_status"] == "out_of_distribution"),
            "signal_quality": float(prediction["signal_quality"]),
            "binary_probability": float(prediction["binary_probability"]),
            "review": float(prediction["decision_status"] != "assisted_review"),
        }
        with self._lock:
            self._records.append(record)
            return self.summary()

    def summary(self) -> DriftSummary:
        with self._lock:
            if not self._records:
                return DriftSummary(
                    n=0,
                    mean_domain_similarity=float("nan"),
                    out_of_distribution_rate=float("nan"),
                    mean_signal_quality=float("nan"),
                    mean_binary_probability=float("nan"),
                    review_rate=float("nan"),
                    alert=False,
                )
            matrix = {
                key: np.asarray([row[key] for row in self._records], dtype=float)
                for key in self._records[0]
            }
            n = len(self._records)
            mean_domain_similarity = float(matrix["domain_similarity"].mean())
            ood_rate = float(matrix["ood"].mean())
            mean_signal_quality = float(matrix["signal_quality"].mean())
            alert = (
                n >= self.minimum_n
                and (
                    mean_domain_similarity < self.minimum_domain_similarity
                    or ood_rate > self.maximum_ood_rate
                    or mean_signal_quality < self.minimum_signal_quality
                )
            )
            return DriftSummary(
                n=n,
                mean_domain_similarity=mean_domain_similarity,
                out_of_distribution_rate=ood_rate,
                mean_signal_quality=mean_signal_quality,
                mean_binary_probability=float(matrix["binary_probability"].mean()),
                review_rate=float(matrix["review"].mean()),
                alert=alert,
            )
