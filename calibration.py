
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Tuple
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


class BinaryPlattScaler:
    def __init__(self) -> None:
        self.model = LogisticRegression(solver="lbfgs")

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "BinaryPlattScaler":
        logits = np.log(
            np.clip(probabilities, 1e-6, 1 - 1e-6)
            / np.clip(1 - probabilities, 1e-6, 1)
        )
        self.model.fit(logits.reshape(-1, 1), labels.astype(int))
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        logits = np.log(
            np.clip(probabilities, 1e-6, 1 - 1e-6)
            / np.clip(1 - probabilities, 1e-6, 1)
        )
        return self.model.predict_proba(logits.reshape(-1, 1))[:, 1]

    def to_dict(self) -> Dict[str, object]:
        return {
            "coef": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
            "classes": self.model.classes_.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BinaryPlattScaler":
        obj = cls()
        obj.model.classes_ = np.asarray(payload["classes"])
        obj.model.coef_ = np.asarray(payload["coef"], dtype=float)
        obj.model.intercept_ = np.asarray(payload["intercept"], dtype=float)
        obj.model.n_features_in_ = 1
        return obj


class MulticlassTemperatureScaler:
    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "MulticlassTemperatureScaler":
        x = torch.tensor(logits, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        log_t = torch.tensor([0.0], requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=100)

        def closure():
            optimizer.zero_grad()
            temperature = log_t.exp().clamp(0.05, 20.0)
            loss = F.cross_entropy(x / temperature, y)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(log_t.detach().exp().clamp(0.05, 20.0))
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=float) / self.temperature

    def transform_probabilities(self, logits: np.ndarray) -> np.ndarray:
        scaled = self.transform_logits(logits)
        scaled -= scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        return exp / exp.sum(axis=1, keepdims=True)

    def to_dict(self) -> Dict[str, float]:
        return {"temperature": self.temperature}

    @classmethod
    def from_dict(cls, payload: Mapping[str, float]) -> "MulticlassTemperatureScaler":
        return cls(float(payload["temperature"]))


def higher_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot calibrate conformal prediction with zero scores.")
    level = min(math.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    try:
        return float(np.quantile(scores, level, method="higher"))
    except TypeError:
        return float(np.quantile(scores, level, interpolation="higher"))


@dataclass
class BinaryMondrianConformal:
    q0: float
    q1: float
    alpha: float

    @classmethod
    def fit(
        cls,
        probabilities: np.ndarray,
        labels: np.ndarray,
        alpha: float,
    ) -> "BinaryMondrianConformal":
        labels = labels.astype(int)
        return cls(
            q0=higher_quantile(probabilities[labels == 0], alpha),
            q1=higher_quantile(1.0 - probabilities[labels == 1], alpha),
            alpha=float(alpha),
        )

    def predict_sets(
        self,
        probabilities: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        include0 = probabilities <= self.q0
        include1 = (1.0 - probabilities) <= self.q1
        size = include0.astype(int) + include1.astype(int)
        return include0, include1, size

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, float]) -> "BinaryMondrianConformal":
        return cls(
            q0=float(payload["q0"]),
            q1=float(payload["q1"]),
            alpha=float(payload["alpha"]),
        )


@dataclass
class MulticlassMondrianConformal:
    q_by_class: List[float]
    alpha: float

    @classmethod
    def fit(
        cls,
        probabilities: np.ndarray,
        labels: np.ndarray,
        alpha: float,
    ) -> "MulticlassMondrianConformal":
        labels = labels.astype(int)
        q_values: List[float] = []
        for class_index in range(probabilities.shape[1]):
            class_probabilities = probabilities[labels == class_index, class_index]
            if len(class_probabilities) == 0:
                q_values.append(1.0)
            else:
                q_values.append(higher_quantile(1.0 - class_probabilities, alpha))
        return cls(q_by_class=q_values, alpha=float(alpha))

    def predict_mask(self, probabilities: np.ndarray) -> np.ndarray:
        q = np.asarray(self.q_by_class, dtype=float)[None, :]
        return (1.0 - probabilities) <= q

    def prediction_sets(self, probabilities: np.ndarray) -> List[List[int]]:
        mask = self.predict_mask(probabilities)
        return [np.flatnonzero(row).tolist() for row in mask]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MulticlassMondrianConformal":
        return cls(
            q_by_class=[float(value) for value in payload["q_by_class"]],
            alpha=float(payload["alpha"]),
        )
