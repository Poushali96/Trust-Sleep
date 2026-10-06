
from __future__ import annotations
from typing import Callable, Dict, Optional
import numpy as np


def subject_bootstrap_ci(
    values: np.ndarray,
    subject_ids: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    repetitions: int = 2000,
    seed: int = 271828,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    subjects = np.unique(subject_ids)
    estimates = []
    for _ in range(repetitions):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([np.flatnonzero(subject_ids == subject) for subject in sampled])
        estimates.append(statistic(values[indices]))
    estimates = np.asarray(estimates, dtype=float)
    return {
        "estimate": float(statistic(values)),
        "lower_95": float(np.quantile(estimates, .025)),
        "upper_95": float(np.quantile(estimates, .975)),
    }


def paired_subject_bootstrap(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    subject_ids: np.ndarray,
    repetitions: int = 2000,
    seed: int = 271828,
) -> Dict[str, float]:
    if len(metric_a) != len(metric_b):
        raise ValueError("Paired arrays must have equal length.")
    difference = np.asarray(metric_a) - np.asarray(metric_b)
    ci = subject_bootstrap_ci(
        difference, np.asarray(subject_ids),
        statistic=np.mean, repetitions=repetitions, seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    subjects = np.unique(subject_ids)
    samples = []
    for _ in range(repetitions):
        selected = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([np.flatnonzero(subject_ids == s) for s in selected])
        samples.append(difference[idx].mean())
    samples = np.asarray(samples)
    ci["p_two_sided"] = float(
        min(1.0, 2 * min(np.mean(samples <= 0), np.mean(samples >= 0)))
    )
    return ci
