
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict
import numpy as np


@dataclass
class TrustPolicy:
    cdt_threshold: float
    cera_threshold: float
    cas_threshold: float
    subtype_probability_threshold: float
    maximum_false_accept_rate: float
    validation_coverage: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def calibrate_policy(
    cdt: np.ndarray,
    cera: np.ndarray,
    cas: np.ndarray,
    subtype_probability: np.ndarray,
    correct: np.ndarray,
    max_false_accept_rate: float = 0.05,
) -> TrustPolicy:
    """
    Select the highest-coverage threshold tuple whose accepted predictions have
    false-acceptance rate below the validation target.
    """
    grids = {
        "cdt": np.quantile(cdt, np.linspace(.3, .9, 13)),
        "cera": np.quantile(cera, np.linspace(.3, .9, 13)),
        "cas": np.quantile(cas, np.linspace(.3, .9, 13)),
        "prob": np.quantile(subtype_probability, np.linspace(.3, .9, 13)),
    }
    best = None
    for tc in grids["cdt"]:
        for te in grids["cera"]:
            for ta in grids["cas"]:
                for tp in grids["prob"]:
                    accepted = (
                        (cdt >= tc) & (cera >= te) & (cas >= ta)
                        & (subtype_probability >= tp)
                    )
                    if not accepted.any():
                        continue
                    false_accept = 1.0 - correct[accepted].mean()
                    coverage = accepted.mean()
                    if false_accept <= max_false_accept_rate:
                        candidate = (coverage, tc, te, ta, tp)
                        if best is None or candidate[0] > best[0]:
                            best = candidate
    if best is None:
        return TrustPolicy(1.0, 1.0, 1.0, 1.0, max_false_accept_rate, 0.0)
    coverage, tc, te, ta, tp = best
    return TrustPolicy(
        float(tc), float(te), float(ta), float(tp),
        float(max_false_accept_rate), float(coverage),
    )
