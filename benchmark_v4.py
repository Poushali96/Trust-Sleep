
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from statistics_v4 import paired_subject_bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed-predictions", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    proposed = pd.read_csv(args.proposed_predictions)
    baseline = pd.read_csv(args.baseline_predictions)
    keys = ["sample_id", "subject_id", "y_true"]
    merged = proposed.merge(
        baseline, on=keys, suffixes=("_proposed", "_baseline")
    )
    proposed_correct = (
        merged["prediction_proposed"].to_numpy()
        == merged["y_true"].to_numpy()
    ).astype(float)
    baseline_correct = (
        merged["prediction_baseline"].to_numpy()
        == merged["y_true"].to_numpy()
    ).astype(float)

    result = paired_subject_bootstrap(
        proposed_correct,
        baseline_correct,
        merged["subject_id"].to_numpy(),
        repetitions=2000,
    )
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
