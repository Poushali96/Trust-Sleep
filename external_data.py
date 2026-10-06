
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import numpy as np
import pandas as pd


PHYSIONET_2018 = {
    "name": "PhysioNet/CinC Challenge 2018",
    "landing_page": "https://physionet.org/content/challenge-2018/1.0.0/",
    "notes": (
        "Contains obstructive, central, mixed apnea and hypopnea annotations. "
        "The full dataset is large; verify access terms before downloading."
    ),
}
SHHS = {
    "name": "Sleep Heart Health Study",
    "landing_page": "https://physionet.org/content/shhpsgdb/1.0.0/",
    "notes": (
        "Respiratory annotations include obstructive apnea, central apnea, and hypopnea. "
        "Access requires approval; mixed-apnea availability varies by release."
    ),
}


def validate_manifest(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "sample_id", "npz_path", "binary_label", "event_label",
        "subtype_label", "subject_id", "source", "split",
        "sampling_rate_hz", "window_seconds",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    allowed_splits = {"train", "validation", "conformal", "test"}
    if not set(frame["split"]).issubset(allowed_splits):
        raise ValueError("split must be train, validation, conformal, or test")

    allowed_subtypes = {"obstructive", "central", "mixed", "indeterminate"}
    if not set(frame["subtype_label"]).issubset(allowed_subtypes):
        raise ValueError("Invalid subtype_label")

    subject_split_counts = frame.groupby("subject_id")["split"].nunique()
    leaked = subject_split_counts[subject_split_counts > 1]
    if len(leaked):
        raise ValueError(f"Subjects appear in multiple splits: {leaked.index[:10].tolist()}")

    for file_path in frame["npz_path"]:
        payload = np.load(file_path)
        signal = payload["signal"]
        if signal.ndim != 2 or signal.shape[1] != 4:
            raise ValueError(f"{file_path}: signal must be [time,4]")
        if "modality_present" not in payload:
            raise ValueError(f"{file_path}: modality_present is required")
    return frame


def write_dataset_plan(path: str) -> None:
    plan = {
        "recommended_primary_source": PHYSIONET_2018,
        "optional_secondary_source": SHHS,
        "required_harmonized_channels": ["chest", "flow", "pulse", "spo2"],
        "recommended_additional_channel": "abdominal_effort",
        "split_policy": {
            "train": 0.70,
            "validation": 0.10,
            "conformal": 0.10,
            "test": 0.10,
            "unit": "subject",
        },
        "do_not": [
            "Do not place the same subject in multiple splits.",
            "Do not tune on the external test partition.",
            "Do not redistribute credentialed data.",
        ],
    }
    Path(path).write_text(json.dumps(plan, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-manifest")
    parser.add_argument("--write-plan")
    args = parser.parse_args()
    if args.validate_manifest:
        frame = validate_manifest(args.validate_manifest)
        print(frame.groupby(["source", "split", "subtype_label"]).size())
    if args.write_plan:
        write_dataset_plan(args.write_plan)


if __name__ == "__main__":
    main()
