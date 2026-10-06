
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from datasets import load_local_hdf
from inference_v5 import TrustSleepDeploymentEngine
from statistics_v4 import subject_bootstrap_ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-h5", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--ensemble-bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    engine = TrustSleepDeploymentEngine(args.ensemble_bundle)
    samples = load_local_hdf(
        args.test_h5,
        args.metadata,
        domain_label=0,
    )

    rows = []
    for index, sample in enumerate(
        tqdm(samples, desc="Independent local test")
    ):
        result = engine.predict(
            sample.signal,
            sample.metadata_raw,
            session_id=sample.recording_id,
            case_id=sample.sample_id,
            timestamp=float(index),
        )
        rows.append({
            "sample_id": sample.sample_id,
            "subject_id": sample.subject_id,
            "recording_id": sample.recording_id,
            "y_true": sample.binary_label,
            "probability": result.binary_probability,
            "prediction": int(result.binary_label == "apnea"),
            "subtype_hypothesis": result.subtype_hypothesis,
            "subtype_confidence": result.subtype_probability,
            "cdt": result.cdt,
            "cera": result.cera,
            "cas": result.cas,
            "decision": result.decision_status,
            "domain_status": result.domain_status,
            "subtype_locally_validated": False,
        })

    frame = pd.DataFrame(rows)
    frame.to_csv(
        output / "local_independent_predictions.csv",
        index=False,
    )
    y = frame["y_true"].to_numpy()
    probability = frame["probability"].to_numpy()
    prediction = frame["prediction"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(
        y, prediction, labels=[0, 1]
    ).ravel()

    metrics = {
        "Accuracy": accuracy_score(y, prediction),
        "Balanced Accuracy": balanced_accuracy_score(y, prediction),
        "Precision": precision_score(y, prediction, zero_division=0),
        "Recall": recall_score(y, prediction, zero_division=0),
        "Specificity": tn / max(tn+fp, 1),
        "F1": f1_score(y, prediction, zero_division=0),
        "ROC-AUC": roc_auc_score(y, probability),
        "Average Precision": average_precision_score(y, probability),
        "Assisted Review Rate": float(
            np.mean(frame["decision"] == "assisted_review")
        ),
        "Clinician Review Rate": float(
            np.mean(frame["decision"] != "assisted_review")
        ),
    }
    pd.DataFrame([metrics]).to_csv(
        output / "local_independent_metrics.csv",
        index=False,
    )

    accuracy_ci = subject_bootstrap_ci(
        (prediction == y).astype(float),
        frame["subject_id"].to_numpy(),
        repetitions=2000,
    )
    (output / "accuracy_subject_bootstrap_ci.json").write_text(
        json.dumps(accuracy_ci, indent=2)
    )


if __name__ == "__main__":
    main()
