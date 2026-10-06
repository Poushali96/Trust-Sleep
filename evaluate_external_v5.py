
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from torch.utils.data import DataLoader

from calibration import (
    MulticlassMondrianConformal,
    MulticlassTemperatureScaler,
)
from datasets import SignalStandardizer
from metadata_scaler import MetadataStandardizer
from model_v4 import HierarchicalTrustSleepV4, SUBTYPE_CLASSES
from statistics_v4 import subject_bootstrap_ci
from train_multisource_v5 import V4Dataset, collate_factory, load_external_v4


@torch.inference_mode()
def member_logits(model, loader, device):
    subtype, signal_subtype, labels, subjects = [], [], [], []
    for batch in loader:
        output = model(
            batch["signal"].to(device),
            batch["metadata"].to(device),
            batch["modality_present"].to(device),
        )
        subtype.append(output.subtype_logits.cpu().numpy())
        signal_subtype.append(output.signal_subtype_logits.cpu().numpy())
        labels.append(batch["subtype_label"].numpy())
        subjects.extend(batch["subject_id"])
    return (
        np.vstack(subtype),
        np.vstack(signal_subtype),
        np.concatenate(labels).astype(int),
        np.asarray(subjects),
    )


def multiclass_ece(labels, probabilities, bins=15):
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == labels).astype(float)
    edges = np.linspace(0, 1, bins+1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (
            (confidence >= low)
            & (confidence < high if high < 1 else confidence <= high)
        )
        if mask.any():
            value += mask.mean() * abs(
                correct[mask].mean() - confidence[mask].mean()
            )
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-bundle", required=True)
    parser.add_argument("--external-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(args.ensemble_bundle, map_location="cpu")

    standardizer = SignalStandardizer.from_dict(
        bundle["signal_standardizer"]
    )
    metadata_standardizer = MetadataStandardizer.from_dict(
        bundle["metadata_standardizer"]
    )
    external = load_external_v4(args.external_manifest)
    loader = DataLoader(
        V4Dataset(external["test"]),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_factory(
            standardizer,
            metadata_standardizer,
        ),
        num_workers=0,
    )

    subtype_members = []
    signal_members = []
    labels = subjects = None
    for checkpoint_path in bundle["checkpoint_paths"]:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = HierarchicalTrustSleepV4(
            metadata_dim=24,
            max_context_weight=float(
                checkpoint.get("config", {}).get(
                    "max_context_weight", 0.20
                )
            ),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        subtype, signal_subtype, y, subject = member_logits(
            model, loader, device
        )
        subtype_members.append(subtype)
        signal_members.append(signal_subtype)
        labels = y
        subjects = subject

    logits = np.mean(subtype_members, axis=0)
    signal_logits = np.mean(signal_members, axis=0)
    scaler = MulticlassTemperatureScaler.from_dict(
        bundle["subtype_temperature"]
    )
    conformal = MulticlassMondrianConformal.from_dict(
        bundle["subtype_conformal"]
    )
    probabilities = scaler.transform_probabilities(logits)
    signal_probabilities = scaler.transform_probabilities(signal_logits)
    prediction = probabilities.argmax(axis=1)
    signal_prediction = signal_probabilities.argmax(axis=1)
    prediction_sets = conformal.prediction_sets(probabilities)

    result = {
        "Accuracy": accuracy_score(labels, prediction),
        "Balanced Accuracy": balanced_accuracy_score(labels, prediction),
        "Macro-F1": f1_score(
            labels, prediction, average="macro", zero_division=0
        ),
        "NLL": log_loss(labels, probabilities, labels=list(range(4))),
        "ECE-15": multiclass_ece(labels, probabilities),
        "Context Reversal Rate": float(
            np.mean(prediction != signal_prediction)
        ),
        "Conformal Coverage": float(np.mean([
            label in prediction_set
            for label, prediction_set in zip(labels, prediction_sets)
        ])),
        "Singleton Rate": float(np.mean([
            len(prediction_set) == 1
            for prediction_set in prediction_sets
        ])),
    }

    for index, name in enumerate(SUBTYPE_CLASSES):
        mask = labels == index
        result[f"{name}_F1"] = f1_score(
            labels == index,
            prediction == index,
            zero_division=0,
        )
        result[f"{name}_Coverage"] = (
            float(np.mean([
                index in prediction_sets[row]
                for row in np.flatnonzero(mask)
            ]))
            if mask.any()
            else np.nan
        )

    pd.DataFrame([result]).to_csv(
        output / "external_test_metrics.csv",
        index=False,
    )
    pd.DataFrame(
        confusion_matrix(labels, prediction)
    ).to_csv(
        output / "external_test_confusion_matrix.csv",
        index=False,
    )

    per_sample = pd.DataFrame({
        "subject_id": subjects,
        "true_subtype": [
            SUBTYPE_CLASSES[index] for index in labels
        ],
        "predicted_subtype": [
            SUBTYPE_CLASSES[index] for index in prediction
        ],
        "signal_only_subtype": [
            SUBTYPE_CLASSES[index] for index in signal_prediction
        ],
        "confidence": probabilities.max(axis=1),
        "prediction_set": [
            json.dumps([
                SUBTYPE_CLASSES[index]
                for index in prediction_set
            ])
            for prediction_set in prediction_sets
        ],
        "correct": prediction == labels,
    })
    for index, name in enumerate(SUBTYPE_CLASSES):
        per_sample[f"probability_{name}"] = probabilities[:, index]
    per_sample.to_csv(
        output / "external_test_predictions.csv",
        index=False,
    )

    accuracy_ci = subject_bootstrap_ci(
        (prediction == labels).astype(float),
        subjects,
        repetitions=2000,
    )
    (output / "external_test_accuracy_ci.json").write_text(
        json.dumps(accuracy_ci, indent=2)
    )


if __name__ == "__main__":
    main()
