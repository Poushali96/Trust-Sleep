
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from calibration import (
    BinaryMondrianConformal,
    BinaryPlattScaler,
    MulticlassMondrianConformal,
    MulticlassTemperatureScaler,
)
from datasets import SignalStandardizer, load_local_hdf
from metadata_scaler import MetadataStandardizer
from model_v4 import HierarchicalTrustSleepV4
from train_multisource_v5 import (
    Config,
    V4Dataset,
    collate_factory,
    load_external_v4,
    split_local,
    wrap_local,
)
from trust_policy import calibrate_policy


@torch.inference_mode()
def predict_member(model, loader, device):
    binary, subtype, signal_subtype, embedding, labels_binary, labels_subtype = (
        [], [], [], [], [], []
    )
    for batch in loader:
        output = model(
            batch["signal"].to(device),
            batch["metadata"].to(device),
            batch["modality_present"].to(device),
        )
        binary.append(torch.sigmoid(output.binary_logit).cpu().numpy())
        subtype.append(output.subtype_logits.cpu().numpy())
        signal_subtype.append(output.signal_subtype_logits.cpu().numpy())
        embedding.append(output.signal_embedding.cpu().numpy())
        labels_binary.append(batch["binary_label"].numpy())
        labels_subtype.append(batch["subtype_label"].numpy())
    return {
        "binary": np.concatenate(binary),
        "subtype": np.vstack(subtype),
        "signal_subtype": np.vstack(signal_subtype),
        "embedding": np.vstack(embedding),
        "y_binary": np.concatenate(labels_binary).astype(int),
        "y_subtype": np.concatenate(labels_subtype).astype(int),
    }


def build_loader(
    samples,
    standardizer,
    metadata_standardizer,
    batch_size,
):
    return DataLoader(
        V4Dataset(samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_factory(
            standardizer,
            metadata_standardizer,
        ),
        num_workers=0,
    )


def fit_ood_reference(embeddings: np.ndarray) -> Dict[str, object]:
    mean = embeddings.mean(axis=0)
    covariance = np.cov(embeddings, rowvar=False)
    diagonal = np.diag(np.diag(covariance))
    covariance = 0.90 * covariance + 0.10 * diagonal
    covariance += np.eye(covariance.shape[0]) * 1e-4
    precision = np.linalg.pinv(covariance)
    delta = embeddings - mean
    distances = np.sqrt(
        np.maximum(np.einsum("bi,ij,bj->b", delta, precision, delta), 0)
    )
    return {
        "mean": mean.astype(np.float32).tolist(),
        "precision": precision.astype(np.float32).tolist(),
        "distance_q95": float(np.quantile(distances, 0.95)),
        "distance_q99": float(np.quantile(distances, 0.99)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-manifest", required=True)
    parser.add_argument("--local-h5", required=True)
    parser.add_argument("--external-manifest", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads(Path(args.ensemble_manifest).read_text())
    checkpoint_paths = manifest["checkpoints"]
    if len(checkpoint_paths) < 2:
        raise ValueError("Ensemble calibration requires at least two members.")

    first = torch.load(checkpoint_paths[0], map_location="cpu")
    standardizer = SignalStandardizer.from_dict(first["signal_standardizer"])
    metadata_standardizer = MetadataStandardizer.from_dict(first["metadata_standardizer"])

    local = load_local_hdf(args.local_h5, args.metadata, domain_label=0)
    cfg = Config(
        seed=args.seed,
        split_seed=int(manifest.get("split_seed", args.seed)),
        alpha=args.alpha,
        batch_size=args.batch_size,
    )
    _, local_validation, local_conformal = split_local(local, cfg)
    external = load_external_v4(args.external_manifest)

    local_validation_loader = build_loader(
        wrap_local(local_validation, "validation"),
        standardizer,
        metadata_standardizer,
        args.batch_size,
    )
    local_conformal_loader = build_loader(
        wrap_local(local_conformal, "conformal"),
        standardizer,
        metadata_standardizer,
        args.batch_size,
    )
    external_validation_loader = build_loader(
        external["validation"], standardizer, metadata_standardizer, args.batch_size
    )
    external_conformal_loader = build_loader(
        external["conformal"], standardizer, metadata_standardizer, args.batch_size
    )
    external_train_loader = build_loader(
        external["train"], standardizer, metadata_standardizer, args.batch_size
    )

    outputs = {
        "local_validation": [],
        "local_conformal": [],
        "external_validation": [],
        "external_conformal": [],
        "external_train": [],
    }

    for checkpoint_path in checkpoint_paths:
        payload = torch.load(checkpoint_path, map_location=device)
        model = HierarchicalTrustSleepV4(
            metadata_dim=24,
            max_context_weight=float(
                payload.get("config", {}).get("max_context_weight", 0.20)
            ),
        ).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        outputs["local_validation"].append(
            predict_member(model, local_validation_loader, device)
        )
        outputs["local_conformal"].append(
            predict_member(model, local_conformal_loader, device)
        )
        outputs["external_validation"].append(
            predict_member(model, external_validation_loader, device)
        )
        outputs["external_conformal"].append(
            predict_member(model, external_conformal_loader, device)
        )
        outputs["external_train"].append(
            predict_member(model, external_train_loader, device)
        )

    local_validation_binary = np.mean(
        [item["binary"] for item in outputs["local_validation"]], axis=0
    )
    local_conformal_binary = np.mean(
        [item["binary"] for item in outputs["local_conformal"]], axis=0
    )
    y_local_validation = outputs["local_validation"][0]["y_binary"]
    y_local_conformal = outputs["local_conformal"][0]["y_binary"]

    binary_scaler = BinaryPlattScaler().fit(
        local_validation_binary, y_local_validation
    )
    calibrated_validation = binary_scaler.transform(local_validation_binary)
    candidates = np.unique(calibrated_validation)
    scores = [
        (
            2
            * np.mean(
                (calibrated_validation >= threshold)
                & (y_local_validation == 1)
            )
            / max(
                np.mean(calibrated_validation >= threshold)
                + np.mean(y_local_validation == 1),
                1e-12,
            )
        )
        for threshold in candidates
    ]
    operating_threshold = float(candidates[int(np.argmax(scores))])
    binary_conformal = BinaryMondrianConformal.fit(
        binary_scaler.transform(local_conformal_binary),
        y_local_conformal,
        args.alpha,
    )

    external_validation_logits = np.mean(
        [item["subtype"] for item in outputs["external_validation"]], axis=0
    )
    external_conformal_logits = np.mean(
        [item["subtype"] for item in outputs["external_conformal"]], axis=0
    )
    y_external_validation = outputs["external_validation"][0]["y_subtype"]
    y_external_conformal = outputs["external_conformal"][0]["y_subtype"]

    subtype_scaler = MulticlassTemperatureScaler().fit(
        external_validation_logits, y_external_validation
    )
    subtype_conformal = MulticlassMondrianConformal.fit(
        subtype_scaler.transform_probabilities(external_conformal_logits),
        y_external_conformal,
        args.alpha,
    )

    validation_probabilities = subtype_scaler.transform_probabilities(
        external_validation_logits
    )
    validation_prediction = validation_probabilities.argmax(axis=1)
    validation_correct = validation_prediction == y_external_validation
    signal_logits = np.mean(
        [item["signal_subtype"] for item in outputs["external_validation"]],
        axis=0,
    )
    signal_prediction = signal_logits.argmax(axis=1)
    context_agreement = signal_prediction == validation_prediction
    subtype_confidence = validation_probabilities.max(axis=1)

    # Conservative empirical trust-policy proxies, fitted on validation only.
    cdt = 0.75 * subtype_confidence + 0.25 * context_agreement.astype(float)
    cera = 0.60 * context_agreement.astype(float) + 0.40
    cas = 0.55 * cdt + 0.45 * cera
    trust_policy = calibrate_policy(
        cdt,
        cera,
        cas,
        subtype_confidence,
        validation_correct.astype(float),
        max_false_accept_rate=0.05,
    )

    external_train_embedding = np.mean(
        [item["embedding"] for item in outputs["external_train"]], axis=0
    )

    bundle = {
        "bundle_version": "5.0.0",
        "checkpoint_paths": checkpoint_paths,
        "member_count": len(checkpoint_paths),
        "split_seed": int(manifest.get("split_seed", args.seed)),
        "signal_standardizer": standardizer.to_dict(),
        "metadata_standardizer": metadata_standardizer.to_dict(),
        "operating_threshold": operating_threshold,
        "binary_platt": binary_scaler.to_dict(),
        "binary_conformal": binary_conformal.to_dict(),
        "subtype_temperature": subtype_scaler.to_dict(),
        "subtype_conformal": subtype_conformal.to_dict(),
        "ood_reference": fit_ood_reference(external_train_embedding),
        "trust_policy": trust_policy.to_dict(),
        "calibration_sources": {
            "binary_probability": "local validation subjects",
            "binary_conformal": "local conformal subjects",
            "subtype_probability": "external validation subjects",
            "subtype_conformal": "external conformal subjects",
        },
        "research_only": True,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    Path(str(output) + ".json").write_text(
        json.dumps(
            {key: value for key, value in bundle.items() if key != "checkpoint_paths"}
            | {"checkpoint_paths": checkpoint_paths},
            indent=2,
        )
    )
    print("Saved ensemble-calibrated bundle:", output)


if __name__ == "__main__":
    main()
