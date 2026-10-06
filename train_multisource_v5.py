
from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from calibration import (
    BinaryMondrianConformal,
    BinaryPlattScaler,
    MulticlassMondrianConformal,
    MulticlassTemperatureScaler,
)
from datasets import (
    META_FIELDS,
    Sample,
    SignalStandardizer,
    load_local_hdf,
    metadata_vector,
    stable_subject_hash,
)
from model_v4 import HierarchicalTrustSleepV4, SUBTYPE_CLASSES
from metadata_scaler import MetadataStandardizer
from trust_policy import calibrate_policy


@dataclass
class V4Sample:
    sample: Sample
    modality_present: np.ndarray
    localization_target: np.ndarray
    localization_available: float
    source_split: str


class V4Dataset(Dataset):
    def __init__(self, samples: Sequence[V4Sample]):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        sample = item.sample
        return {
            "signal": torch.from_numpy(sample.signal).float(),
            "metadata": torch.from_numpy(sample.metadata).float(),
            "binary_label": torch.tensor(sample.binary_label, dtype=torch.float32),
            "event_label": torch.tensor(sample.event_label, dtype=torch.long),
            "subtype_label": torch.tensor(sample.subtype_label, dtype=torch.long),
            "domain_label": torch.tensor(sample.domain_label, dtype=torch.long),
            "modality_present": torch.from_numpy(item.modality_present).float(),
            "localization_target": torch.from_numpy(item.localization_target).float(),
            "localization_available": torch.tensor(item.localization_available, dtype=torch.float32),
            "sample_id": sample.sample_id,
            "subject_id": sample.subject_id,
            "recording_id": sample.recording_id,
        }


@dataclass
class Config:
    seed: int = 271828
    split_seed: int = 271828
    epochs: int = 20
    patience: int = 5
    batch_size: int = 512
    alpha: float = 0.10
    validation_fraction: float = 0.10
    conformal_fraction: float = 0.10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    localization_weight: float = 0.35
    context_prior_weight: float = 0.05
    max_context_weight: float = 0.20


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_local(samples: Sequence[Sample], cfg: Config):
    labels = np.asarray([s.binary_label for s in samples])
    groups = np.asarray([s.subject_id for s in samples])
    idx = np.arange(len(samples))
    hold = cfg.validation_fraction + cfg.conformal_fraction

    for attempt in range(500):
        first = GroupShuffleSplit(
            n_splits=1, test_size=hold, random_state=cfg.split_seed + attempt
        )
        train_idx, hold_idx = next(first.split(idx, labels, groups))
        second = GroupShuffleSplit(
            n_splits=1,
            test_size=cfg.conformal_fraction / hold,
            random_state=cfg.split_seed + 1000 + attempt,
        )
        val_rel, conf_rel = next(second.split(
            hold_idx, labels[hold_idx], groups[hold_idx]
        ))
        val_idx = hold_idx[val_rel]
        conf_idx = hold_idx[conf_rel]
        parts = [train_idx, val_idx, conf_idx]
        if all(set(labels[p]) == {0, 1} for p in parts):
            return [[samples[i] for i in p] for p in parts]
    raise RuntimeError("Could not create class-complete subject-disjoint local splits.")


def load_external_v4(manifest_path: str) -> Dict[str, List[V4Sample]]:
    frame = pd.read_csv(manifest_path)
    required = {
        "sample_id", "npz_path", "binary_label", "event_label",
        "subtype_label", "subject_id", "split",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing external manifest columns: {sorted(missing)}")

    event_map = {
        "no_event": 0, "apnea": 1, "hypopnea": 2, "artifact_or_uncertain": 3
    }
    subtype_map = {name: i for i, name in enumerate(SUBTYPE_CLASSES)}
    outputs = {name: [] for name in ("train", "validation", "conformal", "test")}

    # Hard split leakage check.
    if (frame.groupby("subject_id")["split"].nunique() > 1).any():
        raise RuntimeError("An external subject appears in multiple splits.")

    for _, row in frame.iterrows():
        split = str(row["split"])
        if split not in outputs:
            raise ValueError(f"Unsupported split: {split}")
        payload = np.load(row["npz_path"])
        signal = np.asarray(payload["signal"], np.float32)
        if signal.ndim != 2 or signal.shape[1] != 4:
            raise ValueError(f"{row['npz_path']}: signal must be [time,4]")
        modality_present = np.asarray(
            payload.get("modality_present", np.ones(4)), np.float32
        )
        if modality_present.shape != (4,):
            raise ValueError("modality_present must have shape [4]")

        if "localization_target" in payload:
            localization = np.asarray(payload["localization_target"], np.float32)
            localization_available = 1.0
        else:
            # Do not fabricate event boundaries. Localization loss is masked when
            # event-level boundaries are unavailable.
            localization = np.zeros(signal.shape[0], np.float32)
            localization_available = 0.0

        sample = Sample(
            signal=np.nan_to_num(signal),
            metadata=metadata_vector(row),
            metadata_raw=row.to_dict(),
            binary_label=int(row["binary_label"]),
            event_label=event_map[str(row["event_label"])],
            subtype_label=subtype_map[str(row["subtype_label"])],
            domain_label=1,
            sample_id=str(row["sample_id"]),
            subject_id=str(row["subject_id"]),
            recording_id=str(row.get("recording_id", row["subject_id"])),
        )
        outputs[split].append(V4Sample(
            sample=sample,
            modality_present=modality_present,
            localization_target=localization,
            localization_available=localization_available,
            source_split=split,
        ))

    for split in outputs:
        if not outputs[split]:
            raise RuntimeError(f"External split '{split}' is empty.")
    return outputs


def wrap_local(samples: Sequence[Sample], split: str) -> List[V4Sample]:
    wrapped = []
    for sample in samples:
        # Local data have binary epoch labels but no event boundaries. Keep the
        # localization target masked rather than inventing a middle-third event.
        localization = np.zeros(sample.signal.shape[0], np.float32)
        wrapped.append(V4Sample(
            sample=sample,
            modality_present=np.ones(4, np.float32),
            localization_target=localization,
            localization_available=0.0,
            source_split=split,
        ))
    return wrapped


def collate_factory(
    standardizer: SignalStandardizer,
    metadata_standardizer: MetadataStandardizer | None = None,
):
    def collate(batch):
        result: Dict[str, Any] = {}
        signals, localization = [], []
        for item in batch:
            x = standardizer.transform(item["signal"].numpy())
            x = torch.from_numpy(x).T[None]
            x = F.interpolate(x, size=30, mode="linear", align_corners=False)[0].T
            signals.append(x)

            target = item["localization_target"][None, None]
            target = F.interpolate(target, size=30, mode="nearest")[0, 0]
            localization.append(target)

        result["signal"] = torch.stack(signals)
        result["localization_target"] = torch.stack(localization)
        metadata_batch = torch.stack([item["metadata"] for item in batch])
        if metadata_standardizer is not None:
            metadata_np = metadata_standardizer.transform(metadata_batch.numpy())
            metadata_batch = torch.from_numpy(metadata_np)
        result["metadata"] = metadata_batch
        for key in (
            "binary_label", "event_label", "subtype_label",
            "domain_label", "modality_present", "localization_available",
        ):
            result[key] = torch.stack([item[key] for item in batch])
        for key in ("sample_id", "subject_id", "recording_id"):
            result[key] = [item[key] for item in batch]
        return result
    return collate


def make_loader(
    samples: Sequence[V4Sample],
    standardizer: SignalStandardizer,
    cfg: Config,
    device: torch.device,
    train: bool,
    metadata_standardizer: MetadataStandardizer | None = None,
):
    sampler = None
    shuffle = False
    if train:
        domains = np.asarray([s.sample.domain_label for s in samples])
        counts = np.bincount(domains, minlength=2)
        domain_weights = 1.0 / np.maximum(counts, 1)
        weights = domain_weights[domains]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(samples),
            replacement=True,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
    return DataLoader(
        V4Dataset(samples),
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=collate_factory(standardizer, metadata_standardizer),
        num_workers=cfg.num_workers if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
    )


def masked_ce(logits, labels):
    valid = labels != -100
    return (
        F.cross_entropy(logits[valid], labels[valid])
        if valid.any() else logits.sum() * 0
    )


def train_epoch(model, loader, optimizer, scaler, device, cfg, epoch):
    model.train()
    rows = []
    domain_strength = min(1.0, epoch / max(cfg.epochs * .5, 1))
    for batch in tqdm(loader, leave=False, desc=f"epoch {epoch}"):
        optimizer.zero_grad(set_to_none=True)
        x = batch["signal"].to(device)
        m = batch["metadata"].to(device)
        present = batch["modality_present"].to(device)
        yb = batch["binary_label"].to(device)
        ye = batch["event_label"].to(device)
        ys = batch["subtype_label"].to(device)
        yd = batch["domain_label"].to(device)
        yl = batch["localization_target"].to(device)
        yl_available = batch["localization_available"].to(device)

        # Random modality masking with explicit presence update.
        random_drop = (torch.rand_like(present) < .12).float()
        present_aug = present * (1 - random_drop)

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(x, m, present_aug, domain_strength)
            binary_loss = F.binary_cross_entropy_with_logits(output.binary_logit, yb)
            event_loss = masked_ce(output.event_logits, ye)
            subtype_loss = masked_ce(output.signal_subtype_logits, ys)
            final_subtype_loss = masked_ce(output.subtype_logits, ys)
            domain_loss = F.cross_entropy(output.domain_logits, yd)
            localization_per_sample = F.binary_cross_entropy_with_logits(
                output.localization_logits, yl, reduction="none"
            ).mean(dim=1)
            if yl_available.sum() > 0:
                localization_loss = (
                    localization_per_sample * yl_available
                ).sum() / yl_available.sum()
            else:
                localization_loss = output.localization_logits.sum() * 0.0
            context_magnitude = output.context_prior_logits.abs().mean()
            loss = (
                binary_loss
                + .55 * event_loss
                + .75 * subtype_loss
                + .25 * final_subtype_loss
                + .10 * domain_loss
                + cfg.localization_weight * localization_loss
                + cfg.context_prior_weight * context_magnitude
                + 1e-4 * output.gates.mean()
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        rows.append({
            "loss": float(loss.detach().cpu()),
            "binary_loss": float(binary_loss.detach().cpu()),
            "subtype_loss": float(subtype_loss.detach().cpu()),
            "localization_loss": float(localization_loss.detach().cpu()),
        })
    return pd.DataFrame(rows).mean().to_dict()


@torch.inference_mode()
def infer(model, loader, device):
    model.eval()
    fields = {
        "binary": [], "event": [], "signal_subtype": [], "subtype": [],
        "context": [], "localization": [], "y_binary": [], "y_subtype": [],
        "embedding": [],
    }
    for batch in loader:
        output = model(
            batch["signal"].to(device),
            batch["metadata"].to(device),
            batch["modality_present"].to(device),
        )
        fields["binary"].append(torch.sigmoid(output.binary_logit).cpu().numpy())
        fields["event"].append(output.event_logits.cpu().numpy())
        fields["signal_subtype"].append(output.signal_subtype_logits.cpu().numpy())
        fields["subtype"].append(output.subtype_logits.cpu().numpy())
        fields["context"].append(output.context_prior_logits.cpu().numpy())
        fields["localization"].append(torch.sigmoid(output.localization_logits).cpu().numpy())
        fields["embedding"].append(output.signal_embedding.cpu().numpy())
        fields["y_binary"].append(batch["binary_label"].numpy())
        fields["y_subtype"].append(batch["subtype_label"].numpy())
    return {
        key: np.concatenate(value, axis=0) if key not in {
            "event", "signal_subtype", "subtype", "context", "localization", "embedding"
        } else np.vstack(value)
        for key, value in fields.items()
    }


def threshold_f1(y, p):
    thresholds = np.unique(p)
    scores = [f1_score(y, p >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def fit_ood(embedding):
    mean = embedding.mean(0)
    covariance = np.cov(embedding, rowvar=False)
    covariance = .95 * covariance + .05 * np.diag(np.diag(covariance))
    covariance += np.eye(covariance.shape[0]) * 1e-4
    precision = np.linalg.pinv(covariance)
    delta = embedding - mean
    distance = np.sqrt(np.maximum(np.einsum("bi,ij,bj->b", delta, precision, delta), 0))
    return {
        "mean": mean.tolist(),
        "precision": precision.tolist(),
        "distance_q95": float(np.quantile(distance, .95)),
        "distance_q99": float(np.quantile(distance, .99)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-h5", required=True)
    parser.add_argument("--external-manifest", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=.10)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--split-seed", type=int, default=271828)
    args = parser.parse_args()

    cfg = Config(
        seed=args.seed,
        split_seed=args.split_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        alpha=args.alpha,
    )
    seed_all(cfg.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local = load_local_hdf(args.local_h5, args.metadata, domain_label=0)
    local_train, local_val, local_conf = split_local(local, cfg)
    external = load_external_v4(args.external_manifest)

    standardizer = SignalStandardizer.fit(
        local_train + [item.sample for item in external["train"]]
    )
    metadata_standardizer = MetadataStandardizer.fit(
        [sample.metadata for sample in local_train]
        + [item.sample.metadata for item in external["train"]]
    )
    train_samples = wrap_local(local_train, "train") + external["train"]
    local_val_loader = make_loader(
        wrap_local(local_val, "validation"), standardizer, cfg, device, False,
        metadata_standardizer,
    )
    local_conf_loader = make_loader(
        wrap_local(local_conf, "conformal"), standardizer, cfg, device, False,
        metadata_standardizer,
    )
    ext_val_loader = make_loader(
        external["validation"], standardizer, cfg, device, False,
        metadata_standardizer,
    )
    ext_conf_loader = make_loader(
        external["conformal"], standardizer, cfg, device, False,
        metadata_standardizer,
    )
    ext_train_loader = make_loader(
        external["train"], standardizer, cfg, device, False,
        metadata_standardizer,
    )
    train_loader = make_loader(
        train_samples, standardizer, cfg, device, True,
        metadata_standardizer,
    )

    model = HierarchicalTrustSleepV4(
        metadata_dim=24, max_context_weight=cfg.max_context_weight
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scaler = GradScaler(device=device.type, enabled=device.type == "cuda")

    best_state, best_score, bad = None, -np.inf, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        row = train_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)
        lv = infer(model, local_val_loader, device)
        ev = infer(model, ext_val_loader, device)
        threshold = threshold_f1(lv["y_binary"], lv["binary"])
        binary_f1 = f1_score(lv["y_binary"], lv["binary"] >= threshold, zero_division=0)
        subtype_f1 = f1_score(
            ev["y_subtype"], ev["signal_subtype"].argmax(1),
            average="macro", zero_division=0,
        )
        score = binary_f1 + .30 * subtype_f1
        row.update({
            "epoch": epoch,
            "local_binary_f1": binary_f1,
            "external_signal_subtype_macro_f1": subtype_f1,
            "selection_score": score,
        })
        history.append(row)
        print(row)
        if score > best_score + 1e-5:
            best_state = copy.deepcopy(model.state_dict())
            best_score = score
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    model.load_state_dict(best_state)
    lv, lc = infer(model, local_val_loader, device), infer(model, local_conf_loader, device)
    ev, ec = infer(model, ext_val_loader, device), infer(model, ext_conf_loader, device)
    et = infer(model, ext_train_loader, device)

    binary_scaler = BinaryPlattScaler().fit(lv["binary"], lv["y_binary"])
    calibrated_lv = binary_scaler.transform(lv["binary"])
    operating_threshold = threshold_f1(lv["y_binary"], calibrated_lv)
    binary_conformal = BinaryMondrianConformal.fit(
        binary_scaler.transform(lc["binary"]), lc["y_binary"], cfg.alpha
    )

    subtype_scaler = MulticlassTemperatureScaler().fit(
        ev["subtype"], ev["y_subtype"]
    )
    subtype_conformal = MulticlassMondrianConformal.fit(
        subtype_scaler.transform_probabilities(ec["subtype"]),
        ec["y_subtype"],
        cfg.alpha,
    )

    # Context reversal audit.
    signal_class = ev["signal_subtype"].argmax(1)
    final_class = ev["subtype"].argmax(1)
    context_reversal_rate = float(np.mean(signal_class != final_class))

    # Initial empirical trust policy uses validation correctness and proxy indices.
    subtype_probability = subtype_scaler.transform_probabilities(ev["subtype"]).max(1)
    subtype_correct = final_class == ev["y_subtype"]
    cdt = .70 * subtype_probability + .30 * (signal_class == final_class)
    cera = .60 * (signal_class == final_class) + .40
    cas = .55 * cdt + .45 * cera
    trust_policy = calibrate_policy(
        cdt, cera, cas, subtype_probability, subtype_correct.astype(float)
    )

    checkpoint = {
        "model_state": model.state_dict(),
        "model_class": "HierarchicalTrustSleepV4",
        "model_version": "4.0.0",
        "signal_standardizer": standardizer.to_dict(),
        "metadata_standardizer": metadata_standardizer.to_dict(),
        "localization_supervision": {
            "local_available": 0,
            "external_train_available": int(sum(item.localization_available > 0 for item in external["train"])),
            "external_train_total": len(external["train"]),
        },
        "operating_threshold": operating_threshold,
        "binary_platt": binary_scaler.to_dict(),
        "binary_conformal": binary_conformal.to_dict(),
        "subtype_temperature": subtype_scaler.to_dict(),
        "subtype_conformal": subtype_conformal.to_dict(),
        "ood_reference": fit_ood(et["embedding"]),
        "trust_policy": trust_policy.to_dict(),
        "context_reversal_rate_validation": context_reversal_rate,
        "local_development_subject_hashes": sorted({
            stable_subject_hash(s.subject_id) for s in local
        }),
        "external_test_manifest": args.external_manifest,
        "config": asdict(cfg),
        "research_only": True,
    }
    torch.save(checkpoint, output_dir / "hierarchical_trust_sleep_v4.pt")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    Path(output_dir / "training_summary.json").write_text(json.dumps({
        "best_selection_score": best_score,
        "context_reversal_rate_validation": context_reversal_rate,
        "trust_policy": trust_policy.to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
