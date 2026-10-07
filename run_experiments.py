# ================================================================
# TRUST-SLEEP — CLEAN LEAKAGE-FREE PUBLICATION RERUN
#
# ONE-CELL GOOGLE COLAB PIPELINE
#
# Corrected design:
#   experiment1b.h5 = MASTER DATASET
#
# Recording-disjoint frozen split:
#   60% train
#   10% early-stopping validation
#    5% Platt calibration
#    5% operating-threshold selection
#   10% conformal calibration
#   10% final untouched test
#
# EXPERIMENT 1:
#   - single Full Trust-Sleep model (seed 271828)
#   - predictive performance
#   - CDT / CERA / CAS
#   - trust vs correctness / uncertainty
#   - selective prediction
#   - perturbation robustness
#
# EXPERIMENT 2:
#   - Full vs Raw-only
#   - five seeds
#   - seed-level and ensemble results
#   - independent calibration
#   - conformal prediction
#   - paired recording-bootstrap CIs
#
# ALL OUTPUTS ARE SAVED TO GOOGLE DRIVE.
# ================================================================

# ------------------------------------------------
# 0. INSTALL / IMPORT
# ------------------------------------------------


import os
import sys
import json
import math
import time
import random
import hashlib
import platform
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

from scipy.stats import pearsonr, spearmanr

from tqdm.auto import tqdm

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ------------------------------------------------
# 1. PORTABLE USER CONFIGURATION
# ------------------------------------------------

import argparse

_parser = argparse.ArgumentParser(
    description="Reproduce the Trust-Sleep leakage-controlled experiments."
)
_parser.add_argument(
    "--data",
    default=os.environ.get("TRUSTSLEEP_DATA", "data/experiment1b.h5"),
    help="Path to the restricted master HDF5 dataset (experiment1b.h5).",
)
_parser.add_argument(
    "--output",
    default=os.environ.get("TRUSTSLEEP_OUTPUT", "results/runs"),
    help="Directory in which a timestamped run folder will be created.",
)
_args = _parser.parse_args()

MASTER_H5 = Path(_args.data).expanduser().resolve()
OUTPUT_ROOT = Path(_args.output).expanduser().resolve()
RUN_NAME = datetime.now().strftime("clean_rerun_%Y%m%d_%H%M%S")
OUT = OUTPUT_ROOT / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = OUT / "checkpoints"
TABLE_DIR = OUT / "tables"
FIGURE_DIR = OUT / "figures"
AUDIT_DIR = OUT / "audit"
for p in [CHECKPOINT_DIR, TABLE_DIR, FIGURE_DIR, AUDIT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------
# 2. LOCATE MASTER DATASET
# ------------------------------------------------

def find_master_h5():
    candidates = [
        MASTER_H5,
        Path.cwd() / "data" / "experiment1b.h5",
        Path("/content/experiment1b.h5"),
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "Could not find experiment1b.h5. Pass it explicitly with "
        "--data /path/to/experiment1b.h5. The clinical dataset is not "
        "distributed with this repository."
    )


MASTER_H5 = find_master_h5()

print("\nMASTER DATASET:")
print(MASTER_H5)

print("\nOUTPUT DIRECTORY:")
print(OUT)


# ------------------------------------------------
# 4. GLOBAL EXPERIMENT SETTINGS
# ------------------------------------------------

SIGNALS = (
    "psg_chest_rn_mean",
    "psg_flow_dr_rn_mean",
    "psg_pulse_rn_mean",
    "psg_spo2_rn_mean",
)

SEEDS = (
    271828,
    314159,
    161803,
    141421,
    173205,
)

SPLIT_NAMES = (
    "train",
    "validation",
    "platt",
    "threshold",
    "conformal",
    "test",
)

TARGET_FRACTIONS = {
    "train": 0.60,
    "validation": 0.10,
    "platt": 0.05,
    "threshold": 0.05,
    "conformal": 0.10,
    "test": 0.10,
}

MAX_EPOCHS = 20
PATIENCE = 5

TRAIN_BATCH = 1024
EVAL_BATCH = 4096

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.35

CONFORMAL_ALPHA = 0.10

BOOTSTRAP_REPS = 1000

# Experiment 1 MC-dropout passes.
MC_DROPOUT_PASSES = 15

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDEVICE:", DEVICE)

if DEVICE.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ------------------------------------------------
# 5. REPRODUCIBILITY
# ------------------------------------------------

def seed_everything(seed):

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        ":4096:8",
    )

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    except Exception:
        pass


seed_everything(SEEDS[0])


# ------------------------------------------------
# 6. SAVE RUNTIME INFORMATION
# ------------------------------------------------

runtime = {
    "timestamp": datetime.now().isoformat(),
    "master_h5": str(MASTER_H5),
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "device": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    ),
    "seeds": list(SEEDS),
}

(
    OUT / "runtime_environment.json"
).write_text(
    json.dumps(
        runtime,
        indent=2,
    )
)


# ------------------------------------------------
# 7. HDF5 DATA LOADING
# ------------------------------------------------

def label_from_key(key):

    leaf = key.rsplit("/", 1)[-1]

    if leaf.endswith("_0"):
        return 0

    if leaf.endswith("_1"):
        return 1

    raise ValueError(
        f"Cannot infer label from {key}"
    )


def recording_from_key(key):

    return key.strip("/").split("/")[0]


def get_epoch_keys(h5_path):

    with pd.HDFStore(
        str(h5_path),
        mode="r",
    ) as store:

        keys = sorted([
            k
            for k in store.keys()
            if "/e" in k.lower()
            and k.rsplit("/", 1)[-1].endswith(
                ("_0", "_1")
            )
        ])

    if not keys:
        raise RuntimeError(
            "No labelled epoch keys found."
        )

    return keys


def load_master_dataset(h5_path):

    keys = get_epoch_keys(h5_path)

    print(
        f"\nLoading {len(keys):,} epochs "
        f"from {Path(h5_path).name}"
    )

    xs = []
    ys = []
    groups = []
    missing_fraction = []

    with pd.HDFStore(
        str(h5_path),
        mode="r",
    ) as store:

        for i, key in enumerate(
            tqdm(keys)
        ):

            frame = store.get(key)

            missing_cols = [
                c
                for c in SIGNALS
                if c not in frame.columns
            ]

            if missing_cols:
                raise RuntimeError(
                    f"{key}: missing {missing_cols}"
                )

            arr = (
                frame.loc[:, list(SIGNALS)]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .to_numpy(
                    dtype=np.float32
                )
            )

            if arr.shape != (30, 4):
                raise RuntimeError(
                    f"{key}: got shape "
                    f"{arr.shape}, expected (30,4)"
                )

            arr[
                ~np.isfinite(arr)
            ] = np.nan

            xs.append(arr)

            ys.append(
                label_from_key(key)
            )

            groups.append(
                recording_from_key(key)
            )

            missing_fraction.append(
                np.isnan(arr).mean()
            )

    x = np.stack(xs).astype(
        np.float32
    )

    y = np.asarray(
        ys,
        dtype=np.int64,
    )

    groups = np.asarray(
        groups,
        dtype=object,
    )

    keys = np.asarray(
        keys,
        dtype=object,
    )

    missing_fraction = np.asarray(
        missing_fraction,
        dtype=np.float32,
    )

    print("\nDATASET SUMMARY")
    print("epochs:", len(y))
    print("recordings:", len(np.unique(groups)))
    print("positive:", int(y.sum()))
    print("negative:", int((y == 0).sum()))
    print("positive rate:", float(y.mean()))

    return (
        x,
        y,
        groups,
        keys,
        missing_fraction,
    )


(
    X_RAW,
    Y,
    RECORDING,
    SOURCE_KEY,
    MISSING_FRACTION,
) = load_master_dataset(
    MASTER_H5
)


# ------------------------------------------------
# 8. RECORDING-LEVEL SPLIT CONSTRUCTION
# ------------------------------------------------

def split_candidate(
    indices,
    test_fraction,
    seed,
    attempts=2000,
):

    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    best = None

    for attempt in range(attempts):

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_fraction,
            random_state=seed + attempt,
        )

        left_rel, right_rel = next(
            splitter.split(
                indices,
                Y[indices],
                RECORDING[indices],
            )
        )

        left = indices[left_rel]
        right = indices[right_rel]

        if len(np.unique(Y[left])) < 2:
            continue

        if len(np.unique(Y[right])) < 2:
            continue

        if set(
            RECORDING[left]
        ) & set(
            RECORDING[right]
        ):
            raise RuntimeError(
                "Internal recording leakage."
            )

        actual_fraction = (
            len(right) / len(indices)
        )

        parent_rate = Y[
            indices
        ].mean()

        score = (
            abs(
                actual_fraction
                - test_fraction
            )
            + abs(
                Y[left].mean()
                - parent_rate
            )
            + abs(
                Y[right].mean()
                - parent_rate
            )
        )

        if (
            best is None
            or score < best[0]
        ):
            best = (
                score,
                left,
                right,
            )

    if best is None:
        raise RuntimeError(
            "Could not construct "
            "recording-level split."
        )

    return (
        np.sort(best[1]),
        np.sort(best[2]),
    )


ALL = np.arange(
    len(Y),
    dtype=np.int64,
)

# 60 / 40
TRAIN_IDX, REST40 = split_candidate(
    ALL,
    0.40,
    271828,
)

# 30 / 10
REST30, VAL_IDX = split_candidate(
    REST40,
    0.25,
    314159,
)

# 25 / 5
REST25, PLATT_IDX = split_candidate(
    REST30,
    1 / 6,
    161803,
)

# 20 / 5
REST20, THRESHOLD_IDX = split_candidate(
    REST25,
    0.20,
    141421,
)

# 10 / 10
CONFORMAL_IDX, TEST_IDX = split_candidate(
    REST20,
    0.50,
    173205,
)

SPLITS = {
    "train": TRAIN_IDX,
    "validation": VAL_IDX,
    "platt": PLATT_IDX,
    "threshold": THRESHOLD_IDX,
    "conformal": CONFORMAL_IDX,
    "test": TEST_IDX,
}


# ------------------------------------------------
# 9. HARD SPLIT AUDIT
# ------------------------------------------------

def audit_splits():

    all_indices = np.concatenate(
        list(SPLITS.values())
    )

    if len(all_indices) != len(Y):
        raise RuntimeError(
            "Splits do not cover entire dataset."
        )

    if len(
        np.unique(all_indices)
    ) != len(all_indices):
        raise RuntimeError(
            "Epoch-level split overlap."
        )

    for i, left in enumerate(
        SPLIT_NAMES
    ):

        li = SPLITS[left]

        if set(
            np.unique(Y[li])
        ) != {0, 1}:
            raise RuntimeError(
                f"{left} lacks a class."
            )

        for right in SPLIT_NAMES[
            i + 1:
        ]:

            ri = SPLITS[right]

            overlap = (
                set(RECORDING[li])
                & set(RECORDING[ri])
            )

            if overlap:
                raise RuntimeError(
                    f"RECORDING LEAKAGE "
                    f"{left} vs {right}: "
                    f"{list(overlap)[:5]}"
                )

    print(
        "\n✓ NO RECORDING OVERLAP "
        "BETWEEN ANY SPLITS."
    )


audit_splits()


# ------------------------------------------------
# 10. SAVE SPLIT ARTIFACTS
# ------------------------------------------------

np.savez_compressed(
    AUDIT_DIR / "clean_split_indices.npz",
    **SPLITS,
)

summary_rows = []

epoch_rows = []

for split_name, idx in SPLITS.items():

    labels = Y[idx]

    summary_rows.append({
        "split": split_name,
        "epochs": len(idx),
        "epoch_fraction": len(idx) / len(Y),
        "recordings": len(
            np.unique(RECORDING[idx])
        ),
        "positive_epochs": int(
            labels.sum()
        ),
        "negative_epochs": int(
            (labels == 0).sum()
        ),
        "positive_rate": float(
            labels.mean()
        ),
    })

    for ii in idx:

        epoch_rows.append({
            "index": int(ii),
            "source_key": str(
                SOURCE_KEY[ii]
            ),
            "recording_id": str(
                RECORDING[ii]
            ),
            "label": int(Y[ii]),
            "split": split_name,
        })


SPLIT_SUMMARY = pd.DataFrame(
    summary_rows
)

SPLIT_SUMMARY.to_csv(
    AUDIT_DIR / "split_summary.csv",
    index=False,
)

pd.DataFrame(
    epoch_rows
).to_csv(
    AUDIT_DIR / "epoch_split_audit.csv",
    index=False,
)

print("\nCLEAN SPLIT:")
print(SPLIT_SUMMARY)


# ------------------------------------------------
# 11. TRAIN-ONLY PREPROCESSING
# ------------------------------------------------

def fit_preprocessor(x):

    mean = np.nanmean(
        x,
        axis=(0, 1),
        keepdims=True,
    )

    std = np.nanstd(
        x,
        axis=(0, 1),
        keepdims=True,
    )

    std = np.maximum(
        std,
        1e-6,
    )

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
    )


CHANNEL_MEAN, CHANNEL_STD = (
    fit_preprocessor(
        X_RAW[TRAIN_IDX]
    )
)


def transform_raw(x):

    fill = np.broadcast_to(
        CHANNEL_MEAN,
        x.shape,
    )

    x = np.where(
        np.isfinite(x),
        x,
        fill,
    )

    x = (
        x - CHANNEL_MEAN
    ) / CHANNEL_STD

    return x.astype(
        np.float32
    )


X = transform_raw(
    X_RAW
)

np.savez_compressed(
    AUDIT_DIR / "preprocessing_parameters.npz",
    channel_mean=CHANNEL_MEAN,
    channel_std=CHANNEL_STD,
)


# ------------------------------------------------
# 12. EVENT DESCRIPTOR FUNCTION
# ------------------------------------------------

def event_features_numpy(x):

    eps = 1e-8

    mean = x.mean(1)
    std = x.std(1)

    minimum = x.min(1)
    maximum = x.max(1)

    rng = maximum - minimum

    dx = np.diff(
        x,
        axis=1,
    )

    slope_mean = dx.mean(1)
    slope_std = dx.std(1)

    negative_drop = np.maximum(
        -dx,
        0,
    ).mean(1)

    positive_rise = np.maximum(
        dx,
        0,
    ).mean(1)

    instability = (
        slope_std
        / (std + eps)
    )

    low_tail = np.maximum(
        mean[:, None, :] - x,
        0,
    ).mean(1)

    high_tail = np.maximum(
        x - mean[:, None, :],
        0,
    ).mean(1)

    density = (
        np.abs(dx)
        > slope_std[:, None, :]
    ).mean(1)

    descriptors = np.stack(
        [
            mean,
            std,
            minimum,
            maximum,
            rng,
            slope_mean,
            slope_std,
            negative_drop,
            positive_rise,
            instability,
            low_tail,
            high_tail,
            density,
        ],
        axis=-1,
    )

    return descriptors.reshape(
        len(x),
        -1,
    ).astype(
        np.float32
    )


EVENT_NUMPY = event_features_numpy(
    X
)


# ------------------------------------------------
# 13. TORCH DATASET
# ------------------------------------------------

class PSGDataset(Dataset):

    def __init__(
        self,
        x,
        y,
    ):

        self.x = torch.from_numpy(
            x
        ).float()

        self.y = torch.from_numpy(
            y.astype(
                np.float32
            )
        )

    def __len__(self):

        return len(
            self.y
        )

    def __getitem__(
        self,
        idx,
    ):

        return (
            self.x[idx],
            self.y[idx],
        )


def make_loader(
    indices,
    train=False,
    batch_size=None,
    seed=271828,
):

    if batch_size is None:

        batch_size = (
            TRAIN_BATCH
            if train
            else EVAL_BATCH
        )

    ds = PSGDataset(
        X[indices],
        Y[indices],
    )

    if not train:

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(
                DEVICE.type == "cuda"
            ),
        )

    labels = Y[indices]

    counts = np.bincount(
        labels,
        minlength=2,
    )

    class_weight = (
        1.0
        / np.maximum(
            counts,
            1,
        )
    )

    sample_weight = (
        class_weight[labels]
    )

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weight,
            dtype=torch.double,
        ),
        num_samples=len(indices),
        replacement=True,
        generator=torch.Generator().manual_seed(
            seed
        ),
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )


# ------------------------------------------------
# 14. CHECKPOINT-FAITHFUL ARCHITECTURE
# ------------------------------------------------

class RawPath(nn.Module):

    def __init__(
        self,
        kernel,
        dropout=DROPOUT,
    ):

        super().__init__()

        pad = kernel // 2

        self.net = nn.Sequential(

            nn.Conv1d(
                4,
                32,
                kernel,
                padding=pad,
                bias=False,
            ),

            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.MaxPool1d(2),

            nn.Dropout(
                dropout
            ),

            nn.Conv1d(
                32,
                64,
                kernel,
                padding=pad,
                bias=False,
            ),

            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(
                1
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.net(
            x
        ).squeeze(-1)


class RawBackbone(nn.Module):

    def __init__(
        self,
        dropout=DROPOUT,
    ):

        super().__init__()

        self.paths = nn.ModuleList([
            RawPath(
                3,
                dropout,
            ),
            RawPath(
                7,
                dropout,
            ),
            RawPath(
                15,
                dropout,
            ),
        ])

    def forward(
        self,
        signal,
    ):

        # [B,T,C] -> [B,C,T]
        x = signal.transpose(
            1,
            2,
        )

        z = [
            path(x)
            for path in self.paths
        ]

        return torch.cat(
            z,
            dim=1,
        )


class EventDescriptorLayer(nn.Module):

    bases = (
        "mean",
        "std",
        "min",
        "max",
        "range",
        "slope_mean",
        "slope_std",
        "negative_drop",
        "positive_rise",
        "instability",
        "low_tail",
        "high_tail",
        "event_density",
    )

    def forward(
        self,
        x,
    ):

        mean = x.mean(1)

        std = x.std(
            1,
            unbiased=False,
        )

        xmin = x.amin(1)
        xmax = x.amax(1)

        dx = (
            x[:, 1:]
            - x[:, :-1]
        )

        slope_std = dx.std(
            1,
            unbiased=False,
        )

        return torch.cat(
            [
                mean,
                std,
                xmin,
                xmax,
                xmax - xmin,
                dx.mean(1),
                slope_std,
                F.relu(-dx).mean(1),
                F.relu(dx).mean(1),
                slope_std / (
                    std + 1e-6
                ),
                F.relu(
                    mean[:, None] - x
                ).mean(1),
                F.relu(
                    x - mean[:, None]
                ).mean(1),
                (
                    dx.abs()
                    > slope_std[:, None]
                ).float().mean(1),
            ],
            dim=1,
        )


class FullTrustSleep(nn.Module):

    def __init__(
        self,
        dropout=DROPOUT,
    ):

        super().__init__()

        self.raw = RawBackbone(
            dropout
        )

        self.descriptors = (
            EventDescriptorLayer()
        )

        self.gate = nn.Sequential(
            nn.LayerNorm(52),
            nn.Linear(
                52,
                52,
            ),
            nn.Sigmoid(),
        )

        self.event_encoder = nn.Sequential(

            nn.LayerNorm(52),

            nn.Linear(
                52,
                128,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                80,
            ),

            nn.GELU(),
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                272,
                256,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                256,
                128,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                1,
            ),
        )

    def forward(
        self,
        signal,
        return_explanation=False,
    ):

        raw = self.raw(
            signal
        )

        desc = self.descriptors(
            signal
        )

        gates = self.gate(
            desc
        )

        contribution = (
            desc * gates
        )

        event = self.event_encoder(
            contribution
        )

        fused = torch.cat(
            [
                raw,
                event,
            ],
            dim=1,
        )

        logit = self.classifier(
            fused
        ).squeeze(1)

        if return_explanation:

            return (
                logit,
                desc,
                gates,
                contribution,
            )

        return logit


class RawOnly(nn.Module):

    def __init__(
        self,
        dropout=DROPOUT,
    ):

        super().__init__()

        self.raw = RawBackbone(
            dropout
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                192,
                256,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                256,
                128,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                1,
            ),
        )

    def forward(
        self,
        signal,
    ):

        raw = self.raw(
            signal
        )

        return self.classifier(
            raw
        ).squeeze(1)


def create_model(
    variant,
):

    if variant == "full":
        return FullTrustSleep()

    if variant == "raw_only":
        return RawOnly()

    raise ValueError(
        variant
    )


# ------------------------------------------------
# 15. TRAIN / PREDICT UTILITIES
# ------------------------------------------------

@torch.no_grad()
def predict_logits(
    model,
    indices,
):

    model.eval()

    loader = make_loader(
        indices,
        train=False,
    )

    output = []

    for xb, _ in loader:

        xb = xb.to(
            DEVICE,
            non_blocking=True,
        )

        with autocast(
            device_type=DEVICE.type,
            enabled=(
                DEVICE.type == "cuda"
            ),
        ):
            logits = model(
                xb
            )

        output.append(
            logits.float()
            .cpu()
            .numpy()
        )

    return np.concatenate(
        output
    )


def validation_bce(
    model,
    indices,
):

    logits = predict_logits(
        model,
        indices,
    )

    logits_t = torch.tensor(
        logits,
        dtype=torch.float32,
    )

    y_t = torch.tensor(
        Y[indices],
        dtype=torch.float32,
    )

    return float(
        F.binary_cross_entropy_with_logits(
            logits_t,
            y_t,
        )
    )


def train_one_model(
    variant,
    seed,
):

    seed_everything(
        seed
    )

    path = (
        CHECKPOINT_DIR
        / f"{variant}_seed_{seed}.pt"
    )

    history_path = (
        CHECKPOINT_DIR
        / f"{variant}_seed_{seed}_history.csv"
    )

    if path.exists():

        print(
            "Checkpoint already exists:",
            path.name,
        )

        return path

    model = create_model(
        variant
    ).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = GradScaler(
        DEVICE.type,
        enabled=(
            DEVICE.type == "cuda"
        ),
    )

    loader = make_loader(
        TRAIN_IDX,
        train=True,
        seed=seed,
    )

    best_state = None
    best_loss = np.inf

    bad_epochs = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        losses = []

        for xb, yb in tqdm(
            loader,
            desc=(
                f"{variant} "
                f"{seed} "
                f"epoch {epoch}"
            ),
            leave=False,
        ):

            xb = xb.to(
                DEVICE,
                non_blocking=True,
            )

            yb = yb.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with autocast(
                device_type=DEVICE.type,
                enabled=(
                    DEVICE.type
                    == "cuda"
                ),
            ):

                logits = model(
                    xb
                )

                loss = (
                    F.binary_cross_entropy_with_logits(
                        logits,
                        yb,
                    )
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            losses.append(
                float(
                    loss.detach()
                    .cpu()
                )
            )

        val_loss = validation_bce(
            model,
            VAL_IDX,
        )

        row = {
            "epoch": epoch,
            "train_bce": float(
                np.mean(losses)
            ),
            "validation_bce": val_loss,
        }

        history.append(
            row
        )

        print(
            variant,
            seed,
            row,
        )

        if (
            val_loss
            < best_loss - 1e-8
        ):

            best_loss = val_loss

            best_state = {
                k:
                v.detach()
                .cpu()
                .clone()
                for k, v
                in model.state_dict().items()
            }

            bad_epochs = 0

        else:

            bad_epochs += 1

            if bad_epochs >= PATIENCE:
                print(
                    "Early stopping."
                )
                break

    payload = {
        "variant": variant,
        "seed": seed,
        "model_state": best_state,
        "best_validation_bce": best_loss,
        "architecture": (
            "checkpoint-faithful "
            "three-path raw CNN"
        ),
    }

    torch.save(
        payload,
        path,
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return path


def load_model(
    checkpoint,
):

    payload = torch.load(
        checkpoint,
        map_location="cpu",
    )

    model = create_model(
        payload["variant"]
    )

    model.load_state_dict(
        payload["model_state"]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ------------------------------------------------
# 16. TRAIN FULL + RAW-ONLY, FIVE SEEDS
# ------------------------------------------------

CHECKPOINTS = {
    "full": [],
    "raw_only": [],
}

for variant in [
    "full",
    "raw_only",
]:

    print(
        "\n========================================"
    )
    print(
        "TRAINING:",
        variant
    )
    print(
        "========================================"
    )

    for seed in SEEDS:

        cp = train_one_model(
            variant,
            seed,
        )

        CHECKPOINTS[
            variant
        ].append(
            cp
        )


# ------------------------------------------------
# 17. PROBABILITY CALIBRATION
# ------------------------------------------------

def fit_platt_from_logits(
    logits,
    labels,
):

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    model.fit(
        np.asarray(
            logits
        ).reshape(-1, 1),
        np.asarray(
            labels
        ).astype(int),
    )

    return model


def apply_platt(
    model,
    logits,
):

    return model.predict_proba(
        np.asarray(
            logits
        ).reshape(-1, 1)
    )[:, 1]


def best_f1_threshold(
    y,
    p,
):

    thresholds = np.unique(
        p
    )

    best_t = None
    best_score = -1

    for t in thresholds:

        score = f1_score(
            y,
            p >= t,
            zero_division=0,
        )

        if (
            score > best_score
            or (
                score == best_score
                and (
                    best_t is None
                    or t > best_t
                )
            )
        ):

            best_score = score
            best_t = float(t)

    return float(
        best_t
    )


def conformal_quantile(
    y,
    p,
    alpha=0.10,
):

    scores = np.where(
        y == 1,
        1.0 - p,
        p,
    )

    n = len(scores)

    level = min(
        math.ceil(
            (n + 1)
            * (1 - alpha)
        ) / n,
        1.0,
    )

    try:

        return float(
            np.quantile(
                scores,
                level,
                method="higher",
            )
        )

    except TypeError:

        return float(
            np.quantile(
                scores,
                level,
                interpolation="higher",
            )
        )


def conformal_sets(
    p,
    q,
):

    include0 = (
        p <= q
    )

    include1 = (
        (1 - p) <= q
    )

    size = (
        include0.astype(int)
        + include1.astype(int)
    )

    return (
        include0,
        include1,
        size,
    )


# ------------------------------------------------
# 18. METRICS
# ------------------------------------------------

def ece_score(
    y,
    p,
    bins=15,
):

    edges = np.linspace(
        0,
        1,
        bins + 1,
    )

    index = np.digitize(
        p,
        edges[1:-1],
    )

    ece = 0.0

    for b in range(
        bins
    ):

        mask = (
            index == b
        )

        if not mask.any():
            continue

        confidence = (
            p[mask].mean()
        )

        observed = (
            y[mask].mean()
        )

        ece += (
            mask.mean()
            * abs(
                confidence
                - observed
            )
        )

    return float(
        ece
    )


def metric_dict(
    y,
    p,
    threshold,
):

    pred = (
        p >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            pred,
            labels=[0, 1],
        ).ravel()
    )

    return {
        "accuracy":
            accuracy_score(
                y,
                pred,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                pred,
            ),

        "precision":
            precision_score(
                y,
                pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y,
                pred,
                zero_division=0,
            ),

        "specificity":
            tn / max(
                tn + fp,
                1,
            ),

        "f1":
            f1_score(
                y,
                pred,
                zero_division=0,
            ),

        "roc_auc":
            roc_auc_score(
                y,
                p,
            ),

        "average_precision":
            average_precision_score(
                y,
                p,
            ),

        "brier":
            brier_score_loss(
                y,
                p,
            ),

        "ece":
            ece_score(
                y,
                p,
            ),

        "threshold":
            threshold,

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ------------------------------------------------
# 19. SEED-LEVEL EXPERIMENT 2 EVALUATION
# ------------------------------------------------

seed_result_rows = []

seed_prediction_frames = []

SEED_ARTIFACTS = {}

for variant in [
    "full",
    "raw_only",
]:

    SEED_ARTIFACTS[
        variant
    ] = []

    for seed, cp in zip(
        SEEDS,
        CHECKPOINTS[variant],
    ):

        print(
            "\nEvaluating",
            variant,
            seed,
        )

        model = load_model(
            cp
        )

        platt_logits = predict_logits(
            model,
            PLATT_IDX,
        )

        platt = fit_platt_from_logits(
            platt_logits,
            Y[PLATT_IDX],
        )

        threshold_logits = predict_logits(
            model,
            THRESHOLD_IDX,
        )

        threshold_prob = apply_platt(
            platt,
            threshold_logits,
        )

        threshold = best_f1_threshold(
            Y[THRESHOLD_IDX],
            threshold_prob,
        )

        conformal_logits = predict_logits(
            model,
            CONFORMAL_IDX,
        )

        conformal_prob = apply_platt(
            platt,
            conformal_logits,
        )

        q = conformal_quantile(
            Y[CONFORMAL_IDX],
            conformal_prob,
            CONFORMAL_ALPHA,
        )

        test_logits = predict_logits(
            model,
            TEST_IDX,
        )

        test_prob = apply_platt(
            platt,
            test_logits,
        )

        metrics = metric_dict(
            Y[TEST_IDX],
            test_prob,
            threshold,
        )

        inc0, inc1, set_size = (
            conformal_sets(
                test_prob,
                q,
            )
        )

        coverage = np.mean(
            np.where(
                Y[TEST_IDX] == 0,
                inc0,
                inc1,
            )
        )

        metrics.update({
            "variant": variant,
            "seed": seed,
            "conformal_q": q,
            "conformal_coverage":
                float(
                    coverage
                ),
            "singleton_rate":
                float(
                    np.mean(
                        set_size == 1
                    )
                ),
            "empty_rate":
                float(
                    np.mean(
                        set_size == 0
                    )
                ),
            "doubleton_rate":
                float(
                    np.mean(
                        set_size == 2
                    )
                ),
        })

        seed_result_rows.append(
            metrics
        )

        frame = pd.DataFrame({
            "variant": variant,
            "seed": seed,
            "recording_id":
                RECORDING[TEST_IDX],
            "source_key":
                SOURCE_KEY[TEST_IDX],
            "y_true":
                Y[TEST_IDX],
            "logit":
                test_logits,
            "probability":
                test_prob,
            "prediction":
                (
                    test_prob
                    >= threshold
                ).astype(int),
            "conformal_size":
                set_size,
        })

        seed_prediction_frames.append(
            frame
        )

        SEED_ARTIFACTS[
            variant
        ].append({
            "seed": seed,
            "checkpoint": cp,
            "platt": platt,
            "threshold": threshold,
            "q": q,
        })

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


SEED_RESULTS = pd.DataFrame(
    seed_result_rows
)

SEED_RESULTS.to_csv(
    TABLE_DIR
    / "Experiment2_seed_level_metrics.csv",
    index=False,
)

SEED_PREDICTIONS = pd.concat(
    seed_prediction_frames,
    ignore_index=True,
)

SEED_PREDICTIONS.to_csv(
    TABLE_DIR
    / "Experiment2_seed_level_predictions.csv",
    index=False,
)


# ------------------------------------------------
# 20. FIVE-SEED ENSEMBLE EVALUATION
# ------------------------------------------------

ENSEMBLE_OUTPUT = {}

ensemble_rows = []

ensemble_prediction_frames = []

for variant in [
    "full",
    "raw_only",
]:

    print(
        "\nFive-seed ensemble:",
        variant,
    )

    models = [
        load_model(cp)
        for cp
        in CHECKPOINTS[
            variant
        ]
    ]

    def mean_logits(indices):

        logits = [
            predict_logits(
                model,
                indices,
            )
            for model
            in models
        ]

        return np.mean(
            np.stack(
                logits,
                axis=0,
            ),
            axis=0,
        )

    platt_logits = mean_logits(
        PLATT_IDX
    )

    platt = fit_platt_from_logits(
        platt_logits,
        Y[PLATT_IDX],
    )

    threshold_logits = mean_logits(
        THRESHOLD_IDX
    )

    threshold_prob = apply_platt(
        platt,
        threshold_logits,
    )

    threshold = best_f1_threshold(
        Y[THRESHOLD_IDX],
        threshold_prob,
    )

    conformal_logits = mean_logits(
        CONFORMAL_IDX
    )

    conformal_prob = apply_platt(
        platt,
        conformal_logits,
    )

    q = conformal_quantile(
        Y[CONFORMAL_IDX],
        conformal_prob,
        CONFORMAL_ALPHA,
    )

    test_member_logits = np.stack(
        [
            predict_logits(
                model,
                TEST_IDX,
            )
            for model
            in models
        ],
        axis=0,
    )

    test_logits = (
        test_member_logits.mean(
            axis=0
        )
    )

    test_prob = apply_platt(
        platt,
        test_logits,
    )

    ensemble_variance = (
        test_member_logits.var(
            axis=0
        )
    )

    metrics = metric_dict(
        Y[TEST_IDX],
        test_prob,
        threshold,
    )

    inc0, inc1, size = (
        conformal_sets(
            test_prob,
            q,
        )
    )

    coverage = np.mean(
        np.where(
            Y[TEST_IDX] == 0,
            inc0,
            inc1,
        )
    )

    metrics.update({
        "variant": variant,
        "seed_count": len(SEEDS),
        "conformal_q": q,
        "conformal_coverage":
            float(
                coverage
            ),
        "singleton_rate":
            float(
                np.mean(
                    size == 1
                )
            ),
        "empty_rate":
            float(
                np.mean(
                    size == 0
                )
            ),
        "doubleton_rate":
            float(
                np.mean(
                    size == 2
                )
            ),
    })

    ensemble_rows.append(
        metrics
    )

    pred = (
        test_prob
        >= threshold
    ).astype(int)

    frame = pd.DataFrame({
        "variant": variant,
        "recording_id":
            RECORDING[TEST_IDX],
        "source_key":
            SOURCE_KEY[TEST_IDX],
        "y_true":
            Y[TEST_IDX],
        "probability":
            test_prob,
        "prediction":
            pred,
        "correct":
            (
                pred
                == Y[TEST_IDX]
            ).astype(int),
        "between_seed_logit_variance":
            ensemble_variance,
        "conformal_set_size":
            size,
    })

    ensemble_prediction_frames.append(
        frame
    )

    ENSEMBLE_OUTPUT[
        variant
    ] = {
        "models": models,
        "platt": platt,
        "threshold": threshold,
        "q": q,
        "probability": test_prob,
        "prediction": pred,
        "variance": ensemble_variance,
        "metrics": metrics,
    }


ENSEMBLE_METRICS = pd.DataFrame(
    ensemble_rows
)

ENSEMBLE_METRICS.to_csv(
    TABLE_DIR
    / "Experiment2_ensemble_metrics.csv",
    index=False,
)

ENSEMBLE_PREDICTIONS = pd.concat(
    ensemble_prediction_frames,
    ignore_index=True,
)

ENSEMBLE_PREDICTIONS.to_csv(
    TABLE_DIR
    / "Experiment2_ensemble_predictions.csv",
    index=False,
)


# ------------------------------------------------
# 21. SEED MEAN ± SD
# ------------------------------------------------

metric_columns = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "average_precision",
    "brier",
    "ece",
    "conformal_coverage",
]

seed_summary_rows = []

for variant in [
    "full",
    "raw_only",
]:

    sub = SEED_RESULTS[
        SEED_RESULTS.variant
        == variant
    ]

    row = {
        "variant": variant,
    }

    for metric in metric_columns:

        row[
            metric + "_mean"
        ] = sub[
            metric
        ].mean()

        row[
            metric + "_sd"
        ] = sub[
            metric
        ].std(
            ddof=1
        )

    seed_summary_rows.append(
        row
    )


SEED_SUMMARY = pd.DataFrame(
    seed_summary_rows
)

SEED_SUMMARY.to_csv(
    TABLE_DIR
    / "Experiment2_five_seed_mean_sd.csv",
    index=False,
)


# ------------------------------------------------
# 22. PAIRED RECORDING-LEVEL BOOTSTRAP
# ------------------------------------------------

def metric_by_name(
    metric_name,
    y,
    p,
    pred,
):

    if metric_name == "accuracy":
        return accuracy_score(
            y,
            pred,
        )

    if metric_name == "balanced_accuracy":
        return balanced_accuracy_score(
            y,
            pred,
        )

    if metric_name == "precision":
        return precision_score(
            y,
            pred,
            zero_division=0,
        )

    if metric_name == "recall":
        return recall_score(
            y,
            pred,
            zero_division=0,
        )

    if metric_name == "f1":
        return f1_score(
            y,
            pred,
            zero_division=0,
        )

    if metric_name == "specificity":

        tn, fp, _, _ = (
            confusion_matrix(
                y,
                pred,
                labels=[0, 1],
            ).ravel()
        )

        return tn / max(
            tn + fp,
            1,
        )

    if metric_name == "roc_auc":

        if len(np.unique(y)) < 2:
            return np.nan

        return roc_auc_score(
            y,
            p,
        )

    if metric_name == "average_precision":

        if len(np.unique(y)) < 2:
            return np.nan

        return average_precision_score(
            y,
            p,
        )

    if metric_name == "brier":

        return brier_score_loss(
            y,
            p,
        )

    raise ValueError(
        metric_name
    )


FULL = ENSEMBLE_OUTPUT[
    "full"
]

RAW = ENSEMBLE_OUTPUT[
    "raw_only"
]

test_groups = RECORDING[
    TEST_IDX
]

unique_groups = np.unique(
    test_groups
)

group_indices = {
    g:
    np.where(
        test_groups == g
    )[0]
    for g
    in unique_groups
}

BOOT_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "average_precision",
    "brier",
]

rng = np.random.default_rng(
    271828
)

bootstrap_rows = []

for b in tqdm(
    range(BOOTSTRAP_REPS),
    desc="Paired recording bootstrap",
):

    sampled_groups = rng.choice(
        unique_groups,
        size=len(
            unique_groups
        ),
        replace=True,
    )

    idx = np.concatenate(
        [
            group_indices[g]
            for g
            in sampled_groups
        ]
    )

    yb = Y[
        TEST_IDX
    ][idx]

    full_p = FULL[
        "probability"
    ][idx]

    raw_p = RAW[
        "probability"
    ][idx]

    full_pred = FULL[
        "prediction"
    ][idx]

    raw_pred = RAW[
        "prediction"
    ][idx]

    row = {
        "bootstrap": b,
    }

    for metric in BOOT_METRICS:

        f = metric_by_name(
            metric,
            yb,
            full_p,
            full_pred,
        )

        r = metric_by_name(
            metric,
            yb,
            raw_p,
            raw_pred,
        )

        row[
            metric
        ] = (
            f - r
        )

    bootstrap_rows.append(
        row
    )


BOOTSTRAP = pd.DataFrame(
    bootstrap_rows
)

BOOTSTRAP.to_csv(
    TABLE_DIR
    / "Experiment2_paired_recording_bootstrap_raw.csv",
    index=False,
)

ci_rows = []

for metric in BOOT_METRICS:

    values = (
        BOOTSTRAP[
            metric
        ]
        .dropna()
        .to_numpy()
    )

    point_full = metric_by_name(
        metric,
        Y[TEST_IDX],
        FULL["probability"],
        FULL["prediction"],
    )

    point_raw = metric_by_name(
        metric,
        Y[TEST_IDX],
        RAW["probability"],
        RAW["prediction"],
    )

    ci_rows.append({
        "metric": metric,
        "full": point_full,
        "raw_only": point_raw,
        "difference_full_minus_raw":
            point_full - point_raw,
        "bootstrap_lower_95":
            np.quantile(
                values,
                0.025,
            ),
        "bootstrap_upper_95":
            np.quantile(
                values,
                0.975,
            ),
    })


BOOTSTRAP_CI = pd.DataFrame(
    ci_rows
)

BOOTSTRAP_CI.to_csv(
    TABLE_DIR
    / "Experiment2_paired_recording_bootstrap_CI.csv",
    index=False,
)


# ------------------------------------------------
# 23. EXPERIMENT 1 — SINGLE FULL MODEL
# ------------------------------------------------

EXP1_MODEL = load_model(
    CHECKPOINTS[
        "full"
    ][0]
)

EXP1_SEED = SEEDS[0]

# Dedicated calibration
exp1_platt_logits = predict_logits(
    EXP1_MODEL,
    PLATT_IDX,
)

exp1_platt = fit_platt_from_logits(
    exp1_platt_logits,
    Y[PLATT_IDX],
)

exp1_threshold_logits = predict_logits(
    EXP1_MODEL,
    THRESHOLD_IDX,
)

exp1_threshold_prob = apply_platt(
    exp1_platt,
    exp1_threshold_logits,
)

exp1_threshold = best_f1_threshold(
    Y[THRESHOLD_IDX],
    exp1_threshold_prob,
)

exp1_conformal_logits = predict_logits(
    EXP1_MODEL,
    CONFORMAL_IDX,
)

exp1_conformal_prob = apply_platt(
    exp1_platt,
    exp1_conformal_logits,
)

exp1_q = conformal_quantile(
    Y[CONFORMAL_IDX],
    exp1_conformal_prob,
    CONFORMAL_ALPHA,
)

exp1_test_logits = predict_logits(
    EXP1_MODEL,
    TEST_IDX,
)

exp1_prob = apply_platt(
    exp1_platt,
    exp1_test_logits,
)

exp1_pred = (
    exp1_prob
    >= exp1_threshold
).astype(int)

EXP1_METRICS = metric_dict(
    Y[TEST_IDX],
    exp1_prob,
    exp1_threshold,
)

EXP1_METRICS.update({
    "seed": EXP1_SEED,
    "conformal_q": exp1_q,
})

pd.DataFrame(
    [EXP1_METRICS]
).to_csv(
    TABLE_DIR
    / "Experiment1_predictive_metrics.csv",
    index=False,
)


# ------------------------------------------------
# 24. EXPERIMENT 1 — MC-DROPOUT UNCERTAINTY
# ------------------------------------------------

def enable_dropout_only(
    model,
):

    for module in model.modules():

        if isinstance(
            module,
            nn.Dropout,
        ):

            module.train()


@torch.no_grad()
def mc_dropout_probs(
    model,
    indices,
    platt,
    passes=15,
):

    loader = make_loader(
        indices,
        train=False,
    )

    all_passes = []

    for p in range(
        passes
    ):

        model.eval()
        enable_dropout_only(
            model
        )

        pass_logits = []

        for xb, _ in loader:

            xb = xb.to(
                DEVICE
            )

            logits = model(
                xb
            )

            pass_logits.append(
                logits.cpu()
                .numpy()
            )

        logits = np.concatenate(
            pass_logits
        )

        probs = apply_platt(
            platt,
            logits,
        )

        all_passes.append(
            probs
        )

    arr = np.stack(
        all_passes,
        axis=0,
    )

    return (
        arr.mean(0),
        arr.var(0),
    )


_, exp1_conf_uncertainty = (
    mc_dropout_probs(
        EXP1_MODEL,
        CONFORMAL_IDX,
        exp1_platt,
        passes=MC_DROPOUT_PASSES,
    )
)

_, exp1_test_uncertainty = (
    mc_dropout_probs(
        EXP1_MODEL,
        TEST_IDX,
        exp1_platt,
        passes=MC_DROPOUT_PASSES,
    )
)


# ------------------------------------------------
# 25. EXPERIMENT 1 — EXPLANATION CONTRIBUTIONS
# ------------------------------------------------

@torch.no_grad()
def get_explanations(
    model,
    indices,
    perturb=False,
    perturb_seed=271828,
):

    model.eval()

    loader = make_loader(
        indices,
        train=False,
    )

    contributions = []

    rng = np.random.default_rng(
        perturb_seed
    )

    for xb, _ in loader:

        if perturb:

            arr = xb.numpy()

            noise = rng.normal(
                0,
                0.02,
                size=arr.shape,
            ).astype(
                np.float32
            )

            mask = (
                rng.random(
                    arr.shape
                )
                < 0.01
            )

            arr = arr + noise
            arr[mask] = 0

            xb = torch.from_numpy(
                arr
            )

        xb = xb.to(
            DEVICE
        )

        (
            _,
            _,
            _,
            contribution,
        ) = model(
            xb,
            return_explanation=True,
        )

        contributions.append(
            contribution
            .cpu()
            .numpy()
        )

    return np.vstack(
        contributions
    )


PHI = get_explanations(
    EXP1_MODEL,
    TEST_IDX,
)

PHI_PERT = get_explanations(
    EXP1_MODEL,
    TEST_IDX,
    perturb=True,
)


# ------------------------------------------------
# 26. TRUST COMPONENT HELPERS
# ------------------------------------------------

def robust_scale01(
    values,
    reference,
):

    lo = np.quantile(
        reference,
        0.01,
    )

    hi = np.quantile(
        reference,
        0.99,
    )

    return np.clip(
        (
            values - lo
        )
        / max(
            hi - lo,
            1e-12,
        ),
        0,
        1,
    )


# Prediction confidence
confidence = np.maximum(
    exp1_prob,
    1 - exp1_prob,
)

uncertainty_reliability = (
    1
    - robust_scale01(
        exp1_test_uncertainty,
        exp1_conf_uncertainty,
    )
)


# Conformal reliability
inc0, inc1, exp1_set_size = (
    conformal_sets(
        exp1_prob,
        exp1_q,
    )
)

conformal_reliability = np.where(
    exp1_set_size == 1,
    1.0,
    0.25,
)


# Physiological evidence from descriptor magnitude
test_event = EVENT_NUMPY[
    TEST_IDX
]

conf_event = EVENT_NUMPY[
    CONFORMAL_IDX
]

test_event_mag = (
    np.abs(
        test_event
    ).mean(1)
)

conf_event_mag = (
    np.abs(
        conf_event
    ).mean(1)
)

phys_evidence = robust_scale01(
    test_event_mag,
    conf_event_mag,
)


# Signal quality
raw_test = X_RAW[
    TEST_IDX
]

finite_score = (
    1
    - MISSING_FRACTION[
        TEST_IDX
    ]
)

variation = np.nanstd(
    raw_test,
    axis=1,
).mean(1)

conf_variation = np.nanstd(
    X_RAW[
        CONFORMAL_IDX
    ],
    axis=1,
).mean(1)

variation_score = robust_scale01(
    variation,
    conf_variation,
)

signal_quality = np.clip(
    0.5 * finite_score
    + 0.5 * variation_score,
    0,
    1,
)


# CDT
CDT = np.clip(
    0.30 * confidence
    + 0.25 * uncertainty_reliability
    + 0.20 * conformal_reliability
    + 0.15 * phys_evidence
    + 0.10 * signal_quality,
    0,
    1,
)


# ------------------------------------------------
# 27. CERA COMPONENTS
# ------------------------------------------------

# S: perturbation stability
denominator = max(
    np.linalg.norm(
        PHI,
        axis=1,
    ).max(),
    1e-8,
)

S = np.clip(
    1
    - (
        np.linalg.norm(
            PHI - PHI_PERT,
            axis=1,
        )
        / denominator
    ),
    0,
    1,
)


# T: temporal consistency
T = np.ones(
    len(TEST_IDX),
    dtype=float,
)

test_recording = RECORDING[
    TEST_IDX
]

for recording in np.unique(
    test_recording
):

    idx = np.where(
        test_recording
        == recording
    )[0]

    if len(idx) <= 1:
        continue

    # SOURCE_KEY is sorted globally, so retain test order.
    for j in range(
        1,
        len(idx),
    ):

        current = idx[j]
        previous = idx[j - 1]

        T[current] = np.clip(
            1
            - (
                np.linalg.norm(
                    PHI[current]
                    - PHI[previous]
                )
                / denominator
            ),
            0,
            1,
        )


# R_exp: consistency between prediction confidence and CDT
R_EXP = np.clip(
    1
    - np.abs(
        CDT - confidence
    ),
    0,
    1,
)


# P_phys: share of attribution magnitude on
# change / instability / tail / density descriptors.
#
# Descriptor layout from EventDescriptorLayer is concatenated
# by descriptor block across four channels:
# [mean(4), std(4), ..., event_density(4)].
important_descriptor_blocks = [
    7,   # negative_drop
    8,   # positive_rise
    9,   # instability
    10,  # low_tail
    11,  # high_tail
    12,  # event_density
]

important_indices = []

for block in important_descriptor_blocks:

    important_indices.extend(
        range(
            block * 4,
            block * 4 + 4,
        )
    )

abs_phi = np.abs(
    PHI
)

P_PHYS = (
    abs_phi[
        :,
        important_indices
    ].sum(1)
    / (
        abs_phi.sum(1)
        + 1e-8
    )
)

P_PHYS = np.clip(
    P_PHYS,
    0,
    1,
)


# H: automated interpretability/concentration heuristic.
sorted_mass = np.sort(
    abs_phi,
    axis=1,
)[:, ::-1]

top_k = max(
    1,
    int(
        round(
            abs_phi.shape[1]
            * 0.20
        )
    ),
)

H = (
    sorted_mass[
        :,
        :top_k
    ].sum(1)
    / (
        sorted_mass.sum(1)
        + 1e-8
    )
)

H = np.clip(
    H,
    0,
    1,
)


# Original Experiment 1 five-component CERA
CERA = np.clip(
    (
        S
        + T
        + R_EXP
        + P_PHYS
        + H
    )
    / 5.0,
    0,
    1,
)


CAS = np.clip(
    0.60 * CDT
    + 0.40 * CERA,
    0,
    1,
)


# ------------------------------------------------
# 28. SAVE EXPERIMENT 1 PREDICTIONS
# ------------------------------------------------

correct = (
    exp1_pred
    == Y[TEST_IDX]
).astype(int)

EXP1_PRED = pd.DataFrame({
    "recording_id":
        RECORDING[TEST_IDX],

    "source_key":
        SOURCE_KEY[TEST_IDX],

    "y_true":
        Y[TEST_IDX],

    "probability":
        exp1_prob,

    "prediction":
        exp1_pred,

    "correct":
        correct,

    "uncertainty":
        exp1_test_uncertainty,

    "confidence":
        confidence,

    "uncertainty_reliability":
        uncertainty_reliability,

    "conformal_reliability":
        conformal_reliability,

    "physiological_evidence":
        phys_evidence,

    "signal_quality":
        signal_quality,

    "cera_stability":
        S,

    "cera_temporal":
        T,

    "cera_reliability":
        R_EXP,

    "cera_phys_plausibility":
        P_PHYS,

    "cera_interpretability_heuristic":
        H,

    "cdt":
        CDT,

    "cera":
        CERA,

    "cas":
        CAS,

    "conformal_set_size":
        exp1_set_size,
})

EXP1_PRED.to_csv(
    TABLE_DIR
    / "Experiment1_test_predictions_trust.csv",
    index=False,
)


# ------------------------------------------------
# 29. EXPERIMENT 1 TRUST STATISTICS
# ------------------------------------------------

def safe_corr(
    func,
    x,
    y,
):

    try:
        return float(
            func(
                x,
                y,
            )[0]
        )
    except Exception:
        return np.nan


trust_rows = []

for score_name in [
    "cdt",
    "cera",
    "cas",
]:

    score = EXP1_PRED[
        score_name
    ].to_numpy()

    trust_rows.append({
        "score": score_name.upper(),

        "pearson_correctness":
            safe_corr(
                pearsonr,
                score,
                correct,
            ),

        "spearman_correctness":
            safe_corr(
                spearmanr,
                score,
                correct,
            ),

        "pearson_uncertainty":
            safe_corr(
                pearsonr,
                score,
                exp1_test_uncertainty,
            ),

        "spearman_uncertainty":
            safe_corr(
                spearmanr,
                score,
                exp1_test_uncertainty,
            ),

        "mean_correct":
            float(
                score[
                    correct == 1
                ].mean()
            ),

        "mean_incorrect":
            float(
                score[
                    correct == 0
                ].mean()
            ),
    })


TRUST_STATS = pd.DataFrame(
    trust_rows
)

TRUST_STATS.to_csv(
    TABLE_DIR
    / "Experiment1_trust_correlations.csv",
    index=False,
)


# ------------------------------------------------
# 30. EXPERIMENT 1 SELECTIVE PREDICTION
# ------------------------------------------------

selective_rows = []

for score_name in [
    "cdt",
    "cas",
]:

    score = EXP1_PRED[
        score_name
    ].to_numpy()

    for threshold in np.linspace(
        0,
        1,
        101,
    ):

        accepted = (
            score >= threshold
        )

        n = int(
            accepted.sum()
        )

        if n < 10:
            continue

        y_acc = Y[
            TEST_IDX
        ][accepted]

        p_acc = exp1_pred[
            accepted
        ]

        selective_rows.append({
            "score":
                score_name.upper(),

            "threshold":
                threshold,

            "coverage":
                float(
                    accepted.mean()
                ),

            "n_accepted":
                n,

            "accuracy":
                accuracy_score(
                    y_acc,
                    p_acc,
                ),

            "balanced_accuracy":
                balanced_accuracy_score(
                    y_acc,
                    p_acc,
                )
                if len(
                    np.unique(
                        y_acc
                    )
                ) == 2
                else np.nan,

            "precision":
                precision_score(
                    y_acc,
                    p_acc,
                    zero_division=0,
                ),

            "recall":
                recall_score(
                    y_acc,
                    p_acc,
                    zero_division=0,
                ),

            "f1":
                f1_score(
                    y_acc,
                    p_acc,
                    zero_division=0,
                ),

            "accepted_error":
                1
                - accuracy_score(
                    y_acc,
                    p_acc,
                ),
        })


SELECTIVE = pd.DataFrame(
    selective_rows
)

SELECTIVE.to_csv(
    TABLE_DIR
    / "Experiment1_selective_prediction.csv",
    index=False,
)


# ------------------------------------------------
# 31. EXPERIMENT 1 PRIVACY-MOTIVATED
#     PERTURBATION ROBUSTNESS
# ------------------------------------------------

epsilon_rows = []

rng = np.random.default_rng(
    271828
)

# IMPORTANT:
# This is robustness to epsilon-parameterized Laplace perturbations.
# It is NOT a formal differential privacy guarantee.

for epsilon in [
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
]:

    perturbed = X[
        TEST_IDX
    ].copy()

    # Prespecified base perturbation magnitude in normalized units.
    scale = (
        0.05
        / epsilon
    )

    perturbed += rng.laplace(
        0,
        scale,
        size=perturbed.shape,
    ).astype(
        np.float32
    )

    ds = PSGDataset(
        perturbed,
        Y[TEST_IDX],
    )

    loader = DataLoader(
        ds,
        batch_size=EVAL_BATCH,
        shuffle=False,
        num_workers=0,
    )

    logits_list = []

    EXP1_MODEL.eval()

    with torch.no_grad():

        for xb, _ in loader:

            xb = xb.to(
                DEVICE
            )

            logits_list.append(
                EXP1_MODEL(
                    xb
                )
                .cpu()
                .numpy()
            )

    logits = np.concatenate(
        logits_list
    )

    probs = apply_platt(
        exp1_platt,
        logits,
    )

    metrics = metric_dict(
        Y[TEST_IDX],
        probs,
        exp1_threshold,
    )

    epsilon_rows.append({
        "epsilon_parameter":
            epsilon,

        "laplace_scale":
            scale,

        **metrics,
    })


PERTURBATION = pd.DataFrame(
    epsilon_rows
)

PERTURBATION.to_csv(
    TABLE_DIR
    / "Experiment1_privacy_motivated_perturbation.csv",
    index=False,
)


# ------------------------------------------------
# 32. CLEAN BASELINES
# ------------------------------------------------

print(
    "\nRunning clean baselines..."
)

# Use event descriptors for classical models.
E = EVENT_NUMPY

E_mean = E[
    TRAIN_IDX
].mean(
    axis=0,
)

E_std = np.maximum(
    E[
        TRAIN_IDX
    ].std(
        axis=0,
    ),
    1e-6,
)

E_STD = (
    (
        E - E_mean
    )
    / E_std
).astype(
    np.float32
)


def evaluate_sklearn_baseline(
    name,
    model,
):

    model.fit(
        E_STD[
            TRAIN_IDX
        ],
        Y[
            TRAIN_IDX
        ],
    )

    # Raw probability / decision score
    def get_prob(idx):

        if hasattr(
            model,
            "predict_proba",
        ):

            return model.predict_proba(
                E_STD[idx]
            )[:, 1]

        score = model.decision_function(
            E_STD[idx]
        )

        return 1 / (
            1 + np.exp(-score)
        )

    p_platt_raw = get_prob(
        PLATT_IDX
    )

    # Use logit of baseline probabilities as calibration input.
    eps = 1e-6

    logit_platt = np.log(
        np.clip(
            p_platt_raw,
            eps,
            1 - eps,
        )
        / np.clip(
            1 - p_platt_raw,
            eps,
            1,
        )
    )

    platt = fit_platt_from_logits(
        logit_platt,
        Y[
            PLATT_IDX
        ],
    )

    def calibrated(idx):

        p = get_prob(
            idx
        )

        logit = np.log(
            np.clip(
                p,
                eps,
                1 - eps,
            )
            / np.clip(
                1 - p,
                eps,
                1,
            )
        )

        return apply_platt(
            platt,
            logit,
        )

    p_threshold = calibrated(
        THRESHOLD_IDX
    )

    threshold = best_f1_threshold(
        Y[
            THRESHOLD_IDX
        ],
        p_threshold,
    )

    p_test = calibrated(
        TEST_IDX
    )

    metrics = metric_dict(
        Y[
            TEST_IDX
        ],
        p_test,
        threshold,
    )

    metrics[
        "method"
    ] = name

    return metrics


baseline_rows = []

lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="lbfgs",
)

baseline_rows.append(
    evaluate_sklearn_baseline(
        "Logistic Regression",
        lr,
    )
)

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=271828,
    n_jobs=-1,
)

baseline_rows.append(
    evaluate_sklearn_baseline(
        "Random Forest",
        rf,
    )
)

# Use Raw-only seed 1 as matched neural raw baseline.
raw_seed_metric = (
    SEED_RESULTS[
        (
            SEED_RESULTS.variant
            == "raw_only"
        )
        & (
            SEED_RESULTS.seed
            == SEEDS[0]
        )
    ]
    .iloc[0]
    .to_dict()
)

baseline_rows.append({
    "method": "1D-CNN / Raw-only",
    **{
        k: raw_seed_metric[k]
        for k in [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "roc_auc",
            "average_precision",
            "brier",
            "ece",
            "threshold",
        ]
    },
})

exp1_baseline = {
    "method": "Trust-Sleep",
    **{
        k: EXP1_METRICS[k]
        for k in [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "roc_auc",
            "average_precision",
            "brier",
            "ece",
            "threshold",
        ]
    },
}

baseline_rows.append(
    exp1_baseline
)

BASELINES = pd.DataFrame(
    baseline_rows
)

BASELINES.to_csv(
    TABLE_DIR
    / "Experiment1_baseline_comparison.csv",
    index=False,
)


# ------------------------------------------------
# 33. FIGURES
# ------------------------------------------------

# Exp2 five-seed performance
for metric in [
    "f1",
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
]:

    plt.figure(
        figsize=(6, 4)
    )

    values = []

    errors = []

    labels = []

    for variant in [
        "full",
        "raw_only",
    ]:

        sub = SEED_RESULTS[
            SEED_RESULTS.variant
            == variant
        ]

        values.append(
            sub[
                metric
            ].mean()
        )

        errors.append(
            sub[
                metric
            ].std(
                ddof=1
            )
        )

        labels.append(
            variant
        )

    x = np.arange(
        len(labels)
    )

    plt.bar(
        x,
        values,
        yerr=errors,
        capsize=5,
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel(
        metric.replace(
            "_",
            " ",
        ).upper()
    )

    plt.title(
        f"Five-seed Full vs Raw-only: {metric}"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / f"Experiment2_{metric}.png",
        dpi=300,
    )

    plt.close()


# Experiment 1 trust by correctness
plt.figure(
    figsize=(7, 4)
)

names = [
    "CDT",
    "CERA",
    "CAS",
]

correct_means = [
    CDT[
        correct == 1
    ].mean(),
    CERA[
        correct == 1
    ].mean(),
    CAS[
        correct == 1
    ].mean(),
]

incorrect_means = [
    CDT[
        correct == 0
    ].mean(),
    CERA[
        correct == 0
    ].mean(),
    CAS[
        correct == 0
    ].mean(),
]

x = np.arange(3)

width = 0.35

plt.bar(
    x - width / 2,
    correct_means,
    width,
    label="Correct",
)

plt.bar(
    x + width / 2,
    incorrect_means,
    width,
    label="Incorrect",
)

plt.xticks(
    x,
    names,
)

plt.ylabel(
    "Mean trust score"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "Experiment1_trust_correctness.png",
    dpi=300,
)

plt.close()


# Selective prediction risk-coverage
plt.figure(
    figsize=(6, 4)
)

for score_name in [
    "CDT",
    "CAS",
]:

    s = SELECTIVE[
        SELECTIVE.score
        == score_name
    ].sort_values(
        "coverage"
    )

    plt.plot(
        s.coverage,
        s.accepted_error,
        label=score_name,
    )

plt.xlabel(
    "Coverage"
)

plt.ylabel(
    "Accepted-set error"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "Experiment1_risk_coverage.png",
    dpi=300,
)

plt.close()


# ------------------------------------------------
# 34. FINAL SUMMARY TABLE
# ------------------------------------------------

FINAL_SUMMARY = {
    "master_dataset":
        str(MASTER_H5),

    "output_directory":
        str(OUT),

    "split":
        TARGET_FRACTIONS,

    "recording_leakage_detected":
        False,

    "experiment1_seed":
        EXP1_SEED,

    "experiment1_metrics":
        {
            k:
            float(v)
            if isinstance(
                v,
                (
                    float,
                    np.floating,
                    int,
                    np.integer,
                ),
            )
            else v
            for k, v
            in EXP1_METRICS.items()
        },

    "experiment2_full_ensemble":
        {
            k:
            float(v)
            if isinstance(
                v,
                (
                    float,
                    np.floating,
                    int,
                    np.integer,
                ),
            )
            else v
            for k, v
            in FULL[
                "metrics"
            ].items()
        },

    "experiment2_raw_ensemble":
        {
            k:
            float(v)
            if isinstance(
                v,
                (
                    float,
                    np.floating,
                    int,
                    np.integer,
                ),
            )
            else v
            for k, v
            in RAW[
                "metrics"
            ].items()
        },
}

(
    OUT / "FINAL_SUMMARY.json"
).write_text(
    json.dumps(
        FINAL_SUMMARY,
        indent=2,
    )
)


# ------------------------------------------------
# 35. README FOR THIS CLEAN RUN
# ------------------------------------------------

readme = f"""
TRUST-SLEEP CLEAN PUBLICATION RERUN
===================================

Master dataset
--------------
{MASTER_H5}

The previously used experiment1a.h5 development file was not used.

Reason
------
experiment1a.h5 was verified to overlap with experiment1b.h5.
Therefore experiment1b.h5 was treated as the complete master dataset
and repartitioned from scratch at recording level.

Frozen recording-level split
----------------------------
60% train
10% early-stopping validation
5% Platt probability calibration
5% threshold selection
10% conformal calibration
10% final test

The final test recording set was excluded from:
- model fitting
- preprocessing estimation
- early stopping
- probability calibration
- threshold selection
- conformal calibration

Experiment 1
------------
Single Full Trust-Sleep model using seed 271828.
Includes predictive evaluation, CDT/CERA/CAS analysis,
selective prediction, and perturbation robustness.

Experiment 2
------------
Five-seed Full vs Raw-only comparison.
Seeds:
{SEEDS}

All seeds and both model configurations use exactly the same
recording-level data partition.

Important
---------
The epsilon-parameterized Laplace analysis is a perturbation-robustness
experiment only. It does not establish formal differential privacy.

Primary output tables
---------------------
tables/Experiment1_predictive_metrics.csv
tables/Experiment1_test_predictions_trust.csv
tables/Experiment1_trust_correlations.csv
tables/Experiment1_selective_prediction.csv
tables/Experiment1_privacy_motivated_perturbation.csv
tables/Experiment1_baseline_comparison.csv

tables/Experiment2_seed_level_metrics.csv
tables/Experiment2_five_seed_mean_sd.csv
tables/Experiment2_ensemble_metrics.csv
tables/Experiment2_paired_recording_bootstrap_CI.csv

Audit files
-----------
audit/split_summary.csv
audit/epoch_split_audit.csv
audit/clean_split_indices.npz
audit/preprocessing_parameters.npz

Figures
-------
figures/
"""

(
    OUT / "README_CLEAN_RERUN.txt"
).write_text(
    readme
)


# ------------------------------------------------
# 36. ZIP THE RESULTS FOR EASY DOWNLOAD/BACKUP
# ------------------------------------------------

import shutil

zip_path = shutil.make_archive(
    str(OUT),
    "zip",
    root_dir=str(
        OUT.parent
    ),
    base_dir=OUT.name,
)

print(
    "\n=========================================="
)

print(
    "CLEAN RERUN COMPLETE"
)

print(
    "=========================================="
)

print(
    "\nResults saved permanently in Drive:"
)

print(
    OUT
)

print(
    "\nZIP backup:"
)

print(
    zip_path
)

print(
    "\nSplit summary:"
)

print(
    SPLIT_SUMMARY
)

print(
    "\nExperiment 1:"
)

print(
    pd.DataFrame(
        [EXP1_METRICS]
    )
)

print(
    "\nExperiment 1 trust correlations:"
)

print(
    TRUST_STATS
)

print(
    "\nExperiment 2 five-seed mean ± SD:"
)

print(
    SEED_SUMMARY
)

print(
    "\nExperiment 2 ensemble metrics:"
)

print(
    ENSEMBLE_METRICS
)

print(
    "\nPaired Full - Raw bootstrap CIs:"
)

print(
    BOOTSTRAP_CI
)

print(
    "\nBaseline comparison:"
)

print(
    BASELINES
)
