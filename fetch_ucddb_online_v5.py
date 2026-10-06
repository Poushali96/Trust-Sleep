
from __future__ import annotations
import argparse, json, re, subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

BASE_URL = "https://physionet.org/files/ucddb/1.0.0/"
EVENT_RE = re.compile(
    r"^\s*(\d{2}:\d{2}:\d{2})\s+(APNEA|HYP)-([OCM])\s+(?:\S+\s+)?(\d+(?:\.\d+)?)"
)
SUBTYPE = {"O": "obstructive", "C": "central", "M": "mixed"}

@dataclass
class Event:
    clock: str
    event_label: str
    subtype_label: str
    duration: float

def run(command: Sequence[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)

def download(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination / "ucddb-1.0.0"
    if not (root / "RECORDS").exists():
        run([
            "wget", "-r", "-N", "-c", "-np",
            "--no-host-directories", "--cut-dirs=3",
            "-P", str(root), BASE_URL,
        ])
    if (root / "RECORDS").exists():
        return root
    matches = list(root.rglob("RECORDS"))
    if not matches:
        raise RuntimeError("UCDDB download completed but RECORDS was not found.")
    return matches[0].parent

def parse_events(path: Path) -> List[Event]:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        match = EVENT_RE.match(line)
        if not match:
            continue
        clock, kind, subtype, duration = match.groups()
        rows.append(Event(
            clock=clock,
            event_label="apnea" if kind == "APNEA" else "hypopnea",
            subtype_label=SUBTYPE[subtype],
            duration=float(duration),
        ))
    return rows

def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())

def channel_index(labels: Sequence[str], aliases: Sequence[str]) -> Optional[int]:
    normalized = [norm(label) for label in labels]
    for alias in aliases:
        target = norm(alias)
        for i, label in enumerate(normalized):
            if target in label:
                return i
    return None

def seconds(clock: str) -> int:
    h, m, s = map(int, clock.split(":"))
    return h * 3600 + m * 60 + s

def offset(clock: str, start: datetime) -> float:
    value = seconds(clock)
    start_value = start.hour * 3600 + start.minute * 60 + start.second
    delta = value - start_value
    if delta < -43200:
        delta += 86400
    elif delta > 43200:
        delta -= 86400
    return float(delta)

def resample(values: np.ndarray, source_hz: float, target_hz: float, duration: float) -> np.ndarray:
    n = max(8, int(round(duration * target_hz)))
    if len(values) < 2:
        return np.zeros(n, np.float32)
    old_t = np.arange(len(values)) / source_hz
    new_t = np.arange(n) / target_hz
    return np.interp(new_t, old_t, values, left=values[0], right=values[-1]).astype(np.float32)

def split_subjects(subjects: Sequence[str], seed: int) -> Dict[str, str]:
    subjects = list(np.random.default_rng(seed).permutation(sorted(subjects)))
    n = len(subjects)
    n_test = max(2, round(.12 * n))
    n_conf = max(2, round(.12 * n))
    n_val = max(2, round(.12 * n))
    n_train = n - n_test - n_conf - n_val
    groups = {
        "train": subjects[:n_train],
        "validation": subjects[n_train:n_train+n_val],
        "conformal": subjects[n_train+n_val:n_train+n_val+n_conf],
        "test": subjects[n_train+n_val+n_conf:],
    }
    return {subject: split for split, members in groups.items() for subject in members}

def subject_metadata(dataset_dir: Path) -> Dict[str, Dict[str, object]]:
    path = dataset_dir / "SubjectDetails.xls"
    if not path.exists():
        return {}
    try:
        frame = pd.read_excel(path)
    except Exception:
        return {}
    result = {}
    for _, row in frame.iterrows():
        values = row.to_dict()
        subject = None
        for value in values.values():
            match = re.search(r"(?:ucddb)?0*(\d{1,3})$", str(value).strip().lower())
            if match:
                subject = f"ucddb{int(match.group(1)):03d}"
                break
        if subject is None:
            continue
        lower = {str(k).lower(): v for k, v in values.items()}
        def find(*terms):
            for key, value in lower.items():
                if any(term in key for term in terms):
                    return value
            return np.nan
        result[subject] = {
            "age": find("age"),
            "sex": find("sex", "gender"),
            "bmi": find("bmi"),
            "ahi": find("ahi"),
        }
    return result

def prepare_subject(
    record_path: Path,
    annotation_path: Path,
    subject_id: str,
    split: str,
    metadata: Dict[str, object],
    output_dir: Path,
    target_hz: float,
    context_seconds: float,
    non_event_ratio: float,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    import pyedflib
    reader = pyedflib.EdfReader(str(record_path))
    labels = list(reader.getSignalLabels())
    start_dt = reader.getStartdatetime()
    total_duration = float(reader.getFileDuration())
    aliases = {
        "chest": ("ribcage", "thorax", "thoracic", "chest"),
        "flow": ("airflow", "oro-nasal", "oronasal", "nasal"),
        "pulse": ("pulse", "heart rate", "heartrate"),
        "spo2": ("spo2", "sao2", "oxygen saturation", "oximetry"),
    }
    indices = {name: channel_index(labels, choices) for name, choices in aliases.items()}
    rates = {
        name: float(reader.getSampleFrequency(index)) if index is not None else target_hz
        for name, index in indices.items()
    }

    def read_window(start_s: float, end_s: float):
        duration = end_s - start_s
        channels, present = [], []
        for name in ("chest", "flow", "pulse", "spo2"):
            index = indices[name]
            if index is None:
                channels.append(np.zeros(int(round(duration * target_hz)), np.float32))
                present.append(0.0)
                continue
            hz = rates[name]
            start_sample = max(0, int(round(start_s * hz)))
            count = min(
                int(round(duration * hz)),
                max(reader.getNSamples()[index] - start_sample, 0),
            )
            values = reader.readSignal(index, start_sample, count) if count > 0 else np.zeros(1)
            channels.append(resample(np.asarray(values, np.float32), hz, target_hz, duration))
            present.append(1.0)
        n = min(map(len, channels))
        return np.column_stack([channel[:n] for channel in channels]), np.asarray(present, np.float32)

    rows, occupied = [], []
    for event_index, event in enumerate(parse_events(annotation_path)):
        event_start = offset(event.clock, start_dt)
        event_end = event_start + event.duration
        window_start = max(0.0, event_start - context_seconds)
        window_end = min(total_duration, event_end + context_seconds)
        if window_end - window_start < 8:
            continue
        signal, present = read_window(window_start, window_end)
        localization = np.zeros(len(signal), np.float32)
        a = int(round((event_start-window_start) * target_hz))
        b = int(round((event_end-window_start) * target_hz))
        localization[max(a, 0):min(b, len(signal))] = 1.0
        sample_id = f"{subject_id}_event_{event_index:05d}"
        npz = output_dir / f"{sample_id}.npz"
        np.savez_compressed(npz, signal=signal.astype(np.float32), modality_present=present, localization_target=localization)
        rows.append({
            "sample_id": sample_id,
            "npz_path": str(npz.resolve()),
            "binary_label": 1,
            "event_label": event.event_label,
            "subtype_label": event.subtype_label,
            "subject_id": subject_id,
            "recording_id": subject_id,
            "split": split,
            "source": "ucddb",
            "sampling_rate_hz": target_hz,
            "window_seconds": len(signal) / target_hz,
            **metadata,
        })
        occupied.append((window_start, window_end))

    target_background = int(round(len(rows) * non_event_ratio))
    created = attempts = 0
    while created < target_background and attempts < max(100, target_background * 20):
        attempts += 1
        if total_duration <= 30:
            break
        start_s = float(rng.uniform(0, total_duration - 30))
        end_s = start_s + 30
        if any(not (end_s <= lo or start_s >= hi) for lo, hi in occupied):
            continue
        signal, present = read_window(start_s, end_s)
        sample_id = f"{subject_id}_background_{created:05d}"
        npz = output_dir / f"{sample_id}.npz"
        np.savez_compressed(
            npz,
            signal=signal.astype(np.float32),
            modality_present=present,
            localization_target=np.zeros(len(signal), np.float32),
        )
        rows.append({
            "sample_id": sample_id,
            "npz_path": str(npz.resolve()),
            "binary_label": 0,
            "event_label": "no_event",
            "subtype_label": "indeterminate",
            "subject_id": subject_id,
            "recording_id": subject_id,
            "split": split,
            "source": "ucddb",
            "sampling_rate_hz": target_hz,
            "window_seconds": len(signal) / target_hz,
            **metadata,
        })
        created += 1

    reader.close()
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", default="/content/external_data")
    parser.add_argument("--manifest-out", default="/content/external_manifest_v5.csv")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--context-seconds", type=float, default=10.0)
    parser.add_argument("--non-event-ratio", type=float, default=.50)
    args = parser.parse_args()

    destination = Path(args.destination)
    dataset_dir = download(destination)
    record_names = [
        line.strip()
        for line in (dataset_dir / "RECORDS").read_text().splitlines()
        if line.strip().endswith(".rec")
    ]
    subjects = [Path(name).stem for name in record_names]
    splits = split_subjects(subjects, args.seed)
    metadata = subject_metadata(dataset_dir)
    prepared = destination / "ucddb_prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    for record_name, subject in zip(record_names, subjects):
        annotation = dataset_dir / f"{subject}_respevt.txt"
        record = dataset_dir / record_name
        if not record.exists() or not annotation.exists():
            print("Skipping incomplete subject:", subject)
            continue
        print("Preparing", subject, splits[subject])
        rows.extend(prepare_subject(
            record, annotation, subject, splits[subject],
            metadata.get(subject, {}), prepared,
            args.target_hz, args.context_seconds,
            args.non_event_ratio, rng,
        ))

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No external samples were created.")
    manifest_path = Path(args.manifest_out)
    manifest.to_csv(manifest_path, index=False)

    metadata_columns = [c for c in ("subject_id", "age", "sex", "bmi", "ahi") if c in manifest]
    metadata_path = manifest_path.with_name(manifest_path.stem + "_subject_metadata.csv")
    manifest[metadata_columns].drop_duplicates("subject_id").to_csv(metadata_path, index=False)

    summary = {
        "manifest": str(manifest_path),
        "subject_metadata": str(metadata_path),
        "subjects": int(manifest.subject_id.nunique()),
        "samples": int(len(manifest)),
        "split_counts": manifest.groupby("split").size().to_dict(),
        "subtype_counts": manifest.groupby("subtype_label").size().to_dict(),
        "limitations": [
            "UCDDB has only 25 adult subjects.",
            "It is suitable as a supplemental external source, not the sole final benchmark.",
            "Public external metadata do not replace metadata for local experiment subjects.",
        ],
    }
    manifest_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
