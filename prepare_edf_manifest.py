
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd


def resample_linear(signal: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    if source_hz <= 0 or target_hz <= 0:
        raise ValueError("Sampling rates must be positive.")
    duration = len(signal) / source_hz
    target_n = max(1, int(round(duration * target_hz)))
    old_t = np.linspace(0, duration, len(signal), endpoint=False)
    new_t = np.linspace(0, duration, target_n, endpoint=False)
    return np.interp(new_t, old_t, signal).astype(np.float32)


def prepare_from_edf_and_annotations(
    edf_path: str,
    annotation_csv: str,
    output_dir: str,
    channel_map: Dict[str, str],
    target_hz: float = 10.0,
    context_seconds: float = 10.0,
) -> pd.DataFrame:
    """
    Generic raw PSG adapter.

    annotation_csv columns:
      subject_id, recording_id, start_seconds, duration_seconds,
      event_label, subtype_label, split

    EDF channel_map keys:
      chest, flow, pulse, spo2
    """
    try:
        import pyedflib
    except ImportError as exc:
        raise RuntimeError("Install pyedflib to prepare EDF recordings.") from exc

    annotations = pd.read_csv(annotation_csv)
    required = {
        "subject_id","recording_id","start_seconds","duration_seconds",
        "event_label","subtype_label","split",
    }
    missing = required - set(annotations)
    if missing:
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reader = pyedflib.EdfReader(edf_path)
    labels = list(reader.getSignalLabels())
    label_to_index = {label: i for i, label in enumerate(labels)}

    signals, rates, present = {}, {}, {}
    for canonical in ("chest","flow","pulse","spo2"):
        source_name = channel_map.get(canonical)
        if source_name and source_name in label_to_index:
            index = label_to_index[source_name]
            signals[canonical] = reader.readSignal(index).astype(np.float32)
            rates[canonical] = float(reader.getSampleFrequency(index))
            present[canonical] = 1.0
        else:
            signals[canonical] = None
            rates[canonical] = target_hz
            present[canonical] = 0.0
    reader.close()

    rows = []
    for row_index, row in annotations.iterrows():
        start = max(0.0, float(row["start_seconds"]) - context_seconds)
        end = (
            float(row["start_seconds"])
            + float(row["duration_seconds"])
            + context_seconds
        )
        channel_segments = []
        reference_n = None
        for canonical in ("chest","flow","pulse","spo2"):
            if signals[canonical] is None:
                channel_segments.append(None)
                continue
            rate = rates[canonical]
            i0, i1 = int(round(start*rate)), int(round(end*rate))
            segment = signals[canonical][i0:i1]
            segment = resample_linear(segment, rate, target_hz)
            reference_n = len(segment) if reference_n is None else min(reference_n, len(segment))
            channel_segments.append(segment)

        if reference_n is None or reference_n < 8:
            continue
        aligned = []
        for segment in channel_segments:
            aligned.append(
                np.zeros(reference_n, np.float32)
                if segment is None else segment[:reference_n]
            )
        signal = np.column_stack(aligned).astype(np.float32)

        localization = np.zeros(reference_n, np.float32)
        event_start = int(round(context_seconds * target_hz))
        event_end = min(
            reference_n,
            event_start + int(round(float(row["duration_seconds"]) * target_hz)),
        )
        localization[event_start:event_end] = 1.0

        sample_id = f"{row['recording_id']}_{row_index:06d}"
        path = output / f"{sample_id}.npz"
        np.savez_compressed(
            path,
            signal=signal,
            modality_present=np.asarray(
                [present[name] for name in ("chest","flow","pulse","spo2")],
                np.float32,
            ),
            localization_target=localization,
        )
        rows.append({
            "sample_id": sample_id,
            "npz_path": str(path),
            "binary_label": int(str(row["event_label"]) != "no_event"),
            "event_label": row["event_label"],
            "subtype_label": row["subtype_label"],
            "subject_id": row["subject_id"],
            "recording_id": row["recording_id"],
            "split": row["split"],
            "source": Path(edf_path).stem,
            "sampling_rate_hz": target_hz,
            "window_seconds": reference_n / target_hz,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--channel-map-json", required=True)
    parser.add_argument("--target-hz", type=float, default=10.0)
    args = parser.parse_args()

    channel_map = json.loads(Path(args.channel_map_json).read_text())
    frame = prepare_from_edf_and_annotations(
        args.edf, args.annotations, args.output_dir,
        channel_map, args.target_hz,
    )
    frame.to_csv(args.manifest_out, index=False)


if __name__ == "__main__":
    import json
    main()
