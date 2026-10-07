"""Create a tiny synthetic HDF5 file matching the Trust-Sleep input schema.

This file is only for checking that the pipeline can execute. It cannot reproduce
the scientific results and must not be used for scientific conclusions.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

COLS = [
    "psg_chest_rn_mean",
    "psg_flow_dr_rn_mean",
    "psg_pulse_rn_mean",
    "psg_spo2_rn_mean",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/synthetic_demo.h5")
    ap.add_argument("--recordings", type=int, default=60)
    ap.add_argument("--epochs-per-recording", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if out.exists(): out.unlink()
    with pd.HDFStore(out, mode="w") as store:
        for r in range(args.recordings):
            for e in range(args.epochs_per_recording):
                label = int((e + r) % 5 == 0)
                x = rng.normal(0, 1, size=(30, 4)).astype("float32")
                if label:
                    x[:, 1] -= 0.6
                    x[:, 3] -= 0.4
                frame = pd.DataFrame(x, columns=COLS)
                key = f"/r{r:04d}/e{e:06d}_{label}"
                store.put(key, frame, format="fixed")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
