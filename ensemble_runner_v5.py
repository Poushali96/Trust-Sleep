
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List

DEFAULT_SEEDS = (271828, 314159, 161803, 141421, 173205)


def run(command: List[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-h5", required=True)
    parser.add_argument("--external-manifest", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--split-seed", type=int, default=271828)
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = []

    for seed in args.seeds:
        seed_output = output / f"seed_{seed}"
        command = [
            "python",
            "train_multisource_v5.py",
            "--local-h5",
            args.local_h5,
            "--external-manifest",
            args.external_manifest,
            "--output",
            str(seed_output),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(seed),
            "--split-seed",
            str(args.split_seed),
        ]
        if args.metadata:
            command.extend(["--metadata", args.metadata])
        run(command)
        checkpoints.append(
            str(seed_output / "hierarchical_trust_sleep_v4.pt")
        )

    manifest = {
        "package_version": "5.0.0",
        "model_seeds": args.seeds,
        "split_seed": args.split_seed,
        "checkpoints": checkpoints,
        "external_manifest": args.external_manifest,
        "research_only": True,
    }
    (output / "ensemble_manifest_v5.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print("Saved:", output / "ensemble_manifest_v5.json")


if __name__ == "__main__":
    main()
