
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source_bundle = Path(args.ensemble_bundle).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    bundle = torch.load(source_bundle, map_location="cpu")
    relative_paths = []
    for index, checkpoint in enumerate(bundle["checkpoint_paths"]):
        checkpoint = Path(checkpoint).resolve()
        destination = checkpoints_dir / f"member_{index:02d}.pt"
        shutil.copy2(checkpoint, destination)
        relative_paths.append(str(destination.relative_to(output)))

    bundle["checkpoint_paths"] = relative_paths
    destination_bundle = output / "ensemble_bundle_v5.pt"
    torch.save(bundle, destination_bundle)

    manifest = {
        "bundle": destination_bundle.name,
        "checkpoints": relative_paths,
        "bundle_version": bundle.get("bundle_version", "5.0.0"),
        "member_count": len(relative_paths),
        "research_only": bundle.get("research_only", True),
    }
    (output / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print("Created self-contained model package:", output)


if __name__ == "__main__":
    main()
