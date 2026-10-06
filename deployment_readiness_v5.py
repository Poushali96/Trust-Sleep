
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import torch


REQUIRED_BUNDLE_KEYS = {
    "bundle_version",
    "checkpoint_paths",
    "member_count",
    "signal_standardizer",
    "metadata_standardizer",
    "operating_threshold",
    "binary_platt",
    "binary_conformal",
    "subtype_temperature",
    "subtype_conformal",
    "ood_reference",
    "trust_policy",
}


def check_bundle(path: str) -> Dict[str, Any]:
    bundle_path = Path(path).resolve()
    bundle = torch.load(bundle_path, map_location="cpu")
    missing = sorted(REQUIRED_BUNDLE_KEYS - set(bundle))
    checkpoint_results = []
    for checkpoint in bundle.get("checkpoint_paths", []):
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = bundle_path.parent / checkpoint_path
        checkpoint_results.append({
            "path": str(checkpoint_path),
            "exists": checkpoint_path.exists(),
        })

    policy = bundle.get("trust_policy", {})
    policy_complete = all(
        key in policy
        for key in (
            "cdt_threshold",
            "cera_threshold",
            "cas_threshold",
            "subtype_probability_threshold",
        )
    )
    ready = (
        not missing
        and checkpoint_results
        and all(item["exists"] for item in checkpoint_results)
        and policy_complete
    )
    return {
        "ready": ready,
        "bundle": str(bundle_path),
        "missing_keys": missing,
        "checkpoint_results": checkpoint_results,
        "policy_complete": policy_complete,
        "research_only": bundle.get("research_only", True),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = check_bundle(args.bundle)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text)
    if not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
