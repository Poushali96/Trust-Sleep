
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def priority(prediction: Dict[str, Any]) -> float:
    """
    Prioritize cases that are uncertain, out of distribution, physiologically
    inconsistent, or likely to improve rare-subtype supervision.
    """
    uncertainty = min(
        1.0,
        float(prediction.get("ensemble_variance", 0.0)) / 0.02,
    )
    conformal_ambiguity = min(
        1.0,
        max(0, len(prediction.get("subtype_conformal_set", [])) - 1) / 3,
    )
    low_cas = 1.0 - float(prediction.get("cas", 0.0))
    ood = float(
        prediction.get("domain_status") == "out_of_distribution"
    )
    inconsistent = float(
        prediction.get("physiological_audit", {}).get("status")
        != "consistent"
    )
    rare_subtype = float(
        prediction.get("subtype_hypothesis") in {"central", "mixed"}
    )
    return (
        0.25 * uncertainty
        + 0.20 * conformal_ambiguity
        + 0.20 * low_cas
        + 0.15 * ood
        + 0.10 * inconsistent
        + 0.10 * rare_subtype
    )


def export_sqlite_queue(
    database_path: str,
    output_csv: str,
    limit: int,
) -> None:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT case_id, session_id, timestamp, model_version,
                   prediction_json
            FROM audit_events
            ORDER BY timestamp DESC
            """
        ).fetchall()

    queue: List[Dict[str, Any]] = []
    seen = set()
    for case_id, session_id, timestamp, model_version, payload in rows:
        if case_id in seen:
            continue
        seen.add(case_id)
        prediction = json.loads(payload)
        queue.append({
            "case_id": case_id,
            "session_id": session_id,
            "timestamp": timestamp,
            "model_version": model_version,
            "priority": priority(prediction),
            "subtype_hypothesis": prediction.get("subtype_hypothesis"),
            "subtype_probability": prediction.get("subtype_probability"),
            "cas": prediction.get("cas"),
            "domain_status": prediction.get("domain_status"),
            "decision_status": prediction.get("decision_status"),
        })

    frame = pd.DataFrame(queue).sort_values(
        "priority",
        ascending=False,
    ).head(limit)
    frame.to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite-db", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    export_sqlite_queue(args.sqlite_db, args.output, args.limit)


if __name__ == "__main__":
    main()
