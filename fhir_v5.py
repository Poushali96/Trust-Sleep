
from __future__ import annotations
from typing import Any, Dict, Mapping
from datetime import datetime, timezone


def prediction_to_fhir_bundle(
    prediction: Mapping[str, Any],
    patient_reference: str,
    encounter_reference: str | None = None,
) -> Dict[str, Any]:
    """
    Creates a FHIR R4 Bundle for export. It does not transmit PHI.

    The subtype is explicitly represented as a preliminary interpretation,
    not a confirmed diagnosis.
    """
    issued = datetime.now(timezone.utc).isoformat()
    case_id = str(prediction["case_id"])
    binary_probability = float(prediction["binary_probability"])
    subtype = str(prediction["subtype_hypothesis"])
    subtype_probability = float(prediction["subtype_probability"])

    observation_id = f"trust-sleep-apnea-{case_id}"
    report_id = f"trust-sleep-report-{case_id}"

    observation = {
        "resourceType": "Observation",
        "id": observation_id,
        "status": "preliminary",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "survey",
                "display": "Survey",
            }]
        }],
        "code": {
            "text": "AI-assisted sleep apnea event probability"
        },
        "subject": {"reference": patient_reference},
        "issued": issued,
        "valueQuantity": {
            "value": binary_probability,
            "unit": "probability",
            "system": "http://unitsofmeasure.org",
            "code": "1",
        },
        "component": [
            {
                "code": {"text": "Transferred subtype hypothesis"},
                "valueCodeableConcept": {"text": subtype},
            },
            {
                "code": {"text": "Transferred subtype probability"},
                "valueQuantity": {
                    "value": subtype_probability,
                    "unit": "probability",
                    "system": "http://unitsofmeasure.org",
                    "code": "1",
                },
            },
            {
                "code": {"text": "Clinical decision trust index"},
                "valueQuantity": {
                    "value": float(prediction["cdt"]),
                    "unit": "index",
                },
            },
            {
                "code": {"text": "Explanation reliability index"},
                "valueQuantity": {
                    "value": float(prediction["cera"]),
                    "unit": "index",
                },
            },
            {
                "code": {"text": "Combined assurance index"},
                "valueQuantity": {
                    "value": float(prediction["cas"]),
                    "unit": "index",
                },
            },
        ],
        "note": [{
            "text": (
                "Research decision support only. Subtype was transferred from "
                "external PSG labels and requires clinician confirmation."
            )
        }],
    }
    if encounter_reference:
        observation["encounter"] = {"reference": encounter_reference}

    report = {
        "resourceType": "DiagnosticReport",
        "id": report_id,
        "status": "preliminary",
        "code": {"text": "Trust-Sleep respiratory event review"},
        "subject": {"reference": patient_reference},
        "issued": issued,
        "result": [{"reference": f"Observation/{observation_id}"}],
        "conclusion": prediction["deterministic_explanation"]["explanation"],
        "conclusionCode": [{
            "text": (
                f"Suspected {subtype} pattern; clinician confirmation required"
            )
        }],
    }
    if encounter_reference:
        report["encounter"] = {"reference": encounter_reference}

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": observation},
            {"resource": report},
        ],
    }
