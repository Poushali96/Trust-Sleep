
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional
import json
import urllib.request


SYSTEM_RULES = """
You are a clinical-language formatter for a research sleep-apnea decision-support system.
Use only facts in the supplied JSON. Do not add diagnoses, measurements, treatments,
causes, probabilities, patient history, or recommendations that are not explicitly present.
Use uncertainty language: suspected, compatible with, may represent.
Always state that transferred subtype predictions require clinician confirmation.
Never convert CDT, CERA, or CAS into probabilities.
Return JSON with keys: headline, explanation, trust_summary, action, limitation.
"""


@dataclass
class LLMExplanation:
    headline: str
    explanation: str
    trust_summary: str
    action: str
    limitation: str
    provider: str
    validated: bool


def deterministic_explanation(payload: Mapping[str, Any]) -> LLMExplanation:
    subtype = payload["subtype_hypothesis"]
    probability = payload["subtype_probability"]
    evidence = payload["audit"]["evidence"]
    trust = payload["trust"]
    decision = payload["decision"]

    explanation = " ".join(evidence[:4]) or (
        "The available physiological evidence was insufficient for a detailed mechanism explanation."
    )
    return LLMExplanation(
        headline=f"Apnea event: suspected {subtype} pattern",
        explanation=(
            f"The transferred subtype model assigned {subtype} as the leading hypothesis "
            f"({probability:.0%}). {explanation}"
        ),
        trust_summary=(
            f"CDT {trust['cdt']:.2f} ({trust['cdt_level']}), "
            f"CERA {trust['cera']:.2f} ({trust['cera_level']}), "
            f"CAS {trust['cas']:.2f} ({trust['cas_level']})."
        ),
        action=decision["status"].replace("_", " ").capitalize(),
        limitation=(
            "The subtype was learned from external PSG labels and requires clinician confirmation."
        ),
        provider="deterministic_template",
        validated=True,
    )


def _validate_generated(result: Dict[str, Any], payload: Mapping[str, Any]) -> bool:
    required = {"headline", "explanation", "trust_summary", "action", "limitation"}
    if set(result) != required:
        return False
    if not all(isinstance(result[key], str) and result[key].strip() for key in required):
        return False

    combined = " ".join(result[key] for key in required).lower()
    forbidden = (
        "confirmed diagnosis",
        "definitive diagnosis",
        "start treatment",
        "prescribe",
        "no clinician review",
        "autonomous diagnosis",
    )
    if any(term in combined for term in forbidden):
        return False

    subtype = str(payload["subtype_hypothesis"]).lower()
    if subtype not in combined or "clinician" not in combined:
        return False

    # Do not permit the formatter to introduce percentages not present
    # in the structured payload.
    import re
    generated_percentages = set(re.findall(r"\b\d{1,3}%", combined))
    allowed_percentages = {
        f"{float(payload['subtype_probability']):.0%}"
    }
    return generated_percentages.issubset(allowed_percentages)


def explain_with_endpoint(
    payload: Mapping[str, Any],
    endpoint: Optional[str] = None,
    timeout: float = 20.0,
) -> LLMExplanation:
    """
    Optional local/private LLM endpoint.

    Expected endpoint contract:
      POST JSON {"system": ..., "payload": ...}
      response JSON {"headline", "explanation", "trust_summary", "action", "limitation"}

    Falls back to the deterministic renderer on any error or validation failure.
    """
    fallback = deterministic_explanation(payload)
    if not endpoint:
        return fallback

    request = urllib.request.Request(
        endpoint,
        data=json.dumps({"system": SYSTEM_RULES, "payload": payload}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not _validate_generated(result, payload):
            return fallback
        return LLMExplanation(
            **result,
            provider="private_endpoint",
            validated=True,
        )
    except Exception:
        return fallback
