
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

SIGNALS = ("chest", "flow", "pulse", "spo2")

st.set_page_config(
    page_title="Trust-Sleep Clinical Workstation",
    page_icon="🫁",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top:.7rem;max-width:1800px;}
[data-testid="stMetric"] {
  background:#fff;border:1px solid #e7e9ee;padding:12px;border-radius:14px;
}
.review {
  background:#fff4f2;border:1px solid #ffccc7;border-left:6px solid #d92d20;
  padding:14px;border-radius:12px;
}
.assisted {
  background:#fffbe6;border:1px solid #ffe58f;border-left:6px solid #d48806;
  padding:14px;border-radius:12px;
}
.ok {
  background:#f6ffed;border:1px solid #b7eb8f;border-left:6px solid #389e0d;
  padding:14px;border-radius:12px;
}
.small {font-size:.88rem;color:#667085;}
</style>
""", unsafe_allow_html=True)


def synthetic_source(length: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    t = np.arange(length)
    chest = np.sin(2*np.pi*t/5) + rng.normal(0, .05, length)
    flow = np.sin(2*np.pi*t/5+.1) + rng.normal(0, .05, length)
    pulse = 72 + 2*np.sin(2*np.pi*t/20) + rng.normal(0, .2, length)
    spo2 = 96 + rng.normal(0, .08, length)

    for start in (120, 400, 700):
        stop = min(start+40, length)
        flow[start:stop] *= .08
        chest[start:stop] *= .90
        ds, de = min(start+20, length), min(start+65, length)
        if de > ds:
            spo2[ds:de] -= np.linspace(0, 4.5, de-ds)
        ps, pe = min(start+40, length), min(start+75, length)
        if pe > ps:
            pulse[ps:pe] += np.linspace(0, 8, pe-ps)

    return pd.DataFrame({
        "timestamp": t,
        "chest": chest,
        "flow": flow,
        "pulse": pulse,
        "spo2": spo2,
    })


def waveform_figure(frame: pd.DataFrame, prediction: Dict[str, Any]) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=.03,
        subplot_titles=("Thoracic effort", "Airflow", "Pulse", "SpO₂"),
    )
    for row, channel in enumerate(SIGNALS, start=1):
        fig.add_trace(
            go.Scatter(
                x=frame["timestamp"],
                y=frame[channel],
                mode="lines",
                name=channel,
                hovertemplate="time=%{x}<br>value=%{y:.3f}<extra></extra>",
            ),
            row=row, col=1,
        )

    start = prediction.get("event_start_index")
    end = prediction.get("event_end_index")
    if start is not None and end is not None and len(frame):
        x0 = frame["timestamp"].iloc[min(int(start), len(frame)-1)]
        x1 = frame["timestamp"].iloc[min(int(end), len(frame)-1)]
        for row in range(1, 5):
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor="rgba(245,158,11,.13)",
                line_width=0,
                row=row, col=1,
            )

    fig.update_layout(
        height=760,
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=30, r=20, t=50, b=30),
        uirevision="trust-sleep-v5",
    )
    return fig


def trend_figure(history: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for label, column in (
        ("Apnea probability", "binary_probability"),
        ("Subtype confidence", "subtype_probability"),
        ("CDT", "cdt"),
        ("CERA", "cera"),
        ("CAS", "cas"),
    ):
        fig.add_trace(
            go.Scatter(
                x=history["time"], y=history[column],
                mode="lines", name=label,
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=history["time"],
            y=history["ensemble_variance"],
            mode="lines",
            name="Ensemble variance",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(range=[0, 1], title_text="Probability / trust", secondary_y=False)
    fig.update_yaxes(title_text="Variance", secondary_y=True)
    fig.update_layout(height=340, hovermode="x unified")
    return fig


def request_json(
    method: str,
    url: str,
    api_key: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    headers = {"x-api-key": api_key} if api_key else {}
    response = requests.request(
        method,
        url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Backend {response.status_code}: {response.text}"
        )
    return response.json()


def initialize() -> None:
    defaults = {
        "source": synthetic_source(),
        "cursor": 0,
        "running": False,
        "case_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "history": [],
        "prediction": None,
        "llm_explanation": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize()

st.title("Trust-Sleep Clinical Workstation")
st.caption(
    "Backend-connected real-time respiratory monitoring with ensemble-calibrated "
    "predictions, conformal uncertainty, event localization, physiological auditing, "
    "trust indices, clinician feedback, and optional LLM wording."
)

with st.sidebar:
    st.header("Backend")
    backend_url = st.text_input(
        "Backend URL",
        os.getenv("TRUST_SLEEP_BACKEND_URL", "http://backend:8000"),
    ).rstrip("/")
    api_key = st.text_input(
        "API key",
        os.getenv("TRUST_SLEEP_API_KEY", ""),
        type="password",
    )

    if st.button("Check backend", use_container_width=True):
        try:
            ready = request_json(
                "GET",
                f"{backend_url}/health/ready",
                api_key,
            )
            st.success(
                f"Ready • bundle {ready['bundle_version']} • "
                f"{ready['ensemble_members']} models"
            )
        except Exception as exc:
            st.error(str(exc))

    st.header("Monitoring source")
    source_mode = st.radio(
        "Source",
        ("Synthetic replay", "Uploaded CSV replay", "Live appended CSV"),
    )
    uploaded = st.file_uploader(
        "CSV: timestamp(optional), chest, flow, pulse, spo2",
        type=["csv"],
    )
    live_path = st.text_input("Live CSV path", "/data/live_signals.csv")
    window_points = st.slider("Rolling window", 30, 600, 180, 10)
    chunk_points = st.slider("Samples per update", 1, 30, 5)
    refresh_seconds = st.slider("Refresh seconds", .2, 5.0, 1.0, .2)

    st.header("Patient context")
    age = st.number_input("Age", 0, 110, 50)
    bmi = st.number_input("BMI", 10.0, 80.0, 27.0)
    heart_failure = st.checkbox("Heart failure")
    stroke_history = st.checkbox("Stroke history")
    opioid_use = st.checkbox("Opioid use")
    altitude_m = st.number_input("Altitude (m)", 0, 6000, 0)
    tonsillar = st.checkbox("Tonsillar hypertrophy")

    st.header("Available signals")
    chest_present = st.checkbox("Chest effort", True)
    flow_present = st.checkbox("Airflow", True)
    pulse_present = st.checkbox("Pulse", True)
    spo2_present = st.checkbox("SpO₂", True)

    start_col, stop_col = st.columns(2)
    if start_col.button("▶ Start", type="primary", use_container_width=True):
        st.session_state.running = True
    if stop_col.button("■ Stop", use_container_width=True):
        st.session_state.running = False

    if st.button("New case", use_container_width=True):
        st.session_state.case_id = str(uuid.uuid4())
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.cursor = 0
        st.session_state.history = []
        st.session_state.prediction = None
        st.session_state.llm_explanation = None
        st.session_state.running = False
        st.rerun()


if source_mode == "Uploaded CSV replay" and uploaded is not None:
    source = pd.read_csv(uploaded)
    if "timestamp" not in source:
        source["timestamp"] = np.arange(len(source))
    if not set(SIGNALS).issubset(source.columns):
        st.error(f"CSV must contain {list(SIGNALS)}")
        st.stop()
    st.session_state.source = source[["timestamp", *SIGNALS]]

elif source_mode == "Live appended CSV":
    path = Path(live_path)
    if path.exists():
        source = pd.read_csv(path)
        if "timestamp" not in source:
            source["timestamp"] = np.arange(len(source))
        if not set(SIGNALS).issubset(source.columns):
            st.error(f"Live CSV must contain {list(SIGNALS)}")
            st.stop()
        st.session_state.source = source[["timestamp", *SIGNALS]]
        st.session_state.cursor = len(source)
    else:
        st.warning("Waiting for live CSV file.")

elif source_mode == "Synthetic replay" and st.session_state.cursor == 0:
    st.session_state.source = synthetic_source()


if st.session_state.running and source_mode != "Live appended CSV":
    st.session_state.cursor = min(
        len(st.session_state.source),
        st.session_state.cursor + chunk_points,
    )

end = max(
    st.session_state.cursor,
    min(window_points, len(st.session_state.source)),
)
start = max(0, end-window_points)
window = st.session_state.source.iloc[start:end].copy()

metadata = {
    "age": age,
    "bmi": bmi,
    "heart_failure": heart_failure,
    "stroke_history": stroke_history,
    "opioid_use": opioid_use,
    "altitude_m": altitude_m,
    "tonsillar_hypertrophy": tonsillar,
}
modality_present = [
    float(chest_present),
    float(flow_present),
    float(pulse_present),
    float(spo2_present),
]

if len(window) >= 8:
    payload = {
        "case_id": st.session_state.case_id,
        "session_id": st.session_state.session_id,
        "signal": window[list(SIGNALS)].to_numpy(float).tolist(),
        "metadata": metadata,
        "modality_present": modality_present,
        "timestamp": float(window["timestamp"].iloc[-1]),
    }
    try:
        prediction = request_json(
            "POST",
            f"{backend_url}/v1/predict",
            api_key,
            payload,
            timeout=60,
        )
        st.session_state.prediction = prediction
        st.session_state.history.append({
            "time": float(window["timestamp"].iloc[-1]),
            "binary_probability": prediction["binary_probability"],
            "subtype_probability": prediction["subtype_probability"],
            "cdt": prediction["cdt"],
            "cera": prediction["cera"],
            "cas": prediction["cas"],
            "ensemble_variance": prediction["ensemble_variance"],
            "decision": prediction["decision_status"],
        })
        st.session_state.history = st.session_state.history[-300:]
    except Exception as exc:
        st.error(f"Inference unavailable: {exc}")

prediction = st.session_state.prediction
if prediction is None:
    st.info("Start monitoring or provide at least eight signal samples.")
    if st.session_state.running:
        time.sleep(refresh_seconds)
        st.rerun()
    st.stop()

metrics = st.columns(6)
metrics[0].metric("Apnea probability", f"{prediction['binary_probability']:.1%}")
metrics[1].metric("Binary set", ", ".join(prediction["binary_conformal_set"]) or "empty")
metrics[2].metric("Subtype", prediction["subtype_hypothesis"].title())
metrics[3].metric("Subtype confidence", f"{prediction['subtype_probability']:.1%}")
metrics[4].metric("CDT", f"{prediction['cdt']:.2f}", prediction["cdt_level"])
metrics[5].metric("CAS", f"{prediction['cas']:.2f}", prediction["cas_level"])

deterministic = prediction["deterministic_explanation"]
css = "assisted" if prediction["decision_status"] == "assisted_review" else "review"
st.markdown(
    f'<div class="{css}">'
    f'<b>{prediction["decision_status"].replace("_"," ").title()}</b><br>'
    f'<b>{deterministic["headline"]}</b><br>'
    f'{deterministic["explanation"]}<br>'
    f'<span class="small">{deterministic["trust_summary"]} '
    f'{deterministic["limitation"]}</span></div>',
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "Live waveforms",
    "Where is the problem?",
    "Monitoring trends",
    "Signal quality",
    "Trust & conformal",
    "Doctor explanation",
    "Clinician feedback",
    "Audit",
])

with tabs[0]:
    st.plotly_chart(
        waveform_figure(window, prediction),
        use_container_width=True,
    )

with tabs[1]:
    left, right = st.columns(2)
    audit = prediction["physiological_audit"]
    with left:
        st.subheader("Learned event location")
        if prediction["event_start_index"] is None:
            st.info("No localized interval crossed the event-localization threshold.")
        else:
            st.warning(
                f"Samples {prediction['event_start_index']}–"
                f"{prediction['event_end_index']} • peak "
                f"{prediction['localization_peak']:.2f}"
            )
        st.subheader("Why")
        for statement in audit["evidence"]:
            st.write("•", statement)
        for statement in audit["conflicts"]:
            st.warning(statement)
    with right:
        st.dataframe(pd.DataFrame({
            "component": [
                "Airflow reduction", "Preserved effort", "Reduced effort",
                "Effort return", "Delayed desaturation", "Pulse recovery",
                "Rule consistency",
            ],
            "score": [
                audit["airflow_reduction"],
                audit["preserved_effort"],
                audit["reduced_effort"],
                audit["effort_return"],
                audit["delayed_desaturation"],
                audit["pulse_recovery"],
                audit["rule_consistency"],
            ],
        }), hide_index=True, use_container_width=True)
        st.write("Signal-only subtype:", prediction["signal_only_subtype"].title())
        st.write("Context-adjusted subtype:", prediction["subtype_hypothesis"].title())
        if prediction["context_changed_class"]:
            st.error("Context changed the signal-only subtype; review is mandatory.")

with tabs[2]:
    history = pd.DataFrame(st.session_state.history)
    if len(history) > 1:
        st.plotly_chart(trend_figure(history), use_container_width=True)
        st.dataframe(history.tail(30), hide_index=True, use_container_width=True)
    else:
        st.info("More monitoring cycles are needed.")

with tabs[3]:
    cols = st.columns(4)
    for column, name, present in zip(
        cols,
        ("Chest effort", "Airflow", "Pulse", "SpO₂"),
        modality_present,
    ):
        column.metric(name, "available" if present else "missing")
    st.metric("Combined signal-quality index", f"{prediction['signal_quality']:.2f}")
    if not chest_present or not flow_present:
        st.error(
            "Airflow and thoracic effort are required for reliable mechanism review."
        )

with tabs[4]:
    cols = st.columns(3)
    cols[0].metric("CDT", f"{prediction['cdt']:.2f}", prediction["cdt_level"])
    cols[1].metric("CERA", f"{prediction['cera']:.2f}", prediction["cera_level"])
    cols[2].metric("CAS", f"{prediction['cas']:.2f}", prediction["cas_level"])
    st.write("Binary conformal set:", prediction["binary_conformal_set"])
    st.write("Subtype conformal set:", prediction["subtype_conformal_set"])
    st.write("Domain status:", prediction["domain_status"])
    st.write("Domain similarity:", f"{prediction['domain_similarity']:.2f}")
    st.write("Ensemble variance:", f"{prediction['ensemble_variance']:.5f}")
    st.info("CDT, CERA, and CAS are indices, not probabilities.")
    st.bar_chart(pd.DataFrame(
        {"probability": prediction["subtype_probabilities"]}
    ))

with tabs[5]:
    st.subheader("Deterministic explanation")
    st.markdown(
        f"**{deterministic['headline']}**\n\n"
        f"{deterministic['explanation']}\n\n"
        f"**Trust:** {deterministic['trust_summary']}\n\n"
        f"**Action:** {deterministic['action']}\n\n"
        f"**Limitation:** {deterministic['limitation']}"
    )
    st.divider()
    st.subheader("Optional LLM wording")
    st.caption(
        "The LLM is never called automatically. It can only rephrase the locked result."
    )
    if st.button("Generate LLM explanation"):
        try:
            st.session_state.llm_explanation = request_json(
                "POST",
                f"{backend_url}/v1/explain",
                api_key,
                {"prediction": prediction},
                timeout=60,
            )
        except Exception as exc:
            st.error(str(exc))
    if st.session_state.llm_explanation:
        llm = st.session_state.llm_explanation
        st.markdown(
            f"**{llm['headline']}**\n\n"
            f"{llm['explanation']}\n\n"
            f"**Trust:** {llm['trust_summary']}\n\n"
            f"**Action:** {llm['action']}\n\n"
            f"**Limitation:** {llm['limitation']}"
        )

with tabs[6]:
    st.subheader("Review reasons")
    for reason in prediction["review_reasons"]:
        st.write("•", reason)
    clinician_id = st.text_input("Clinician ID")
    action = st.selectbox(
        "Action",
        (
            "confirm_event", "reject_event", "change_subtype",
            "mark_artifact", "request_second_review",
        ),
    )
    corrected_subtype = st.selectbox(
        "Corrected subtype",
        ("", "obstructive", "central", "mixed", "indeterminate"),
    )
    note = st.text_area("Clinical note")
    if st.button("Save clinician feedback"):
        if not clinician_id:
            st.error("Clinician ID is required.")
        else:
            try:
                request_json(
                    "POST",
                    f"{backend_url}/v1/feedback",
                    api_key,
                    {
                        "case_id": st.session_state.case_id,
                        "clinician_id": clinician_id,
                        "action": action,
                        "corrected_subtype": corrected_subtype or None,
                        "note": note or None,
                    },
                )
                st.success("Feedback saved.")
            except Exception as exc:
                st.error(str(exc))

with tabs[7]:
    audit_record = dict(prediction)
    if st.session_state.llm_explanation:
        audit_record["llm_explanation"] = st.session_state.llm_explanation
    st.json(audit_record)
    st.download_button(
        "Download audit JSON",
        data=json.dumps(audit_record, indent=2),
        file_name=f"trust_sleep_{st.session_state.case_id}.json",
        mime="application/json",
    )

if st.session_state.running:
    time.sleep(refresh_seconds)
    st.rerun()
