
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from audit_store_v5 import AuditStore
from drift_monitor_v5 import DriftMonitor
from fhir_v5 import prediction_to_fhir_bundle
from inference_v5 import TrustSleepDeploymentEngine
from streaming_v5 import StreamBufferStore


BUNDLE_PATH = os.getenv(
    "TRUST_SLEEP_ENSEMBLE_BUNDLE",
    "ensemble_bundle_v5.pt",
)
DATABASE_URL = os.getenv(
    "TRUST_SLEEP_DATABASE_URL",
    "sqlite:///trust_sleep_audit.sqlite3",
)
API_KEY = os.getenv("TRUST_SLEEP_API_KEY", "")
LLM_ENDPOINT = os.getenv("TRUST_SLEEP_LLM_ENDPOINT")
MAX_STREAM_POINTS = int(os.getenv("TRUST_SLEEP_MAX_STREAM_POINTS", "600"))

app = FastAPI(
    title="Trust-Sleep Deployment API",
    version="5.0.0",
    description=(
        "Research deployment backend for real-time sleep-apnea decision support. "
        "Subtype outputs require clinician confirmation."
    ),
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "TRUST_SLEEP_ALLOWED_ORIGINS",
        "http://localhost:8501",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

engine: Optional[TrustSleepDeploymentEngine] = None
audit_store: Optional[AuditStore] = None
stream_store = StreamBufferStore(max_points=MAX_STREAM_POINTS)
drift_monitor = DriftMonitor()
startup_error: Optional[str] = None


def require_api_key(
    x_api_key: Optional[str] = Header(default=None),
) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")


@app.on_event("startup")
def startup() -> None:
    global engine, audit_store, startup_error
    try:
        engine = TrustSleepDeploymentEngine(
            BUNDLE_PATH,
            llm_endpoint=None,
        )
        audit_store = AuditStore(DATABASE_URL)
        startup_error = None
    except Exception as exc:
        startup_error = f"{type(exc).__name__}: {exc}"
        engine = None
        audit_store = None


class PredictRequest(BaseModel):
    case_id: str
    session_id: str
    signal: List[List[float]] = Field(description="[time,4]")
    metadata: Dict[str, Any] = {}
    modality_present: Optional[List[float]] = None
    timestamp: Optional[float] = None


class ExplainRequest(BaseModel):
    prediction: Dict[str, Any]


class FeedbackRequest(BaseModel):
    case_id: str
    clinician_id: str
    action: str
    corrected_subtype: Optional[str] = None
    note: Optional[str] = None


class StreamChunk(BaseModel):
    case_id: str
    session_id: str
    timestamps: List[float]
    signal_rows: List[List[float]]
    metadata: Dict[str, Any] = {}
    modality_present: Optional[List[float]] = None
    infer: bool = True


class FHIRExportRequest(BaseModel):
    prediction: Dict[str, Any]
    patient_reference: str
    encounter_reference: Optional[str] = None


@app.get("/health/live")
def health_live() -> Dict[str, Any]:
    return {"status": "alive", "time": time.time()}


@app.get("/health/ready")
def health_ready() -> Dict[str, Any]:
    if engine is None or audit_store is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "error": startup_error,
            },
        )
    return {
        "status": "ready",
        "bundle_version": engine.bundle_version,
        "ensemble_members": len(engine.models),
        "database_backend": audit_store.backend,
    }


@app.post("/v1/predict", dependencies=[Depends(require_api_key)])
def predict(request: PredictRequest) -> Dict[str, Any]:
    if engine is None or audit_store is None:
        raise HTTPException(status_code=503, detail="Backend is not ready.")
    try:
        signal = np.asarray(request.signal, dtype=np.float32)
        result = engine.predict(
            signal,
            request.metadata,
            session_id=request.session_id,
            case_id=request.case_id,
            timestamp=request.timestamp,
            modality_present=(
                None
                if request.modality_present is None
                else np.asarray(request.modality_present, dtype=np.float32)
            ),
        )
        payload = result.to_dict()
        drift_monitor.update(payload)
        audit_store.log_prediction(
            case_id=request.case_id,
            session_id=request.session_id,
            model_version=engine.bundle_version,
            input_checksum=engine.input_checksum(signal),
            prediction=payload,
        )
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/explain", dependencies=[Depends(require_api_key)])
def explain(request: ExplainRequest) -> Dict[str, Any]:
    if engine is None:
        raise HTTPException(status_code=503, detail="Backend is not ready.")
    if not LLM_ENDPOINT:
        raise HTTPException(
            status_code=409,
            detail="No private LLM endpoint is configured.",
        )
    return engine.generate_llm_explanation(
        request.prediction,
        endpoint=LLM_ENDPOINT,
    )


@app.post("/v1/feedback", dependencies=[Depends(require_api_key)])
def feedback(request: FeedbackRequest) -> Dict[str, str]:
    if audit_store is None or engine is None:
        raise HTTPException(status_code=503, detail="Backend is not ready.")
    audit_store.add_feedback(
        case_id=request.case_id,
        clinician_id=request.clinician_id,
        action=request.action,
        corrected_subtype=request.corrected_subtype,
        note=request.note,
        model_version=engine.bundle_version,
        artifact=request.action == "mark_artifact",
        second_review=request.action == "request_second_review",
    )
    return {"status": "saved"}


@app.post("/v1/stream/chunk", dependencies=[Depends(require_api_key)])
def stream_chunk(request: StreamChunk) -> Dict[str, Any]:
    if engine is None or audit_store is None:
        raise HTTPException(status_code=503, detail="Backend is not ready.")
    try:
        stream_store.append(
            request.session_id,
            request.timestamps,
            request.signal_rows,
        )
        if not request.infer:
            return {"status": "buffered", "samples": len(request.signal_rows)}

        timestamps, signal = stream_store.snapshot(request.session_id)
        result = engine.predict(
            signal,
            request.metadata,
            session_id=request.session_id,
            case_id=request.case_id,
            timestamp=float(timestamps[-1]),
            modality_present=(
                None
                if request.modality_present is None
                else np.asarray(request.modality_present, dtype=np.float32)
            ),
        )
        payload = result.to_dict()
        drift_monitor.update(payload)
        audit_store.log_prediction(
            case_id=request.case_id,
            session_id=request.session_id,
            model_version=engine.bundle_version,
            input_checksum=engine.input_checksum(signal),
            prediction=payload,
        )
        return {
            "status": "predicted",
            "window_samples": len(signal),
            "prediction": payload,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v1/drift", dependencies=[Depends(require_api_key)])
def drift_status() -> Dict[str, Any]:
    return drift_monitor.summary().__dict__


@app.post("/v1/fhir/export", dependencies=[Depends(require_api_key)])
def fhir_export(request: FHIRExportRequest) -> Dict[str, Any]:
    return prediction_to_fhir_bundle(
        request.prediction,
        patient_reference=request.patient_reference,
        encounter_reference=request.encounter_reference,
    )


@app.websocket("/v1/ws/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str) -> None:
    supplied_key = websocket.headers.get("x-api-key")
    if API_KEY and supplied_key != API_KEY:
        await websocket.close(code=4401)
        return
    await websocket.accept()

    if engine is None or audit_store is None:
        await websocket.send_json({
            "type": "error",
            "detail": "Backend is not ready.",
        })
        await websocket.close(code=1013)
        return

    try:
        while True:
            message = await websocket.receive_json()
            timestamps = message.get("timestamps", [])
            rows = message.get("signal_rows", [])
            case_id = str(message.get("case_id", session_id))
            metadata = message.get("metadata", {})
            modality_present = message.get("modality_present")
            infer_now = bool(message.get("infer", True))

            stream_store.append(session_id, timestamps, rows)
            await websocket.send_json({
                "type": "ack",
                "received": len(rows),
            })

            if not infer_now:
                continue

            try:
                buffered_timestamps, signal = stream_store.snapshot(session_id)
            except ValueError as exc:
                await websocket.send_json({
                    "type": "waiting",
                    "detail": str(exc),
                })
                continue

            result = await asyncio.to_thread(
                engine.predict,
                signal,
                metadata,
                session_id,
                case_id,
                float(buffered_timestamps[-1]),
                (
                    None
                    if modality_present is None
                    else np.asarray(modality_present, dtype=np.float32)
                ),
            )
            payload = result.to_dict()
            drift_monitor.update(payload)
            await asyncio.to_thread(
                audit_store.log_prediction,
                case_id=case_id,
                session_id=session_id,
                model_version=engine.bundle_version,
                input_checksum=engine.input_checksum(signal),
                prediction=payload,
            )
            await websocket.send_json({
                "type": "prediction",
                "payload": payload,
            })

    except WebSocketDisconnect:
        stream_store.clear(session_id)
    except Exception as exc:
        await websocket.send_json({
            "type": "error",
            "detail": f"{type(exc).__name__}: {exc}",
        })
        await websocket.close(code=1011)
