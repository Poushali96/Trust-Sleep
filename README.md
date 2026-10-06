
# Trust-Sleep Deployment v5

A deployment-oriented research package for:

- local binary sleep-apnea detection;
- externally transferred obstructive, central, mixed, and indeterminate hypotheses;
- five-member deep ensembles;
- ensemble-level probability calibration;
- binary and subtype Mondrian conformal prediction;
- learned event localization using real event-boundary supervision only;
- signal-only subtype inference with a bounded patient-context prior;
- explicit modality-presence masks;
- OOD/domain-similarity monitoring;
- CDT, CERA, and CAS trust indices;
- persistent audit logging and clinician feedback;
- active-learning case selection;
- FHIR R4 export;
- REST and WebSocket real-time inference;
- a backend-connected doctor-facing dashboard;
- optional LLM wording through a separate button only.

## Important boundary

This is a research and validation system, not a cleared autonomous medical device.

`experiment1b.h5` can validate local binary apnea detection because it has binary labels.
It cannot measure local obstructive, central, or mixed accuracy. Local subtype outputs remain
externally transferred hypotheses requiring clinician confirmation.

## Correct data flow

```text
experiment1a.h5
├── local training subjects
├── local validation subjects
└── local binary conformal subjects

external subtype-labelled PSG
├── external training subjects
├── external validation subjects
├── external subtype conformal subjects
└── untouched external subtype test subjects

experiment1b.h5
└── one-time independent local binary test
```

All five ensemble members use the same subject split. Their random initialization and
optimization seed differ; the data partition does not.

## External manifest

Required columns:

```csv
sample_id,npz_path,binary_label,event_label,subtype_label,subject_id,recording_id,split,source,sampling_rate_hz,window_seconds
```

Each NPZ contains:

```text
signal                [time, 4]
modality_present      [4]
localization_target   [time]  # optional but required for localization training
```

Signal order is:

```text
chest, airflow, pulse, SpO2
```

Valid splits:

```text
train, validation, conformal, test
```

A subject must appear in exactly one split.

## 1. Validate external data

```bash
python external_data.py \
  --validate-manifest /data/external_manifest_v5.csv
```

## 2. Train five members

```bash
python ensemble_runner_v5.py \
  --local-h5 /data/experiment1a.h5 \
  --external-manifest /data/external_manifest_v5.csv \
  --metadata /data/metadata.csv \
  --output /models/training \
  --split-seed 271828 \
  --epochs 20 \
  --batch-size 512
```

## 3. Fit ensemble-level calibration

Calibration must be fitted to the ensemble mean, not borrowed from one member.

```bash
python ensemble_calibration_v5.py \
  --ensemble-manifest /models/training/ensemble_manifest_v5.json \
  --local-h5 /data/experiment1a.h5 \
  --external-manifest /data/external_manifest_v5.csv \
  --metadata /data/metadata.csv \
  --output /models/training/ensemble_bundle_v5.pt
```

## 4. Evaluate the untouched external subtype test

```bash
python evaluate_external_v5.py \
  --ensemble-bundle /models/training/ensemble_bundle_v5.pt \
  --external-manifest /data/external_manifest_v5.csv \
  --output /results/external_test
```

## 5. Evaluate the independent local binary test

```bash
python evaluate_local_v5.py \
  --test-h5 /data/experiment1b.h5 \
  --metadata /data/metadata.csv \
  --ensemble-bundle /models/training/ensemble_bundle_v5.pt \
  --output /results/local_test
```

## 6. Create a portable model package

```bash
python package_model_bundle_v5.py \
  --ensemble-bundle /models/training/ensemble_bundle_v5.pt \
  --output-dir ./model_package
```

Check it:

```bash
python deployment_readiness_v5.py \
  --bundle ./model_package/ensemble_bundle_v5.pt
```

## 7. Run the backend and dashboard

```bash
cp .env.example .env
# Edit .env
docker compose up --build -d
```

- Dashboard: `http://localhost:8501`
- Backend liveness: `http://localhost:8000/health/live`
- Backend readiness: `http://localhost:8000/health/ready`

## Real-time integration

### REST window inference

```text
POST /v1/predict
```

### Incremental streaming

```text
POST /v1/stream/chunk
```

### WebSocket device stream

```text
WS /v1/ws/{session_id}
```

Recommended hospital flow:

```text
PSG or bedside device
  -> device integration gateway
  -> authenticated WebSocket
  -> rolling buffer
  -> ensemble inference
  -> PostgreSQL audit
  -> dashboard
  -> clinician feedback
  -> active-learning queue
```

## API endpoints

```text
GET  /health/live
GET  /health/ready
POST /v1/predict
POST /v1/stream/chunk
WS   /v1/ws/{session_id}
POST /v1/explain
POST /v1/feedback
GET  /v1/drift
POST /v1/fhir/export
```

`/v1/predict` never calls an LLM.

`/v1/explain` is an explicit, separate request. The LLM can only rephrase the locked
structured result. It cannot alter probabilities, conformal sets, trust scores, event
boundaries, or the review decision.

## Dashboard

The doctor-facing dashboard includes:

- live chest, airflow, pulse, and SpO2 waveforms;
- learned candidate-event highlighting;
- signal-only and context-adjusted subtype views;
- physiological evidence and conflicts;
- rolling probability and trust trends;
- signal availability and quality;
- binary and subtype conformal sets;
- subtype alternatives;
- CDT, CERA, and CAS;
- deterministic doctor-language explanation;
- separate optional LLM explanation button;
- clinician confirmation, rejection, correction, artifact, and second-review controls;
- downloadable audit JSON.

## Active learning

Export high-value clinician-review cases from the SQLite demo database:

```bash
python active_learning_v5.py \
  --sqlite-db trust_sleep_audit.sqlite3 \
  --output active_learning_queue.csv \
  --limit 500
```

In production, implement the same query against PostgreSQL and route selected cases for
blinded sleep-specialist adjudication.

## Data that still need more training

Highest-priority additions:

1. Local event-level obstructive, central, mixed, hypopnea, and artifact labels.
2. Both thoracic and abdominal effort channels.
3. True event start/end annotations.
4. Sleep stage, body position, arousal, and snoring channels.
5. Pediatric data trained and calibrated separately.
6. Multiple hospitals, devices, sensor vendors, and demographic groups.
7. Rare central and mixed events with adjudication by multiple scorers.
8. Prospective streaming data from the intended hospital workflow.
9. Negative controls and difficult artifacts.
10. Outcome-linked clinician corrections for continual post-deployment monitoring.

## Main limitations

- Local subtype accuracy remains unknown without local subtype labels.
- One chest-effort channel is weaker than thoracic plus abdominal effort.
- Trust indices are validation-derived operational indices, not probabilities.
- OOD detection needs validation against real unseen hospitals and devices.
- External annotation definitions and sensor characteristics may differ from local data.
- A 30-point representation may be inadequate if it does not preserve event morphology.
- The dashboard does not replace a sleep-scoring workstation.
- API-key authentication is only a reference control; production should use hospital SSO/OIDC.
- Docker Compose is a reference deployment, not a high-availability architecture.
- Prospective validation, cybersecurity, human-factors testing, quality management,
  clinical safety review, and regulatory assessment remain mandatory.

See `DEPLOYMENT.md` for the production architecture and operational controls.
