
# Deployment guide

## 1. Train and calibrate

Train all ensemble members with one fixed split seed:

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

Fit calibration and conformal predictors to the ensemble mean:

```bash
python ensemble_calibration_v5.py \
  --ensemble-manifest /models/training/ensemble_manifest_v5.json \
  --local-h5 /data/experiment1a.h5 \
  --external-manifest /data/external_manifest_v5.csv \
  --metadata /data/metadata.csv \
  --output /models/training/ensemble_bundle_v5.pt
```

Create a portable model package:

```bash
python package_model_bundle_v5.py \
  --ensemble-bundle /models/training/ensemble_bundle_v5.pt \
  --output-dir ./model_package
```

## 2. Configure

```bash
cp .env.example .env
```

Replace both secrets. Keep the LLM endpoint empty unless a private endpoint has been approved.

## 3. Start

```bash
docker compose up --build -d
```

- Dashboard: `http://localhost:8501`
- Backend readiness: `http://localhost:8000/health/ready`

## 4. Real-time device connection

Preferred production path:

```text
PSG/device gateway
  -> authenticated WebSocket /v1/ws/{session_id}
  -> rolling stream buffer
  -> ensemble inference
  -> audit database
  -> dashboard
```

A device or gateway sends:

```json
{
  "case_id": "case-123",
  "timestamps": [1.0, 1.1],
  "signal_rows": [
    [0.1, 0.2, 72.0, 96.0],
    [0.2, 0.1, 72.2, 96.0]
  ],
  "metadata": {"age": 64, "bmi": 32},
  "modality_present": [1, 1, 1, 1],
  "infer": true
}
```

Signal order is always:

```text
chest, airflow, pulse, SpO2
```

## 5. Hospital production requirements not supplied by Docker Compose

The compose file is a reference deployment, not regulatory clearance. A hospital deployment
still needs:

- approved identity provider and OIDC/SSO;
- TLS termination and certificate rotation;
- network segmentation;
- secrets manager;
- encrypted database backups;
- PHI retention policy;
- disaster recovery;
- security assessment;
- clinical safety case;
- model-change control;
- prospective shadow-mode validation;
- local IT and biomedical engineering approval;
- applicable medical-device regulatory review.

Use an enterprise reverse proxy/API gateway and replace the demonstration API key with
hospital identity and authorization.
