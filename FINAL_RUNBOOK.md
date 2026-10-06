# Trust-Sleep Final v5.1 Runbook

## What this package includes

- Online external-data fetcher and manifest builder:
  `fetch_ucddb_online_v5.py`
- Local binary HDF5 training and testing:
  `experiment1a.h5` for development, `experiment1b.h5` for independent test
- Five-seed ensemble training:
  `ensemble_runner_v5.py`
- Ensemble-level calibration and conformal prediction:
  `ensemble_calibration_v5.py`
- External subtype test:
  `evaluate_external_v5.py`
- Local independent binary test:
  `evaluate_local_v5.py`
- Deployment backend:
  `backend_v5.py`
- Doctor-facing dashboard:
  `dashboard_v5.py`
- Docker deployment:
  `docker-compose.yml`, `Dockerfile.backend`, `Dockerfile.dashboard`
- Audit, streaming, FHIR export, drift monitoring, active learning, and optional LLM wording.

## Colab quick start

```bash
cd /content/trust_sleep_final_v5_1
pip install -r requirements.txt

python fetch_ucddb_online_v5.py \
  --destination /content/external_data \
  --manifest-out /content/external_manifest_v5.csv

python external_data.py \
  --validate-manifest /content/external_manifest_v5.csv

python ensemble_runner_v5.py \
  --local-h5 /content/experiment1a.h5 \
  --external-manifest /content/external_manifest_v5.csv \
  --output /content/v5_training \
  --split-seed 271828 \
  --epochs 20 \
  --batch-size 512

python ensemble_calibration_v5.py \
  --ensemble-manifest /content/v5_training/ensemble_manifest_v5.json \
  --local-h5 /content/experiment1a.h5 \
  --external-manifest /content/external_manifest_v5.csv \
  --output /content/v5_training/ensemble_bundle_v5.pt

python evaluate_external_v5.py \
  --ensemble-bundle /content/v5_training/ensemble_bundle_v5.pt \
  --external-manifest /content/external_manifest_v5.csv \
  --output /content/v5_external_test

python evaluate_local_v5.py \
  --test-h5 /content/experiment1b.h5 \
  --ensemble-bundle /content/v5_training/ensemble_bundle_v5.pt \
  --output /content/v5_local_test
```

## Metadata rule

The online script creates public metadata for UCDDB subjects only. It must not be used
as metadata for `experiment1a.h5` or `experiment1b.h5`. Local metadata must come from the
local study or hospital export. If unavailable, omit `--metadata`.

## LLM rule

The LLM is never called automatically. The deterministic explanation is the source of
truth. Optional LLM wording is only triggered by the separate dashboard button or
`POST /v1/explain`.
