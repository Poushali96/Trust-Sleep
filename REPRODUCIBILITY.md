# Reproducibility protocol

## Frozen recording-level partition

The master dataset is repartitioned from scratch with mutually exclusive recording groups:

- 60% training
- 10% early-stopping validation
- 5% Platt probability calibration
- 5% operating-threshold selection
- 10% conformal calibration
- 10% final untouched test

The final test recordings are excluded from model fitting, preprocessing estimation, early stopping, probability calibration, threshold selection, and conformal calibration.

## Experiments

**Experiment 1** uses the Full Trust-Sleep model at seed `271828` for predictive performance, CDT/CERA/CAS analyses, selective prediction, and perturbation robustness.

**Experiment 2** compares the Full model against a matched Raw-Signal-only model over five prespecified seeds: `271828`, `314159`, `161803`, `141421`, `173205`. Both variants use the same frozen recording-level split and the same preprocessing/training/calibration/evaluation protocol. The intended architectural difference is the event-centric branch.

## Determinism

The code sets Python, NumPy, and PyTorch seeds, disables cuDNN benchmarking, requests deterministic algorithms where supported, and records runtime metadata. GPU kernels and library updates can still produce small numerical differences across hardware/software stacks.

## Reference runtime

The supplied reference run records Python 3.13.15, NumPy 2.1.3, pandas 2.2.3, PyTorch 2.11.0+cu130, CUDA 13.0, cuDNN 92700, and an NVIDIA Tesla T4. See `results/reference/runtime_environment.json`.

## Reference-result check

After a full rerun, compare your aggregate CSV files against `results/reference/tables/`. Exact equality is expected when the same data and compatible deterministic runtime are used; otherwise small floating-point differences may occur.
