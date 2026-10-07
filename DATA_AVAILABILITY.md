# Data availability and expected input format

The pediatric PSG dataset used in the Trust-Sleep study is clinical/restricted data and is **not redistributed in this repository**. Reproducibility therefore has two levels:

1. **Computational reproducibility:** all analysis/training code, fixed seeds, split protocol, aggregate reference tables, and figures are provided.
2. **Exact numerical rerun:** requires authorized access to the same master HDF5 file, `experiment1b.h5`.

## Expected HDF5 structure

The pipeline uses a pandas `HDFStore`. Each labeled epoch is stored at a key of the form:

```text
/<recording_id>/e<epoch_id>_0
/<recording_id>/e<epoch_id>_1
```

The final suffix encodes the binary label (`0` = non-apnea, `1` = apnea). Each epoch must contain exactly 30 rows and the following four numeric columns:

```text
psg_chest_rn_mean
psg_flow_dr_rn_mean
psg_pulse_rn_mean
psg_spo2_rn_mean
```

The pipeline derives the recording identifier from the first path component and performs all splitting at **recording level**, so every epoch from one recording remains in exactly one partition.

## Why row-level outputs are not in the public package

The original clean rerun generated epoch-level audit and prediction files containing pseudonymous recording IDs and source keys. Those row-level artifacts are intentionally excluded from this Git-ready public package. They are regenerated automatically when an authorized user reruns the pipeline on the restricted dataset. Only aggregate tables/figures needed to verify the reported paper results are included under `results/reference/`.
