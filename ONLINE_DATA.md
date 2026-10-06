# Automatic online external-data preparation

Run in Colab:

```bash
cd /content/trust_sleep_deployment_v5_1
pip install -r requirements.txt

python fetch_ucddb_online_v5.py \
  --destination /content/external_data \
  --manifest-out /content/external_manifest_v5.csv
```

The script automatically:

1. downloads the open-access UCD sleep-apnea PSG dataset from PhysioNet;
2. reads respiratory-event annotations;
3. extracts obstructive, central, mixed, hypopnea, and background windows;
4. creates event-localization targets;
5. records missing channels with explicit modality masks;
6. creates subject-disjoint train, validation, conformal, and test partitions;
7. writes `/content/external_manifest_v5.csv`;
8. writes `/content/external_manifest_v5_subject_metadata.csv`.

Public external metadata describe UCDDB subjects only. They are not metadata for
`experiment1a.h5` or `experiment1b.h5`.

Local metadata must come from the local study, HDF keys, or a hospital export. The
software must never search the internet for private patient attributes.
