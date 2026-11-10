# Phase 5 · Statute identification

[Experiment notebook](run.ipynb)

## Experiment records

All six [submission predictions](submissions) were recovered from the individually named local ZIPs. The full Kaggle session archive, fitted thresholds, development predictions and per-run checkpoint records were unavailable. The notebook contains the training and calibration procedure.

[Run and checkpoint manifest](run_manifest.json)

## Local evaluation

Use matching held-out lists of records with `id` and `gold` or `pred` fields. Each label list contains integer statute IDs. The local scorer averages over IDs from zero through the largest gold ID. Run from this phase directory.

```sh
python evaluate_phase5.py --truth truth.json --prediction prediction.json
```

## Codabench submissions

| Submission ID | File name | Date shown | Status | Score shown | Predictions |
| ---: | --- | --- | --- | ---: | --- |
| 1013054 | `prmptp5.zip` | 2026-10-27 18:26 | Finished | 0.1622 | [JSON](submissions/1013054/prediction.json) |
| 1013053 | `kdp5.zip` | 2026-10-27 18:26 | Finished | 0.1307 | [JSON](submissions/1013053/prediction.json) |
| 1012798 | `adapterp5.zip` | 2026-10-27 16:38 | Finished | 0.2218 | [JSON](submissions/1012798/prediction.json) |
| 1012793 | `lormap5.zip` | 2026-10-27 16:36 | Finished | 0.2298 | [JSON](submissions/1012793/prediction.json) |
| 1012499 | `partialp5.zip` | 2026-10-27 14:33 | Finished | 0.1488 | [JSON](submissions/1012499/prediction.json) |
| 1012181 | `fullp5.zip` | 2026-10-27 12:19 | Finished | **0.2685** | [JSON](submissions/1012181/prediction.json) |
