# Phase 5 · Statute identification

[Statute identification notebook](statute_identification.ipynb)

## Kaggle inputs and run order

1. Attach the LEXWeave foundation and `Phase_5_LSI` data. Set `LOCAL_DATA` to the data root or enable the authenticated download. Use a writable `WORK` folder.
2. Sections 2–6 load the original model, build the statute vocabulary, cache document windows, define adaptations and calibrate thresholds. Run the correctness checks in section 8 before starting section 9.
3. Keep `EXPERIMENTS = ["full"]` for the selected run. To run the six-way comparison, use `["full", "partial", "lorma", "adapter", "prompt", "kd"]`. KD selects a completed teacher by selection-set macro-F1. Set `EXPORT_ALL_VARIANTS = True` to generate all six test exports.
4. Sections 10–11 write the comparison and predictions. The selected configuration writes `runs/full/submission.zip`; its local prediction file is linked in the submission table.

## Results

Full fine-tuning scored 0.2685 macro-F1 on Codabench, followed by LoRMA at 0.2298 and adapters at 0.2218. The [six prediction files](runs) are grouped by method.

[Run and checkpoint manifest](run_manifest.json)

## Local evaluation

The evaluator reports micro-F1 and exact label-set accuracy.

Use matching held-out lists of records with `id` and `gold` or `pred` fields. Each label list contains integer statute IDs. The local scorer averages over IDs from zero through the largest gold ID. Run from this phase directory.

```sh
python evaluate_phase5.py --truth truth.json --prediction prediction.json
```

## Codabench submissions

| Submission ID | File name | Date shown | Status | Score shown | Predictions |
| ---: | --- | --- | --- | ---: | --- |
| 1013054 | `prmptp5.zip` | 2026-10-27 18:26 | Finished | 0.1622 | [JSON](runs/prompt/prediction.json) |
| 1013053 | `kdp5.zip` | 2026-10-27 18:26 | Finished | 0.1307 | [JSON](runs/kd/prediction.json) |
| 1012798 | `adapterp5.zip` | 2026-10-27 16:38 | Finished | 0.2218 | [JSON](runs/adapter/prediction.json) |
| 1012793 | `lormap5.zip` | 2026-10-27 16:36 | Finished | 0.2298 | [JSON](runs/lorma/prediction.json) |
| 1012499 | `partialp5.zip` | 2026-10-27 14:33 | Finished | 0.1488 | [JSON](runs/partial/prediction.json) |
| 1012181 | `fullp5.zip` | 2026-10-27 12:19 | Finished | **0.2685** | [JSON](runs/full/prediction.json) |
