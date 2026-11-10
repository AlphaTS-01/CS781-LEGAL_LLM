# Phase 4 · Judgment outcome prediction

[Experiment notebook](run.ipynb)

## Experiment records

The [five saved runs](runs) include development probabilities, reference labels, selected thresholds and training history. The historical fold contains repeated case IDs. Evaluate the ordered saved rows with `--saved-run` to reproduce the reported macro-F1. The separate truth/prediction interface expects unique submission IDs.

[Run and checkpoint manifest](run_manifest.json) · [Archived Kaggle cell source](session_source.py)

| Validation run | Thresholded macro-F1 |
| --- | ---: |
| [E0_full](runs/E0_full/done.json) | 0.6940 |
| [E1_lorma](runs/E1_lorma/done.json) | 0.5131 |
| [E2_adalora](runs/E2_adalora/done.json) | 0.6581 |
| [E3_adapter](runs/E3_adapter/done.json) | 0.6608 |
| [E4_bitfit](runs/E4_bitfit/done.json) | 0.6289 |

## Local evaluation

The example uses the included E0 records. For paired JSON exports, pass `--truth` and `--prediction` with lists of records containing `id` and an integer `gold` or `pred` field. Run from this phase directory.

```sh
python evaluate_phase4.py --saved-run runs/E0_full/dev_predictions.json
```

## Codabench submissions

| Submission ID | File name | Date shown | Status | Score shown | Predictions |
| ---: | --- | --- | --- | ---: | --- |
| 1007875 | `e3p4.zip` | 2026-10-21 15:03 | Finished | 0.6704 | [JSON](runs/E3_adapter/prediction.json) |
| 1007874 | `e1p4.zip` | 2026-10-21 15:02 | Finished | 0.4940 | [JSON](runs/E1_lorma/prediction.json) |
| 1007871 | `e2p4.zip` | 2026-10-21 15:01 | Preparing | — | [JSON](runs/E1_lorma/prediction.json) |
| 1007383 | `e0_p4.zip` | 2026-10-21 11:13 | Finished | **0.6752** | [JSON](runs/E0_full/prediction.json) |
