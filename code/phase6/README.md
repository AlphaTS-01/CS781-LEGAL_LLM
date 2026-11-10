# Phase 6 · Legal summarization

[Experiment notebook](run.ipynb)

## Experiment records

The [method comparison](runs/comparison.csv) records six completed runs. Its ROUGE-1 and ROUGE-L scores are from held-out summaries. The Codabench table uses the displayed test BERTScore F1. Each method folder contains generated validation summaries, inference metadata and test predictions. Reference summaries remain in the source data and original Kaggle archive.

[Run and checkpoint manifest](run_manifest.json) · [Archived Kaggle cell source](session_source.py)

| Validation run | ROUGE-L F1 |
| --- | ---: |
| [lora](runs/lora/metrics.json) | 0.1637 |
| [lorma](runs/lorma/metrics.json) | 0.1685 |
| [houlsby](runs/houlsby/metrics.json) | 0.1522 |
| [pfeiffer](runs/pfeiffer/metrics.json) | 0.1526 |
| [prompt](runs/prompt/metrics.json) | 0.1487 |
| [kd](runs/kd/metrics.json) | 0.1647 |

## Local evaluation

Supply reference summaries keyed by case ID and a prediction export with `id` and `summary` fields. Install `torch` and `bert-score` to compute English BERTScore F1. The first run downloads the scoring model. Run from this phase directory.

```sh
python evaluate_phase6.py --truth truth.json --prediction prediction.json
```

## Codabench submissions

| Submission ID | File name | Date shown | Status | Score shown | Predictions |
| ---: | --- | --- | --- | ---: | --- |
| 1029566 | `pmpt6.zip` | 2026-11-03 12:49 | Finished | 0.8162 | [JSON](runs/prompt/prediction.json) |
| 1029351 | `pmpt6.zip` | 2026-11-03 11:00 | Failed | — | Unverified |
| 1029350 | `hbyp6.zip` | 2026-11-03 11:00 | Finished | 0.8253 | [JSON](runs/houlsby/prediction.json) |
| 1029348 | `pfp6.zip` | 2026-11-03 10:59 | Finished | 0.8263 | [JSON](runs/pfeiffer/prediction.json) |
| 1029346 | `lormap6.zip` | 2026-11-03 10:59 | Finished | **0.8276** | [JSON](runs/lorma/prediction.json) |
| 1029345 | `lorap6.zip` | 2026-11-03 10:58 | Finished | 0.8269 | [JSON](runs/lora/prediction.json) |
