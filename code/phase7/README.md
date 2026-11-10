# Phase 7 · Prior case retrieval

[Experiment notebook](run.ipynb)

## Experiment records

The first training stage produced a [dense LoRMA baseline](runs/lorma/metrics.json). The later [hybrid comparison](final_six_hybrid/comparison.json) covers the five rerankers applied to the lexical/dense baseline. The [final predictions](final_predictions) are the six exports matched to the submission filenames. The LoRMA export uses full-text TF-IDF at weight 2 and filters each query from its own results.

[Run and checkpoint manifest](run_manifest.json) · [Archived Kaggle cell source](session_source.py)

## Local evaluation

Use the labelled dev query file and its candidate ID list. The evaluator checks that every query has exactly seven unique ranked candidates. The example evaluates the dense baseline. Run from this phase directory.

```sh
python evaluate_phase7.py --truth dev.json --prediction runs/lorma/dev_rankings.json --candidate-ids candidate_ids.json
```

## Codabench submissions

| Submission ID | File name | Date shown | Status | Score shown | Predictions |
| ---: | --- | --- | --- | ---: | --- |
| 1040966 | `prompt_hybrid.zip` | 2026-11-08 01:20 | Finished | 0.3473 | [JSON](final_predictions/prompt_hybrid/prediction.json) |
| 1040965 | `lorma_confirmed.zip` | 2026-11-08 01:20 | Finished | **0.3538** | [JSON](final_predictions/fulltext_w2.0_exclude_self_True/prediction.json) |
| 1040964 | `kd_hybrid.zip` | 2026-11-08 01:20 | Finished | 0.3467 | [JSON](final_predictions/kd_hybrid/prediction.json) |
| 1040963 | `partial_hybrid.zip` | 2026-11-08 01:20 | Finished | 0.3473 | [JSON](final_predictions/partial_hybrid/prediction.json) |
| 1040962 | `ia3_hybrid.zip` | 2026-11-08 01:20 | Finished | 0.3467 | [JSON](final_predictions/ia3_hybrid/prediction.json) |
| 1040961 | `full_hybrid.zip` | 2026-11-08 01:20 | Finished | 0.3455 | [JSON](final_predictions/full_hybrid/prediction.json) |
