# Phase 7 · Prior case retrieval

[Prior case retrieval notebook](prior_case_retrieval.ipynb)

## Kaggle inputs and run order

1. Attach the original LEXWeave foundation and the `Phase_7_PCR` Train, Dev and Test query/candidate files. The [dataset revision](runs/download.json) and [input hashes](runs/inputs.json) identify the recorded inputs. Use `RESUME_FROM` to restore a saved run to writable storage.
2. For the winning LoRMA submission, run the [retrieval notebook](prior_case_retrieval.ipynb) with `SMOKE = False` and `EXPERIMENTS = ["lorma"]`. Sections 1–6 set up, validate the retrieval pools and define the encoder. Sections 7–8 train or restore the model, rank candidates and write the dense export.
3. Run **Fuse the dense and full-text rankings**. It combines the dense top seven and the TF-IDF top fifty, applies lexical weight two, filters the query's own ID, and writes `runs/lorma/lorma_confirmed.zip`.
4. To reproduce the full comparison, use the [hybrid reranker notebook](hybrid_rerankers.ipynb). It includes the same setup, runs the five rerankers, evaluates lexical weights on dev, rebuilds the selected LoRMA predictions and exports the six final ZIPs. Restoring the original checkpoint, indexes and reranker weights avoids repeating their training. The [hybrid comparison](final_six_hybrid/comparison.json) records dev F1 gains and the number of changed test queries.

## Results

The first training stage produced a [dense LoRMA baseline](runs/lorma/metrics.json). The later [hybrid comparison](final_six_hybrid/comparison.json) covers the five rerankers applied to the lexical/dense baseline. The [final predictions](final_predictions) are the six exports matched to the submission filenames. The LoRMA export uses full-text TF-IDF at weight 2 and filters each query from its own results.

[Run and checkpoint manifest](run_manifest.json) · [Archived Kaggle cell source](session_source.py)

| Hybrid reranker | Dev micro-F1@7 | Changed test queries | Codabench micro-F1@7 |
| --- | ---: | ---: | ---: |
| [full](final_predictions/full_hybrid/prediction.json) | 0.3469 | 67 | 0.3455 |
| [partial](final_predictions/partial_hybrid/prediction.json) | 0.3469 | 69 | 0.3473 |
| [prompt](final_predictions/prompt_hybrid/prediction.json) | 0.3469 | 70 | 0.3473 |
| [ia3](final_predictions/ia3_hybrid/prediction.json) | 0.3446 | 74 | 0.3467 |
| [kd](final_predictions/kd_hybrid/prediction.json) | 0.3469 | 60 | 0.3467 |

## Local evaluation

Append `--output scores.json` to save the aggregate scores and TP, FP and FN counts.

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
