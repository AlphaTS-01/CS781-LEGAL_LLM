# Phase 3 · Rhetorical role prediction

[Rhetorical roles notebook](rhetorical_roles_experiments.ipynb)

## Kaggle inputs and run order

1. Attach the LEXWeave foundation archive and the official Phase 3 rhetorical-role splits. Set `CS781_FOUNDATION` to the archive or extracted model directory and `CS781_WORK` to a writable phase-specific folder.
2. Run sections 0–4 for setup, model loading, CL/IT data checks and sentence windows. Sections 5–8 define the head, adaptations, evaluation and submission writer.
3. In section 9, use `RUN_QUEUE = ["E2_lora"]` for the selected run. For a comparison, add experiment names from the notebook's registry. Put the teacher before the student when selecting distillation. Use the same working folder to reuse finished runs.
4. Use the selected run's `submission.zip` for test predictions. For local scoring, use its held-out predictions and both domain reference maps.

## Results

Each [run directory](runs) contains held-out predictions, training history and the test export. The training log reports macro-F1 over observed labels and a separate score over the full 13-label vocabulary. The local evaluator reports the latter. The [CL](validation/CL_labels.json) and [IT](validation/IT_labels.json) reference maps contain the validation labels keyed by case ID.

[Run and checkpoint manifest](run_manifest.json) · [Archived Kaggle cell source](session_source.py)

| Validation run | Macro-F1, 13 labels |
| --- | ---: |
| [E0_full](runs/E0_full/done.json) | 0.4376 |
| [E1_lorma](runs/E1_lorma/done.json) | 0.4272 |
| [E2_lora](runs/E2_lora/done.json) | 0.4401 |
| [E3_rslora](runs/E3_rslora/done.json) | 0.4373 |
| [E4_ia3](runs/E4_ia3/done.json) | 0.4576 |
| [E5_prompt](runs/E5_prompt/done.json) | 0.4327 |
| [E6_kd](runs/E6_kd/done.json) | 0.4416 |

## Local evaluation

The evaluator also reports macro-precision, macro-recall and document and sentence counts.

The command below uses included validation files for E2 LoRA. Run from this phase directory.

```sh
python evaluate_phase3.py --truth validation/CL_labels.json --truth-it validation/IT_labels.json --labels label_vocab.json --prediction runs/E2_lora/dev_predictions.json
```

## Codabench submissions

| Submission ID | File name | Date shown | Status | Score shown | Predictions |
| ---: | --- | --- | --- | ---: | --- |
| 993600 | `e4_ia3.zip` | 2026-10-16 20:11 | Finished | 0.4185 | [JSON](runs/E4_ia3/prediction.json) |
| 993467 | `kd.zip` | 2026-10-16 19:43 | Finished | 0.4505 | [JSON](runs/E6_kd/prediction.json) |
| 993275 | `prompt.zip` | 2026-10-16 18:59 | Finished | 0.4199 | [JSON](runs/E5_prompt/prediction.json) |
| 993274 | `e4_ia3.zip` | 2026-10-16 18:59 | Cancelled | — | — |
| 993272 | `e3rslora.zip` | 2026-10-16 18:59 | Finished | 0.4442 | [JSON](runs/E3_rslora/prediction.json) |
| 993270 | `e2lora.zip` | 2026-10-16 18:58 | Finished | **0.4792** | [JSON](runs/E2_lora/prediction.json) |
| 992773 | `e1lormap3.zip` | 2026-10-16 15:50 | Finished | 0.4171 | [JSON](runs/E1_lorma/prediction.json) |
| 992611 | `submission.zip` | 2026-10-16 14:57 | Finished | 0.4306 | [JSON](runs/E0_full/prediction.json) |
