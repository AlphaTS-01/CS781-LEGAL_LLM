# Phase 2 · Named entity recognition

[Experiment notebook](model1_lexweave/ner_lora_crf.ipynb)

## Kaggle inputs and run order

1. Attach the Phase 1 LEXWeave input and the official Phase 2 LNER files, or enable the notebook's authenticated data download. The three foundation files and input selector are described below.
2. Run the notebook through **Train and evaluate the tagger** to prepare the document split, character offsets, token windows, model registry and scoring functions.
3. In **Train and score a recovered experiment**, leave `RUN` at `E3_lora_M1_lexweave` for the winning LoRA + CRF submission, or choose one of the other recorded configurations. For E6 distillation, the notebook restores the E1 teacher or trains it first. The [teacher–student comparison](model1_lexweave/teacher_student.csv) includes parameter counts, training times and strict macro-F1.
4. Collect `OUT_DIR/<experiment>/submission.zip`. The separate continued E3 run starts from the original E3 weights. Its historical continuation and interrupted-run recovery cells are in the archived session source.

## Experiment records

The Kaggle registry resolves E3 to the custom LEXWeave backbone. The predictions for E3 match the local export named `e3_lora.zip` across every document. The continuation run matches `final.zip`. The [E3 configuration and metrics](model1_lexweave/E3_lora_M1_lexweave/metrics.json) record rank 16, alpha 32, a CRF head and a 512-token window.

Attach `model.pt`, `tokenizer.json` and the [LEXWeave implementation](model1_lexweave/lexweave_model.py) in the same Kaggle input directory. The [E3 notebook](model1_lexweave/ner_lora_crf.ipynb) uses the recovered training functions and saved configuration. Set `CS781_FOUNDATION` to select the bundle explicitly.

[Run and checkpoint manifest](checkpoint_manifest.json) · [Archived Kaggle cell source](session_source.py)

| Validation run | Strict macro-F1 |
| --- | ---: |
| [E1_full_M1_lexweave](model1_lexweave/E1_full_M1_lexweave/metrics.json) | 0.4161 |
| [E2_lorma_M1_lexweave](model1_lexweave/E2_lorma_M1_lexweave/metrics.json) | 0.4768 |
| [E3_lora_cont_M1_lexweave](model1_lexweave/E3_lora_cont_M1_lexweave/metrics.json) | 0.4648 |
| [E3_lora_M1_lexweave](model1_lexweave/E3_lora_M1_lexweave/metrics.json) | 0.4496 |
| [E5_partial_M1_lexweave](model1_lexweave/E5_partial_M1_lexweave/metrics.json) | 0.4674 |
| [E6_kd_M1_lexweave](model1_lexweave/E6_kd_M1_lexweave/metrics.json) | 0.4474 |

## Local evaluation

The JSON report also gives macro-precision, macro-recall and per-label scores.

Both JSON files map document IDs to lists of `[start, end, label]` character spans. The scorer computes strict macro-F1 over the supplied label vocabulary. Run from this phase directory.

```sh
python evaluate_phase2.py --truth truth.json --prediction prediction.json --labels labels.txt
```

## Codabench submissions

| Submission ID | File name | Date shown | Status | Score shown | Predictions |
| ---: | --- | --- | --- | ---: | --- |
| 978777 | `final.zip` | 2026-10-11 23:55 | Finished | 0.4858 | [JSON](model1_lexweave/E3_lora_cont_M1_lexweave/prediction.json) |
| 977900 | `e6_kd.zip` | 2026-10-11 19:38 | Finished | 0.4629 | [JSON](model1_lexweave/E6_kd_M1_lexweave/prediction.json) |
| 977798 | `e5_partial.zip` | 2026-10-11 19:06 | Finished | 0.4853 | [JSON](model1_lexweave/E5_partial_M1_lexweave/prediction.json) |
| 977791 | `e3_lora.zip` | 2026-10-11 19:05 | Finished | **0.4957** | [JSON](model1_lexweave/E3_lora_M1_lexweave/prediction.json) |
| 977696 | `E2_lorma.zip` | 2026-10-11 18:42 | Finished | 0.4956 | [JSON](model1_lexweave/E2_lorma_M1_lexweave/prediction.json) |
| 977563 | `E1_full.zip` | 2026-10-11 18:14 | Finished | 0.4486 | [JSON](model1_lexweave/E1_full_M1_lexweave/prediction.json) |
