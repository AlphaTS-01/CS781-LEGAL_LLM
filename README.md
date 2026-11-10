# CS781 · Legal Language Modelling

Tejas Shrivastava · CS781 capstone

This project trains compact language models on legal text and adapts them for six downstream tasks. The [experiment guide](code/README.md) lists the selected configurations, input paths, prediction exports, and reported Codabench scores. The [capstone report](report/CS781-CP-Tejas-Shrivastava-231091-report.pdf) covers the architecture choices and comparisons across submissions.

## Running the notebooks

1. Open the phase guide below. Attach the official phase data and the foundation checkpoint named in the guide to a GPU-enabled Kaggle notebook. The [experiment guide](code/README.md) lists each selected export.
2. For pretraining, select the model folder and check the tokenizer against the [prepared-data manifest](code/phase1/shared/prepared_data_manifest.json). For downstream tasks, use the Phase 1 foundation required by that notebook. The [foundation checkpoint manifest](code/phase1/checkpoint_manifest.json) and [NER checkpoint manifest](code/phase2/checkpoint_manifest.json) identify the archived weights.
3. Follow the phase-specific run order for setup, training or restore, validation and test export. Each submission ZIP contains `prediction.json` at its root. Saved validation predictions and the phase evaluator can be used independently of training.
4. Run local scoring from the corresponding phase folder. Phases 2–5 and 7 use the Python standard library. Phases 1 and 6 use `torch` and `bert-score`; Phase 6 also uses `rouge-score`. The first BERTScore evaluation requires access to its English scoring model.

## Architectures, methods and submissions

| Phase guide | Best finished submission | Method | Kaggle |
| --- | --- | --- | --- |
| [1. Legal language modelling](code/phase1/README.md) | 0.8307 BERTScore F1 | LEXWeave continuation and decode-v2 fallback | [RelayLSTM](https://www.kaggle.com/code/tejass73/relaylstmp1) · [BridgeFormer](https://www.kaggle.com/code/tejass73/bridgp1) |
| [2. Legal named entity recognition](code/phase2/README.md) | 0.4957 strict macro-F1 | LEXWeave with rank-16 LoRA and a CRF head | [phase2n1](https://www.kaggle.com/code/tejass73/phase2n1) |
| [3. Rhetorical role prediction](code/phase3/README.md) | 0.4792 macro-F1 | Sentence-role classification with LoRA | [phase3n](https://www.kaggle.com/code/tejass73/phase3n) |
| [4. Court judgment prediction](code/phase4/README.md) | 0.6752 macro-F1 | Full fine-tuning with a dev-selected decision threshold | [phase4n](https://www.kaggle.com/code/tejass73/phase4n) |
| [5. Legal statute identification](code/phase5/README.md) | 0.2685 macro-F1 | Full fine-tuning with multilabel threshold calibration | Notebook link pending |
| [6. Legal summarization](code/phase6/README.md) | 0.8276 BERTScore F1 | LoRMA+ with summary-only training loss | [phase6n](https://www.kaggle.com/code/tejass73/phase6n) |
| [7. Prior case retrieval](code/phase7/README.md) | 0.3538 micro-F1@7 | LoRMA dense retrieval with full-text TF-IDF and self-match filtering | [phase7n](https://www.kaggle.com/code/tejass73/phase7n) |

## Experiment records

Open a phase guide in the table above for submission IDs, predictions, local evaluator commands and links to saved run metadata. The local evaluators compute the primary metric on labelled validation examples. Codabench scores come from the submitted test runs.

The Phase 1, 2, 3, 4, 6 and 7 guides incorporate saved Kaggle session outputs. Phase 5 includes the training notebook and six submission exports. Its saved training session is still missing. Each run manifest identifies the available source files and records hashes for recovered prediction exports and available checkpoints. Kaggle links above point to notebook pages.

Phase 1 v2 was the highest finished submission at 0.8307. Codabench displayed 0.8312 for v3 with status Failed.
