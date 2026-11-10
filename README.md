# CS781 Legal Language Modelling

Tejas Shrivastava · CS781 capstone

Phase 1 explores four models for legal text continuation: LEXWeave, BridgeFormer, AnchorGPT and RelayLSTM. The six downstream tasks use LEXWeave for entity recognition, rhetorical role prediction, judgment prediction, statute identification, summarization and prior-case retrieval. The [capstone report](report/CS781-CP-Tejas-Shrivastava-231091-report.pdf) describes the architectures and experiments.

## Architectures, methods and submissions

| Phase guide | Best finished submission | Method | Notebook |
| --- | --- | --- | --- |
| [1. Legal language modelling](code/phase1/README.md) | 0.8307 BERTScore F1 | LEXWeave continuation and decode-v2 fallback | [RelayLSTM](https://www.kaggle.com/code/tejass73/relaylstmp1) · [BridgeFormer](https://www.kaggle.com/code/tejass73/bridgp1) |
| [2. Legal named entity recognition](code/phase2/README.md) | 0.4957 strict macro-F1 | LEXWeave with rank-16 LoRA and a CRF head | [phase2n1](https://www.kaggle.com/code/tejass73/phase2n1) |
| [3. Rhetorical role prediction](code/phase3/README.md) | 0.4792 macro-F1 | Sentence-role classification with LoRA | [phase3n](https://www.kaggle.com/code/tejass73/phase3n) |
| [4. Court judgment prediction](code/phase4/README.md) | 0.6752 macro-F1 | Full fine-tuning with a dev-selected decision threshold | [phase4n](https://www.kaggle.com/code/tejass73/phase4n) |
| [5. Legal statute identification](code/phase5/README.md) | 0.2685 macro-F1 | Full fine-tuning with multilabel threshold calibration | [Statute identification](code/phase5/statute_identification.ipynb) |
| [6. Legal summarization](code/phase6/README.md) | 0.8276 BERTScore F1 | LoRMA+ with summary-only training loss | [phase6n](https://www.kaggle.com/code/tejass73/phase6n) |
| [7. Prior case retrieval](code/phase7/README.md) | 0.3538 micro-F1@7 | LoRMA dense retrieval with full-text TF-IDF and self-match filtering | [phase7n](https://www.kaggle.com/code/tejass73/phase7n) |

## Running the notebooks

Open the phase guide for its notebook, data layout and run order. Training uses a GPU-enabled Kaggle environment and the official Exploration-Lab/CS-781-Capstone splits. Each model uses its matching Phase 1 tokenizer and checkpoint. The [tokenizer manifest](code/phase1/shared/prepared_data_manifest.json) and [checkpoint manifest](code/phase1/checkpoint_manifest.json) identify those files.

Each test submission is a ZIP containing `prediction.json` at the root. The [submission index](code/README.md) links to the selected predictions for each phase. The tables in the phase guides give the Codabench submission IDs and scores.

For local evaluation, run the scoring command from its phase directory with labelled validation data. Phases 2–5 and 7 use the Python standard library. Phases 1 and 6 use `torch` and `bert-score`, and Phase 6 also uses `rouge-score`. BERTScore downloads its English scoring model on first use.
