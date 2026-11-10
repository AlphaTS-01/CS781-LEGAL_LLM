# CS781 · Legal Language Modelling

Tejas Shrivastava · CS781 capstone

This project trains compact language models on legal text and adapts them for six downstream tasks. The [experiment guide](code/README.md) lists the selected configurations, input paths, prediction exports, and reported Codabench scores. The [capstone report](report/CS781-CP-Tejas-Shrivastava-231091-report.pdf) covers the architecture choices and comparisons across submissions.

| Phase | Submitted code | Kaggle experiments | Best finished score |
| --- | --- | --- | ---: |
| 1 · Legal continuation | [Four architecture notebooks and prediction merge](code/phase1/README.md) | [Relay LSTM](https://www.kaggle.com/code/tejass73/relaylstmp1) · [Bridge](https://www.kaggle.com/code/tejass73/bridgp1) | [0.8307](code/phase1/README.md#codabench-submissions) |
| 2 · Named entity recognition | [NER notebook](code/phase2/run.ipynb) | [phase2n1](https://www.kaggle.com/code/tejass73/phase2n1) | [0.4957](code/phase2/README.md#codabench-submissions) |
| 3 · Rhetorical role prediction | [Role classification notebook](code/phase3/run.ipynb) | [phase3n](https://www.kaggle.com/code/tejass73/phase3n) | [0.4792](code/phase3/README.md#codabench-submissions) |
| 4 · Judgment outcome prediction | [Judgment prediction notebook](code/phase4/run.ipynb) | [phase4n](https://www.kaggle.com/code/tejass73/phase4n) | [0.6752](code/phase4/README.md#codabench-submissions) |
| 5 · Statute identification | [Statute identification notebook](code/phase5/run.ipynb) | Link pending | [0.2685](code/phase5/README.md#codabench-submissions) |
| 6 · Legal summarization | [Summarization notebook](code/phase6/run.ipynb) | [phase6n](https://www.kaggle.com/code/tejass73/phase6n) | [0.8276](code/phase6/README.md#codabench-submissions) |
| 7 · Prior case retrieval | [Retrieval and reranking notebook](code/phase7/run.ipynb) | [phase7n](https://www.kaggle.com/code/tejass73/phase7n) | [0.3538](code/phase7/README.md#codabench-submissions) |

## Running a phase

Open the phase notebook in a GPU-enabled Jupyter environment. Attach the official phase data and the foundation checkpoint specified in the experiment guide, configure the input and working paths, then run the cells in order. Each phase uses its own output directory.

## Experiment records

Open a phase's score in the table above for submission IDs, predictions, local evaluator commands and links to saved run metadata. The local evaluators compute the primary metric on labelled validation examples. Codabench scores come from the submitted test runs.

The Phase 1, 2, 3, 4, 6 and 7 guides incorporate saved Kaggle session outputs. Phase 5 includes the training notebook and six submission exports. Its saved training session is still missing. Each run manifest identifies the available source files and records hashes for recovered prediction exports and available checkpoints. Kaggle links above point to notebook pages.

Phase 1 v2 was the highest finished submission at 0.8307. Codabench displayed 0.8312 for v3 with status Failed.
