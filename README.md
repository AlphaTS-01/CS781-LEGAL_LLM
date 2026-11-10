# CS781 Legal NLP Capstone

Tejas Shrivastava · CS781

The project starts with a legal text continuation task. Given the opening of a passage, each model generates the next 128 words. Phase 1 compares four architectures. The six downstream tasks all start from **LEXWeave**, the 95.2-million-parameter causal decoder developed for this project.

LEXWeave combines three ways of reading the preceding text. Self-attention reads earlier tokens directly. A second attention branch reads summaries of completed 32-token chunks. A convolution branch picks up phrase patterns at three different spacings. At each token, a learned gate controls how much each branch contributes to the next layer. The same pretrained decoder can then supply token representations for tagging, document representations for classification and retrieval, or next-token predictions for summarization. [Model implementation](code/phase2/model1_lexweave/lexweave_model.py) · [Capstone report](report/CS781-CP-Tejas-Shrivastava-231091-report.pdf)

## Running the notebooks

1. Open the phase guide below and import its experiment notebook into Kaggle. Enable a GPU for training and generation.
2. Attach the official `Exploration-Lab/CS-781-Capstone` data with its phase and split directories intact. For notebooks that download the data, enable Internet access and supply an authorized Hugging Face token through `HF_TOKEN`.
3. For Phase 1, select the model and its matching tokenizer. For downstream experiments, attach the LEXWeave foundation export. The NER notebook expects `model.pt`, `tokenizer.json` and `lexweave_model.py` in the same directory. Each phase guide specifies the inputs for its run. [Checkpoint manifest](code/phase1/checkpoint_manifest.json) · [Tokenizer manifest](code/phase1/shared/prepared_data_manifest.json)
4. Run the setup, data and model-definition cells before the experiment cells. Write outputs to a working directory such as `/kaggle/working`. To resume training, restore the checkpoint and configuration described in the phase guide.
5. Use the test export from the selected run. Each submission ZIP contains `prediction.json` at the root. The [submission index](code/README.md) links the selected predictions to their Codabench IDs.

For local scoring, run the evaluator from its phase directory with predictions and references from the same validation split. Phases 2–5 and 7 use the Python standard library. The continuation and summarization evaluators use BERTScore, and summarization also uses ROUGE. Install the scoring dependencies with the following command.

```sh
python -m pip install torch bert-score rouge-score
```

The first BERTScore evaluation downloads its English scoring model.

## Architecture, methods, and submissions

The other three Phase 1 models carry context through a passage in different ways. **BridgeFormer** encodes the full prompt before decoding the continuation. Its decoder attends to both the original prompt tokens and a shorter sequence of learned representations. **AnchorGPT** is a causal Transformer that feeds the representation from the start of each four-layer group back into its blocks through a learned gate. **RelayLSTM** carries a recurrent state through each token sequence and adds causal attention every third block. [Phase 1 notebooks and runs](code/phase1/README.md)

| Phase guide | Best finished submission | Architecture and method |
| --- | --- | --- |
| [1. Legal language model pretraining](code/phase1/README.md) | LEXWeave v2 · **0.8307 BERTScore F1** | Continue training the decoder on legal text and generate a 128-word continuation. The v2 export combines the primary generation pass for cases 1–249 with predictions from an earlier decoding run. |
| [2. Legal named entity recognition](code/phase2/README.md) | E3 LoRA + CRF · **0.4957 strict macro-F1** | Train rank-16 low-rank updates to the decoder. A conditional random field decodes a sequence of entity tags. The notebook converts these tags to character offsets and entity labels for exact-span scoring. |
| [3. Rhetorical role prediction](code/phase3/README.md) | E2 LoRA · **0.4792 macro-F1** | Pack consecutive sentences into overlapping windows so each sentence can be classified in context. Adapt the decoder with rank-16 LoRA and predict a role such as fact, issue or ruling for each sentence. |
| [4. Court judgment prediction](code/phase4/README.md) | E0 full fine-tuning · **0.6752 macro-F1** | Fine-tune the whole decoder on up to four 512-token windows from the end of a judgment. Attention pools the window representations into a case-level prediction. Select the binary decision threshold on the development set. |
| [5. Legal statute identification](code/phase5/README.md) | Full fine-tuning · **0.2685 macro-F1** | Sample windows across a judgment, average token representations within each window and use attention to combine the windows. Score each statute independently and calibrate the thresholds so a case can receive multiple labels. |
| [6. Abstractive legal summarization](code/phase6/README.md) | LoRMA+ · **0.8276 BERTScore F1** | Condition the decoder on passages from the beginning, middle and end of the judgment. Train low-rank updates on the outputs of frozen linear layers and compute training loss only on the reference summary tokens. |
| [7. Prior case retrieval](code/phase7/README.md) | LoRMA + full-text TF-IDF · **0.3538 micro-F1@7** | Combine the model’s top seven cases with the top fifty from full-text TF-IDF. Give the lexical ranking twice the weight in reciprocal-rank fusion, remove the query case itself and return seven prior cases. |

Kaggle notebooks: [RelayLSTM](https://www.kaggle.com/code/tejass73/relaylstmp1) · [BridgeFormer](https://www.kaggle.com/code/tejass73/bridgp1) · [Entity recognition](https://www.kaggle.com/code/tejass73/phase2n1) · [Rhetorical roles](https://www.kaggle.com/code/tejass73/phase3n) · [Judgment prediction](https://www.kaggle.com/code/tejass73/phase4n) · [Summarization](https://www.kaggle.com/code/tejass73/phase6n) · [Retrieval](https://www.kaggle.com/code/tejass73/phase7n)
