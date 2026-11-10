# Phase 1 · Legal continuation

| Architecture | Notebook | Experiment |
| --- | --- | --- |
| **LEXWeave (Lexy)** | [Main model and continuation training](model1_lexweave/lexweave.ipynb) | Custom causal decoder. This notebook includes the model definition and continues training from the LEXWeave foundation export. |
| BridgeFormer | [Kaggle training notebook](model2_bridgeformer/bridgeformer.ipynb) | Encoder–decoder with a bidirectional prompt encoder and gated cross-attention in the continuation decoder. |
| AnchorGPT | [Training notebook](model3_anchor_gpt/anchor_gpt.ipynb) | GPT-style causal decoder trained from random weights. |
| RelayLSTM | [Training notebook](model4_relay_lstm/relay_lstm.ipynb) | Recurrent hybrid trained from random weights. |

Each notebook includes its model definition and experiment settings. Start with the setup and path configuration cells for the selected architecture. The [merge script](merge.py) combines saved primary and fallback generations for the LEXWeave v2 submission.

## Kaggle inputs and run order

1. Choose the architecture in the table above. For LEXWeave, attach its foundation export and set `FOUNDATION_PATH` in section 1. BridgeFormer, AnchorGPT and RelayLSTM have their own model, cache and backup settings.
2. Run setup and corpus preparation before the model cells. The [shared preparation notebook](shared/prepare_corpus.ipynb) contains the RelayLSTM data stages. The [prepared-data manifest](shared/prepared_data_manifest.json) identifies the matching tokenizers and links the LEXWeave raw and packed manifests by path.
3. In LEXWeave, run sections 5–8 to define, check, benchmark and train the model. Sections 9–12 restore the best checkpoint, inspect held-out continuations, generate predictions and export artifacts. The other three notebooks follow the same sequence under their own section headings.
4. Use the matching checkpoint and tokenizer for generation. The [checkpoint manifest](checkpoint_manifest.json) records the original archive member names, sizes and SHA-256 digests. To reproduce the historical v2 merge, use the primary generations, decode-v2 fallback and official test prompts with `python merge.py primary.json fallback.zip test.json v2_submission.zip`.

The [RelayLSTM generation settings](model4_relay_lstm/generation/manifest.json) identify the checkpoint and decoding configuration. Its [generation log](model4_relay_lstm/generation/generation_info.json) records all 1,000 case IDs.

## Results

[LEXWeave foundation metrics](model1_lexweave/metrics.json) record 95,232,768 parameters and checkpoint step 5840. The [BridgeFormer history](model2_bridgeformer/history.jsonl) and [RelayLSTM history](model4_relay_lstm/history.jsonl) record their training trajectories. The [AnchorGPT v2 predictions](model3_anchor_gpt/submissions/v2/prediction.json) accompany its notebook.

The [v2 submission](model1_lexweave/submissions/v2/prediction.json) combines primary generations for case IDs 1–249 with the [decode-v2 fallback](model1_lexweave/submissions/decode_v2/prediction.json). All 196 entries in the [saved primary file](model1_lexweave/primary_partial.json) match the corresponding submitted continuations.

[Run and checkpoint manifest](checkpoint_manifest.json) · [Archived Kaggle cell source](session_source.py)

## Local evaluation

Reports English BERTScore F1, whitespace-token ROUGE-L F1 and unsmoothed corpus BLEU-4.

Both JSON files map case IDs to objects with a `text` field. Install `torch` and `bert-score` for English BERTScore F1. The first run downloads the scoring model. Run from this phase directory.

```sh
python evaluate_phase1.py --truth truth.json --prediction prediction.json
```

## Codabench submissions

| Submission ID | File name | Date shown | Status | Score shown | Predictions |
| ---: | --- | --- | --- | ---: | --- |
| 965457 | `v3_submission.zip` | 2026-10-06 23:53 | Failed | 0.8312 | [JSON](model1_lexweave/submissions/v3/prediction.json) |
| 965367 | `v2_submission.zip` | 2026-10-06 23:16 | Finished | **0.8307** | [JSON](model1_lexweave/submissions/v2/prediction.json) |
| 964365 | `bridge_200m_full_v1_cd0e257a4203.zip` | 2026-10-06 15:03 | Finished | 0.8067 | [JSON](model2_bridgeformer/submission/prediction.json) |
| 963879 | `relay_lstm_200m_full_v1_7fd2769a59d.zip` | 2026-10-06 11:10 | Finished | 0.8250 | [JSON](model4_relay_lstm/prediction.json) |
| 962399 | `anchor_gpt_150m_full_v2_submission_.zip` | 2026-10-05 18:22 | Finished | 0.8286 | [JSON](model3_anchor_gpt/submissions/v2/prediction.json) |
| 961282 | `anchor_gpt_150m_full_v1_849b3d6003c.zip` | 2026-10-05 10:01 | Finished | 0.8240 | [JSON](model3_anchor_gpt/submissions/v1/prediction.json) |
| 958463 | `submission_lexweave_decode_v2.zip` | 2026-10-03 17:26 | Finished | 0.8300 | [JSON](model1_lexweave/submissions/decode_v2/prediction.json) |
