# Phase 1 · Legal continuation

| Architecture | Notebook | Experiment |
| --- | --- | --- |
| **LEXWeave (Lexy)** | [Main model and continuation training](lexweave.ipynb) | Custom causal decoder. This notebook includes the model definition and continues training from the LEXWeave foundation export. |
| BridgeFormer | [Kaggle training notebook](bridgeformer.ipynb) | Encoder–decoder with a bidirectional prompt encoder and gated cross-attention in the continuation decoder. |
| AnchorGPT | [Training notebook](anchor_gpt.ipynb) | GPT-style causal decoder trained from random weights. |
| RelayLSTM | [Training notebook](relay_lstm.ipynb) | Recurrent hybrid trained from random weights. |

Each notebook includes its model definition and experiment settings. Start with the setup and path configuration cells for the selected architecture. The [merge script](merge.py) combines saved primary and fallback generations for the LEXWeave v2 submission. The [experiment guide](../README.md) gives the command and expected inputs.

## Experiment records

[LEXWeave foundation metrics](runs/lexweave/metrics.json) record 95,232,768 parameters and checkpoint step 5840. The [BridgeFormer history](runs/bridgeformer/history.jsonl) and [RelayLSTM history](runs/relay_lstm/history.jsonl) record their training trajectories. The [AnchorGPT v2 predictions](submissions/962399/prediction.json) accompany its notebook. An AnchorGPT checkpoint and training history were unavailable.

The archived v2 merge used 249 primary case IDs. All 196 [recovered primary records](runs/lexweave/primary_partial.json) match the [submitted v2 output](submissions/965367/prediction.json). The [decode-v2 fallback](submissions/958463/prediction.json) is available. The other primary records are needed to replay the full merge from its original inputs.

[Run and checkpoint manifest](run_manifest.json) · [Archived Kaggle cell source](session_source.py)

## Local evaluation

Both JSON files map case IDs to objects with a `text` field. Install `torch` and `bert-score` for English BERTScore F1. The first run downloads the scoring model. Run from this phase directory.

```sh
python evaluate_phase1.py --truth truth.json --prediction prediction.json
```

## Codabench submissions

| Submission ID | File name | Date shown | Status | Score shown | Predictions |
| ---: | --- | --- | --- | ---: | --- |
| 965457 | `v3_submission.zip` | 2026-10-06 23:53 | Failed | 0.8312 | [JSON](submissions/965457/prediction.json) |
| 965367 | `v2_submission.zip` | 2026-10-06 23:16 | Finished | **0.8307** | [JSON](submissions/965367/prediction.json) |
| 964365 | `bridge_200m_full_v1_cd0e257a4203.zip` | 2026-10-06 15:03 | Finished | 0.8067 | [JSON](submissions/964365/prediction.json) |
| 963879 | `relay_lstm_200m_full_v1_7fd2769a59d.zip` | 2026-10-06 11:10 | Finished | 0.8250 | [JSON](runs/relay_lstm/prediction.json) |
| 962399 | `anchor_gpt_150m_full_v2_submission_.zip` | 2026-10-05 18:22 | Finished | 0.8286 | [JSON](submissions/962399/prediction.json) |
| 961282 | `anchor_gpt_150m_full_v1_849b3d6003c.zip` | 2026-10-05 10:01 | Finished | 0.8240 | [JSON](submissions/961282/prediction.json) |
| 958463 | `submission_lexweave_decode_v2.zip` | 2026-10-03 17:26 | Finished | 0.8300 | [JSON](submissions/958463/prediction.json) |
