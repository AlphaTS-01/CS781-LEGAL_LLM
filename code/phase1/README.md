# Phase 1 · Legal continuation

| Architecture | Notebook | Experiment |
| --- | --- | --- |
| **LEXWeave (Lexy)** | [Main model and continuation training](lexweave.ipynb) | Custom causal decoder. This notebook includes the model definition and continues training from the LEXWeave foundation export. |
| BridgeFormer | [Kaggle training notebook](bridgeformer.ipynb) | Encoder–decoder with a bidirectional prompt encoder and gated cross-attention in the continuation decoder. |
| AnchorGPT | [Training notebook](anchor_gpt.ipynb) | GPT-style causal decoder trained from random weights. |
| RelayLSTM | [Training notebook](relay_lstm.ipynb) | Recurrent hybrid trained from random weights. |

Each notebook includes its model definition and experiment settings. Start with the setup and path configuration cells for the selected architecture. The [merge script](merge.py) combines saved primary and fallback generations for the LEXWeave v2 submission. The [experiment guide](../README.md) gives the command and expected inputs.

## Codabench submissions

| Submission ID | File name | Date shown | Status | Score shown |
| ---: | --- | --- | --- | ---: |
| 965457 | `v3_submission.zip` | 2026-10-06 23:53 | Failed | 0.8312 |
| 965367 | `v2_submission.zip` | 2026-10-06 23:16 | Finished | **0.8307** |
| 964365 | `bridge_200m_full_v1_cd0e257a4203.zip` | 2026-10-06 15:03 | Finished | 0.8067 |
| 963879 | `relay_lstm_200m_full_v1_7fd2769a59d.zip` | 2026-10-06 11:10 | Finished | 0.8250 |
| 962399 | `anchor_gpt_150m_full_v2_submission_.zip` | 2026-10-05 18:22 | Finished | 0.8286 |
| 961282 | `anchor_gpt_150m_full_v1_849b3d6003c.zip` | 2026-10-05 10:01 | Finished | 0.8240 |
| 958463 | `submission_lexweave_decode_v2.zip` | 2026-10-03 17:26 | Finished | 0.8300 |
