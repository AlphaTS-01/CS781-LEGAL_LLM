# Phase 1 · Legal continuation

| Architecture | Notebook | Experiment |
| --- | --- | --- |
| **LEXWeave (Lexy)** | [Main model and continuation training](lexweave.ipynb) | Custom causal decoder. This notebook includes the model definition and continues training from the LEXWeave foundation export. |
| BridgeFormer | [Kaggle training notebook](bridgeformer.ipynb) | Encoder–decoder with a bidirectional prompt encoder and gated cross-attention in the continuation decoder. |
| AnchorGPT | [Training notebook](anchor_gpt.ipynb) | GPT-style causal decoder trained from random weights. |
| RelayLSTM | [Training notebook](relay_lstm.ipynb) | Recurrent hybrid trained from random weights. |

Each notebook includes its model definition and experiment settings. Start with the setup and path configuration cells for the selected architecture. The [merge script](merge.py) combines saved primary and fallback generations for the LEXWeave v2 submission. The [experiment guide](../README.md) gives the command and expected inputs.
