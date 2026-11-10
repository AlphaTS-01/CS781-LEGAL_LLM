# Selected submissions

The table lists the highest-scoring finished submission for each phase. Each filename links to the predictions inside the submitted ZIP.

| Phase | Configuration | Submission ID | Prediction export | Codabench score |
| --- | --- | ---: | --- | ---: |
| [1. Legal continuation](phase1/README.md) | LEXWeave continuation with decode-v2 fallback | 965367 | [v2_submission.zip](phase1/model1_lexweave/submissions/v2/prediction.json) | 0.8307 |
| [2. Named entity recognition](phase2/README.md) | E3 LoRA with a CRF head | 977791 | [e3_lora.zip](phase2/model1_lexweave/E3_lora_M1_lexweave/prediction.json) | 0.4957 |
| [3. Rhetorical roles](phase3/README.md) | E2 LoRA | 993270 | [e2lora.zip](phase3/runs/E2_lora/prediction.json) | 0.4792 |
| [4. Judgment outcome](phase4/README.md) | E0 full fine-tuning | 1007383 | [e0_p4.zip](phase4/runs/E0_full/prediction.json) | 0.6752 |
| [5. Statute identification](phase5/README.md) | Full fine-tuning | 1012181 | [fullp5.zip](phase5/runs/full/prediction.json) | 0.2685 |
| [6. Summarization](phase6/README.md) | LoRMA+ | 1029346 | [lormap6.zip](phase6/runs/lorma/prediction.json) | 0.8276 |
| [7. Prior case retrieval](phase7/README.md) | LoRMA with full-text TF-IDF and self-match filtering | 1040965 | [lorma_confirmed.zip](phase7/final_predictions/fulltext_w2.0_exclude_self_True/prediction.json) | 0.3538 |
