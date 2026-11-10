# CS781 capstone submission

Phase 1 includes four architecture notebooks. Start with [LEXWeave (Lexy)](phase1/lexweave.ipynb) for the main model. Phases 2 through 7 each use `run.ipynb`. Open the notebook in a CUDA-enabled Jupyter environment, set its data and foundation paths in the configuration section, and run the cells in order. Use the official Exploration-Lab/CS-781-Capstone splits and a fresh working directory for each phase. Downloads from the gated dataset require an authorized account. The phase guides link to recovered training records, submission predictions and local evaluators. Large checkpoint files stay in the named source archives. Their SHA-256 digests appear in the run manifests.

| Phase | Selected configuration | Finished submission | Score |
| --- | --- | --- | ---: |
| 1 | LEXWeave continuation and fallback merge | v2_submission.zip | 0.8307 |
| 2 | E3 LoRA, CRF head | e3_lora.zip | 0.4957 |
| 3 | E2 LoRA | e2lora.zip | 0.4792 |
| 4 | E0 full fine-tuning | e0_p4.zip | 0.6752 |
| 5 | Full fine-tuning | fullp5.zip | 0.2685 |
| 6 | LoRMA+ | lormap6.zip | 0.8276 |
| 7 | LoRMA, full-text TF-IDF and self-match filtering | lorma_confirmed.zip | 0.3538 |

## Inputs and exports

1. **Continuation.** Attach the original LEXWeave foundation export, including its tokenizer and manifests, then run the continuation notebook. The original configuration and tokenizer manifest are included for identification. The historical v2 archive used primary generations through case 249 and the earlier decode-v2 output as fallback. Run `python merge.py primary.json fallback.zip test.json v2_submission.zip` in phase1. The reconstructed merge was checked against all 1,000 archived v2 outputs using the overlapping saved primary generations. A fresh training run can generate different text.
2. **NER.** Attach the LEXWeave checkpoint, tokenizer and model definition together, then run the [E3 notebook](phase2/run.ipynb). Its predictions match submission 977791. The [phase guide](phase2/README.md) documents the original and continued LoRA runs.
3. **Rhetorical roles.** Point CS781_FOUNDATION to the LEXWeave archive and CS781_WORK to the working directory. E2 uses rank-16 additive LoRA on the sentence-role model. Its experiment directory contains submission.zip.
4. **Judgment outcomes.** Use the same foundation settings as Phase 3 with a separate CS781_WORK. E0 trains all model parameters and exports one case-level decision per test case.
5. **Statutes.** Set LOCAL_DATA to Phase_5_LSI, or set it to None for the dataset download. Attach the foundation to the Kaggle notebook. Full tuning uses the original document and time caps, followed by threshold calibration. Retrieve runs/full/submission.zip from WORK.
6. **Summaries.** Set LOCAL_DATA to Phase_6_Summ or use authenticated download. Attach the foundation. The selected run writes its summary export under runs/lorma.
7. **Retrieval.** Attach the foundation and Phase_7_PCR inputs. The last cell combines the dense top seven and the full-text top fifty with lexical weight two, removes the query ID, and writes runs/lorma/lorma_confirmed.zip. The archived weight-two, self-filtered predictions were checked against the confirmed submission and matched for every query.
