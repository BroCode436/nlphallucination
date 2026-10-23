# Reproduction from a Git checkout

Install `requirements_chc_v2.txt` in a Python 3.12 virtual environment. The experiment environment used PyTorch 2.11.0 with CUDA 12.8, Transformers 5.19.0, scikit-learn 1.9.1 and joblib 1.5.3; saved sklearn/joblib models may be incompatible with other versions. Use `outputs/chc_v2/environment.json` for exact recorded versions. CUDA-enabled PyTorch must match the machine/driver; CPU operation is slower.

Run software fixtures with `python -m pytest tests_chc tests test_model.py test_integration.py`. These fixtures do not validate research accuracy. The development release passed 109 tests. Live/transformer tests may skip when external checkpoints are absent. Full real inference needs the frozen models identified by snapshot in the paper and manifests; do not confuse a skipped test with successful inference.

Recorded results and the 31 small policy checkpoints are included under `outputs/chc_v2/pilot`. The six graph `.pt` files are deliberately included despite the general large-weight ignore rule. Intermediate optimizer states, model-download caches, backups and development-run duplicates remain local. The original correction-validity joblib is included. Historical DistilBERT/FLAN weights and RAGTruth processed data must be obtained or regenerated separately; this source checkout is not a complete offline distribution of every dependency.

Set `CHC_V2_CACHE` to a writable model/cache directory, otherwise the application uses `outputs/chc_v2/cache`. Set `CHC_V2_RESULTS` only when choosing a different recorded experiment directory. `python app.py --port 8767` serves the old verifier and `/chc/propagate`; the PowerShell launcher is optional.

Source rows for the bounded HotpotQA protocol are included in `data/chc_v2` with attribution in `DATA_NOTICE.md`. To rerun generation into a fresh directory:

```powershell
$env:CHC_V2_CACHE='outputs/chc_v2/cache'
python scripts/run_chc_v2.py --cache $env:CHC_V2_CACHE --hotpot data/chc_v2 --output outputs/chc_v2/reproduction --stage pilot --seeds 101
python scripts/train_chc_v2.py --cache $env:CHC_V2_CACHE --stage outputs/chc_v2/reproduction --judge-device cpu
python scripts/finalize_chc_v2.py --cache $env:CHC_V2_CACHE --stage outputs/chc_v2/reproduction
```

These commands default to NLI risk when no detector report is supplied. To reproduce the reported calibrated-risk condition exactly, provide `--detector-report outputs/chc_v2/detector/detector_report.json` during collection and obtain the original DistilBERT checkpoint. Optional historical transfer comparisons additionally require `--historical` pointing to the previous CHC policy checkpoint directory. Do not claim these optional conditions were reproduced when the files are missing.

The recorded model source hashes and training fingerprints describe the original experiment. Do not overwrite frozen results when changing a fitting recipe. Create a new output directory; training reuse checks intentionally reject incompatible inputs/configuration. Exact floating-point outputs can differ across hardware and dependency versions.

The paper and reports distinguish automatic judgments from human review. No independent human factual/repair labels have been collected. Keep the negative answer-accuracy finding, eligibility coverage and live failure analysis in any derivative report.
