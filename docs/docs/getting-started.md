# Getting Started

Follow these steps to get your environment ready for extracting features and generating plots across various elections.

## 1. Environment Requirements

This project utilizes advanced NLP libraries (like `spaCy` and `transformers`) which require specific runtime environments.

### SpaCy Models
Ensure you have the required `en_core_web_sm` model downloaded for the Named Entity Recognition (NER) step within the Appellatives extraction:
```bash
python -m spacy download en_core_web_sm
```

### PyTorch & CUDA
If you have an NVIDIA GPU, this repository will automatically detect and utilize **CUDA with FP16 precision** to massively accelerate the BART-MNLI political leaning classifications (on my hardware it reduced time from almost 2 hours to 18 minutes). For the best performance, ensure your PyTorch build matches your system's CUDA version.

---

## 2. Setting Up an Election & Modifying Configurations

Our architecture defines multiple election campaigns through `bias_analysis/election_configs.py`. To configure a new race, add your parameters into the `ELECTION_CONFIGS` dictionary mapping:

* **Candidates**: A list of targets (e.g. `["Trump", "Harris", "Biden"]`).
* **Semantic Filters**: Match phrases to capture NLI text embeddings and Appellatives correctly.
* **SCWG Demographics**: Calibrate ideological bases to ensure post-stratification correctly maps to reality.
* **Milestones**: Supply specific dashboard milestones like convention dates and debates.

### Directory Structure Requirements

Ensure your data is positioned correctly before running the endpoints. Path targets are isolated by the `--election` flag provided to the runners. For instance, if you establish `us24` in your config, create the corresponding `raw` data directories first:

* **Raw Data:** Put your initial Twitter `.jsonl` or `.csv` files into `data/raw/<election>/`
* **Processed Data:** Feature tables (`.csv`) get generated into `data/processed/<election>/`
* **Reports:** Final analyses and statistics are output to `reports/<election>/`
* **Figures:** Plot images and SCWG HTML dashboards are generated in `reports/figures/<election>/`

---

## 3. Running the Tests

The test suite requires no raw data as all tests use in-memory fixtures or temporary files. Run it from the project root:

```bash
make test
# or directly:
python -m pytest tests/ -v
```

The suite covers four modules:

| File | What it tests |
|---|---|
| `test_election_configs.py` | Config completeness, probability constraints, key consistency |
| `test_dataset.py` | All JSONL loaders, path helpers, PredictIt CSV parsing |
| `test_features.py` | `smart_candidate_match` across exact, fuzzy, regex, and edge cases |
| `test_train.py` | Post-stratification frame, GLM/OLS fitting, `poststratify` output |
