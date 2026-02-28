# Getting Started

Follow these steps to get your environment ready for extracting features and generating plots.

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

## 2. Directory Structure

Ensure your data is positioned correctly before running extraction endpoints:

* **Raw Data:** Put your initial Twitter `.jsonl` or `.csv` files into `data/raw/`
* **Processed Data:** Feature tables (`.csv`) get generated into `data/processed/`
* **Reports:** Final Pearson correlation CSVs and summary JSONs are generated to `reports/`
* **Figures:** Visualizations and Heatmaps are output to `reports/figures/`
