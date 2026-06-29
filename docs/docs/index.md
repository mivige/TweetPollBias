# TweetPollBias Analysis

Welcome to the **TweetPollBias** documentation!

## Project Overview

This repository conducts a deep social-poll analysis of multiple electoral cycles on Twitter (US 2016, US 2020, US 2024). The primary goal is to extract **bias markers** (candidate ordering, formal vs. informal appellative usage, political leaning, sentiment, and cognitive biases) and study how these markers relate to **audience bias** and the final **poll outcome bias** using a Sample-Calibrated Weighted GLM (SCWG).

## Core Capabilities

1. **Config-Driven Architecture:** Dynamically define datasets, NLI semantic matching profiles, demographic priors, and key milestones on a per-election basis in `election_configs.py`.
2. **Feature Extraction:** NLP extraction of:
    * **Candidate Order:** How poll option placement affects votes.
    * **Appellatives:** Measuring formality bias using `spaCy` NER and the van den Berg (2019) naming taxonomy.
    * **Political Leaning:** Zero-shot classification powered by `facebook/bart-large-mnli` (with CUDA FP16 support).
    * **Sentiment & Toxicity:** VADER sentiment and Detoxify toxicity scoring per poll.
    * **Cognitive Biases:** LLM-based detection of six cognitive biases via Gemini API.
3. **End-to-End Pipeline:** A unified runner script (`run_pipeline.py`) that executes feature extraction, SCWG modeling, and interactive visualization dashboard generation for any registered election in one go.
4. **Rich Visualizations:** Interactive SCWG tracking dashboards via HTML, alongside Pearson correlation heatmaps and KDE scatter plots.

Use the sidebar navigation to read about getting set up, extracting features, and generating plots.
