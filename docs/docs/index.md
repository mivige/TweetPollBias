# TweetPollBias Analysis

Welcome to the **TweetPollBias** documentation!

## Project Overview

This repository conducts a deep social-poll analysis of multiple electoral cycles on Twitter (e.g., US 2016, US 2020, US 2024). Our primary goal is to extract **bias markers**, such as candidate ordering, formal vs. informal appellative usage and political leaning, and study how these markers relate to **audience bias** and the final **poll outcome bias** using Multilevel Regression and Poststratification (MRP).

## Core Capabilities

1. **Config-Driven Architecture:** Dynamically define datasets, NLI semantic matching profiles, demographic priors, and key milestones on a per-election basis in `election_configs.py`.
2. **Feature Extraction:** NLP extraction of:
   * **Candidate Order:** How poll option placement affects votes.
   * **Appellatives:** Measuring formality bias using `spaCy` NER.
   * **Political Leaning:** Zero-shot classification powered by `facebook/bart-large-mnli` (with CUDA FP16 support).
3. **End-to-End Pipeline:** A unified runner script (`run_pipeline.py`) that seamlessly executes feature extraction, MRP modeling, and interactive visualization dashboard generation for any registered election in one go.
4. **Rich Visualizations:** Interactive MRP tracking dashboards via HTML, alongside Pearson correlation heatmaps and KDE scatter plots.

Use the sidebar navigation to read about getting setup, extracting features, and generating your plots.
