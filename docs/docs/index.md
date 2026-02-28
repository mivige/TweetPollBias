# TweetPollBias Analysis

Welcome to the **TweetPollBias** documentation!

## Project Overview

This repository conducts a deep social-poll analysis of the 2020 US Presidential Election cycle on Twitter. Our primary goal is to extract **bias markers**, such as candidate ordering, formal vs. informal appellative usage, and political leaning, and study how these markers relate to **audience bias** and the final **poll outcome bias**.

## Core Capabilities

1. **Foundational Datasets:** Cleans and processes raw Twitter datasets.
2. **Feature Extraction:** NLP extraction of:
   * **Candidate Order:** How poll option placement affects votes.
   * **Appellatives:** Measuring formality bias using `spaCy` NER.
   * **Political Leaning:** Zero-shot classification powered by `facebook/bart-large-mnli` (with CUDA FP16 support).
3. **Statistical Correlation:** End-to-end Pearson correlation with False Discovery Rate (FDR) corrections.
4. **Rich Visualizations:** Scatter plots and correlation heatmaps.

Use the sidebar navigation to read about getting setup, extracting features, and generating your plots.
