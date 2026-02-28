# Feature Extraction

The `features.py` module exposes Typer CLI commands to execute various stages of the data pipeline. You can run these commands easily from the root.

## Available Commands

### 1. Extract Candidate Order

Extracts the positional placement of Candidates (Trump vs Biden) inside the poll voting options.
```bash
python -m bias_analysis.features candidate-order
```
*Outputs to:* `data/processed/candidate_order_features.csv`

---

### 2. Extract Formal vs Informal Appellatives

Utilizes `spaCy` NER to sift through the Tweet texts and poll options to extract the exact nickname/title used to refer to candidates, labelling them as Neutral, Formal (e.g., "President Trump", "Joe Biden"), or Informal (e.g., "Sleepy Joe", "Donnie").
```bash
python -m bias_analysis.features formal-vs-informal
```
*Outputs to:* `data/processed/formal_informal_appellatives.csv`

---

### 3. Analyze Political Leaning

Uses `facebook/bart-large-mnli` to parse the underlying political affiliation, conservative/liberal leaning, and pro/anti sentiment of the tweet.
```bash
# To run on the full dataset:
python -m bias_analysis.features political-leaning

# To limit to X samples for testing:
python -m bias_analysis.features political-leaning --max-samples 100
```
*Outputs to:* `data/processed/political_leaning_features.csv`

---

### 4. Pearson Correlation Analysis

Aggregates the extracted features above to compute author bias, audience bias, and poll outcomes, followed by a rigorous multi-marker Pearson Correlation using bootstrapping and FDR-correction.
```bash
python -m bias_analysis.features pearson-correlation
```
*Outputs to:* 
* `reports/pearson_correlation_results.csv`
* `reports/correlation_analysis_summary.json`
* `reports/figures/assumption_check_distributions.png`
* `reports/figures/final_correlation_heatmap.png`
