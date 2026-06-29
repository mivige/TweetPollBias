# Feature Extraction

The `features.py` module exposes Typer CLI commands to execute various stages of the data pipeline. You can run these commands easily from the root, or use the unified `run_pipeline.py` script to do it all at once.

## Running the Full Pipeline

The easiest way to extract all features, build the SCWG models, and generate dashboards is using the unified runner:
```bash
python run_pipeline.py --election us24
```
You can skip specific phases using flags like `--skip-modeling` or `--skip-plots`.

---

## Individual Step Commands

If you need to run granular feature extraction, you can call them directly. Make sure to specify the `--election` code (e.g., `us20`, `us24`, `us16`).

### 1. Extract Candidate Order

Extracts the positional placement of Candidates inside the poll voting options.
```bash
python -m bias_analysis.features candidate-order --election us24
```
*Outputs to:* `data/processed/<election>/candidate_order_features.csv`

---

### 2. Extract Formal vs Informal Appellatives

Utilizes `spaCy` NER to sift through the Tweet texts and poll options to extract the exact nickname/title used to refer to candidates, labelling them as Neutral, Formal (e.g., "President Trump", "Joe Biden"), or Informal (e.g., "Sleepy Joe", "Donnie").
```bash
python -m bias_analysis.features formal-vs-informal --election us24
```
*Outputs to:* `data/processed/<election>/formal_informal_appellatives.csv`

---

### 3. Analyze Political Leaning

Uses `facebook/bart-large-mnli` to parse the underlying political affiliation, conservative/liberal leaning, and pro/anti sentiment of the tweet.
```bash
# To run on the full dataset for an election:
python -m bias_analysis.features political-leaning --election us24

# To limit to X samples for testing:
python -m bias_analysis.features political-leaning --election us24 --max-samples 100
```
*Outputs to:* `data/processed/<election>/political_leaning_features.csv`

---

### 4. Sentiment & Toxicity Analysis

Computes VADER sentiment scores and Detoxify toxicity scores for each poll tweet.
```bash
python -m bias_analysis.features sentiment-and-toxicity --election us24
```
*Outputs to:* `data/processed/<election>/sentiment_toxicity_features.csv`

---

### 5. Cognitive Bias Extraction

Uses the Gemini LLM to detect six cognitive biases in each poll (confirmation, anchoring, availability heuristic, social desirability, acquiescence, demand characteristics). Requires a `GEMINI_API_KEY` in your `.env` file.
```bash
python -m bias_analysis.llm_bias_extraction extract-biases --election us24
```
*Outputs to:* `data/processed/<election>/cognitive_biases.jsonl`

---

### 6. Pearson Correlation Analysis

Aggregates the extracted features above to compute author bias, audience bias, and poll outcomes, followed by a rigorous multi-marker Pearson Correlation using bootstrapping and FDR-correction.
```bash
python -m bias_analysis.features pearson-correlation --election us24
```
*Outputs to:* 

* `reports/<election>/pearson_correlation_results.csv`
* `reports/<election>/correlation_analysis_summary.json`
* `reports/figures/<election>/assumption_check_distributions.png`
* `reports/figures/<election>/final_correlation_heatmap.png`
