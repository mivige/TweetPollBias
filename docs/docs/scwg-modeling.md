# SCWG Modeling & Post-stratification

This project uses a **Sample-Calibrated Weighted GLM (SCWG)** to translate raw Twitter poll data into bias-adjusted electoral estimates.

## 1. What is SCWG?

SCWG is a frequentist pipeline that fits a Binomial GLM weighted by poll size, then post-stratifies the predictions over a calibrated population frame. In this project it is used to:

1. **Model** the relationship between individual poll characteristics (partisanship, gender, age, bias markers) and the "positive candidate" vote share in a Twitter poll.
2. **Post-stratify** the model's predictions by weighting them according to the actual demographic and partisan breakdown of the registered voting population, producing a bias-adjusted electoral estimate.

## 2. The Modeling Phase

For each election, two models are fit and compared:

* **Full GLM** – Binomial GLM with Logit link, weighted by `total_votes`, including all bias markers as predictors.
* **Baseline GLM** – same family/link but without bias markers, used as the reference for AIC comparison.

An **OLS reference model** (full and baseline) is also produced for interpretability.

### Predictors
* `audience_mean_partisanship`: Average political leaning of the poll's audience (retweeters/favoriters).
* `author_partisanship`: Political leaning of the poll's creator.
* `candidate_order`: Positional bias (positive candidate position minus negative candidate position, scaled).
* `formality_bias`: Formality differential of the appellatives used for each candidate (van den Berg 2019 scale).
* `positive_ideology_score`: NLI-inferred conservative leaning of the tweet text.
* `undirected_sentiment`, `sentiment_intensity`, `toxicity_score`: Sentiment and toxicity of the tweet.
* `bias_*`: Six cognitive bias indicators detected by the Gemini LLM (confirmation, anchoring, availability, social desirability, acquiescence, demand characteristics).

## 3. Post-stratification Frame

The calibration population is defined in `election_configs.py` for each election under the `"scwg"` key. Strata cross **partisan × ideological** dimensions:

* **Partisan Strata:** Republican, Democrat, Independent (with real-electorate weights).
* **Ideological Strata:** Conservative, Moderate, Liberal.
* **Demographic Profiles:** Gender and Age distributions within each partisan group.

The post-stratification frame uses a **zero-bias counterfactual** (all bias markers are set to 0) so the SCWG estimate reflects what the polls would have shown in a perfectly neutral environment.

Example configuration in `election_configs.py`:
```python
"scwg": {
    "partisan_strata": {"Republican": 0.33, "Democrat": 0.36, "Independent": 0.31},
    "ideological_strata": {"Conservative": 0.35, "Moderate": 0.39, "Liberal": 0.26},
    # ... demographic profiles, ideology offsets, actual results
}
```

## 4. The Interactive Dashboard

Results are exported to an HTML dashboard at `reports/<election>/scwg_bias_dashboard.html`.

### Dashboard Features
* **Time-Series Tracking:** Full SCWG estimate and baseline tracked over the campaign timeline.
* **Rolling Averages:** 7-day and 14-day rolling windows for trend smoothing.
* **Milestone Integration:** Vertical markers for debates, conventions, and major events.
* **Comparative Baselines:** Prediction market prices and actual election result for MAE calculation.
