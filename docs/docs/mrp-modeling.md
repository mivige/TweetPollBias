# MRP Modeling & Post-stratification

This project utilizes **Multilevel Regression and Post-stratification (MRP)** to translate raw Twitter user data into representative electoral estimates.

## 1. What is MRP?

MRP is a powerful statistical technique used to estimate preferences in small geographic areas or demographic sub-groups. In this project, we use it to:
1. **Model** the relationship between individual user characteristics (partisanship, gender, age) and their "vote" in a Twitter poll.
2. **Post-stratify** these estimates by weighting them according to the actual demographic and partisan breakdown of the registered voting population.

## 2. The Modeling Phase

For each election, we fit a **Generalized Linear Model (GLM)** with a Binomial distribution and Logit link.

### Predictors
The model uses several individual-level predictors extracted during the Feature Phase:
* `audience_mean_partisanship`: Average political leaning of the poll's audience.
* `author_partisanship`: Political leaning of the poll's creator.
* `candidate_order`: Positional bias (Target vs Opponent placement).
* `formality_bias`: Formality of the appellatives used for candidates.
* `positive_ideology_score`: NLI-inferred political leaning of the tweet text.

## 3. Post-stratification Strata

The "truth" population is defined in `election_configs.py` for each election cycle. We typically stratify by:
* **Partisan Strata:** Republican, Democrat, Independent.
* **Ideological Strata:** Conservative, Moderate, Liberal.
* **Demographic Profiles:** Gender and Age distributions within those groups.

Example configuration in `election_configs.py`:
```python
"mrp": {
    "partisan_strata": {"Republican": 0.33, "Democrat": 0.36, "Independent": 0.31},
    "ideological_strata": {"Conservative": 0.35, "Moderate": 0.39, "Liberal": 0.26},
    # ... profiles defined for each group
}
```

## 4. The Interactive Dashboard

The results are exported to a high-density **HTML Dashboard** located in `reports/<election>/mrp_bias_dashboard.html`.

### Dashboard Features
* **Time-Series Tracking:** Visualizes predicted vs. actual vote share over time.
* **Rolling Averages:** 7-day and 14-day rolling windows for trend analysis.
* **Milestone Integration:** Dynamic vertical markers for debates, conventions, and major events.
* **Comparative Baselines:** Prediction market data and "Actual Share" targets for MAE calculation.
