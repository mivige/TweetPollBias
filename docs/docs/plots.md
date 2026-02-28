# Visualizations

The `plots.py` module utilizes Python's `seaborn` and `matplotlib` to convert generated processed data into stylized, high-quality graphs that convey the relationships between bias markers and poll outcomes.

## Generating All Plots

To easily execute all visualization pipelines in sequence:
```bash
python -m bias_analysis.plots run-all
```

---

## Individual Plot Commands

You can trigger the generation of exact plot relationships individually:

### Candidate Order Influence
Plots how positional ordering directly influences the final win percentage.
```bash
python -m bias_analysis.plots candidate-order
```

### Appellatives
Visualizes Formal versus Informal sentiment breakdowns.
```bash
python -m bias_analysis.plots appellatives
```

### Political Leaning Distribution
A massive scatter/violin hybrid plot showing vote distributions grouped by pro/neutral/anti leanings.
```bash
python -m bias_analysis.plots leaning
```

### The Bias Relationship Scatterplots
A specialized visualization comparing **Audience Bias (X-axis)** versus **Poll Outcome (Y-Axis)**, with individual hue layers for dichotomized markers (over/under Median).
```bash
python -m bias_analysis.plots bias-relationship-scatter
```
*Outputs to:* `reports/figures/scatter_candidate_order_bias.png`, etc.
