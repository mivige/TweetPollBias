# Visualizations

The `plots.py` module utilizes Python's `seaborn` and `matplotlib` to convert generated processed data into stylized, high-quality graphs that convey the relationships between bias markers and poll outcomes. It also handles generating the final interactive HTML SCWG Bias Dashboard.

## Generating All Plots

To easily execute all visualization pipelines in sequence for a specific election:
```bash
python -m bias_analysis.plots run-all --election us24
```
*Key Output:* `reports/<election>/scwg_bias_dashboard.html` (Interactive SCWG Dashboard)

---

## Individual Plot Commands

You can trigger the generation of exact plot relationships individually:

### Candidate Order Influence
Plots how positional ordering directly influences the final win percentage.
```bash
python -m bias_analysis.plots candidate-order --election us24
```

### Appellatives
Visualizes Formal versus Informal sentiment breakdowns.
```bash
python -m bias_analysis.plots appellatives --election us24
```

### Political Leaning Distribution
A massive scatter/violin hybrid plot showing vote distributions grouped by pro/neutral/anti leanings.
```bash
python -m bias_analysis.plots leaning --election us24
```

### The Bias Relationship Scatterplots
A specialized visualization comparing **Audience Bias (X-axis)** versus **Poll Outcome (Y-Axis)**, with individual hue layers for dichotomized markers (over/under Median).
```bash
python -m bias_analysis.plots bias-relationship-scatter --election us24
```
*Outputs to:* `reports/figures/<election>/scatter_candidate_order_bias.png`, etc.
