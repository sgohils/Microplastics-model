# Hypothesis Explanation

## Hypothesis
### Primary Hypothesis (H1)
A spatiotemporal graph neural network that explicitly incorporates real river-network connectivity topology will achieve statistically significantly lower prediction error (measured by RMSE) on held-out test data compared to the best-performing conventional machine learning baseline (XGBoost), when both are evaluated on spatially and temporally held-out microplastic observations.

### Null Hypothesis (H0)
There is no statistically significant difference in prediction error (RMSE) between a graph neural network incorporating river-network connectivity and the best-performing conventional machine learning baseline (XGBoost), when evaluated on held-out test data. Any observed differences are within the bounds of random variation.

### Secondary Hypotheses
**H2 (Graph Structure):** Real river-network topology provides significantly better prediction than geographic nearest-neighbor or fully-connected graph structures (p < 0.05, paired t-test on RMSE across random seeds).

**H3 (Temporal Information):** Models incorporating temporal lag features achieve significantly lower prediction error than models using only contemporaneous environmental conditions (p < 0.05, paired permutation test on MAE).

**H4 (Sparse Data Robustness):** The graph-based approach maintains superior prediction accuracy relative to baselines across all data sparsity levels (10%, 25%, 50%, 75%, 100%), with the performance gap widening as training data decreases.

**H5 (Spatial Generalization):** The graph-based model demonstrates superior generalization to unseen watersheds compared to conventional baselines, as measured by performance degradation rate between random and spatial holdout splits.

**H6 (Topology Control):** Randomized river topology provides no significant predictive benefit over no graph structure, confirming that meaningful topology contributes to model performance.

**H7 (Uncertainty Calibration):** Prediction uncertainty (measured by ensemble standard deviation) increases significantly in poorly monitored regions and during extreme weather events compared to well-monitored, normal conditions.

## Rationale
The hypotheses are grounded in the scientific premise that hydrological connectivity influences contaminant transport. By explicitly modeling the directed flow of water through river segments, the GNN should capture upstream influences on microplastic concentration that purely geographic models miss. Temporal lag features account for antecedent conditions that may mobilize or transport microplastics. Robustness and generalization hypotheses test whether these advantages hold under challenging real-world conditions (sparse monitoring, extreme events, unseen locations).

## What result would have disproved the hypothesis?
The primary hypothesis (H1) would be disproved if:
- The GNN with real river topology does not achieve a statistically significant reduction in RMSE compared to XGBoost on held-out spatial/temporal/spatiotemporal test sets (p ≥ 0.05 in a paired permutation test).
- The GNN's performance is equal to or worse than XGBoost, or the difference is not statistically significant after correcting for multiple comparisons.
Additionally, if secondary hypotheses fail (e.g., randomized topology performs as well as real topology, or temporal lags do not improve performance), it would suggest that the assumed mechanisms (hydrological connectivity, antecedent conditions) are not driving predictive gains.

In our actual results (see `experiments/results/final_summary.json` and `statistical_comparison.json`), the GNN did not achieve statistically significant improvement over XGBoost (p > 0.05 for both RMSE and MAE). Thus, the null hypothesis (H0) cannot be rejected, and H1 is not supported. However, the study provides valuable insights into the limitations of modeling sparse microplastic data and the importance of rigorous baseline comparisons.