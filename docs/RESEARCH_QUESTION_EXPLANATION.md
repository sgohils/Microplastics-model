# Research Question Explanation

## The question
Does explicitly representing physical river-network connectivity in a spatiotemporal graph neural network architecture provide measurably superior prediction of microplastic concentrations compared to conventional machine learning approaches that rely only on geographic proximity and environmental covariates, when evaluated on previously unseen watersheds and time periods?

## Why it matters
Microplastic pollution is a pervasive environmental issue, and rivers are the main pathways transporting plastics from land to ocean. Predicting where microplastics accumulate helps target monitoring and remediation efforts. However, observations are sparse and expensive to collect. If river-network topology improves predictions, it could enhance the efficiency of monitoring networks by leveraging hydrological connectivity rather than relying solely on dense in-situ sampling.

## What was unknown
Prior work applied graph neural networks to river networks for streamflow or water quality prediction, but it was unclear whether explicitly representing hydrological connectivity (as opposed to geographic proximity) adds predictive value for microplastic transport, especially given the sparse and heterogeneous nature of microplastic observations. It was also unknown how model performance degrades under sparse data conditions and whether graph-based models generalize better to unseen locations or times.

## What was tested
We compared a spatiotemporal GNN that uses real river-network topology (nodes = river segments, edges = downstream flow direction, weighted by upstream drainage area) against conventional machine learning models (Ridge Regression, Random Forest, XGBoost, MLP) that use the same environmental and temporal features but without graph structure. We evaluated performance under:
- Random split (baseline)
- Spatial holdout (unseen sub-watersheds)
- Temporal holdout (unseen time periods)
- Spatiotemporal holdout (unseen space and time)
Additionally, we conducted ablation studies to isolate the contribution of graph structure and temporal encoding, and we tested robustness to sparse monitoring and extreme events.

## What would count as evidence
Evidence supporting a positive answer would be:
1. The GNN with real river topology achieves significantly lower prediction error (e.g., RMSE, MAE) than the best conventional baseline (XGBoost) on held-out spatial/temporal/spatiotemporal test sets, with statistical significance (e.g., paired permutation test, p < 0.05).
2. Ablation shows that removing the graph component (while keeping temporal and features) leads to a measurable drop in performance.
3. The GNN demonstrates better generalization (smaller performance degradation) on spatial or temporal holdouts compared to baselines.
4. Uncertainty estimates are well-calibrated, indicating reliable predictions.

Conversely, if the GNN does not outperform baselines, or if graph ablation shows no drop in performance, or if uncertainty is poorly calibrated, the evidence would not support the hypothesis.