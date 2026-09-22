# Elevator Pitch

## Problem
Microplastic pollution is a growing environmental concern, and predicting its transport in rivers is crucial for effective monitoring. However, observations are spatially sparse and expensive to collect.

## Gap
Many environmental prediction models treat each monitoring location as an independent data point, using only local environmental features to predict microplastic concentration. This ignores the fact that rivers are directional networks: water flows from upstream to downstream, carrying pollutants with it.

## Research
I investigated whether explicitly representing physical river-network connectivity in a spatiotemporal graph neural network improves prediction of microplastic concentrations compared to conventional machine learning baselines.

## Method
I gathered microplastic observation data from the USGS ScienceBase, harmonized units, and matched observations to the NHDPlus river network. I extracted environmental covariates from USGS streamgages, Daymet meteorological data, and NLCD/NED geographic data, adding lagged features for antecedent conditions while preventing data leakage. I represented the river network as a directed, weighted graph where nodes are river segments, edges represent downstream flow, and edge weights reflect upstream drainage area. I built a spatiotemporal GNN combining a GRU for temporal encoding and GraphSAGE layers for spatial message passing over this graph.

## Result
The GNN achieved moderate prediction accuracy but did not significantly outperform the best conventional baseline (XGBoost). Ablation studies indicated modest contributions from graph structure (~0.14 MAE improvement) and temporal encoding (~0.07 MAE improvement). The model showed reasonable spatial generalization but poor temporal generalization and extrapolation.

## Importance
This work provides a rigorous framework for evaluating graph-based methods in sparse environmental datasets and highlights the importance of leakage prevention, strong baselines, and holdout validation. **This study tests whether river-network structure can provide useful information for predicting microplastic pollution beyond conventional feature-based models.**