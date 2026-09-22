# 3-Minute Presentation Script

## Opening
"Thank you for the opportunity to present my research. I'm [Student Name] from [School Name], and today I'll share my work on predicting microplastic transport through river networks using a spatiotemporal graph neural network."

## Problem (30 seconds)
"Microplastic pollution is a growing environmental concern, and predicting its transport in rivers is crucial for effective monitoring. However, observations are spatially sparse and expensive to collect. Microplastics enter rivers from sources like urban runoff and can travel downstream, meaning that pollution observed at one location may have originated far upstream. Because rivers are interconnected networks, understanding transport requires considering how pollution moves from upstream to downstream areas."

## Knowledge Gap and Research Question (30 seconds)
"Many environmental prediction models treat each monitoring location as an independent data point, using only local environmental features to predict microplastic concentration. This ignores the fact that rivers are directional networks: water flows from upstream to downstream, carrying pollutants with it. My research question is: Does explicitly representing physical river-network connectivity in a spatiotemporal graph neural network improve prediction of microplastic concentrations compared to conventional machine learning baselines?"

## Method (45 seconds)
"I gathered microplastic observation data from the USGS ScienceBase for the Delaware River (2018), Great Lakes tributaries (2014-2015), and Northeastern U.S. streams (2017-2018). I harmonized units and matched each observation to the NHDPlus river network. I extracted environmental covariates from USGS streamgages, Daymet meteorological data, and NLCD/NED geographic data, adding lagged features for antecedent conditions while preventing data leakage. I represented the river network as a directed, weighted graph where nodes are river segments, edges represent downstream flow, and edge weights reflect upstream drainage area. I built a spatiotemporal GNN combining a GRU for temporal encoding and GraphSAGE layers for spatial message passing over this graph, and compared it against strong baselines like XGBoost and Random Forest using rigorous experimental design: random split, spatial holdout, temporal holdout, spatiotemporal holdout, ablation studies, sparse monitoring experiments, and uncertainty quantification, with test set protection and leakage prevention."

## Results (45 seconds)
"The GNN achieved moderate prediction accuracy but did not significantly outperform the best conventional baseline (XGBoost). Ablation studies indicated modest contributions from graph structure (~0.14 MAE improvement) and temporal encoding (~0.07 MAE improvement). The model showed reasonable spatial generalization but poor temporal generalization and extrapolation. Uncertainty estimates were under-dispersed."

## Conclusion (30 seconds)
"In summary, this study found that explicitly modeling river-network connectivity as a graph did not yield a statistically significant improvement in microplastic prediction over strong baselines. However, the graph-based approach captures some predictive signal from river topology and temporal dynamics, as evidenced by ablation studies. The work provides a rigorous framework for evaluating graph-based methods in sparse environmental datasets and highlights the importance of leakage prevention, strong baselines, and holdout validation. **This study tests whether river-network structure can provide useful information for predicting microplastic pollution beyond conventional feature-based models.**"

## Closing
"Thank you for your attention. I'm happy to answer any questions."