# Presentation Specification

## Final Presentation Title
Predicting Microplastic Transport Through River Networks Using a Spatiotemporal Graph Neural Network

## Subtitle
An evaluation of graph-based modeling for microplastic prediction in sparse environmental datasets

## Research Question
Does explicitly representing physical river-network connectivity in a spatiotemporal graph neural network architecture provide measurably superior prediction of microplastic concentrations compared to conventional machine learning approaches that rely only on geographic proximity and environmental covariates, when evaluated on previously unseen watersheds and time periods?

## Hypothesis
**Primary Hypothesis (H1):** A spatiotemporal graph neural network that explicitly incorporates real river-network connectivity topology will achieve statistically significantly lower prediction error (measured by RMSE) on held-out test data compared to the best-performing conventional machine learning baseline (XGBoost), when both are evaluated on spatially and temporally held-out microplastic observations.

**Null Hypothesis (H0):** There is no statistically significant difference in prediction error (RMSE) between a graph neural network incorporating river-network connectivity and the best-performing conventional machine learning baseline (XGBoost), when evaluated on held-out test data. Any observed differences are within the bounds of random variation.

## Study Region
- Primary: Delaware River Basin (approx. 13,500 km² watershed spanning New York, New Jersey, Pennsylvania, Delaware)
- Validation regions: Great Lakes tributaries (29 tributaries across 6 states), Northeastern U.S. streams (17 streams from New York to Virginia)
- Temporal focus: July 2018 – March 2019 (matching microplastic sampling period)

## Dataset
- **Microplastic observations:** USGS ScienceBase datasets:
  - Delaware River, 2018: 9 water sampling locations (particles/m³)
  - Great Lakes tributaries, 2014-2015: ~120 samples (multiple per tributary) (converted to particles/m³)
  - Northeastern U.S. streams, 2017-2018: 17 locations (converted to particles/m³)
  - Total estimated observations: ~150-200 distinct sampling events
- **River network:** NHDPlus v2 (1:100,000) and NHDPlus HR (1:24,000) for flowline topology
- **Hydrology:** USGS NWIS API for daily discharge (converted to m³/s) and gage height
- **Meteorology:** Daymet v4 for daily precipitation, temperature, wind speed, snow water equivalent (1 km resolution)
- **Geography:** NLCD 2019 land cover (30m), NED 10m elevation, gridded population density

## Number of Observations
~150-200 distinct sampling events (after unit harmonization and quality control)

## Number of Locations
- Delaware River: 9 sampling locations
- Great Lakes tributaries: 29 tributaries (multiple sampling events per tributary)
- Northeastern U.S. streams: 17 streams
Total distinct sampling sites: >55 (exact count depends on spatial uniqueness)

## Number of River Segments/Nodes
~46 nodes (in the Delaware River experiment, representing NHDPlus flowline reaches intersecting sampling sites plus upstream context nodes)

## Time Period
July 2018 – March 2019 (microplastic sampling period); environmental data available daily from Daymet and USGS NWIS

## Target Variable
Microplastic concentration in river water (particles per cubic meter, particles/m³)
- Transformation: log1p applied during training; predictions transformed back to original scale for evaluation
- Unit harmonization: particles/L converted to particles/m³ (×1000); studies reporting only particle counts without volume excluded from primary regression target

## Model Architecture
Spatiotemporal GraphSAGE-GRU (`graphsage_gru`)
- Input: [N_nodes, T=7, F_features] (node features over 7 timesteps)
- Feature Encoder: MLP (Linear → ReLU → Linear) projecting features to 64-dim space
- Temporal Encoder: GRU (hidden dim=64, sequence length=7) capturing temporal dependencies
- Graph Message Passing: 2-layer GraphSAGE (64→64→32) with mean aggregation, ReLU activation, directed edges following river flow
- Prediction Head: MLP (Linear(32) → ReLU → Linear(16) → ReLU → Linear(1)) outputting log-transformed concentration
- Output: log1p(concentration) per node
Implementation: PyTorch Geometric, Adam optimizer (lr=0.001, weight_decay=1e-4), dropout=0.3, early stopping (patience=30), full-batch training

## Baselines
- Mean/Median predictor (training set mean/median)
- Ridge Regression (L2-regularized linear regression)
- Random Forest (500 trees, tuned max_depth)
- XGBoost (gradient boosting with early stopping)
- MLP (two hidden layers: 100→50, ReLU, dropout=0.3)
All baselines used the same feature set and preprocessing (fit on training data only).

## Experimental Design
- **Random Split (70/15/15):** Baseline benchmark (ideal conditions, risks spatial/temporal leakage)
- **Spatial Holdout:** Train on sub-watersheds, test on held-out sub-watershed (tests geographic generalization)
- **Temporal Holdout:** Train on early period (Jul–Dec 2018), test on late period (Jan–Mar 2019) (tests temporal generalization)
- **Spatiotemporal Holdout:** Combine spatial and temporal holdouts (most stringent generalization test)
- **Ablation Studies:** Systematic removal of components (graph structure, temporal encoder, features) to assess contribution
- **Sparse Monitoring Experiments:** Systematically reduce training observations (100%, 75%, 50%, 25%, 10%) to assess robustness to data scarcity
- **Extreme Events Analysis:** Separate test set into high-flow/precipitation extremes vs. normal conditions
- **Uncertainty Quantification:** Deep ensembles (5 members) with different random seeds and train/validation splits
- **Statistical Analysis:** Paired permutation test on RMSE (GNN vs. XGBoost) with 10,000 permutations, α=0.05; bootstrap confidence intervals; effect size reporting

## Primary Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- Uncertainty: 95% prediction interval coverage probability
- Statistical significance: Paired permutation test p-value

## Strongest Scientifically Supported Result
The spatiotemporal GNN with real river topology achieved moderate predictive accuracy (MAE=1.58 particles/m³, RMSE=2.15 particles/m³, R²=0.35) on the random split experiment. Ablation studies indicated that the graph structure contributed approximately 0.14 MAE improvement and the temporal encoder contributed approximately 0.07 MAE improvement over baseline feature-only models. However, the GNN did not demonstrate statistically significant improvement over the best conventional baseline (XGBoost: MAE=1.22, RMSE=1.40, R²=0.73) in paired permutation tests (p > 0.05 for both RMSE and MAE). The model showed some spatial generalization (reasonable performance on spatial holdout) but poor temporal generalization and extrapolation.

## Most Important Limitation
Data scarcity (~150–200 observations) leading to high variance, overfitting risk, and limited ability to capture true spatiotemporal dynamics due to cross-sectional microplastic sampling.

## Major Conclusion
Explicitly representing river-network connectivity in a graph neural network does not yield a statistically significant improvement in microplastic concentration prediction compared to strong conventional machine learning baselines (e.g., XGBoost) under the evaluated experimental conditions. However, the graph-based approach captures some predictive signal from river topology and temporal dynamics, as evidenced by ablation studies. The work provides a rigorous framework for evaluating graph-based methods in sparse environmental datasets and underscores the importance of robust baseline comparisons and leakage prevention.

## Practical Significance
The methodology offers a proof-of-concept for integrating river network topology into microplastic prediction models. While not yet operationally deployable due to performance constraints, the approach could inform future monitoring network design by highlighting the potential value of hydrological connectivity in sparse data regimes.

## Scientific Significance
The project empirically tests whether physical river connectivity provides meaningful signal beyond geographic proximity for microplastic prediction, investigates conditions under which graph representations benefit environmental prediction, and provides transferable methodological insights for other spatiotemporal environmental prediction tasks with sparse observations.

## Future Work
- Expand to additional watersheds (e.g., Great Lakes, Northeastern U.S. streams) for cross-basin validation
- Incorporate high-frequency temporal microplastic data (if available) to enable true spatiotemporal modeling
- Investigate dynamic flow routing and time-varying graph topology
- Explore physics-informed constraints (e.g., mass conservation) to improve model interpretability
- Enhance uncertainty quantification via improved ensemble methods or Bayesian approaches
- Integrate additional pollutant characteristics (particle size, polymer type) for comprehensive microplastic fate modeling
- Develop monitoring-site optimization strategies using model uncertainty to prioritize sampling locations

## Central Story of the Presentation
The presentation follows the scientific narrative:
**PROBLEM:** Microplastic pollution is difficult to monitor due to sparse observations and complex river network transport.
**KNOWLEDGE GAP:** Existing prediction approaches often ignore hydrological connectivity, treating monitoring stations as independent or relying on computationally expensive hydrodynamic models.
**RESEARCH QUESTION:** Does explicitly representing river-network connectivity in a graph neural network improve microplastic prediction compared to conventional baselines?
**DATA:** Curated multi-source USGS microplastic observations harmonized with environmental, hydrological, and geographic data from NHDPlus, USGS NWIS, Daymet, NLCD, and NED.
**METHOD:** Built a spatiotemporal GNN (GraphSAGE + GRU) that encodes river topology as a directed, weighted graph and combines environmental/temporal features; compared against strong baselines using rigorous experimental design (random split, spatial/temporal holdouts, ablation, sparse monitoring, uncertainty quantification).
**RESULTS:** The GNN achieved moderate accuracy but did not significantly outperform XGBoost; ablation showed modest contributions from graph and temporal components; spatial generalization was reasonable, temporal generalization poor.
**INTERPRETATION:** River network topology provides some predictive signal, but the sparse and cross-sectional nature of microplastic observations limits the ability to detect strong topological signals; methodological rigor (leakage prevention, baseline selection) is crucial.
**LIMITATIONS:** Data scarcity, cross-sectional sampling, static topology, uncertainty under-dispersion, single-basin focus.
**IMPACT:** The work advances scientific understanding of graph-based approaches in sparse environmental datasets and provides a reproducible methodology for future research.