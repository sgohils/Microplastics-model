# Methods Documentation

## Study Region
The primary study region is the Delaware River Basin (approximately 13,500 km²), spanning parts of New York, New Jersey, Pennsylvania, and Delaware. Validation regions include Great Lakes tributaries (29 tributaries across 6 states) and Northeastern U.S. streams (17 streams from New York to Virginia). The temporal focus is July 2018 – March 2019, matching the microplastic sampling period.

## Data Sources
- **Microplastic observations**: USGS ScienceBase datasets (Delaware River 2018, Great Lakes Tributaries 2014-2015, Northeastern U.S. Streams 2017-2018). Units were harmonized to particles per cubic meter (particles/m³).
- **River network**: NHDPlus v2 (1:100,000) and NHDPlus HR (1:24,000) for flowline topology, stream order, slope, velocity, and drainage area.
- **Hydrology**: USGS NWIS via API for daily discharge (converted to m³/s) and gage height.
- **Meteorology**: Daymet v4 for daily precipitation, temperature, wind speed, and snow water equivalent at 1 km resolution.
- **Geography**: NLCD 2019 land cover (30m), NED 10m elevation, and gridded population density.

## Observation Processing
Microplastic observations were screened for methodological metadata, unit consistency (converted to particles/m³ where necessary), and physically implausible values (negative concentrations). Samples with insufficient metadata or concentrations below detection thresholds were flagged but not used as primary regression targets if volume measurements were missing.

## Spatial Processing
River network nodes were defined as NHDPlus flowline reaches. Microplastic sampling locations were spatially joined to intersecting reaches; for the Delaware River, all reaches within the watershed were included as nodes (~46 nodes in the experiment), with microplastic targets only at sampled reaches. Environmental covariates were extracted for all nodes via spatial matching (centroid extraction for gridded data, nearest streamgage for discharge/temperature within 50 km using inverse-distance weighting).

## Temporal Processing
Data were aligned to daily timesteps. For each microplastic sampling date, we extracted:
- Same-day environmental conditions (precipitation, temperature, discharge).
- Lagged features: 1-day, 3-day, and 7-day lags for precipitation and discharge (to avoid leakage, only past data used).
- Temporal calendar features: month, season.
All features were matched to the date of observation, ensuring no future information leakage.

## River Graph Construction
- **Nodes**: River segments (NHDPlus flowline reaches). Each node carries static and time-varying environmental features.
- **Edges**: Directed edges following NHDPlus hydrologic sequencing (upstream → downstream).
- **Edge weights**: Upstream drainage area (flow accumulation), representing relative flow contribution.
- **Edge features** (optional): flow direction indicator, stream order difference.
- **Graph construction for holdouts**: For spatial/temporal holdouts, the graph was constructed using only training-set nodes plus their immediate upstream/downstream neighbors to prevent test-to-train leakage.

## Feature Engineering
Static features (per node): elevation, slope, drainage area, stream order, land-cover fractions (forest, urban, agricultural), impervious surface, population density.
Temporal features (per node, per timestep): precipitation, 7-day cumulative precipitation, max/min temperature, discharge (from nearest streamgage), month, season.
Lagged features: precipitation and discharge at 1-, 3-, and 7-day lags.
Features excluded: microplastic measurements from upstream/downstream locations (to prevent target leakage).

## Target Definition
- **Definition**: Microplastic concentration in river water.
- **Units**: particles per cubic meter (particles/m³).
- **Type**: Continuous regression.
- **Transformation**: log1p (log(1+x)) applied to target for training to stabilize variance; predictions were transformed back to original scale for evaluation.
- **Handling incompatible units**: Converted particles/L to particles/m³ (×1000); studies reporting only particle counts without volume were excluded from the primary regression target.

## Train/Validation/Test Design
We employed multiple split strategies to assess different aspects of generalization:
1. **Random Split (70/15/15)**: Randomly shuffled observations split into training, validation, and test sets. Provides an upper bound on performance under ideal conditions (but risks spatial/temporal leakage).
2. **Spatial Holdout**: Trained on a subset of sub-watersheds, tested on a completely held-out sub-watershed (e.g., train on Delaware River main stem + western tributaries, test on eastern tributaries).
3. **Temporal Holdout**: Trained on early period (July–December 2018), tested on late period (January–March 2019).
4. **Spatiotemporal Holdout**: Combined spatial and temporal holdouts (e.g., train on western Delaware River in January 2019, test on eastern Delaware River in July 2018) – the most stringent generalization test.
For each split, we reported performance across multiple random seeds (where applicable) and used 5-fold temporal-aware cross-validation for robustness. Given the limited sample size (~150–200 observations), we also used leave-one-location-out cross-validation for spatial generalization tests and reported bootstrap confidence intervals.

## Baselines
We compared against strong conventional baselines to ensure a fair evaluation:
- Mean/Median predictor (predicts training set mean/median).
- Ridge Regression (L2-regularized linear regression).
- Random Forest (500 trees, tuned max_depth).
- XGBoost (gradient boosting with early stopping).
- MLP (two hidden layers: 100→50, ReLU, dropout=0.3).
All baselines used the same feature set and preprocessing (fit on training data only) to isolate the value of graph structure.

## GNN Architecture
Our spatiotemporal GNN combines a GRU (temporal encoder) and GraphSAGE (spatial message passing) with an MLP prediction head:
- **Input**: [N_nodes, T=7, F_features] (node features over 7 timesteps).
- **Feature Encoder**: MLP (Linear → ReLU → Linear) projecting features to 64-dim space.
- **Temporal Encoder**: GRU (hidden dim=64, sequence length=7) capturing temporal dependencies.
- **Graph Message Passing**: 2-layer GraphSAGE (64→64→32) with mean aggregation, ReLU activation, and directed edges following river flow.
- **Prediction Head**: MLP (Linear(32) → ReLU → Linear(16) → ReLU → Linear(1)) outputting log-transformed concentration.
- **Output**: log1p(concentration) per node; transformed back to original scale for evaluation.
Implementation details: PyTorch Geometric framework, Adam optimizer (lr=0.001, weight_decay=1e-4), dropout=0.3, early stopping (patience=30), batch size=full batch (due to small dataset), sequence length=7 timesteps.

## Training
- **Loss**: Mean Squared Error (MSE) on log-transformed target.
- **Optimizer**: Adam with learning rate 0.001 and weight decay 1e-4.
- **Regularization**: Dropout (0.3) and L2 weight decay to mitigate overfitting.
- **Early Stopping**: Based on validation RMSE (log-transformed) with patience=30 epochs and minimum delta=0.001.
- **Ensemble**: 5 ensemble members trained with different random seeds and train/validation splits for uncertainty estimation.
- **Hardware**: CPU-compatible (Intel i5/Ryzen 5+, 16GB RAM); optional GPU (NVIDIA GTX 1660+ 6GB VRAM) for faster training.
- **Runtime**: ~2–4 hours for all experiments on CPU.

## Hyperparameter Optimization
Hyperparameters were selected based on balancing expressiveness with overfitting risk given the small dataset:
- Hidden dimension: 64 (chosen over 128 to reduce overfitting).
- GNN layers: 2 (sufficient receptive depth for river networks).
- GRU layers: 1 (simpler than LSTM, effective for short sequences).
- Dropout: 0.3.
- Weight decay: 1e-4.
- Learning rate: 0.001.
- Sequence length: 7 days (based on hydrological memory and ablation).
No extensive hyperparameter search was performed; values were motivated by literature and pilot experiments.

## Ablations
We conducted component ablation studies to isolate the value of each module:
- **Model A**: Environmental Only (static geographic + instantaneous environmental, no temporal lags, no graph).
- **Model B**: Environmental + Temporal (added lagged precipitation/discharge and temporal calendar features, no graph).
- **Model C**: Environmental + Graph (static graph structure, no temporal encoder).
- **Model D**: Environmental + Graph + Temporal (graph + temporal encoder, no feature encoder beyond baseline).
- **Model E**: Full Spatiotemporal GNN (feature encoder + GRU + GraphSAGE + MLP head, with uncertainty via deep ensembles).
Comparisons (A vs. B, B vs. C, C vs. D, D vs. E, A vs. E) quantified the contribution of temporal information, static graph structure, temporal encoding, and full architecture.

## Generalization Testing
- **Spatial holdout**: Left-one-sub-watershed-out; trained on remaining sub-watersheds, tested on held-out.
- **Temporal holdout**: Split by time (early vs. late); ensured no temporal leakage in feature construction.
- **Spatiotemporal holdout**: Combined spatial and temporal holdouts.
- **Sparse monitoring**: Systematically reduced training observations (100%, 75%, 50%, 25%, 10%) and evaluated on a fixed test set to assess robustness to data scarcity.
- **Extreme events**: Separated test set into high-flow/precipitation extremes vs. normal conditions and compared error metrics.

## Uncertainty
We used deep ensembles (5 members) trained with different random seeds and different train/validation splits (same test set). Final prediction = mean across ensemble members. Uncertainty estimate = standard deviation across members. 95% prediction interval = [mean − 1.96×std, mean + 1.96×std]. We reported coverage probability (percentage of observations falling within the interval) and noted under-/over-dispersion.

## Statistical Analysis
- **Primary test**: Paired permutation test on RMSE (GNN vs. XGBoost) with 10,000 permutations, α=0.05.
- **Multiple comparisons correction**: Bonferroni where appropriate (e.g., when comparing multiple baselines or ablation models).
- **Effect size**: Cohen's d for mean differences; relative improvement percentage.
- **Confidence intervals**: Bootstrap (1,000 resamples, 95% CI) for RMSE differences.
- **Cross-validation**: 5-fold with random seeds {42, 123, 456, 789, 101} for variability assessment.
- **Assumptions verification**: Shapiro-Wilk test on residuals for normality; Breusch-Pagan test for homoscedasticity; Moran's I for spatial/temporal autocorrelation in test errors.