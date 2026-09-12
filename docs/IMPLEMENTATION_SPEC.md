# Implementation Specification

This document summarizes the scientific decisions from Stage 1 (Research Specification) for the implementation team.

## Research Parameters

- **Primary Research Question:** Does explicit representation of physical river-network connectivity in a graph neural network architecture improve prediction of microplastic concentrations compared with conventional machine learning models that rely only on geographic proximity and environmental covariates, when evaluated on held-out spatial and temporal data?

- **Primary Hypothesis (H1):** GNN with real river topology will achieve significantly lower RMSE than best baseline (XGBoost) on held-out data
- **Null Hypothesis (H0):** No significant difference between GNN and best baseline
- **Significance Level:** α = 0.05

## Study Region

- **Primary:** Delaware River Basin (13,500 km² watershed spanning NY, NJ, PA, DE)
- **Validation:** Great Lakes tributaries (29 tributaries, 6 states)
- **Temporal Focus:** July 2018 - March 2019

## Target Variable

- **Definition:** Microplastic concentration in river water
- **Units:** particles per cubic meter (particles/m³)
- **Type:** Continuous regression
- **Transformation:** log1p transform for training; inverse transform for evaluation
- **Handling incompatible units:** Convert particles/L → particles/m³ (×1000)

## Datasets

### Primary (Microplastic Observations)
1. **Delaware River, 2018** - USGS ScienceBase, DOI:10.5066/P9QVIVX3
   - 9 water sampling locations, July 2018 - March 2019
   - Variables: concentration (particles/m³), particle type, morphology
   - Access: CSV download from ScienceBase

2. **Great Lakes Tributaries, 2014-2015** - USGS
   - 29 tributaries across 6 states
   - Multiple sampling events per tributary
   - Units: particles/km² (surface), convertible to volumetric

3. **Northeastern U.S. Streams, 2017-2018** - USGS
   - 17 streams from NY to VA

### Environmental Covariates
1. **River Network:** NHDPlus v2 (1:100,000) + NHDPlus HR (1:24,000)
   - Flowlines with hydrologic sequencing
   - Stream order, slope, velocity, drainage area
   - Flow direction and connectivity

2. **Hydrology:** USGS NWIS via API
   - Daily discharge (cfs → m³/s), gage height
   - ~20+ streamgages in Delaware River Basin

3. **Meteorology:** Daymet v4
   - Daily 1km resolution: precipitation, temperature, wind, radiation
   - 1980-present coverage

4. **Geography:** 
   - NLCD 2019 land cover (30m resolution)
   - NED 10m elevation
   - Gridded population (1km resolution)

## Graph Specification

### Node Definition
- **Node = River segment** (NHDPlus flowline reach)
- Nodes aligned to microplastic sampling locations via spatial join
- Each node carries environmental covariates at the sampling date

### Edge Definition
- **Edge = Flow-direction connectivity** (from NHDPlus hydrologic sequencing)
- Edges follow downstream flow direction: upstream → downstream
- Edge weights = upstream drainage area (larger upstream contributing more)

### Edge Features
- Drainage area contribution (proportional to upstream nodes)
- Flow direction indicator (binary)
- Stream order difference

## Feature Set

### Static Features (per node)
- Elevation (m)
- Slope (degrees)
- Drainage area (km²)
- Stream order (dimensionless)
- Land cover fractions (forest, urban, agricultural - %)
- Impervious surface (%)
- Population density (people/km²)

### Temporal Features (per node, per timestep)
- Precipitation (mm, daily)
- 7-day cumulative precipitation (mm)
- Max/min temperature (°C)
- Discharge (m³/s, from nearest streamgage, daily)
- Month (1-12), Season (categorical)

### Lagged Features
- Precipitation: 1-day, 3-day, 7-day lags
- Discharge: 1-day, 3-day, 7-day lags

### Features Excluded
- Microplastic measurements from upstream/downstream locations (target leakage)

## Temporal Design

- **Granularity:** Daily alignment
- **History window:** 7 days (sequence length for temporal encoder)
- **Lag periods:** 1, 3, 7 days (empirically tested via ablation)

## Train/Validation/Test Splits

1. **Random Split (70/15/15):** Baseline benchmark
2. **Spatial Holdout:** Leave one sub-watershed (e.g., lower Delaware) out
3. **Temporal Holdout:** Train on July-Dec 2018, test on Jan-Mar 2019
4. **Spatiotemporal Holdout:** Combine spatial and temporal holdouts (hardest)

## Baseline Models

1. Mean/Median predictor (absolute minimum baseline)
2. Ridge Regression (linear with regularization)
3. Random Forest (500 trees, max_depth tuned)
4. XGBoost (learning_rate=0.01, max_depth=6, early stopping=50)
5. MLP (2 hidden layers: 100→50, ReLU, dropout=0.3)

## Proposed GNN Architecture

### Spatiotemporal GNN (GraphSAGE + GRU)

```
Input: [N_nodes, T_timesteps, F_features]
↓ Feature Encoder: Linear(F) → ReLU → Linear(64)
↓ Temporal Encoder: GRU(64→64, 7 timesteps)
↓ Graph Message Passing: GraphSAGE(64→64→32), 2 layers, ReLU
↓ Prediction Head: Linear(32)→ReLU→Linear(16)→ReLU→Linear(1)
↓ Output: log1p(concentration)
```

### Implementation Details
- **Framework:** PyTorch Geometric
- **GNN layers:** 2-layer GraphSAGE with mean aggregation
- **Temporal:** 1-layer GRU (hidden dim 64, sequence length 7)
- **Loss:** MSE on log-transformed target
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-4)
- **Regularization:** 0.3 dropout, early stopping (patience=30)
- **Ensemble:** 5 members for uncertainty (different seeds, splits)

## Experiments

| Exp | Model | Spatial Split | Temporal | Graph | Purpose |
|-----|-------|---------------|----------|-------|---------|
| E1-E5 | Baseline models | Random | Random | None | Benchmark |
| E6 | Full GNN | Random | Random | River | Initial test |
| E7-E10 | GNN ablations | Random | Random | Varies | Component analysis |
| E11 | Full GNN | Spatial | Same | River | Spatial generalization |
| E12 | Full GNN | Same | Temporal | River | Temporal generalization |
| E13 | Full GNN | Spatial | Temporal | River | Hard generalization |
| E14 | Full GNN | Same | Same | Randomized | Topology control |
| E15 | Full GNN | Same | Same | Geo-KNN | Graph structure |
| E16 | Full GNN | Same | Same | Haversine | Graph structure |
| E17 | All models | Same | Same | Varies | Sparse data |
| E18 | Full GNN | Spatial | Same | River | Unseen watershed |
| E19 | Full GNN | Same | Temporal | River | Temporal extrapolation |
| E20 | Full GNN | Same | Same | River | Extreme events |
| E21 | Full GNN | Same | Same | River | Uncertainty |

## Statistical Analysis

- **Primary test:** Paired permutation test on RMSE (GNN vs. XGBoost), 10,000 permutations, α=0.05
- **Multiple comparisons correction:** Bonferroni where appropriate
- **Effect size:** Cohen's d
- **Confidence intervals:** Bootstrap (1000 resamples, 95% CI)
- **Cross-validation:** 5-fold with random seeds {42, 123, 456, 789, 101}

## Uncertainty Method

**Deep Ensembles (5 members)**
- Different random initializations
- Different train/validation splits (same test set)
- Prediction = mean(ensemble)
- Uncertainty = std(ensemble)
- 95% prediction interval = [mean - 1.96×std, mean + 1.96×std]

## Evaluation Metrics

- **Primary:** RMSE, MAE, R²
- **Secondary:** Median Absolute Error, Explained Variance
- **Uncertainty:** Coverage probability (95% PI should contain 95% of observations)

## Computational Requirements

- **Hardware:** CPU-compatible (Intel i5/Ryzen 5+, 16GB RAM)
- **Optional GPU:** NVIDIA GTX 1660+ (6GB VRAM) for faster training
- **Storage:** ~2GB for datasets, ~500MB for models/results
- **Runtime:** 2-4 hours total on CPU for all experiments

## Data Integrity Rules

1. No synthetic microplastic measurements for primary analysis
2. No fabricating observations or results
3. All preprocessing fitted on training data only
4. Target leakage prevention: no microplastic features
5. Temporal leakage prevention: only past data in features
6. Spatial leakage prevention: no test-to-train edges in cross-validation
