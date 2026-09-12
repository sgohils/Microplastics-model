# Implementation Assumptions

This document records scientifically defensible assumptions made during implementation where the Stage 1 specification was ambiguous or required implementation decisions.

## 1. Data Availability and Access

### Assumption 1.1: USGS ScienceBase Access
**Decision:** Use the publicly accessible CSV data from USGS ScienceBase (DOI:10.5066/P9QVIVX3) for Delaware River microplastic observations.
**Rationale:** This is the primary, confirmed publicly available dataset.
**Risk:** ScienceBase may require authentication for API access. If so, data will be downloaded manually or via alternative endpoint.

### Assumption 1.2: Data Integration Across Studies
**Decision:** Only integrate datasets after unit harmonization. Measurements in particles/L will be converted to particles/m³ (×1000). Studies reporting only particle counts without volume will be excluded from the primary regression target.
**Rationale:** Maintains scientific validity while maximizing data utilization.
**Risk:** Reduced sample size if incompatible datasets must be excluded.

## 2. Graph Construction

### Assumption 2.1: Node Definition
**Decision:** Nodes represent river segments (NHDPlus flowline reaches) that spatially intersect microplastic sampling locations, with the option to include upstream river segments as additional context nodes.
**Rationale:** This represents actual river topology rather than abstract point locations, enabling the graph to capture real hydrological connectivity.
**Alternative Considered:** Monitoring stations as nodes. Rejected because the sample size (9 Delaware River + 29 Great Lakes + 17 NE streams ≈ 55 unique locations) provides insufficient nodes for meaningful message passing.

### Assumption 2.2: Node Expansion
**Decision:** For the Delaware River dataset, expand to include all NHDPlus flowline reaches within the watershed as additional graph nodes, with microplastic sampling locations mapped to their intersecting reaches. Environmental covariates will be extracted for all nodes, but the target (microplastic concentration) will only be available for sampled reaches.
**Rationale:** This increases the graph size to a meaningful number of nodes (~100-500 reaches) while maintaining real topology.
**Alternative Considered:** Use only the 9 sampled locations. Rejected as insufficient for GNN message passing.

### Assumption 2.3: Edge Weighting
**Decision:** Edge weights will be proportional to upstream drainage area, representing the relative contribution of upstream flow to downstream transport.
**Rationale:** Physically motivated - larger upstream areas contribute more water and potentially more microplastics.
**Alternative Considered:** Unit weights. Rejected as it loses physical information.

## 3. Feature Engineering

### Assumption 3.1: Streamgage Interpolation
**Decision:** When a microplastic sampling location doesn't have a co-located streamgage, interpolate discharge from the 3 nearest streamgages using inverse-distance weighting, using only streamgages within 50 km.
**Rationale:** Most sampling locations don't have direct streamgage data; spatial interpolation is standard hydrological practice.
**Risk:** Interpolation introduces uncertainty; this is flagged in uncertainty analysis.

### Assumption 3.2: Environmental Data Extraction
**Decision:** For gridded data (Daymet, NLCD, NED), extract values at the centroid of the river segment polygon.
**Rationale:** Centroid extraction is standard practice and computationally efficient for large datasets.
**Alternative Considered:** Average of all intersecting grid cells. Rejected for computational simplicity unless significant differences found.

### Assumption 3.3: Lagged Feature Selection
**Decision:** Use 1-day, 3-day, and 7-day lags for temporal features. The 7-day window is chosen because: (1) it represents approximately one week of hydrological memory, (2) it's a common choice in hydrological ML literature, (3) it balances signal retention with computational complexity.
**Rationale:** Based on the understanding that microplastic transport is influenced by recent precipitation and flow conditions, with memory effects extending several days.
**Risk:** If optimal lag differs, this may reduce model performance. Ablation experiments will investigate alternative lags.

## 4. Model Architecture

### Assumption 4.1: Model Capacity
**Decision:** Use a moderate model capacity (hidden_dim=64, 2 GNN layers, 1 GRU layer) to balance expressiveness with overfitting risk.
**Rationale:** With ~150-200 observations, excessive model complexity will lead to overfitting. Regularization (dropout=0.3, weight decay=1e-4) further mitigates this.
**Alternative Considered:** Larger models (hidden_dim=128, 3 GNN layers). Rejected due to overfitting risk.

### Assumption 4.2: Temporal Resolution for GNN
**Decision:** The temporal component will use a fixed 7-day sequence length for the GRU. For observations without complete 7-day environmental history, the sequence will be padded with the earliest available data or filled with NaN and masked.
**Rationale:** Provides consistent input dimensionality for training.
**Risk:** May introduce bias for observations in data-poor periods.

## 5. Training and Validation

### Assumption 5.1: Early Stopping Metric
**Decision:** Use validation RMSE on log-transformed target for early stopping, with patience=30 epochs and minimum delta=0.001.
**Rationale:** RMSE penalizes large errors, appropriate for concentration predictions. Log-transform stabilizes variance.
**Alternative Considered:** MAE. Rejected as less sensitive to outliers which may be meaningful in this domain.

### Assumption 5.2: Batch Size for Graph Data
**Decision:** Use full-batch training rather than mini-batch training for the GNN, given the small dataset size.
**Rationale:** With ~100-500 nodes and ~150-200 observations, mini-batching provides no efficiency benefit and complicates temporal sequence handling.
**Alternative Considered:** Mini-batch training. Rejected as unnecessary overhead.

## 6. Statistical Analysis

### Assumption 6.1: Normality Assumption Relaxation
**Decision:** While permutation tests don't require normality assumptions, bootstrap confidence intervals will be computed with 1000 resamples. If the bootstrap distribution is highly non-normal, bias-corrected and accelerated (BCa) intervals will be used.
**Rationale:** Robust approach that doesn't rely on distributional assumptions.

### Assumption 6.2: Effect Size Reporting
**Decision:** Report both absolute RMSE differences and relative improvement (%) as effect sizes. Cohen's d will also be computed where appropriate.
**Rationale:** Provides interpretable measures for science fair presentation.

## 7. Visualization

### Assumption 7.1: Map Projections
**Decision:** Use EPSG:4326 (WGS84) for all maps, with optional local projections for distance calculations.
**Rationale:** Standard for web mapping and widely understood.

### Assumption 7.2: Figure Style
**Decision:** Use clean, scientific style figures (no decorative elements, standard fonts, accessible color palettes).
**Rationale:** Appropriate for science fair presentation and potential publication.

## 8. Computational Constraints

### Assumption 8.1: Runtime Budget
**Decision:** Target total runtime of <4 hours on CPU for all experiments.
**Rationale:** Practical for student researcher environment while allowing sufficient model iterations.
**Mitigation:** Use efficient implementations, limit hyperparameter search space, cache intermediate results.

### Assumption 8.2: Memory Management
**Decision:** Process data in chunks where necessary to maintain <8GB RAM usage.
**Rationale:** Ensure compatibility with consumer hardware.

## 9. Handling Insufficient Data

### Assumption 9.1: Minimum Viable Dataset
**Decision:** If real microplastic observations fall below 50 valid samples after quality control, the project will:
1. Document the limitation transparently
2. Fall back to a simpler modeling approach (no temporal component, basic graph)
3. Clearly label any synthetic data experiments as separate from the real-data experiment
**Rationale:** Maintains scientific integrity while producing meaningful results.

### Assumption 9.2: Cross-Study Integration
**Decision:** When combining datasets from different studies, apply strict unit conversion and document the heterogeneity. Studies with incompatible units (e.g., only mass concentration without volumetric data) will be excluded from the primary analysis.
**Rationale:** Scientifically defensible while maximizing usable data.

## 10. Uncertainty Quantification

### Assumption 10.1: Ensemble Size
**Decision:** Use 5 ensemble members as specified in Stage 1.
**Rationale:** Balances uncertainty estimation quality with computational cost.
**Trade-off:** More members would provide better uncertainty estimates but increase runtime.

### Assumption 10.2: Coverage Threshold
**Decision:** Report coverage at the nominal 95% level. No calibration adjustment will be applied unless coverage deviates by more than ±10 percentage points.
**Rationale:** Raw ensemble coverage is more transparent for evaluation.
