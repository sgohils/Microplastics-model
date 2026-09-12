# MICROPLASTIC TRANSPORT GNN RESEARCH & SYSTEM SPECIFICATION

A complete scientific blueprint for developing a spatiotemporal graph neural network to predict microplastic transport in river systems.

---

## 1. EXECUTIVE SUMMARY

This document specifies a complete scientific research blueprint for predicting microplastic transport through river networks using a spatiotemporal graph neural network (GNN). The project compares the predictive power of physically-connected river-network topology against conventional machine learning baselines, testing whether explicit representation of hydrological connectivity improves forecasting of microplastic concentrations at unseen locations and times.

Based on data availability assessment, the project will focus on the Delaware River Basin as the primary study region, leveraging USGS microplastic observation datasets (Delaware River 2018, Great Lakes tributaries 2014-2015, northeastern U.S. streams 2017-2018), NHDPlus river network data, USGS streamflow data, Daymet meteorological data, and multispectral land cover data.

The specification defines a rigorous methodology with 12 controlled experiments including graph ablation, topology randomization controls, spatial/temporal holdouts, sparse-data degradation studies, and uncertainty quantification. The document establishes strict scientific rules including prohibition of data fabrication, mandatory train/test separation, and clear distinction between observed, derived, and synthetic data.

---

## 2. SCIENTIFIC PROBLEM

### Analysis of Research Question Feasibility

**Original Question:** *Can a spatiotemporal graph neural network incorporating river-network connectivity, hydrological conditions, meteorological conditions, and geographic characteristics predict microplastic pollution more accurately than conventional machine-learning models at previously unseen locations and times?*

#### Scientific Significance
- Microplastic pollution is a critical global environmental concern affecting aquatic ecosystems, food webs, and human health
- River systems are the primary conduits transporting microplastics from terrestrial sources to oceans
- Understanding and predicting microplastic transport patterns is essential for developing targeted monitoring strategies and remediation policies
- Current predictive approaches rely heavily on physically-based hydrodynamic models that are computationally expensive and often inadequate for real-time forecasting

#### Feasibility
**Moderately Feasible with Important Caveats:**
- **Data Constraints:** Publicly available microplastic observations are limited in quantity, spatial coverage, and temporal frequency. The Delaware River dataset contains 9 sampling locations with single sampling events (July 2018 - March 2019), yielding approximately 9 water samples with concentration measurements. This represents a very small dataset.
- **Temporal Limitations:** Most microplastic datasets are cross-sectional (single time-point sampling) rather than time series, limiting the ability to train true spatiotemporal models.
- **Spatial Constraints:** Datasets are typically watershed-specific, making spatial generalization testing possible but temporally constrained.

#### Available Data (Confirmed Sources)
1. **USGS Microplastic Observations:**
   - Delaware River, 2018 (9 locations, water + sediment samples)
   - Great Lakes tributaries, 2014-2015 (29 tributaries, multiple timepoints)
   - Northeastern U.S. streams, 2017-2018 (17 streams)
   - St. Croix/Mississippi National River, 2015 (8 locations)

2. **River Network Topology:** NHDPlus v2 and NHDPlus HR (1:100,000 and 1:24,000 scale)

3. **Hydrology:** USGS Water Services API, NWIS (streamflow, discharge)

4. **Meteorology:** Daymet (daily 1km resolution gridded data)

5. **Geography:** NLCD land cover, NED elevation, WGS climate data

#### Measurable Variables
- **Primary target:** Microplastic concentration (particles per cubic meter or particles per liter)
- **Predictor variables:** Discharge, precipitation, temperature, elevation, slope, drainage area, land cover, flow velocity, stream order
- **Graph topology:** River connectivity from NHDPlus flow direction networks

#### Confounding Factors
- **Sampling methodology differences:** Different studies use different collection methods (nets, grab samples), analysis techniques (FTIR, visual identification), and size classifications
- **Seasonality:** Most samples collected during specific seasons, not year-round
- **Spatial heterogeneity:** Microplastic concentrations vary significantly within short distances due to point sources
- **Temporal mismatch:** Environmental data collection frequency (daily) far exceeds sampling frequency of microplastic measurements
- **Non-stationarity:** Land use changes, population growth, and climate change affect baselines over time

#### Potential Sources of Bias
- **Selection bias:** Sampling sites often chosen for accessibility, not representativeness
- **Temporal aggregation bias:** Daily/monthly environmental aggregates may miss short-duration events that drive microplastic transport
- **Spatial interpolation bias:** Environmental covariates interpolated to river segments may not represent local conditions
- **Publication bias:** Available datasets may over-represent certain regions or conditions

#### Expected Difficulty
**High:** The fundamental challenge is that microplastic observations are sparse, expensive to collect, and methodologically heterogeneous. Training a data-hungry deep learning model on such limited data requires careful regularization and ablation study design.

#### Novelty Assessment
- **GNN for microplastic prediction:** Novel in combining river topology with microplastic prediction
- **Graph ablation approach:** Novel comparison between physical connectivity and geographic proximity specifically for microplastics
- **Sparse-data robustness:** Novel investigation of whether GNN topology helps in data-scarce scenarios
- **NOT** the first use of GNNs for river networks (extensive prior work exists for streamflow prediction)
- **NOT** the first application of ML to water quality prediction

#### Refinement of Research Question

The original question is scientifically sound but needs refinement to account for data limitations:

**Refined Research Question:** *Does explicitly representing physical river-network connectivity in a graph neural network architecture improve prediction of microplastic concentrations compared with conventional machine learning models that rely only on geographic proximity and environmental covariates, when evaluated on held-out spatial and temporal data?*

---

## 3. RESEARCH QUESTIONS

### Primary Research Question

Does explicit representation of physical river-network connectivity in a spatiotemporal graph neural network architecture provide measurably superior prediction of microplastic concentrations compared to conventional machine learning approaches that rely on geographic proximity and environmental covariates, when evaluated on previously unseen watersheds and time periods?

### Secondary Research Questions

1. Does incorporating river-network topology improve prediction accuracy compared to geographic nearest-neighbor or fully-connected graph representations?
2. Does adding temporal information (lagged environmental conditions) improve microplastic concentration predictions?
3. How does model performance degrade as training observations are progressively reduced, and does the graph-based approach maintain superiority under sparse data conditions?
4. Can the model generalize to previously unseen watersheds, and does graph-based modeling improve spatial generalization?
5. Which environmental variables contribute most strongly to prediction accuracy as measured by permutation importance and SHAP values?
6. Does prediction uncertainty increase in poorly monitored regions, unseen watersheds, and during extreme weather events?
7. Does randomized river topology provide any predictive benefit over no graph structure, indicating whether meaningful topology matters?

---

## 4. HYPOTHESES

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

---

## 5. SCIENTIFIC SIGNIFICANCE

This project addresses a critical gap in environmental monitoring: the ability to predict microplastic transport in data-sparse river systems. Unlike existing approaches that rely on computationally expensive hydrodynamic simulations or treat monitoring stations independently, this project tests whether graph-based representations of river networks can encode connectivity information that improves predictive accuracy.

The scientific contribution lies in:
1. Empirically testing whether physical river connectivity provides meaningful signal beyond geographic proximity
2. Investigating the conditions under which graph representations are beneficial for environmental prediction
3. Providing methodological insights transferable to other spatiotemporal environmental prediction tasks

This is particularly relevant for high-school science fair competition because:
- The project directly addresses an environmental crisis (microplastic pollution) that affects local communities
- The methodology is accessible yet scientifically rigorous
- The results have practical implications for environmental monitoring and policy

---

## 6. LITERATURE REVIEW STRATEGY

### Microplastic Transport Literature

**Search Areas:**
- Freshwater microplastic transport mechanisms in river systems
- Downstream transport patterns and retention processes
- Influence of hydrological conditions on microplastic movement
- Relationship between river discharge and microplastic concentrations
- Seasonal variability in microplastic distributions

**Key Sources to Consult:**
- *Modeling the transport of microplastics along river networks* (Portillo De Arbeloa & Marzadri, Science of the Total Environment, 2024) - Integrates advection-dispersion equation with anthropogenic loads
- *Microplastics in the Delaware River* (Baldwin et al., USGS Fact Sheet 2020-3071) - Provides baseline data and methodology
- *Microplastics profile along the Rhine River* (Scientific Reports, 2015) - Demonstrates downstream transport patterns
- *River plastic emissions to the world's oceans* (PMC, 2017) - Global context for riverine transport

### Machine Learning in Environmental Science Literature

**Search Areas:**
- Environmental pollution prediction using machine learning
- Water quality prediction models
- Spatial machine learning for environmental data
- Time-series prediction in hydrological contexts

**Key Sources:**
- *Spatiotemporal prediction of water quality and ecological risk assessment in a river basin using T-GCN based on river network topology constraints* (Scientific Reports, 2026) - Directly relevant T-GCN application
- Various Random Forest and XGBoost applications in water quality prediction

### Graph Neural Networks for Hydrology Literature

**Search Areas:**
- Graph neural networks for river network modeling
- Physics-guided GNNs for hydrological applications
- Spatiotemporal graph neural networks for environmental forecasting
- Traffic/network forecasting techniques adapted for hydrology
- River network topology representation in GNN architectures

**Key Sources:**
- *A graph neural network (GNN) approach to basin-scale river network learning* (Hydrol. Earth Syst. Sci., 2022) - Established precedent for GNN in river networks
- *Spatiotemporal graph neural networks for analyzing the influence mechanisms of river hydrodynamics on microplastic transport processes* (Scientific Reports, 2025) - Most directly relevant recent work
- *The Merit of River Network Topology for Neural Flood Forecasting* (ICML 2024) - Important cautionary note about topology benefits
- *HydroGAT: Distributed Heterogeneous Graph Attention Transformer* (ACM GIS 2025) - State-of-the-art architecture reference

### Scientific Gap Analysis

**What Has Been Done:**
- Physics-informed models exist for microplastic transport (advection-dispersion equations)
- GNNs have been applied to river network modeling for streamflow prediction
- T-GCN architectures have been used for water quality prediction with topology constraints
- A recent 2025 paper specifically used spatiotemporal GNN for microplastic transport prediction

**What Remains Uncertain:**
- Whether explicit river topology provides benefit over geographic proximity specifically for microplastic prediction (the 2025 paper found R² > 0.89, but direct comparison with baselines was not emphasized)
- How model performance degrades under sparse data conditions
- Whether the benefit is consistent across different environmental conditions
- Quantification of upstream influence on downstream predictions

**What This Project Can Test:**
- Direct A/B comparison of physical topology vs. geographic proximity graphs
- Graph randomization controls to test topology specificity
- Sparse-data robustness of graph-based approaches
- Transfer of GNN techniques from hydrology to microplastic prediction in an accessible, science-fair-appropriate manner

---

## 7. DATA FEASIBILITY AUDIT

### Microplastic Observations

**Dataset 1: Delaware River Microplastics, 2018**
- Source: USGS, ScienceBase
- DOI: 10.5066/P9QVIVX3
- Spatial Coverage: Delaware River and tributaries (New York, Pennsylvania, New Jersey, Delaware)
- Temporal Coverage: July 2018 - March 2019
- Number of Observations: 9 water sampling locations
- Variables: Particle count, concentration (particles/m³), particle type, morphology
- Units: particles per cubic meter (water), particles per kg dry weight (sediment)
- Sampling Methodology: Grab sampling and net sampling
- Data Access: https://doi.org/10.5066/P9QVIVX3 (CSV format)
- Known Biases: Limited spatial/temporal coverage, baseflow conditions only

**Dataset 2: Great Lakes Tributaries Microplastics, 2014-2015**
- Source: USGS
- Spatial Coverage: 29 tributaries across 6 states (Great Lakes region)
- Temporal Coverage: Spring 2014 - Spring 2015
- Number of Observations: ~120 samples (multiple per tributary)
- Variables: Particle count, concentration (particles/km²), morphology
- Temporal Resolution: Multiple sampling events per location
- Units: particles per km², particles per m³
- Sampling Methodology: Manta trawl, surface sampling
- Data Access: Available via USGS ScienceBase

**Dataset 3: Northeastern U.S. Streams, 2017-2018**
- Source: USGS
- Spatial Coverage: 17 streams from New York to Virginia
- Temporal Coverage: 2017-2018
- Number of Observations: 17 locations
- Units: particles/L (reported in some literature)
- Sampling Methodology: Various USGS protocols

**Combined Assessment:**
- Total estimated observations: ~150-200 distinct sampling events
- Geographic distribution: Primarily U.S. Northeast and Great Lakes
- Temporal distribution: Cross-sectional studies, not continuous monitoring
- Critical limitation: No dataset provides both high spatial density and temporal continuity sufficient for training a robust spatiotemporal GNN from real observations alone

### Hydrology Data

**USGS Water Services API (NWIS)**
- Source: https://api.waterdata.usgs.gov/
- Variables: Discharge (cfs), gage height (ft), water temperature (°C)
- Spatial Coverage: 13,500+ monitoring stations nationwide
- Temporal Coverage: Continuous real-time data at many stations, with historical records spanning decades
- Temporal Resolution: 15-minute to daily
- Units: Cubic feet per second (cfs), feet (ft)
- Known Biases: Streamgages not co-located with microplastic sampling sites
- Compatibility: Excellent - same geographic regions

**NHDPlus v2 Streamflow Estimates**
- Source: EPA/USGS
- Variables: Mean annual/monthly discharge, velocity, slope
- Spatial Resolution: 1:100,000 scale
- Temporal Coverage: Long-term averages
- Units: Various (cubic meters per second, meters per second)
- Known Biases: Modeled estimates, not direct observations
- Compatibility: Good for static features

### Meteorology Data

**Daymet**
- Source: Oak Ridge National Laboratory (https://daymet.ornl.gov/)
- Variables: Precipitation, max/min temperature, shortwave radiation, humidity, snow water equivalent
- Spatial Coverage: North America
- Spatial Resolution: 1 km x 1 km
- Temporal Coverage: 1980-present (daily)
- Temporal Resolution: Daily
- Units: Precipitation (mm), temperature (°C), radiation (W/m²)
- License: Public domain
- Known Biases: Gridded interpolation may smooth local extremes
- Compatibility: Excellent - can extract data for any river location

### Geography Data

**National Land Cover Database (NLCD)**
- Source: USGS (https://www.usgs.gov/landsat-missions/land-cover)
- Variables: Land cover classification (22 classes), impervious surface
- Spatial Resolution: 30m (CONUS), 10m (NLCD 2016+)
- Temporal Coverage: 2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021
- Spatial Coverage: CONUS
- Units: Categorical land cover classes, percentage impervious surface
- License: Public domain
- Known Biases: Static over short time periods, classification accuracy varies

**National Elevation Dataset (NED)**
- Source: USGS
- Variables: Elevation, slope, aspect
- Spatial Resolution: 10m (CONUS)
- Spatial Coverage: CONUS
- Units: Meters
- License: Public domain

**NHDPlus HR Catchment Characteristics**
- Source: USGS
- Variables: Drainage area, stream order, river slope, flow accumulation
- Spatial Resolution: 1:24,000 scale
- License: Public domain
- Compatibility: Built for river network analysis

### River Network Data

**NHDPlus v2**
- Source: EPA/USGS (https://www.epa.gov/waterdata/nhdplus-national-hydrography-dataset-plus)
- Variables: Stream segments, connectivity, flow direction, stream order, slope, velocity
- Spatial Resolution: 1:100,000 scale
- Spatial Coverage: CONUS
- Temporal Coverage: Static snapshots
- Format: Shapefile, File Geodatabase
- License: Public domain
- Known Biases: Scale limitations may miss small tributaries

**NHDPlus HR**
- Source: USGS (https://www.usgs.gov/national-hydrography/access-national-hydrography-products)
- Variables: High-resolution stream network, catchments, flow direction
- Spatial Resolution: 1:24,000 scale
- Spatial Coverage: CONUS, Hawaii, Puerto Rico, Guam
- Format: File Geodatabase
- License: Public domain
- Known Biases: Irregular update schedule

---

## 8. TARGET VARIABLE

### Selected Target Variable

**Microplastic concentration in river water (particles per cubic meter)**

### Rationale for Selection

1. **Scientific Relevance:** Water column concentration is the most relevant metric for understanding transport dynamics and ecological exposure
2. **Data Availability:** Available from USGS Delaware River dataset (particles/m³) and other sources
3. **Direct Measurement:** Represents actual microplastic burden, not inferred from proxies
4. **Comparability:** Multiple datasets report this metric (though units vary)

### Measurement Method

1. **Collection:** Water samples collected via grab sampling or net tows
2. **Processing:** Filtration, digestion of organic matter, identification via FTIR spectroscopy
3. **Quantification:** Particle count per unit volume
4. **Units:** Particles per cubic meter (particles/m³)

### Expected Distribution

- **Highly right-skewed:** Most measurements will be low, with occasional high spikes
- **Range:** Expected from ~1 to ~1,000+ particles/m³ based on literature
- **Spatial variability:** Significant variation due to point sources, urban runoff
- **Temporal variability:** Higher concentrations expected during storm events

### Transformation

- **Log transformation recommended:** Apply log1p (log(1+x)) transformation to normalize distribution
- **Justification:** Microplastic concentrations are positive and right-skewed
- **Validation:** Q-Q plots and Shapiro-Wilk tests on transformed values

### Limitations

1. **Unit inconsistency:** Different studies report different units (particles/m³, particles/L, particles/km²)
2. **Methodological heterogeneity:** Different sampling and analysis protocols affect comparability
3. **Detection limits:** Some studies report "below detection limit" as zero
4. **Temporal mismatch:** Point-in-time measurements vs. continuous environmental data

### Handling Incompatible Units

- **Standardization approach:** Convert all measurements to particles per cubic meter (particles/m³)
  - 1 particles/L = 1,000 particles/m³
- **Documentation requirement:** Record original units for each observation
- **Quality filtering:** Exclude observations with insufficient methodological metadata

---

## 9. STUDY REGION

### Selected Study Area: Delaware River Basin

**Rationale:**
1. **Data Availability:** The USGS Delaware River microplastic dataset provides the most comprehensive microplastic observations in a single watershed
2. **Infrastructure Data:** Extensive USGS streamgaging network (20+ stations)
3. **Meteorological Coverage:** Daymet provides complete daily data for the entire basin
4. **River Network Quality:** NHDPlus v2 and HR available with full topology
5. **Geographic Diversity:** Basin spans from headwaters in New York to tidal reaches in Delaware, encompassing urban, suburban, and rural environments

**Spatial Extent:** Approximately 13,500 km² watershed covering parts of New York, New Jersey, Pennsylvania, and Delaware

**Temporal Focus:** July 2018 - March 2019 (matching the microplastic sampling period)

### Alternative Datasets for Validation

1. **Great Lakes Tributaries (2014-2015):** 29 tributaries across 6 states - can serve as secondary spatial validation
2. **Northeastern U.S. Streams (2017-2018):** 17 streams - additional geographic diversity

### Future Expansion Strategy

1. **Phase 1:** Delaware River Basin focused analysis
2. **Phase 2:** Add Great Lakes tributary data for cross-basin validation
3. **Phase 3:** Incorporate synthetic data generation for methodological testing (explicitly labeled as synthetic)
4. **Phase 4:** Expand to other well-monitored basins if data becomes available

---

## 10. FEATURE SPECIFICATION

### Node Features

#### Hydrological Features
| Feature | Unit | Source | Spatial Resolution | Temporal Resolution | Expected Relevance | Leakage Risk |
|---------|------|--------|-------------------|-------------------|-------------------|--------------|
| Discharge | m³/s (converted from cfs) | USGS NWIS | Point (streamgage) | Daily | High - drives transport | Medium - must align temporally |
| Flow velocity | m/s | NHDPlus HR | Reach-level | Static (annual mean) | High - particle transport rate | Low |
| Drainage area | km² | NHDPlus HR | Reach-level | Static | High - correlates with flow | Low |
| Stream order | dimensionless | NHDPlus HR | Reach-level | Static | Medium - indicates connectivity | Low |
| River slope | degrees | NHDPlus HR | Reach-level | Static | Medium - affects deposition | Low |
| Gage height | m (converted from ft) | USGS NWIS | Point | Daily | Medium | Medium |

#### Meteorological Features
| Feature | Unit | Source | Spatial Resolution | Temporal Resolution | Expected Relevance | Leakage Risk |
|---------|------|--------|-------------------|-------------------|-------------------|--------------|
| Precipitation | mm | Daymet | 1 km² | Daily | High - source and transport driver | Low if lagged properly |
| Cumulative precipitation (7-day) | mm | Daymet | 1 km² | Daily | High - antecedent moisture | Low |
| Max temperature | °C | Daymet | 1 km² | Daily | Medium - affects plastic properties | Low |
| Wind speed | m/s | Daymet | 1 km² | Daily | Low-Medium | Low |
| Snow water equivalent | mm | Daymet | 1 km² | Daily | Medium - seasonal storage | Low |

#### Geographic Features
| Feature | Unit | Source | Spatial Resolution | Temporal Resolution | Expected Relevance | Leakage Risk |
|---------|------|--------|-------------------|-------------------|-------------------|--------------|
| Elevation | meters | NED | 10m | Static | High - gravity-driven flow | Low |
| Land cover (forest) | % | NLCD | 30m | Static | Medium | Low |
| Land cover (urban) | % | NLCD | 30m | Static | High - microplastic source | Low |
| Impervious surface | % | NLCD | 30m | Static | High - runoff generation | Low |
| Population density | people/km² | Gridded census | 1 km² | Static | High - pollution source | Low |
| Slope | degrees | NED-derived | 10m | Static | Medium | Low |

#### Temporal Features
| Feature | Unit | Source | Temporal Resolution | Expected Relevance | Leakage Risk |
|---------|------|--------|-------------------|-------------------|--------------|
| Month | 1-12 | Calendar | Daily | High - seasonality | None |
| Season | categorical | Calendar | Daily | High - seasonal patterns | None |
| Year | numeric | Calendar | Daily | Medium - trend detection | Low |
| Lagged precipitation (1d, 3d, 7d) | mm | Computed | Daily | High - antecedent conditions | Medium - must fit on train only |
| Lagged discharge (1d, 3d, 7d) | m³/s | USGS | Daily | High - flow memory | Medium - must fit on train only |

#### Pollution Features (Conditional Use)
| Feature | Unit | Source | Leakage Risk |
|---------|------|--------|--------------|
| Upstream microplastic concentration | particles/m³ | Same dataset | HIGH - target leakage |
| Downstream microplastic concentration | particles/m³ | Same dataset | HIGH - target leakage |

**Important:** Upstream/downstream microplastic measurements must NOT be used as features due to target leakage. Only environmental covariates may be used as predictors.

### Feature Engineering Pipeline

1. **Spatial Aggregation:** For each river segment node, extract:
   - Mean/max/min values from intersecting grid cells (Daymet, NLCD, NED)
   - Upstream accumulated values (drainage area, population)
   - Nearest streamgage values for discharge/temperature

2. **Temporal Alignment:** Match all features to the date of microplastic sampling

3. **Lag Features:** Compute 1-day, 3-day, and 7-day lags for:
   - Precipitation (cumulative and daily averages)
   - Discharge (from nearest streamgage)

4. **Normalization:** Apply standardization (z-score) fitted only on training data

---

## 11. TEMPORAL DESIGN

### Temporal Granularity

**Primary: Daily alignment** with microplastic sampling events as observation timestamps.

### Justification

1. **Microplastic sampling frequency:** Typically monthly or less frequent
2. **Meteorological data:** Available daily from Daymet
3. **Streamflow data:** Available daily from USGS
4. **Computational efficiency:** Daily resolution manageable on consumer hardware

### Lag Window Design

| Lag Period | Rationale |
|------------|-----------|
| 1 day | Immediate antecedent conditions |
| 3 days | Short-term memory of system |
| 7 days | Weekly hydrological cycle |
| 30 days | Monthly integration of conditions |

### Temporal Experiments

1. **No temporal features:** Only static geographic and instantaneous environmental features
2. **1-day lags:** Includes 1-day lagged precipitation and discharge
3. **7-day lags:** Includes 1, 3, and 7-day lagged features
4. **30-day lags:** Full temporal history features

Lag selection will be determined empirically through ablation experiments, not arbitrarily specified.

---

## 12. DATA CLEANING PLAN

### Data Validation Steps

1. **Microplastic Data Validation:**
   - Verify units are consistent (particles/m³)
   - Remove samples with insufficient methodological metadata
   - Check for physically impossible values (negative concentrations)
   - Document original units for all converted values

2. **Hydrological Data Validation:**
   - Remove streamgage readings with quality flags
   - Interpolate missing daily values using linear interpolation between adjacent valid measurements
   - Flag and handle zero-flow readings appropriately

3. **Meteorological Data Validation:**
   - Verify Daymet data covers all sampling dates
   - Handle missing grid cells through nearest-neighbor interpolation
   - Remove extreme outliers (> 5 standard deviations from station mean)

4. **Geospatial Data Validation:**
   - Verify all river segments have valid coordinates
   - Check NHDPlus connectivity for topological consistency
   - Validate DEM-derived slope calculations

### Missing Data Handling

1. **Streamgage data:** If a streamgage lacks measurements on a sampling date, interpolate from adjacent dates. If >7 days missing, exclude from that node's features.
2. **Daymet data:** Grid cells missing data will be filled using inverse-distance weighting from neighboring cells.
3. **Microplastic observations:** Samples without concentration measurements will be excluded.
4. **Land cover data:** NLCD data is static and complete for CONUS.

### Quality Control Thresholds

- Microplastic samples with particle counts < 5 will be flagged as potentially below detection limit
- Streamflow measurements with quality "Poor" rating will be excluded
- Temperature readings > 35°C or < -5°C will be checked against neighboring stations

---

## 13. DATA LEAKAGE PREVENTION

### Identified Leakage Risks and Mitigation Strategies

#### 1. Future Information in Temporal Features
- **Risk:** Using future precipitation or discharge data to predict past microplastic concentrations
- **Mitigation:** All temporal features must be lagged by at least 1 day. For 7-day cumulative precipitation, the most recent day included must be before the sampling date.

#### 2. Target-Derived Features
- **Risk:** Using upstream or downstream microplastic measurements as features
- **Mitigation:** No microplastic data from any location will be used as predictive features. Only environmental covariates are permitted.

#### 3. Cross-Contamination via Environmental Data Interpolation
- **Risk:** Interpolating streamgage data to river segments that are actually in the test set
- **Mitigation:** Environmental data interpolation will use only training-set streamgages to avoid information transfer.

#### 4. Preprocessing Leakage
- **Risk:** Fitting scalers, encoders, or imputation models on full dataset
- **Mitigation:** All data transformations will be fitted only on training data and then applied to validation/test data. This includes StandardScaler, MinMaxScaler, and any categorical encoders.

#### 5. Spatial Leakage via Graph Construction
- **Risk:** Including edges from test locations to train locations in the graph, allowing indirect information flow
- **Mitigation:** For spatial holdout experiments, the graph will be constructed with only training-set nodes plus their direct upstream/downstream neighbors. Test nodes will have their immediate neighborhood included but no distant test-to-test connections.

#### 6. Sampling Date Leakage
- **Risk:** Using environmental conditions on the exact sampling date (which includes the day's precipitation/discharge that may correlate with the sampling event itself)
- **Mitigation:** Features represent conditions as of the end of the previous day, not the sampling date.

#### 7. Duplicate Observations
- **Risk:** Same location sampled multiple times, creating pseudo-replication
- **Mitigation:** Verify each (location, date) combination is unique. If duplicates exist, retain only the first observation.

#### 8. Station Normalization
- **Risk:** Normalizing streamflow by station mean (leaking all data including future)
- **Mitigation:** Station-specific normalization parameters computed from training period only.

---

## 14. TRAIN/VALIDATION/TEST DESIGN

### Experiment A: Random Split (70/15/15)
- **Purpose:** Establish baseline performance under ideal conditions
- **Method:** Randomly shuffle all observations, split into train/validation/test
- **Rationale:** Standard ML benchmark; establishes upper bound on achievable performance
- **Warning:** This is NOT the primary scientific experiment due to high risk of spatial/temporal leakage

### Experiment B: Spatial Holdout
- **Purpose:** Test model generalization to unseen geography
- **Method:** Leave one major tributary/sub-watershed completely out of training
- **Example:** Train on Delaware River main stem + western tributaries, test on eastern tributaries
- **Rationale:** Determines whether the model can predict in locations where it has never seen microplastic data

### Experiment C: Temporal Holdout
- **Purpose:** Test model generalization to unseen time periods
- **Method:** Use first 80% of temporal data for training, last 20% for testing
- **Example:** Train on Jul-Dec 2018, test on Jan-Mar 2019
- **Rationale:** Tests whether learned patterns generalize to future conditions

### Experiment D: Spatiotemporal Holdout (Most Critical)
- **Purpose:** Test hardest generalization scenario
- **Method:** Hold out both geographic regions AND time periods
- **Example:** Train on western Delaware River (Jan 2019), test on eastern Delaware River (July 2018)
- **Rationale:** This is the strongest test of whether river network topology provides meaningful information

### Additional Split Strategies

**K-Fold Cross-Validation (Temporal-aware):**
- For all experiments, use 5-fold cross-validation with random seeds
- For temporal holdout experiments, ensure folds respect temporal ordering
- Report mean and standard deviation across folds

**Small Dataset Considerations:**
- Given limited observations (~150-200), use leave-one-location-out cross-validation for spatial generalization tests
- Report confidence intervals using bootstrap resampling

---

## 15. BASELINE MODELS

### Baseline 1: Mean/Median Predictor
- **Implementation:** Predict the training set mean/median for all test observations
- **Rationale:** Establishes absolute minimum performance; any reasonable model should outperform this
- **Metrics:** MAE, RMSE, R²

### Baseline 2: Ridge Regression
- **Implementation:** Linear regression with L2 regularization, features standardized
- **Rationale:** Linear relationships between environmental covariates and microplastic concentration are theoretically motivated
- **Parameters:** α tuned via grid search {0.01, 0.1, 1.0, 10.0, 100.0}
- **Justification:** Simple, interpretable baseline that handles collinearity

### Baseline 3: Random Forest
- **Implementation:** scikit-learn RandomForestRegressor
- **Rationale:** Non-linear relationships, handles mixed feature types, robust to outliers
- **Parameters:** n_estimators=500, max_depth tuned {10, 20, 50, None}, max_features="sqrt"
- **Justification:** Strong general-purpose baseline, widely used in environmental ML

### Baseline 4: XGBoost
- **Implementation:** XGBoost regressor
- **Rationale:** State-of-the-art gradient boosting, typically best-performing tabular baseline
- **Parameters:** learning_rate=0.01, max_depth=6, n_estimators=1000, early_stopping_rounds=50
- **Justification:** Must be included as the strongest conventional baseline; GNN should not be compared only against weak models

### Baseline 5: Multilayer Perceptron (MLP)
- **Implementation:** scikit-learn MLPRegressor
- **Rationale:** Neural network without graph structure, tests whether GNN benefits come from graph or just neural architecture
- **Parameters:** hidden_layer_sizes=(100, 50), activation="relu", solver="adam", alpha=0.01
- **Justification:** Tests whether benefits are specifically from graph structure vs. neural representation learning

---

## 16. PROPOSED MODEL ARCHITECTURE

### Architecture: Spatiotemporal Graph Neural Network with Message Passing

```
Environmental Features (per node, per timestep)
        ↓
Feature Encoder (MLP: Linear → ReLU → Linear)
        ↓
Temporal Encoder (GRU: captures temporal dependencies)
        ↓
Graph Message Passing (GCN/GraphSAGE layers)
        ↓
Node Representation Aggregator
        ↓
Prediction Head (Linear → ReLU → Linear → Output)
        ↓
Microplastic Concentration Prediction
```

### Detailed Component Design

#### 1. Feature Encoder
- **Type:** Multi-layer perceptron (2 layers)
- **Input:** Node features at each timestep (dimension varies based on feature selection)
- **Hidden dimension:** 64
- **Activation:** ReLU
- **Purpose:** Projects heterogeneous features into a common embedding space

#### 2. Temporal Encoder
- **Type:** Gated Recurrent Unit (GRU)
- **Input:** Sequence of encoded feature vectors
- **Hidden dimension:** 64
- **Sequence length:** 7 timesteps (7-day history)
- **Purpose:** Captures temporal dependencies in environmental conditions

#### 3. Graph Message Passing
- **Type:** 2-layer GraphSAGE (GraphSAC)
- **Hidden dimension:** 64 → 32
- **Aggregation:** Mean aggregation with ReLU activation
- **Directionality:** Directed edges following river flow direction
- **Edge features:** Optional - can include distance, flow direction
- **Purpose:** Propagates information along river connectivity network

#### 4. Prediction Head
- **Type:** MLP regression head
- **Architecture:** Linear(32) → ReLU → Linear(16) → ReLU → Linear(1)
- **Output:** Log-transformed microplastic concentration
- **Purpose:** Maps final node representation to concentration prediction

### Architecture Justification

**Why GraphSAGE over GCN/GAT:**
- GraphSAGE is simpler and more computationally efficient
- Does not require the full adjacency matrix (scales better)
- Mean aggregation naturally captures upstream/downstream averaging
- Avoids attention computation overhead

**Why GRU over LSTM:**
- Simpler architecture with fewer parameters
- Computationally lighter for small datasets
- GRU has been successfully applied in temporal GNN literature (T-GCN)

**Why 2 message-passing layers:**
- Sufficient receptive field given typical river network depth (2-3 degrees of upstream/downstream connectivity per node)
- Deeper layers lead to over-smoothing on small networks
- Empirically validated in similar GNN applications

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 0.001 | Standard Adam default |
| Batch size | 32 | Balance between efficiency and gradient stability |
| Hidden dimension | 64 | Sufficient capacity without overfitting |
| Message passing layers | 2 | Appropriate receptive field |
| Dropout | 0.3 | Regularization for small dataset |
| L2 weight decay | 0.0001 | Prevent overfitting |
| Epochs | 500 (with early stopping) | Sufficient training with patience=30 |

---

## 17. UNCERTAINTY METHOD

### Approach: Deep Ensembles

**Rationale:**
- Simple to implement and well-understood
- Provides both aleatoric and epistemic uncertainty
- Computationally feasible for small models
- No additional architectural complexity

### Implementation Details

1. **Number of ensemble members:** 5
2. **Training:** Each member trained with different random initialization and different train/validation split
3. **Prediction:** Mean of ensemble predictions as final prediction
4. **Uncertainty estimate:** Standard deviation across ensemble members
5. **Prediction intervals:** 95% interval = [mean - 1.96×std, mean + 1.96×std]

### Uncertainty Analysis Experiments

1. **Well-monitored regions:** Compare uncertainty at locations with >3 observations
2. **Sparse regions:** Compare uncertainty at locations with ≤3 observations
3. **Unseen watersheds:** Compare uncertainty on spatial holdout data
4. **Extreme events:** Compare uncertainty during high-flow vs. normal flow conditions
5. **Calibration testing:** Verify that 95% prediction intervals contain true values ~95% of the time

---

## 18. ABLATION EXPERIMENTS

### Model Progression Study

#### Model A: Environmental Only
- **Features:** Static geographic + instantaneous environmental conditions
- **Temporal:** No lag features (current day only)
- **Graph:** No graph structure (MLP or flat feature vector)
- **Purpose:** Establishes baseline with minimal information

#### Model B: Environmental + Geographic
- **Features:** Model A + lagged precipitation, discharge, and temporal calendar features
- **Temporal:** 7-day lagged features
- **Graph:** No graph structure
- **Purpose:** Tests value of temporal information alone

#### Model C: Environmental + Graph
- **Features:** Model B
- **Graph:** River connectivity graph with message passing
- **Temporal:** No temporal encoder
- **Purpose:** Tests whether static graph information adds value

#### Model D: Environmental + Graph + Temporal
- **Features:** Model B
- **Graph:** River connectivity graph with message passing
- **Temporal:** GRU-based temporal encoder
- **Purpose:** Full model with graph and temporal integration

#### Model E: Full Spatiotemporal GNN (Proposed Model)
- **Features:** All available features including engineered lags
- **Graph:** River connectivity with 2-layer GraphSAGE
- **Temporal:** GRU encoder with 7-timestep history
- **Uncertainty:** Deep ensemble (5 members)
- **Purpose:** Proposed full architecture

#### Optional Model F: Full Model + Physics Constraint
- **Note:** Physics-informed constraints are NOT included unless validated against established literature. The advection-dispersion relationship between upstream and downstream concentrations could potentially inform a consistency loss, but this is deferred pending further investigation.

### Conclusion Testing

| Comparison | Scientific Conclusion |
|------------|----------------------|
| A vs. B | Does temporal information improve predictions? |
| B vs. C | Does static graph information improve predictions? |
| C vs. D | Does temporal encoding add beyond static graph? |
| D vs. E | Does full architecture improve over partial components? |
| A vs. E | Overall improvement of proposed model vs. minimal model |

---

## 19. GRAPH CONTROL EXPERIMENTS

### Experiment 17a: Graph Structure Comparison

| Graph Type | Node Definition | Edge Definition | Edge Weighting |
|------------|-----------------|-----------------|----------------|
| No Graph | River segment | None | None |
| Geographic KNN | Monitoring locations | k=5 nearest neighbors (Euclidean) | Distance-based |
| Haversine Threshold | Monitoring locations | All nodes within 50 km | Inverse distance |
| **River Connectivity** | **River segments** | **NHDPlus flow direction** | **Flow accumulation** |

**Purpose:** Determine whether physical connectivity provides information beyond geographic proximity.

### Experiment 17b: Topology Randomization Control

**Control Method:**
1. Preserve node features and edge weights
2. Randomly shuffle edge connections while maintaining:
   - Same number of edges
   - Same in-degree and out-degree distribution
   - No self-loops
   - No duplicate edges
3. Use configuration model for degree-preserving randomization

**Purpose:** Test whether benefits are specific to meaningful topology or simply from having a graph structure.

**Expected Outcome:** If real topology provides benefit, randomized topology should perform at or below the "no graph" baseline.

---

## 20. SPATIAL GENERALIZATION EXPERIMENT

### Design

**Watershed Partitioning:**
- Delaware River Basin divided into 3-4 sub-watersheds based on major tributaries
- Sub-watershed 1: Upper Delaware (New York headwaters)
- Sub-watershed 2: Middle Delaware ( Pennsylvania section)
- Sub-watershed 3: Lower Delaware (New Jersey/Delaware tidal reach)
- Sub-watershed 4: Major tributaries (Lehigh, Schuylkill, etc.)

**Experiment Protocol:**
1. Train on 3 sub-watersheds, test on 1 held-out sub-watershed
2. Rotate which sub-watershed is held out
3. Compare XGBoost, MLP, and GNN performance

**Metric:** RMSE degradation = (test_RMSE - train_RMSE) / train_RMSE

**Hypothesis:** GNN will show smaller degradation due to shared river topology patterns.

---

## 21. TEMPORAL GENERALIZATION EXPERIMENT

### Design

**Temporal Partitioning:**
- Early period: July 2018 - December 2018 (training)
- Late period: January 2019 - March 2019 (testing)

**Protocol:**
1. Train all models on early period
2. Test on late period (no overlap)
3. Ensure all features available at prediction time use only historical data

**Important:** No future data used in feature construction for either the temporal lag features or the environmental covariates.

---

## 22. SPARSE-DATA EXPERIMENT

### Design

**Data Reduction Levels:**
- 100%: Full training set
- 75%: Remove 25% of training observations (random sampling)
- 50%: Remove 50% of training observations
- 25%: Remove 75% of training observations
- 10%: Remove 90% of training observations

**Protocol:**
1. For each sparsity level, train: XGBoost, MLP, and GNN
2. Evaluate on the same held-out test set
3. Use multiple random seeds (5) for each level
4. Plot performance curves: RMSE vs. training data percentage

**Metric:** Normalized RMSE = (model_RMSE - baseline_RMSE) / baseline_RMSE

**Hypothesis:** GNN degradation rate will be lower than baselines, especially at sparse levels.

---

## 23. EXTREME-EVENT EXPERIMENT

### Design

**Event Definition:**
- **High flow:** Streamflow > 90th percentile for the station
- **Extreme precipitation:** Daily precipitation > 25mm (1 inch)
- **Normal conditions:** All other observations

**Protocol:**
1. Stratify test set into extreme and normal event categories
2. Compute prediction error separately for each category
3. Compare error ratios: extreme_error / normal_error

**Secondary analysis:** Compute prediction uncertainty separately for extreme vs. normal events

---

## 24. EXPLAINABILITY ANALYSIS

### Methods

1. **Permutation Importance:**
   - Shuffle each feature individually and measure performance drop
   - Computed on validation set (not test set)
   - Repeated 10 times for stability

2. **SHAP Values:**
   - Use TreeSHAP for XGBoost models
   - For GNN, use GNNExplainer or Integrated Gradients
   - Focus on top contributing features

3. **GNN Attention/Edge Importance:**
   - If using GAT variants, analyze attention weights
   - Examine which upstream nodes contribute most to downstream predictions

### Critical Distinction

All explainability analysis is explicitly framed as **feature importance for prediction**, NOT **causation**. The document will clearly state that statistical association does not imply causal relationship between environmental features and microplastic pollution.

---

## 25. ERROR ANALYSIS

### Error Stratification Categories

1. **High rainfall events:** Errors correlated with >20mm daily precipitation
2. **High discharge events:** Errors correlated with >90th percentile streamflow
3. **Extreme concentrations:** Errors at top 10% of observed concentrations
4. **Sparse observations:** Errors at locations with ≤5 historical observations
5. **Specific watersheds:** Per-watershed error analysis
6. **Urban areas:** High vs. low impervious surface coverage
7. **Unusual seasons:** Spring runoff vs. baseflow conditions
8. **Sampling method differences:** Net vs. grab samples (if metadata available)

### Analysis Method

For each category, compute:
- Mean absolute error
- Error bias (mean error)
- Correlation between predicted and observed
- Coverage of prediction intervals

---

## 26. STATISTICAL ANALYSIS PLAN

### Primary Testing Approach

1. **Multiple Random Seeds:** Run all experiments with 5 random seeds
2. **Primary Comparison:** GNN vs. XGBoost on random split (E3 vs. E2)
3. **Significance Test:** Paired permutation t-test on RMSE differences
   - Assumptions: Independent observations, normally distributed errors (verified by Shapiro-Wilk)
   - Null: Mean difference = 0

### Secondary Analyses

1. **Bootstrap Confidence Intervals:**
   - 1000 bootstrap resamples of test set
   - 95% confidence intervals for RMSE difference

2. **Effect Sizes:**
   - Cohen's d for mean performance differences
   - Relative improvement percentage

3. **ANOVA for Ablation:**
   - Repeated measures ANOVA across model variants
   - Post-hoc Tukey HSD for pairwise comparisons

### Assumptions Verification

- **Normality:** Shapiro-Wilk test on residuals
- **Homoscedasticity:** Breusch-Pagan test
- **Independence:** Verify no spatial/temporal autocorrelation in test errors using Moran's I

---

## 27. VISUALIZATION PLAN

### Required Visualizations

1. **Study Area Map:** River network with observation locations, colored by concentration
2. **Performance Comparison Charts:** Bar charts with error bars showing RMSE/MAE/R² by model
3. **Ablation Progression:** Line plot showing performance improvement across model variants
4. **Spatial Generalization Maps:** Predicted vs. observed concentrations on maps
5. **Temporal Trends:** Time series of predictions vs. observations
6. **Uncertainty Visualization:** Prediction intervals overlaid on predictions
7. **Feature Importance:** SHAP summary plots and permutation importance bar charts
8. **Graph Structure:** Visualization of river network with edge weights
9. **Error Heatmaps:** Geographic distribution of prediction errors
10. **Extreme Event Comparison:** Error distributions for normal vs. extreme events

### Tools
- matplotlib (primary), seaborn (statistical plots)
- plotly (interactive exploration)
- NetworkX + matplotlib (graph visualization)
- geopandas (spatial maps)

---

## 28. SOFTWARE ARCHITECTURE

### Recommended Technology Stack

**Core ML Libraries:**
- PyTorch (neural network framework)
- PyTorch Geometric (GNN implementation)
- scikit-learn (baseline models, preprocessing)

**Geospatial Libraries:**
- geopandas (spatial data handling)
- rasterio (raster data)
- shapely (geometric operations)
- pyproj (coordinate transformations)
- networkx (graph analysis, as backup)

**Data Science:**
- pandas (data manipulation)
- numpy (numerical computing)
- scipy (statistical functions)

**Visualization:**
- matplotlib, seaborn (static plots)
- plotly (interactive visualizations)

### Dependency Management
All dependencies are available on PyPI/conda. No special licensing required. Total installation footprint: ~500MB.

### Hardware Requirements

- **CPU-only:** Intel i5/Ryzen 5 or better, 16GB RAM
- **GPU (optional):** NVIDIA GTX 1660 or better with 6GB VRAM
- **Storage:** 10GB for datasets, 2GB for code/results
- **Training time:** 2-10 hours on CPU, 30 minutes-2 hours on GPU

---

## 29. REPOSITORY STRUCTURE

```
microplastic-gnn/
├── configs/
│   ├── config.yaml          # Main configuration file
│   ├── models/              # Model hyperparameters
│   └── experiments/         # Experiment-specific settings
├── data/
│   ├── raw/                 # Downloaded raw datasets
│   ├── processed/           # Cleaned and formatted data
│   ├── features/            # Engineered feature matrices
│   └── graphs/              # Constructed graph objects
├── models/
│   ├── gnn.py               # GNN architecture
│   ├── baselines.py         # Baseline model implementations
│   └── ensemble.py          # Ensemble wrapper
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── downloaders.py   # Data download utilities
│   │   ├── validators.py    # Data quality checks
│   │   └── processors.py    # Data cleaning pipelines
│   ├── geospatial/
│   │   ├── river_network.py # NHDPlus processing
│   │   └── spatial_join.py  # Spatial aggregation
│   ├── graph/
│   │   ├── builder.py       # Graph construction
│   │   └── randomization.py # Topology controls
│   ├── models/
│   │   ├── architecture.py  # GNN model definition
│   │   └── training.py      # Training loop
│   ├── training/
│   │   ├── trainer.py       # Main training logic
│   │   └── callbacks.py     # Early stopping, etc.
│   ├── evaluation/
│   │   ├── metrics.py       # Performance metrics
│   │   ├── ablation.py      # Ablation analysis
│   │   └── statistics.py    # Statistical tests
│   ├── explainability/
│   │   ├── shap_analysis.py
│   │   └── importance.py
│   └── visualization/
│       ├── plots.py         # Static plots
│       └── maps.py          # Spatial visualizations
├── scripts/
│   ├── run_experiment.py    # Main experiment runner
│   ├── reproduce_all.sh     # Reproduce all results
│   └── make_figures.py      # Generate publication figures
├── experiments/
│   ├── e01_baseline.py
│   ├── e02_xgboost.py
│   ├── e03_gnn_initial.py
│   ├── e04_spatial_holdout.py
│   ├── e05_temporal_holdout.py
│   ├── e06_spatiotemporal_holdout.py
│   ├── e07_randomized_topology.py
│   ├── e08_graph_structure_comparison.py
│   ├── e09_ablation.py
│   ├── e10_sparse_data.py
│   ├── e11_generalization.py
│   ├── e12_temporal_generalization.py
│   └── e13_extreme_events.py
├── results/
│   ├── logs/                # Training logs
│   ├── metrics/             # Performance metrics
│   ├── figures/             # Generated figures
│   └── reports/             # Summary reports
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_graph.py
└── documentation/
    ├── methodology.md       # Detailed methodology
    ├── data_spec.md         # Data specifications
    └── model_spec.md        # Model architecture spec
```

### Directory Specifications

**configs/** - YAML/JSON configuration files for reproducible experiments
**data/raw/** - Downloaded datasets with original structure preserved
**data/processed/** - Cleaned datasets ready for feature engineering
**data/features/** - Computed feature matrices with documentation
**data/graphs/** - Serialized graph objects (PyG Data objects saved as .pt)
**models/** - Model architecture and training code
**notebooks/** - Exploratory analysis and visualization notebooks
**src/** - Modular source code library
**scripts/** - Executable scripts for running experiments
**experiments/** - Self-contained experiment scripts
**results/** - Output artifacts organized by type
**tests/** - Unit and integration tests
**documentation/** - Technical documentation and specifications

---

## 30. REPRODUCIBILITY REQUIREMENTS

### Code Versioning
- Git repository with clear commit history
- Semantic version tags for major milestones
- Branch strategy: main + feature/experiment branches

### Random Seed Management
- Global seed setting (numpy, torch, random)
- Per-experiment seed recording
- Minimum 5 random seeds per experiment
- Results reported as mean ± standard deviation

### Dependency Management
- requirements.txt for pinned versions
- environment.yml for conda environments
- Docker container specification for full reproducibility

### Documentation Requirements
- All data sources documented with citations
- Preprocessing steps recorded as executable code
- Experiment configurations stored as version-controlled YAML files
- Results include provenance (which code version, which seed)

### Computational Reproducibility
- All experiments must be runnable with single command: `python scripts/reproduce_all.sh`
- Expected runtime documented for each experiment
- Output artifacts automatically saved and timestamped

---

## 31. SCIENTIFIC LIMITATIONS

### Data Limitations
1. **Sample Size:** Limited microplastic observations (~150-200 samples total across all datasets) severely constrains model complexity and training
2. **Temporal Coverage:** No continuous time series of microplastic measurements; most data are single time-point cross-sections
3. **Methodological Heterogeneity:** Different studies use incompatible sampling and analysis methods, making data integration challenging
4. **Geographic Bias:** USGS datasets primarily cover U.S. Northeast and Great Lakes regions
5. **Scale Mismatch:** Microplastic samples represent point-in-time conditions while environmental data are continuous

### Methodological Limitations
1. **Target Variable:** Microplastic concentration is inherently stochastic; deterministic prediction has fundamental limits
2. **Non-stationarity:** Relationships between environmental conditions and microplastic transport may change over time due to land use change, climate change, and pollution control measures
3. **Transport Complexity:** Actual microplastic transport involves complex physics (sedimentation, degradation, biological uptake) not fully captured by environmental covariates
4. **Spatial Resolution:** River segments may integrate conditions from large areas, potentially obscuring important local effects

### Model Limitations
1. **Extrapolation:** Models may fail when environmental conditions exceed training range
2. **Uncertainty Quantification:** Ensemble methods under-disperse uncertainty estimates
3. **Feature Leakage:** Despite precautions, some subtle form of leakage may remain
4. **Overfitting Risk:** Small datasets increase risk of overfitting to idiosyncrasies

### Interpretability Limitations
1. **Feature Importance:** Permutation importance and SHAP values measure predictive utility, not causal importance
2. **Graph Attention:** Edge weights in GNN may not reflect physical transport mechanisms
3. **Correlation vs. Causation:** Environmental features correlated with microplastic concentrations may not causally influence transport

---

## 32. NOVELTY ASSESSMENT

### Areas of Genuine Novelty

1. **Combined microplastic + GNN + topology comparison:** While GNNs have been applied to water quality prediction with topology constraints (T-GCN paper, 2026), direct comparison of physical topology vs. geographic proximity for microplastic prediction appears to be novel to this project scope.

2. **Graph randomization control for microplastics:** The topology randomization experiment (comparing real vs. shuffled river networks) is a rigorous control that has not been extensively applied to microplastic prediction specifically.

3. **Sparse-data robustness investigation:** While common in other domains, systematic testing of GNN robustness under progressively reduced training data specifically for microplastic prediction is novel.

4. **Systematic ablation across full model stack:** The complete ablation study (A→E) with controlled components represents a thorough methodological contribution.

### Areas of Existing Work

1. **GNN for river networks:** Extensively studied (Sun et al. 2022, Kirschstein & Sun 2024, HydroGAT 2025)
2. **T-GCN for water quality:** Applied (Scientific Reports 2026 paper)
3. **Physics-informed ML for hydrology:** Well-established literature
4. **Spatiotemporal GNN for environmental prediction:** Common technique

### Scientific Contribution Statement

The project's primary contribution is **empirical validation** of whether explicit river-network topology provides predictive value for microplastic forecasting compared to simpler geographic representations, under realistic data scarcity conditions. This addresses an open question identified in the literature (Kirschstein & Sun 2024: "whether GNNs actually benefit from network topology").

The contribution is methodological and empirical rather than algorithmic. The novelty lies in the **experimental design** and **comparative evaluation** rather than the model architecture itself.

---

## 33. EXPERIMENT MATRIX

| Experiment | Model | Spatial Split | Temporal Split | Graph | Purpose |
|----------|-------|---------------|----------------|-------|---------|
| E1 | Mean/Median | Random | Random | None | Absolute baseline |
| E2 | Ridge | Random | Random | None | Linear baseline |
| E3 | Random Forest | Random | Random | None | Strong baseline |
| E4 | XGBoost | Random | Random | None | Strongest baseline |
| E5 | MLP | Random | Random | None | Neural baseline |
| E6 | GNN (full) | Random | Random | River | Initial GNN test |
| E7 | GNN (ablation A) | Random | Random | None | Environmental only |
| E8 | GNN (ablation B) | Random | Random | None | Env + temporal |
| E9 | GNN (ablation C) | Random | Random | River | + graph (no temporal encoder) |
| E10 | GNN (ablation D) | Random | Random | River | + temporal encoder |
| E11 | GNN (full) | Spatial holdout | Same | River | Spatial generalization |
| E12 | GNN (full) | Same | Temporal holdout | River | Temporal generalization |
| E13 | GNN (full) | Spatial holdout | Temporal holdout | River | Hard generalization |
| E14 | GNN (full) | Same | Same | Randomized | Topology control |
| E15 | GNN (full) | Same | Same | Geographic KNN | Graph structure comparison |
| E16 | GNN (full) | Same | Same | Haversine | Graph structure comparison |
| E17 | Models (all baselines) | Same | Same | River | Sparse data experiment |
| E18 | GNN (full) | Spatial | Same | River | Unseen watershed comparison |
| E19 | GNN (full) | Same | Temporal | River | Temporal extrapolation |
| E20 | GNN (full) | Same | Same | River | Extreme event analysis |
| E21 | GNN (full) | Same | Same | River | Uncertainty analysis |

---

## 34. IMPLEMENTATION HANDOFF SPECIFICATION

### Research Parameters

**Research Question:** Does explicit river-network connectivity in a GNN improve microplastic prediction vs. baselines?

**Hypothesis Test:** Paired permutation test on RMSE, α = 0.05

**Effect Size Target:** Detect ≥10% RMSE improvement

**Sample Size:** 150-200 microplastic observations (limited by available data)

### Data Requirements

| Dataset | Source | URL | Format |
|---------|--------|-----|--------|
| Microplastic observations | USGS ScienceBase | DOI:10.5066/P9QVIVX3 | CSV |
| NHDPlus river network | EPA | https://www.epa.gov/waterdata/nhdplus | Shapefile/GDB |
| Streamflow data | USGS NWIS | https://api.waterdata.usgs.gov/ | JSON/CSV |
| Meteorology | Daymet | https://daymet.ornl.gov/ | NetCDF/CSV |
| Land cover | NLCD | https://www.usgs.gov/landsat-missions/land-cover | GeoTIFF |
| Elevation | NED | https://www.usgs.gov/core-science-support/ngtoc/national-elevation-dataset | GeoTIFF |

### Model Specifications

**Input dimension:** Variable based on feature selection
**Hidden dimension:** 64
**Message passing layers:** 2 (GraphSAGE)
**Temporal encoder:** GRU (hidden dim 64, seq len 7)
**Output dimension:** 1 (log-concentration)
**Ensemble size:** 5 for uncertainty
**Training:** Adam optimizer, lr=0.001, 500 epochs with early stopping (patience=30)

### Evaluation Protocol

1. **Metrics:** MAE, RMSE, R², Explained Variance, Median Absolute Error
2. **Cross-validation:** 5-fold with random seeds {42, 123, 456, 789, 101}
3. **Statistical tests:** Paired permutation tests (10,000 permutations)
4. **Uncertainty validation:** Coverage probability of 95% prediction intervals

### Output Requirements

1. **Predictions:** CSV with columns: [location_id, sample_date, observed, predicted, lower_95, upper_95]
2. **Metrics:** JSON with per-experiment metrics and statistical test results
3. **Figures:** All visualization plan items saved as PNG/PDF
4. **Report:** Comprehensive results summary in Markdown
5. **Code:** Fully documented, executable notebooks and scripts

### Data Handling Protocol

1. **All preprocessing must use only training data for fitting**
2. **Temporal alignment: features from day t-1 for prediction on day t**
3. **No microplastic measurements used as features (target leakage prevention)**
4. **Spatial joins: upstream accumulated features computed only from training locations**

### Quality Assurance

1. **Reproducibility check:** Re-run with same seeds must produce identical results
2. **Gradient flow verification:** Monitor gradient norms during training
3. **Overfitting detection:** Monitor train/validation gap
4. **Statistical significance:** All comparisons tested at α=0.05 level

---