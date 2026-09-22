Slide Number: 5
Slide Title: Building the dataset
Purpose: Describe the data sources and integration process.
Main Message: The dataset combines USGS microplastic observations with environmental, hydrological, and geographic data from multiple sources.
Exact Text/Copy:
Data sources:
• Microplastic observations: USGS ScienceBase (Delaware River 2018, Great Lakes tributaries 2014-2015, Northeastern U.S. streams 2017-2018)
• River network: NHDPlus v2 and NHDPlus HR (flowline topology, slope, velocity, drainage area)
• Hydrology: USGS NWIS API (daily discharge, gage height)
• Meteorology: Daymet v4 (daily precipitation, temperature, wind, snow water equivalent)
• Geography: NLCD 2019 land cover, NED 10m elevation, gridded population density
Total observations: ~150-200 distinct sampling events (after unit harmonization and quality control)
Spatial extent: Primarily U.S. Northeast and Great Lakes region
Temporal extent: July 2018 – March 2019 (microplastic sampling period)
Target variable: Microplastic concentration in river water (particles per cubic meter)
Preprocessing:
• Unit conversion: particles/L → particles/m³ (×1000)
• log1p transformation of target for training
• Z-score normalization of features (fit on training data only)
• Lagged features: 1-day, 3-day, 7-day lags for precipitation and discharge (using only past data to prevent leakage)
• No microplastic measurements used as features (to prevent target leakage)
Recommended Visual: A data pipeline diagram showing:
Microplastic observations → Geographic coordinates → River network → Hydrology → Meteorology → Land use/geography → Temporal features → Integrated graph dataset
Figure/Table Requirement: Create a simple flowchart illustrating the integration of diverse data sources into a unified dataset for graph-based modeling.
Speaker Notes:
- 30-second explanation: "I combined microplastic data from the USGS with river network, weather, and land cover data."
- 60–90-second explanation: "To build the dataset, I started with three USGS microplastic observation datasets: the Delaware River (2018), Great Lakes tributaries (2014-2015), and Northeastern U.S. streams (2017-2018). I harmonized units to particles per cubic meter, converting from particles per liter where necessary. I then matched each observation to the corresponding river segment in the NHDPlus network and extracted environmental covariates: hydrology from USGS streamgages (with inverse-distance weighting for non-co-located sites), meteorology from Daymet gridded data, and geography from NLCD and NED. Temporal features included lagged precipitation and discharge to capture antecedent conditions, ensuring that only past data was used to avoid leakage. Features were normalized using z-scores computed solely on the training set."
- Technical explanation: "The preprocessing pipeline strictly adhered to leakage prevention principles: all normalization parameters (mean, standard deviation) were computed exclusively on the training set and applied to validation and test sets. Lagged features were constructed such that for a given observation date, only environmental data from prior days were used. Microplastic measurements from any location were never used as predictive features for any other location, eliminating target leakage."
- One-sentence explanation: "I gathered microplastic observations from the USGS and matched them with river, weather, and land use data."
Transition to Next Slide: "With the data ready, I then represented the river system as a graph."