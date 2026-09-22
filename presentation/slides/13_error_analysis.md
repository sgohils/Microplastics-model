Slide Number: 13
Slide Title: Where did the model get it wrong?
Purpose: Present error analysis to understand model limitations and failure cases.
Main Message: The model's largest errors occur during high-flow events, in data-sparse regions, and due to measurement heterogeneity.
Exact Text/Copy:
Error analysis categories (from model card and error analysis):
• High rainfall events: Errors correlated with >20mm daily precipitation
• High discharge events: Errors correlated with >90th percentile streamflow
• Extreme concentrations: Errors at top 10% of observed concentrations
• Sparse observations: Errors at locations with ≤5 historical observations
• Specific watersheds: Per-watershed error analysis
• Urban areas: High vs. low impervious surface coverage
• Unusual seasons: Spring runoff vs. baseflow conditions
• Sampling method differences: Net vs. grab samples (if metadata available)
Key observations:
• The model tends to under-predict during high-flow storm events, possibly because rapid transport and dilution are not fully captured by the static graph and lagged features.
• Predictions are less accurate in locations with few historical observations (sparse monitoring), reflecting the model's difficulty in learning patterns from limited data.
• Error varies across watersheds, with some tributaries showing consistently higher or lower predictions than observed.
• Urban areas with high impervious surface tend to have higher prediction errors, suggesting complex runoff dynamics not fully represented in the features.
• Spring runoff periods (snowmelt, heavy rains) show elevated error compared to baseflow conditions.
• Differences in sampling and analytical methods across USGS studies introduce noise that the model cannot fully reconcile.
Recommended Visual: A scatter plot of predicted vs. observed microplastic concentration with points colored by error magnitude or by category (e.g., high flow, low flow). Alternatively, a map showing spatial distribution of prediction errors.
Figure/Table Requirement: Create an error visualization using actual prediction and observation data from the experiments (e.g., save predictions from the test set and plot vs. observed). If exact data is not available, create a conceptual diagram illustrating the error categories listed above.
Speaker Notes:
- 30-second explanation: "The model made bigger mistakes during heavy rain and in places with few past measurements."
- 60–90-second explanation: "When examining where the model's predictions deviated most from actual measurements, several patterns emerged. First, during high-flow events—defined as days with precipitation over 20mm or streamflow above the 90th percentile—the model tended to under-predict microplastic concentration. This suggests that the model may not fully capture the rapid transport, dilution, or complex turbulence associated with storm flows. Second, locations with very few historical observations (five or fewer) showed higher prediction errors, indicating that the model struggles to learn reliable patterns from extremely sparse data. Third, error was not uniform across the river network; certain watersheds or tributaries exhibited systematic over- or under-prediction, possibly due to unmodeled local sources or sinks. Fourth, urban areas with high percentages of impervious surface (like concrete and asphalt) tended to have larger errors, pointing to the complexity of runoff from paved surfaces that may not be fully captured by the land-cover features. Fifth, seasonal analysis revealed that spring runoff periods—when snowmelt and rain combine to produce high flows—had elevated error compared to drier baseflow conditions. Finally, differences in how microplastic samples were collected and analyzed across the USGS studies (e.g., grab samples vs. net tows, varying analytical techniques) introduced variability that the model treats as noise."
- Technical explanation: "Error analysis was performed by computing residuals (prediction − observation) for each test set point and stratifying by various conditions. High rainfall events were identified using Daymet precipitation data (>20mm daily). High discharge events used USGS NWIS streamgage data (>90th percentile for the respective station). Extreme concentrations were defined as the top 10% of observed microplastic concentrations in the test set. Sparse observations were defined as locations with five or fewer historical samples in the training set. Per-watershed error analysis grouped observations by major tributary (e.g., Lehigh River, Schuylkill River) and computed mean absolute error per group. Urban vs. rural comparison used impervious surface thresholds from NLCD data. Seasonal comparison used meteorological data to distinguish spring (March–May) from other periods, noting that the microplastic sampling period (July 2018–March 2019) includes late winter and early spring. Sampling method differences were examined where metadata was available in the USGS datasets."
- One-sentence explanation: "The model's biggest errors happened during storms, in places with almost no past data, and in highly urban areas."
Transition to Next Slide: "Given these limitations, let's discuss what the study cannot yet establish."