Slide Number: 12
Slide Title: What factors influenced predictions?
Purpose: Describe the feature importance analysis to understand what the model learned.
Main Message: Feature importance analysis (permutation importance and SHAP values) identified key environmental and temporal drivers of microplastic concentration predictions.
Exact Text/Copy:
Methods used:
• Permutation importance: Shuffle each feature individually and measure performance drop on validation set
• SHAP values: For tree-based baselines (XGBoost, Random Forest); for GNN, consider GNNExplainer or gradient-based approximations
Feature groups examined:
• Hydrology: Discharge, precipitation, temperature, flow velocity, gage height
• Geography: Elevation, slope, drainage area, stream order, land-cover fractions, impervious surface, population density
• Temporal conditions: Month, season, lagged precipitation/discharge (1/3/7 days)
• Human activity: Impervious surface, population density, urban land cover
• Connectivity: Graph structure (via message passing)
Key findings (based on available analyses):
• Permutation importance on XGBoost baselines showed that discharge, precipitation, and impervious surface were among the top-ranking features.
• SHAP values for XGBoost highlighted similar hydrological and land-use features as major contributors.
• For the GNN, message-passing analysis indicated that upstream nodes contribute to downstream predictions, with edge weights (flow accumulation) modulating the strength of influence.
• Temporal features (lagged precipitation and discharge) showed measurable importance in ablation studies.
Note: The GNN's internal representations are not directly interpretable as feature importance in the same way as tree-based models; however, ablation studies and message-passing examinations provide insight into what drives predictions.
Recommended Visual: A bar chart showing feature importance scores (e.g., mean |SHAP value| or permutation importance score) for the top 10 features, grouped by category (hydrology, geography, temporal, human activity). Alternatively, a diagram showing upstream influence on a downstream node via message passing.
Figure/Table Requirement: Create a feature importance bar chart using actual SHAP or permutation importance values from the analysis (if available). If exact values are not available, create a schematic illustrating the concept of upstream influence in the graph and label it as conceptual.
Speaker Notes:
- 30-second explanation: "The model found that water flow, rain, and land use were important for making predictions."
- 60–90-second explanation: "To understand what the model learned, I analyzed feature importance using multiple approaches. For the conventional baselines (XGBoost, Random Forest), I computed permutation importance—measuring how much prediction error increases when a feature's values are randomly shuffled—and SHAP values, which estimate each feature's contribution to individual predictions. These analyses consistently showed that hydrological features like discharge and precipitation, along with land-use indicators such as impervious surface and population density, were strong predictors of microplastic concentration. For the GNN itself, direct feature attribution is more complex due to the message-passing mechanism; however, by examining how predictions change when altering upstream conditions or edge weights, I observed that upstream nodes do influence downstream predictions, with the strength of influence modulated by the graph's edge weights (flow accumulation). Ablation studies further confirmed that removing temporal features (lagged precipitation and discharge) increased prediction error, indicating that antecedent hydrological conditions provide useful information. Importantly, none of these analyses imply causation—they only identify statistical associations between features and predicted concentrations."
- Technical explanation: "Permutation importance was computed on the validation set for each baseline model by iterating over features, shuffling their values, and measuring the increase in MAE or RMSE. SHAP values were calculated using the TreeSHAP algorithm for XGBoost and Random Forest models. For the GNN, since it is not a tree-based model, exact SHAP values are not directly applicable; however, approximation methods such as KernelSHAP or gradient-based techniques can be used to estimate feature contributions. Message-passing analysis involved computing the sensitivity of a node's prediction to changes in neighboring nodes' features or edge weights, revealing the flow of information through the graph. All analyses were conducted on validation data to avoid overfitting to the test set."
- One-sentence explanation: "The model paid the most attention to water flow, rainfall, and how built-up the area is."
Transition to Next Slide: "Let's look at where the model made its biggest mistakes."