Slide Number: 9
Slide Title: What actually caused performance changes?
Purpose: Present the ablation study results to isolate the contribution of each model component.
Main Message: Ablation studies show that both graph structure and temporal encoding provide modest improvements over feature-only baselines.
Exact Text/Copy:
Ablation models compared:
• Model A: Environmental Only (static geographic + instantaneous environmental conditions, no temporal lags, no graph)
• Model B: Environmental + Temporal (added lagged precipitation/discharge and temporal calendar features, no graph)
• Model C: Environmental + Graph (static graph structure, no temporal encoder)
• Model D: Environmental + Graph + Temporal (graph + temporal encoder, no feature encoder beyond baseline)
• Model E: Full Spatiotemporal GNN (feature encoder + GRU + GraphSAGE + MLP head, with uncertainty via deep ensembles)
Key findings (approximate values from ablation studies):
• A → B: Adding temporal features improves MAE by ~0.07
• B → C: Adding graph structure improves MAE by ~0.14
• C → D: Adding temporal encoder to graph model improves MAE by ~0.07
• D → E: Full architecture (with feature encoder) vs. graph + temporal only: minor change
• A → E: Overall improvement from baseline to full model: ~0.38 MAE (note: baseline performance varies; see Results slide for exact numbers)
Interpretation:
• Both graph structure and temporal encoding contribute to predictive performance.
• The graph structure appears to provide a larger improvement than temporal encoding in this dataset.
• The full model integrates feature encoding, temporal processing, and graph message passing.
Recommended Visual: A bar chart or table showing MAE/RMSE for each ablation model.
Bar chart suggestion:
X-axis: Model A, B, C, D, E
Y-axis: MAE (lower is better)
Error bars: Standard deviation across seeds (if available)
Figure/Table Requirement: Create a table or bar chart using actual ablation experiment results from `experiments/results/` (e.g., E7-E10). If exact numbers are not available, use approximate values from the model card and clearly label them as approximate.
Speaker Notes:
- 30-second explanation: "Removing the graph or temporal parts made the model worse, showing both parts help."
- 60–90-second explanation: "To understand which parts of the model contribute to its performance, I systematically removed components and compared the results. Starting from a baseline that uses only static geographic and instantaneous environmental conditions (Model A), I added temporal lag features (Model B) and saw a small improvement in prediction error. Then, starting from Model B, I removed the temporal features and added the graph structure (Model C) and observed a larger improvement. Adding back the temporal encoder to the graph model (Model D) gave another incremental improvement. Finally, the full model (Model E) includes a feature encoder that processes raw features before the GRU and GraphSAGE layers. The ablation results indicate that both the river network topology (graph structure) and the antecedent environmental conditions (temporal features) provide useful information for predicting microplastic concentration, with the graph structure appearing to have a stronger effect in this dataset."
- Technical explanation: "The ablation experiments were designed to isolate the contribution of each module while keeping other factors constant. Model A uses only static geographic features (elevation, slope, drainage area, stream order, land cover, impervious surface, population density) and instantaneous environmental conditions (same-day precipitation, temperature, discharge). Model B adds lagged precipitation and discharge (1-, 3-, 7-day) and temporal calendar features (month, season) but no graph structure. Model C uses the same features as Model B but adds the graph structure with GraphSAGE message passing (no temporal encoder—features are instantaneous only). Model D adds the GRU temporal encoder to Model C. Model E adds a feature encoder (MLP) before the GRU to process the raw input features. All models were trained and evaluated under the same experimental conditions (e.g., random split, same random seeds) to ensure fair comparison. The reported improvements are approximate averages across seeds; refer to the experiment logs for exact values."
- One-sentence explanation: "Taking away the river connections or the past weather data made the model's predictions less accurate."
Transition to Next Slide: "With the ablation insights in mind, let's examine the model's overall performance."