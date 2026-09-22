# Judge Question Database

## Scientific Questions

**Q1: What is the main scientific question this study addresses?**  
A: Whether explicitly representing physical river-network connectivity in a spatiotemporal graph neural network improves prediction of microplastic concentrations compared to conventional machine learning baselines that rely only on geographic proximity and environmental covariates.  
Evidence: Research question in `docs/RESEARCH_QUESTION_EXPLANATION.md` and `docs/PRESENTATION_SPECIFICATION.md`.  

**Q2: Why is predicting microplastic transport in rivers scientifically important?**  
A: Microplastic pollution is a critical global environmental concern affecting aquatic ecosystems, food webs, and human health; rivers are the primary conduits transporting microplastics from land to ocean; understanding transport patterns helps develop targeted monitoring and remediation strategies.  
Evidence: `docs/microplastic_gnn_research_specification.md`, Section 2 (Scientific Problem).  

**Q3: What specific aspect of river networks does your model focus on?**  
A: The model focuses on the directed hydrological connectivity of river segments, where edges represent downstream flow and edge weights are proportional to upstream drainage area (flow accumulation).  
Evidence: `docs/microplastic_gnn_research_specification.md`, Section 10 (FEATURE SPECIFICATION) and `docs/ASSUMPTIONS.md`, Assumption 2.2 and 2.3.  

**Q4: How does your model differ from a standard machine learning model that uses the same features?**  
A: The GNN incorporates graph structure via message passing, allowing each node to aggregate information from its neighbors (upstream/downstream), whereas standard models treat each location independently.  
Evidence: `docs/METHODS.md`, Sections on River Graph Construction and GNN Architecture; `docs/PRESENTATION_SLIDE_07_MODEL.md`.  

**Q5: What does the ablation study reveal about the contribution of graph structure?**  
A: Ablation studies indicate that removing the graph structure increases prediction error by approximately 0.14 MAE, suggesting that river topology provides modest predictive information beyond temporal and static features.  
Evidence: `docs/MODEL_CARD.md`, Ablation section; `docs/RESULTS_SUMMARY.md`, Ablation Results.  

## Environmental Questions

**Q6: How do microplastics enter rivers, and what factors influence their transport?**  
A: Microplastics enter rivers via urban runoff, wastewater discharge, and atmospheric deposition. Transport is influenced by water flow velocity, turbulence, particle settling, and hydrological conditions such as discharge and precipitation.  
Evidence: `docs/microplastic_gnn_research_specification.md`, Section 2 (Scientific Problem) and Section 5 (Scientific Significance).  

**Q7: Why might storm events lead to higher prediction errors in your model?**  
A: During high-flow events, rapid transport, dilution, and complex turbulence may not be fully captured by the static graph and lagged environmental features, leading to under-prediction.  
Evidence: `docs/PRESENTATION_SLIDE_13_ERROR_ANALYSIS.md`, High rainfall events and High discharge events; `docs/ERROR_ANALYSIS.md` (if available) or `docs/MODEL_CARD.md` limitations.  

**Q8: How does land use, specifically impervious surface, affect microplastic predictions?**  
A: Higher impervious surface is associated with greater runoff and microplastic loading, but the model may not fully capture the complexity of runoff dynamics from paved surfaces, leading to higher prediction errors in urban areas.  
Evidence: `docs/PRESENTATION_SLIDE_13_ERROR_ANALYSIS.md`, Urban areas; `docs/METHODS.md`, Geographic Features.  

## Machine Learning Questions

**Q9: Why did you choose a GraphSAGE-GRU architecture over other GNN variants (e.g., GCN, GAT)?**  
A: GraphSAGE is computationally efficient and does not require the full adjacency matrix; GRU is simpler than LSTM and effective for short sequences; both were chosen to balance expressiveness with overfitting risk given the small dataset.  
Evidence: `docs/microplastic_gnn_research_specification.md`, Section 16 (ARCHITECTURE JUSTIFICATION); `docs/ASSUMPTIONS.md`, Assumption 4.1 and 4.2.  

**Q10: How do you prevent overfitting in your model?**  
A: We use dropout (0.3), weight decay (1e-4), early stopping (patience=30), limit model capacity (hidden_dim=64), and ensure all preprocessing is fit on training data only.  
Evidence: `docs/ASSUMPTIONS.md`, Assumption 5.1 and 5.2; `docs/METHODS.md`, Training and Validation; `docs/MODEL_CARD.md`, Training details.  

**Q11: What baselines did you compare against, and why were they chosen?**  
A: We compared against Mean/Median predictor, Ridge Regression, Random Forest, XGBoost, and MLP. These represent a range from simple to strong conventional baselines, ensuring that any improvement of the GNN is meaningful.  
Evidence: `docs/METHODS.md`, Baselines; `docs/PRESENTATION_SPECIFICATION.md`, Baselines; `docs/IMPLEMENTATION_SPEC.md`, Section 17 (BASELINE MODELS).  

**Q12: How is the uncertainty of predictions quantified?**  
A: Uncertainty is estimated using deep ensembles (5 members) trained with different random seeds and train/validation splits; the final prediction is the mean across members, and uncertainty is the standard deviation.  
Evidence: `docs/ASSUMPTIONS.md`, Assumption 10.1 and 10.2; `docs/METHODS.md`, Uncertainty; `docs/MODEL_CARD.md`, Uncertainty Method.  

## Data Questions

**Q13: What are the sources and limitations of your microplastic observation data?**  
A: Data come from USGS ScienceBase: Delaware River 2018, Great Lakes tributaries 2014-2015, and Northeastern U.S. streams 2017-2018. Limitations include sparse sampling (~150-200 events), cross-sectional collection, unit heterogeneity, and methodological differences across studies.  
Evidence: `docs/microplastic_gnn_research_specification.md`, Section 7 (DATA FEASIBILITY AUDIT); `docs/DATA_SOURCES.md` (if available) or `docs/REPRODUCIBILITY.md`.  

**Q14: How did you handle incompatible units in microplastic data?**  
A: We converted all measurements to particles per cubic meter (particles/m³), using the conversion 1 particle/L = 1,000 particles/m³, and recorded original units for each observation. Studies reporting only particle counts without volume were excluded from the primary regression target.  
Evidence: `docs/microplastic_gnn_research_specification.md`, Section 8 (TARGET VARIABLE), Handling Incompatible Units.  

**Q15: How did you prevent data leakage in your feature engineering?**  
A: We used only past data for lagged features (e.g., 1-day lag uses previous day's conditions), fitted all normalization parameters (mean, standard deviation) exclusively on the training set, and excluded microplastic measurements from features to avoid target leakage.  
Evidence: `docs/ASSUMPTIONS.md`, Assumption 3.1 and 3.2; `docs/METHODS.md`, Spatial Processing, Temporal Processing, and Data Leakage Prevention; `docs/IMPLEMENTATION_SPEC.md`, Section 16 (DATA LEAKAGE PREVENTION).  

## Statistical Questions

**Q16: What statistical test did you use to compare the GNN and XGBoost, and what was the result?**  
A: We used a paired permutation test on RMSE and MAE with 10,000 permutations, α=0.05. The p-values were 0.25 for RMSE and 0.37 for MAE, indicating that the difference in performance was not statistically significant.  
Evidence: `docs/experiments/results/statistical_comparison.json`, ttest_rmse and ttest_mae; `docs/METHODS.md`, Statistical Analysis; `docs/RESULTS_SUMMARY.md`, Statistical Analysis.  

**Q17: What effect size did you observe between the GNN and XGBoost?**  
A: The effect size (Cohen's d) was approximately 0.40 in favor of XGBoost (i.e., GNN had higher error), indicating a medium-sized difference but in the opposite direction of the hypothesis.  
Evidence: `docs/experiments/results/statistical_comparison.json`, effect_size section; `docs/RESULTS_SUMMARY.md`, Statistical Analysis.  

**Q18: How did you assess the calibration of your uncertainty estimates?**  
A: We computed the coverage probability of the 95% prediction intervals, which was approximately 85%, indicating under-dispersion (intervals too narrow).  
Evidence: `docs/ASSUMPTIONS.md`, Assumption 10.2; `docs/MODEL_CARD.md`, Uncertainty results; `docs/RESULTS_SUMMARY.md`, Uncertainty Results.  

## Graph Questions

**Q19: How does message passing work in your GNN, and what does it allow the model to do?**  
A: In each GraphSAGE layer, each node aggregates feature vectors from its neighbors (using mean aggregation) and updates its own representation, enabling information to propagate upstream and downstream through the river network.  
Evidence: `docs/microplastic_gnn_research_specification.md`, Section 16 (ARCHITECTURE JUSTIFICATION); `docs/METHODS.md`, Graph Message Passing; `docs/PRESENTATION_SLIDE_07_MODEL.md`.  

**Q20: What is the purpose of using randomized graph topology as a control?**  
A: Randomized topology preserves the degree distribution but destroys meaningful hydrological connections, allowing us to test whether the specific arrangement of edges (real topology) provides predictive benefit beyond simply having a graph structure.  
Evidence: `docs/microplastic_gnn_research_specification.md`, Section 19 (GRAPH CONTROL EXPERIMENT); `docs/ASSUMPTIONS.md`, Assumption 2.3 (Edge Weighting) and related.  

**Q21: Did randomized topology provide any predictive benefit?**  
A: No—the GNN with randomized topology performed worse than the real topology GNN and was closer to the "no graph" baseline, indicating that meaningful topology contributes to model performance.  
Evidence: `docs/experiments/results/final_summary.json`, E6_gnn_random_topology; `docs/RESULTS_SUMMARY.md`, GNN Results.  

## Experimental Design Questions

**Q22: What is a spatiotemporal holdout, and why is it considered the most stringent test?**  
A: A spatiotemporal holdout combines spatial and temporal holdouts—for example, training on a subset of sub-watersheds in one time period and testing on a held-out sub-watershed in a different time period—assessing whether the model can generalize to new locations and new times simultaneously.  
Evidence: `docs/microplastic_gnn_research_specification.md`, Section 14 (TRAIN/VALIDATION/TEST DESIGN); `docs/METHODS.md`, Train/Validation/Test Design; `docs/PRESENTATION_SLIDE_08_EXPERIMENTAL_DESIGN.md`.  

**Q23: How did you ensure that your test set was not used for model tuning?**  
A: We used a separate validation set for hyperparameter selection and early stopping; the test set was kept completely frozen until final evaluation.  
Evidence: `docs/ASSUMPTIONS.md`, Assumption 5.1 (Early Stopping Metric); `docs/METHODS.md`, Training and Validation; `docs/PRESENTATION_SLIDE_08_EXPERIMENTAL_DESIGN.md`, Critical safeguards.  

**Q24: What is the sparse monitoring experiment, and what does it test?**  
A: The sparse monitoring experiment systematically reduces the number of training observations (e.g., to 75%, 50%, 25%, 10%) and evaluates performance on a fixed test set to assess how model robustness degrades with decreasing data availability.  
Evidence: `docs/microplastic_gnn_research_specification.md`, Section 22 (SPARSE-DATA EXPERIMENT); `docs/METHODS.md`, Sparse monitoring experiments; `docs/RESULTS_SUMMARY.md`, Sparse Monitoring.  

## Limitations

**Q25: What is the most important limitation of your study, and how does it affect your conclusions?**  
A: The most important limitation is data scarcity (~150-200 observations) and cross-sectional sampling, which limits the ability to capture true spatiotemporal dynamics and increases variance and overfitting risk. This means the model may not realize the full potential of graph-based approaches, and conclusions are constrained by the available data.  
Evidence: `docs/microplastic_gnn_research_specification.md`, Section 7 (DATA FEASIBILITY AUDIT) and Section 6 (Scientific Gap Analysis); `docs/PRESENTATION_SLIDE_14_LIMITATIONS.md`, Data limitations; `docs/MODEL_CARD.md`, Limitations.  

**Q26: How does the static nature of your river network graph affect the model's applicability?**  
A: The graph is based on long-term average topology and does not capture short-term changes in flow routing during storms, droughts, or dam operations, which may limit the model's ability to predict under dynamic hydrological conditions.  
Evidence: `docs/ASSUMPTIONS.md`, Assumption 2.3 (Edge Weighting) and related; `docs/PRESENTATION_SLIDE_14_LIMITATIONS.md`, Graph limitations; `docs/MODEL_CARD.md`, Limitations.  

**Q27: Why might your uncertainty estimates be under-dispersed?**  
A: Under-dispersion can arise from insufficient ensemble diversity, overconfidence in model predictions, or failure to capture all sources of variability. In this case, the limited dataset and small ensemble size (5 members) may contribute to under-dispersed uncertainty.  
Evidence: `docs/ASSUMPTIONS.md`, Assumption 10.2; `docs/MODEL_CARD.md`, Uncertainty limitations; `docs/RESULTS_SUMMARY.md`, Uncertainty Results.  

## Real-World Impact

**Q28: Could this model be used today for operational microplastic monitoring?**  
A: No—the model is not yet ready for operational use due to modest performance, lack of statistical significance over baselines, and uncertainty under-dispersion. It serves as a proof-of-concept for evaluating graph-based approaches in sparse data regimes.  
Evidence: `docs/PRESENTATION_SLIDE_14_LIMITATIONS.md`, Real-world deployment limitations; `docs/PRESENTATION_SLIDE_15_SIGNIFICANCE_FUTURE_WORK.md`, Future work – Long-term; `docs/MODEL_CARD.md`, Limitations.  

**Q29: How could this model assist in environmental monitoring or pollution surveillance?**  
A: The model could help identify river segments with high prediction uncertainty, suggesting locations where additional monitoring would be valuable, or provide screening-level estimates to prioritize sampling efforts. However, these applications require further validation and uncertainty calibration.  
Evidence: `docs/PRESENTATION_SLIDE_35_MONITORING_OPTIMIZATION_EXPLANATION.md` (if exists) or `docs/IMPACT_ANALYSIS.md` (if available); `docs/PRESENTATION_SLIDE_15_SIGNIFICANCE_FUTURE_WORK.md`, Medium-priority future work.  

**Q30: What future work would most improve confidence in your results?**  
A: Expanding to additional watersheds for cross-basin validation, acquiring high-frequency temporal microplastic data to enable true spatiotemporal modeling, and incorporating dynamic flow routing to better capture changing network topology would most improve scientific confidence.  
Evidence: `docs/PRESENTATION_SLIDE_15_SIGNIFICANCE_FUTURE_WORK.md`, Highest priority future work; `docs/FUTURE_WORK.md` (if available); `docs/microplastic_gnn_research_specification.md`, Section 20 (FUTURE EXPERIMENTS).  

For brevity, we have included a representative set of questions. Additional questions can be generated following the same pattern.