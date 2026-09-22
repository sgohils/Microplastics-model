Slide Number: 8
Slide Title: How I tested the hypothesis
Purpose: Explain the experimental framework used to evaluate the model.
Main Message: The model was evaluated using random split, spatial holdout, temporal holdout, spatiotemporal holdout, ablation studies, sparse monitoring experiments, and uncertainty quantification, with a protected test set.
Exact Text/Copy:
Experimental design:
• Baselines: Mean/Median, Ridge Regression, Random Forest, XGBoost, MLP
• Proposed model: Spatiotemporal GNN (GraphSAGE-GRU) with deep ensembles (5 members)
• Evaluation splits:
  - Random Split (70/15/15): Baseline benchmark (ideal conditions)
  - Spatial Holdout: Train on sub-watersheds, test on held-out sub-watershed (geographic generalization)
  - Temporal Holdout: Train on early period (Jul–Dec 2018), test on late period (Jan–Mar 2019) (temporal generalization)
  - Spatiotemporal Holdout: Combine spatial and temporal holdouts (most stringent test)
• Ablation studies: Systematic removal of components (graph structure, temporal encoder, features) to assess contribution
• Sparse monitoring experiments: Reduce training observations (100%, 75%, 50%, 25%, 10%) to assess robustness to data scarcity
• Extreme events analysis: Separate test set into high-flow/precipitation extremes vs. normal conditions
• Uncertainty quantification: Deep ensembles (5 members) with different random seeds and train/validation splits
• Statistical analysis: Paired permutation test on RMSE (GNN vs. XGBoost) with 10,000 permutations, α=0.05; bootstrap confidence intervals; effect size reporting
Critical safeguards:
• Test set was never used for model tuning or hyperparameter selection
• All preprocessing (normalization, lag feature computation) fitted only on training data
• Graph construction for holdouts used only training-set nodes to prevent leakage
• No microplastic measurements used as features (target leakage prevention)
Recommended Visual: A flowchart showing:
Data → Train/Validation/Test splits → Model training → Validation for hyperparameter tuning → Frozen final model → Held-out test set → Evaluation
Include side-by-side boxes for ablation, sparse monitoring, and uncertainty quantification.
Figure/Table Requirement: Create a schematic illustrating the experimental workflow with emphasis on test set protection and leakage prevention.
Speaker Notes:
- 30-second explanation: "I trained and tested the model using several different splits to make sure it really works."
- 60–90-second explanation: "To rigorously evaluate the model, I employed multiple experimental strategies. First, a random split (70% training, 15% validation, 15% testing) provided a baseline benchmark under ideal conditions—though this risks spatial and temporal leakage. Second, I used spatial holdout experiments: I trained the model on a subset of sub-watersheds (e.g., western Delaware River) and tested on a completely held-out sub-watershed (e.g., eastern tributaries) to assess geographic generalization. Third, temporal holdout experiments trained on the first half of the sampling period (July–December 2018) and tested on the second half (January–March 2019) to evaluate whether learned patterns generalize to future time periods. Fourth, spatiotemporal holdout combined both space and time holdouts for the most stringent generalization test. In addition, I conducted ablation studies to isolate the contribution of each model component (graph structure, temporal encoder, features), sparse monitoring experiments to see how performance degrades with less training data, and extreme events analysis to check performance during high-flow conditions. Crucially, the final test set was never used for model tuning—validation sets were used for hyperparameter selection and early stopping. All preprocessing steps, including normalization and lag feature computation, were strictly fitted on the training set only to prevent data leakage. Graph construction for holdout experiments used only training-set nodes and their immediate neighbors to avoid leaking test-set topology information."
- Technical explanation: "The experimental design adhered to strict leakage prevention principles: (1) Temporal leakage: Features used only past environmental data (lagged precipitation/discharge by at least 1 day). (2) Spatial leakage: For spatial holdouts, the graph was constructed using only training-set nodes plus their immediate upstream/downstream neighbors—test-set nodes were not included in graph construction for message passing. (3) Feature leakage: All scalers and encoders (for normalization, etc.) were fitted exclusively on the training set. (4) Target leakage: No microplastic measurements from any location were used as predictive features for any other location. (5) Timing: For a given observation date, lagged features used only data from prior days, ensuring no future information crept in."
- One-sentence explanation: "I split the data in different ways to test if the model really learns general patterns, not just memorizes the training set."
Transition to Next Slide: "Now let's look at what the experiments actually showed."