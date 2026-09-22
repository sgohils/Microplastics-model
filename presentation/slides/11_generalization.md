Slide Number: 11
Slide Title: Does the model generalize beyond the training data?
Purpose: Evaluate the model's ability to predict in unseen locations and time periods.
Main Message: The GNN shows reasonable spatial generalization but poor temporal generalization and extrapolation.
Exact Text/Copy:
Spatial holdout (E11_spatial_holdout):
• MAE = 0.77 particles/m³
• RMSE = 0.86 particles/m³
• R² = -1.06
Temporal holdout (E12_temporal_holdout):
• MAE = 2.72 particles/m³
• RMSE = 3.61 particles/m³
• R² = -0.61
Spatiotemporal holdout (E13_spatiotemporal_holdout):
• MAE = 1.22 particles/m³
• RMSE = 1.32 particles/m³
• R² = -2.18
Temporal extrapolation (E19_temporal_extrapolation):
• MAE = 2.72 particles/m³
• RMSE = 3.61 particles/m³
• R² = -0.61
Note:
• Negative R² indicates that the model performs worse than simply predicting the mean of the training set (due to high variance relative to mean in the test set).
• Spatial holdout results are based on holding out one sub-watershed (e.g., eastern tributaries) and training on the remainder.
• Temporal holdout splits the data by time: early period (July–December 2018) vs. late period (January–March 2019).
• Spatiotemporal holdout combines both space and time holdouts (e.g., train on western Delaware River in January 2019, test on eastern Delaware River in July 2018).
• Temporal extrapolation is equivalent to temporal holdout in this dataset.
Recommended Visual: A bar chart showing MAE (or RMSE) for each holdout experiment, with the random split result for comparison. Include error bars if available.
Figure/Table Requirement: Create a table or bar chart using the exact numbers from `experiments/results/final_summary.json` (E11, E12, E13, E19) and the random split result (E6_gnn_real_topology). If variance across seeds is available, include error bars.
Speaker Notes:
- 30-second explanation: "The model works okay in new locations but poorly in new time periods."
- 60–90-second explanation: "To assess whether the model has learned generalizable patterns, I tested it on data it had not seen during training. In spatial holdout experiments—where I trained on a subset of sub-watersheds and tested on a completely held-out sub-watershed—the GNN achieved an MAE of 0.77 and RMSE of 0.86. While the R² is negative (indicating high error relative to the variance of the test set), the absolute error is lower than some baselines in other experiments, suggesting the model has captured some spatial patterns that transfer across geography. In stark contrast, temporal holdout experiments—where I trained on the first half of the sampling period and tested on the second half—showed much higher error: MAE of 2.72 and RMSE of 3.61. This indicates that the model does not generalize well to future time periods, likely because the microplastic observations are cross-sectional (single time points) and the GRU only sees environmental history, not lagged microplastic measurements. Spatiotemporal holdout, which combines both space and time holdouts, yielded intermediate results (MAE=1.22, RMSE=1.32). Temporal extrapolation tests confirmed the poor temporal generalization. These results suggest that while the model has learned some spatial relationships that transfer to new locations, it struggles to capture temporal dynamics that would allow forecasting future conditions—a limitation driven by the sparse, cross-sectional nature of the microplastic observations."
- Technical explanation: "The spatial holdout experiment (E11) involved partitioning the Delaware River Basin into sub-watersheds based on major tributaries, training on a subset, and testing on a completely held-out sub-watershed. The graph construction for this experiment used only training-set nodes and their immediate upstream/downstream neighbors to prevent leakage. The temporal holdout experiment (E12) split the data chronologically: observations from July–December 2018 for training, January–March 2019 for testing. The spatiotemporal holdout (E13) combined both partitions. The temporal extrapolation experiment (E19) used the same split as the temporal holdout. All experiments used the same model architecture and hyperparameters, with training conducted on the respective training sets. The negative R² values arise when the model's prediction error is larger than the variance of the test set around its mean—a common occurrence when the test set has high variability and the model's predictions are inaccurate."
- One-sentence explanation: "The model works alright in new places on the river but not for predicting future times."
Transition to Next Slide: "What factors did the model find most important for making predictions?"