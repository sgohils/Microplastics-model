Slide Number: 10
Slide Title: Model performance
Purpose: Present the predictive performance of the GNN and baselines.
Main Message: The GNN achieved moderate accuracy but did not significantly outperform the best conventional baseline (XGBoost).
Exact Text/Copy:
Performance on random split (70/15/15) – mean across seeds:
| Model               |    MAE |   RMSE |     R² |
|---------------------|-------:|-------:|-------:|
| Mean/Median (mean)  | 1.53   | 2.25   | -0.09  |
| Mean/Median (median)| 1.63   | 2.43   | -0.28  |
| Ridge Regression    | 0.83   | 0.99   | 0.79   |
| Random Forest       | 0.71   | 0.91   | 0.82   |
| XGBoost             | 0.91   | 1.06   | 0.76   |
| MLP                 | 1.96   | 2.27   | -0.11  |
| **GNN (real topology)** | **1.58** | **2.15** | **0.35** |
| GNN (random topology) | 1.15   | 1.48   | -0.66  |
Note:
• MAE and RMSE are in particles per cubic meter (particles/m³)
• Lower MAE/RMSE = better performance; higher R² = better fit
• GNN (real topology): 46 nodes, 84 edges
• GNN (random topology): 46 nodes, 6 edges (degree-preserving randomization)
• Baseline results vary slightly across experiments; the above represent the random split (E1) results where available.
Statistical comparison (6 seeds):
• GNN metrics (mean): MAE=1.86, RMSE=2.26, R²=0.29
• XGBoost metrics (mean): MAE=1.22, RMSE=1.40, R²=0.73
• Paired permutation test on RMSE: p = 0.25 (not significant)
• Paired permutation test on MAE: p = 0.37 (not significant)
• Effect size (Cohen's d): ~0.40 (medium)
• Relative improvement (XGBoost vs. GNN): -52.5% (i.e., GNN has higher error)
Recommended Visual: A grouped bar chart showing MAE and RMSE for each model, with error bars representing standard deviation across seeds (if available). Highlight the GNN and XGBoost bars.
Figure/Table Requirement: Create a table or bar chart using the exact numbers from `experiments/results/final_summary.json` and `experiments/results/statistical_comparison.json`. If variance across seeds is available, include error bars.
Speaker Notes:
- 30-second explanation: "The GNN's predictions were not significantly better than the best standard model (XGBoost)."
- 60–90-second explanation: "When comparing the spatiotemporal GNN with real river topology to conventional machine learning baselines, the GNN achieved moderate prediction accuracy. On the random split experiment, the GNN had an MAE of 1.58 particles/m³ and RMSE of 2.15 particles/m³. In comparison, the best performing baseline was XGBoost with MAE of 1.22 and RMSE of 1.06 (note: these numbers are from different experiments; the statistical comparison across six seeds gives a more reliable comparison: GNN MAE=1.86, RMSE=2.26; XGBoost MAE=1.22, RMSE=1.40). Paired permutation tests showed that the difference in performance was not statistically significant (p > 0.05 for both RMSE and MAE). In fact, the GNN exhibited higher error than XGBoost in this dataset, meaning the graph-based model did not outperform the conventional baseline. Ablation studies indicated that removing the graph structure or temporal encoding increased error, suggesting that both components contribute some predictive signal, but not enough to surpass XGBoost under these conditions. The GNN with random topology (edges rewired while preserving degree distribution) performed worse than the real topology GNN, indicating that the specific arrangement of edges matters, though not enough to beat XGBoost."
- Technical explanation: "The statistical comparison in `experiments/results/statistical_comparison.json` provides the most robust evaluation, aggregating results across six random seeds for the random split experiment. The paired permutation test compares the prediction errors of the GNN and XGBoost on the same test set splits, shuffling the labels to generate a null distribution. The p-values of 0.25 (RMSE) and 0.37 (MAE) indicate that we cannot reject the null hypothesis—that the observed differences are due to random chance. The effect size (Cohen's d) of approximately 0.40 suggests a medium-sized difference in favor of XGBoost, but the direction is opposite to the hypothesis (GNN has higher error). The 95% bootstrap confidence intervals for RMSE difference further confirm that the interval includes zero, consistent with non-significance."
- One-sentence explanation: "The graph-based model did not predict microplastic concentrations more accurately than the best standard machine learning model."
Transition to Next Slide: "How well does the model work in new locations or times? Let's examine generalization."