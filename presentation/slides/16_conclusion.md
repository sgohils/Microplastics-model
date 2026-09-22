Slide Number: 16
Slide Title: Conclusion
Purpose: Summarize the findings and respond to the research question.
Main Message: The study found that explicit river-network connectivity does not significantly improve microplastic prediction over strong baselines, but the graph-based approach captures some predictive signal and provides a rigorous framework for future work.
Exact Text/Copy:
What I tested:
Whether explicitly representing physical river-network connectivity in a spatiotemporal graph neural network improves prediction of microplastic concentrations compared to conventional machine learning baselines.
What I found:
• The GNN with real river topology did not achieve statistically significant improvement over XGBoost (p > 0.05 for RMSE and MAE)
• Ablation studies indicated modest contributions from graph structure (~0.14 MAE improvement) and temporal encoding (~0.07 MAE improvement)
• The model showed reasonable spatial generalization but poor temporal generalization and extrapolation
• Uncertainty estimates were under-dispersed (95% PI coverage ~85%)
What it means:
• River network topology provides some predictive signal for microplastic concentration, but the sparse and cross-sectional nature of the observations limits the ability to detect strong topological advantages over conventional baselines
• The study underscores the importance of rigorous experimental design in environmental machine learning: leakage prevention, strong baselines, ablation studies, and holdout validation are essential for drawing trustworthy conclusions
• The graph-based approach is not yet ready for operational use but offers a methodological framework for evaluating complex environmental systems in sparse data regimes
What remains uncertain:
• Whether higher-frequency microplastic observations or dynamic flow routing would reveal a stronger benefit of river connectivity
• How the model would perform in independent watersheds with different hydrological regimes
• The optimal temporal depth and graph architecture for this specific prediction task
Finish with:
**This study tests whether river-network structure can provide useful information for predicting microplastic pollution beyond conventional feature-based models.**
Recommended Visual: None (conclusion slide; consider using the title graphic or a simple summary diagram)
Figure/Table Requirement: None
Speaker Notes:
- 30-second explanation: "The river connections didn't beat the best standard model, but the graph method still gave us useful insights."
- 60–90-second explanation: "To answer the original question: I found that explicitly modeling river-network connectivity as a graph did not yield a statistically significant improvement in microplastic prediction compared to the best conventional machine learning baseline (XGBoost). The GNN's predictions were not significantly better, and in fact showed higher error in this dataset. However, the study was not a failure—it revealed that both the graph structure and temporal encoding contribute modest improvements over feature-only baselines, as shown by ablation studies. The model demonstrated some ability to generalize to new locations (spatial holdout) but struggled with future time periods (temporal holdout), reflecting the cross-sectional nature of the microplastic samples. Uncertainty quantification showed room for improvement. These results suggest that while river connectivity does carry predictive information, the current dataset's limitations prevent us from realizing its full potential. The work's main value lies in its rigorous methodology: by carefully preventing data leakage, comparing against strong baselines, and systematically testing model components, we have generated trustworthy insights about what the graph-based approach can and cannot do. This provides a solid foundation for future research, whether that means gathering more data, incorporating dynamic flow features, or exploring physical constraints."
- Technical explanation: "The conclusion is drawn directly from the experimental results: the paired permutation test failed to reject the null hypothesis, the ablation study quantified component contributions, and the holdout experiments evaluated generalization. The limitation regarding temporal generalization is supported by the cross-sectional nature of the target variable, as documented in the data feasibility audit. The uncertainty under-dispersion is reported in the model card. The future work suggestions are limited to those directly supported by the study's weaknesses."
- One-sentence explanation: "Using river connections in the model helped a little, but not enough to clearly beat the standard approach."
Transition to Next Slide: (End of presentation)