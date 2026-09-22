Slide Number: 4
Slide Title: Research Question and Hypothesis
Purpose: Clearly state the study's research question and hypothesis.
Main Message: The study tests whether explicit river-network connectivity improves microplastic prediction compared to conventional baselines.
Exact Text/Copy:
Research Question:
Does explicitly representing physical river-network connectivity in a spatiotemporal graph neural network architecture provide measurably superior prediction of microplastic concentrations compared to conventional machine learning approaches that rely only on geographic proximity and environmental covariates, when evaluated on previously unseen watersheds and time periods?
Hypothesis:
A spatiotemporal graph neural network that explicitly incorporates real river-network connectivity topology will achieve statistically significantly lower prediction error (measured by RMSE) on held-out test data compared to the best-performing conventional machine learning baseline (XGBoost), when both are evaluated on spatially and temporally held-out microplastic observations.
Null Hypothesis:
There is no statistically significant difference in prediction error (RMSE) between a graph neural network incorporating river-network connectivity and the best-performing conventional machine learning baseline (XGBoost), when evaluated on held-out test data. Any observed differences are within the bounds of random variation.
Recommended Visual: None (text-focused slide; consider using icons for question and hypothesis)
Figure/Table Requirement: None
Speaker Notes:
- 30-second explanation: "I asked whether modeling river connections as a graph improves microplastic psychic predictions."
- 60–90-second explanation: "The core of this study is a direct comparison: I built a graph neural network that explicitly represents the river network's hydrological connectivity—where nodes are river segments and edges represent downstream flow—and compared its predictive performance against strong conventional machine learning models that use the same environmental and temporal features but without graph structure. The hypothesis was that the graph-based model would produce significantly lower prediction errors on unseen data. The null hypothesis states that any observed difference is due to random chance."
- Technical explanation: "The hypothesis was tested using paired permutation tests on prediction errors (RMSE and MAE) across multiple random seeds, with spatial, temporal, and spatiotemporal holdout experiments to evaluate generalization. Significance was set at α=0.05."
- One-sentence explanation: "I tested if adding river connections to the model improves its predictions."
Transition to Next Slide: "To answer this question, I first needed to gather and prepare the data."