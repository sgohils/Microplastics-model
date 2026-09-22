Slide Number: 14
Slide Title: What this study cannot yet establish
Purpose: Clearly state the study's limitations to ensure scientific honesty.
Main Message: The study faces constraints related to data sparsity, temporal dynamics, graph staticness, and uncertainty quantification.
Exact Text/Copy:
Limitations:
• Data limitations
  - Sparse microplastic observations (~150-200 distinct sampling events)
  - Cross-sectional sampling (single time points per location) limits true spatiotemporal modeling
  - Unit and methodological heterogeneity across studies
• Geographic limitations
  - Primary focus on Delaware River Basin; limited validation on independent watersheds
  - Validation regions (Great Lakes tributaries, Northeastern U.S. streams) used for secondary analysis but not primary model training
• Temporal limitations
  - Microplastic samples are infrequent (monthly or less); environmental data are daily
  - GRU only sees environmental history, not lagged microplastic measurements
  - No true temporal signal in the target variable
• Graph limitations
  - Static topology based on long-term average NHDPlus data; does not capture dynamic flow routing during storms or droughts
  - Edge weights based on upstream drainage area may not fully represent transport dynamics
  - Graph construction does not account for temporary flow reversals or backwater effects
• Modeling limitations
  - Modest performance gains over baselines; not statistically significant
  - Uncertainty estimates are under-dispersed (95% PI coverage ~85%)
  - Limited temporal depth (7-day history) may miss longer antecedent conditions
  - Message-passing depth (2 layers) limits receptive field to ~2-3 degrees of upstream/downstream separation
• Real-world deployment limitations
  - Not ready for operational use in environmental monitoring or pollution surveillance
  - Requires further validation on independent datasets and watersheds
  - Computational requirements are modest but still need calibration and maintenance
Recommended Visual: None (text-focused slide; consider using icons for each limitation category)
Figure/Table Requirement: None
Speaker Notes:
- 30-second explanation: "The study is limited by sparse data, snapshot observations, and a static river network model."
- 60–90-second explanation: "This study faces several important constraints that affect the interpretation of its results. First, the dataset is relatively small: only about 150 to 200 distinct microplastic sampling events were available after harmonizing units and quality control. These observations are cross-sectional, meaning each location was sampled at one or a few points in time, not as a continuous time series. This limits the ability to model true temporal dynamics in microplastic concentrations—the GRU can only use environmental history as a proxy, not actual past microplastic measurements. Second, while the study includes validation data from the Great Lakes tributaries and Northeastern U.S. streams, the primary model development and testing focused on the Delaware River Basin, so findings may not generalize to other geographic regions without further testing. Third, the graph representation of the river network is static, based on long-term average topology from NHDPlus. It does not capture short-term changes in flow routing during flood events, droughts, or dam operations. Edge weights are derived from upstream drainage area, which is a reasonable proxy for flow contribution but may not fully represent the nuances of microplastic transport. Fourth, the model's performance gains over conventional baselines are modest and not statistically significant in this dataset. Uncertainty quantification using deep ensembles showed under-dispersion, meaning the predicted confidence intervals were too narrow compared to actual errors. Fifth, the model relies on a 7-day history of environmental conditions, which may not capture longer antecedent conditions relevant to microplastic transport. Finally, the model is not yet ready for real-world deployment; it serves as a proof-of-concept for evaluating graph-based approaches in sparse data regimes."
- Technical explanation: "Each limitation is supported by specific evidence from the experiments and data assessment. Data sparsity is quantified in the DATA FEASIBILITY AUDIT section of the research specification (~150-200 observations). Cross-sectional nature is documented in the same section: 'Temporal distribution: Cross-sectional studies, not continuous monitoring.' Geographic limitations are noted in the STUDY REGION section: primary focus on Delaware River Basin, with alternative datasets for validation. Graph limitations are described in the ASSUMPTIONS.md file (e.g., Assumption 2.1: Node Definition, Assumption 2.3: Edge Weighting) and the MODEL_CARD.md limitations section. Modeling limitations are inferred from the experimental results: modest improvement over baselines, uncertainty under-dispersion in MODEL_CARD.md, and ablation study effect sizes. Real-world deployment limitations are a synthesis of the above: the model's current performance and uncertainty characteristics do not meet thresholds for operational reliance."
- One-sentence explanation: "The study is limited by having few measurements, snapshots in time, and a fixed map of river connections."
Transition to Next Slide: "Despite these limitations, the work still enables certain insights and future directions."