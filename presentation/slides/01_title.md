Slide Number: 1
Slide Title: Predicting Microplastic Transport Through River Networks Using a Spatiotemporal Graph Neural Network
Purpose: Introduce the project and establish context.
Main Message: The project investigates whether explicitly modeling river-network connectivity improves microplastic concentration predictions compared to conventional machine learning models.
Exact Text/Copy:
Title: Predicting Microplastic Transport Through River Networks Using a Spatiotemporal Graph Neural Network
Researcher: [Student Name]
School: [School Name]
Competition: [Competition Name]
Category: [Environmental Science / Computer Science / Engineering]
Year: 2026
Recommended Visual: A clean, scientifically styled hero visualization showing:
- A map of the Delaware River Basin with monitoring locations marked
- Inset of a river network graph (nodes = river segments, edges = flow direction)
- Arrows indicating data flow from observations → graph → model → prediction
Figure/Table Requirement: None (this is a title slide; visual is background/representative)
Speaker Notes:
- 30-second explanation: "I studied whether representing river networks as graphs helps predict microplastic pollution better than standard machine learning models."
- 60–90-second explanation: "Microplastic pollution is a growing environmental concern, and predicting its transport in rivers is crucial for effective monitoring. However, observations are sparse and expensive to collect. This project tests whether explicitly encoding the river network's hydrological connectivity—as a graph where nodes are river segments and edges represent downstream flow—enables a graph neural network to capture useful patterns that conventional models miss. I built a spatiotemporal GNN combining graph structure with temporal environmental data and compared it against strong baselines like XGBoost and Random Forest."
- Technical explanation: "The model architecture consists of a feature encoder for static and temporal environmental variables, a GRU to capture 7-day antecedent conditions, and GraphSAGE layers to propagate information along the directed river graph. The output predicts log-transformed microplastic concentration per node."
- One-sentence explanation: "I used a graph neural network to see if river connectivity helps predict microplastic pollution."
Transition to Next Slide: "To understand why this approach might work, let's first look at the problem of microplastic monitoring."