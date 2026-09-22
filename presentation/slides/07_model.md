Slide Number: 7
Slide Title: Spatiotemporal Graph Neural Network
Purpose: Describe the project's GNN architecture.
Main Message: The model combines a GRU for temporal encoding and GraphSAGE for spatial message passing over the river graph.
Exact Text/Copy:
Architecture diagram:
Environmental Data (per node, per timestep)
       ↓
Feature Encoder (MLP: Linear → ReLU → Linear)
       ↓
Temporal Encoder (GRU: captures 7-day antecedent conditions)
       ↓
Graph Message Passing (GraphSAGE: 2 layers, mean aggregation)
       ↓
Prediction Head (MLP: Linear → ReLU → Linear)
       ↓
Output: log-transformed microplastic concentration per node
Key dimensions:
• Input features (F): ~32 (static + temporal + lagged)
• Feature encoder output: 64 dimensions
• GRU hidden state: 64 dimensions
• GraphSAGE output: 32 dimensions → 16 dimensions (after two layers)
• Prediction head output: 1 dimension (log-concentration)
Implementation details:
• Framework: PyTorch Geometric
• Optimizer: Adam (learning rate=0.001, weight_decay=1e-4)
• Regularization: Dropout=0.3, early stopping (patience=30 epochs)
• Loss: Mean Squared Error on log-transformed target
• Ensemble: 5 members for uncertainty estimation (different random seeds, train/validation splits)
Recommended Visual: A clean architecture diagram showing the flow from input features through feature encoder, GRU, GraphSAGE layers, and prediction head to output. Label each component with its function and dimensions.
Figure/Table Requirement: Create a detailed architecture diagram matching the implemented model, using consistent labeling and dimensions.
Speaker Notes:
- 30-second explanation: "The model has two main parts: one for processing time sequences and one for spreading information across the river network."
- 60–90-second explanation: "The spatiotemporal GNN processes data in four stages. First, a feature encoder transforms raw environmental and temporal features (like elevation, precipitation, discharge, and their lags) into a common 64-dimensional embedding space. Second, a GRU (Gated Recurrent Unit) takes the sequence of encoded features over the past 7 days and captures temporal dependencies, producing a 64-dimensional state that summarizes antecedent hydrological and meteorological conditions. Third, this temporal encoding is passed through two GraphSAGE layers, which aggregate information from neighboring nodes in the river graph using mean aggregation—allowing each node to collect information from its upstream and downstream neighbors. After two layers, the representation is 32-dimensional. Finally, a prediction head maps this representation to a single value: the log-transformed microplastic concentration. To estimate uncertainty, I trained five ensemble members with different random seeds and data splits; the final prediction is the mean across members, and uncertainty is the standard deviation."
- Technical explanation: "The feature encoder is a two-layer MLP with ReLU activation, projecting the input feature vector (dimension F ≈ 32) to a 64-dimensional space. The GRU processes a sequence of length 7 (one timestep per day for the past week) of these 64-dimensional vectors, outputting a 64-dimensional hidden state. The first GraphSAGE layer takes the 64-dimensional node features, aggregates neighbor information via mean pooling, applies ReLU, and outputs 64-dimensional features. The second GraphSAGE layer repeats the process, outputting 32-dimensional features. The prediction head is a two-layer MLP with ReLU activation, mapping 32-dimensional features to a 1-dimensional output (log-concentration). Dropout of 0.3 is applied after each hidden layer in the feature encoder and prediction head. The GRU and GraphSAGE layers also incorporate dropout. The ensemble members are trained with different random seeds and different 85/15 train/validation splits (same test set)."
- One-sentence explanation: "The model looks at the past week of weather and flow at each river segment, then shares information with connected segments to make a prediction."
Transition to Next Slide: "To test whether this design actually works, I ran a series of rigorous experiments."