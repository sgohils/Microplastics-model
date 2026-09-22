# Technical Explanations

## Microplastics
### Technical explanation
Microplastics are plastic particles less than 5 mm in diameter, originating from the breakdown of larger plastic debris or manufactured as small particles (e.g., microbeads). They are persistent in the environment and can be ingested by aquatic organisms, leading to ecological and health impacts.

### Simple explanation
Tiny plastic pieces that pollute water and harm wildlife.

### One-sentence explanation
Microplastics are small plastic pollutants in rivers that can harm aquatic life.

### Why it matters to this project
Understanding microplastic transport helps predict where they accumulate, aiding in mitigation efforts.

## River Networks
### Technical explanation
A river network is a directed graph where nodes represent river segments (or reaches) and edges represent the direction of water flow (from upstream to downstream). The network captures the hierarchical structure of tributaries merging into main stems.

### Simple explanation
The system of rivers and streams that water flows through, like branches of a tree joining together.

### One-sentence explanation
River networks show how water moves from small streams to larger rivers.

### Why it matters to this model
The graph structure allows the model to propagate information upstream/downstream, capturing hydrological connectivity that influences microplastic transport.

## Graph Theory
### Technical explanation
Graph theory studies structures made of nodes (vertices) connected by edges (links). In this project, we use a directed, weighted graph where edge weights represent flow accumulation (upstream drainage area).

### Simple explanation
The math of networks, like maps of connections.

### One-sentence explanation
Graph theory helps us model how river segments are connected.

### Why it matters to this project
It provides the framework for representing river connectivity and applying graph neural networks.

## Nodes and Edges
### Technical explanation
- **Nodes**: Represent river segments (NHDPlus flowline reaches) that intersect microplastic sampling locations or are part of the upstream context.
- **Edges**: Represent hydrological connectivity (flow direction) from upstream to downstream reaches, weighted by upstream drainage area (flow accumulation).

### Simple explanation
Nodes are points on the river; edges are the flow directions between them.

### One-sentence explanation
Nodes are river sections, edges show which way water flows.

### Why it matters to this project
Correctly defining nodes and edges ensures the GNN processes meaningful hydrological relationships.

## Features
### Technical explanation
Features are measurable properties used as inputs to the model. They include:
- Static: elevation, slope, drainage area, stream order, land cover, impervious surface, population density.
- Temporal: precipitation, temperature, discharge (with 1-, 3-, 7-day lags), month, season.
- Derived: lagged temporal features to capture antecedent conditions.

### Simple explanation
Information about each river segment, like its height, land use, and recent rain.

### One-sentence explanation
Features describe the river segment's environment over time.

### Why it matters to this project
Features provide the environmental context that influences microplastic concentration at each node.

## Machine Learning
### Technical explanation
Machine learning involves training algorithms to make predictions from data. We compare a Graph Neural Network (GNN) against conventional baselines (Ridge Regression, Random Forest, XGBoost, MLP) to test whether adding graph structure improves prediction.

### Simple explanation
Teaching computers to find patterns in data to make predictions.

### One-sentence explanation
ML helps us predict microplastic levels using river and weather data.

### Why it matters to this project
It allows us to quantitatively test the hypothesis that river network structure improves predictions.

## Training, Validation, Test Set
### Technical explanation
- **Training set**: Used to fit model parameters.
- **Validation set**: Used to tune hyperparameters and prevent overfitting (early stopping).
- **Test set**: Held-out data used for final unbiased evaluation of model performance.
We use random split (70/15/15) for baseline comparison and spatial/temporal holdouts for generalization testing.

### Simple explanation
Dividing data into parts to train, tune, and test the model fairly.

### One-sentence explanation
We split data to train the model, check its settings, and see how well it works on new data.

### Why it matters to this project
Proper splitting ensures our performance estimates are unbiased and reflect real-world generalization.

## Overfitting
### Technical explanation
Overfitting occurs when a model learns noise or idiosyncrasies in the training data, leading to poor performance on unseen data. We mitigate overfitting via dropout (0.3), weight decay (1e-4), early stopping, and limiting model capacity (hidden_dim=64).

### Simple explanation
When the model memorizes the training data instead of learning general patterns.

### One-sentence explanation
Overfitting is when the model works well on training data but fails on new data.

### Why it matters to this project
Avoiding overfitting ensures our results are reliable and not due to memorizing the small dataset.

## GNN (Graph Neural Network)
### Technical explanation
Our GNN combines a GRU (temporal encoder) and GraphSAGE (spatial message passing). Input: node features over 7 time steps. The GRU captures temporal dependencies, then GraphSAGE aggregates information from neighboring nodes (upstream/downstream) over 2 layers. An MLP head predicts log-transformed microplastic concentration.

### Simple explanation
A type of neural network that understands data on networks (like rivers).

### One-sentence explanation
A GNN processes river network data by combining temporal and spatial information.

### Why it matters to this project
It allows us to explicitly model river connectivity and test its predictive value.

## Message Passing
### Technical explanation
In GraphSAGE, each node aggregates feature vectors from its neighbors (using mean aggregation) and updates its own representation. This process repeats for multiple layers, allowing information to propagate further across the graph.

### Simple explanation
Nodes share information with their neighbors to update their understanding.

### One-sentence explanation
Message passing lets river segments share information with connected segments.

### Why it matters to this project
It enables the model to use upstream/downstream conditions to predict microplastic concentration at a given segment.

## Temporal Modeling
### Technical explanation
We use a GRU to process sequences of environmental features (7-day history) at each node, capturing how antecedent conditions (e.g., recent rain) influence current microplastic levels.

### Simple explanation
Modeling how past weather and flow affect the present.

### One-sentence explanation
Temporal modeling uses recent environmental data to improve predictions.

### Why it matters to this project
It tests whether antecedent hydrological conditions improve microplastic prediction beyond instantaneous conditions.

## Uncertainty
### Technical explanation
We quantify prediction uncertainty using deep ensembles (5 members) trained with different random seeds and data splits. The final prediction is the mean of ensemble members; uncertainty is the standard deviation across members. We report 95% prediction intervals as [mean ± 1.96×std].

### Simple explanation
A measure of how confident we are in our predictions.

### One-sentence explanation
Uncertainty tells us how reliable our predictions are.

### Why it matters to this project
It helps assess the reliability of predictions, especially in data-sparse regions.

## Ablation Study
### Technical explanation
We systematically remove components (graph structure, temporal encoder, features) to assess their contribution to performance. Models compared:
- Environmental Only (no temporal, no graph)
- Environmental + Graph (no temporal)
- Environmental + Temporal (no graph)
- Full Spatiotemporal GNN (graph + temporal)
This isolates the value of each component.

### Simple explanation
Removing parts of the model to see what each part does.

### One-sentence explanation
Ablation studies test the importance of each model component.

### Why it matters to this project
It shows whether the graph and temporal components actually improve predictions.

## Baselines
### Technical explanation
We compare against conventional machine learning models:
- Mean/Median predictor (baseline)
- Ridge Regression (linear with L2 regularization)
- Random Forest (ensemble of decision trees)
- XGBoost (gradient boosting)
- MLP (multilayer perceptron)
These establish performance bounds to judge whether the GNN adds value.

### Simple explanation
Simple models we compare our complex model against.

### One-sentence explanation
Baselines are simpler models to see if our GNN is actually better.

### Why it matters to this project
It ensures we are not comparing the GNN to overly weak models, making any improvement meaningful.

## Generalization
### Technical explanation
Generalization measures how well a model performs on data not seen during training (e.g., different locations or time periods). We test:
- Spatial holdout: train on sub-watersheds, test on held-out sub-watershed.
- Temporal holdout: train on early period, test on late period.
- Spatiotemporal holdout: combine spatial and temporal holdouts.
This evaluates whether the model captures transferable patterns.

### Simple explanation
How well the model works on new places or times.

### One-sentence explanation
Generalization tests if the model works where it hasn't seen data before.

### Why it matters to this project
It shows whether the learned patterns (e.g., river connectivity) are useful beyond the training data.

## Data Leakage
### Technical explanation
Data leakage occurs when information from the test set inadvertently influences training, leading to overly optimistic performance. We prevent leakage by:
- Using only past data for features (no future information).
- Ensuring graph construction for holdouts uses only training-set nodes (no test-to-train edges).
- Fitting scalers and encoders on training data only.
- Excluding microplastic measurements from features (to avoid target leakage).

### Simple explanation
Accidentally letting test data affect training, making results look better than they are.

### One-sentence explanation
Data leakage is a mistake that makes the model seem better than it really is.

### Why it matters to this project
Preventing leakage ensures our performance estimates are trustworthy and scientifically valid.

## RMSE, MAE, R²
### Technical explanation
- **RMSE (Root Mean Squared Error)**: Square root of the average squared differences between predictions and observations. Penalizes large errors.
- **MAE (Mean Absolute Error)**: Average absolute differences between predictions and observations.
- **R² (Coefficient of Determination)**: Proportion of variance in observations explained by the model (1 = perfect fit, 0 = mean model, negative = worse than mean).
We report all three to capture different aspects of error.

### Simple explanation
Metrics that measure how close our predictions are to the actual measurements.

### One-sentence explanation
RMSE, MAE, and R² tell us how accurate our predictions are.

### Why it matters to this project
They provide standardized ways to compare model performance and interpret error magnitude.