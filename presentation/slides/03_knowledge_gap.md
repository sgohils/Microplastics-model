Slide Number: 3
Slide Title: Existing prediction approaches can miss river connectivity
Purpose: Highlight the scientific gap that the study addresses.
Main Message: Conventional models treat locations independently or use geographic proximity, ignoring hydrological connectivity.
Exact Text/Copy:
Conventional approach:
Location → environmental features → prediction
Ignores that nearby locations may not be hydrologically connected.
Proposed approach:
Upstream river segments
↓
River connectivity (hydrological flow)
↓
Environmental conditions
↓
Temporal conditions
↓
Spatiotemporal GNN
↓
Prediction
Explicitly models how upstream conditions influence downstream locations via river network topology.
Recommended Visual: Two side-by-side diagrams:
Left: Conventional approach - independent locations with features pointing to prediction.
Right: Proposed approach - river network with upstream segments influencing downstream nodes through graph connections.
Figure/Table Requirement: Create a simple schematic showing the difference between treating locations as independent nodes vs. connecting them via river flow edges.
Speaker Notes:
- 30-second explanation: "Standard models look at each location separately, but rivers connect locations through flow."
- 60–90-second explanation: "Many environmental prediction models, including machine learning approaches, treat each monitoring location as an independent data point, using only local environmental features to predict microplastic concentration. This ignores the fact that rivers are directional networks: water flows from upstream to downstream, carrying pollutants with it. Two nearby locations may not be connected if they are on different tributaries, while distant locations on the same river thread can be strongly connected. By representing the river system as a graph where nodes are river segments and edges represent downstream flow, we allow the model to explicitly use upstream information to predict downstream conditions."
- Technical explanation: "In graph-based models, the adjacency matrix or edge list encodes the network structure. For river networks, edges are directed according to flow direction, and edge weights can represent flow accumulation or travel time. Message-passing neural networks then aggregate information from neighboring nodes, enabling upstream/downstream information flow."
- One-sentence explanation: "Instead of treating river locations as independent dots, we connect them based on how water actually flows."
Transition to Next Slide: "Let's now define the specific research question and hypothesis."