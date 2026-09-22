Slide Number: 6
Slide Title: I represented the river system as a graph
Purpose: Explain how the river network was converted into a graph structure suitable for a GNN.
Main Message: Nodes represent river segments, edges represent downstream flow, and edge weights reflect flow accumulation.
Exact Text/Copy:
Nodes:
• Definition: River segments (NHDPlus flowline reaches) that intersect microplastic sampling locations or are part of the upstream context
• Count: ~46 nodes (in the Delaware River experiment, including upstream context nodes)
• Features per node: Static (elevation, slope, drainage area, stream order, land-cover fractions, impervious surface, population density) + Temporal (precipitation, temperature, discharge, lags 1/3/7 days, month/season)
Edges:
• Definition: Directed edges following NHDPlus hydrologic sequencing (upstream → downstream)
• Weight: Upstream drainage area (flow accumulation), representing relative flow contribution
• Optional features: Flow direction indicator, stream order difference
Graph properties:
• Directed (flow direction matters)
• Weighted (edge weights proportional to upstream contributing area)
• Static (based on long-term average topology; does not capture dynamic flow variations)
Recommended Visual: A clear graph diagram showing:
• Nodes as circles or rectangles labeled with segment IDs or locations
• Arrows indicating edge direction (upstream → downstream)
• Edge thickness varying to represent weight (flow accumulation)
• Example layout:
  Node A (upstream)
     ↓ (thick edge)
  Node B (midstream)
     ↙     ↘
  Node C   Node D (downstream tributaries)
Figure/Table Requirement: Create a simple schematic of a river network graph with labeled nodes, directed edges, and variable edge weights to illustrate the representation.
Speaker Notes:
- 30-second explanation: "I turned the river network into a graph where nodes are river segments and arrows show downstream flow."
- 60–90-second explanation: "The river network was represented as a directed, weighted graph. Each node corresponds to a river segment (NHDPlus flowline reach) that either contains a microplastic sampling location or is part of the upstream context. Each node carries static features like elevation, slope, and land cover, as well as temporal features like precipitation and temperature with lagged antecedent conditions. Edges connect nodes in the direction of water flow (upstream to downstream), with edge weights proportional to the upstream drainage area (flow accumulation)—larger upstream areas contribute more water and potentially more microplastics. This weighting reflects the hydrological intuition that a segment receiving flow from a large watershed experiences greater transport potential than one fed by a small tributary. The graph is static, based on long-term average topology from NHDPlus, and does not capture short-term changes in flow routing during storms or droughts."
- Technical explanation: "The graph construction process involved spatially joining microplastic sampling locations to NHDPlus flowline reaches to assign nodes. For the Delaware River experiment, all reaches within the watershed were included as nodes to increase graph size and improve message passing, with microplastic targets available only at sampled reaches. Edge weights were computed as the ratio of upstream drainage area of the child node to the parent node, or simply the child node's drainage area for simplicity. Edge features could include binary flow direction and normalized stream order difference."
- One-sentence explanation: "Each dot is a river segment, and arrows show which way water flows, with thicker arrows meaning more water upstream."
Transition to Next Slide: "Next, I combined this graph with environmental and temporal data to build the model."