"""Visualize the river network graph with observations."""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from pathlib import Path
import numpy as np

from src.utils.config import load_config
from src.geospatial.spatial import RiverNetworkBuilder
from src.utils.seed import set_seed

config = load_config('config.yaml')
set_seed(42)

# Build river network
builder = RiverNetworkBuilder(config)
river_network = builder.build_delaware_river_network()

# Add observation locations
import pandas as pd
features = pd.read_csv(Path(config['data']['processed_dir']) / 'features_processed.csv')
builder.add_microplastic_locations(features)

graph = builder.graph

# Create visualization
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Separate river segments and observation nodes
river_nodes = [n for n in graph.nodes() if not str(n).startswith('obs_')]
obs_nodes = [n for n in graph.nodes() if str(n).startswith('obs_')]

# Get positions for river segments (from geometry midpoints)
pos = {}
for node_id in river_nodes:
    node_data = graph.nodes[node_id]
    if 'geometry' in node_data:
        geom = node_data['geometry']
        centroid = geom.centroid
        pos[node_id] = (centroid.x, centroid.y)
    else:
        pos[node_id] = (0, 0)

# Get positions for observations
for obs_id in obs_nodes:
    node_data = graph.nodes[obs_id]
    pos[obs_id] = (node_data['longitude'], node_data['latitude'])

# Draw river segments
river_edges = [(u, v) for u, v in graph.edges() 
               if not str(u).startswith('obs_') and not str(v).startswith('obs_')]
flow_edges = [(u, v) for u, v in graph.edges() 
              if str(u).startswith('obs_') or str(v).startswith('obs_')]

# River network edges (flow connections)
nx.draw_networkx_edges(graph, pos, edgelist=river_edges,
                       edge_color='blue', width=2, alpha=0.6,
                       arrows=True, arrowsize=20, ax=ax)

# Observation connections (red, dashed)
nx.draw_networkx_edges(graph, pos, edgelist=flow_edges,
                       edge_color='red', width=0.5, alpha=0.3,
                       style='dashed', ax=ax)

# Draw river segment nodes
river_labels = {n: graph.nodes[n].get('name', str(n)) for n in river_nodes}
nx.draw_networkx_nodes(graph, pos, nodelist=river_nodes,
                       node_color='lightblue', node_size=1500,
                       edgecolors='darkblue', linewidths=2, ax=ax)

# Draw observation nodes
obs_concentrations = [graph.nodes[n]['concentration'] for n in obs_nodes]
# Filter out None values for color mapping
valid_concentrations = [c for c in obs_concentrations if c is not None]
if valid_concentrations:
    vmin, vmax = min(valid_concentrations), max(valid_concentrations)
    obs_concentrations = [c if c is not None else np.median(valid_concentrations) for c in obs_concentrations]
else:
    vmin, vmax = 0, 1
    obs_concentrations = [0.5 for _ in obs_concentrations]

nx.draw_networkx_nodes(graph, pos, nodelist=obs_nodes,
                       node_color=obs_concentrations, cmap='viridis',
                       node_size=200, vmin=vmin, vmax=vmax, ax=ax)

# Colorbar for observations
sm = plt.cm.ScalarMappable(cmap='viridis', 
                           norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
cbar.set_label('Microplastic Concentration (particles/m³)', fontsize=10)

# Labels
nx.draw_networkx_labels(graph, pos, labels=river_labels,
                        font_size=8, font_weight='bold', ax=ax)

# Obs node labels (obs_index)
obs_labels = {n: str(graph.nodes[n]['obs_id']) for n in obs_nodes}
nx.draw_networkx_labels(graph, pos, labels=obs_labels,
                        font_size=6, font_color='white', ax=ax)

ax.set_title('Delaware River Network with Microplastic Observations', fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=11)
ax.set_ylabel('Latitude', fontsize=11)

# Legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='lightblue', edgecolor='darkblue', label='River Segments'),
    Patch(facecolor='purple', label='Observations (color=concentration)'),
    Line2D([0], [0], color='blue', linewidth=2, alpha=0.6, label='Flow edges'),
    Line2D([0], [0], color='red', linewidth=0.5, alpha=0.3, linestyle='--', label='Obs-segment connections'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

plt.tight_layout()

# Save
output_path = Path(config['paths']['figures_dir']) / 'river_network_graph.png'
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'Graph visualization saved to: {output_path}')

# Also save graph as GraphML for external viewers (skip - not supported for complex attrs)
# graphml_path = Path(config['paths']['figures_dir']) / 'river_network.graphml'
# nx.write_graphml(graph, graphml_path)
# print(f'GraphML file saved to: {graphml_path}')

# Print graph statistics
print(f'\nGraph Statistics:')
print(f'  River segments: {len(river_nodes)}')
print(f'  Observation nodes: {len(obs_nodes)}')
print(f'  Total nodes: {graph.number_of_nodes()}')
print(f'  Total edges: {graph.number_of_edges()}')
print(f'  Flow edges: {len(river_edges)}')
print(f'  Obs connections: {len(flow_edges)}')

for node_id in river_nodes:
    node_data = graph.nodes[node_id]
    print(f'  Segment {node_id} ({node_data.get("name", "?")}): '
          f'order={node_data.get("stream_order", "?")}, '
          f'area={node_data.get("drainage_area", "?")} km², '
          f'discharge={node_data.get("avg_discharge_cfs", "?")} cfs')
