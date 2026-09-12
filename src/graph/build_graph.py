"""Graph construction for river network and PyTorch Geometric conversion."""
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import pickle
import torch
from torch_geometric.data import Data, Batch

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds and manages graphs for microplastic prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graphs: Dict[str, nx.DiGraph] = {}
    
    def build_river_graph(
        self,
        river_network: nx.DiGraph,
        observations: pd.DataFrame,
        node_features: Dict[int, np.ndarray],
        targets: Dict[int, float]
    ) -> Data:
        """Build PyTorch Geometric graph for river network.
        
        Args:
            river_network: NetworkX river network graph
            observations: Microplastic observations
            node_features: Feature matrix for each node
            targets: Target values for each node
        
        Returns:
            PyTorch Geometric Data object
        """
        # Get all nodes
        nodes = sorted(river_network.nodes())
        node_id_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Build edge index
        edges = []
        edge_attrs = []
        
        for u, v, data in river_network.edges(data=True):
            if u in node_id_to_idx and v in node_id_to_idx:
                edges.append([node_id_to_idx[u], node_id_to_idx[v]])
                edge_attrs.append([
                    data.get('length_km', 1.0),
                    data.get('slope', 0.001),
                ])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor([[1.0, 0.001]], dtype=torch.float)
        
        # Build node feature tensor
        feature_dim = len(next(iter(node_features.values())))
        x = torch.zeros((len(nodes), feature_dim), dtype=torch.float)
        y = torch.zeros(len(nodes), dtype=torch.float)
        obs_mask = torch.zeros(len(nodes), dtype=torch.bool)
        
        for node, idx in node_id_to_idx.items():
            if node in node_features:
                x[idx] = torch.tensor(node_features[node], dtype=torch.float)
            if node in targets:
                y[idx] = torch.tensor(targets[node], dtype=torch.float)
                obs_mask[idx] = True
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            obs_mask=obs_mask,
            node_ids=nodes,
            node_id_to_idx=node_id_to_idx,
            num_nodes=len(nodes)
        )
        
        return data
    
    def build_geographic_graph(
        self,
        coordinates: List[Tuple[float, float]],
        features: np.ndarray,
        targets: np.ndarray,
        k: int = 5
    ) -> Data:
        """Build geographic nearest-neighbor graph.
        
        Args:
            coordinates: List of (lat, lon) tuples
            features: Feature matrix
            targets: Target values
            k: Number of nearest neighbors
        
        Returns:
            PyTorch Geometric Data object
        """
        from sklearn.neighbors import NearestNeighbors
        
        coords_rad = np.radians(coordinates)
        nbrs = NearestNeighbors(
            n_neighbors=min(k+1, len(coordinates)),
            metric='haversine'
        ).fit(coords_rad)
        
        distances, indices = nbrs.kneighbors(coords_rad)
        
        edges = []
        edge_attrs = []
        
        for i in range(len(coordinates)):
            for j_idx in range(1, len(indices[i])):
                j = indices[i][j_idx]
                dist = distances[i][j_idx]
                # Add bidirectional edges with inverse-distance weighting
                edges.append([i, j])
                edges.append([j, i])
                weight = 1.0 / (dist * 6371 + 1e-6)  # Inverse distance in km
                edge_attrs.append([dist * 6371, weight])
                edge_attrs.append([dist * 6371, weight])
        
        # Remove duplicate edges
        unique_edges = {}
        for edge, attr in zip(edges, edge_attrs):
            key = (edge[0], edge[1])
            if key not in unique_edges:
                unique_edges[key] = attr
        
        edges = list(unique_edges.keys())
        edge_attrs = list(unique_edges.values())
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor([[1.0, 1.0]], dtype=torch.float)
        
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(targets, dtype=torch.float)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=len(features)
        )
        
        return data
    
    def build_temporal_graph_sequence(
        self,
        graph_data: Data,
        temporal_features: np.ndarray,
        sequence_length: int = 7
    ) -> List[Data]:
        """Build sequence of graph snapshots for temporal modeling.
        
        Args:
            graph_data: Base graph structure
            temporal_features: Shape (num_nodes, sequence_length, feature_dim)
            sequence_length: Number of timesteps
        
        Returns:
            List of Data objects for each timestep
        """
        snapshots = []
        
        for t in range(sequence_length):
            snapshot = Data(
                x=torch.tensor(temporal_features[:, t, :], dtype=torch.float),
                edge_index=graph_data.edge_index.clone(),
                edge_attr=graph_data.edge_attr.clone(),
                y=graph_data.y.clone(),
                obs_mask=graph_data.obs_mask.clone(),
                num_nodes=graph_data.num_nodes
            )
            snapshots.append(snapshot)
        
        return snapshots
    
    def validate_graph(self, data: Data) -> Dict[str, Any]:
        """Validate a PyTorch Geometric graph.
        
        Args:
            data: PyG Data object
        
        Returns:
            Dictionary of validation results
        """
        results = {
            'num_nodes': data.num_nodes,
            'num_edges': data.edge_index.size(1) if data.edge_index is not None else 0,
            'has_isolated_nodes': bool((data.edge_index[0] != torch.arange(data.num_nodes)).all() and 
                                       (data.edge_index[1] != torch.arange(data.num_nodes)).all()),
            'has_self_loops': bool(torch.any(data.edge_index[0] == data.edge_index[1])),
            'is_directed': bool(data.edge_index.size(1) > 0 and 
                               not torch.any(torch.flip(data.edge_index, [0]) == data.edge_index)),
            'feature_shape': tuple(data.x.shape),
            'target_shape': tuple(data.y.shape),
            'obs_mask_sum': int(data.obs_mask.sum().item()) if hasattr(data, 'obs_mask') else 0,
        }
        
        return results


def get_river_graph_statistics(graph: nx.DiGraph) -> Dict[str, Any]:
    """Compute statistics for river network graph.
    
    Args:
        graph: NetworkX graph
    
    Returns:
        Dictionary of statistics
    """
    degrees = [d for n, d in graph.degree()]
    
    return {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'num_components': nx.number_weakly_connected_components(graph) 
                         if graph.is_directed() else nx.number_connected_components(graph),
        'isolated_nodes': len(list(nx.isolates(graph.to_undirected()))),
        'mean_degree': 2 * graph.number_of_edges() / max(1, graph.number_of_nodes()),
        'median_degree': float(np.median(degrees)) if degrees else 0,
        'max_degree': max(degrees) if degrees else 0,
        'min_degree': min(degrees) if degrees else 0,
        'density': nx.density(graph),
        'avg_shortest_path': nx.average_shortest_path_length(graph) if nx.is_weakly_connected(graph) else float('inf'),
    }


def build_graph_for_experiment(
    config: Dict[str, Any],
    observations: pd.DataFrame,
    graph_type: str = "river"
) -> Data:
    """Build graph for a specific experiment type.
    
    Args:
        config: Configuration dictionary
        observations: Microplastic observations
        graph_type: Type of graph ('river', 'geographic', 'haversine', 'random')
    
    Returns:
        PyTorch Geometric Data object
    """
    builder = GraphBuilder(config)
    
    # Get coordinates
    coordinates = list(zip(observations['latitude'].values, observations['longitude'].values))
    
    # Get features and targets
    feature_cols = [c for c in observations.columns 
                   if c not in ['latitude', 'longitude', 'date', 'concentration_particles_per_m3',
                               'location_name', 'watershed', 'source_dataset']]
    
    features = observations[feature_cols].values.astype(np.float32)
    targets = observations['concentration_particles_per_m3'].values.astype(np.float32)
    
    if graph_type == "geographic":
        return builder.build_geographic_graph(coordinates, features, targets, k=5)
    elif graph_type == "haversine":
        # Build haversine threshold graph
        from src.graph.river_topology import build_haversine_graph, create_graph_for_pytorch_geometric
        G = build_haversine_graph(coordinates, max_distance_km=50.0)
        return builder.build_river_graph(G, observations, {}, dict(enumerate(targets)))
    elif graph_type == "river":
        # Full river network graph
        from src.geospatial.spatial import build_river_network_graph
        G = build_river_network_graph(config)
        return builder.build_river_graph(G, observations, {}, dict(enumerate(targets)))
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")


def save_graph_data(data: Data, path: Path) -> None:
    """Save PyTorch Geometric graph data.
    
    Args:
        data: PyG Data object
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    logger.info(f"Graph data saved to {path}")


def load_graph_data(path: Path) -> Data:
    """Load PyTorch Geometric graph data.
    
    Args:
        path: Path to saved graph
    
    Returns:
        PyG Data object
    """
    data = torch.load(path, weights_only=False)
    logger.info(f"Graph data loaded from {path}")
    return data
