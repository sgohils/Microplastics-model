"""River network topology construction and graph randomization."""
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


def randomize_topology(graph: nx.DiGraph, 
                       seed: int = 42) -> nx.DiGraph:
    """Randomize graph topology while preserving degree distribution.
    
    Uses the configuration model to create a graph with the same
    in-degree and out-degree distribution but randomized edge connections.
    This tests whether benefits come from meaningful topology or simply
    from having any graph structure.
    
    Args:
        graph: Input DiGraph
        seed: Random seed
    
    Returns:
        Randomized DiGraph with same degree distribution
    """
    np.random.seed(seed)
    
    # Extract degree sequences (only for non-observation nodes)
    in_degree_seq = [d for n, d in graph.in_degree() if not str(n).startswith('obs_')]
    out_degree_seq = [d for n, d in graph.out_degree() if not str(n).startswith('obs_')]
    
    # Use configuration model for directed graphs
    try:
        randomized = nx.configuration_model(
            in_degree_seq, out_degree_seq, 
            create_using=nx.DiGraph,
            seed=seed
        )
    except Exception:
        # Fallback: simple edge shuffling
        randomized = graph.copy()
        edges = list(randomized.edges())
        
        # Shuffle targets while keeping sources
        sources = [e[0] for e in edges if not str(e[0]).startswith('obs_') and not str(e[1]).startswith('obs_')]
        targets = [e[1] for e in edges if not str(e[0]).startswith('obs_') and not str(e[1]).startswith('obs_')]
        
        np.random.shuffle(targets)
        
        randomized.remove_edges_from(edges)
        for s, t in zip(sources, targets):
            if s != t:  # No self-loops
                randomized.add_edge(s, t)
    
    return randomized


def build_geographic_graph(coordinates: List[Tuple[float, float]],
                          k: int = 5) -> nx.Graph:
    """Build a geographic nearest-neighbor graph.
    
    Creates an undirected graph where each node is connected to its
    k nearest neighbors based on geographic distance.
    
    Args:
        coordinates: List of (lat, lon) tuples
        k: Number of nearest neighbors
    
    Returns:
        NetworkX undirected graph
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import haversine
    
    coords_rad = np.radians(coordinates)
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(coordinates)), 
                           metric='haversine').fit(coords_rad)
    
    distances, indices = nbrs.kneighbors(coords_rad)
    
    G = nx.Graph()
    for i, coord in enumerate(coordinates):
        G.add_node(i, pos=coord)
    
    for i in range(len(coordinates)):
        for j_idx in range(1, min(k+1, len(indices[i]))):  # Skip self (index 0)
            j = indices[i][j_idx]
            dist = distances[i][j_idx]
            # Use inverse distance as weight
            G.add_edge(i, j, weight=1.0 / (dist + 1e-6))
    
    return G


def build_haversine_graph(coordinates: List[Tuple[float, float]],
                         max_distance_km: float = 50.0) -> nx.Graph:
    """Build a graph based on haversine distance threshold.
    
    Args:
        coordinates: List of (lat, lon) tuples
        max_distance_km: Maximum distance for edge creation (km)
    
    Returns:
        NetworkX undirected graph
    """
    G = nx.Graph()
    for i, coord in enumerate(coordinates):
        G.add_node(i, pos=coord)
    
    for i in range(len(coordinates)):
        for j in range(i+1, len(coordinates)):
            # Calculate haversine distance
            lat1, lon1 = np.radians(coordinates[i])
            lat2, lon2 = np.radians(coordinates[j])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            dist = 6371 * c  # Earth radius in km
            
            if dist <= max_distance_km:
                G.add_edge(i, j, weight=dist)
    
    return G


def compute_graph_statistics(graph: nx.Graph) -> Dict[str, Any]:
    """Compute comprehensive graph statistics.
    
    Args:
        graph: NetworkX graph
    
    Returns:
        Dictionary of graph statistics
    """
    stats = {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'is_directed': graph.is_directed(),
        'num_components': nx.number_connected_components(graph) if not graph.is_directed() else 'N/A',
        'isolated_nodes': len(list(nx.isolates(graph))),
        'mean_degree': 2 * graph.number_of_edges() / max(1, graph.number_of_nodes()),
        'density': nx.density(graph),
    }
    
    # Compute degree statistics
    if graph.number_of_nodes() > 0:
        degrees = [d for n, d in graph.degree()]
        stats['median_degree'] = float(np.median(degrees))
        stats['max_degree'] = max(degrees)
        stats['min_degree'] = min(degrees)
    else:
        stats['median_degree'] = 0
        stats['max_degree'] = 0
        stats['min_degree'] = 0
    
    return stats


def validate_graph_directions(graph: nx.DiGraph) -> List[str]:
    """Validate that graph edges follow correct flow direction.
    
    Args:
        graph: Directed graph with river segments
    
    Returns:
        List of validation warnings
    """
    warnings = []
    
    # Check for cycles (rivers shouldn't have downstream-to-upstream cycles)
    try:
        cycles = list(nx.simple_cycles(graph))
        if cycles:
            warnings.append(f"Found {len(cycles)} cycles in graph - rivers should be directed acyclic")
    except Exception as e:
        warnings.append(f"Cycle detection failed: {e}")
    
    # Check for self-loops
    self_loops = list(nx.selfloop_edges(graph))
    if self_loops:
        warnings.append(f"Found {len(self_loops)} self-loops")
    
    return warnings


def create_graph_for_pytorch_geometric(
    graph: nx.DiGraph,
    node_features: np.ndarray,
    node_labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray
) -> Any:
    """Convert NetworkX graph to PyTorch Geometric Data object.
    
    Args:
        graph: NetworkX graph
        node_features: Node feature matrix
        node_labels: Node labels (targets)
        train_mask: Training node mask
        val_mask: Validation node mask
        test_mask: Test node mask
    
    Returns:
        PyTorch Geometric Data object
    """
    from torch_geometric.data import Data
    import torch
    
    # Convert to edge index format
    edges = list(graph.edges())
    if not edges:
        # Handle empty edge case
        edge_index = torch.tensor([[0], [0]], dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(node_labels, dtype=torch.float)
    train_mask_t = torch.tensor(train_mask, dtype=torch.bool)
    val_mask_t = torch.tensor(val_mask, dtype=torch.bool)
    test_mask_t = torch.tensor(test_mask, dtype=torch.bool)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask_t,
        val_mask=val_mask_t,
        test_mask=test_mask_t
    )
    
    return data


def create_temporal_graph_sequence(
    graph: nx.DiGraph,
    feature_dict: Dict[int, np.ndarray],
    sequence_length: int
) -> List[Any]:
    """Create a sequence of graph snapshots for temporal modeling.
    
    Args:
        graph: Static graph structure
        feature_dict: Dictionary mapping timestep to node features
        sequence_length: Number of timesteps
    
    Returns:
        List of PyTorch Geometric Data objects
    """
    snapshots = []
    
    for t in range(sequence_length):
        if t in feature_dict:
            features = feature_dict[t]
            data = create_graph_for_pytorch_geometric(
                graph, features, np.zeros(len(features)),
                np.ones(len(features), dtype=bool),
                np.zeros(len(features), dtype=bool),
                np.zeros(len(features), dtype=bool)
            )
            snapshots.append(data)
    
    return snapshots
