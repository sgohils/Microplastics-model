"""GNN model training experiment."""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data, DataLoader
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import networkx as nx

from src.models.temporal_gnn import SpatioTemporalGNN, GraphModelFactory
from src.training.train_gnn import train_model, load_checkpoint
from src.evaluation.metrics import compute_metrics
from src.evaluation.uncertainty import DeepEnsemble
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.geospatial.spatial import build_river_network_graph, RiverNetworkBuilder
from src.graph.river_topology import randomize_topology

logger = logging.getLogger(__name__)


class TemporalGraphDataset(torch.utils.data.Dataset):
    """Dataset for temporal graph data."""
    
    def __init__(self, graph_data_list: List[Data], 
                 targets: np.ndarray):
        """
        Args:
            graph_data_list: List of temporal graph snapshots
            targets: Target values for observation nodes
        """
        self.graph_data_list = graph_data_list
        self.targets = targets
    
    def __len__(self):
        return len(self.graph_data_list)
    
    def __getitem__(self, idx):
        data = self.graph_data_list[idx]
        # y is already set in the graph_data; update with targets if needed
        # Don't overwrite y - use the data.y that was already set
        return data


def build_river_topology_graph(
    features: pd.DataFrame,
    config: Dict[str, Any],
    use_real_topology: bool = True,
    seed: int = 42
) -> nx.DiGraph:
    """Build a graph using real river topology or random topology for control.
    
    Args:
        features: Processed feature dataframe
        config: Configuration dictionary
        use_real_topology: If True, use real river network; if False, use random topology
        seed: Random seed for reproducibility
    
    Returns:
        NetworkX DiGraph with observation nodes connected to river network
    """
    if use_real_topology:
        builder = RiverNetworkBuilder(config)
        river_network = builder.build_delaware_river_network()
        builder.add_microplastic_locations(features)
        graph = builder.graph
    else:
        # Build random topology as control (same nodes, random edges)
        builder = RiverNetworkBuilder(config)
        river_network = builder.build_delaware_river_network()
        builder.add_microplastic_locations(features)
        graph = randomize_topology(builder.graph, seed=seed)
    
    return graph


def get_feature_cols(features: pd.DataFrame) -> List[str]:
    """Get feature columns excluding target and metadata."""
    exclude = [
        'target', 'obs_index', 'location_name', 'date',
        'date_parsed', 'source_dataset', 'watershed',
        'latitude', 'longitude', 'original_units',
        'citation', 'notes', 'particle_types', 'basin',
        'source_url', 'sampling_method', 'nearest_segment_id',
        'has_max', 'below_detection_limit', 'season',
        'particle_count'
    ]
    return [c for c in features.columns if c not in exclude]


def prepare_temporal_graph_data(
    features: pd.DataFrame,
    config: Dict[str, Any],
    sequence_length: int = 7,
    use_real_topology: bool = True,
    seed: int = 42
) -> Tuple[List[Data], List[int], nx.DiGraph]:
    """Prepare temporal graph data for GNN training using river topology.
    
    Args:
        features: Processed feature dataframe
        config: Configuration dictionary
        sequence_length: Number of timesteps
        use_real_topology: If True, use real river network topology
        seed: Random seed
    
    Returns:
        Tuple of (list of Data objects, observation indices, NetworkX graph)
    """
    from torch_geometric.data import Data
    from sklearn.preprocessing import StandardScaler
    
    feature_cols = get_feature_cols(features)
    
    # Handle NaN values
    features_clean = features[feature_cols].copy()
    for col in feature_cols:
        if features_clean[col].dtype == 'object':
            features_clean[col] = pd.Categorical(features_clean[col]).codes
        features_clean[col] = features_clean[col].fillna(features_clean[col].median())
    
    # Normalize
    scaler = StandardScaler()
    feature_values = scaler.fit_transform(features_clean.values)
    
    n_nodes = len(features)
    n_features = feature_values.shape[1]
    
    # Create temporal sequences: repeat features for each timestep
    temporal_features = np.stack(
        [feature_values] * sequence_length, 
        axis=1
    )  # Shape: (n_nodes, seq_len, n_features)
    
    # Build real river topology graph - but filter to only observation nodes
    # and create edges between observations connected through river segments
    river_graph = build_river_topology_graph(
        features, config, 
        use_real_topology=use_real_topology,
        seed=seed
    )
    
    # Filter: only keep observation nodes (obs_*)
    obs_nodes = [n for n in river_graph.nodes() if str(n).startswith('obs_')]
    obs_to_idx = {str(n): i for i, n in enumerate(obs_nodes)}
    
    # Create edges between observation nodes that share a common river segment
    # or are connected through the river network
    edges = []
    for i, obs_i in enumerate(obs_nodes):
        nearest_seg_i = river_graph.nodes[obs_i].get('nearest_segment')
        for j, obs_j in enumerate(obs_nodes):
            if i >= j:
                continue
            nearest_seg_j = river_graph.nodes[obs_j].get('nearest_segment')
            
            # Connect observations on the same or connected segments
            if nearest_seg_i is not None and nearest_seg_i == nearest_seg_j:
                edges.append([i, j])
                edges.append([j, i])
            elif nearest_seg_i is not None and nearest_seg_j is not None:
                # Check if segments are connected (upstream/downstream)
                try:
                    if (nx.has_path(river_graph, nearest_seg_i, nearest_seg_j) or
                        nx.has_path(river_graph, nearest_seg_j, nearest_seg_i)):
                        edges.append([i, j])
                        edges.append([j, i])
                except nx.NetworkXNoPath:
                    pass
    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # Fallback: connect each observation to its 3 nearest neighbors by distance
        from src.graph.river_topology import build_geographic_graph
        coords = list(zip(features['latitude'].values, features['longitude'].values))
        geo_graph = build_geographic_graph(coords, k=3)
        edge_list = list(geo_graph.edges())
        if edge_list:
            edge_index = torch.tensor(
                [[e[0], e[1]] for e in edge_list],
                dtype=torch.long
            ).t().contiguous()
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long).t().contiguous()
    
    # Create graph data - only observation nodes
    # Apply log1p transform to target for training consistency with baselines
    targets = np.log1p(features['target'].values)
    graph_data = Data(
        x=torch.tensor(temporal_features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(targets, dtype=torch.float),
        num_nodes=n_nodes
    )
    
    # Create train/val/test masks based on random split
    n_samples = len(features)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_end = int(0.7 * n_samples)
    val_end = int(0.85 * n_samples)
    
    train_mask = np.zeros(n_samples, dtype=bool)
    val_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)
    
    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True
    
    graph_data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    graph_data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    graph_data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
    graph_data.obs_mask = torch.ones(n_samples, dtype=torch.bool)
    
    return [graph_data], list(range(n_samples)), river_graph


def run_gnn_experiment(
    config: Dict[str, Any],
    seed: int = 42,
    experiment_name: str = "E6_initial_gnn",
    model_type: str = "st-gnn",
    use_real_topology: bool = True
) -> Dict[str, Any]:
    """Run a single GNN experiment.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
        experiment_name: Name for the experiment
        model_type: Type of GNN model
        use_real_topology: If True, use real river network topology
    
    Returns:
        Dictionary with experiment results
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load processed features
    processed_dir = Path(config['data']['processed_dir'])
    feature_file = processed_dir / "features_processed.csv"
    
    if not feature_file.exists():
        logger.error("Processed features not found. Run data pipeline first.")
        return {'error': 'No processed data found'}
    
    features = pd.read_csv(feature_file)
    logger.info(f"Loaded {len(features)} observations for GNN experiment")
    
    # Prepare data with river topology
    sequence_length = config.get('model', {}).get('sequence_length', 7)
    graph_data_list, obs_indices, river_graph = prepare_temporal_graph_data(
        features, config, sequence_length,
        use_real_topology=use_real_topology,
        seed=seed
    )
    
    # Create data loaders
    train_dataset = TemporalGraphDataset(
        graph_data_list, 
        features['target'].values
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.get('training', {}).get('batch_size', 32),
        shuffle=True
    )
    
    # Update model config with actual input dimension
    model_config = config.get('model', {})
    feature_cols = [c for c in features.columns 
                   if c not in ['target', 'obs_index', 'location_name', 'date',
                               'date_parsed', 'source_dataset', 'watershed',
                               'latitude', 'longitude', 'original_units',
                               'citation', 'notes', 'particle_types', 'basin',
                               'source_url', 'sampling_method', 'nearest_segment_id',
                               'has_max', 'below_detection_limit', 'season',
                               'particle_count']]
    
    model_config['input_dim'] = len(feature_cols)
    
    # Create model
    model = GraphModelFactory.create(model_type, model_config)
    model = model.to(device)
    
    # Setup paths
    results_dir = Path(config['experiments']['results_dir'])
    checkpoint_path = results_dir / 'checkpoints' / f'{experiment_name}_seed{seed}.pt'
    
    # Train model
    training_config = {
        'training': config.get('training', {}),
        'model': model_config,
        'general': config.get('general', {})
    }
    
    train_results = train_model(
        model, train_loader, train_loader,  # Use train as val for simplicity
        training_config,
        checkpoint_path=checkpoint_path,
        seed=seed,
        device=device
    )
    
    # Evaluate
    model.eval()
    
    # Get predictions on test set only
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            preds = model(batch)
            targets = batch.y
            
            # Use test mask for evaluation
            if hasattr(batch, 'test_mask'):
                test_mask = batch.test_mask
                preds = preds[test_mask]
                targets = targets[test_mask]
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Transform back to original scale (targets were log1p transformed)
    all_preds_orig = np.expm1(all_preds)
    all_targets_orig = np.expm1(all_targets)
    all_preds_orig = np.maximum(all_preds_orig, 0)
    
    # Compute metrics on original scale
    metrics = compute_metrics(all_targets_orig, all_preds_orig)
    
    results = {
        'experiment_name': experiment_name,
        'model_type': model_type,
        'seed': seed,
        'use_real_topology': use_real_topology,
        'river_graph_stats': {
            'num_nodes': river_graph.number_of_nodes(),
            'num_edges': river_graph.number_of_edges(),
        },
        'training_results': train_results,
        'test_metrics': metrics,
        'config': config,
        'checkpoint_path': str(checkpoint_path)
    }
    
    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f'{experiment_name}_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Experiment {experiment_name} completed:")
    logger.info(f"  Test MAE: {metrics['mae']:.4f}")
    logger.info(f"  Test RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  Test R²: {metrics['r2']:.4f}")
    
    return results


def run_experiments_e2_e6(config: Dict[str, Any]) -> Dict[str, Dict]:
    """Run experiments E2-E6 (baseline comparisons and initial GNN).
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Dictionary with all experiment results
    """
    results = {}
    seeds = config.get('general', {}).get('seeds', [42, 123, 456, 789, 101])
    
    # For this initial implementation, run with a single seed due to
    # data limitations (small dataset)
    seed = seeds[0] if isinstance(seeds, list) else 42
    
    # Run GNN experiment
    results['gnn'] = run_gnn_experiment(
        config, seed=seed, 
        experiment_name="E6_initial_gnn",
        model_type="st-gnn"
    )
    
    return results
