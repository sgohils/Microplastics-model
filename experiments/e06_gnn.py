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

from src.models.temporal_gnn import SpatioTemporalGNN, GraphModelFactory
from src.training.train_gnn import train_model, load_checkpoint
from src.evaluation.metrics import compute_metrics
from src.evaluation.uncertainty import DeepEnsemble
from src.utils.config import load_config
from src.utils.seed import set_seed

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
        data.y = torch.tensor(self.targets[idx], dtype=torch.float)
        return data


def prepare_temporal_graph_data(
    features: pd.DataFrame,
    config: Dict[str, Any],
    sequence_length: int = 7
) -> Tuple[List[Data], List[int]]:
    """Prepare temporal graph data for GNN training.
    
    Args:
        features: Processed feature dataframe
        config: Configuration dictionary
        sequence_length: Number of timesteps
    
    Returns:
        Tuple of (list of Data objects, observation indices)
    """
    from torch_geometric.data import Data
    
    # Get feature columns
    feature_cols = [c for c in features.columns 
                   if c not in ['target', 'obs_index', 'location_name', 'date',
                               'date_parsed', 'source_dataset', 'watershed',
                               'latitude', 'longitude', 'original_units',
                               'citation', 'notes', 'particle_types', 'basin',
                               'source_url', 'sampling_method', 'nearest_segment_id',
                               'has_max', 'below_detection_limit', 'season',
                               'particle_count']]
    
    # Handle NaN values
    features_clean = features[feature_cols].copy()
    for col in feature_cols:
        if features_clean[col].dtype == 'object':
            features_clean[col] = pd.Categorical(features_clean[col]).codes
        features_clean[col] = features_clean[col].fillna(features_clean[col].median())
    
    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feature_values = scaler.fit_transform(features_clean.values)
    
    # Create temporal sequences
    # For each observation, create a sequence of the same features
    # (since we don't have multi-temporal environmental data in the base implementation)
    n_nodes = len(features)
    n_features = feature_values.shape[1]
    
    # Create sequence: repeat features for each timestep
    temporal_features = np.stack(
        [feature_values] * sequence_length, 
        axis=1
    )  # Shape: (n_nodes, seq_len, n_features)
    
    # Create graph (simple fully connected for now - real topology in separate module)
    # In real implementation, this would use the river network topology
    edge_index = torch.tensor([
        [i for i in range(n_nodes) for j in range(n_nodes) if i != j],
        [j for i in range(n_nodes) for j in range(n_nodes) if i != j]
    ], dtype=torch.long)
    
    # Create graph data
    graph_data = Data(
        x=torch.tensor(temporal_features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(features['target'].values, dtype=torch.float),
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
    
    return [graph_data], list(range(n_samples))


def run_gnn_experiment(
    config: Dict[str, Any],
    seed: int = 42,
    experiment_name: str = "E6_initial_gnn",
    model_type: str = "st-gnn"
) -> Dict[str, Any]:
    """Run a single GNN experiment.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
        experiment_name: Name for the experiment
        model_type: Type of GNN model
    
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
    
    # Prepare data
    sequence_length = config.get('model', {}).get('sequence_length', 7)
    graph_data_list, obs_indices = prepare_temporal_graph_data(
        features, config, sequence_length
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
    
    # Get predictions on all data
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            if hasattr(batch, 'test_mask'):
                preds = model(batch)[batch.test_mask]
                targets = batch.y[batch.test_mask]
            else:
                preds = model(batch)
                targets = batch.y
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Transform back to original scale
    all_preds_orig = np.expm1(np.array(all_preds))
    all_targets_orig = np.expm1(np.array(all_targets))
    all_preds_orig = np.maximum(all_preds_orig, 0)
    
    # Compute metrics
    metrics = compute_metrics(all_targets_orig, all_preds_orig)
    
    results = {
        'experiment_name': experiment_name,
        'model_type': model_type,
        'seed': seed,
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
