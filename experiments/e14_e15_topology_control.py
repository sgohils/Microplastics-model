"""E14-E15: Topology ablation experiments."""
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from typing import Dict, Any

from src.graph.build_graph import get_river_graph_statistics
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def run_experiment_e14_edge_ablation(
    config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Experiment E14: Edge type ablation.
    
    Remove different edge types (flow, proximity, similarity) to measure
    their contribution to model performance.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
    
    Returns:
        Results dictionary
    """
    set_seed(seed)
    
    logger.info("Running E14: Edge type ablation")
    
    # This experiment requires the full graph construction pipeline
    # For now, we'll return a structure that can be filled during full runs
    
    results = {
        'experiment': 'E14_edge_ablation',
        'seed': seed,
        'edge_types_tested': ['flow_only', 'proximity_only', 'similarity_only', 'all_edges'],
        'results': {}
    }
    
    for edge_type in ['flow_only', 'proximity_only', 'similarity_only', 'all_edges']:
        results['results'][edge_type] = {
            'rmse': None,
            'mae': None,
            'r2': None,
            'description': f'Performance with {edge_type.replace("_", " ")}'
        }
    
    logger.warning("E14 requires full graph pipeline. Results placeholder created.")
    
    return results


def run_experiment_e15_node_ablation(
    config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Experiment E15: Node feature ablation.
    
    Remove different node feature types to measure their contribution.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
    
    Returns:
        Results dictionary
    """
    set_seed(seed)
    
    logger.info("Running E15: Node feature ablation")
    
    processed_dir = Path(config['data']['processed_dir'])
    features_df = pd.read_csv(processed_dir / "features_processed.csv")
    
    feature_groups = {
        'hydrological': [c for c in features_df.columns if 'discharge' in c.lower() 
                        or 'flow' in c.lower() or 'precip' in c.lower()
                        or 'temperature' in c.lower() or 'elevation' in c.lower()],
        'spatial': [c for c in features_df.columns if 'lat' in c.lower() 
                   or 'lon' in c.lower() or 'coordinate' in c.lower()],
        'temporal': [c for c in features_df.columns if 'day_of' in c.lower()
                    or 'month' in c.lower() or 'season' in c.lower()
                    or 'hour' in c.lower()],
        'watershed': [c for c in features_df.columns if 'watershed' in c.lower()
                     or 'basin' in c.lower() or 'area' in c.lower()],
        'soil': [c for c in features_df.columns if 'soil' in c.lower()
                or 'clay' in c.lower() or 'sand' in c.lower()],
        'landuse': [c for c in features_df.columns if 'landuse' in c.lower()
                   or 'impervious' in c.lower() or 'forest' in c.lower()
                   or 'agriculture' in c.lower()],
    }
    
    results = {
        'experiment': 'E15_node_ablation',
        'seed': seed,
        'feature_groups': {k: v for k, v in feature_groups.items()},
        'results': {}
    }
    
    # For each feature group, remove it and record which features were removed
    for group_name, group_features in feature_groups.items():
        results['results'][f'remove_{group_name}'] = {
            'features_removed': group_features,
            'n_features_removed': len(group_features),
            'rmse': None,
            'mae': None,
            'r2': None
        }
    
    logger.warning("E15 requires full model training. Results structure created.")
    
    return results


def run_experiment_e21_topology_comparison(
    config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Experiment E21: Graph topology comparison.
    
    Compare different graph construction methods.
    """
    set_seed(seed)
    
    logger.info("Running E21: Topology comparison")
    
    results = {
        'experiment': 'E21_topology_comparison',
        'seed': seed,
        'topologies': ['full_topology', 'flow_only', 'proximity_knn'],
        'results': {}
    }
    
    for topology in ['full_topology', 'flow_only', 'proximity_knn']:
        results['results'][topology] = {
            'n_nodes': None,
            'n_edges': None,
            'rmse': None,
            'mae': None,
            'r2': None
        }
    
    return results
