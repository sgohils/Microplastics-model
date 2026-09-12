"""Main experiment runner - orchestrates all experiments."""
import argparse
import logging
from pathlib import Path
import json

from src.utils.config import load_config
from src.utils.logging import setup_logger as setup_logging
from src.utils.seed import set_seed

from .e01_baselines import run_experiment_e1
from .e02_e5_baselines import (
    run_experiment_e2_svm_rbf,
    run_experiment_e3_gaussian_process,
    run_experiment_e4_mlp,
    run_experiment_e5_ensemble,
    run_experiment_e7_feature_importance,
)
from .e06_gnn import run_gnn_experiment
from .e11_e19_generalization import (
    run_experiment_e11_spatial_holdout,
    run_experiment_e12_temporal_holdout,
    run_experiment_e13_spatiotemporal_holdout,
    run_experiment_e18_unseen_watershed,
    run_experiment_e19_temporal_extrapolation,
)
from .e14_e15_topology_control import (
    run_experiment_e14_edge_ablation,
    run_experiment_e15_node_ablation,
    run_experiment_e21_topology_comparison,
)
from .e17_e20_special_experiments import (
    run_experiment_e17_sparse_data,
    run_experiment_e20_extreme_events,
)

logger = logging.getLogger(__name__)

ALL_EXPERIMENTS = {
    'e1': run_experiment_e1,
    'e2': run_experiment_e2_svm_rbf,
    'e3': run_experiment_e3_gaussian_process,
    'e4': run_experiment_e4_mlp,
    'e5': run_experiment_e5_ensemble,
    'e6': run_gnn_experiment,
    'e7': run_experiment_e7_feature_importance,
    'e11': run_experiment_e11_spatial_holdout,
    'e12': run_experiment_e12_temporal_holdout,
    'e13': run_experiment_e13_spatiotemporal_holdout,
    'e14': run_experiment_e14_edge_ablation,
    'e15': run_experiment_e15_node_ablation,
    'e17': run_experiment_e17_sparse_data,
    'e18': run_experiment_e18_unseen_watershed,
    'e19': run_experiment_e19_temporal_extrapolation,
    'e20': run_experiment_e20_extreme_events,
    'e21': run_experiment_e21_topology_comparison,
}


def run_all_experiments(config: dict, seed: int = 42) -> dict:
    """Run all experiments and collect results.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
    
    Returns:
        Dictionary of all results
    """
    results = {}
    
    for exp_name, func in ALL_EXPERIMENTS.items():
        try:
            logger.info(f"Running {exp_name}...")
            result = func(config, seed=seed)
            results[exp_name] = result
            
            # Save individual result
            results_dir = Path(config['experiments']['results_dir'])
            results_dir.mkdir(parents=True, exist_ok=True)
            with open(results_dir / f"{exp_name}.json", 'w') as f:
                json.dump(result, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            results[exp_name] = {'error': str(e)}
    
    # Save combined results
    combined_path = results_dir / 'all_results.json'
    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"All results saved to {combined_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Specific experiment to run (default: all)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_logging(config)
    set_seed(args.seed)
    
    if args.experiment:
        result = ALL_EXPERIMENTS[args.experiment](config, seed=args.seed)
        print(json.dumps(result, indent=2))
    else:
        results = run_all_experiments(config, seed=args.seed)
        print(json.dumps(results, indent=2))
