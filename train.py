"""Main training entry point."""
import argparse
import json
from pathlib import Path
import logging
import torch

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.pipeline import run_data_pipeline
from experiments.e01_baselines import run_baseline_experiments
from experiments.e06_gnn import run_experiments_e2_e6

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Microplastic Transport GNN Training Pipeline"
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--experiment', type=str, default='all',
        help='Experiment to run (baselines, gnn, all)'
    )
    parser.add_argument(
        '--output', type=str, default='experiments/results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config['experiments'] = {'results_dir': args.output}
    
    # Set seed
    set_seed(args.seed)
    
    # Check data
    processed_dir = Path(config['data']['processed_dir'])
    if not (processed_dir / "features_processed.csv").exists():
        logger.info("Processed data not found. Running data pipeline...")
        run_data_pipeline(config)
    
    # Run experiments
    if args.experiment in ['baselines', 'all']:
        logger.info("Running baseline model experiments...")
        baseline_results = run_baseline_experiments(config, seed=args.seed)
        logger.info("Baseline experiments complete")
    
    if args.experiment in ['gnn', 'all']:
        logger.info("Running GNN experiments...")
        gnn_results = run_experiments_e2_e6(config)
        logger.info("GNN experiments complete")
    
    logger.info("All experiments complete")


if __name__ == '__main__':
    main()
