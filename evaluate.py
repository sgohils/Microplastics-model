"""Evaluation entry point."""
import argparse
import json
from pathlib import Path
import logging

from src.utils.config import load_config
from src.evaluation.metrics import compute_metrics
from src.evaluation.uncertainty import DeepEnsemble
from src.evaluation.statistical_tests import permutation_test

logger = logging.getLogger(__name__)


def main():
    """Evaluate trained models."""
    parser = argparse.ArgumentParser(
        description="Evaluate Microplastic Prediction Models"
    )
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output', type=str, default='experiments/results')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    logger.info(f"Evaluating model: {args.model}")


if __name__ == '__main__':
    main()
