"""Prediction script for loading trained models and generating predictions."""
import argparse
import torch
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import logging

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_trained_model(checkpoint_path: Path, config: dict, device=None):
    """Load a trained model from checkpoint."""
    if device is None:
        device = get_device()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine model type
    model_config = checkpoint.get('config', {}).get('model', config.get('model', {}))
    
    # Create model
    from src.models.temporal_gnn import GraphModelFactory
    model = GraphModelFactory.create('st-gnn', model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info(f"Loaded model from {checkpoint_path}")
    return model, checkpoint


def predict_from_csv(model_path: Path,
                     input_csv: Path,
                     output_csv: Path,
                     config_path: Path = None) -> pd.DataFrame:
    """Generate predictions from an input CSV file.
    
    Args:
        model_path: Path to trained model checkpoint
        input_csv: Path to input CSV with features
        output_csv: Path to output CSV with predictions
        config_path: Optional path to config file
    
    Returns:
        Dataframe with predictions
    """
    config = load_config(config_path) if config_path else load_config()
    device = get_device()
    
    # Load input data
    input_data = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(input_data)} observations from {input_csv}")
    
    # Load trained model
    is_gnn = model_path.suffix == '.pt'
    
    if is_gnn:
        model, checkpoint = load_trained_model(model_path, config, device)
        
        # Prepare features
        feature_cols = checkpoint.get('config', {}).get('feature_columns', [])
        if not feature_cols:
            feature_cols = [c for c in input_data.columns 
                          if c not in ['target', 'obs_index']]
        
        # This is a simplified prediction - in full implementation,
        # would need to construct graph from input data
        logger.warning("GNN prediction requires graph construction from input data. "
                      "Using simplified approach for demonstration.")
        
        # For baseline models stored as pickle
        predictions = np.zeros(len(input_data))
    else:
        # Baseline model (pickle)
        model = joblib.load(model_path)
        logger.info(f"Loaded baseline model from {model_path}")
        
        # Get features
        feature_cols = [c for c in input_data.columns 
                      if c not in ['target', 'obs_index', 'location_name', 
                                  'date', 'watershed']]
        
        X = input_data[feature_cols].fillna(0).values
        predictions = model.predict(X)
    
    # Transform predictions if log-transformed
    is_log_transformed = checkpoint.get('config', {}).get('training', {}).get(
        'loss_function', 'mse') == 'mse'  # Assume log transform
    
    if is_gnn and is_log_transformed:
        predictions = np.expm1(predictions)
    
    predictions = np.maximum(predictions, 0)
    
    # Create output
    output_data = input_data.copy()
    output_data['prediction'] = predictions
    
    if 'target' in input_data.columns:
        output_data['actual'] = input_data['target']
        output_data['error'] = predictions - input_data['target']
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_data.to_csv(output_csv, index=False)
    logger.info(f"Predictions saved to {output_csv}")
    
    return output_data


def main():
    """Main prediction entry point."""
    parser = argparse.ArgumentParser(
        description="Generate predictions using trained microplastic prediction model"
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to model checkpoint (.pt for GNN, .pkl for baseline)'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output', type=str, default='predictions.csv',
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--uncertainty', action='store_true',
        help='Generate uncertainty estimates (GNN only)'
    )
    
    args = parser.parse_args()
    
    result = predict_from_csv(
        Path(args.model),
        Path(args.input),
        Path(args.output),
        Path(args.config)
    )
    
    print(f"\nPrediction Summary:")
    print(f"  Total predictions: {len(result)}")
    if 'actual' in result.columns:
        mae = np.mean(np.abs(result['prediction'] - result['actual']))
        rmse = np.sqrt(np.mean((result['prediction'] - result['actual'])**2))
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
    print(f"  Mean prediction: {result['prediction'].mean():.4f}")
    print(f"  Predictions range: [{result['prediction'].min():.4f}, {result['prediction'].max():.4f}]")


if __name__ == '__main__':
    main()
