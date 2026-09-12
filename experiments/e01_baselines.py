"""Baseline model experiments."""
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from sklearn.model_selection import train_test_split

from src.models.baselines import train_baseline_models
from src.evaluation.metrics import compute_metrics, compare_models
from src.data.feature_engineering import normalize_features
from src.utils.config import load_config
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def run_baseline_experiments(config: dict, seed: int = 42) -> dict:
    """Run all baseline model experiments.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
    
    Returns:
        Dictionary with all results
    """
    set_seed(seed)
    
    # Load processed data
    processed_dir = Path(config['data']['processed_dir'])
    feature_file = processed_dir / "features_processed.csv"
    
    if not feature_file.exists():
        logger.warning("Processed features not found. Running data processing first...")
        from src.pipeline import run_data_pipeline
        run_data_pipeline(config)
    
    df = pd.read_csv(feature_file)
    logger.info(f"Loaded {len(df)} observations for baseline experiments")
    
    # Prepare features and target
    feature_cols = [c for c in df.columns 
                   if c not in ['target', 'obs_index', 'location_name', 'date',
                               'date_parsed', 'source_dataset', 'watershed',
                               'latitude', 'longitude', 'original_units',
                               'citation', 'notes', 'particle_types', 'basin',
                               'source_url', 'sampling_method', 'nearest_segment_id',
                               'has_max', 'below_detection_limit', 'season',
                               'particle_count']]
    
    X = df[feature_cols].fillna(df[feature_cols].median()).values
    y = df['target'].values
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target statistics: mean={np.mean(y):.4f}, "
                f"std={np.std(y):.4f}, min={np.min(y):.4f}, max={np.max(y):.4f}")
    
    # Apply log transformation
    y_log = np.log1p(y)
    logger.info(f"Log-transformed target: mean={np.mean(y_log):.4f}, std={np.std(y_log):.4f}")
    
    # Split data (random split for baseline benchmark)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_log, test_size=0.3, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed
    )
    
    # Normalize features (fit only on training data)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Train baseline models
    results_dir = Path(config['experiments']['results_dir']) / 'baselines'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_results = train_baseline_models(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        X_test_scaled, y_test,
        config,
        output_dir=results_dir
    )
    
    # Compute detailed test metrics
    detailed_results = {}
    for model_name, result in baseline_results.items():
        if 'error' in result:
            continue
        
        # Load model for predictions
        import joblib
        model_path = results_dir / f"{model_name}_model.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            y_pred = model.predict(X_test_scaled)
            
            # Transform back to original scale
            y_test_orig = np.expm1(y_test)
            y_pred_orig = np.expm1(y_pred)
            
            # Ensure non-negative
            y_pred_orig = np.maximum(y_pred_orig, 0)
            
            metrics_orig = compute_metrics(y_test_orig, y_pred_orig)
            metrics_log = compute_metrics(y_test, y_pred)
            
            detailed_results[model_name] = {
                'metrics_original_scale': metrics_orig,
                'metrics_log_scale': metrics_log,
                'train_results': {k: v for k, v in result.items() if k.startswith('train_')},
                'val_results': {k: v for k, v in result.items() if k.startswith('val_')},
                'n_train': len(X_train),
                'n_val': len(X_val),
                'n_test': len(X_test),
                'seed': seed
            }
            
            logger.info(f"  {model_name}: Test MAE={metrics_orig['mae']:.4f}, "
                       f"RMSE={metrics_orig['rmse']:.4f}, R²={metrics_orig['r2']:.4f}")
    
    # Save comparison table
    comparison = compare_models(detailed_results, metric='rmse')
    comparison.to_csv(results_dir / 'model_comparison.csv', index=False)
    
    # Save detailed results
    with open(results_dir / 'detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    return {
        'baseline_results': detailed_results,
        'model_comparison': comparison.to_dict(),
        'feature_columns': feature_cols,
        'data_splits': {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test)
        },
        'seed': seed
    }


def run_experiment_e1(config: dict = None, seed: int = 42):
    """Experiment E1: Mean/Median predictor baseline."""
    if config is None:
        config = load_config()
    return run_baseline_experiments(config, seed=seed)


if __name__ == '__main__':
    run_experiment_e1()
